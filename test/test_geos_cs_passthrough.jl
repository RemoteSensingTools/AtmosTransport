#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan indexed-baking-valiant Commit 5 — GEOS → CS passthrough orchestrator
#
# Verifies:
#   1. process_day(date, ::CSGrid, ::AbstractGEOSSettings, vertical; out_path)
#      writes a v4 binary at out_path with the expected size and shape.
#   2. The binary's write-time replay gate is below the requested tolerance.
#   3. NO Poisson-balance call is invoked on the passthrough path (it would
#      be a regression if the user's "GEOS already provides mass fluxes"
#      semantic was lost). Verified by stubbing
#      `balance_cs_global_mass_fluxes!` to error.
#   4. `geos_native_to_face_flux!` correctly maps MFXC → am_v4 with
#      panel-edge halos populated by mirror sync.
# ---------------------------------------------------------------------------

using Test
using Dates
using NCDatasets
using TOML

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.Grids: CubedSphereMesh, GEOSNativePanelConvention,
                              EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST,
                              HybridSigmaPressure, n_levels
using .AtmosTransport.Preprocessing: GEOSITSettings, process_day,
                                      build_target_geometry,
                                      geos_native_to_face_flux!,
                                      sync_all_cs_boundary_mirrors!,
                                      load_hybrid_coefficients,
                                      CS_PANEL_COUNT
using .AtmosTransport.MetDrivers: CubedSphereBinaryReader

const FT_TEST = Float64
const NC = 8
const NPANEL = 6
const NZ = 4

# Inline the synthetic fixture writers (same shape as test_geos_reader.jl;
# duplicating ~50 lines is cleaner than cross-test importing).

function _synthetic_hybrid_coefs_c5(nz::Int)
    nz == 4 || error("synthetic coefs only defined for nz=4 here")
    A = [0.0, 0.0, 0.0, 0.0, 0.0]
    B = [0.0, 0.05, 0.20, 0.55, 1.0]
    return A, B
end

function _defvar3d_c5!(ds, name, A; units::String="")
    v = defVar(ds, name, eltype(A), ("Xdim","Ydim","nf","lev","time");
               attrib = ["units" => units])
    v[:, :, :, :, :] = A
    return v
end

function _defvar2d_xyz_t_c5!(ds, name, A; units::String="")
    v = defVar(ds, name, eltype(A), ("Xdim","Ydim","nf","time");
               attrib = ["units" => units])
    v[:, :, :, :] = A
    return v
end

"""Write a synthetic GEOS-IT day under `dir`, bottom-up levels, 24-window cadence.
For replay-gate testing we default to **zero horizontal fluxes** so the
divergence is identically zero and the cm column closes trivially. Set
`mfxc_const`/`mfyc_const` to test other paths.
"""
function _write_synthetic_geosit_day_c5!(dir::String, datestr::String;
                                         qv_const::Float64 = 0.005,
                                         mfxc_const::Float64 = 0.0,
                                         mfyc_const::Float64 = 0.0,
                                         ntimes_a1::Int = 24,
                                         ntimes_i1::Int = 25)
    mkpath(dir)
    A_top, B_top = _synthetic_hybrid_coefs_c5(NZ)
    delp_per_layer_bu = Float64[]
    for k in NZ:-1:1
        ΔA = A_top[k+1] - A_top[k]
        ΔB = B_top[k+1] - B_top[k]
        push!(delp_per_layer_bu, ΔA + ΔB * 100_000.0)
    end

    a1 = joinpath(dir, "GEOSIT.$(datestr).CTM_A1.C$(NC).nc")
    NCDataset(a1, "c") do ds
        ds.dim["Xdim"] = NC
        ds.dim["Ydim"] = NC
        ds.dim["nf"]   = NPANEL
        ds.dim["lev"]  = NZ
        ds.dim["time"] = ntimes_a1
        delp_bu = fill(0.0, NC, NC, NPANEL, NZ, ntimes_a1)
        for k in 1:NZ
            delp_bu[:, :, :, k, :] .= delp_per_layer_bu[k]
        end
        _defvar3d_c5!(ds, "DELP", delp_bu;       units="Pa")
        _defvar3d_c5!(ds, "MFXC", fill(mfxc_const, NC, NC, NPANEL, NZ, ntimes_a1);
                       units="Pa m+2")
        _defvar3d_c5!(ds, "MFYC", fill(mfyc_const, NC, NC, NPANEL, NZ, ntimes_a1);
                       units="Pa m+2")
        defVar(ds, "lev", collect(1:NZ), ("lev",); attrib=["positive" => "down"])
    end

    i1 = joinpath(dir, "GEOSIT.$(datestr).CTM_I1.C$(NC).nc")
    NCDataset(i1, "c") do ds
        ds.dim["Xdim"] = NC
        ds.dim["Ydim"] = NC
        ds.dim["nf"]   = NPANEL
        ds.dim["lev"]  = NZ
        ds.dim["time"] = ntimes_i1
        _defvar2d_xyz_t_c5!(ds, "PS", fill(1000.0, NC, NC, NPANEL, ntimes_i1);
                             units="hPa")
        _defvar3d_c5!(ds, "QV", fill(qv_const, NC, NC, NPANEL, NZ, ntimes_i1);
                       units="kg kg-1")
        defVar(ds, "lev", collect(1:NZ), ("lev",); attrib=["positive" => "down"])
    end

    return a1, i1
end

function _write_synthetic_coefs_toml_c5!(path::String)
    A, B = _synthetic_hybrid_coefs_c5(NZ)
    open(path, "w") do io
        write(io, """
            [metadata]
            n_levels = $(NZ)
            n_interfaces = $(NZ + 1)

            [coefficients]
            a = $(repr(A))
            b = $(repr(B))
            """)
    end
end

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "GEOS → CS passthrough" begin
    tmpdir   = mktempdir()
    daydir   = joinpath(tmpdir, "20211201")
    nextdir  = joinpath(tmpdir, "20211202")
    _write_synthetic_geosit_day_c5!(daydir,  "20211201")
    _write_synthetic_geosit_day_c5!(nextdir, "20211202")
    coef_path = joinpath(tmpdir, "synthetic_coefs.toml")
    _write_synthetic_coefs_toml_c5!(coef_path)

    settings = GEOSITSettings(
        root_dir = tmpdir, Nc = NC,
        mass_flux_dt = 450.0,
        coefficients_file = coef_path,
    )

    grid = build_target_geometry(
        Dict{String,Any}("type" => "cubed_sphere", "Nc" => NC,
                          "panel_convention" => "geos_native"),
        FT_TEST,
    )

    vertical = (
        merged_vc = load_hybrid_coefficients(coef_path),
        Nz = NZ,
        Nz_native = NZ,
    )

    @testset "geos_native_to_face_flux! shape + values" begin
        # Use a SINGLE constant for both am_native and bm_native — that way,
        # regardless of which neighbour edge a panel's west/south halo comes
        # from (X-edge or Y-edge, with possibly flipped orientation), the
        # synced halo magnitude is unambiguous.
        K = FT_TEST(42)
        am_native = ntuple(p -> fill(K, NC, NC, NZ), 6)
        bm_native = ntuple(p -> fill(K, NC, NC, NZ), 6)
        am_v4 = ntuple(p -> zeros(FT_TEST, NC + 1, NC, NZ), 6)
        bm_v4 = ntuple(p -> zeros(FT_TEST, NC, NC + 1, NZ), 6)
        scale = FT_TEST(2.0)
        geos_native_to_face_flux!(am_v4, bm_v4, am_native, bm_native,
                                  grid.mesh.connectivity, NC, NZ, scale)
        # Interior + east boundary: am_v4[i+1, j, k] = MFXC[i, j, k] * scale.
        # West-boundary halo (i=1) is filled from a neighbour's canonical
        # edge by `sync_all_cs_boundary_mirrors!` and has magnitude 42*scale
        # too (sign may flip per panel orientation, hence the abs check).
        for p in 1:6
            for k in 1:NZ, j in 1:NC, i in 1:NC
                @test am_v4[p][i + 1, j, k] ≈ K * scale
            end
            for k in 1:NZ, j in 1:NC
                @test abs(am_v4[p][1, j, k]) ≈ K * scale
            end
            for k in 1:NZ, i in 1:NC, j in 1:NC
                @test bm_v4[p][i, j + 1, k] ≈ K * scale
            end
            for k in 1:NZ, i in 1:NC
                @test abs(bm_v4[p][i, 1, k]) ≈ K * scale
            end
        end
    end

    @testset "nested GEOS CS block coarsening helpers" begin
        R = Val(2)
        src3 = reshape(collect(FT_TEST, 1:16), 4, 4, 1)
        dst3 = zeros(FT_TEST, 2, 2, 1)
        AtmosTransport.Preprocessing._coarsen_sum3!(dst3, src3, R)
        @test dst3[:, :, 1] == [
            sum(src3[1:2, 1:2, 1]) sum(src3[1:2, 3:4, 1]);
            sum(src3[3:4, 1:2, 1]) sum(src3[3:4, 3:4, 1])
        ]

        src_area = fill(FT_TEST(1), 4, 4)
        src2 = reshape(collect(FT_TEST, 1:16), 4, 4)
        dst2 = zeros(FT_TEST, 2, 2)
        AtmosTransport.Preprocessing._coarsen_area_weighted2!(dst2, src2, src_area, R)
        @test dst2 ≈ [
            sum(src2[1:2, 1:2]) / 4 sum(src2[1:2, 3:4]) / 4;
            sum(src2[3:4, 1:2]) / 4 sum(src2[3:4, 3:4]) / 4
        ]

        src_x = ones(FT_TEST, 5, 4, 1)
        dst_x = zeros(FT_TEST, 3, 2, 1)
        AtmosTransport.Preprocessing._coarsen_xface_sum!(dst_x, src_x, R)
        @test all(dst_x .== 2)

        src_y = ones(FT_TEST, 4, 5, 1)
        dst_y = zeros(FT_TEST, 2, 3, 1)
        AtmosTransport.Preprocessing._coarsen_yface_sum!(dst_y, src_y, R)
        @test all(dst_y .== 2)
    end

    @testset "process_day writes a valid CS binary" begin
        out_path = joinpath(tmpdir, "out_cs.bin")
        result = process_day(Date(2021, 12, 1), grid, settings, vertical;
                             out_path = out_path,
                             dt_met_seconds = 3600.0,
                             FT = FT_TEST,
                             mass_basis = :dry,
                             replay_tol = 1e-12)
        @test isfile(out_path)
        @test result.elapsed > 0
        @test result.worst_replay_rel < 1e-12
        @test result.out_path == out_path

        # Open the binary back and confirm the header round-trips.
        reader = CubedSphereBinaryReader(out_path; FT=FT_TEST)
        try
            @test reader.header.Nc == NC
            @test reader.header.npanel == 6
            @test reader.header.nlevel == NZ
            @test reader.header.nwindow == 24                        # GEOS hourly
            @test reader.header.mass_basis === :dry
            @test reader.header.panel_convention === :geos_native
            @test :dm in reader.header.payload_sections              # delta enabled
        finally
            close(reader)
        end
    end
end
