#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# test_ll_to_cs_regrid_script.jl — plan 40 Commit 3
#
# Exercises `regrid_ll_binary_to_cs` end-to-end: writes a tiny LL v4 fixture,
# regrids it to a C4 cubed-sphere binary, and verifies
#   1. the output file exists and its header is readable by
#      `CubedSphereBinaryReader` / `inspect_binary`,
#   2. the Poisson-balanced CS output satisfies invariant 13
#      (`max(|cm|/m) < 1e-7` on F32 / `< 1e-12` on F64),
#   3. the total CS air mass matches the LL source to within conservative-
#      regrid tolerance.
#
# This is the unit-level validation the existing `regrid_ll_binary_to_cs`
# function has lacked: today it is only exercised transitively through
# `process_day`. The thin CLI wrapper
# `scripts/preprocessing/regrid_ll_transport_binary_to_cs.jl` is the
# deployment surface; this test locks in the numerical contract.
# ---------------------------------------------------------------------------

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Preprocessing: regrid_ll_binary_to_cs, build_target_geometry

# LL fixture: small but non-trivial. Pressure varies with latitude so
# `recover_ll_cell_center_winds!` sees a genuine ∇p field, and air mass
# varies with B-coefficient so the regridded totals aren't all identical.
function _ll_fixture_binary(path::AbstractString;
                            FT::Type{<:AbstractFloat} = Float64,
                            Nx::Int = 24, Ny::Int = 13, Nz::Int = 4,
                            nwindow::Int = 2)
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    A_ifc = FT[0, 2500, 5000, 7500, 10000]
    B_ifc = FT[0, 0.1, 0.3, 0.6, 1.0]
    vertical = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)

    # ps varies with latitude: lower at poles, higher at equator.
    ps = Array{FT}(undef, Nx, Ny)
    for j in 1:Ny
        φ = FT(mesh.φᶜ[j])
        ps_col = FT(101_300) - FT(2_000) * FT(sind(φ)^2)
        for i in 1:Nx
            ps[i, j] = ps_col
        end
    end

    # m[i,j,k] = (A[k+1]-A[k]) + (B[k+1]-B[k]) * ps[i,j], divided by g ≈ 9.81.
    g = FT(9.81)
    m = Array{FT}(undef, Nx, Ny, Nz)
    for k in 1:Nz
        dA = A_ifc[k + 1] - A_ifc[k]
        dB = B_ifc[k + 1] - B_ifc[k]
        for j in 1:Ny, i in 1:Nx
            m[i, j, k] = (dA + dB * ps[i, j]) / g
        end
    end

    # Start with zero fluxes — tests the core regrid + Poisson + cm-diagnose
    # path without committing to a particular wind field. Post-balance
    # `max(|cm|/m)` should be at solver tolerance, `dm` between windows is
    # zero, `cm` stays zero.
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    cm = zeros(FT, Nx, Ny, Nz + 1)

    windows = [ (m = m, am = am, bm = bm, cm = cm, ps = ps) for _ in 1:nwindow ]

    write_transport_binary(path, grid, windows;
                           FT = FT,
                           dt_met_seconds = 3600.0,
                           half_dt_seconds = 1800.0,
                           steps_per_window = 2,
                           mass_basis = :dry,
                           source_flux_sampling = :window_start_endpoint,
                           flux_sampling = :window_constant)
    return (; Nx, Ny, Nz, nwindow, m_total = sum(m) * nwindow)
end

@testset "plan 40 Commit 3 — regrid_ll_binary_to_cs end-to-end" begin

    @testset "F64 static LL → C4 CS, 2 windows" begin
        mktempdir() do dir
            ll_path = joinpath(dir, "ll_fixture.bin")
            cs_path = joinpath(dir, "cs_fixture.bin")

            meta = _ll_fixture_binary(ll_path; FT = Float64,
                                       Nx = 24, Ny = 13, Nz = 4, nwindow = 2)

            # Build a C4 CS target geometry with a tempdir regridder cache
            # so the test is hermetic and does not pollute the user's cache.
            cfg_grid = Dict{String, Any}(
                "Nc" => 4,
                "regridder_cache_dir" => joinpath(dir, "cr_cache"),
            )
            cs_grid = build_target_geometry(Val(:cubed_sphere), cfg_grid, Float64)

            regrid_ll_binary_to_cs(ll_path, cs_grid, cs_path;
                                    FT           = Float64,
                                    met_interval = 3600.0,
                                    dt           = 900.0,
                                    mass_basis   = :dry)

            @test isfile(cs_path)

            # `inspect_binary` must round-trip a CS binary written by the
            # regridder. This also exercises the plan-40-review fix to
            # `_peek_grid_type` + the CS `binary_capabilities` method.
            caps = inspect_binary(cs_path; io = devnull)
            @test caps.grid_type === :cubed_sphere
            @test caps.mass_basis === :dry
            @test caps.advection === true
            @test caps.surface_pressure === true
            @test caps.tm5_convection === false
            @test caps.replay_gate === false     # CS writer does not emit deltas

            # Numerical invariants: load via the CS reader and check the
            # total air mass and post-balance cm/m.
            reader = CubedSphereBinaryReader(cs_path; FT = Float64)
            @test reader.header.Nc == 4
            @test reader.header.npanel == 6
            @test reader.header.nwindow == meta.nwindow
            @test reader.header.nlevel == meta.Nz

            # For each window, probe cm/m on the interior panel and confirm
            # global mass agrees with the LL total within conservative-regrid
            # tolerance. We use the raw data segmented by `_cs_section_offset`
            # through the reader's own slicing helpers via a compact ad-hoc
            # walk — load_cs_transport_window! would be the canonical route
            # but is driver-owned; testing at the reader level keeps the
            # dependency minimal.
            Nc = reader.header.Nc
            Nz = reader.header.nlevel
            npanel = reader.header.npanel

            for win in 1:reader.header.nwindow
                # Offset within the window for the `m` section (first in
                # payload_sections for CS: [m, am, bm, cm, ps]).
                # elems_per_window × (win − 1) gets us to this window's base.
                base = reader.header.elems_per_window * (win - 1)
                n_m = npanel * Nc * Nc * Nz
                m_slice = @view reader.data[base + 1 : base + n_m]
                m_total_cs = sum(Float64, m_slice)

                # Conservative LL→CS regrid preserves total mass to the
                # weight-map resolution. Tolerance: 1e-6 relative.
                @test isapprox(m_total_cs, meta.m_total / reader.header.nwindow;
                               rtol = 1e-6)

                # `cm` section sits after m + am + bm.
                n_am = npanel * (Nc + 1) * Nc * Nz
                n_bm = npanel * Nc * (Nc + 1) * Nz
                n_cm = npanel * Nc * Nc * (Nz + 1)
                cm_offset = base + n_m + n_am + n_bm
                cm_slice = @view reader.data[cm_offset + 1 : cm_offset + n_cm]
                max_abs_cm = maximum(abs, cm_slice)

                # With zero input fluxes the reconstructed am/bm, the
                # diagnosed cm, and hence cm/m should all be at
                # Poisson-solver tolerance (far below invariant-13's 1e-7).
                m_ref = Float64(maximum(m_slice))
                @test max_abs_cm / m_ref < 1e-10
            end

            close(reader)
        end
    end

end
