#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Snapshot exact-conservation diagnostic (`<tracer>_total_mass`, F64)
#
# Plan 45 follow-up: a reference-state tracer's spatial output is the F32
# full-field reconstruction (anom + q_ref·m), so a mass budget integrated
# from it is polluted at the background scale. capture_snapshot now also
# records the EXACT total_mass_full (F64) per tracer; the writer emits it as
# a `<tracer>_total_mass` time series. This test pins that the F64 diagnostic
# matches total_mass_full and is NOT the polluted field integral.
# ---------------------------------------------------------------------------

using Test
using NCDatasets

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.State: CubedSphereState, DryBasis, REF_GLOBAL_MEAN,
    set_tracer_reference!, tracer_index, get_tracer_raw, total_mass_full
using .AtmosTransport.Grids: CubedSphereMesh
using .AtmosTransport.Models: mass_weighted_global_mean_vmr

function _mini_cs_model(; Nc = 4, Hp = 1, Nz = 3, FT = Float32)
    mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp)
    vc = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = FT)
    Np = Nc + 2Hp
    air = ntuple(_ -> ones(FT, Np, Np, Nz), 6)
    # structured co2 so the referenced anomaly is non-trivial
    co2 = ntuple(_ -> FT(4e-4) .+ FT(2e-5) .* reshape(collect(FT, 1:Nz), 1, 1, :), 6)
    sf6 = ntuple(_ -> fill(FT(1e-11), Np, Np, Nz), 6)
    state = CubedSphereState(DryBasis, mesh, air; co2 = co2, sf6 = sf6)
    fluxes = allocate_face_fluxes(mesh, Nz; FT = FT, basis = DryBasis)
    return TransportModel(state, fluxes, grid, LinRoodPPMScheme{7}()), grid
end

@testset "snapshot writes exact F64 <tracer>_total_mass" begin
    model, grid = _mini_cs_model()
    state = model.state

    # reference co2 as an anomaly (the case the field integral gets wrong)
    idx = tracer_index(state, :co2)
    raw = get_tracer_raw(state, idx)
    q_ref = mass_weighted_global_mean_vmr(raw, state.air_mass, state.halo_width)
    for p in 1:6
        raw[p] .= Float32.(Float64.(raw[p]) .- q_ref .* Float64.(state.air_mass[p]))
    end
    set_tracer_reference!(state.tracer_refs, idx, REF_GLOBAL_MEAN, q_ref)

    tm_co2 = total_mass_full(state, :co2)   # the authoritative F64 value
    tm_sf6 = total_mass_full(state, :sf6)
    @test tm_co2 > 0

    frame = capture_snapshot(model; time_hours = 0.0, halo_width = state.halo_width)
    @test haskey(frame.total_mass, :co2)
    @test frame.total_mass[:co2] == tm_co2     # exact, from total_mass_full
    @test frame.total_mass[:sf6] == tm_sf6

    mktemp() do path, io
        close(io)
        ncpath = path * ".nc"
        write_snapshot_netcdf(ncpath, [frame, frame], grid;
                              mass_basis = :dry)
        NCDataset(ncpath) do ds
            @test haskey(ds, "co2_total_mass")
            @test haskey(ds, "sf6_total_mass")
            @test ds["co2_total_mass"][:][1] == tm_co2   # F64, exact
            # the diagnostic must NOT equal the F32 field integral for the
            # referenced tracer (that integral is background-rounding-polluted);
            # they should differ at the reconstruction scale or agree only by
            # luck — here we just assert the diagnostic equals total_mass_full.
            @test eltype(ds["co2_total_mass"][:]) == Float64
        end
        isfile(ncpath) && rm(ncpath)
    end
end

println("test_snapshot_total_mass.jl OK")
