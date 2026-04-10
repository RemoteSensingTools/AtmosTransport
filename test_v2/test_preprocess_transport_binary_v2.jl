#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "scripts", "preprocessing", "preprocess_era5_latlon_transport_binary_v2.jl"))
using .AtmosTransportV2

function tiny_latlon_target(FT::Type{<:AbstractFloat}=Float64)
    mesh = LatLonMesh(; FT=FT, size=(4, 3))
    area_by_lat = cell_areas_by_latitude(mesh)
    area = Array{FT}(undef, nx(mesh), ny(mesh))
    for j in 1:ny(mesh)
        @views area[:, j] .= area_by_lat[j]
    end
    return LatLonTargetGeometry(mesh, copy(mesh.λᶜ), copy(mesh.φᶜ), mesh.Δλ, mesh.Δφ, cosd.(mesh.φᶜ), area)
end

@testset "ERA5 lat-lon transport-binary v2 bridge" begin
    FT = Float64
    grid = tiny_latlon_target(FT)
    vc = AtmosTransportV2.HybridSigmaPressure(FT[0, 100, 200], FT[0, 0, 1])
    vertical = (Nz_native = 2, Nz = 2, merged_vc = vc)
    storage = allocate_window_storage(2, FT; include_qv=true)

    Nx, Ny, Nz = nlon(grid), nlat(grid), vertical.Nz
    storage.all_m[1] = fill(FT(1), Nx, Ny, Nz)
    storage.all_am[1] = fill(FT(2), Nx + 1, Ny, Nz)
    storage.all_bm[1] = fill(FT(3), Nx, Ny + 1, Nz)
    storage.all_cm[1] = fill(FT(4), Nx, Ny, Nz + 1)
    storage.all_ps[1] = fill(FT(95000), Nx, Ny)
    storage.all_qv_start[1] = fill(FT(0.01), Nx, Ny, Nz)

    storage.all_m[2] = fill(FT(10), Nx, Ny, Nz)
    storage.all_am[2] = fill(FT(20), Nx + 1, Ny, Nz)
    storage.all_bm[2] = fill(FT(30), Nx, Ny + 1, Nz)
    storage.all_cm[2] = fill(FT(40), Nx, Ny, Nz + 1)
    storage.all_ps[2] = fill(FT(96000), Nx, Ny)
    storage.all_qv_start[2] = fill(FT(0.02), Nx, Ny, Nz)

    last_hour_next = (
        m = fill(FT(100), Nx, Ny, Nz),
        am = fill(FT(200), Nx + 1, Ny, Nz),
        bm = fill(FT(300), Nx, Ny + 1, Nz),
        cm = fill(FT(400), Nx, Ny, Nz + 1),
        qv = fill(FT(0.03), Nx, Ny, Nz),
    )

    settings = (
        include_qv = true,
        mass_basis = :moist,
        met_interval = 3600.0,
        dt = 900.0,
        half_dt = 450.0,
        output_float_type = FT,
    )

    mktemp() do path, io
        close(io)
        pack_seconds, write_seconds = write_transport_binary_v2_from_storage!(path, grid, vertical, storage, settings, last_hour_next)
        @test pack_seconds >= 0
        @test write_seconds >= 0

        reader = TransportBinaryReader(path; FT=FT)
        @test grid_type(reader) == :latlon
        @test horizontal_topology(reader) == :structureddirectional
        @test source_flux_sampling(reader) == :window_start_endpoint
        @test air_mass_sampling(reader) == :window_start_endpoint
        @test flux_sampling(reader) == :window_start_endpoint
        @test flux_kind(reader) == :substep_mass_amount
        @test humidity_sampling(reader) == :window_endpoints
        @test delta_semantics(reader) == :forward_window_endpoint_difference
        @test has_qv_endpoints(reader)
        @test has_flux_delta(reader)

        qv_pair_1 = load_qv_pair_window!(reader, 1)
        qv_pair_2 = load_qv_pair_window!(reader, 2)
        @test qv_pair_1.qv_start == storage.all_qv_start[1]
        @test qv_pair_1.qv_end == storage.all_qv_start[2]
        @test qv_pair_2.qv_end == last_hour_next.qv

        deltas_1 = load_flux_delta_window!(reader, 1)
        deltas_2 = load_flux_delta_window!(reader, 2)
        @test deltas_1.dm == storage.all_m[2] .- storage.all_m[1]
        @test deltas_1.dam == storage.all_am[2] .- storage.all_am[1]
        @test deltas_2.dm == last_hour_next.m .- storage.all_m[2]
        @test deltas_2.dcm == last_hour_next.cm .- storage.all_cm[2]

        close(reader)
    end
end
