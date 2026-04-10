#!/usr/bin/env julia

using Test
using NCDatasets

include(joinpath(@__DIR__, "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2
include(joinpath(@__DIR__, "..", "scripts", "run_transport_binary_v2.jl"))

function write_sequence_binary(path::AbstractString; FT::Type{<:AbstractFloat}=Float64, scale::Real=1)
    Nx, Ny, Nz = 4, 3, 2
    mesh = AtmosTransportV2.LatLonMesh(; FT=FT, Nx=Nx, Ny=Ny)
    vertical = AtmosTransportV2.HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosTransportV2.AtmosGrid(mesh, vertical, AtmosTransportV2.CPU(); FT=FT)

    windows = [begin
        m = fill(FT(scale), Nx, Ny, Nz)
        am = zeros(FT, Nx + 1, Ny, Nz)
        bm = zeros(FT, Nx, Ny + 1, Nz)
        cm = zeros(FT, Nx, Ny, Nz + 1)
        ps = fill(FT(95000), Nx, Ny)
        qv_start = fill(FT(0.01), Nx, Ny, Nz)
        qv_end = fill(FT(0.01), Nx, Ny, Nz)
        dam = zeros(FT, Nx + 1, Ny, Nz)
        dbm = zeros(FT, Nx, Ny + 1, Nz)
        dcm = zeros(FT, Nx, Ny, Nz + 1)
        dm = zeros(FT, Nx, Ny, Nz)
        (; m, am, bm, cm, ps, qv_start, qv_end, dam, dbm, dcm, dm)
    end]

    AtmosTransportV2.write_transport_binary(path, grid, windows;
                           FT=FT,
                           dt_met_seconds=3600.0,
                           half_dt_seconds=450.0,
                           steps_per_window=1,
                           mass_basis=:moist,
                           source_flux_sampling=:window_start_endpoint,
                           flux_sampling=:window_constant,
                           extra_header=Dict(
                               "poisson_balance_target_scale" => 0.5,
                               "poisson_balance_target_semantics" => "forward_window_mass_difference / (2 * steps_per_window)",
                           ))
    return nothing
end

field_value(lon_deg, lat_deg) = 3.5e-4 + 1e-6 * wrapped_longitude_360(lon_deg) + 2e-6 * lat_deg

function write_test_ic_file(path::AbstractString)
    ds = NCDataset(path, "c")
    try
        defDim(ds, "longitude", 4)
        defDim(ds, "latitude", 2)
        defDim(ds, "level", 3)
        defDim(ds, "hlevel", 4)

        vlon = defVar(ds, "longitude", Float64, ("longitude",))
        vlat = defVar(ds, "latitude", Float64, ("latitude",))
        vlev = defVar(ds, "level", Float64, ("level",))
        vhlev = defVar(ds, "hlevel", Float64, ("hlevel",))
        vap = defVar(ds, "ap", Float64, ("hlevel",))
        vbp = defVar(ds, "bp", Float64, ("hlevel",))
        vps = defVar(ds, "Psurf", Float64, ("longitude", "latitude"))
        vco2 = defVar(ds, "CO2", Float64, ("longitude", "latitude", "level"))

        lon = [-135.0, -45.0, 45.0, 135.0]
        lat = [45.0, -45.0]  # descending on purpose
        vlon[:] = lon
        vlat[:] = lat
        vlev[:] = [0.0, 1.0, 2.0]
        vhlev[:] = [0.0, 1.0, 2.0, 3.0]
        vap[:] = [0.0, 0.0, 0.0, 0.0]
        vbp[:] = [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0]
        vps[:, :] = fill(9.0e4, 4, 2)

        raw = Array{Float64}(undef, 4, 2, 3)
        for k in 1:3, j in 1:2, i in 1:4
            raw[i, j, k] = field_value(lon[i], lat[j])
        end
        vco2[:, :, :] = raw
        vco2.attrib["units"] = "mol mol-1"
    finally
        close(ds)
    end
    return nothing
end

function write_test_surface_flux_file(path::AbstractString)
    ds = NCDataset(path, "c")
    try
        defDim(ds, "longitude", 4)
        defDim(ds, "latitude", 2)
        defDim(ds, "time", 2)

        vlon = defVar(ds, "longitude", Float64, ("longitude",))
        vlat = defVar(ds, "latitude", Float64, ("latitude",))
        vtime = defVar(ds, "time", Float64, ("time",))
        vtotal = defVar(ds, "TOTAL", Float64, ("longitude", "latitude", "time"))
        varea = defVar(ds, "cell_area", Float64, ("longitude", "latitude"))

        vlon[:] = [-135.0, -45.0, 45.0, 135.0]
        vlat[:] = [45.0, -45.0]  # descending on purpose
        vtime[:] = [1.0, 2.0]
        raw1 = zeros(Float64, 4, 2)
        raw1[1, 1] = 7 * SECONDS_PER_MONTH
        raw2 = 2 .* raw1
        vtotal[:, :, 1] = raw1
        vtotal[:, :, 2] = raw2
        varea[:, :] = reshape(Float64.(1:8), 4, 2)
        vtotal.attrib["units"] = "kgCO2/month/m2"
    finally
        close(ds)
    end
    return nothing
end

@testset "transport-binary sequence runner" begin
    mktempdir() do dir
        path1 = joinpath(dir, "day1.bin")
        path2 = joinpath(dir, "day2.bin")
        write_sequence_binary(path1; scale=1)
        write_sequence_binary(path2; scale=1)

        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [path1, path2]),
            "numerics" => Dict("float_type" => "Float64"),
            "run" => Dict("scheme" => "upwind", "tracer_name" => "CO2", "start_window" => 1),
            "init" => Dict("kind" => "uniform", "background" => 4.0e-4),
        )

        model = run_sequence([path1, path2], cfg)
        @test AtmosTransportV2.total_air_mass(model.state) ≈ 24.0 atol=eps(Float64) * 100
        @test AtmosTransportV2.total_mass(model.state, :CO2) ≈ 24.0 * 4.0e-4 atol=eps(Float64) * 100
    end
end

@testset "file-based initial conditions support latlon and reduced grids" begin
    mktempdir() do dir
        ic_path = joinpath(dir, "test_ic.nc")
        write_test_ic_file(ic_path)

        init_cfg = Dict{String, Any}(
            "kind" => "file",
            "file" => ic_path,
            "variable" => "CO2",
        )

        vertical = AtmosTransportV2.HybridSigmaPressure(Float64[0, 0, 0], Float64[1, 0.5, 0])

        latlon_mesh = AtmosTransportV2.LatLonMesh(; FT=Float64, Nx=4, Ny=2)
        latlon_grid = AtmosTransportV2.AtmosGrid(latlon_mesh, vertical, AtmosTransportV2.CPU(); FT=Float64)
        air_mass_ll = ones(Float64, 4, 2, 2)
        q_ll = build_initial_mixing_ratio(air_mass_ll, latlon_grid, init_cfg)
        for j in 1:2, i in 1:4
            expected = field_value(latlon_mesh.λᶜ[i], latlon_mesh.φᶜ[j])
            @test q_ll[i, j, 1] ≈ expected atol=1e-12
            @test q_ll[i, j, 2] ≈ expected atol=1e-12
        end

        reduced_mesh = AtmosTransportV2.ReducedGaussianMesh([-45.0, 45.0], [4, 4]; FT=Float64)
        reduced_grid = AtmosTransportV2.AtmosGrid(reduced_mesh, vertical, AtmosTransportV2.CPU(); FT=Float64)
        air_mass_rg = ones(Float64, AtmosTransportV2.ncells(reduced_mesh), 2)
        q_rg = build_initial_mixing_ratio(air_mass_rg, reduced_grid, init_cfg)
        for j in 1:AtmosTransportV2.nrings(reduced_mesh)
            lons = AtmosTransportV2.ring_longitudes(reduced_mesh, j)
            lat = reduced_mesh.latitudes[j]
            for i in eachindex(lons)
                c = AtmosTransportV2.cell_index(reduced_mesh, i, j)
                expected = field_value(lons[i], lat)
                @test q_rg[c, 1] ≈ expected atol=1e-12
                @test q_rg[c, 2] ≈ expected atol=1e-12
            end
        end
    end
end

@testset "surface-flux loader supports latlon and reduced grids" begin
    mktempdir() do dir
        flux_path = joinpath(dir, "gridfed.nc")
        write_test_surface_flux_file(flux_path)

        flux_cfg = Dict{String, Any}(
            "kind" => "gridfed_fossil_co2",
            "file" => flux_path,
            "time_index" => 1,
        )

        vertical = AtmosTransportV2.HybridSigmaPressure(Float64[0, 0], Float64[1, 0])
        native_total = 7.0

        latlon_mesh = AtmosTransportV2.LatLonMesh(; FT=Float64, Nx=4, Ny=2)
        latlon_grid = AtmosTransportV2.AtmosGrid(latlon_mesh, vertical, AtmosTransportV2.CPU(); FT=Float64)
        ll_source = build_surface_flux_source(latlon_grid, :fossil_co2, flux_cfg, Float64)
        @test sum(ll_source.cell_mass_rate) ≈ native_total rtol=1e-12

        reduced_mesh = AtmosTransportV2.ReducedGaussianMesh([-67.5, -22.5, 22.5, 67.5], [8, 8, 8, 8]; FT=Float64)
        reduced_grid = AtmosTransportV2.AtmosGrid(reduced_mesh, vertical, AtmosTransportV2.CPU(); FT=Float64)
        rg_source = build_surface_flux_source(reduced_grid, :fossil_co2, flux_cfg, Float64)
        @test sum(rg_source.cell_mass_rate) ≈ native_total rtol=1e-12
    end
end

@testset "transport-binary sequence runner supports multi-tracer surface flux sources" begin
    mktempdir() do dir
        path = joinpath(dir, "day1.bin")
        flux_path = joinpath(dir, "gridfed.nc")
        write_sequence_binary(path; scale=1)
        write_test_surface_flux_file(flux_path)

        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [path]),
            "numerics" => Dict("float_type" => "Float64"),
            "run" => Dict("scheme" => "upwind", "start_window" => 1),
            "tracers" => Dict(
                "natural_co2" => Dict(
                    "init" => Dict("kind" => "uniform", "background" => 4.0e-4),
                ),
                "fossil_co2" => Dict(
                    "init" => Dict("kind" => "uniform", "background" => 0.0),
                    "surface_flux" => Dict(
                        "kind" => "gridfed_fossil_co2",
                        "file" => flux_path,
                        "time_index" => 1,
                    ),
                ),
            ),
        )

        model = run_sequence([path], cfg)
        mesh = AtmosTransportV2.LatLonMesh(; FT=Float64, Nx=4, Ny=3)

        @test AtmosTransportV2.total_air_mass(model.state) ≈ 24.0 atol=eps(Float64) * 100
        @test AtmosTransportV2.total_mass(model.state, :natural_co2) ≈ 24.0 * 4.0e-4 atol=eps(Float64) * 100
        @test AtmosTransportV2.total_mass(model.state, :fossil_co2) ≈ 7.0 * 3600.0 rtol=1e-12
    end
end
