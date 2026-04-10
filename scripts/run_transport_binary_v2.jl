#!/usr/bin/env julia

using Logging
using Printf
using TOML

include(joinpath(@__DIR__, "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2

@inline wrapped_longitude_distance(lon, lon0) = abs(mod(lon - lon0 + 180, 360) - 180)

function build_initial_mixing_ratio(air_mass::AbstractArray{FT}, mesh::AtmosTransportV2.LatLonMesh{FT}, cfg) where FT
    kind = Symbol(lowercase(String(get(cfg, "kind", "uniform"))))
    background = FT(get(cfg, "background", 4.0e-4))
    if kind === :uniform
        return fill(background, size(air_mass))
    elseif kind === :gaussian_blob
        lon0 = FT(get(cfg, "lon0_deg", 0.0))
        lat0 = FT(get(cfg, "lat0_deg", 0.0))
        sigma_lon = FT(get(cfg, "sigma_lon_deg", 10.0))
        sigma_lat = FT(get(cfg, "sigma_lat_deg", 10.0))
        amplitude = FT(get(cfg, "amplitude", background))
        q = Array{FT}(undef, size(air_mass))
        for k in axes(q, 3), j in axes(q, 2), i in axes(q, 1)
            dlon = wrapped_longitude_distance(mesh.λᶜ[i], lon0)
            dlat = mesh.φᶜ[j] - lat0
            q[i, j, k] = background + amplitude * exp(-FT(0.5) * ((dlon / sigma_lon)^2 + (dlat / sigma_lat)^2))
        end
        return q
    else
        throw(ArgumentError("unsupported init.kind=$(kind) for LatLonMesh"))
    end
end

function build_initial_mixing_ratio(air_mass::AbstractArray{FT}, mesh::AtmosTransportV2.ReducedGaussianMesh{FT}, cfg) where FT
    kind = Symbol(lowercase(String(get(cfg, "kind", "uniform"))))
    background = FT(get(cfg, "background", 4.0e-4))
    if kind === :uniform
        return fill(background, size(air_mass))
    elseif kind === :gaussian_blob
        lon0 = FT(get(cfg, "lon0_deg", 0.0))
        lat0 = FT(get(cfg, "lat0_deg", 0.0))
        sigma_lon = FT(get(cfg, "sigma_lon_deg", 10.0))
        sigma_lat = FT(get(cfg, "sigma_lat_deg", 10.0))
        amplitude = FT(get(cfg, "amplitude", background))
        q = Array{FT}(undef, size(air_mass))
        for j in 1:nrings(mesh)
            lats = mesh.latitudes[j]
            lons = ring_longitudes(mesh, j)
            for i in eachindex(lons)
                c = cell_index(mesh, i, j)
                dlon = wrapped_longitude_distance(lons[i], lon0)
                dlat = lats - lat0
                value = background + amplitude * exp(-FT(0.5) * ((dlon / sigma_lon)^2 + (dlat / sigma_lat)^2))
                @views q[c, :] .= value
            end
        end
        return q
    else
        throw(ArgumentError("unsupported init.kind=$(kind) for ReducedGaussianMesh"))
    end
end

function make_model(driver::AtmosTransportV2.TransportBinaryDriver; FT::Type{<:AbstractFloat}, scheme_name::Symbol, tracer_name::Symbol, init_cfg)
    grid = AtmosTransportV2.driver_grid(driver)
    window = AtmosTransportV2.load_transport_window(driver, 1)
    air_mass = copy(window.air_mass)
    q = build_initial_mixing_ratio(air_mass, grid.horizontal, init_cfg)
    rm = q .* air_mass

    basis_type = AtmosTransportV2.air_mass_basis(driver) == :dry ? AtmosTransportV2.DryBasis : AtmosTransportV2.MoistBasis
    tracer_tuple = NamedTuple{(tracer_name,)}((rm,))
    state = if basis_type === AtmosTransportV2.DryBasis
        AtmosTransportV2.CellState{AtmosTransportV2.DryBasis, typeof(air_mass), typeof(tracer_tuple)}(air_mass, tracer_tuple)
    else
        AtmosTransportV2.CellState{AtmosTransportV2.MoistBasis, typeof(air_mass), typeof(tracer_tuple)}(air_mass, tracer_tuple)
    end
    fluxes = AtmosTransportV2.allocate_face_fluxes(grid.horizontal, AtmosTransportV2.nlevels(grid); FT=FT, basis=basis_type)
    scheme = scheme_name == :slopes ? AtmosTransportV2.SlopesScheme() : AtmosTransportV2.UpwindScheme()
    return AtmosTransportV2.TransportModel(state, fluxes, grid, scheme)
end

function run_sequence(binary_paths::Vector{String}, cfg)
    FT = Symbol(get(get(cfg, "numerics", Dict{String, Any}()), "float_type", "Float64")) == :Float32 ? Float32 : Float64
    scheme_name = Symbol(lowercase(String(get(get(cfg, "run", Dict{String, Any}()), "scheme", "upwind"))))
    tracer_name = Symbol(get(get(cfg, "run", Dict{String, Any}()), "tracer_name", "CO2"))
    start_window = Int(get(get(cfg, "run", Dict{String, Any}()), "start_window", 1))
    stop_window_override = get(get(cfg, "run", Dict{String, Any}()), "stop_window", nothing)
    init_cfg = get(cfg, "init", Dict{String, Any}())

    isempty(binary_paths) && throw(ArgumentError("no binary_paths configured"))

    first_driver = AtmosTransportV2.TransportBinaryDriver(first(binary_paths); FT=FT, arch=AtmosTransportV2.CPU())
    model = make_model(first_driver; FT=FT, scheme_name=scheme_name, tracer_name=tracer_name, init_cfg=init_cfg)
    m0 = AtmosTransportV2.total_air_mass(model.state)
    rm0 = AtmosTransportV2.total_mass(model.state, tracer_name)

    for (idx, path) in enumerate(binary_paths)
        driver = idx == 1 ? first_driver : AtmosTransportV2.TransportBinaryDriver(path; FT=FT, arch=AtmosTransportV2.CPU())
        stop_window = stop_window_override === nothing ? AtmosTransportV2.total_windows(driver) : Int(stop_window_override)
        initialize_air_mass = idx == 1
        sim = AtmosTransportV2.DrivenSimulation(model, driver;
                               start_window=start_window,
                               stop_window=stop_window,
                               initialize_air_mass=initialize_air_mass)
        if !initialize_air_mass
            boundary_rel = maximum(abs.(model.state.air_mass .- sim.window.air_mass)) / max(maximum(abs.(sim.window.air_mass)), eps(FT))
            @info @sprintf("Boundary air-mass mismatch before %s: %.3e", basename(path), boundary_rel)
        end
        @info @sprintf("Running %s with %s on %s (%d windows)", basename(path), scheme_name, summary(AtmosTransportV2.driver_grid(driver).horizontal), stop_window - start_window + 1)
        AtmosTransportV2.run!(sim)
        close(driver)
    end

    m1 = AtmosTransportV2.total_air_mass(model.state)
    rm1 = AtmosTransportV2.total_mass(model.state, tracer_name)
    @info @sprintf("Final air-mass change vs initial state:  %.3e", (m1 - m0) / m0)
    @info @sprintf("Final tracer-mass drift:                 %.3e", (rm1 - rm0) / rm0)
    return model
end

function main()
    base_logger = ConsoleLogger(stderr, Logging.Info; show_limited=false)
    global_logger(base_logger)

    isempty(ARGS) && error("Usage: julia --project=. scripts/run_transport_binary_v2.jl config.toml")
    cfg = TOML.parsefile(expanduser(ARGS[1]))
    binary_paths = [expanduser(String(p)) for p in get(get(cfg, "input", Dict{String, Any}()), "binary_paths", String[])]
    run_sequence(binary_paths, cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
