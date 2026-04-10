#!/usr/bin/env julia

using Logging
using Printf
using TOML
using NCDatasets
using Adapt

include(joinpath(@__DIR__, "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2

@inline wrapped_longitude_distance(lon, lon0) = abs(mod(lon - lon0 + 180, 360) - 180)
@inline wrapped_longitude_360(lon) = mod(lon, 360)

@inline _init_kind(cfg) = Symbol(lowercase(String(get(cfg, "kind", "uniform"))))
@inline _is_file_init_kind(kind::Symbol) = kind in (:file, :netcdf, :file_field, :catrine_co2)
@inline _use_gpu(cfg) = Bool(get(get(cfg, "architecture", Dict{String, Any}()), "use_gpu", false))

struct FileInitialConditionSource{FT}
    raw           :: Array{FT, 3}
    lon           :: Vector{Float64}
    lat           :: Vector{Float64}
    ap            :: Vector{Float64}
    bp            :: Vector{Float64}
    psurf         :: Matrix{Float64}
    needs_vinterp :: Bool
end

function _ic_find_coord(ds, candidates::Vector{String})
    for name in candidates
        haskey(ds, name) && return name
    end
    return nothing
end

function _bilinear_bracket(val::Float64, arr::Vector{Float64})
    N = length(arr)
    N == 1 && return (1, 0.0)
    val <= arr[1] && return (1, 0.0)
    val >= arr[N] && return (N, 0.0)
    lo, hi = 1, N
    while hi - lo > 1
        mid = (lo + hi) >> 1
        if arr[mid] <= val
            lo = mid
        else
            hi = mid
        end
    end
    denom = arr[hi] - arr[lo]
    w = denom > 0 ? (val - arr[lo]) / denom : 0.0
    return lo, w
end

function _periodic_bilinear_bracket(val::Float64, arr::Vector{Float64})
    N = length(arr)
    N == 1 && return (1, 0.0)
    Δ = arr[2] - arr[1]
    Δ > 0 || throw(ArgumentError("longitude coordinate must be strictly increasing"))
    u = mod(val - arr[1], 360.0) / Δ
    ilo0 = floor(Int, u)
    return mod1(ilo0 + 1, N), u - ilo0
end

function _horizontal_interp_weights(lon::Real, lat::Real, lon_src::Vector{Float64}, lat_src::Vector{Float64})
    lon_m = wrapped_longitude_360(Float64(lon))
    ilo, wx = _periodic_bilinear_bracket(lon_m, lon_src)
    ihi = ilo == length(lon_src) ? 1 : ilo + 1
    jlo, wy = _bilinear_bracket(Float64(lat), lat_src)
    jhi = min(jlo + 1, length(lat_src))
    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy
    return ilo, ihi, jlo, jhi, w00, w10, w01, w11
end

function _sample_bilinear_profile!(dest::AbstractVector{FT},
                                   raw::Array{FT, 3},
                                   lon_src::Vector{Float64},
                                   lat_src::Vector{Float64},
                                   lon::Real,
                                   lat::Real) where FT
    ilo, ihi, jlo, jhi, w00, w10, w01, w11 = _horizontal_interp_weights(lon, lat, lon_src, lat_src)
    @inbounds for k in eachindex(dest)
        dest[k] = FT(w00 * raw[ilo, jlo, k] +
                     w10 * raw[ihi, jlo, k] +
                     w01 * raw[ilo, jhi, k] +
                     w11 * raw[ihi, jhi, k])
    end
    return nothing
end

function _sample_bilinear_scalar(raw::Matrix{Float64},
                                 lon_src::Vector{Float64},
                                 lat_src::Vector{Float64},
                                 lon::Real,
                                 lat::Real)
    ilo, ihi, jlo, jhi, w00, w10, w01, w11 = _horizontal_interp_weights(lon, lat, lon_src, lat_src)
    return w00 * raw[ilo, jlo] +
           w10 * raw[ihi, jlo] +
           w01 * raw[ilo, jhi] +
           w11 * raw[ihi, jhi]
end

function _resolve_file_init(cfg, kind::Symbol)
    default_file, default_variable = if kind === :catrine_co2
        ("~/data/AtmosTransport/catrine/InitialConditions/startCO2_202112010000.nc", "CO2")
    else
        ("", "")
    end
    file = expanduser(String(get(cfg, "file", default_file)))
    variable = String(get(cfg, "variable", default_variable))
    isempty(file) && throw(ArgumentError("file-based init.kind=$(kind) requires init.file"))
    isempty(variable) && throw(ArgumentError("file-based init.kind=$(kind) requires init.variable"))
    time_index = Int(get(cfg, "time_index", 1))
    return file, variable, time_index
end

function _load_file_initial_condition_source(cfg, ::Type{FT}, Nz_target::Integer) where FT
    kind = _init_kind(cfg)
    file, variable, time_index = _resolve_file_init(cfg, kind)
    isfile(file) || throw(ArgumentError("initial-condition file not found: $file"))

    ds = NCDataset(file)
    try
        lon_var = _ic_find_coord(ds, ["lon", "longitude", "x"])
        lat_var = _ic_find_coord(ds, ["lat", "latitude", "y"])
        lev_var = _ic_find_coord(ds, ["lev", "level", "plev", "z", "hybrid", "nhym"])
        isnothing(lon_var) && throw(ArgumentError("could not find longitude coordinate in $file"))
        isnothing(lat_var) && throw(ArgumentError("could not find latitude coordinate in $file"))
        isnothing(lev_var) && throw(ArgumentError("could not find vertical coordinate in $file"))
        haskey(ds, variable) || throw(ArgumentError("variable '$variable' not found in $file"))

        lon_src = Float64.(ds[lon_var][:])
        lat_src = Float64.(ds[lat_var][:])
        lev_src = Float64.(ds[lev_var][:])

        raw_var = ds[variable]
        raw = if ndims(raw_var) == 4
            FT.(nomissing(raw_var[:, :, :, time_index], zero(FT)))
        elseif ndims(raw_var) == 3
            FT.(nomissing(raw_var[:, :, :], zero(FT)))
        else
            throw(ArgumentError("variable '$variable' must be 3D or 4D, got ndims=$(ndims(raw_var))"))
        end

        has_hybrid = haskey(ds, "ap") && haskey(ds, "bp") && haskey(ds, "Psurf")
        ap = has_hybrid ? Float64.(ds["ap"][:]) : Float64[]
        bp = has_hybrid ? Float64.(ds["bp"][:]) : Float64[]
        psurf = has_hybrid ? Float64.(nomissing(ds["Psurf"][:, :], 101325.0)) : zeros(Float64, 0, 0)

        if length(lat_src) > 1 && lat_src[1] > lat_src[end]
            raw = raw[:, end:-1:1, :]
            lat_src = reverse(lat_src)
            if has_hybrid
                psurf = psurf[:, end:-1:1]
            end
        end

        if minimum(lon_src) < 0
            split = findfirst(>=(0), lon_src)
            if split !== nothing
                idx = vcat(split:length(lon_src), 1:split-1)
                lon_src = mod.(lon_src[idx], 360.0)
                raw = raw[idx, :, :]
                if has_hybrid
                    psurf = psurf[idx, :]
                end
            end
        end

        if length(lev_src) > 1 && lev_src[1] > lev_src[end]
            raw = raw[:, :, end:-1:1]
            lev_src = reverse(lev_src)
            if has_hybrid
                ap = reverse(ap)
                bp = reverse(bp)
            end
        end

        needs_vinterp = has_hybrid && size(raw, 3) != Nz_target
        return FileInitialConditionSource{FT}(raw, lon_src, lat_src, ap, bp, psurf, needs_vinterp)
    finally
        close(ds)
    end
end

function _interpolate_log_pressure_profile!(dest::AbstractVector{FT},
                                            src_q::AbstractVector{FT},
                                            air_mass_col,
                                            ap::Vector{Float64},
                                            bp::Vector{Float64},
                                            ps_src::Float64,
                                            area::Float64,
                                            g::Float64) where FT
    Nsrc = length(src_q)
    Nz = length(dest)

    src_p_half = Vector{Float64}(undef, Nsrc + 1)
    @inbounds for k in 1:(Nsrc + 1)
        src_p_half[k] = ap[k] + bp[k] * ps_src
    end
    src_p_mid = Vector{Float64}(undef, Nsrc)
    @inbounds for k in 1:Nsrc
        src_p_mid[k] = 0.5 * (src_p_half[k] + src_p_half[k + 1])
    end

    tgt_p_half = Vector{Float64}(undef, Nz + 1)
    tgt_p_half[1] = 0.0
    @inbounds for k in 1:Nz
        dp = Float64(air_mass_col[k]) * g / area
        tgt_p_half[k + 1] = tgt_p_half[k] + dp
    end

    ks = 1
    @inbounds for k in 1:Nz
        p_tgt = 0.5 * (tgt_p_half[k] + tgt_p_half[k + 1])
        if p_tgt >= src_p_mid[1]
            dest[k] = src_q[1]
        elseif p_tgt <= src_p_mid[end]
            dest[k] = src_q[end]
        else
            while ks < Nsrc && src_p_mid[ks + 1] > p_tgt
                ks += 1
            end
            lp1 = log(max(src_p_mid[ks], floatmin(Float64)))
            lp2 = log(max(src_p_mid[ks + 1], floatmin(Float64)))
            lpt = log(max(p_tgt, floatmin(Float64)))
            w = (lpt - lp1) / (lp2 - lp1)
            dest[k] = FT(src_q[ks] + w * (src_q[ks + 1] - src_q[ks]))
        end
    end
    return nothing
end

function _copy_profile!(dest::AbstractVector{FT}, src_q::AbstractVector{FT}) where FT
    fill!(dest, zero(FT))
    Nz_use = min(length(dest), length(src_q))
    @views copyto!(dest[1:Nz_use], src_q[1:Nz_use])
    return nothing
end

function ensure_gpu_runtime!(cfg)
    _use_gpu(cfg) || return false
    Sys.isapple() && throw(ArgumentError("transport-binary v2 GPU path is only wired for CUDA hosts"))
    if !isdefined(Main, :CUDA)
        Core.eval(Main, :(using CUDA))
    end
    CUDA = Core.eval(Main, :CUDA)
    Base.invokelatest(getproperty(CUDA, :functional)) ||
        throw(ArgumentError("CUDA runtime is not functional on this host"))
    Base.invokelatest(getproperty(CUDA, :allowscalar), false)
    return true
end

function backend_array_adapter(cfg)
    if _use_gpu(cfg)
        ensure_gpu_runtime!(cfg)
        return getproperty(Core.eval(Main, :CUDA), :CuArray)
    end
    return Array
end

function backend_label(cfg)
    if _use_gpu(cfg)
        ensure_gpu_runtime!(cfg)
        CUDA = Core.eval(Main, :CUDA)
        device_name = Base.invokelatest(getproperty(CUDA, :name), Base.invokelatest(getproperty(CUDA, :device)))
        return "GPU (CUDA, $(device_name))"
    end
    return "CPU"
end

function synchronize_backend!(cfg)
    if _use_gpu(cfg)
        ensure_gpu_runtime!(cfg)
        Base.invokelatest(getproperty(Core.eval(Main, :CUDA), :synchronize))
    end
    return nothing
end

function build_initial_mixing_ratio(air_mass::AbstractArray{FT}, mesh::AtmosTransportV2.LatLonMesh{FT}, cfg) where FT
    kind = _init_kind(cfg)
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
    kind = _init_kind(cfg)
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
        for j in 1:AtmosTransportV2.nrings(mesh)
            lats = mesh.latitudes[j]
            lons = AtmosTransportV2.ring_longitudes(mesh, j)
            for i in eachindex(lons)
                c = AtmosTransportV2.cell_index(mesh, i, j)
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

function build_initial_mixing_ratio(air_mass::AbstractArray{FT},
                                    grid::AtmosTransportV2.AtmosGrid{<:AtmosTransportV2.LatLonMesh},
                                    cfg) where FT
    kind = _init_kind(cfg)
    _is_file_init_kind(kind) || return build_initial_mixing_ratio(air_mass, grid.horizontal, cfg)

    source = _load_file_initial_condition_source(cfg, FT, size(air_mass, 3))
    mesh = grid.horizontal
    q = Array{FT}(undef, size(air_mass))
    src_q = Vector{FT}(undef, size(source.raw, 3))
    g = Float64(AtmosTransportV2.gravity(grid))

    for j in axes(q, 2)
        area = Float64(AtmosTransportV2.cell_area(mesh, 1, j))
        lat = mesh.φᶜ[j]
        for i in axes(q, 1)
            lon = mesh.λᶜ[i]
            _sample_bilinear_profile!(src_q, source.raw, source.lon, source.lat, lon, lat)
            if source.needs_vinterp
                ps_src = _sample_bilinear_scalar(source.psurf, source.lon, source.lat, lon, lat)
                _interpolate_log_pressure_profile!(@view(q[i, j, :]), src_q, @view(air_mass[i, j, :]),
                                                  source.ap, source.bp, ps_src, area, g)
            else
                _copy_profile!(@view(q[i, j, :]), src_q)
            end
        end
    end

    return q
end

function build_initial_mixing_ratio(air_mass::AbstractArray{FT},
                                    grid::AtmosTransportV2.AtmosGrid{<:AtmosTransportV2.ReducedGaussianMesh},
                                    cfg) where FT
    kind = _init_kind(cfg)
    _is_file_init_kind(kind) || return build_initial_mixing_ratio(air_mass, grid.horizontal, cfg)

    source = _load_file_initial_condition_source(cfg, FT, size(air_mass, 2))
    mesh = grid.horizontal
    q = Array{FT}(undef, size(air_mass))
    src_q = Vector{FT}(undef, size(source.raw, 3))
    g = Float64(AtmosTransportV2.gravity(grid))

    for j in 1:AtmosTransportV2.nrings(mesh)
        lat = mesh.latitudes[j]
        lons = AtmosTransportV2.ring_longitudes(mesh, j)
        for i in eachindex(lons)
            c = AtmosTransportV2.cell_index(mesh, i, j)
            lon = lons[i]
            _sample_bilinear_profile!(src_q, source.raw, source.lon, source.lat, lon, lat)
            if source.needs_vinterp
                ps_src = _sample_bilinear_scalar(source.psurf, source.lon, source.lat, lon, lat)
                area = Float64(AtmosTransportV2.cell_area(mesh, c))
                _interpolate_log_pressure_profile!(@view(q[c, :]), src_q, @view(air_mass[c, :]),
                                                  source.ap, source.bp, ps_src, area, g)
            else
                _copy_profile!(@view(q[c, :]), src_q)
            end
        end
    end

    return q
end

function make_model(driver::AtmosTransportV2.TransportBinaryDriver;
                    FT::Type{<:AbstractFloat},
                    scheme_name::Symbol,
                    tracer_name::Symbol,
                    init_cfg,
                    cfg)
    grid = AtmosTransportV2.driver_grid(driver)
    window = AtmosTransportV2.load_transport_window(driver, 1)
    air_mass = copy(window.air_mass)
    q = build_initial_mixing_ratio(air_mass, grid, init_cfg)
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
    model = AtmosTransportV2.TransportModel(state, fluxes, grid, scheme)
    adaptor = backend_array_adapter(cfg)
    return adaptor === Array ? model : Adapt.adapt(adaptor, model)
end

function run_sequence(binary_paths::Vector{String}, cfg)
    FT = Symbol(get(get(cfg, "numerics", Dict{String, Any}()), "float_type", "Float64")) == :Float32 ? Float32 : Float64
    run_cfg = get(cfg, "run", Dict{String, Any}())
    scheme_name = Symbol(lowercase(String(get(run_cfg, "scheme", "upwind"))))
    tracer_name = Symbol(get(run_cfg, "tracer_name", "CO2"))
    start_window = Int(get(run_cfg, "start_window", 1))
    stop_window_override = get(run_cfg, "stop_window", nothing)
    reset_air_mass_each_window = Bool(get(run_cfg, "reset_air_mass_each_window", false))
    init_cfg = get(cfg, "init", Dict{String, Any}())

    isempty(binary_paths) && throw(ArgumentError("no binary_paths configured"))
    ensure_gpu_runtime!(cfg)

    first_driver = AtmosTransportV2.TransportBinaryDriver(first(binary_paths); FT=FT, arch=AtmosTransportV2.CPU())
    model = make_model(first_driver; FT=FT, scheme_name=scheme_name, tracer_name=tracer_name, init_cfg=init_cfg, cfg=cfg)
    m0 = AtmosTransportV2.total_air_mass(model.state)
    rm0 = AtmosTransportV2.total_mass(model.state, tracer_name)
    @info "Backend: $(backend_label(cfg))"

    for (idx, path) in enumerate(binary_paths)
        driver = idx == 1 ? first_driver : AtmosTransportV2.TransportBinaryDriver(path; FT=FT, arch=AtmosTransportV2.CPU())
        stop_window = stop_window_override === nothing ? AtmosTransportV2.total_windows(driver) : Int(stop_window_override)
        initialize_air_mass = idx == 1
        sim = AtmosTransportV2.DrivenSimulation(model, driver;
                               start_window=start_window,
                               stop_window=stop_window,
                               initialize_air_mass=initialize_air_mass,
                               reset_air_mass_each_window=reset_air_mass_each_window)
        if !initialize_air_mass
            boundary_rel = maximum(abs.(model.state.air_mass .- sim.window.air_mass)) / max(maximum(abs.(sim.window.air_mass)), eps(FT))
            @info @sprintf("Boundary air-mass mismatch before %s: %.3e", basename(path), boundary_rel)
        end
        @info @sprintf("Running %s with %s on %s (%d windows)", basename(path), scheme_name, summary(AtmosTransportV2.driver_grid(driver).horizontal), stop_window - start_window + 1)
        synchronize_backend!(cfg)
        t0 = time()
        AtmosTransportV2.run!(sim)
        synchronize_backend!(cfg)
        @info @sprintf("Finished %s in %.2f s", basename(path), time() - t0)
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
