#!/usr/bin/env julia

using Printf
using TOML
using NCDatasets

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
include(joinpath(@__DIR__, "..", "run_transport_binary_v2.jl"))

function usage()
    println("""
    Usage:
      julia --project=. scripts/diagnostics/debug_reduced_gaussian_instability.jl step-trace <config.toml> [nsteps]
      julia --project=. scripts/diagnostics/debug_reduced_gaussian_instability.jl outgoing-cfl <config.toml> [nwindows]
      julia --project=. scripts/diagnostics/debug_reduced_gaussian_instability.jl compare-snapshots <latlon.nc> <reduced.nc>

    Commands:
      step-trace        Run a reduced or lat-lon config step-by-step and print state extrema.
      outgoing-cfl      Report per-window outgoing-mass ratios for reduced-grid horizontal/vertical sweeps.
      compare-snapshots Compare lat-lon and reduced snapshot NetCDFs along equatorial and meridional cuts.
    """)
end

function _load_run_config(path::AbstractString)
    cfg = TOML.parsefile(expanduser(path))
    input_cfg = get(cfg, "input", Dict{String, Any}())
    run_cfg = get(cfg, "run", Dict{String, Any}())
    num_cfg = get(cfg, "numerics", Dict{String, Any}())
    init_cfg = get(cfg, "init", Dict{String, Any}())

    binary_paths = [expanduser(String(p)) for p in get(input_cfg, "binary_paths", String[])]
    isempty(binary_paths) && throw(ArgumentError("no input.binary_paths configured in $(path)"))

    FT = Symbol(get(num_cfg, "float_type", "Float64")) == :Float32 ? Float32 : Float64
    scheme_name = Symbol(lowercase(String(get(run_cfg, "scheme", "upwind"))))
    tracer_name = Symbol(get(run_cfg, "tracer_name", "CO2"))
    start_window = Int(get(run_cfg, "start_window", 1))
    stop_window = get(run_cfg, "stop_window", nothing)
    return (; cfg, binary_paths, FT, scheme_name, tracer_name, start_window, stop_window, init_cfg)
end

function _env_flag(name::AbstractString, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "true" : "false")))
    return raw in ("1", "true", "yes", "on")
end

function _build_sim(path::AbstractString; stop_window_override=nothing,
                    reset_air_mass_each_window::Bool = true)
    rcfg = _load_run_config(path)
    driver = AtmosTransport.TransportBinaryDriver(first(rcfg.binary_paths); FT=rcfg.FT, arch=AtmosTransport.CPU())
    stop_window = stop_window_override === nothing ?
        (rcfg.stop_window === nothing ? AtmosTransport.total_windows(driver) : Int(rcfg.stop_window)) :
        Int(stop_window_override)
    model = make_model(driver; FT=rcfg.FT,
                       scheme_name=rcfg.scheme_name,
                       tracer_name=rcfg.tracer_name,
                       init_cfg=rcfg.init_cfg)
    sim = AtmosTransport.DrivenSimulation(model, driver;
                                            start_window=rcfg.start_window,
                                            stop_window=stop_window,
                                            initialize_air_mass=true,
                                            reset_air_mass_each_window=reset_air_mass_each_window)
    return (; rcfg..., driver, model, sim)
end

function _q_extrema(state, tracer_name::Symbol)
    rm = getproperty(state.tracers, tracer_name)
    m = state.air_mass
    q = rm ./ max.(m, eps(eltype(m)))
    return minimum(q), maximum(q)
end

function step_trace(config_path::AbstractString, nsteps::Int)
    reset_each_window = _env_flag("AT_DEBUG_RESET_AIR_MASS_EACH_WINDOW", true)
    run = _build_sim(config_path; reset_air_mass_each_window=reset_each_window)
    step_fn = getfield(AtmosTransport, Symbol("step!"))
    rm0 = AtmosTransport.total_mass(run.model.state, run.tracer_name)

    println("reset_air_mass_each_window = $(reset_each_window)")
    println("iter win sub time_h min_m max_m min_q max_q tracer_drift")
    for step in 0:nsteps
        qmin, qmax = _q_extrema(run.model.state, run.tracer_name)
        m = run.model.state.air_mass
        drift = (AtmosTransport.total_mass(run.model.state, run.tracer_name) - rm0) / rm0
        @printf("%3d %3d %3d %6.2f %.3e %.3e %.3e %.3e %.3e\n",
                run.sim.iteration,
                AtmosTransport.window_index(run.sim),
                AtmosTransport.substep_index(run.sim),
                run.sim.time / 3600,
                minimum(m), maximum(m), qmin, qmax, drift)
        if step == nsteps || !all(isfinite.(m)) || minimum(m) <= zero(eltype(m)) || !all(isfinite.(getproperty(run.model.state.tracers, run.tracer_name)))
            break
        end
        step_fn(run.sim)
    end
    close(run.driver)
    return nothing
end

function _face_endpoints(mesh::AtmosTransport.ReducedGaussianMesh)
    nface = AtmosTransport.nfaces(mesh)
    left = Vector{Int32}(undef, nface)
    right = Vector{Int32}(undef, nface)
    for f in 1:nface
        l, r = AtmosTransport.face_cells(mesh, f)
        left[f] = Int32(l)
        right[f] = Int32(r)
    end
    return left, right
end

function _max_outgoing_ratio(horizontal_flux::AbstractMatrix{FT},
                             air_mass::AbstractMatrix{FT},
                             left::Vector{Int32},
                             right::Vector{Int32},
                             face_range) where FT
    nc = size(air_mass, 1)
    nz = size(air_mass, 2)
    outgoing = zeros(FT, nc)
    max_ratio = zero(FT)
    where_cell = 0
    where_level = 0
    where_out = zero(FT)
    where_m = zero(FT)

    for k in 1:nz
        outgoing .= zero(FT)
        @inbounds for f in face_range
            l = left[f]
            r = right[f]
            if l > 0 && r > 0
                F = horizontal_flux[f, k]
                if F >= zero(FT)
                    outgoing[l] += F
                else
                    outgoing[r] -= F
                end
            end
        end
        ratios = outgoing ./ max.(air_mass[:, k], eps(FT))
        ratio_k, idx = findmax(ratios)
        if ratio_k > max_ratio
            max_ratio = ratio_k
            where_cell = idx
            where_level = k
            where_out = outgoing[idx]
            where_m = air_mass[idx, k]
        end
    end

    return max_ratio, where_cell, where_level, where_out, where_m
end

function _max_vertical_outgoing_ratio(cm::AbstractMatrix{FT}, air_mass::AbstractMatrix{FT}) where FT
    nc = size(air_mass, 1)
    nz = size(air_mass, 2)
    max_ratio = zero(FT)
    where_cell = 0
    where_level = 0
    where_out = zero(FT)
    where_m = zero(FT)

    @inbounds for k in 1:nz
        out_k = max.(cm[:, k], zero(FT)) .+ max.(-cm[:, k + 1], zero(FT))
        ratios = out_k ./ max.(air_mass[:, k], eps(FT))
        ratio_k, idx = findmax(ratios)
        if ratio_k > max_ratio
            max_ratio = ratio_k
            where_cell = idx
            where_level = k
            where_out = out_k[idx]
            where_m = air_mass[idx, k]
        end
    end

    return max_ratio, where_cell, where_level, where_out, where_m
end

function outgoing_cfl(config_path::AbstractString, nwindows::Int)
    run = _build_sim(config_path; stop_window_override=max(1, nwindows))
    mesh = run.model.grid.horizontal
    mesh isa AtmosTransport.ReducedGaussianMesh ||
        throw(ArgumentError("outgoing-cfl only supports ReducedGaussianMesh configs"))

    left, right = _face_endpoints(mesh)
    zonal_faces = 1:AtmosTransport.ncells(mesh)
    meridional_faces = (AtmosTransport.ncells(mesh) + 1):AtmosTransport.nfaces(mesh)
    total_windows = min(nwindows, AtmosTransport.total_windows(run.driver))

    println("win horiz_total zonal_only merid_only vertical")
    for win in 1:total_windows
        window = AtmosTransport.load_transport_window(run.driver, win)
        h_all = _max_outgoing_ratio(window.fluxes.horizontal_flux, window.air_mass, left, right, eachindex(left))
        h_zonal = _max_outgoing_ratio(window.fluxes.horizontal_flux, window.air_mass, left, right, zonal_faces)
        h_merid = _max_outgoing_ratio(window.fluxes.horizontal_flux, window.air_mass, left, right, meridional_faces)
        v_all = _max_vertical_outgoing_ratio(window.fluxes.cm, window.air_mass)

        @printf("%3d  %.6f  %.6f  %.6f  %.6f\n",
                win, h_all[1], h_zonal[1], h_merid[1], v_all[1])
        @printf("     h_all:   cell=%d level=%d out=%.3e m=%.3e\n", h_all[2], h_all[3], h_all[4], h_all[5])
        @printf("     h_zonal: cell=%d level=%d out=%.3e m=%.3e\n", h_zonal[2], h_zonal[3], h_zonal[4], h_zonal[5])
        @printf("     h_merid: cell=%d level=%d out=%.3e m=%.3e\n", h_merid[2], h_merid[3], h_merid[4], h_merid[5])
        @printf("     v_all:   cell=%d level=%d out=%.3e m=%.3e\n", v_all[2], v_all[3], v_all[4], v_all[5])
    end

    close(run.driver)
    return nothing
end

@inline _wrap_lon360(x) = mod(x, 360)
@inline _wrap_lon180(x) = mod(x + 180, 360) - 180

function _nearest_reduced_equator_profile(lon_cell, lat_cell, values)
    target_lat = lat_cell[argmin(abs.(lat_cell))]
    mask = abs.(lat_cell .- target_lat) .< 1e-10
    lons = _wrap_lon360.(lon_cell[mask])
    vals = values[mask]
    order = sortperm(lons)
    return lons[order], vals[order], target_lat
end

function _sample_periodic_nearest(lons_src, vals_src, lons_target)
    out = similar(lons_target, eltype(vals_src))
    @inbounds for i in eachindex(lons_target)
        lon_t = _wrap_lon360(lons_target[i])
        best_j = 1
        best_d = Inf
        for j in eachindex(lons_src)
            d = abs(mod(lons_src[j] - lon_t + 180, 360) - 180)
            if d < best_d
                best_d = d
                best_j = j
            end
        end
        out[i] = vals_src[best_j]
    end
    return out
end

function _reduced_meridian_profile(lon_cell, lat_cell, values; lon0=0.0)
    ring_lats = sort(unique(lat_cell))
    prof = similar(ring_lats)
    for (j, lat) in enumerate(ring_lats)
        mask = abs.(lat_cell .- lat) .< 1e-10
        ring_lons = lon_cell[mask]
        ring_vals = values[mask]
        best_i = argmin(abs.(_wrap_lon180.(ring_lons .- lon0)))
        prof[j] = ring_vals[best_i]
    end
    return ring_lats, prof
end

function _sample_lat_nearest(lats_src, vals_src, lats_target)
    out = similar(lats_target, eltype(vals_src))
    @inbounds for i in eachindex(lats_target)
        best_j = argmin(abs.(lats_src .- lats_target[i]))
        out[i] = vals_src[best_j]
    end
    return out
end

function compare_snapshots(latlon_path::AbstractString, reduced_path::AbstractString)
    ds_ll = NCDataset(expanduser(latlon_path))
    ds_rg = NCDataset(expanduser(reduced_path))
    try
        times_ll = vec(ds_ll["time_hours"][:])
        times_rg = vec(ds_rg["time_hours"][:])
        lon_ll = vec(ds_ll["lon"][:])
        lat_ll = vec(ds_ll["lat"][:])
        col_ll = ds_ll["co2_column_mean"][:, :, :]
        col_rg = ds_rg["co2_column_mean"][:, :]
        lon_rg = vec(ds_rg["lon_cell"][:])
        lat_rg = vec(ds_rg["lat_cell"][:])

        common_times = sort(collect(intersect(Set(round.(Int, times_ll)), Set(round.(Int, times_rg)))))
        isempty(common_times) && throw(ArgumentError("no common snapshot times between files"))

        j_eq = argmin(abs.(lat_ll))
        i_lon0 = argmin(abs.(_wrap_lon180.(lon_ll)))

        println("time_h eq_max_abs eq_at_lon mer_max_abs mer_at_lat")
        for hour in common_times
            it_ll = findfirst(t -> round(Int, t) == hour, times_ll)
            it_rg = findfirst(t -> round(Int, t) == hour, times_rg)
            ll_eq = vec(col_ll[it_ll, j_eq, :])
            rg_lons, rg_eq, rg_lat = _nearest_reduced_equator_profile(lon_rg, lat_rg, vec(col_rg[it_rg, :]))
            rg_eq_on_ll = _sample_periodic_nearest(rg_lons, rg_eq, _wrap_lon360.(lon_ll))

            lon_mask = abs.(_wrap_lon180.(lon_ll)) .<= 40
            eq_diff = abs.(ll_eq[lon_mask] .- rg_eq_on_ll[lon_mask])
            eq_idx_local = argmax(eq_diff)
            eq_lons = lon_ll[lon_mask]

            ll_mer = vec(col_ll[it_ll, :, i_lon0])
            rg_lats, rg_mer = _reduced_meridian_profile(lon_rg, lat_rg, vec(col_rg[it_rg, :]))
            rg_mer_on_ll = _sample_lat_nearest(rg_lats, rg_mer, lat_ll)

            lat_mask = abs.(lat_ll) .<= 30
            mer_diff = abs.(ll_mer[lat_mask] .- rg_mer_on_ll[lat_mask])
            mer_idx_local = argmax(mer_diff)
            mer_lats = lat_ll[lat_mask]

            @printf("%6d  %.3e  %7.2f  %.3e  %7.2f\n",
                    hour,
                    eq_diff[eq_idx_local], eq_lons[eq_idx_local],
                    mer_diff[mer_idx_local], mer_lats[mer_idx_local])
        end
        rg_eq_lat = _nearest_reduced_equator_profile(lon_rg, lat_rg, vec(col_rg[findfirst(t -> round(Int, t) == first(common_times), times_rg), :]))[3]
        println("equatorial reduced ring latitude = $(lat_ll[j_eq]) latlon, $(rg_eq_lat) reduced")
    finally
        close(ds_ll)
        close(ds_rg)
    end
    return nothing
end

function main(args)
    isempty(args) && (usage(); return 1)
    cmd = args[1]
    if cmd == "step-trace"
        length(args) < 2 && throw(ArgumentError("step-trace requires <config.toml>"))
        nsteps = length(args) >= 3 ? parse(Int, args[3]) : 48
        step_trace(args[2], nsteps)
    elseif cmd == "outgoing-cfl"
        length(args) < 2 && throw(ArgumentError("outgoing-cfl requires <config.toml>"))
        nwindows = length(args) >= 3 ? parse(Int, args[3]) : 6
        outgoing_cfl(args[2], nwindows)
    elseif cmd == "compare-snapshots"
        length(args) < 3 && throw(ArgumentError("compare-snapshots requires <latlon.nc> <reduced.nc>"))
        compare_snapshots(args[2], args[3])
    else
        throw(ArgumentError("unknown command: $(cmd)"))
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
