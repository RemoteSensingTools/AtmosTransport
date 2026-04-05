#!/usr/bin/env julia
# ============================================================================
# LL vs TM5 first-step mass-evolution audit
#
# Runs the real lat-lon transport path through `run!`, captures scalar stage
# metrics for the first dynamic intervals, compares the LL state against
# TM5-reference mass-flow formulas, and writes:
#   1. JSON summary table of all captured stages
#   2. NetCDF file with selected 2D snapshots
#
# Usage:
#   julia --project=. scripts/diagnostics/diagnose_ll_transport_sweep.jl \
#       --config config/runs/era5_v4_test_1win.toml \
#       --windows 2 --dyn-intervals 2 \
#       --output /tmp/era5_v4_ll_audit.nc \
#       --tm5-reference cpu
# ============================================================================

using TOML

function _usage()
    println(stderr, "Usage: julia --project=. scripts/diagnostics/diagnose_ll_transport_sweep.jl [options]")
    println(stderr, "")
    println(stderr, "Options:")
    println(stderr, "  --config <path>          Run configuration TOML (required)")
    println(stderr, "  --windows <n>            Max windows to run (default: 2)")
    println(stderr, "  --dyn-intervals <n>      Dynamic intervals to capture (default: 2)")
    println(stderr, "  --output <path>          Output NetCDF path (default: /tmp/<config>_ll_audit.nc)")
    println(stderr, "  --tm5-reference <mode>   cpu|off (default: cpu)")
    println(stderr, "  --flat-ic [value]        Use uniform wet-VMR IC; default is mean of configured IC")
    println(stderr, "  --flat-dry-ic [value]    Use uniform dry-VMR IC; default is mean of configured IC")
    exit(1)
end

function _parse_cli(args::Vector{String})
    opts = Dict{String, Any}(
        "config" => "",
        "windows" => 2,
        "dyn_intervals" => 2,
        "output" => "",
        "tm5_reference" => "cpu",
        "flat_ic" => nothing,
        "flat_dry_ic" => nothing,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            _usage()
        elseif arg == "--config"
            i == length(args) && _usage()
            opts["config"] = args[i + 1]
            i += 2
        elseif startswith(arg, "--config=")
            opts["config"] = split(arg, "=", limit=2)[2]
            i += 1
        elseif arg == "--windows"
            i == length(args) && _usage()
            opts["windows"] = parse(Int, args[i + 1])
            i += 2
        elseif startswith(arg, "--windows=")
            opts["windows"] = parse(Int, split(arg, "=", limit=2)[2])
            i += 1
        elseif arg == "--dyn-intervals"
            i == length(args) && _usage()
            opts["dyn_intervals"] = parse(Int, args[i + 1])
            i += 2
        elseif startswith(arg, "--dyn-intervals=")
            opts["dyn_intervals"] = parse(Int, split(arg, "=", limit=2)[2])
            i += 1
        elseif arg == "--output"
            i == length(args) && _usage()
            opts["output"] = args[i + 1]
            i += 2
        elseif startswith(arg, "--output=")
            opts["output"] = split(arg, "=", limit=2)[2]
            i += 1
        elseif arg == "--tm5-reference"
            if i < length(args) && !startswith(args[i + 1], "--")
                opts["tm5_reference"] = lowercase(args[i + 1])
                i += 2
            else
                opts["tm5_reference"] = "cpu"
                i += 1
            end
        elseif startswith(arg, "--tm5-reference=")
            opts["tm5_reference"] = lowercase(split(arg, "=", limit=2)[2])
            i += 1
        elseif arg == "--flat-ic"
            if i < length(args) && !startswith(args[i + 1], "--")
                val = args[i + 1]
                opts["flat_ic"] = lowercase(val) == "auto" ? "auto" : parse(Float64, val)
                i += 2
            else
                opts["flat_ic"] = "auto"
                i += 1
            end
        elseif startswith(arg, "--flat-ic=")
            val = split(arg, "=", limit=2)[2]
            opts["flat_ic"] = lowercase(val) == "auto" ? "auto" : parse(Float64, val)
            i += 1
        elseif arg == "--flat-dry-ic"
            if i < length(args) && !startswith(args[i + 1], "--")
                val = args[i + 1]
                opts["flat_dry_ic"] = lowercase(val) == "auto" ? "auto" : parse(Float64, val)
                i += 2
            else
                opts["flat_dry_ic"] = "auto"
                i += 1
            end
        elseif startswith(arg, "--flat-dry-ic=")
            val = split(arg, "=", limit=2)[2]
            opts["flat_dry_ic"] = lowercase(val) == "auto" ? "auto" : parse(Float64, val)
            i += 1
        else
            println(stderr, "Unknown option: $arg")
            _usage()
        end
    end

    isempty(opts["config"]) && _usage()
    opts["flat_ic"] !== nothing && opts["flat_dry_ic"] !== nothing &&
        error("Use only one of --flat-ic or --flat-dry-ic")
    return opts
end

const OPTS = _parse_cli(ARGS)
const RAW_CONFIG = TOML.parsefile(OPTS["config"])

if get(get(RAW_CONFIG, "architecture", Dict()), "use_gpu", false)
    if Sys.isapple()
        using Metal
    else
        using CUDA
        CUDA.allowscalar(false)
    end
end

using Downloads
using JSON3
using NCDatasets
using Printf
using Dates
using Statistics
using AtmosTransport
using AtmosTransport.IO: build_model_from_config
using AtmosTransport.Grids: LatitudeLongitudeGrid
import AtmosTransport.Models: run!, LLTransportAudit, LL_AUDIT

function _default_output_path(config_path::String)
    stem = replace(basename(config_path), ".toml" => "")
    return "/tmp/$(stem)_ll_audit.nc"
end

function _summary_path(nc_path::String)
    return replace(nc_path, r"\.nc$" => ".json")
end

function _deepcopy_config(cfg::Dict)
    return deepcopy(cfg)
end

function _configured_primary_tracer(cfg::Dict)
    tracers_cfg = get(cfg, "tracers", Dict{String, Any}())
    haskey(tracers_cfg, "co2") && return :co2
    isempty(tracers_cfg) && return :co2
    return Symbol(first(keys(tracers_cfg)))
end

function _resolve_tracer_ic_source(cfg::Dict, tracer::Symbol)
    ic_cfg = get(cfg, "initial_conditions", Dict{String, Any}())
    tracer_key = String(tracer)

    if haskey(ic_cfg, tracer_key) && ic_cfg[tracer_key] isa Dict
        tracer_ic = ic_cfg[tracer_key]
        haskey(tracer_ic, "file") || error("Tracer $tracer does not have a file-backed IC in config")
        return (
            file=expanduser(String(tracer_ic["file"])),
            variable=String(get(tracer_ic, "variable", uppercase(tracer_key))),
            time_index=Int(get(tracer_ic, "time_index", 1)),
        )
    elseif haskey(ic_cfg, "file")
        return (
            file=expanduser(String(ic_cfg["file"])),
            variable=String(get(ic_cfg, tracer_key, uppercase(tracer_key))),
            time_index=Int(get(ic_cfg, "time_index", 1)),
        )
    end

    error("Could not resolve file-backed initial condition for tracer $tracer")
end

function _configured_ic_mean(cfg::Dict, tracer::Symbol)
    src = _resolve_tracer_ic_source(cfg, tracer)
    isfile(src.file) || error("Initial condition file not found: $(src.file)")
    ds = NCDataset(src.file)
    try
        haskey(ds, src.variable) || error("IC variable $(src.variable) not found in $(src.file)")
        raw_var = ds[src.variable]
        raw = if ndims(raw_var) == 4
            Float64.(nomissing(raw_var[:, :, :, src.time_index], NaN))
        elseif ndims(raw_var) == 3
            Float64.(nomissing(raw_var[:, :, :], NaN))
        else
            error("IC variable $(src.variable) is $(ndims(raw_var))D, expected 3D or 4D")
        end
        vals = raw[isfinite.(raw)]
        isempty(vals) && error("Configured IC contains no finite values for tracer $tracer")
        return mean(vals)
    finally
        close(ds)
    end
end

function _apply_flat_ic!(cfg::Dict, tracer::Symbol, value::Float64; basis::Symbol=:wet)
    ic_cfg = get!(cfg, "initial_conditions", Dict{String, Any}())
    delete!(ic_cfg, "file")
    delete!(ic_cfg, "time_index")
    key = basis == :dry ? "uniform_dry_value" : "uniform_value"
    ic_cfg[String(tracer)] = Dict{String, Any}(key => value)
    return cfg
end

function _force_transport_only!(cfg::Dict; windows::Int, suppress_qv::Bool=false)
    met_cfg = get!(cfg, "met_data", Dict{String, Any}())
    met_cfg["max_windows"] = windows
    if suppress_qv
        met_cfg["disable_qv"] = true
        met_cfg["qv_directory"] = ""
    end

    adv_cfg = get!(cfg, "advection", Dict{String, Any}())
    adv_cfg["debug_first_window"] = false
    adv_cfg["debug_window"] = nothing
    adv_cfg["debug_all_substeps"] = false

    out_cfg = Dict{String, Any}()
    out_cfg["filename"] = ""
    cfg["output"] = out_cfg

    cfg["convection"] = Dict("type" => "none")
    cfg["diffusion"] = Dict("type" => "none")
    cfg["chemistry"] = Dict("type" => "none")

    tracers_cfg = get!(cfg, "tracers", Dict{String, Any}())
    for (_, tcfg_any) in tracers_cfg
        tcfg_any isa Dict || continue
        tcfg_any["emission"] = "none"
        tcfg_any["chemistry"] = "none"
    end

    return cfg
end

@inline _wrap_lon_360(lon::Float64) = lon < 0.0 ? lon + 360.0 : lon

function _point_in_ring(lon::Float64, lat::Float64, ring::Vector{NTuple{2, Float64}})
    inside = false
    n = length(ring)
    n < 3 && return false
    j = n
    @inbounds for i in 1:n
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat)
            x_cross = (xj - xi) * (lat - yi) / (yj - yi + eps(Float64)) + xi
            lon < x_cross && (inside = !inside)
        end
        j = i
    end
    return inside
end

function _parse_polygon_rings(coords, use_360::Bool)
    rings = Vector{Vector{NTuple{2, Float64}}}()
    lon_min = Inf
    lon_max = -Inf
    lat_min = Inf
    lat_max = -Inf
    for ring_coords in coords
        ring = Vector{NTuple{2, Float64}}()
        for pt in ring_coords
            lon = Float64(pt[1])
            lat = Float64(pt[2])
            use_360 && (lon = _wrap_lon_360(lon))
            push!(ring, (lon, lat))
            lon_min = min(lon_min, lon)
            lon_max = max(lon_max, lon)
            lat_min = min(lat_min, lat)
            lat_max = max(lat_max, lat)
        end
        push!(rings, ring)
    end
    return (bbox=(lon_min, lon_max, lat_min, lat_max), rings=rings)
end

function _load_land_polygons(cache_path::String, use_360::Bool)
    if !isfile(cache_path)
        mkpath(dirname(cache_path))
        Downloads.download(
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson",
            cache_path,
        )
    end

    raw = JSON3.read(read(cache_path, String))
    polys = Any[]
    for feature in raw["features"]
        geom = feature["geometry"]
        gtype = String(geom["type"])
        if gtype == "Polygon"
            push!(polys, _parse_polygon_rings(geom["coordinates"], use_360))
        elseif gtype == "MultiPolygon"
            for poly_coords in geom["coordinates"]
                push!(polys, _parse_polygon_rings(poly_coords, use_360))
            end
        end
    end
    return polys
end

function _point_on_land(lon::Float64, lat::Float64, polygons)
    for poly in polygons
        lon_min, lon_max, lat_min, lat_max = poly.bbox
        lon < lon_min && continue
        lon > lon_max && continue
        lat < lat_min && continue
        lat > lat_max && continue
        rings = poly.rings
        _point_in_ring(lon, lat, rings[1]) || continue
        in_hole = false
        for hole in Iterators.drop(rings, 1)
            if _point_in_ring(lon, lat, hole)
                in_hole = true
                break
            end
        end
        in_hole || return true
    end
    return false
end

function _build_sh_ocean_mask(lons::Vector{Float64}, lats::Vector{Float64})
    use_360 = minimum(lons) >= 0.0
    polygons = _load_land_polygons("/tmp/ne_110m_land.geojson", use_360)
    mask = falses(length(lons), length(lats))
    @info "Building SH-ocean mask for $(length(lons))×$(length(lats)) grid"
    for j in eachindex(lats)
        lat = lats[j]
        (-60.0 <= lat <= -20.0) || continue
        for i in eachindex(lons)
            lon = lons[i]
            mask[i, j] = !_point_on_land(lon, lat, polygons)
        end
    end
    return mask
end

function _select_tracer_name(model)
    names = collect(keys(model.tracers))
    :co2 in names && return :co2
    return first(names)
end

function _build_model_for_audit(raw_cfg::Dict;
                                windows::Int,
                                suppress_qv::Bool=false,
                                flat_ic_value::Union{Nothing, Float64}=nothing,
                                flat_ic_basis::Symbol=:wet)
    cfg = _deepcopy_config(raw_cfg)
    _force_transport_only!(cfg; windows, suppress_qv)
    if flat_ic_value !== nothing
        _apply_flat_ic!(cfg, _configured_primary_tracer(cfg), flat_ic_value; basis=flat_ic_basis)
    end
    return build_model_from_config(cfg), cfg
end

function _run_ll_audit(raw_cfg::Dict;
                       windows::Int,
                       dyn_intervals::Int,
                       include_tm5_reference::Bool,
                       flat_ic_value::Union{Nothing, Float64},
                       flat_ic_basis::Symbol,
                       sh_ocean_mask::BitMatrix,
                       zonal_lat_index::Int)
    model, cfg = _build_model_for_audit(raw_cfg; windows, suppress_qv=false, flat_ic_value, flat_ic_basis)
    grid = model.grid
    grid isa LatitudeLongitudeGrid || error("LL audit only supports LatitudeLongitudeGrid")

    audit = LLTransportAudit(;
        tracer=_select_tracer_name(model),
        max_windows=windows,
        max_dyn_intervals=dyn_intervals,
        capture_snapshots=true,
        include_tm5_reference=include_tm5_reference,
        sh_ocean_mask=sh_ocean_mask,
        zonal_lat_index=zonal_lat_index,
        lon_values=Float64.(grid.λᶜ_cpu),
        lat_values=Float64.(grid.φᶜ_cpu),
    )
    audit.metadata["config_path"] = OPTS["config"]
    audit.metadata["tm5_reference_mode"] = include_tm5_reference ? "cpu" : "off"
    audit.metadata["windows_requested"] = windows
    audit.metadata["dyn_intervals_requested"] = dyn_intervals
    audit.metadata["suppress_qv"] = false
    audit.metadata["flat_ic_value"] = flat_ic_value
    audit.metadata["flat_ic_basis"] = String(flat_ic_basis)

    LL_AUDIT[] = audit
    try
        run!(model)
    finally
        LL_AUDIT[] = nothing
    end
    return audit, cfg
end

function _run_ll_audit_qv_suppressed(raw_cfg::Dict;
                                     windows::Int,
                                     dyn_intervals::Int,
                                     include_tm5_reference::Bool,
                                     flat_ic_value::Union{Nothing, Float64},
                                     flat_ic_basis::Symbol,
                                     sh_ocean_mask::BitMatrix,
                                     zonal_lat_index::Int)
    model, _ = _build_model_for_audit(raw_cfg; windows, suppress_qv=true, flat_ic_value, flat_ic_basis)
    grid = model.grid
    grid isa LatitudeLongitudeGrid || error("LL audit only supports LatitudeLongitudeGrid")

    audit = LLTransportAudit(;
        tracer=_select_tracer_name(model),
        max_windows=windows,
        max_dyn_intervals=dyn_intervals,
        capture_snapshots=false,
        include_tm5_reference=include_tm5_reference,
        sh_ocean_mask=sh_ocean_mask,
        zonal_lat_index=zonal_lat_index,
        lon_values=Float64.(grid.λᶜ_cpu),
        lat_values=Float64.(grid.φᶜ_cpu),
    )
    audit.metadata["config_path"] = OPTS["config"]
    audit.metadata["tm5_reference_mode"] = include_tm5_reference ? "cpu" : "off"
    audit.metadata["windows_requested"] = windows
    audit.metadata["dyn_intervals_requested"] = dyn_intervals
    audit.metadata["suppress_qv"] = true
    audit.metadata["flat_ic_value"] = flat_ic_value
    audit.metadata["flat_ic_basis"] = String(flat_ic_basis)

    LL_AUDIT[] = audit
    try
        run!(model)
    finally
        LL_AUDIT[] = nothing
    end
    return audit
end

function _first_breach(stages)
    for stage in stages
        if stage.threshold_surface_breach || stage.threshold_column_breach
            return stage
        end
    end
    return nothing
end

function _compare_qv_runs(with_qv, without_qv)
    n = min(length(with_qv.stages), length(without_qv.stages))
    fields = (
        :m_min, :m_max, :rm_min, :rm_max,
        :surface_sh_ocean_std_ppm, :surface_zonal50_std_ppm,
        :column_sh_ocean_std_ppm, :column_zonal50_std_ppm,
        :tm5_cm_absmax_diff, :tm5_cm_rms_diff, :tm5_x_nloop_max,
    )
    max_abs_diff = 0.0
    first_diff = nothing
    for idx in 1:n
        a = with_qv.stages[idx]
        b = without_qv.stages[idx]
        for fld in fields
            av = getproperty(a, fld)
            bv = getproperty(b, fld)
            if av isa Integer && bv isa Integer
                diff = abs(Float64(av - bv))
            else
                avf = Float64(av)
                bvf = Float64(bv)
                diff = if isfinite(avf) && isfinite(bvf)
                    abs(avf - bvf)
                elseif isnan(avf) && isnan(bvf)
                    0.0
                else
                    Inf
                end
            end
            if diff > max_abs_diff
                max_abs_diff = diff
                first_diff = (stage_index=idx, label=a.label, field=String(fld), with_qv=av, without_qv=bv, abs_diff=diff)
            end
        end
    end
    return Dict(
        "matched_stages" => n,
        "max_abs_diff" => max_abs_diff,
        "identical" => max_abs_diff == 0.0,
        "first_diff" => first_diff,
    )
end

function _write_snapshot_netcdf(path::String, audit::LLTransportAudit)
    snaps = sort(collect(audit.snapshots); by=first)
    lon = audit.lon_values
    lat = audit.lat_values
    Ns = length(snaps)

    ds = NCDataset(path, "c")
    try
        defDim(ds, "lon", length(lon))
        defDim(ds, "lat", length(lat))
        defDim(ds, "snapshot", Ns)

        ds.attrib["title"] = "LL first-step transport audit snapshots"
        ds.attrib["config_path"] = String(get(audit.metadata, "config_path", ""))
        ds.attrib["tm5_reference_mode"] = String(get(audit.metadata, "tm5_reference_mode", ""))

        lon_var = defVar(ds, "lon", Float64, ("lon",))
        lat_var = defVar(ds, "lat", Float64, ("lat",))
        lon_var[:] = lon
        lat_var[:] = lat

        snapshot_stage = defVar(ds, "snapshot_stage_index", Int32, ("snapshot",))
        surface_var = defVar(ds, "surface_ppm", Float32, ("lon", "lat", "snapshot"))
        column_var = defVar(ds, "column_ppm", Float32, ("lon", "lat", "snapshot"))
        m_var = defVar(ds, "m_surface", Float32, ("lon", "lat", "snapshot"))
        cm_var = defVar(ds, "cm_lowest", Float32, ("lon", "lat", "snapshot"))

        surface_stack = fill(Float32(NaN), length(lon), length(lat), Ns)
        column_stack = fill(Float32(NaN), length(lon), length(lat), Ns)
        m_stack = fill(Float32(NaN), length(lon), length(lat), Ns)
        cm_stack = fill(Float32(NaN), length(lon), length(lat), Ns)
        stage_idx = Array{Int32}(undef, Ns)

        for (n, (idx, snap)) in enumerate(snaps)
            stage_idx[n] = Int32(idx)
            surface_stack[:, :, n] .= snap.surface_ppm
            column_stack[:, :, n] .= snap.column_ppm
            m_stack[:, :, n] .= snap.m_surface
            if snap.cm_lowest !== nothing
                cm_stack[:, :, n] .= snap.cm_lowest
            end
        end

        snapshot_stage[:] = stage_idx
        surface_var[:, :, :] = surface_stack
        column_var[:, :, :] = column_stack
        m_var[:, :, :] = m_stack
        cm_var[:, :, :] = cm_stack
    finally
        close(ds)
    end
    return path
end

_json_safe(x::Nothing) = nothing
_json_safe(x::Bool) = x
_json_safe(x::Integer) = x
_json_safe(x::AbstractString) = x
_json_safe(x::Symbol) = String(x)
_json_safe(x::AbstractFloat) = isfinite(x) ? x : nothing
_json_safe(x::NamedTuple) = Dict(String(k) => _json_safe(getproperty(x, k)) for k in keys(x))
_json_safe(x::AbstractVector) = [_json_safe(v) for v in x]
_json_safe(x::Dict) = Dict(String(k) => _json_safe(v) for (k, v) in x)
_json_safe(x) = x

_summary_get(x::NamedTuple, key::String) = getproperty(x, Symbol(key))
_summary_get(x::Dict, key::String) = x[key]

function _write_summary_json(path::String, audit::LLTransportAudit, qv_compare)
    mkpath(dirname(path))
    stage_dicts = [_json_safe(stage) for stage in audit.stages]
    payload = Dict(
        "metadata" => _json_safe(audit.metadata),
        "stage_count" => length(audit.stages),
        "snapshot_stage_indices" => sort(collect(keys(audit.snapshots))),
        "stages" => stage_dicts,
        "qv_compare" => _json_safe(qv_compare),
    )
    open(path, "w") do io
        JSON3.pretty(io, payload)
    end
    return path
end

function _print_summary(audit::LLTransportAudit, qv_compare, summary_path::String, nc_path::String)
    println("="^100)
    println("LL FIRST-STEP MASS EVOLUTION AUDIT")
    println("="^100)
    println("Captured stages: $(length(audit.stages))")
    println("Captured snapshots: $(length(audit.snapshots))")

    first_breach = _first_breach(audit.stages)
    if first_breach === nothing
        println("First threshold breach: none")
    else
        @printf("First threshold breach: stage=%s dyn=%d substep=%d micro=%d/%d surface_std=%.4f ppm column_std=%.4f ppm\n",
                first_breach.label, first_breach.dyn_interval, first_breach.substep,
                first_breach.micro, first_breach.refinement,
                first_breach.surface_sh_ocean_std_ppm,
                first_breach.column_sh_ocean_std_ppm)
    end

    if !isempty(audit.stages)
        first_stage = audit.stages[1]
        last_stage = audit.stages[end]
        @printf("Initial SH-ocean std: surface=%.6f ppm column=%.6f ppm\n",
                first_stage.surface_sh_ocean_std_ppm, first_stage.column_sh_ocean_std_ppm)
        @printf("Final captured SH-ocean std: surface=%.6f ppm column=%.6f ppm\n",
                last_stage.surface_sh_ocean_std_ppm, last_stage.column_sh_ocean_std_ppm)
        @printf("Worst hotspot captured: lon=%.2f lat=%.2f value=%.3f ppm deviation=%.3f ppm\n",
                last_stage.surface_hotspot_lon, last_stage.surface_hotspot_lat,
                last_stage.surface_hotspot_ppm, last_stage.surface_hotspot_dev_ppm)
    end

    println("QV sanity comparison:")
    println("  matched stages: $(qv_compare["matched_stages"])")
    println("  identical: $(qv_compare["identical"])")
    println("  max abs diff: $(qv_compare["max_abs_diff"])")
    if qv_compare["first_diff"] !== nothing
        fd = qv_compare["first_diff"]
        println("  first diff: stage=$(_summary_get(fd, "stage_index")) $(_summary_get(fd, "label")) field=$(_summary_get(fd, "field")) with_qv=$(_summary_get(fd, "with_qv")) without_qv=$(_summary_get(fd, "without_qv"))")
    end

    println("Summary JSON: $summary_path")
    println("Snapshot NetCDF: $nc_path")
end

function main()
    out_nc = isempty(OPTS["output"]) ? _default_output_path(OPTS["config"]) : OPTS["output"]
    summary_json = _summary_path(out_nc)
    include_tm5_reference = OPTS["tm5_reference"] != "off"
    flat_ic_basis = OPTS["flat_dry_ic"] === nothing ? :wet : :dry
    flat_ic_opt = OPTS["flat_dry_ic"] === nothing ? OPTS["flat_ic"] : OPTS["flat_dry_ic"]
    flat_ic_value = if flat_ic_opt === nothing
        nothing
    else
        tracer = _configured_primary_tracer(RAW_CONFIG)
        flat_ic_opt == "auto" ? _configured_ic_mean(RAW_CONFIG, tracer) : Float64(flat_ic_opt)
    end

    preview_model, _ = _build_model_for_audit(RAW_CONFIG;
                                              windows=OPTS["windows"],
                                              suppress_qv=false,
                                              flat_ic_value=flat_ic_value,
                                              flat_ic_basis=flat_ic_basis)
    grid = preview_model.grid
    grid isa LatitudeLongitudeGrid || error("LL audit only supports LatitudeLongitudeGrid")
    lon = Float64.(grid.λᶜ_cpu)
    lat = Float64.(grid.φᶜ_cpu)
    sh_ocean_mask = _build_sh_ocean_mask(lon, lat)
    zonal_lat_index = argmin(abs.(lat .+ 50.0))

    audit, _ = _run_ll_audit(RAW_CONFIG;
                             windows=OPTS["windows"],
                             dyn_intervals=OPTS["dyn_intervals"],
                             include_tm5_reference=include_tm5_reference,
                             flat_ic_value=flat_ic_value,
                             flat_ic_basis=flat_ic_basis,
                             sh_ocean_mask=sh_ocean_mask,
                             zonal_lat_index=zonal_lat_index)

    audit_qv_off = _run_ll_audit_qv_suppressed(RAW_CONFIG;
                                               windows=1,
                                               dyn_intervals=OPTS["dyn_intervals"],
                                               include_tm5_reference=include_tm5_reference,
                                               flat_ic_value=flat_ic_value,
                                               flat_ic_basis=flat_ic_basis,
                                               sh_ocean_mask=sh_ocean_mask,
                                               zonal_lat_index=zonal_lat_index)
    qv_compare = _compare_qv_runs(audit, audit_qv_off)

    _write_summary_json(summary_json, audit, qv_compare)
    _write_snapshot_netcdf(out_nc, audit)
    _print_summary(audit, qv_compare, summary_json, out_nc)
end

main()
