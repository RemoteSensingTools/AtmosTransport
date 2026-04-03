#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Time-aligned binary comparison: ERA5 LL v3 vs GEOS-IT C180 v4
#
# Compares preprocessed transport binaries at matched hourly UTC windows,
# verifies the source time-axis mapping used by the preprocessors, and reports:
#   - LL fine-grid and TM5-reduced donor-mass CFL diagnostics
#   - LL cm boundary residuals and merged-level z-CFL hotspots
#   - CS DELP / QV / PS sanity checks and continuity-derived cm diagnostics
#   - Cross-path per-hour summaries, latitude-band summaries, fixed probes,
#     and a ranked suspect list tied to TM5 reference routines
#
# Usage:
#   julia --project=. scripts/diagnostics/compare_transport_binaries.jl \
#       --date 2021-12-01 \
#       --era5-bin /temp1/atmos_transport/era5_daily_v3 \
#       --cs-bin /temp1/catrine/met/geosit_c180/massflux_v4_nfs/massflux_v4 \
#       --json-out /tmp/compare_20211201.json
# ---------------------------------------------------------------------------

using AtmosTransport
using AtmosTransport.Architectures: CPU
using AtmosTransport.Grids: CubedSphereGrid, compute_reduced_grid_tm5, fill_panel_halos!
using AtmosTransport.IO: MassFluxBinaryReader, load_window!,
                         CSBinaryReader, load_cs_window!, load_cs_qv_ps_window!,
                         default_met_config, build_vertical_coordinate
using Dates
using JSON3
using NCDatasets
using Printf
using Statistics
using TOML

const FT = Float32
const ERA5_HEADER_BYTES = 16384
const CS_HEADER_BYTES = 8192
const GRAV = 9.80665f0
const EARTH_RADIUS = 6.371e6

atexit() do
    try
        flush(stdout)
        flush(stderr)
    catch
    end
    ccall(:_exit, Cvoid, (Cint,), 0)
end

const DEFAULT_ERA5_RUN_CONFIG =
    "config/runs/catrine_era5_hourly_merged_dec2021.toml"
const DEFAULT_ERA5_PREPROC_CONFIG =
    "config/preprocessing/catrine_era5_daily_v3.toml"
const DEFAULT_CS_RUN_CONFIG =
    "config/runs/catrine_geosit_c180_gchp_v4_7d_perremap.toml"

const DEFAULT_PROBES = [
    (label = "eq_000E", lon =   0.0, lat =   0.0),
    (label = "nh_045N", lon =   0.0, lat =  45.0),
    (label = "nh_075N", lon =   0.0, lat =  75.0),
    (label = "sh_075S", lon =   0.0, lat = -75.0),
    (label = "atl_060N", lon = -30.0, lat =  60.0),
    (label = "pac_060N", lon =-150.0, lat =  60.0),
]

const LAT_BANDS = [
    (label = "90S_60S", min = -90.0, max = -60.0),
    (label = "60S_30S", min = -60.0, max = -30.0),
    (label = "30S_00S", min = -30.0, max =   0.0),
    (label = "00N_30N", min =   0.0, max =  30.0),
    (label = "30N_60N", min =  30.0, max =  60.0),
    (label = "60N_90N", min =  60.0, max =  90.0),
]

const TM5_REFERENCE = Dict(
    "pole_row_handling" =>
        "TM5 reduced-grid clustering/redistribution in deps/tm5/base/src/redgridZoom.F90 (read_redgrid, uni2red, red2uni). red2uni redistributes with air-mass weighting.",
    "cm_reconstruction" =>
        "TM5 hybrid-coordinate vertical closure in deps/tm5/base/src/advect_tools.F90:dynam0 uses the B-term correction when constructing vertical mass fluxes.",
    "vertical_merging" =>
        "TM5 level selection/combination uses explicit TLevelInfo + FillLevels logic in deps/tm5/base/src/grid_type_hyb.F90 and deps/tm5/base/src/tmm.F90, not a generic min-Δp heuristic.",
    "latitude_convention" =>
        "TM5 global lat-lon regions in deps/tm5/rc/include/tm5_regions.rc and reduced-grid setup in deps/tm5/base/src/redgridZoom.F90 assume pole-bounded latitude bands (e.g. 1x1 => jm=180), not extra pole-centered rows.",
    "mass_flux_dt_or_ordering" =>
        "TM5 expects internally consistent time-averaging and level ordering before transport; this script checks binary/header conventions against the preprocessors and run configs."
)

struct LLBuffers{T}
    m::Array{T, 3}
    am::Array{T, 3}
    bm::Array{T, 3}
    cm::Array{T, 3}
    ps::Array{T, 2}
end

struct CSBuffers{T}
    delp::NTuple{6, Array{T, 3}}
    am::NTuple{6, Array{T, 3}}
    bm::NTuple{6, Array{T, 3}}
    qv_start::NTuple{6, Array{T, 3}}
    qv_end::NTuple{6, Array{T, 3}}
    ps_start::NTuple{6, Array{T, 2}}
    ps_end::NTuple{6, Array{T, 2}}
    qv_halo::NTuple{6, Array{T, 3}}
    m_halo::NTuple{6, Array{T, 3}}
    cm::NTuple{6, Array{T, 3}}
end

normalize_lon(lon::Real) = mod(Float64(lon) + 180.0, 360.0) - 180.0
lon_diff(a::Real, b::Real) = abs(mod(normalize_lon(a) - normalize_lon(b) + 180.0, 360.0) - 180.0)
fmt_dt(dt::DateTime) = Dates.format(dt, dateformat"yyyy-mm-ddTHH:MM:SS")
fmt_hour(dt::DateTime) = Dates.format(dt, dateformat"yyyy-mm-dd HH:MM")

function haversine_deg(lon1, lat1, lon2, lat2)
    λ1 = deg2rad(normalize_lon(lon1))
    λ2 = deg2rad(normalize_lon(lon2))
    φ1 = deg2rad(Float64(lat1))
    φ2 = deg2rad(Float64(lat2))
    dλ = λ2 - λ1
    dφ = φ2 - φ1
    a = sin(dφ / 2)^2 + cos(φ1) * cos(φ2) * sin(dλ / 2)^2
    return 2 * EARTH_RADIUS * atan(sqrt(a), sqrt(max(0.0, 1 - a)))
end

function parse_args()
    args = Dict{String, String}()
    i = 1
    while i <= length(ARGS)
        if startswith(ARGS[i], "--")
            key = ARGS[i][3:end]
            if i < length(ARGS) && !startswith(ARGS[i + 1], "--")
                args[key] = ARGS[i + 1]
                i += 2
            else
                args[key] = "true"
                i += 1
            end
        else
            i += 1
        end
    end

    date = Date(get(args, "date", "2021-12-01"))
    topn = parse(Int, get(args, "topn", "8"))

    return (
        date = date,
        era5_bin = get(args, "era5-bin", ""),
        cs_bin = get(args, "cs-bin", ""),
        windows = get(args, "windows", ""),
        hours = get(args, "hours", ""),
        json_out = get(args, "json-out", ""),
        era5_run_config = get(args, "era5-run-config", DEFAULT_ERA5_RUN_CONFIG),
        era5_preproc_config = get(args, "era5-preprocess-config", DEFAULT_ERA5_PREPROC_CONFIG),
        cs_run_config = get(args, "cs-run-config", DEFAULT_CS_RUN_CONFIG),
        era5_source_file = get(args, "era5-source-file", ""),
        cs_raw_dir = get(args, "cs-raw-dir", ""),
        cs_gridspec = get(args, "cs-gridspec", ""),
        topn = topn,
    )
end

function parse_window_list(spec::String)
    isempty(strip(spec)) && return Int[]
    wins = Int[]
    for raw in split(spec, ',')
        token = strip(raw)
        isempty(token) && continue
        if occursin('-', token)
            a, b = split(token, '-', limit = 2)
            ia = parse(Int, strip(a))
            ib = parse(Int, strip(b))
            append!(wins, ia:ib)
        else
            push!(wins, parse(Int, token))
        end
    end
    return sort!(unique!(wins))
end

function parse_hour_list(date::Date, spec::String)
    isempty(strip(spec)) && return DateTime[]
    hours = DateTime[]
    for raw in split(spec, ',')
        token = strip(raw)
        isempty(token) && continue
        if occursin('-', token)
            a, b = split(token, '-', limit = 2)
            h1 = parse(Int, first(split(strip(a), ':')))
            h2 = parse(Int, first(split(strip(b), ':')))
            for h in h1:h2
                push!(hours, DateTime(date) + Hour(h))
            end
        else
            h = parse(Int, first(split(token, ':')))
            push!(hours, DateTime(date) + Hour(h))
        end
    end
    return sort!(unique!(hours))
end

function read_json_header(path::String, nbytes::Int)
    open(path, "r") do io
        hdr = read(io, min(nbytes, filesize(path)))
        json_end = something(findfirst(==(0x00), hdr), length(hdr) + 1) - 1
        return JSON3.read(String(hdr[1:json_end]))
    end
end

function resolve_existing(path::String)
    p = expanduser(path)
    isfile(p) || isdir(p) || error("Path not found: $p")
    return p
end

function resolve_era5_bin(path_or_dir::String, date::Date, run_cfg_path::String)
    datestr = Dates.format(date, "yyyymmdd")
    name = "era5_v3_$(datestr)_merged1000Pa_float32.bin"
    if !isempty(path_or_dir)
        p = expanduser(path_or_dir)
        return isdir(p) ? joinpath(p, name) : p
    end
    run_cfg = TOML.parsefile(run_cfg_path)
    base = expanduser(run_cfg["met_data"]["directory"])
    return joinpath(base, name)
end

function _resolve_stale_cs_dir(dir::String)
    base = basename(dir)
    parent = dirname(dir)
    return joinpath(parent, base * "_nfs", base)
end

function resolve_cs_bin(path_or_dir::String, date::Date, run_cfg_path::String)
    datestr = Dates.format(date, "yyyymmdd")
    name = "geosfp_cs_$(datestr)_float32.bin"

    if !isempty(path_or_dir)
        p = expanduser(path_or_dir)
        if isdir(p)
            return joinpath(p, name)
        end
        return p
    end

    run_cfg = TOML.parsefile(run_cfg_path)
    base = expanduser(run_cfg["met_data"]["preprocessed_dir"])
    if isdir(base)
        return joinpath(base, name)
    end
    alt = _resolve_stale_cs_dir(base)
    isdir(alt) || error("CS preprocessed directory not found: $base (also tried $alt)")
    return joinpath(alt, name)
end

function resolve_cs_raw_dir(path::String, run_cfg_path::String)
    if !isempty(path)
        return expanduser(path)
    end
    run_cfg = TOML.parsefile(run_cfg_path)
    return expanduser(run_cfg["met_data"]["netcdf_dir"])
end

function resolve_cs_gridspec(path::String, run_cfg_path::String)
    if !isempty(path)
        return expanduser(path)
    end
    run_cfg = TOML.parsefile(run_cfg_path)
    return expanduser(run_cfg["met_data"]["coord_file"])
end

function discover_era5_source_file(path::String, date::Date, preproc_cfg_path::String)
    if !isempty(path)
        return expanduser(path)
    end
    cfg = TOML.parsefile(preproc_cfg_path)
    source_dir = expanduser(cfg["input"]["massflux_dir"])
    datestr = Dates.format(date, "yyyymmdd")
    monthstr = Dates.format(date, "yyyymm")
    candidates = [
        joinpath(source_dir, "massflux_era5_spectral_$(datestr)_float32.nc"),
        joinpath(source_dir, "massflux_era5_spectral_$(monthstr)_float32.nc"),
        joinpath(source_dir, "massflux_era5_spectral_$(datestr)_hourly_float32.nc"),
    ]
    for fp in candidates
        isfile(fp) && return fp
    end
    fallback = filter(f -> endswith(f, ".nc") && contains(basename(f), monthstr),
                      readdir(source_dir; join = true))
    isempty(fallback) && error("No ERA5 source NetCDF found in $source_dir for $monthstr")
    return first(sort(fallback))
end

function discover_cs_raw_ctm_a1(raw_dir::String, date::Date, Nc::Int)
    datestr = Dates.format(date, "yyyymmdd")
    daydir = joinpath(raw_dir, datestr)
    isdir(daydir) || error("CS raw day directory not found: $daydir")
    preferred = joinpath(daydir, "GEOSIT.$(datestr).CTM_A1.C$(Nc).nc")
    isfile(preferred) && return preferred
    tag = "CTM_A1.C$(Nc)"
    for f in sort(readdir(daydir))
        if occursin(tag, f) && endswith(f, ".nc")
            return joinpath(daydir, f)
        end
    end
    error("No CTM_A1 raw file found in $daydir matching $tag")
end

function band_index(lat::Real)
    φ = Float64(lat)
    for (idx, band) in pairs(LAT_BANDS)
        if idx == length(LAT_BANDS)
            if φ >= band.min && φ <= band.max
                return idx
            end
        elseif φ >= band.min && φ < band.max
            return idx
        end
    end
    return φ < LAT_BANDS[1].min ? 1 : length(LAT_BANDS)
end

function push_topn!(vec::Vector, item, topn::Int)
    if length(vec) < topn
        push!(vec, item)
        sort!(vec, by = x -> x.value, rev = true)
    elseif item.value > vec[end].value
        vec[end] = item
        sort!(vec, by = x -> x.value, rev = true)
    end
    return nothing
end

function row_stats_finalize(sum_abs::Vector{Float64}, count::Vector{Int})
    out = Vector{Float64}(undef, length(sum_abs))
    for i in eachindex(out)
        out[i] = count[i] > 0 ? sum_abs[i] / count[i] : NaN
    end
    return out
end

function mean_abs_std_max(sum_abs::Float64, sum_sq::Float64, count::Int, max_abs::Float64)
    if count == 0
        return (mean_abs = NaN, rms = NaN, max_abs = NaN)
    end
    return (
        mean_abs = sum_abs / count,
        rms = sqrt(sum_sq / count),
        max_abs = max_abs,
    )
end

function stats_dict(sum_abs::Float64, sum_sq::Float64, count::Int, max_abs::Float64)
    stats = mean_abs_std_max(sum_abs, sum_sq, count, max_abs)
    return Dict(
        "mean_abs" => stats.mean_abs,
        "rms" => stats.rms,
        "max_abs" => stats.max_abs,
    )
end

function merge_group_info(merge_map::Vector{Int}, Nz::Int)
    group_size = ones(Int, Nz)
    native_first = collect(1:Nz)
    native_last = collect(1:Nz)
    isempty(merge_map) && return (group_size = group_size, native_first = native_first, native_last = native_last)

    group_size .= 0
    native_first .= 0
    native_last .= 0
    for (k, km) in pairs(merge_map)
        group_size[km] += 1
        native_first[km] == 0 && (native_first[km] = k)
        native_last[km] = k
    end
    for km in 1:Nz
        if native_first[km] == 0
            native_first[km] = km
            native_last[km] = km
            group_size[km] = 1
        end
    end
    return (group_size = group_size, native_first = native_first, native_last = native_last)
end

function make_ll_buffers(reader::MassFluxBinaryReader{FT}) where FT
    return LLBuffers(
        Array{FT}(undef, reader.Nx, reader.Ny, reader.Nz),
        Array{FT}(undef, reader.Nx + 1, reader.Ny, reader.Nz),
        Array{FT}(undef, reader.Nx, reader.Ny + 1, reader.Nz),
        Array{FT}(undef, reader.Nx, reader.Ny, reader.Nz + 1),
        Array{FT}(undef, reader.Nx, reader.Ny),
    )
end

function make_cs_buffers(reader::CSBinaryReader{FT}) where FT
    Nc, Nz, Hp = reader.Nc, reader.Nz, reader.Hp
    N = Nc + 2Hp
    delp = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), 6)
    bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), 6)
    qv_start = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    qv_end = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    ps_start = ntuple(_ -> zeros(FT, Nc, Nc), 6)
    ps_end = ntuple(_ -> zeros(FT, Nc, Nc), 6)
    qv_halo = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    m_halo = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6)
    return CSBuffers(delp, am, bm, qv_start, qv_end, ps_start, ps_end, qv_halo, m_halo, cm)
end

function copy_core_to_halo!(dst::Array{FT, 3}, src::Array{FT, 3}, Hp::Int) where FT
    Nc = size(src, 1)
    @views dst[Hp + 1:Hp + Nc, Hp + 1:Hp + Nc, :] .= src
    return nothing
end

function find_ll_probes(lons::AbstractVector, lats::AbstractVector)
    out = Dict{String, Tuple{Int, Int}}()
    for probe in DEFAULT_PROBES
        i_best = 1
        j_best = 1
        d_best = Inf
        for (j, lat) in pairs(lats), (i, lon) in pairs(lons)
            d = haversine_deg(probe.lon, probe.lat, lon, lat)
            if d < d_best
                d_best = d
                i_best = i
                j_best = j
            end
        end
        out[probe.label] = (i_best, j_best)
    end
    return out
end

function ll_geometry_summary(lons::AbstractVector, lats::AbstractVector)
    Nx = length(lons)
    Ny = length(lats)

    lon_spacing = Nx > 1 ? median(diff(Float64.(lons))) : 360.0
    lat_spacing = Ny > 1 ? median(diff(Float64.(lats))) : 180.0
    runtime_lon_centers = [Float64(lons[1]) + (i - 1) * lon_spacing for i in 1:Nx]
    runtime_lat_centers = [-90.0 + (j - 0.5) * (180.0 / Ny) for j in 1:Ny]

    max_lon_center_mismatch = maximum(abs.(Float64.(lons) .- runtime_lon_centers))
    max_lat_center_mismatch = maximum(abs.(Float64.(lats) .- runtime_lat_centers))
    tm5_style_ny = round(Int, 180.0 / lat_spacing)
    exact_pole_centers = isapprox(Float64(lats[1]), -90.0; atol = 1e-6) &&
                         isapprox(Float64(lats[end]), 90.0; atol = 1e-6)

    return Dict(
        "Nx" => Nx,
        "Ny" => Ny,
        "lon_spacing_deg" => lon_spacing,
        "lat_spacing_deg" => lat_spacing,
        "binary_has_exact_pole_centers" => exact_pole_centers,
        "tm5_style_ny_from_spacing" => tm5_style_ny,
        "max_runtime_lon_center_mismatch_deg" => max_lon_center_mismatch,
        "max_runtime_lat_center_mismatch_deg" => max_lat_center_mismatch,
        "binary_lat_first_deg" => Float64(lats[1]),
        "binary_lat_last_deg" => Float64(lats[end]),
        "runtime_lat_first_deg" => runtime_lat_centers[1],
        "runtime_lat_last_deg" => runtime_lat_centers[end],
        "sample_rows" => [
            Dict(
                "j" => j,
                "binary_lat_deg" => Float64(lats[j]),
                "runtime_lat_deg" => runtime_lat_centers[j],
                "mismatch_deg" => Float64(lats[j]) - runtime_lat_centers[j],
            ) for j in unique([1, min(2, Ny), max(1, Ny - 1), Ny])
        ],
    )
end

function read_cs_gridspec(path::String)
    ds = NCDataset(path, "r")
    try
        lons = Array{Float64}(ds["lons"][:, :, :])
        lats = Array{Float64}(ds["lats"][:, :, :])
        areas = haskey(ds, "areas") ? Array{Float64}(ds["areas"][:, :, :]) : nothing
        return (lons = lons, lats = lats, areas = areas)
    finally
        close(ds)
    end
end

function find_cs_probes(cs_lons::Array{Float64, 3}, cs_lats::Array{Float64, 3})
    Nc = size(cs_lons, 1)
    out = Dict{String, NamedTuple}()
    for probe in DEFAULT_PROBES
        best = (panel = 1, i = 1, j = 1, distance_m = Inf)
        for p in 1:6, j in 1:Nc, i in 1:Nc
            d = haversine_deg(probe.lon, probe.lat, cs_lons[i, j, p], cs_lats[i, j, p])
            if d < best.distance_m
                best = (panel = p, i = i, j = j, distance_m = d)
            end
        end
        out[probe.label] = best
    end
    return out
end

function era5_window_map(source_file::String, date::Date)
    out = NamedTuple[]
    ds = NCDataset(source_file, "r")
    try
        times = ds["time"][:]
        local_win = 0
        for (src_idx, t) in pairs(times)
            dt = DateTime(t)
            if Date(dt) == date
                local_win += 1
                push!(out, (
                    window = local_win,
                    source_index = src_idx,
                    raw_time = dt,
                    window_start = dt,
                    window_end = dt + Hour(1),
                ))
            end
        end
    finally
        close(ds)
    end
    return out
end

function cs_window_map(raw_file::String)
    out = NamedTuple[]
    ds = NCDataset(raw_file, "r")
    try
        times = ds["time"][:]
        for (idx, t) in pairs(times)
            mid = DateTime(t)
            push!(out, (
                window = idx,
                raw_time = mid,
                window_start = mid - Minute(30),
                window_end = mid + Minute(30),
            ))
        end
    finally
        close(ds)
    end
    return out
end

function select_aligned_windows(date::Date, era_map, cs_map, window_spec::String, hour_spec::String)
    requested_hours = parse_hour_list(date, hour_spec)
    if isempty(requested_hours)
        wins = parse_window_list(window_spec)
        if isempty(wins)
            requested_hours = [entry.window_start for entry in era_map]
        else
            requested_hours = DateTime[]
            for w in wins
                1 <= w <= length(era_map) || error("Requested ERA5 window $w outside 1:$(length(era_map))")
                push!(requested_hours, era_map[w].window_start)
            end
        end
    end

    pairs = NamedTuple[]
    for hour_start in requested_hours
        e_idx = findfirst(entry -> entry.window_start == hour_start, era_map)
        c_idx = findfirst(entry -> entry.window_start == hour_start, cs_map)
        e_idx === nothing && error("No ERA5 source timestamp matched hour $(fmt_hour(hour_start))")
        c_idx === nothing && error("No GEOS-IT raw midpoint matched inferred hour $(fmt_hour(hour_start))")
        push!(pairs, (
            hour_start = hour_start,
            era = era_map[e_idx],
            cs = cs_map[c_idx],
        ))
    end
    return pairs
end

function row_tm5_cfl(am::Array{FT, 3}, m::Array{FT, 3}, row::Int, level::Int, r::Int) where FT
    Nx = size(m, 1)
    if r == 1
        local_max = 0.0
        for i in 1:(Nx + 1)
            il = i == 1 ? Nx : i - 1
            ir = i > Nx ? 1 : i
            am_face = am[i, row, level]
            md = am_face >= 0 ? m[il, row, level] : m[ir, row, level]
            if md > 0
                local_max = max(local_max, abs(am_face) / md)
            end
        end
        return local_max
    end

    Nx_red = Nx ÷ r
    local_max = 0.0
    for ic in 1:Nx_red
        face_index = (ic - 1) * r + 1
        am_face = am[face_index, row, level]
        donor_ic = am_face >= 0 ? (ic == 1 ? Nx_red : ic - 1) : ic
        i0 = (donor_ic - 1) * r + 1
        md = 0.0
        for ii in i0:(i0 + r - 1)
            md += m[ii, row, level]
        end
        if md > 0
            local_max = max(local_max, abs(am_face) / md)
        end
    end
    return local_max
end

function ll_surface_residual_recomputed64(am::Array{FT, 3}, bm::Array{FT, 3},
                                          B_ifc::AbstractVector{<:Real}) where FT
    Nx = size(am, 1) - 1
    Ny = size(am, 2)
    Nz = size(am, 3)
    use_b_correction = length(B_ifc) == Nz + 1

    max_surface_abs = 0.0
    max_binary_minus_recomputed_abs = 0.0
    loc = (i = 1, j = 1)

    @inbounds for j in 1:Ny, i in 1:Nx
        pit = 0.0
        if use_b_correction
            for k in 1:Nz
                pit += (Float64(am[i + 1, j, k]) - Float64(am[i, j, k])) +
                       (Float64(bm[i, j + 1, k]) - Float64(bm[i, j, k]))
            end
        end

        acc = 0.0
        for k in 1:Nz
            div_h = (Float64(am[i + 1, j, k]) - Float64(am[i, j, k])) +
                    (Float64(bm[i, j + 1, k]) - Float64(bm[i, j, k]))
            if use_b_correction
                acc = acc - div_h + (Float64(B_ifc[k + 1]) - Float64(B_ifc[k])) * pit
            else
                acc = acc - div_h
            end
        end

        sfc_abs = abs(acc)
        if sfc_abs > max_surface_abs
            max_surface_abs = sfc_abs
            loc = (i = i, j = j)
        end
    end

    return max_surface_abs, loc
end

function ll_diagnostics!(buf::LLBuffers{FT},
                         reader::MassFluxBinaryReader{FT},
                         ll_hdr,
                         win::Int,
                         reduced_spec,
                         ll_probe_idx::Dict{String, Tuple{Int, Int}},
                         topn::Int) where FT
    load_window!(buf.m, buf.am, buf.bm, buf.cm, buf.ps, reader, win)

    Nx, Ny, Nz = size(buf.m)
    B_ifc = reader.B_ifc
    cluster_sizes = reduced_spec === nothing ? ones(Int, Ny) : reduced_spec.cluster_sizes
    merge_map = haskey(ll_hdr, :merge_map) ? Int.(collect(ll_hdr.merge_map)) : Int[]
    merge_info = merge_group_info(merge_map, Nz)

    fine_row_max = zeros(Float64, Ny)
    tm5_row_max = zeros(Float64, Ny)
    band_x_max = zeros(Float64, length(LAT_BANDS))
    band_x_tm5_max = zeros(Float64, length(LAT_BANDS))
    band_y_max = zeros(Float64, length(LAT_BANDS))
    band_z_max = zeros(Float64, length(LAT_BANDS))
    surf_abs_am_sum = zeros(Float64, length(LAT_BANDS))
    surf_abs_bm_sum = zeros(Float64, length(LAT_BANDS))
    surf_abs_am_count = zeros(Int, length(LAT_BANDS))
    surf_abs_bm_count = zeros(Int, length(LAT_BANDS))
    merged_z_profile = zeros(Float64, Nz)

    x_hotspots = NamedTuple[]
    z_hotspots = NamedTuple[]

    am_sum_abs = 0.0
    am_sum_sq = 0.0
    am_count = 0
    am_max_abs = 0.0

    bm_sum_abs = 0.0
    bm_sum_sq = 0.0
    bm_count = 0
    bm_max_abs = 0.0

    cm_sum_abs = 0.0
    cm_sum_sq = 0.0
    cm_count = 0
    cm_max_abs = 0.0

    x_fine_max = 0.0
    x_fine_loc = (i = 1, j = 1, k = 1, lon = Float64(reader.lons[1]), lat = Float64(reader.lats[1]))

    y_max = 0.0
    y_loc = (i = 1, j = 2, k = 1, lon = Float64(reader.lons[1]), lat = Float64(reader.lats[1]))

    z_max = 0.0
    z_loc = (i = 1, j = 1, k = 2, lon = Float64(reader.lons[1]), lat = Float64(reader.lats[1]), donor_level = 1)

    for k in 1:Nz, j in 1:Ny
        lat = Float64(reader.lats[j])
        band = band_index(lat)

        row_tm5_level_max = row_tm5_cfl(buf.am, buf.m, j, k, cluster_sizes[j])
        tm5_row_max[j] = max(tm5_row_max[j], row_tm5_level_max)
        band_x_tm5_max[band] = max(band_x_tm5_max[band], row_tm5_level_max)

        for i in 1:(Nx + 1)
            am_face = Float64(buf.am[i, j, k])
            am_abs = abs(am_face)
            am_sum_abs += am_abs
            am_sum_sq += am_face^2
            am_count += 1
            am_max_abs = max(am_max_abs, am_abs)

            il = i == 1 ? Nx : i - 1
            ir = i > Nx ? 1 : i
            md = am_face >= 0 ? Float64(buf.m[il, j, k]) : Float64(buf.m[ir, j, k])
            cfl = md > 0 ? am_abs / md : 0.0
            fine_row_max[j] = max(fine_row_max[j], cfl)
            band_x_max[band] = max(band_x_max[band], cfl)

            if k == Nz
                surf_abs_am_sum[band] += am_abs
                surf_abs_am_count[band] += 1
            end

            if cfl > x_fine_max
                x_fine_max = cfl
                x_fine_loc = (i = i, j = j, k = k, lon = Float64(reader.lons[ir]), lat = lat)
            end

            push_topn!(x_hotspots, (
                value = cfl,
                lon = Float64(reader.lons[ir]),
                lat = lat,
                i = i,
                j = j,
                level = k,
                flux = am_face,
                donor_mass = md,
                cluster = cluster_sizes[j],
            ), topn)
        end
    end

    for k in 1:Nz, j in 1:(Ny + 1), i in 1:Nx
        bm_face = Float64(buf.bm[i, j, k])
        bm_abs = abs(bm_face)
        bm_sum_abs += bm_abs
        bm_sum_sq += bm_face^2
        bm_count += 1
        bm_max_abs = max(bm_max_abs, bm_abs)

        if 2 <= j <= Ny
            lat = Float64(reader.lats[j - 1])
            band = band_index(lat)
            md = bm_face >= 0 ? Float64(buf.m[i, j - 1, k]) : Float64(buf.m[i, j, k])
            cfl = md > 0 ? bm_abs / md : 0.0
            band_y_max[band] = max(band_y_max[band], cfl)

            if k == Nz
                surf_abs_bm_sum[band] += bm_abs
                surf_abs_bm_count[band] += 1
            end

            if cfl > y_max
                y_max = cfl
                y_loc = (i = i, j = j, k = k, lon = Float64(reader.lons[i]), lat = lat)
            end
        end
    end

    cm_top_max = 0.0
    cm_sfc_max = 0.0
    cm_top_loc = (i = 1, j = 1)
    cm_sfc_loc = (i = 1, j = 1)

    for j in 1:Ny, i in 1:Nx
        top_abs = abs(Float64(buf.cm[i, j, 1]))
        sfc_abs = abs(Float64(buf.cm[i, j, Nz + 1]))
        if top_abs > cm_top_max
            cm_top_max = top_abs
            cm_top_loc = (i = i, j = j)
        end
        if sfc_abs > cm_sfc_max
            cm_sfc_max = sfc_abs
            cm_sfc_loc = (i = i, j = j)
        end
    end

    cm_sfc64_max, cm_sfc64_loc = ll_surface_residual_recomputed64(buf.am, buf.bm, B_ifc)

    for k in 1:(Nz + 1), j in 1:Ny, i in 1:Nx
        cm_face = Float64(buf.cm[i, j, k])
        cm_abs = abs(cm_face)
        cm_sum_abs += cm_abs
        cm_sum_sq += cm_face^2
        cm_count += 1
        cm_max_abs = max(cm_max_abs, cm_abs)
    end

    for k in 2:Nz, j in 1:Ny, i in 1:Nx
        cm_face = Float64(buf.cm[i, j, k])
        donor_level = cm_face > 0 ? k - 1 : k
        md = cm_face > 0 ? Float64(buf.m[i, j, k - 1]) : Float64(buf.m[i, j, k])
        cfl = md > 0 ? abs(cm_face) / md : 0.0
        lat = Float64(reader.lats[j])
        band = band_index(lat)
        band_z_max[band] = max(band_z_max[band], cfl)
        merged_z_profile[donor_level] = max(merged_z_profile[donor_level], cfl)

        if cfl > z_max
            z_max = cfl
            z_loc = (i = i, j = j, k = k, lon = Float64(reader.lons[i]), lat = lat, donor_level = donor_level)
        end

        push_topn!(z_hotspots, (
            value = cfl,
            lon = Float64(reader.lons[i]),
            lat = lat,
            i = i,
            j = j,
            interface = k,
            donor_level = donor_level,
            flux = cm_face,
            donor_mass = md,
            native_first = merge_info.native_first[donor_level],
            native_last = merge_info.native_last[donor_level],
            group_size = merge_info.group_size[donor_level],
        ), topn)
    end

    ll_probes = Dict{String, Any}()
    for probe in DEFAULT_PROBES
        i, j = ll_probe_idx[probe.label]
        col_mass = sum(Float64.(buf.m[i, j, :]))
        col_max_z = maximum(merged_z_profile)
        ll_probes[probe.label] = Dict(
            "grid_index" => Dict("i" => i, "j" => j),
            "grid_lon" => Float64(reader.lons[i]),
            "grid_lat" => Float64(reader.lats[j]),
            "surface_pressure_pa" => Float64(buf.ps[i, j]),
            "column_mass_kg" => col_mass,
            "surface_mass_kg" => Float64(buf.m[i, j, Nz]),
            "surface_abs_am" => max(abs(Float64(buf.am[min(i, Nx), j, Nz])), abs(Float64(buf.am[min(i + 1, Nx + 1), j, Nz]))),
            "surface_abs_bm" => max(abs(Float64(buf.bm[i, max(j, 1), Nz])), abs(Float64(buf.bm[i, min(j + 1, Ny + 1), Nz]))),
            "column_max_z_cfl" => col_max_z,
        )
    end

    band_rows = Vector{Any}(undef, length(LAT_BANDS))
    surf_am_mean = row_stats_finalize(surf_abs_am_sum, surf_abs_am_count)
    surf_bm_mean = row_stats_finalize(surf_abs_bm_sum, surf_abs_bm_count)
    for (idx, band) in pairs(LAT_BANDS)
        band_rows[idx] = Dict(
            "label" => band.label,
            "lat_min" => band.min,
            "lat_max" => band.max,
            "max_x_cfl_fine" => band_x_max[idx],
            "max_x_cfl_tm5" => band_x_tm5_max[idx],
            "max_y_cfl" => band_y_max[idx],
            "max_z_cfl" => band_z_max[idx],
            "mean_surface_abs_am" => surf_am_mean[idx],
            "mean_surface_abs_bm" => surf_bm_mean[idx],
        )
    end

    return Dict(
        "x_cfl_fine_max" => x_fine_max,
        "x_cfl_fine_location" => Dict("i" => x_fine_loc.i, "j" => x_fine_loc.j, "level" => x_fine_loc.k,
                                      "lon" => x_fine_loc.lon, "lat" => x_fine_loc.lat),
        "x_cfl_tm5_reduced_max" => maximum(tm5_row_max),
        "x_cfl_tm5_row_max" => tm5_row_max,
        "x_cfl_fine_row_max" => fine_row_max,
        "y_cfl_max" => y_max,
        "y_cfl_location" => Dict("i" => y_loc.i, "j" => y_loc.j, "level" => y_loc.k,
                                 "lon" => y_loc.lon, "lat" => y_loc.lat),
        "z_cfl_max" => z_max,
        "z_cfl_location" => Dict("i" => z_loc.i, "j" => z_loc.j, "interface" => z_loc.k,
                                 "donor_level" => z_loc.donor_level,
                                 "lon" => z_loc.lon, "lat" => z_loc.lat),
        "merged_z_cfl_profile" => merged_z_profile,
        "cm_top_residual_max" => cm_top_max,
        "cm_top_residual_location" => Dict("i" => cm_top_loc.i, "j" => cm_top_loc.j,
                                           "lon" => Float64(reader.lons[cm_top_loc.i]),
                                           "lat" => Float64(reader.lats[cm_top_loc.j])),
        "cm_surface_residual_max" => cm_sfc_max,
        "cm_surface_residual_location" => Dict("i" => cm_sfc_loc.i, "j" => cm_sfc_loc.j,
                                               "lon" => Float64(reader.lons[cm_sfc_loc.i]),
                                               "lat" => Float64(reader.lats[cm_sfc_loc.j])),
        "cm_surface_residual_recomputed64_max" => cm_sfc64_max,
        "cm_surface_residual_recomputed64_location" => Dict("i" => cm_sfc64_loc.i, "j" => cm_sfc64_loc.j,
                                                            "lon" => Float64(reader.lons[cm_sfc64_loc.i]),
                                                            "lat" => Float64(reader.lats[cm_sfc64_loc.j])),
        "merge_group_size" => merge_info.group_size,
        "merge_native_first" => merge_info.native_first,
        "merge_native_last" => merge_info.native_last,
        "ps_stats" => Dict(
            "mean" => mean(Float64.(buf.ps)),
            "min" => minimum(Float64.(buf.ps)),
            "max" => maximum(Float64.(buf.ps)),
        ),
        "am_stats" => stats_dict(am_sum_abs, am_sum_sq, am_count, am_max_abs),
        "bm_stats" => stats_dict(bm_sum_abs, bm_sum_sq, bm_count, bm_max_abs),
        "cm_stats" => stats_dict(cm_sum_abs, cm_sum_sq, cm_count, cm_max_abs),
        "latitude_bands" => band_rows,
        "probes" => ll_probes,
        "x_hotspots" => [Dict(
            "value" => h.value,
            "lon" => h.lon,
            "lat" => h.lat,
            "i" => h.i,
            "j" => h.j,
            "level" => h.level,
            "flux" => h.flux,
            "donor_mass" => h.donor_mass,
            "cluster" => h.cluster,
        ) for h in x_hotspots],
        "z_hotspots" => [Dict(
            "value" => h.value,
            "lon" => h.lon,
            "lat" => h.lat,
            "i" => h.i,
            "j" => h.j,
            "interface" => h.interface,
            "donor_level" => h.donor_level,
            "flux" => h.flux,
            "donor_mass" => h.donor_mass,
            "native_first" => h.native_first,
            "native_last" => h.native_last,
            "group_size" => h.group_size,
        ) for h in z_hotspots],
    )
end

function compute_bt_vector(vc)
    Nz = length(vc.A) - 1
    ΔB = zeros(Float64, Nz)
    for k in 1:Nz
        ΔB[k] = vc.B[k + 1] - vc.B[k]
    end
    ΔB_total = vc.B[end] - vc.B[1]
    if abs(ΔB_total) <= eps(Float64)
        return zeros(Float64, Nz)
    end
    return ΔB ./ ΔB_total
end

function compute_cm_panel_cpu!(cm::Array{FT, 3}, am::Array{FT, 3}, bm::Array{FT, 3},
                               bt::AbstractVector{<:Real}, Nc::Int, Nz::Int) where FT
    fill!(cm, zero(FT))
    @inbounds for j in 1:Nc, i in 1:Nc
        pit = zero(Float64)
        for k in 1:Nz
            pit += am[i, j, k] - am[i + 1, j, k] + bm[i, j, k] - bm[i, j + 1, k]
        end
        acc = zero(Float64)
        cm[i, j, 1] = zero(FT)
        for k in 1:Nz
            conv_k = am[i, j, k] - am[i + 1, j, k] + bm[i, j, k] - bm[i, j + 1, k]
            acc += conv_k - bt[k] * pit
            cm[i, j, k + 1] = FT(acc)
        end
    end
    return nothing
end

function cs_diagnostics!(buf::CSBuffers{FT},
                         reader::CSBinaryReader{FT},
                         cs_hdr,
                         cs_grid::CubedSphereGrid{FT},
                         cs_lons::Array{Float64, 3},
                         cs_lats::Array{Float64, 3},
                         cs_areas::Array{Float64, 3},
                         cs_probe_idx::Dict{String, NamedTuple},
                         bt::Vector{Float64},
                         win::Int,
                         dry_correction::Bool,
                         topn::Int) where FT
    load_cs_window!(buf.delp, buf.am, buf.bm, reader, win)
    has_qv_ps = load_cs_qv_ps_window!(buf.qv_start, buf.qv_end, buf.ps_start, buf.ps_end, reader, win)
    has_qv_ps || error("CS v4 diagnostics require embedded QV/PS in $(reader.io.name)")

    Nc, Nz, Hp = reader.Nc, reader.Nz, reader.Hp
    g = GRAV

    for p in 1:6
        copy_core_to_halo!(buf.qv_halo[p], buf.qv_start[p], Hp)
    end
    fill_panel_halos!(buf.qv_halo, cs_grid)

    for p in 1:6
        fill!(buf.m_halo[p], zero(FT))
        @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
            delp_ijk = buf.delp[p][Hp + i, Hp + j, k]
            qfac = dry_correction ? (1 - buf.qv_start[p][i, j, k]) : one(FT)
            buf.m_halo[p][Hp + i, Hp + j, k] = delp_ijk * qfac * FT(cs_areas[i, j, p]) / g
        end
    end
    fill_panel_halos!(buf.m_halo, cs_grid)

    for p in 1:6
        compute_cm_panel_cpu!(buf.cm[p], buf.am[p], buf.bm[p], bt, Nc, Nz)
    end

    x_hotspots = NamedTuple[]
    z_hotspots = NamedTuple[]

    band_x_max = zeros(Float64, length(LAT_BANDS))
    band_y_max = zeros(Float64, length(LAT_BANDS))
    band_z_max = zeros(Float64, length(LAT_BANDS))
    surf_abs_am_sum = zeros(Float64, length(LAT_BANDS))
    surf_abs_bm_sum = zeros(Float64, length(LAT_BANDS))
    surf_abs_am_count = zeros(Int, length(LAT_BANDS))
    surf_abs_bm_count = zeros(Int, length(LAT_BANDS))

    am_sum_abs = 0.0
    am_sum_sq = 0.0
    am_count = 0
    am_max_abs = 0.0

    bm_sum_abs = 0.0
    bm_sum_sq = 0.0
    bm_count = 0
    bm_max_abs = 0.0

    cm_sum_abs = 0.0
    cm_sum_sq = 0.0
    cm_count = 0
    cm_max_abs = 0.0

    x_max = 0.0
    y_max = 0.0
    z_max = 0.0
    x_loc = (panel = 1, i = 1, j = 1, k = 1, lon = cs_lons[1, 1, 1], lat = cs_lats[1, 1, 1])
    y_loc = (panel = 1, i = 1, j = 1, k = 1, lon = cs_lons[1, 1, 1], lat = cs_lats[1, 1, 1])
    z_loc = (panel = 1, i = 1, j = 1, k = 2, donor_level = 1, lon = cs_lons[1, 1, 1], lat = cs_lats[1, 1, 1])

    delp_top_vals = Float64[]
    delp_bot_vals = Float64[]
    ps_col_err_max = 0.0

    for p in 1:6
        delp_core = @view buf.delp[p][Hp + 1:Hp + Nc, Hp + 1:Hp + Nc, :]
        push!(delp_top_vals, mean(Float64.(delp_core[:, :, 1])))
        push!(delp_bot_vals, mean(Float64.(delp_core[:, :, end])))
        for j in 1:Nc, i in 1:Nc
            ps_col = 0.0
            for k in 1:Nz
                ps_col += delp_core[i, j, k]
            end
            ps_col_err_max = max(ps_col_err_max, abs(ps_col - Float64(buf.ps_start[p][i, j])))
        end
    end

    for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:(Nc + 1)
        am_face = Float64(buf.am[p][i, j, k])
        am_abs = abs(am_face)
        am_sum_abs += am_abs
        am_sum_sq += am_face^2
        am_count += 1
        am_max_abs = max(am_max_abs, am_abs)

        ii = Hp + i
        jj = Hp + j
        md = am_face >= 0 ? Float64(buf.m_halo[p][ii - 1, jj, k]) : Float64(buf.m_halo[p][ii, jj, k])
        cfl = md > 0 ? am_abs / md : 0.0

        ci = clamp(i, 1, Nc)
        lat = cs_lats[ci, j, p]
        lon = cs_lons[ci, j, p]
        band = band_index(lat)
        band_x_max[band] = max(band_x_max[band], cfl)
        if k == Nz
            surf_abs_am_sum[band] += am_abs
            surf_abs_am_count[band] += 1
        end

        if cfl > x_max
            x_max = cfl
            x_loc = (panel = p, i = i, j = j, k = k, lon = lon, lat = lat)
        end
        push_topn!(x_hotspots, (
            value = cfl,
            panel = p,
            i = i,
            j = j,
            level = k,
            lon = lon,
            lat = lat,
            flux = am_face,
            donor_mass = md,
        ), topn)
    end

    for p in 1:6, k in 1:Nz, j in 1:(Nc + 1), i in 1:Nc
        bm_face = Float64(buf.bm[p][i, j, k])
        bm_abs = abs(bm_face)
        bm_sum_abs += bm_abs
        bm_sum_sq += bm_face^2
        bm_count += 1
        bm_max_abs = max(bm_max_abs, bm_abs)

        ii = Hp + i
        jj = Hp + j
        md = bm_face >= 0 ? Float64(buf.m_halo[p][ii, jj - 1, k]) : Float64(buf.m_halo[p][ii, jj, k])
        cfl = md > 0 ? bm_abs / md : 0.0

        cj = clamp(j, 1, Nc)
        lat = cs_lats[i, cj, p]
        lon = cs_lons[i, cj, p]
        band = band_index(lat)
        band_y_max[band] = max(band_y_max[band], cfl)
        if k == Nz
            surf_abs_bm_sum[band] += bm_abs
            surf_abs_bm_count[band] += 1
        end

        if cfl > y_max
            y_max = cfl
            y_loc = (panel = p, i = i, j = j, k = k, lon = lon, lat = lat)
        end
    end

    for p in 1:6, k in 1:(Nz + 1), j in 1:Nc, i in 1:Nc
        cm_face = Float64(buf.cm[p][i, j, k])
        cm_abs = abs(cm_face)
        cm_sum_abs += cm_abs
        cm_sum_sq += cm_face^2
        cm_count += 1
        cm_max_abs = max(cm_max_abs, cm_abs)
    end

    for p in 1:6, k in 2:Nz, j in 1:Nc, i in 1:Nc
        cm_face = Float64(buf.cm[p][i, j, k])
        donor_level = cm_face > 0 ? k - 1 : k
        md = cm_face > 0 ? Float64(buf.m_halo[p][Hp + i, Hp + j, k - 1]) :
                           Float64(buf.m_halo[p][Hp + i, Hp + j, k])
        cfl = md > 0 ? abs(cm_face) / md : 0.0
        lon = cs_lons[i, j, p]
        lat = cs_lats[i, j, p]
        band = band_index(lat)
        band_z_max[band] = max(band_z_max[band], cfl)

        if cfl > z_max
            z_max = cfl
            z_loc = (panel = p, i = i, j = j, k = k, donor_level = donor_level, lon = lon, lat = lat)
        end
        push_topn!(z_hotspots, (
            value = cfl,
            panel = p,
            i = i,
            j = j,
            interface = k,
            donor_level = donor_level,
            lon = lon,
            lat = lat,
            flux = cm_face,
            donor_mass = md,
        ), topn)
    end

    dy_approx = 2π * EARTH_RADIUS / (4 * Nc)
    delp_bot_mean = mean(delp_bot_vals)
    am_sfc_rms = sqrt(mean(vcat([vec(Float64.(buf.am[p][:, :, end])) for p in 1:6]...) .^ 2))
    bm_sfc_rms = sqrt(mean(vcat([vec(Float64.(buf.bm[p][:, :, end])) for p in 1:6]...) .^ 2))
    u_sfc_est = am_sfc_rms * GRAV / (delp_bot_mean * dy_approx)
    v_sfc_est = bm_sfc_rms * GRAV / (delp_bot_mean * dy_approx)

    ps_start_all = vcat([vec(Float64.(buf.ps_start[p])) for p in 1:6]...)
    ps_end_all = vcat([vec(Float64.(buf.ps_end[p])) for p in 1:6]...)
    qv_start_all = vcat([vec(Float64.(buf.qv_start[p])) for p in 1:6]...)
    qv_end_all = vcat([vec(Float64.(buf.qv_end[p])) for p in 1:6]...)

    cs_probes = Dict{String, Any}()
    for probe in DEFAULT_PROBES
        idx = cs_probe_idx[probe.label]
        p = idx.panel
        i = idx.i
        j = idx.j
        col_mass = 0.0
        col_max_z = 0.0
        for k in 1:Nz
            col_mass += Float64(buf.m_halo[p][Hp + i, Hp + j, k])
        end
        for k in 2:Nz
            cm_face = Float64(buf.cm[p][i, j, k])
            md = cm_face > 0 ? Float64(buf.m_halo[p][Hp + i, Hp + j, k - 1]) :
                               Float64(buf.m_halo[p][Hp + i, Hp + j, k])
            cfl = md > 0 ? abs(cm_face) / md : 0.0
            col_max_z = max(col_max_z, cfl)
        end
        cs_probes[probe.label] = Dict(
            "panel" => p,
            "i" => i,
            "j" => j,
            "grid_lon" => cs_lons[i, j, p],
            "grid_lat" => cs_lats[i, j, p],
            "distance_m" => idx.distance_m,
            "surface_pressure_start_pa" => Float64(buf.ps_start[p][i, j]),
            "surface_pressure_end_pa" => Float64(buf.ps_end[p][i, j]),
            "column_mass_kg" => col_mass,
            "surface_mass_kg" => Float64(buf.m_halo[p][Hp + i, Hp + j, Nz]),
            "column_max_z_cfl" => col_max_z,
        )
    end

    band_rows = Vector{Any}(undef, length(LAT_BANDS))
    surf_am_mean = row_stats_finalize(surf_abs_am_sum, surf_abs_am_count)
    surf_bm_mean = row_stats_finalize(surf_abs_bm_sum, surf_abs_bm_count)
    for (idx, band) in pairs(LAT_BANDS)
        band_rows[idx] = Dict(
            "label" => band.label,
            "lat_min" => band.min,
            "lat_max" => band.max,
            "max_x_cfl" => band_x_max[idx],
            "max_y_cfl" => band_y_max[idx],
            "max_z_cfl" => band_z_max[idx],
            "mean_surface_abs_am" => surf_am_mean[idx],
            "mean_surface_abs_bm" => surf_bm_mean[idx],
        )
    end

    return Dict(
        "x_cfl_max" => x_max,
        "x_cfl_location" => Dict("panel" => x_loc.panel, "i" => x_loc.i, "j" => x_loc.j,
                                 "level" => x_loc.k, "lon" => x_loc.lon, "lat" => x_loc.lat),
        "y_cfl_max" => y_max,
        "y_cfl_location" => Dict("panel" => y_loc.panel, "i" => y_loc.i, "j" => y_loc.j,
                                 "level" => y_loc.k, "lon" => y_loc.lon, "lat" => y_loc.lat),
        "z_cfl_max" => z_max,
        "z_cfl_location" => Dict("panel" => z_loc.panel, "i" => z_loc.i, "j" => z_loc.j,
                                 "interface" => z_loc.k, "donor_level" => z_loc.donor_level,
                                 "lon" => z_loc.lon, "lat" => z_loc.lat),
        "delp_top_mean_pa" => mean(delp_top_vals),
        "delp_bottom_mean_pa" => delp_bot_mean,
        "column_ps_minus_ps_start_max_abs" => ps_col_err_max,
        "ps_start_stats" => Dict(
            "mean" => mean(ps_start_all),
            "min" => minimum(ps_start_all),
            "max" => maximum(ps_start_all),
        ),
        "ps_window_delta_stats" => Dict(
            "mean" => mean(ps_end_all .- ps_start_all),
            "max_abs" => maximum(abs.(ps_end_all .- ps_start_all)),
        ),
        "qv_window_delta_stats" => Dict(
            "mean" => mean(qv_end_all .- qv_start_all),
            "max_abs" => maximum(abs.(qv_end_all .- qv_start_all)),
        ),
        "surface_wind_estimate" => Dict(
            "u_rms_mps" => u_sfc_est,
            "v_rms_mps" => v_sfc_est,
            "speed_rms_mps" => hypot(u_sfc_est, v_sfc_est),
        ),
        "am_stats" => stats_dict(am_sum_abs, am_sum_sq, am_count, am_max_abs),
        "bm_stats" => stats_dict(bm_sum_abs, bm_sum_sq, bm_count, bm_max_abs),
        "cm_stats" => stats_dict(cm_sum_abs, cm_sum_sq, cm_count, cm_max_abs),
        "latitude_bands" => band_rows,
        "probes" => cs_probes,
        "x_hotspots" => [Dict(
            "value" => h.value,
            "panel" => h.panel,
            "i" => h.i,
            "j" => h.j,
            "level" => h.level,
            "lon" => h.lon,
            "lat" => h.lat,
            "flux" => h.flux,
            "donor_mass" => h.donor_mass,
        ) for h in x_hotspots],
        "z_hotspots" => [Dict(
            "value" => h.value,
            "panel" => h.panel,
            "i" => h.i,
            "j" => h.j,
            "interface" => h.interface,
            "donor_level" => h.donor_level,
            "lon" => h.lon,
            "lat" => h.lat,
            "flux" => h.flux,
            "donor_mass" => h.donor_mass,
        ) for h in z_hotspots],
        "dry_correction_applied_to_mass" => dry_correction,
        "binary_has_qv_ps" => has_qv_ps,
    )
end

function classify_suspects(hour_results, ll_geometry, ll_header, cs_header, era5_run_cfg, cs_run_cfg)
    ll_x_peak = maximum(hr["ll"]["x_cfl_fine_max"] for hr in hour_results)
    ll_x_tm5_peak = maximum(hr["ll"]["x_cfl_tm5_reduced_max"] for hr in hour_results)
    ll_z_peak = maximum(hr["ll"]["z_cfl_max"] for hr in hour_results)
    ll_sfc_resid_peak = maximum(hr["ll"]["cm_surface_residual_max"] for hr in hour_results)
    ll_sfc_resid64_peak = maximum(hr["ll"]["cm_surface_residual_recomputed64_max"] for hr in hour_results)
    cs_x_peak = maximum(hr["cs"]["x_cfl_max"] for hr in hour_results)
    cs_z_peak = maximum(hr["cs"]["z_cfl_max"] for hr in hour_results)

    ll_x_polar_fraction = begin
        lats = Float64[]
        for hr in hour_results
            for h in hr["ll"]["x_hotspots"]
                push!(lats, abs(Float64(h["lat"])))
            end
        end
        isempty(lats) ? NaN : count(>=(89.0), lats) / length(lats)
    end

    ll_z_merged_fraction = begin
        groups = Int[]
        for hr in hour_results
            for h in hr["ll"]["z_hotspots"]
                push!(groups, Int(h["group_size"]))
            end
        end
        isempty(groups) ? NaN : count(>(1), groups) / length(groups)
    end

    suspects = Any[]

    pole_status = if ll_x_peak > 2.0 && ll_x_polar_fraction >= 0.75 && cs_x_peak < 2.0
        "confirmed"
    elseif ll_x_peak > 2.0
        "needs_rerun"
    else
        "rejected"
    end
    push!(suspects, Dict(
        "name" => "pole_row_handling",
        "status" => pole_status,
        "classification" => pole_status == "rejected" ? "shared" : "LL-only",
        "evidence" => @sprintf("LL fine x-CFL peak=%.3f, TM5-reduced peak=%.3f, polar-hotspot fraction=%.2f, CS x-CFL peak=%.3f",
                               ll_x_peak, ll_x_tm5_peak, ll_x_polar_fraction, cs_x_peak),
        "tm5_reference" => TM5_REFERENCE["pole_row_handling"],
    ))

    cm_status = if ll_sfc_resid64_peak > 1e2
        "confirmed"
    elseif ll_sfc_resid64_peak > 1e-2
        "needs_rerun"
    else
        "rejected"
    end
    push!(suspects, Dict(
        "name" => "cm_reconstruction",
        "status" => cm_status,
        "classification" => cm_status == "rejected" ? "shared" : "LL-only",
        "evidence" => @sprintf("LL stored surface cm residual peak=%.3e kg; Float64 recomputed closure peak=%.3e kg",
                               ll_sfc_resid_peak, ll_sfc_resid64_peak),
        "tm5_reference" => TM5_REFERENCE["cm_reconstruction"],
    ))

    merge_status = if ll_z_peak > 1.0 && ll_z_merged_fraction >= 0.5 && cs_z_peak < ll_z_peak
        "confirmed"
    elseif ll_z_peak > 1.0
        "needs_rerun"
    else
        "rejected"
    end
    push!(suspects, Dict(
        "name" => "vertical_merging",
        "status" => merge_status,
        "classification" => merge_status == "rejected" ? "shared" : "LL-only",
        "evidence" => @sprintf("LL z-CFL peak=%.3f, CS z-CFL peak=%.3f, merged-hotspot fraction=%.2f",
                               ll_z_peak, cs_z_peak, ll_z_merged_fraction),
        "tm5_reference" => TM5_REFERENCE["vertical_merging"],
    ))

    lat_status = if Bool(ll_geometry["binary_has_exact_pole_centers"]) &&
                    Float64(ll_geometry["max_runtime_lat_center_mismatch_deg"]) > 0.1
        "confirmed"
    else
        "rejected"
    end
    push!(suspects, Dict(
        "name" => "latitude_convention",
        "status" => lat_status,
        "classification" => lat_status == "rejected" ? "shared" : "LL-only",
        "evidence" => @sprintf("LL header Ny=%d with Δlat≈%.3f° and exact pole rows; runtime uniform-center mismatch max=%.3f°; TM5-style Ny from spacing=%d",
                               Int(ll_geometry["Ny"]),
                               Float64(ll_geometry["lat_spacing_deg"]),
                               Float64(ll_geometry["max_runtime_lat_center_mismatch_deg"]),
                               Int(ll_geometry["tm5_style_ny_from_spacing"])),
        "tm5_reference" => TM5_REFERENCE["latitude_convention"],
    ))

    ll_dt = Float64(era5_run_cfg["met_data"]["dt"])
    ll_bin_dt = Float64(ll_header.dt_seconds)
    cs_mass_flux_dt = Float64(cs_run_cfg["met_data"]["mass_flux_dt"])
    cs_bin_dt = Float64(cs_header.dt_met_seconds)
    cs_vertical_ok = String(get(cs_header, :vertical_order, "unknown")) == "top_to_bottom"
    dt_status = if abs(ll_dt - 2ll_bin_dt) < 1e-6 && abs(cs_mass_flux_dt - cs_bin_dt) < 1e-6 && cs_vertical_ok
        "rejected"
    else
        "confirmed"
    end
    push!(suspects, Dict(
        "name" => "mass_flux_dt_or_ordering",
        "status" => dt_status,
        "classification" => dt_status == "rejected" ? "shared" : "CS-only",
        "evidence" => @sprintf("ERA5 run dt=%.0fs vs binary dt=%.0fs; CS mass_flux_dt=%.0fs vs binary dt=%.0fs; CS vertical_order=%s",
                               ll_dt, ll_bin_dt, cs_mass_flux_dt, cs_bin_dt,
                               String(get(cs_header, :vertical_order, "unknown"))),
        "tm5_reference" => TM5_REFERENCE["mass_flux_dt_or_ordering"],
    ))

    return suspects
end

function print_time_alignment(io::IO, aligned)
    println(io, "\nTime alignment")
    println(io, "--------------")
    println(io, rpad("Hour start", 18), rpad("ERA5 raw", 22), rpad("GEOS-IT raw midpoint", 24), "GEOS-IT window")
    for pair in aligned
        println(io,
                rpad(fmt_hour(pair.hour_start), 18),
                rpad(fmt_dt(pair.era.raw_time), 22),
                rpad(fmt_dt(pair.cs.raw_time), 24),
                fmt_hour(pair.cs.window_start), " -> ", fmt_hour(pair.cs.window_end))
    end
end

function print_hour_summary(io::IO, hour_results)
    println(io, "\nPer-hour summary")
    println(io, "----------------")
    println(io, rpad("Hour", 18),
            rpad("LL x/tm5/y/z", 28),
            rpad("LL cm_sfc(bin/r64)", 24),
            rpad("CS x/y/z", 24),
            "CS |Δps|max")
    for hr in hour_results
        ll = hr["ll"]
        cs = hr["cs"]
        line = @sprintf("%-18s %-28s %-24s %-24s %.3f",
                        hr["hour_start"],
                        @sprintf("%.2f/%.2f/%.2f/%.2f", ll["x_cfl_fine_max"], ll["x_cfl_tm5_reduced_max"],
                                 ll["y_cfl_max"], ll["z_cfl_max"]),
                        @sprintf("%.3e/%.3e", ll["cm_surface_residual_max"],
                                 ll["cm_surface_residual_recomputed64_max"]),
                        @sprintf("%.2f/%.2f/%.2f", cs["x_cfl_max"], cs["y_cfl_max"], cs["z_cfl_max"]),
                        cs["ps_window_delta_stats"]["max_abs"])
        println(io, line)
    end
end

function print_ll_geometry(io::IO, ll_geometry)
    println(io, "\nLL geometry")
    println(io, "-----------")
    println(io, @sprintf("Ny=%d, Δlat≈%.3f°, exact_pole_centers=%s, runtime_center_mismatch_max=%.3f°",
                         Int(ll_geometry["Ny"]),
                         Float64(ll_geometry["lat_spacing_deg"]),
                         string(Bool(ll_geometry["binary_has_exact_pole_centers"])),
                         Float64(ll_geometry["max_runtime_lat_center_mismatch_deg"])))
end

function print_suspects(io::IO, suspects)
    println(io, "\nSuspects")
    println(io, "--------")
    for s in suspects
        println(io, @sprintf("[%s | %s] %s", s["status"], s["classification"], s["name"]))
        println(io, "  ", s["evidence"])
        println(io, "  TM5: ", s["tm5_reference"])
    end
end

function main()
    cfg = parse_args()
    datestr = Dates.format(cfg.date, "yyyymmdd")

    era5_bin = resolve_era5_bin(cfg.era5_bin, cfg.date, cfg.era5_run_config)
    cs_bin = resolve_cs_bin(cfg.cs_bin, cfg.date, cfg.cs_run_config)
    era5_source = discover_era5_source_file(cfg.era5_source_file, cfg.date, cfg.era5_preproc_config)
    cs_raw_dir = resolve_cs_raw_dir(cfg.cs_raw_dir, cfg.cs_run_config)
    cs_gridspec = resolve_cs_gridspec(cfg.cs_gridspec, cfg.cs_run_config)

    isfile(era5_bin) || error("ERA5 binary not found: $era5_bin")
    isfile(cs_bin) || error("CS binary not found: $cs_bin")
    isfile(era5_source) || error("ERA5 source NetCDF not found: $era5_source")
    isfile(cs_gridspec) || error("CS gridspec not found: $cs_gridspec")

    ll_hdr = read_json_header(era5_bin, ERA5_HEADER_BYTES)
    cs_hdr = read_json_header(cs_bin, CS_HEADER_BYTES)

    ll_reader = MassFluxBinaryReader(era5_bin, FT)
    cs_reader = CSBinaryReader(cs_bin, FT)
    ll_buf = make_ll_buffers(ll_reader)
    cs_buf = make_cs_buffers(cs_reader)

    era_map = era5_window_map(era5_source, cfg.date)
    cs_raw_file = discover_cs_raw_ctm_a1(cs_raw_dir, cfg.date, cs_reader.Nc)
    cs_map = cs_window_map(cs_raw_file)
    aligned = select_aligned_windows(cfg.date, era_map, cs_map, cfg.windows, cfg.hours)

    ll_probe_idx = find_ll_probes(ll_reader.lons, ll_reader.lats)
    cs_grid_info = read_cs_gridspec(cs_gridspec)
    cs_probe_idx = find_cs_probes(cs_grid_info.lons, cs_grid_info.lats)

    cs_areas = cs_grid_info.areas
    cs_areas === nothing && error("CS gridspec is missing exact 'areas' field: $cs_gridspec")

    geos_cfg = default_met_config("geosfp")
    vc_geos = build_vertical_coordinate(geos_cfg; FT = FT)
    bt = compute_bt_vector(vc_geos)
    cs_grid = CubedSphereGrid(CPU(); FT = FT, Nc = cs_reader.Nc, vertical = vc_geos, halo = (cs_reader.Hp, 1))

    reduced_spec = compute_reduced_grid_tm5(ll_reader.Nx, Float64.(ll_reader.lats))
    ll_geometry = ll_geometry_summary(ll_reader.lons, ll_reader.lats)

    era5_run_cfg = TOML.parsefile(cfg.era5_run_config)
    cs_run_cfg = TOML.parsefile(cfg.cs_run_config)
    dry_correction = get(get(cs_run_cfg, "advection", Dict{String, Any}()), "dry_correction", true)

    hour_results = Any[]
    for pair in aligned
        ll = ll_diagnostics!(ll_buf, ll_reader, ll_hdr, pair.era.window, reduced_spec, ll_probe_idx, cfg.topn)
        cs = cs_diagnostics!(cs_buf, cs_reader, cs_hdr, cs_grid,
                             cs_grid_info.lons, cs_grid_info.lats, cs_areas,
                             cs_probe_idx, bt, pair.cs.window, dry_correction, cfg.topn)

        push!(hour_results, Dict(
            "hour_start" => fmt_hour(pair.hour_start),
            "era5_window" => pair.era.window,
            "era5_source_time" => fmt_dt(pair.era.raw_time),
            "cs_window" => pair.cs.window,
            "cs_raw_midpoint_time" => fmt_dt(pair.cs.raw_time),
            "ll" => ll,
            "cs" => cs,
        ))
    end

    suspects = classify_suspects(hour_results, ll_geometry, ll_hdr, cs_hdr, era5_run_cfg, cs_run_cfg)

    result = Dict(
        "date" => string(cfg.date),
        "inputs" => Dict(
            "era5_bin" => era5_bin,
            "cs_bin" => cs_bin,
            "era5_source" => era5_source,
            "cs_raw_ctm_a1" => cs_raw_file,
            "cs_gridspec" => cs_gridspec,
            "dry_correction" => dry_correction,
        ),
        "headers" => Dict(
            "era5" => Dict(
                "version" => Int(get(ll_hdr, :version, 1)),
                "Nx" => Int(ll_hdr.Nx),
                "Ny" => Int(ll_hdr.Ny),
                "Nz" => Int(ll_hdr.Nz),
                "Nz_native" => Int(get(ll_hdr, :Nz_native, ll_hdr.Nz)),
                "Nt" => Int(ll_hdr.Nt),
                "dt_seconds" => Float64(ll_hdr.dt_seconds),
                "half_dt_seconds" => Float64(ll_hdr.half_dt_seconds),
                "steps_per_met_window" => Int(ll_hdr.steps_per_met_window),
            ),
            "cs" => Dict(
                "version" => Int(get(cs_hdr, :version, 1)),
                "Nc" => Int(cs_hdr.Nc),
                "Nz" => Int(cs_hdr.Nz),
                "Hp" => Int(cs_hdr.Hp),
                "Nt" => Int(cs_hdr.Nt),
                "dt_met_seconds" => Float64(cs_hdr.dt_met_seconds),
                "vertical_order" => String(get(cs_hdr, :vertical_order, "unknown")),
                "include_qv" => Bool(get(cs_hdr, :include_qv, false)),
                "include_ps" => Bool(get(cs_hdr, :include_ps, false)),
                "include_courant" => Bool(get(cs_hdr, :include_courant, false)),
                "include_area_flux" => Bool(get(cs_hdr, :include_area_flux, false)),
            ),
        ),
        "time_alignment" => [Dict(
            "hour_start" => fmt_hour(pair.hour_start),
            "era5_window" => pair.era.window,
            "era5_source_time" => fmt_dt(pair.era.raw_time),
            "cs_window" => pair.cs.window,
            "cs_raw_midpoint_time" => fmt_dt(pair.cs.raw_time),
            "cs_window_start" => fmt_dt(pair.cs.window_start),
            "cs_window_end" => fmt_dt(pair.cs.window_end),
        ) for pair in aligned],
        "ll_geometry" => ll_geometry,
        "hours" => hour_results,
        "suspects" => suspects,
        "tm5_reference" => TM5_REFERENCE,
    )

    print_time_alignment(stdout, aligned)
    print_ll_geometry(stdout, ll_geometry)
    print_hour_summary(stdout, hour_results)
    print_suspects(stdout, suspects)

    if !isempty(cfg.json_out)
        outpath = expanduser(cfg.json_out)
        mkpath(dirname(outpath))
        open(outpath, "w") do io
            write(io, JSON3.write(result))
        end
        println("\nJSON written: ", outpath)
    end

end

main()
