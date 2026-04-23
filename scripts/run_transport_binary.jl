#!/usr/bin/env julia

using Logging
using Printf
using TOML
using NCDatasets
using Adapt

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Regridding: build_regridder, apply_regridder!

# ---------------------------------------------------------------------------
# Structured / reduced-Gaussian transport-binary runner.
#
# Physics composition uses the shared runtime-recipe schema:
#
#   [advection]  scheme = upwind | slopes | ppm
#   [diffusion]  kind   = none | constant
#   [convection] kind   = none | tm5 | cmfmc
#
# Legacy `[run].scheme` remains accepted as a fallback for advection.
# Convection choices are validated against the binary payload before the run.
# ---------------------------------------------------------------------------

# Wrap longitude to [0, 360) for periodic bilinear interpolation.
# NOTE: the source arrays (Catrine, GridFED) may be in [-180, 180)
# convention; this wraps both to a common [0, 360) before looking up.
@inline wrapped_longitude_distance(lon, lon0) = abs(mod(lon - lon0 + 180, 360) - 180)
@inline wrapped_longitude_360(lon) = mod(lon, 360)

@inline _init_kind(cfg) = Symbol(lowercase(String(get(cfg, "kind", "uniform"))))
@inline _is_file_init_kind(kind::Symbol) = kind in (:file, :netcdf, :file_field, :catrine_co2)
@inline _surface_flux_kind(cfg) = Symbol(lowercase(String(get(cfg, "kind", "none"))))
@inline _use_gpu(cfg) = Bool(get(get(cfg, "architecture", Dict{String, Any}()), "use_gpu", false))

const SECONDS_PER_MONTH = 365.25 * 86400 / 12

"""
    FileInitialConditionSource{FT}

Container for a file-based initial condition (e.g. Catrine startCO2).

# Fields
- `raw`            — 3D mixing ratio field `(nlon_src, nlat_src, nlevel_src)`.
                     Level ordering follows the source file (Catrine: k=1 is
                     SURFACE, k=end is TOA — verify via `ap[1]+bp[1]*ps ≈ ps`).
- `lon`, `lat`     — source coordinate vectors [degrees]. May be in [-180,180)
                     or [0,360); the bilinear sampler wraps to [0,360)
                     internally via `wrapped_longitude_360`.
- `ap`, `bp`       — hybrid half-level coefficients `(nlevel_src + 1)`:
                     `p_half[k] = ap[k] + bp[k] × ps_src`. Units: `ap` [Pa],
                     `bp` [dimensionless].
- `psurf`          — surface pressure `(nlon_src, nlat_src)` [Pa].
- `needs_vinterp`  — `true` if source levels ≠ target levels (triggers
                     log-pressure vertical interpolation in
                     `_interpolate_log_pressure_profile!`).
"""
struct FileInitialConditionSource{FT}
    raw           :: Array{FT, 3}
    lon           :: Vector{Float64}
    lat           :: Vector{Float64}
    ap            :: Vector{Float64}
    bp            :: Vector{Float64}
    psurf         :: Matrix{Float64}
    needs_vinterp :: Bool
end

struct FileSurfaceFluxField{FT}
    raw           :: Array{FT, 2}
    lon           :: Vector{Float64}
    lat           :: Vector{Float64}
    native_total_mass_rate :: Float64
end

struct TransportTracerSpec
    name             :: Symbol
    init_cfg         :: Dict{String, Any}
    surface_flux_cfg :: Dict{String, Any}
end

function _ic_find_coord(ds, candidates::Vector{String})
    for name in candidates
        haskey(ds, name) && return name
    end
    return nothing
end

"""
    _bilinear_bracket(val, arr) -> (lo, w)

Find the 1-based bracket index `lo` and fractional weight `w ∈ [0, 1]`
such that `val ≈ arr[lo] + w × (arr[lo+1] − arr[lo])`. Uses binary
search on a **strictly increasing, non-periodic** array (e.g. latitude).

Clamps: if `val ≤ arr[1]`, returns `(1, 0.0)` (extrapolate to first value).
If `val ≥ arr[end]`, returns `(N, 0.0)` (extrapolate to last value).
"""
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

"""
    _periodic_bilinear_bracket(val, arr) -> (ilo, w)

Periodic (longitude) version of `_bilinear_bracket`. `arr` must be
strictly increasing with uniform spacing `Δ = arr[2] − arr[1]` over a
360° domain (e.g. `[-179.5, -178.5, ..., 179.5]` or `[0.5, 1.5, ..., 359.5]`).

Returns 1-based index `ilo` and fractional weight `w` for the bracket
surrounding `mod(val − arr[1], 360)` in the periodic domain. Wraps both
`val` and the index modulo `length(arr)`.

The caller should also compute `ihi = (ilo == N) ? 1 : ilo + 1` for the
upper bracket (periodic wrap at the last cell).
"""
function _periodic_bilinear_bracket(val::Float64, arr::Vector{Float64})
    N = length(arr)
    N == 1 && return (1, 0.0)
    Δ = arr[2] - arr[1]   # assumed uniform spacing [degrees]
    Δ > 0 || throw(ArgumentError("longitude coordinate must be strictly increasing"))
    u = mod(val - arr[1], 360.0) / Δ   # fractional index into the periodic domain
    ilo0 = floor(Int, u)
    return mod1(ilo0 + 1, N), u - ilo0  # (1-indexed, fractional weight)
end

"""
    _horizontal_interp_weights(lon, lat, lon_src, lat_src)

Compute bilinear interpolation indices and weights for a target point
`(lon, lat)` on source arrays with coordinates `lon_src` [degrees, periodic]
and `lat_src` [degrees, non-periodic clamped].

Returns `(ilo, ihi, jlo, jhi, w00, w10, w01, w11)` where `w00` through
`w11` are the four bilinear weights for cells `(ilo, jlo)`, `(ihi, jlo)`,
`(ilo, jhi)`, `(ihi, jhi)` respectively.

**Longitude convention**: `lon` is first wrapped to [0, 360) via
`wrapped_longitude_360` before the periodic bracket. This handles both
`[-180, 180)` and `[0, 360)` input conventions transparently.
"""
function _horizontal_interp_weights(lon::Real, lat::Real, lon_src::Vector{Float64}, lat_src::Vector{Float64})
    lon_m = wrapped_longitude_360(Float64(lon))  # → [0, 360)
    ilo, wx = _periodic_bilinear_bracket(lon_m, lon_src)
    ihi = ilo == length(lon_src) ? 1 : ilo + 1   # periodic wrap at last cell
    jlo, wy = _bilinear_bracket(Float64(lat), lat_src)
    jhi = min(jlo + 1, length(lat_src))           # clamp at poles (no wrap)
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

function _sample_bilinear_scalar(raw::AbstractMatrix{T},
                                 lon_src::Vector{Float64},
                                 lat_src::Vector{Float64},
                                 lon::Real,
                                 lat::Real) where T
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

function _resolve_surface_flux_file(cfg, kind::Symbol)
    default_file, default_variable = if kind === :gridfed_fossil_co2
        ("~/data/AtmosTransport/catrine/Emissions/gridfed/GCP-GridFEDv2024.0_2021.short.nc", "TOTAL")
    else
        ("", "")
    end
    file = expanduser(String(get(cfg, "file", default_file)))
    variable = String(get(cfg, "variable", default_variable))
    isempty(file) && throw(ArgumentError("surface_flux.kind=$(kind) requires surface_flux.file"))
    isempty(variable) && throw(ArgumentError("surface_flux.kind=$(kind) requires surface_flux.variable"))

    default_time_index = kind === :gridfed_fossil_co2 ? Int(get(cfg, "month", 0)) : 1
    time_index = Int(get(cfg, "time_index", default_time_index))
    if kind === :gridfed_fossil_co2 && time_index < 1
        throw(ArgumentError("surface_flux.kind=gridfed_fossil_co2 requires surface_flux.time_index or surface_flux.month"))
    end
    time_index < 1 && throw(ArgumentError("surface_flux.time_index must be >= 1"))
    return file, variable, time_index
end

function _normalize_units_string(units)
    units_str = String(units)
    return lowercase(replace(strip(units_str), " " => "", "^" => "", "²" => "2"))
end

function _load_file_surface_flux_field(cfg, ::Type{FT}) where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing
    file, variable, time_index = _resolve_surface_flux_file(cfg, kind)
    isfile(file) || throw(ArgumentError("surface-flux file not found: $file"))

    ds = NCDataset(file)
    try
        lon_var = _ic_find_coord(ds, ["lon", "longitude", "x"])
        lat_var = _ic_find_coord(ds, ["lat", "latitude", "y"])
        isnothing(lon_var) && throw(ArgumentError("could not find longitude coordinate in $file"))
        isnothing(lat_var) && throw(ArgumentError("could not find latitude coordinate in $file"))
        haskey(ds, variable) || throw(ArgumentError("variable '$variable' not found in $file"))

        lon_src = Float64.(ds[lon_var][:])
        lat_src = Float64.(ds[lat_var][:])

        raw_var = ds[variable]
        raw = if ndims(raw_var) == 3
            FT.(nomissing(raw_var[:, :, time_index], zero(FT)))
        elseif ndims(raw_var) == 2
            FT.(nomissing(raw_var[:, :], zero(FT)))
        else
            throw(ArgumentError("surface-flux variable '$variable' must be 2D or 3D, got ndims=$(ndims(raw_var))"))
        end

        cell_area = if haskey(ds, "cell_area")
            Float64.(nomissing(ds["cell_area"][:, :], 0.0))
        else
            nothing
        end

        if length(lat_src) > 1 && lat_src[1] > lat_src[end]
            raw = raw[:, end:-1:1]
            lat_src = reverse(lat_src)
            cell_area === nothing || (cell_area = cell_area[:, end:-1:1])
        end

        if minimum(lon_src) < 0
            split = findfirst(>=(0), lon_src)
            if split !== nothing
                idx = vcat(split:length(lon_src), 1:split-1)
                lon_src = mod.(lon_src[idx], 360.0)
                raw = raw[idx, :]
                cell_area === nothing || (cell_area = cell_area[idx, :])
            end
        end

        units_norm = _normalize_units_string(get(raw_var.attrib, "units", ""))
        if kind === :gridfed_fossil_co2 || units_norm == "kgco2/month/m2"
            raw ./= FT(SECONDS_PER_MONTH)
        elseif !(isempty(units_norm) || occursin("/s", units_norm) || occursin("s-1", units_norm))
            throw(ArgumentError("unsupported surface-flux units '$units_norm' in $file; expected kgCO2/month/m2 or per-second flux units"))
        end

        raw .*= FT(get(cfg, "scale", 1.0))
        native_total_mass_rate = cell_area === nothing ? NaN : sum(Float64.(raw) .* cell_area)
        return FileSurfaceFluxField{FT}(raw, lon_src, lat_src, native_total_mass_rate)
    finally
        close(ds)
    end
end

function _renormalize_surface_flux_rate!(rate::AbstractArray{FT}, source::FileSurfaceFluxField) where FT
    isfinite(source.native_total_mass_rate) || return rate
    sampled_total = Float64(sum(rate))
    sampled_total > 0 || return rate
    scale = source.native_total_mass_rate / sampled_total
    rate .*= FT(scale)
    return rate
end

_copy_cfg_dict(cfg) = Dict{String, Any}(String(k) => v for (k, v) in pairs(cfg))

function _tracer_init_cfg(tracer_cfg)
    if haskey(tracer_cfg, "init")
        return _copy_cfg_dict(tracer_cfg["init"])
    end

    cfg = Dict{String, Any}()
    for key in ("kind", "background", "lon0_deg", "lat0_deg", "sigma_lon_deg",
                "sigma_lat_deg", "amplitude", "file", "variable", "time_index")
        haskey(tracer_cfg, key) && (cfg[key] = tracer_cfg[key])
    end
    isempty(cfg) && return Dict{String, Any}("kind" => "uniform", "background" => 0.0)
    return cfg
end

function _tracer_surface_flux_cfg(tracer_cfg)
    if haskey(tracer_cfg, "surface_flux")
        return _copy_cfg_dict(tracer_cfg["surface_flux"])
    end

    cfg = Dict{String, Any}()
    for (src_key, dst_key) in (("surface_flux_kind", "kind"),
                               ("surface_flux_file", "file"),
                               ("surface_flux_variable", "variable"),
                               ("surface_flux_time_index", "time_index"),
                               ("surface_flux_month", "month"),
                               ("surface_flux_scale", "scale"))
        haskey(tracer_cfg, src_key) && (cfg[dst_key] = tracer_cfg[src_key])
    end
    return cfg
end

function _parse_tracer_specs(cfg)
    tracers_cfg = get(cfg, "tracers", nothing)
    tracers_cfg isa AbstractDict || return nothing

    names = sort!(collect(keys(tracers_cfg)))
    isempty(names) && throw(ArgumentError("config has [tracers] but no tracer sections"))
    return Tuple(TransportTracerSpec(Symbol(name),
                                     _tracer_init_cfg(tracers_cfg[name]),
                                     _tracer_surface_flux_cfg(tracers_cfg[name])) for name in names)
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

"""
    _interpolate_log_pressure_profile!(dest, src_q, air_mass_col, ap, bp, ps_src, area, g)

Vertically interpolate a source profile `src_q[1:Nsrc]` onto target model
levels `dest[1:Nz]` using log-pressure linear interpolation.

## Source pressure levels

Source half-level pressures from the IC file's hybrid coordinates:

    src_p_half[k] = ap[k] + bp[k] × ps_src     k = 1..Nsrc+1

**Ordering convention** (Catrine startCO2): `ap[1] = 0, bp[1] = 1.0` →
`src_p_half[1] = ps_src` (SURFACE). `ap[end] = 0, bp[end] = 0` →
`src_p_half[end] = 0` (TOA). So `src_p_mid` is **decreasing** from
surface to TOA (k=1 is surface, k=Nsrc is TOA).

## Target pressure levels

Target half-level pressures from the model's air mass column:

    tgt_p_half[1] = 0          (TOA, model convention k=1 = top)
    tgt_p_half[k+1] = tgt_p_half[k] + air_mass[k] × g / area

So `tgt_p_mid` is **increasing** from TOA to surface (k=1 small, k=Nz large).

## Interpolation

For each target level `k`, find the bracket in `src_p_mid` where
`src_p_mid[ks] > p_tgt ≥ src_p_mid[ks+1]` (decreasing source → advance
`ks` forward for larger p_tgt). Then linear interpolation in log-pressure:

    w = (log p_tgt − log p_src[ks]) / (log p_src[ks+1] − log p_src[ks])
    dest[k] = src_q[ks] + w × (src_q[ks+1] − src_q[ks])

Clamps: if `p_tgt > src_p_mid[1]` (below source surface), use `src_q[1]`.
If `p_tgt < src_p_mid[end]` (above source TOA), use `src_q[end]`.

**NOTE**: the `ks` index is persistent across the `k` loop (not reset)
because `p_tgt` is monotonically **increasing** with `k` (target goes
TOA → surface), so `ks` only needs to advance forward (toward lower src
pressures) — never backward. This relies on the source `src_p_mid` being
monotonically **decreasing**.
"""
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

    # Source half-level pressures: src_p_half[1] = ps (surface), src_p_half[end] = 0 (TOA)
    src_p_half = Vector{Float64}(undef, Nsrc + 1)
    @inbounds for k in 1:(Nsrc + 1)
        src_p_half[k] = ap[k] + bp[k] * ps_src
    end
    # Source mid-level pressures (decreasing: surface → TOA)
    src_p_mid = Vector{Float64}(undef, Nsrc)
    @inbounds for k in 1:Nsrc
        src_p_mid[k] = 0.5 * (src_p_half[k] + src_p_half[k + 1])
    end

    # Target half-level pressures: tgt_p_half[1] = 0 (TOA), cumulating to ps (surface)
    # (model convention: k=1 is TOA, k=Nz is surface-adjacent)
    tgt_p_half = Vector{Float64}(undef, Nz + 1)
    tgt_p_half[1] = 0.0   # TOA boundary
    @inbounds for k in 1:Nz
        dp = Float64(air_mass_col[k]) * g / area   # pressure thickness [Pa]
        tgt_p_half[k + 1] = tgt_p_half[k] + dp
    end

    # Persistent bracket index (advances forward as p_tgt increases with k)
    ks = 1
    @inbounds for k in 1:Nz
        p_tgt = 0.5 * (tgt_p_half[k] + tgt_p_half[k + 1])  # target mid-level [Pa]
        if p_tgt >= src_p_mid[1]
            # Below source surface → clamp to surface-level value
            dest[k] = src_q[1]
        elseif p_tgt <= src_p_mid[end]
            # Above source TOA → clamp to TOA-level value
            dest[k] = src_q[end]
        else
            # Find bracket: src_p_mid[ks] > p_tgt ≥ src_p_mid[ks+1]
            # (src is decreasing, so advance ks until src_p_mid[ks+1] ≤ p_tgt)
            while ks < Nsrc && src_p_mid[ks + 1] > p_tgt
                ks += 1
            end
            # Log-pressure linear interpolation
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

function build_initial_mixing_ratio(air_mass::AbstractArray{FT}, mesh::AtmosTransport.LatLonMesh{FT}, cfg) where FT
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

function build_initial_mixing_ratio(air_mass::AbstractArray{FT}, mesh::AtmosTransport.ReducedGaussianMesh{FT}, cfg) where FT
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
        for j in 1:AtmosTransport.nrings(mesh)
            lats = mesh.latitudes[j]
            lons = AtmosTransport.ring_longitudes(mesh, j)
            for i in eachindex(lons)
                c = AtmosTransport.cell_index(mesh, i, j)
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
                                    grid::AtmosTransport.AtmosGrid{<:AtmosTransport.LatLonMesh},
                                    cfg) where FT
    kind = _init_kind(cfg)
    _is_file_init_kind(kind) || return build_initial_mixing_ratio(air_mass, grid.horizontal, cfg)

    source = _load_file_initial_condition_source(cfg, FT, size(air_mass, 3))
    mesh = grid.horizontal
    q = Array{FT}(undef, size(air_mass))
    src_q = Vector{FT}(undef, size(source.raw, 3))
    g = Float64(AtmosTransport.gravity(grid))

    for j in axes(q, 2)
        area = Float64(AtmosTransport.cell_area(mesh, 1, j))
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
                                    grid::AtmosTransport.AtmosGrid{<:AtmosTransport.ReducedGaussianMesh},
                                    cfg) where FT
    kind = _init_kind(cfg)
    _is_file_init_kind(kind) || return build_initial_mixing_ratio(air_mass, grid.horizontal, cfg)

    source = _load_file_initial_condition_source(cfg, FT, size(air_mass, 2))
    mesh = grid.horizontal
    q = Array{FT}(undef, size(air_mass))
    src_q = Vector{FT}(undef, size(source.raw, 3))
    g = Float64(AtmosTransport.gravity(grid))

    for j in 1:AtmosTransport.nrings(mesh)
        lat = mesh.latitudes[j]
        lons = AtmosTransport.ring_longitudes(mesh, j)
        for i in eachindex(lons)
            c = AtmosTransport.cell_index(mesh, i, j)
            lon = lons[i]
            _sample_bilinear_profile!(src_q, source.raw, source.lon, source.lat, lon, lat)
            if source.needs_vinterp
                ps_src = _sample_bilinear_scalar(source.psurf, source.lon, source.lat, lon, lat)
                area = Float64(AtmosTransport.cell_area(mesh, c))
                _interpolate_log_pressure_profile!(@view(q[c, :]), src_q, @view(air_mass[c, :]),
                                                  source.ap, source.bp, ps_src, area, g)
            else
                _copy_profile!(@view(q[c, :]), src_q)
            end
        end
    end

    return q
end

# ---------------------------------------------------------------------------
# Surface flux regridding: bilinear (legacy) and conservative
# ---------------------------------------------------------------------------

const _REGRID_CACHE_DIR = expanduser("~/.cache/AtmosTransport/cr_regridding")

"""
    _build_emission_source_mesh(source::FileSurfaceFluxField)

Construct a `LatLonMesh` matching the source emission file's regular grid.
The source coordinates after `_load_file_surface_flux_field` are guaranteed
to be in [0, 360) longitude and ascending latitude.
"""
function _build_emission_source_mesh(source::FileSurfaceFluxField)
    Nx_src = length(source.lon)
    Ny_src = length(source.lat)
    dlon = source.lon[2] - source.lon[1]
    dlat = source.lat[2] - source.lat[1]
    # Infer face boundaries from cell centers
    lon_west = source.lon[1] - dlon / 2
    lon_east = source.lon[end] + dlon / 2
    lat_south = source.lat[1] - dlat / 2
    lat_north = source.lat[end] + dlat / 2
    # Clamp to valid ranges (floating point from [-180,180)→[0,360) shift
    # can produce lon_east slightly > 360)
    lat_south = max(lat_south, -90.0)
    lat_north = min(lat_north, 90.0)
    if lon_east - lon_west > 360.0
        lon_east = lon_west + 360.0
    end
    return AtmosTransport.LatLonMesh(; Nx=Nx_src, Ny=Ny_src,
        longitude=(lon_west, lon_east), latitude=(lat_south, lat_north))
end

"""
    _conservative_surface_flux_rate(source, dst_mesh, FT)

Conservatively regrid emission flux density [kg/m²/s] from the source LL grid
to the destination mesh. Returns cell mass rates [kg/s] as a flat vector.
"""
function _conservative_surface_flux_rate(source::FileSurfaceFluxField,
                                         dst_mesh::AtmosTransport.AbstractHorizontalMesh,
                                         ::Type{FT}) where FT
    src_mesh = _build_emission_source_mesh(source)
    regridder = build_regridder(src_mesh, dst_mesh; cache_dir=_REGRID_CACHE_DIR)

    # Flatten source flux density to 1D (column-major: i + (j-1)*Nx)
    src_flat = vec(Float64.(source.raw))
    n_dst = length(regridder.dst_areas)
    dst_flat = zeros(Float64, n_dst)
    apply_regridder!(dst_flat, regridder, src_flat)

    # Convert flux density [kg/m²/s] → mass rate [kg/s] using regridder areas
    rate = Array{FT}(undef, n_dst)
    for c in 1:n_dst
        rate[c] = FT(dst_flat[c] * regridder.dst_areas[c])
    end

    # Verify global mass conservation
    src_total = sum(src_flat .* regridder.src_areas)
    dst_total = sum(Float64.(rate))
    rel_err = abs(dst_total - src_total) / max(abs(src_total), 1e-30)
    @info @sprintf("  Conservative regrid: src_total=%.6e  dst_total=%.6e  rel_err=%.2e kg/s",
                   src_total, dst_total, rel_err)
    rel_err > 1e-6 && @warn @sprintf("  Conservative regrid mass conservation warning: rel_err=%.2e", rel_err)

    return rate
end

"""Parse regridding method from config: "conservative" or "bilinear" (default)."""
_regridding_method(cfg) = Symbol(lowercase(String(get(cfg, "regridding", "bilinear"))))

function build_surface_flux_source(grid::AtmosTransport.AtmosGrid{<:AtmosTransport.LatLonMesh},
                                   tracer_name::Symbol,
                                   cfg,
                                   ::Type{FT}) where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing

    source = _load_file_surface_flux_field(cfg, FT)
    method = _regridding_method(cfg)
    mesh = grid.horizontal

    if method === :conservative
        rate_flat = _conservative_surface_flux_rate(source, mesh, FT)
        # Reshape to (Nx, Ny) for structured grids
        rate = reshape(rate_flat, AtmosTransport.nx(mesh), AtmosTransport.ny(mesh))
    else
        # Legacy bilinear sampling
        rate = Array{FT}(undef, AtmosTransport.nx(mesh), AtmosTransport.ny(mesh))
        for j in axes(rate, 2)
            area = Float64(AtmosTransport.cell_area(mesh, 1, j))
            lat = mesh.φᶜ[j]
            for i in axes(rate, 1)
                lon = mesh.λᶜ[i]
                flux_density = _sample_bilinear_scalar(source.raw, source.lon, source.lat, lon, lat)
                rate[i, j] = FT(flux_density * area)
            end
        end
        _renormalize_surface_flux_rate!(rate, source)
    end

    return AtmosTransport.SurfaceFluxSource(tracer_name, rate)
end

function build_surface_flux_source(grid::AtmosTransport.AtmosGrid{<:AtmosTransport.ReducedGaussianMesh},
                                   tracer_name::Symbol,
                                   cfg,
                                   ::Type{FT}) where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing

    source = _load_file_surface_flux_field(cfg, FT)
    method = _regridding_method(cfg)
    mesh = grid.horizontal

    if method === :conservative
        rate = _conservative_surface_flux_rate(source, mesh, FT)
    else
        # Legacy bilinear sampling
        rate = Array{FT}(undef, AtmosTransport.ncells(mesh))
        for j in 1:AtmosTransport.nrings(mesh)
            lat = mesh.latitudes[j]
            lons = AtmosTransport.ring_longitudes(mesh, j)
            for i in eachindex(lons)
                c = AtmosTransport.cell_index(mesh, i, j)
                flux_density = _sample_bilinear_scalar(source.raw, source.lon, source.lat, lons[i], lat)
                rate[c] = FT(flux_density * Float64(AtmosTransport.cell_area(mesh, c)))
            end
        end
        _renormalize_surface_flux_rate!(rate, source)
    end

    return AtmosTransport.SurfaceFluxSource(tracer_name, rate)
end

function build_surface_flux_sources(grid, tracer_specs, ::Type{FT}) where FT
    sources = Any[]
    for spec in tracer_specs
        source = build_surface_flux_source(grid, spec.name, spec.surface_flux_cfg, FT)
        source === nothing || push!(sources, source)
    end
    return Tuple(sources)
end

function make_model(driver::AtmosTransport.TransportBinaryDriver;
                    FT::Type{<:AbstractFloat},
                    recipe::AtmosTransport.RuntimePhysicsRecipe,
                    tracer_name::Union{Symbol, Nothing}=nothing,
                    init_cfg=nothing,
                    tracer_specs=nothing,
                    cfg=Dict{String, Any}())
    grid = AtmosTransport.driver_grid(driver)
    window = AtmosTransport.load_transport_window(driver, 1)
    air_mass = copy(window.air_mass)

    tracer_specs_tuple = if tracer_specs === nothing
        name = something(tracer_name, :CO2)
        init_cfg_dict = init_cfg === nothing ? Dict{String, Any}("kind" => "uniform", "background" => 0.0) : _copy_cfg_dict(init_cfg)
        (TransportTracerSpec(name, init_cfg_dict, Dict{String, Any}()),)
    else
        Tuple(tracer_specs)
    end
    isempty(tracer_specs_tuple) && throw(ArgumentError("at least one tracer must be configured"))

    tracer_names = Tuple(spec.name for spec in tracer_specs_tuple)
    rm_arrays = map(tracer_specs_tuple) do spec
        q = build_initial_mixing_ratio(air_mass, grid, spec.init_cfg)
        return q .* air_mass
    end

    basis_type = AtmosTransport.air_mass_basis(driver) == :dry ? AtmosTransport.DryBasis : AtmosTransport.MoistBasis
    tracer_tuple = NamedTuple{tracer_names}(Tuple(rm_arrays))
    state = AtmosTransport.CellState(basis_type, air_mass; tracer_tuple...)
    fluxes = AtmosTransport.allocate_face_fluxes(grid.horizontal, AtmosTransport.nlevels(grid); FT=FT, basis=basis_type)
    model = AtmosTransport.TransportModel(state, fluxes, grid, recipe.advection;
                                          diffusion = recipe.diffusion,
                                          convection = recipe.convection)
    adaptor = backend_array_adapter(cfg)
    return adaptor === Array ? model : Adapt.adapt(adaptor, model)
end

"""Capture column-mean mixing ratio for each tracer at current model state.

Handles both structured grids (3D arrays: Nx × Ny × Nz) and face-indexed
grids like reduced Gaussian (2D arrays: ncells × Nz).
"""
function _capture_snapshot(model)
    names = AtmosTransport.tracer_names(model.state)
    m = Array(model.state.air_mass)
    result = Dict{Symbol, Vector{Float64}}()
    for name in names
        rm = Array(getfield(model.state.tracers, name))
        # Column-mean VMR: Σ(rm_k) / Σ(m_k) per column
        if ndims(rm) == 3
            # Structured grid: (Nx, Ny, Nz)
            Nx, Ny, Nz = size(rm)
            col_mean = zeros(Float64, Nx * Ny)
            for j in 1:Ny, i in 1:Nx
                sum_rm = 0.0; sum_m = 0.0
                for k in 1:Nz
                    sum_rm += Float64(rm[i, j, k])
                    sum_m  += Float64(m[i, j, k])
                end
                col_mean[(j-1)*Nx + i] = sum_m > 0 ? sum_rm / sum_m : 0.0
            end
        elseif ndims(rm) == 2
            # Face-indexed grid (reduced Gaussian): (ncells, Nz)
            Nc, Nz = size(rm)
            col_mean = zeros(Float64, Nc)
            for c in 1:Nc
                sum_rm = 0.0; sum_m = 0.0
                for k in 1:Nz
                    sum_rm += Float64(rm[c, k])
                    sum_m  += Float64(m[c, k])
                end
                col_mean[c] = sum_m > 0 ? sum_rm / sum_m : 0.0
            end
        else
            error("_capture_snapshot: unsupported tracer array ndims=$(ndims(rm))")
        end
        result[name] = col_mean
    end
    return result
end

"""Write captured snapshots to a CF-compliant NetCDF file.

For LatLon grids: uses proper `(lon, lat, time)` dimensions so Panoply
renders them directly on a map.

For ReducedGaussian grids: uses `(cell, time)` with auxiliary `lon(cell)`
and `lat(cell)` coordinate variables + CF `coordinates` attribute so
Panoply can do unstructured scatter rendering.
"""
function _write_snapshot_netcdf(path, snapshots, snapshot_hours, grid)
    isempty(snapshots) && return
    mkpath(dirname(path))
    mesh = grid.horizontal
    ntime = length(snapshots)

    NCDataset(path, "c") do ds
        ds.attrib["Conventions"] = "CF-1.8"
        ds.attrib["grid"] = summary(mesh)

        if mesh isa AtmosTransport.LatLonMesh
            _write_snapshot_ll!(ds, snapshots, snapshot_hours, mesh, ntime)
        elseif mesh isa AtmosTransport.ReducedGaussianMesh
            _write_snapshot_rg!(ds, snapshots, snapshot_hours, mesh, ntime)
        else
            error("Unsupported mesh type for snapshot output: $(typeof(mesh))")
        end
    end
    @info "Saved snapshots: $path ($(length(snapshots)) times)"
end

function _write_snapshot_ll!(ds, snapshots, snapshot_hours, mesh, ntime)
    Nx, Ny = AtmosTransport.nx(mesh), AtmosTransport.ny(mesh)

    defDim(ds, "lon", Nx)
    defDim(ds, "lat", Ny)
    defDim(ds, "time", ntime)

    v_lon = defVar(ds, "lon", Float64, ("lon",),
                   attrib=Dict("units" => "degrees_east", "long_name" => "longitude",
                               "standard_name" => "longitude"))
    v_lat = defVar(ds, "lat", Float64, ("lat",),
                   attrib=Dict("units" => "degrees_north", "long_name" => "latitude",
                               "standard_name" => "latitude"))
    v_time = defVar(ds, "time", Float64, ("time",),
                    attrib=Dict("units" => "hours since start", "long_name" => "time"))

    v_lon[:] = Float64.(mesh.λᶜ)
    v_lat[:] = Float64.(mesh.φᶜ)
    v_time[:] = snapshot_hours[1:ntime]

    for name in keys(first(snapshots))
        vname = "$(name)_column_mean"
        v = defVar(ds, vname, Float64, ("lon", "lat", "time"),
                   attrib=Dict("units" => "mol mol-1",
                               "long_name" => "Column-mean $name VMR"))
        for t in 1:ntime
            flat = snapshots[t][name]
            v[:, :, t] = reshape(flat, Nx, Ny)
        end
    end
end

"""Write RG snapshots regridded to a regular LatLon grid for Panoply compatibility.

Uses nearest-ring mapping: each output latitude picks the closest RG ring,
and each output longitude picks the closest cell on that ring. This is exact
for visualization — no smoothing, no weight building.
"""
function _write_snapshot_rg!(ds, snapshots, snapshot_hours, mesh, ntime)
    nr = AtmosTransport.nrings(mesh)

    # Build output grid: nlat = nrings, nlon = max(nlon_per_ring)
    Nlon = maximum(mesh.nlon_per_ring)
    Nlat = nr
    dlon = 360.0 / Nlon
    out_lons = [(i - 0.5) * dlon for i in 1:Nlon]  # cell centers [0, 360)
    out_lats = Float64.(mesh.latitudes)             # ring center latitudes

    # Build nearest-neighbor map: for each (i_lon, j_lat) in output, find the
    # index of the closest RG cell on ring j_lat.
    nn_map = zeros(Int, Nlon, Nlat)
    for j in 1:Nlat
        nlon_ring = mesh.nlon_per_ring[j]
        dlon_ring = 360.0 / nlon_ring
        for i in 1:Nlon
            # Find nearest cell on this ring
            i_ring = round(Int, out_lons[i] / dlon_ring + 0.5)
            i_ring = clamp(i_ring, 1, nlon_ring)
            nn_map[i, j] = AtmosTransport.cell_index(mesh, i_ring, j)
        end
    end

    defDim(ds, "lon", Nlon)
    defDim(ds, "lat", Nlat)
    defDim(ds, "time", ntime)

    ds.attrib["grid_type"] = "reduced_gaussian_regridded"
    ds.attrib["nrings"] = nr
    ds.attrib["regridding"] = "nearest-neighbor from reduced Gaussian"

    v_lon = defVar(ds, "lon", Float64, ("lon",),
                   attrib=Dict("units" => "degrees_east", "long_name" => "longitude",
                               "standard_name" => "longitude"))
    v_lat = defVar(ds, "lat", Float64, ("lat",),
                   attrib=Dict("units" => "degrees_north", "long_name" => "latitude",
                               "standard_name" => "latitude"))
    v_time = defVar(ds, "time", Float64, ("time",),
                    attrib=Dict("units" => "hours since start", "long_name" => "time"))

    v_lon[:] = out_lons
    v_lat[:] = out_lats
    v_time[:] = snapshot_hours[1:ntime]

    for name in keys(first(snapshots))
        vname = "$(name)_column_mean"
        v = defVar(ds, vname, Float64, ("lon", "lat", "time"),
                   attrib=Dict("units" => "mol mol-1",
                               "long_name" => "Column-mean $name VMR"))
        for t in 1:ntime
            flat = snapshots[t][name]
            regridded = zeros(Float64, Nlon, Nlat)
            for j in 1:Nlat, i in 1:Nlon
                regridded[i, j] = flat[nn_map[i, j]]
            end
            v[:, :, t] = regridded
        end
    end
end

function run_sequence(binary_paths::Vector{String}, cfg)
    FT = Symbol(get(get(cfg, "numerics", Dict{String, Any}()), "float_type", "Float64")) == :Float32 ? Float32 : Float64
    run_cfg = get(cfg, "run", Dict{String, Any}())
    start_window = Int(get(run_cfg, "start_window", 1))
    stop_window_override = get(run_cfg, "stop_window", nothing)
    # Plan 39 Commit G: reset_air_mass_each_window removed from DrivenSimulation.
    # Config flag, if present, is silently ignored with a debug note.
    if haskey(run_cfg, "reset_air_mass_each_window")
        @debug "run.reset_air_mass_each_window config key is ignored (plan 39 Commit G removed the flag)"
    end
    init_cfg = get(cfg, "init", Dict{String, Any}())
    tracer_specs = something(_parse_tracer_specs(cfg),
                             (TransportTracerSpec(Symbol(get(run_cfg, "tracer_name", "CO2")),
                                                  _copy_cfg_dict(init_cfg),
                                                  Dict{String, Any}()),))

    # Snapshot configuration
    output_cfg = get(cfg, "output", Dict{String, Any}())
    snapshot_hours = Float64.(get(output_cfg, "snapshot_hours", Float64[]))
    snapshot_file = expanduser(String(get(output_cfg, "snapshot_file", "")))
    do_snapshots = !isempty(snapshot_hours) && !isempty(snapshot_file)

    isempty(binary_paths) && throw(ArgumentError("no binary_paths configured"))
    ensure_gpu_runtime!(cfg)

    first_driver = AtmosTransport.TransportBinaryDriver(first(binary_paths); FT=FT, arch=AtmosTransport.CPU())
    recipe = AtmosTransport.build_runtime_physics_recipe(cfg, first_driver, FT)
    model = make_model(first_driver; FT=FT, recipe=recipe, tracer_specs=tracer_specs, cfg=cfg)
    surface_sources = build_surface_flux_sources(AtmosTransport.driver_grid(first_driver), tracer_specs, FT)
    m0 = AtmosTransport.total_air_mass(model.state)
    tracer_masses0 = Dict(name => AtmosTransport.total_mass(model.state, name) for name in AtmosTransport.tracer_names(model.state))
    source_tracers = Set(source.tracer_name for source in surface_sources)
    @info "Backend: $(backend_label(cfg))"
    @info "Physics: advection=$(nameof(typeof(recipe.advection))) diffusion=$(nameof(typeof(recipe.diffusion))) convection=$(nameof(typeof(recipe.convection)))"
    for source in surface_sources
        @info @sprintf("Surface source %s total mass rate: %.12e kg/s",
                       String(source.tracer_name), Float64(sum(source.cell_mass_rate)))
    end

    # Snapshot state
    snapshots = Dict{Symbol, Vector{Float64}}[]
    snap_idx = 1
    total_elapsed_hours = 0.0

    # Capture initial snapshot (hour 0) if requested
    if do_snapshots && snap_idx <= length(snapshot_hours) && abs(snapshot_hours[snap_idx]) < 0.5
        push!(snapshots, _capture_snapshot(model))
        @info @sprintf("Snapshot %d at t=%.0fh", snap_idx, 0.0)
        snap_idx += 1
    end

    for (idx, path) in enumerate(binary_paths)
        driver = idx == 1 ? first_driver : AtmosTransport.TransportBinaryDriver(path; FT=FT, arch=AtmosTransport.CPU())
        AtmosTransport.validate_runtime_physics_recipe(recipe, driver)
        stop_window = stop_window_override === nothing ? AtmosTransport.total_windows(driver) : Int(stop_window_override)
        initialize_air_mass = idx == 1
        sim = AtmosTransport.DrivenSimulation(model, driver;
                               start_window=start_window,
                               stop_window=stop_window,
                               initialize_air_mass=initialize_air_mass,
                               surface_sources=surface_sources)
        model = sim.model
        if !initialize_air_mass
            boundary_rel = maximum(abs.(model.state.air_mass .- sim.window.air_mass)) / max(maximum(abs.(sim.window.air_mass)), eps(FT))
            @info @sprintf("Boundary air-mass mismatch before %s: %.3e", basename(path), boundary_rel)
        end
        window_hours = Float64(AtmosTransport.window_dt(driver)) / 3600.0
        n_windows = stop_window - start_window + 1
        @info @sprintf("Running %s with %s on %s (%d windows)",
                       basename(path),
                       nameof(typeof(recipe.advection)),
                       summary(AtmosTransport.driver_grid(driver).horizontal),
                       n_windows)
        synchronize_backend!(cfg)
        t0 = time()

        if do_snapshots
            # Window-by-window loop with snapshot capture
            for w in 1:n_windows
                AtmosTransport.run_window!(sim)
                total_elapsed_hours += window_hours
                # Check if current hour matches a snapshot hour
                while snap_idx <= length(snapshot_hours) &&
                      abs(total_elapsed_hours - snapshot_hours[snap_idx]) < 0.5
                    push!(snapshots, _capture_snapshot(model))
                    @info @sprintf("Snapshot %d at t=%.0fh", snap_idx, total_elapsed_hours)
                    snap_idx += 1
                end
            end
        else
            AtmosTransport.run!(sim)
            total_elapsed_hours += n_windows * window_hours
        end

        synchronize_backend!(cfg)
        @info @sprintf("Finished %s in %.2f s", basename(path), time() - t0)
        close(driver)
    end

    # Write snapshots if configured
    if do_snapshots && !isempty(snapshots)
        _write_snapshot_netcdf(snapshot_file, snapshots, snapshot_hours,
                              AtmosTransport.driver_grid(first_driver))
    end

    m1 = AtmosTransport.total_air_mass(model.state)
    @info @sprintf("Final air-mass change vs initial state:  %.3e", (m1 - m0) / m0)
    for name in AtmosTransport.tracer_names(model.state)
        rm0 = Float64(tracer_masses0[name])
        rm1 = Float64(AtmosTransport.total_mass(model.state, name))
        if name in source_tracers
            @info @sprintf("Final tracer mass for %s (with source): %.12e kg", String(name), rm1)
        elseif abs(rm0) > eps(Float64)
            @info @sprintf("Final tracer-mass drift for %s:         %.3e", String(name), (rm1 - rm0) / rm0)
        else
            @info @sprintf("Final tracer mass for %s:               %.12e kg", String(name), rm1)
        end
    end
    return model
end

function main()
    base_logger = ConsoleLogger(stderr, Logging.Info; show_limited=false)
    global_logger(base_logger)

    isempty(ARGS) && error("Usage: julia --project=. scripts/run_transport_binary.jl config.toml")
    cfg = TOML.parsefile(expanduser(ARGS[1]))
    binary_paths = [expanduser(String(p)) for p in get(get(cfg, "input", Dict{String, Any}()), "binary_paths", String[])]
    run_sequence(binary_paths, cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
