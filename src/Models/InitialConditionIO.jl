"""
    InitialConditionIO

Single owner of initial-condition I/O, vertical remap, and
topology-dispatched VMR builders for the unified runtime.

## Public API (exported via `Models` → `AtmosTransport`)

- [`build_initial_mixing_ratio`](@ref) — topology-dispatched builder
  returning **dry VMR** on interior cells. Accepts `kind = uniform |
  gaussian_blob | file | netcdf | file_field | catrine_co2` for LL/RG
  meshes. CS file-based dispatch is added in plan 40 Commit 1c.
- [`pack_initial_tracer_mass`](@ref) — basis-aware VMR → tracer-mass
  conversion. Dispatches on `mass_basis::AbstractMassBasis`:
  - `DryBasis` (default per CLAUDE.md invariant 14): `rm = vmr .* air_mass`.
  - `MoistBasis`: `rm = vmr .* air_mass .* (1 .- qv)` per CLAUDE.md
    invariant 9; `qv` must be supplied.
- [`FileInitialConditionSource`](@ref) — container for a loaded IC
  NetCDF (3D VMR + hybrid coefficients + surface pressure).

## Plan 40 commit provenance

- 1a (2026-04-24, commit `1450204`) — module scaffold.
- 1b (2026-04-24, commit `29fcca5`) — LL/RG hoist (verbatim, bit-exact)
  + basis-aware `pack_initial_tracer_mass` (LL/RG).
- 1c (this file) — CS `build_initial_mixing_ratio` for
  `uniform | file | catrine_co2 | netcdf | file_field`; CS
  `pack_initial_tracer_mass` (DryBasis + MoistBasis, halo-padded
  output with halo zeroed); `_build_source_latlon_mesh` helper for
  LL→CS conservative regridding.
- 1d (this file) — Hoist of surface-flux NetCDF loader (13 helpers
  + `FileSurfaceFluxField` struct) and LL/RG `build_surface_flux_source`
  methods; new CS `build_surface_flux_source` that conservatively
  LL→CS-regrids the flux (the regridder's `dst_areas` × regridded
  density already yields kg/s per cell) and unpacks to
  `NTuple{6, Matrix{FT}}` — satisfying the per-cell kg/s contract
  at `src/Operators/SurfaceFlux/sources.jl:12`.

Private helpers (underscore-prefixed) stay unexported and are
accessed by callers (including `scripts/run_transport_binary.jl`)
via `AtmosTransport.Models.InitialConditionIO.<name>` if needed.
"""
module InitialConditionIO

using NCDatasets

using ..State: AbstractMassBasis, DryBasis, MoistBasis
using ..Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh,
                nrings, ring_longitudes, cell_index, cell_area,
                gravity
# Regridding + Preprocessing are loaded before Models (AtmosTransport.jl,
# plan 40 Commit 1c reorder) so we can pull in the LL→CS conservative
# regridder + panel unpacking helpers for the CS file-based IC path.
using ..Regridding: build_regridder, apply_regridder!
using ..Preprocessing: unpack_flat_to_panels_3d!, unpack_flat_to_panels_2d!,
                       CS_PANEL_COUNT
using ..Operators.SurfaceFlux: SurfaceFluxSource
# Grid accessors used by both LL/RG and CS surface-flux builders
using ..Grids: AbstractHorizontalMesh, nx, ny, ncells

using Printf: @sprintf

# ---------------------------------------------------------------------------
# Longitude-wrap helpers (hoisted from run_transport_binary.jl:29,30)
# ---------------------------------------------------------------------------

# NOTE: the source arrays (Catrine, GridFED) may be in [-180, 180)
# convention; these wrap both to a common [0, 360) before looking up.
@inline wrapped_longitude_distance(lon, lon0) = abs(mod(lon - lon0 + 180, 360) - 180)
@inline wrapped_longitude_360(lon) = mod(lon, 360)

# ---------------------------------------------------------------------------
# Config-kind resolvers (hoisted from run_transport_binary.jl:32,33)
# ---------------------------------------------------------------------------

@inline _init_kind(cfg) = Symbol(lowercase(String(get(cfg, "kind", "uniform"))))
@inline _is_file_init_kind(kind::Symbol) = kind in (:file, :netcdf, :file_field, :catrine_co2)

# ---------------------------------------------------------------------------
# FileInitialConditionSource struct (hoisted from run_transport_binary.jl:59)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Bracket search + bilinear interpolation (hoisted from :82,99,132,157)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# IC config resolver + NetCDF loader (hoisted from :198, :353)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Vertical log-pressure interpolation (hoisted from :466)
# ---------------------------------------------------------------------------

"""
    _interpolate_log_pressure_profile!(dest, src_q, ap, bp, ps_src, A_tgt, B_tgt, ps_tgt)

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

Target half-level pressures from the transport binary's hybrid coordinates and
stored surface pressure:

    tgt_p_half[k] = A_tgt[k] + B_tgt[k] × ps_tgt

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
                                            ap::Vector{Float64},
                                            bp::Vector{Float64},
                                            ps_src::Float64,
                                            A_tgt::AbstractVector{<:Real},
                                            B_tgt::AbstractVector{<:Real},
                                            ps_tgt::Real) where FT
    Nsrc = length(src_q)
    Nz = length(dest)
    length(A_tgt) == Nz + 1 || throw(DimensionMismatch(
        "A_tgt has length $(length(A_tgt)), expected Nz+1 = $(Nz + 1)"))
    length(B_tgt) == Nz + 1 || throw(DimensionMismatch(
        "B_tgt has length $(length(B_tgt)), expected Nz+1 = $(Nz + 1)"))

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

    # Target half-level pressures from the binary's own hybrid coefficients
    # and surface pressure: `p_half[k] = A[k] + B[k] * ps_tgt`. This is
    # *exact* and decouples vertical remap from `air_mass × g / area`,
    # which previously drifted by 9-22% on gnomonic CS because
    # `mesh.cell_areas[i, j]` was inconsistent with the area used by the
    # preprocessor when writing `m`. Visible symptom (2026-04-24): cube
    # panel-outline structure in C48 column-mean IC, dissolved by transport
    # within ~30 h.
    ps_tgt_f = Float64(ps_tgt)
    tgt_p_half = Vector{Float64}(undef, Nz + 1)
    @inbounds for k in 1:(Nz + 1)
        tgt_p_half[k] = Float64(A_tgt[k]) + Float64(B_tgt[k]) * ps_tgt_f
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

# ---------------------------------------------------------------------------
# Topology-dispatched VMR builder (hoisted from :570, :593, :622, :653)
#
# `build_initial_mixing_ratio` returns **dry VMR** on interior cells.
# Shapes:
#   LL: (Nx, Ny, Nz)
#   RG: (ncells, Nz)
#   CS: NTuple{6, Array{FT, 3}} of (Nc, Nc, Nz) — added in plan 40 Commit 1c
# ---------------------------------------------------------------------------

function build_initial_mixing_ratio(air_mass::AbstractArray{FT}, mesh::LatLonMesh{FT}, cfg) where FT
    kind = _init_kind(cfg)
    background = FT(get(cfg, "background", 4.0e-4))
    if kind === :uniform
        return fill(background, size(air_mass))
    elseif kind === :bl_enhanced
        # Flat background + enhancement in the lowest `n_layers` model levels
        # (k = Nz-n_layers+1:Nz, since k=1=TOA, k=Nz=surface). Layer-based
        # threshold follows terrain naturally.
        n_layers = Int(get(cfg, "n_layers", 3))
        enhancement = FT(get(cfg, "enhancement", 1.0e-4))
        Nz = size(air_mass, 3)
        n_layers >= 1 && n_layers <= Nz ||
            throw(ArgumentError("init.kind=bl_enhanced: n_layers=$(n_layers) must satisfy 1 ≤ n_layers ≤ Nz=$(Nz)"))
        q = fill(background, size(air_mass))
        @views q[:, :, (Nz - n_layers + 1):Nz] .+= enhancement
        return q
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

function build_initial_mixing_ratio(air_mass::AbstractArray{FT}, mesh::ReducedGaussianMesh{FT}, cfg) where FT
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

function build_initial_mixing_ratio(air_mass::AbstractArray{FT},
                                    grid::AtmosGrid{<:LatLonMesh},
                                    cfg;
                                    surface_pressure::Union{Nothing, AbstractMatrix} = nothing) where FT
    kind = _init_kind(cfg)
    _is_file_init_kind(kind) || return build_initial_mixing_ratio(air_mass, grid.horizontal, cfg)

    source = _load_file_initial_condition_source(cfg, FT, size(air_mass, 3))
    mesh = grid.horizontal
    q = Array{FT}(undef, size(air_mass))
    src_q = Vector{FT}(undef, size(source.raw, 3))
    A_tgt = grid.vertical.A
    B_tgt = grid.vertical.B
    surface_pressure === nothing && throw(ArgumentError(
        "build_initial_mixing_ratio(::AtmosGrid{<:LatLonMesh}, ...) with " *
        "vertical-interp init kind=$(kind) requires `surface_pressure` (the " *
        "binary's stored ps) so target half-level pressures can be computed " *
        "exactly from the grid's hybrid coefficients. Pass " *
        "`window.surface_pressure` from `load_transport_window` to avoid " *
        "the area-mismatch artifact (2026-04-24)."))
    size(surface_pressure) == size(air_mass)[1:2] || throw(DimensionMismatch(
        "surface_pressure size $(size(surface_pressure)) must match air_mass " *
        "horizontal extent $(size(air_mass)[1:2])"))

    for j in axes(q, 2)
        lat = mesh.φᶜ[j]
        for i in axes(q, 1)
            lon = mesh.λᶜ[i]
            _sample_bilinear_profile!(src_q, source.raw, source.lon, source.lat, lon, lat)
            if source.needs_vinterp
                ps_src = _sample_bilinear_scalar(source.psurf, source.lon, source.lat, lon, lat)
                ps_tgt = Float64(surface_pressure[i, j])
                _interpolate_log_pressure_profile!(@view(q[i, j, :]), src_q,
                                                   source.ap, source.bp, ps_src,
                                                   A_tgt, B_tgt, ps_tgt)
            else
                _copy_profile!(@view(q[i, j, :]), src_q)
            end
        end
    end

    return q
end

function build_initial_mixing_ratio(air_mass::AbstractArray{FT},
                                    grid::AtmosGrid{<:ReducedGaussianMesh},
                                    cfg;
                                    surface_pressure::Union{Nothing, AbstractVector} = nothing) where FT
    kind = _init_kind(cfg)
    _is_file_init_kind(kind) || return build_initial_mixing_ratio(air_mass, grid.horizontal, cfg)

    source = _load_file_initial_condition_source(cfg, FT, size(air_mass, 2))
    mesh = grid.horizontal
    q = Array{FT}(undef, size(air_mass))
    src_q = Vector{FT}(undef, size(source.raw, 3))
    A_tgt = grid.vertical.A
    B_tgt = grid.vertical.B
    surface_pressure === nothing && throw(ArgumentError(
        "build_initial_mixing_ratio(::AtmosGrid{<:ReducedGaussianMesh}, ...) " *
        "with vertical-interp init kind=$(kind) requires `surface_pressure` " *
        "(the binary's stored ps, length = ncells). Pass " *
        "`window.surface_pressure` from `load_transport_window`."))
    length(surface_pressure) == size(air_mass, 1) || throw(DimensionMismatch(
        "surface_pressure length $(length(surface_pressure)) must match " *
        "air_mass cells $(size(air_mass, 1))"))

    for j in 1:nrings(mesh)
        lat = mesh.latitudes[j]
        lons = ring_longitudes(mesh, j)
        for i in eachindex(lons)
            c = cell_index(mesh, i, j)
            lon = lons[i]
            _sample_bilinear_profile!(src_q, source.raw, source.lon, source.lat, lon, lat)
            if source.needs_vinterp
                ps_src = _sample_bilinear_scalar(source.psurf, source.lon, source.lat, lon, lat)
                ps_tgt = Float64(surface_pressure[c])
                _interpolate_log_pressure_profile!(@view(q[c, :]), src_q,
                                                   source.ap, source.bp, ps_src,
                                                   A_tgt, B_tgt, ps_tgt)
            else
                _copy_profile!(@view(q[c, :]), src_q)
            end
        end
    end

    return q
end

# ---------------------------------------------------------------------------
# pack_initial_tracer_mass — basis-aware VMR → tracer-mass conversion
#
# Rule (feedback_vmr_to_mass_basis_aware, 2026-04-24): IC VMRs are dry.
# - DryBasis:   air_mass == m_dry   → rm = vmr .* air_mass
# - MoistBasis: air_mass == m_moist → rm = vmr .* air_mass .* (1 .- qv)
#               (per CLAUDE.md invariant 9)
# ---------------------------------------------------------------------------

"""
    pack_initial_tracer_mass(grid, air_mass, vmr_dry; mass_basis::AbstractMassBasis,
                                                      qv = nothing)

Convert dry volume mixing ratio `vmr_dry` to tracer-mass storage
matching the binary's mass-basis contract. Returns an array of the
same shape as `air_mass`.

## Dispatch

- `mass_basis::DryBasis` — `air_mass` is `m_dry` per CLAUDE.md
  invariant 14. Result: `vmr_dry .* air_mass`. `qv` is ignored.
- `mass_basis::MoistBasis` — `air_mass` is `m_moist` per CLAUDE.md
  invariant 9. Result: `vmr_dry .* air_mass .* (1 .- qv)`. `qv` must
  be supplied from the first transport window; missing `qv` errors.

CS dispatch (per-panel halo packing) is added in plan 40 Commit 1c.

## Arguments

- `grid`           — `AtmosGrid{<:LatLonMesh}` or `AtmosGrid{<:ReducedGaussianMesh}`
                     (CS added in 1c).
- `air_mass`       — storage-shaped air mass from the transport
                     window. Shape matches `vmr_dry`.
- `vmr_dry`        — dry volume mixing ratio, same shape as `air_mass`.
- `mass_basis`     — `DryBasis()` or `MoistBasis()`.
- `qv`             — specific humidity, same shape as `air_mass`;
                     required iff `mass_basis isa MoistBasis`.
"""
function pack_initial_tracer_mass(grid::AtmosGrid, air_mass, vmr_dry;
                                  mass_basis::AbstractMassBasis,
                                  qv = nothing)
    return _pack_tracer_mass(grid, air_mass, vmr_dry, mass_basis, qv)
end

function _pack_tracer_mass(::AtmosGrid{<:LatLonMesh}, air_mass, vmr_dry, ::DryBasis, qv)
    return vmr_dry .* air_mass
end

function _pack_tracer_mass(::AtmosGrid{<:ReducedGaussianMesh}, air_mass, vmr_dry, ::DryBasis, qv)
    return vmr_dry .* air_mass
end

function _pack_tracer_mass(::AtmosGrid{<:LatLonMesh}, air_mass, vmr_dry, ::MoistBasis, qv)
    qv === nothing && throw(ArgumentError(
        "pack_initial_tracer_mass on MoistBasis requires qv (specific humidity) " *
        "from the first transport window; got qv=nothing. See CLAUDE.md invariant 9."))
    size(qv) == size(air_mass) || throw(DimensionMismatch(
        "qv shape $(size(qv)) must match air_mass shape $(size(air_mass))"))
    return vmr_dry .* air_mass .* (1 .- qv)
end

function _pack_tracer_mass(::AtmosGrid{<:ReducedGaussianMesh}, air_mass, vmr_dry, ::MoistBasis, qv)
    qv === nothing && throw(ArgumentError(
        "pack_initial_tracer_mass on MoistBasis requires qv (specific humidity) " *
        "from the first transport window; got qv=nothing. See CLAUDE.md invariant 9."))
    size(qv) == size(air_mass) || throw(DimensionMismatch(
        "qv shape $(size(qv)) must match air_mass shape $(size(air_mass))"))
    return vmr_dry .* air_mass .* (1 .- qv)
end

# ---------------------------------------------------------------------------
# CS file-based IC source mesh construction (plan 40 Commit 1c)
#
# Build a LatLonMesh matching the source NetCDF's lon/lat grid so the
# conservative regridder in `src/Regridding/` can LL→CS the 3D VMR field
# (and the 2D surface-pressure field, when vertical interpolation is
# needed).
#
# The source arrays (after `_load_file_initial_condition_source`) are
# guaranteed to be in [0, 360) longitude + ascending latitude because the
# loader already rolls/reverses them. We infer face boundaries from cell
# centres, assuming uniform spacing (standard for all ERA5 / Catrine /
# GridFED products).
# ---------------------------------------------------------------------------

function _build_source_latlon_mesh(lon_src::Vector{Float64}, lat_src::Vector{Float64}, ::Type{FT}) where FT
    Nx_src = length(lon_src)
    Ny_src = length(lat_src)
    dlon = lon_src[2] - lon_src[1]
    dlat = lat_src[2] - lat_src[1]
    lon_west  = lon_src[1]   - dlon / 2
    lon_east  = lon_src[end] + dlon / 2
    lat_south = lat_src[1]   - dlat / 2
    lat_north = lat_src[end] + dlat / 2
    lat_south = max(lat_south, -90.0)
    lat_north = min(lat_north, 90.0)
    if lon_east - lon_west > 360.0
        lon_east = lon_west + 360.0
    end
    return LatLonMesh(; FT = FT, Nx = Nx_src, Ny = Ny_src,
                      longitude = (lon_west, lon_east),
                      latitude  = (lat_south, lat_north))
end

# ---------------------------------------------------------------------------
# CS build_initial_mixing_ratio (plan 40 Commit 1c)
#
# Returns a 6-tuple of interior `(Nc, Nc, Nz)` VMR arrays. The tuple is
# topology-shaped but halo-free; the halo ring is added later by
# `pack_initial_tracer_mass` so that different halo widths can share the
# same IC builder.
# ---------------------------------------------------------------------------

function build_initial_mixing_ratio(air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                                    grid::AtmosGrid{<:CubedSphereMesh},
                                    cfg;
                                    surface_pressure::Union{Nothing, NTuple{6, <:AbstractMatrix}} = nothing) where FT
    kind = _init_kind(cfg)
    mesh = grid.horizontal
    Nc = mesh.Nc
    Nz = size(air_mass[1], 3)

    if kind === :uniform
        background = FT(get(cfg, "background", 4.0e-4))
        return ntuple(_ -> fill(background, Nc, Nc, Nz), CS_PANEL_COUNT)
    elseif _is_file_init_kind(kind)
        surface_pressure === nothing && throw(ArgumentError(
            "build_initial_mixing_ratio(::AtmosGrid{<:CubedSphereMesh}, ...) " *
            "with vertical-interp init kind=$(kind) requires `surface_pressure` " *
            "(the binary's stored ps as `NTuple{6, Matrix}`). Pass " *
            "`window.surface_pressure` from `load_transport_window` so target " *
            "half-level pressures use the binary's hybrid coefficients exactly. " *
            "Without this, the gnomonic `mesh.cell_areas[i,j]` mismatch with " *
            "the preprocessor's area produces a 9-22% pressure drift and " *
            "visible cube-panel artifacts (2026-04-24)."))
        for p in 1:CS_PANEL_COUNT
            size(surface_pressure[p]) == (Nc, Nc) || throw(DimensionMismatch(
                "surface_pressure[$p] size $(size(surface_pressure[p])) must be ($Nc, $Nc)"))
        end
        return _build_cs_file_ic(grid, air_mass, cfg, FT, surface_pressure)
    else
        throw(ArgumentError(
            "unsupported init.kind=$(kind) for CubedSphereMesh; " *
            "supported: uniform | file | netcdf | file_field | catrine_co2"))
    end
end

function _build_cs_file_ic(grid::AtmosGrid{<:CubedSphereMesh},
                           air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                           cfg, ::Type{FT},
                           ps_tgt_panels::NTuple{6, <:AbstractMatrix}) where FT
    mesh = grid.horizontal
    Nc   = mesh.Nc
    Nz   = size(air_mass[1], 3)
    A_tgt = grid.vertical.A
    B_tgt = grid.vertical.B

    source = _load_file_initial_condition_source(cfg, FT, Nz)
    src_mesh = _build_source_latlon_mesh(source.lon, source.lat, FT)
    regridder = build_regridder(src_mesh, mesh)

    # 3D VMR: (Nx_src, Ny_src, Nlev_src) → 6 × (Nc, Nc, Nlev_src)
    Nlev_src = size(source.raw, 3)
    n_src    = length(source.lon) * length(source.lat)
    n_dst    = CS_PANEL_COUNT * Nc * Nc
    src_flat = Matrix{FT}(undef, n_src, Nlev_src)
    dst_flat = Matrix{FT}(undef, n_dst, Nlev_src)
    copyto!(src_flat, reshape(source.raw, n_src, Nlev_src))
    apply_regridder!(dst_flat, regridder, src_flat)
    vmr_src_levels = ntuple(_ -> Array{FT}(undef, Nc, Nc, Nlev_src), CS_PANEL_COUNT)
    unpack_flat_to_panels_3d!(vmr_src_levels, dst_flat, Nc, Nlev_src)

    # 2D source surface pressure (only if source levels differ from target).
    # NOTE: this is the SOURCE psurf (Catrine), used to build source p-half
    # levels. The TARGET ps comes from `ps_tgt_panels` passed in by the caller
    # — the binary's own ps. Mixing the two cleanly is what fixes the
    # area-mismatch artifact.
    src_ps_panels = if source.needs_vinterp
        src_ps_flat = Vector{Float64}(undef, n_src)
        dst_ps_flat = Vector{Float64}(undef, n_dst)
        copyto!(src_ps_flat, reshape(source.psurf, n_src))
        apply_regridder!(dst_ps_flat, regridder, src_ps_flat)
        panels_ps = ntuple(_ -> Matrix{Float64}(undef, Nc, Nc), CS_PANEL_COUNT)
        unpack_flat_to_panels_2d!(panels_ps, dst_ps_flat, Nc)
        panels_ps
    else
        nothing
    end

    # Vertical remap column-by-column into interior `(Nc, Nc, Nz)` tuple.
    vmr = ntuple(_ -> Array{FT}(undef, Nc, Nc, Nz), CS_PANEL_COUNT)
    src_q = Vector{FT}(undef, Nlev_src)

    for p in 1:CS_PANEL_COUNT
        for j in 1:Nc, i in 1:Nc
            @views copyto!(src_q, vmr_src_levels[p][i, j, :])
            if source.needs_vinterp
                ps_src = Float64(src_ps_panels[p][i, j])
                ps_tgt = Float64(ps_tgt_panels[p][i, j])
                _interpolate_log_pressure_profile!(@view(vmr[p][i, j, :]),
                                                   src_q,
                                                   source.ap, source.bp, ps_src,
                                                   A_tgt, B_tgt, ps_tgt)
            else
                _copy_profile!(@view(vmr[p][i, j, :]), src_q)
            end
        end
    end

    return vmr
end

# ---------------------------------------------------------------------------
# CS pack_initial_tracer_mass (plan 40 Commit 1c)
#
# Takes interior `NTuple{6, (Nc, Nc, Nz)}` VMR + halo-padded
# `NTuple{6, (Nc+2Hp, Nc+2Hp, Nz)}` air_mass. Returns halo-padded tracer
# mass with the halo ring zeroed; halo exchanges during the run populate
# those cells.
# ---------------------------------------------------------------------------

function _pack_tracer_mass(grid::AtmosGrid{<:CubedSphereMesh},
                           air_mass::NTuple{6, <:AbstractArray},
                           vmr_dry::NTuple{6, <:AbstractArray},
                           ::DryBasis,
                           qv)
    return _cs_pack_interior_into_halo(grid, air_mass, vmr_dry, nothing)
end

function _pack_tracer_mass(grid::AtmosGrid{<:CubedSphereMesh},
                           air_mass::NTuple{6, <:AbstractArray},
                           vmr_dry::NTuple{6, <:AbstractArray},
                           ::MoistBasis,
                           qv)
    qv === nothing && throw(ArgumentError(
        "pack_initial_tracer_mass on MoistBasis requires qv (specific humidity) " *
        "from the first transport window; got qv=nothing. See CLAUDE.md invariant 9."))
    qv isa NTuple{6} || throw(ArgumentError(
        "CS pack_initial_tracer_mass on MoistBasis requires qv::NTuple{6}; " *
        "got $(typeof(qv))"))
    return _cs_pack_interior_into_halo(grid, air_mass, vmr_dry, qv)
end

function _cs_pack_interior_into_halo(grid::AtmosGrid{<:CubedSphereMesh},
                                     air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                                     vmr::NTuple{6, <:AbstractArray{FT, 3}},
                                     qv::Union{Nothing, NTuple{6}}) where FT
    mesh = grid.horizontal
    Nc = mesh.Nc
    Hp = mesh.Hp
    out = ntuple(p -> zeros(FT, size(air_mass[p])...), CS_PANEL_COUNT)
    for p in 1:CS_PANEL_COUNT
        size(vmr[p]) == (Nc, Nc, size(air_mass[p], 3)) || throw(DimensionMismatch(
            "CS panel $p: vmr has shape $(size(vmr[p])), expected $((Nc, Nc, size(air_mass[p], 3)))"))
        interior_am = @view air_mass[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]
        interior_out = @view out[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]
        if qv === nothing
            interior_out .= vmr[p] .* interior_am
        else
            size(qv[p]) == size(air_mass[p]) || throw(DimensionMismatch(
                "CS panel $p: qv has shape $(size(qv[p])), expected $(size(air_mass[p]))"))
            interior_qv = @view qv[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]
            interior_out .= vmr[p] .* interior_am .* (1 .- interior_qv)
        end
    end
    return out
end

# ===========================================================================
# Surface-flux loader + builders (plan 40 Commit 1d)
#
# Owns the file-based surface-flux path: NetCDF load + LL→{LL,RG,CS}
# conservative regrid + area integration. Every builder returns
# `SurfaceFluxSource` with cell_mass_rate in **kg/s per cell** (not
# kg/m²/s) — the contract required by
# `src/Operators/SurfaceFlux/sources.jl:12`.
#
# Hoisted verbatim (modulo renames for dependency consolidation) from
# scripts/run_transport_binary.jl:
#   FileSurfaceFluxField, SECONDS_PER_MONTH, _surface_flux_kind,
#   _resolve_surface_flux_file, _normalize_units_string,
#   _load_file_surface_flux_field, _renormalize_surface_flux_rate!,
#   _REGRID_CACHE_DIR, _conservative_surface_flux_rate,
#   _regridding_method, build_surface_flux_source (LL + RG),
#   build_surface_flux_sources.
#
# `_build_emission_source_mesh` is dropped in favour of the shared
# `_build_source_latlon_mesh` introduced for the IC path.
# ===========================================================================

const SECONDS_PER_MONTH = 365.25 * 86400 / 12

const _REGRID_CACHE_DIR = expanduser("~/.cache/AtmosTransport/cr_regridding")

"""
    FileSurfaceFluxField{FT}

2-D flux density `(Nx_src, Ny_src)` loaded from a NetCDF emission file.
Units are kg/m²/s after `_load_file_surface_flux_field` has normalised
the native units. `native_total_mass_rate` is the pre-regrid global
integral, preserved for the bilinear path's renormalisation (the
conservative path does not need it).
"""
struct FileSurfaceFluxField{FT}
    raw                    :: Array{FT, 2}
    lon                    :: Vector{Float64}
    lat                    :: Vector{Float64}
    native_total_mass_rate :: Float64
end

@inline _surface_flux_kind(cfg) = Symbol(lowercase(String(get(cfg, "kind", "none"))))

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

        cell_area_src = if haskey(ds, "cell_area")
            Float64.(nomissing(ds["cell_area"][:, :], 0.0))
        else
            nothing
        end

        if length(lat_src) > 1 && lat_src[1] > lat_src[end]
            raw = raw[:, end:-1:1]
            lat_src = reverse(lat_src)
            cell_area_src === nothing || (cell_area_src = cell_area_src[:, end:-1:1])
        end

        if minimum(lon_src) < 0
            split = findfirst(>=(0), lon_src)
            if split !== nothing
                idx = vcat(split:length(lon_src), 1:split-1)
                lon_src = mod.(lon_src[idx], 360.0)
                raw = raw[idx, :]
                cell_area_src === nothing || (cell_area_src = cell_area_src[idx, :])
            end
        end

        units_norm = _normalize_units_string(get(raw_var.attrib, "units", ""))
        if kind === :gridfed_fossil_co2 || units_norm == "kgco2/month/m2"
            raw ./= FT(SECONDS_PER_MONTH)
        elseif !(isempty(units_norm) || occursin("/s", units_norm) || occursin("s-1", units_norm))
            throw(ArgumentError("unsupported surface-flux units '$units_norm' in $file; expected kgCO2/month/m2 or per-second flux units"))
        end

        raw .*= FT(get(cfg, "scale", 1.0))
        native_total_mass_rate = cell_area_src === nothing ? NaN : sum(Float64.(raw) .* cell_area_src)
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

"""Parse regridding method from config: "conservative" or "bilinear" (default)."""
_regridding_method(cfg) = Symbol(lowercase(String(get(cfg, "regridding", "bilinear"))))

"""
    _conservative_surface_flux_rate(source, dst_mesh, FT) -> Vector{FT}

Conservatively regrid flux density [kg/m²/s] onto `dst_mesh`; return a
flat vector of per-cell mass rates [kg/s] (already area-integrated).
Callers reshape/wrap per topology.
"""
function _conservative_surface_flux_rate(source::FileSurfaceFluxField,
                                         dst_mesh::AbstractHorizontalMesh,
                                         ::Type{FT}) where FT
    src_mesh = _build_source_latlon_mesh(source.lon, source.lat, FT)
    regridder = build_regridder(src_mesh, dst_mesh; cache_dir = _REGRID_CACHE_DIR)

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

    # Report global mass conservation (warn only; conservative regrid is exact to ~FP ulps)
    src_total = sum(src_flat .* regridder.src_areas)
    dst_total = sum(Float64.(rate))
    rel_err = abs(dst_total - src_total) / max(abs(src_total), 1e-30)
    @info @sprintf("  Conservative regrid: src_total=%.6e  dst_total=%.6e  rel_err=%.2e kg/s",
                   src_total, dst_total, rel_err)
    rel_err > 1e-6 && @warn @sprintf("  Conservative regrid mass conservation warning: rel_err=%.2e", rel_err)

    return rate
end

# ---------------------------------------------------------------------------
# build_surface_flux_source — LL / RG / CS
# ---------------------------------------------------------------------------

function build_surface_flux_source(grid::AtmosGrid{<:LatLonMesh},
                                   tracer_name::Symbol, cfg, ::Type{FT}) where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing

    source = _load_file_surface_flux_field(cfg, FT)
    method = _regridding_method(cfg)
    mesh = grid.horizontal

    if method === :conservative
        rate_flat = _conservative_surface_flux_rate(source, mesh, FT)
        rate = reshape(rate_flat, nx(mesh), ny(mesh))
    else
        # Legacy bilinear sampling
        rate = Array{FT}(undef, nx(mesh), ny(mesh))
        for j in axes(rate, 2)
            area = Float64(cell_area(mesh, 1, j))
            lat = mesh.φᶜ[j]
            for i in axes(rate, 1)
                lon = mesh.λᶜ[i]
                flux_density = _sample_bilinear_scalar(source.raw, source.lon, source.lat, lon, lat)
                rate[i, j] = FT(flux_density * area)
            end
        end
        _renormalize_surface_flux_rate!(rate, source)
    end

    return SurfaceFluxSource(tracer_name, rate)
end

function build_surface_flux_source(grid::AtmosGrid{<:ReducedGaussianMesh},
                                   tracer_name::Symbol, cfg, ::Type{FT}) where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing

    source = _load_file_surface_flux_field(cfg, FT)
    method = _regridding_method(cfg)
    mesh = grid.horizontal

    if method === :conservative
        rate = _conservative_surface_flux_rate(source, mesh, FT)
    else
        # Legacy bilinear sampling
        rate = Array{FT}(undef, ncells(mesh))
        for j in 1:nrings(mesh)
            lat = mesh.latitudes[j]
            lons = ring_longitudes(mesh, j)
            for i in eachindex(lons)
                c = cell_index(mesh, i, j)
                flux_density = _sample_bilinear_scalar(source.raw, source.lon, source.lat, lons[i], lat)
                rate[c] = FT(flux_density * Float64(cell_area(mesh, c)))
            end
        end
        _renormalize_surface_flux_rate!(rate, source)
    end

    return SurfaceFluxSource(tracer_name, rate)
end

"""
    build_surface_flux_source(grid::AtmosGrid{<:CubedSphereMesh},
                              tracer_name, cfg, ::Type{FT})

Plan 40 Commit 1d: CS surface-flux builder. Conservatively LL→CS
regrids the 2-D flux density (kg/m²/s) onto the 6 CS panel cell
centres, then multiplies by each panel's cell area to yield per-cell
kg/s — the contract `SurfaceFluxSource` expects
(`src/Operators/SurfaceFlux/sources.jl:12`). Returns a
`SurfaceFluxSource` whose `cell_mass_rate` is an `NTuple{6, Matrix{FT}}`
of interior-only `(Nc, Nc)` panels.

`cfg` must set `kind` (non-`none`); any of the file-based surface-flux
kinds `_load_file_surface_flux_field` understands work
(`gridfed_fossil_co2` or user-supplied `file` + `variable`).
Conservative regridding is enforced — CS bilinear is not supported.
"""
function build_surface_flux_source(grid::AtmosGrid{<:CubedSphereMesh},
                                   tracer_name::Symbol, cfg, ::Type{FT}) where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing

    method = _regridding_method(cfg)
    method === :conservative || @warn "CS surface-flux: `regridding = \"$(method)\"` requested; CS supports conservative only — forcing conservative."

    source = _load_file_surface_flux_field(cfg, FT)
    mesh = grid.horizontal
    Nc = mesh.Nc

    # _conservative_surface_flux_rate already returns kg/s per cell
    # (regridder.dst_areas × regridded flux density), so the panel unpack
    # only needs to reshape the flat `6*Nc^2` vector into 6 × (Nc, Nc).
    rate_flat = _conservative_surface_flux_rate(source, mesh, FT)
    length(rate_flat) == CS_PANEL_COUNT * Nc * Nc || throw(DimensionMismatch(
        "CS surface-flux conservative regrid returned $(length(rate_flat)) cells; expected $(CS_PANEL_COUNT * Nc * Nc)"))

    panels = ntuple(_ -> Matrix{FT}(undef, Nc, Nc), CS_PANEL_COUNT)
    unpack_flat_to_panels_2d!(panels, rate_flat, Nc)

    return SurfaceFluxSource(tracer_name, panels)
end

"""
    build_surface_flux_sources(grid, tracer_specs, ::Type{FT})

Build `SurfaceFluxSource` instances for every tracer spec that requests
one. Returns a tuple (possibly empty) suitable for the
`surface_sources = (…,)` kwarg on `DrivenSimulation`.
"""
function build_surface_flux_sources(grid, tracer_specs, ::Type{FT}) where FT
    sources = Any[]
    for spec in tracer_specs
        source = build_surface_flux_source(grid, spec.name, spec.surface_flux_cfg, FT)
        source === nothing || push!(sources, source)
    end
    return Tuple(sources)
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export FileInitialConditionSource
export build_initial_mixing_ratio
export pack_initial_tracer_mass
export FileSurfaceFluxField
export build_surface_flux_source, build_surface_flux_sources

end # module InitialConditionIO
