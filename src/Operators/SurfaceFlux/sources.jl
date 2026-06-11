"""
    AbstractSurfaceFluxSource

Supertype for all per-tracer surface emission sources consumed by the
surface-flux operator and stored in a [`PerTracerFluxMap`](@ref).

Concrete subtypes:

- [`SurfaceFluxSource`](@ref) — a single static per-cell rate array.
- [`TimeVaryingSurfaceFluxSource`](@ref) — a time series of per-cell
  rate slices plus a sorted `times` vector; the operator linearly
  interpolates to the simulation clock at `apply!` time.

Every subtype carries a `tracer_name :: Symbol` field used by
`flux_for` to key the map.
"""
abstract type AbstractSurfaceFluxSource end

"""
    SurfaceFluxSource{RateT}

A single-tracer surface source: a `tracer_name` plus a `cell_mass_rate`
array supplying model-storage amount added **per cell per second** to the
surface layer.

- `tracer_name :: Symbol` — matches a name in `CellState.tracer_names`.
- `cell_mass_rate :: RateT` — one of:
  - a 2D `(Nx, Ny)` array for structured grids
  - a 1D `(Nc,)` array for face-indexed grids
  - an `NTuple{6}` of 2D `(Nc, Nc)` arrays for cubed-sphere panels
  The rates are already area-integrated. For dry-VMR tracers, file-based
  physical fluxes in kg species/s are converted by the source builder to
  dry-air-equivalent storage units before reaching this struct. The surface
  flux kernel applies `rm_surface += rate × dt` without multiplying by cell
  area.

# Why per-cell rates (not kg/m²/s)

The prognostic tracer is stored per cell, so a per-cell rate × dt is the
natural unit and matches the legacy
`DrivenSimulation._apply_surface_source!` signature. A per-area variant
that multiplies by `cell_area` is deferred to a follow-up.

# Provenance

Originally introduced in `src/Models/DrivenSimulation.jl`, then migrated
to `src/Operators/SurfaceFlux/` so that `SurfaceFluxOperator` can consume
it; the name is still re-exported from `AtmosTransport` for backward
compat with external callers that imported it by fully-qualified name.

# Fields
- `tracer_name :: Symbol`
- `cell_mass_rate :: RateT` — backend-agnostic; `Adapt.adapt` converts
  the array between host and device transparently via
  `Adapt.adapt_structure`.
"""
struct SurfaceFluxSource{RateT, CompT} <: AbstractSurfaceFluxSource
    tracer_name    :: Symbol
    cell_mass_rate :: RateT
    compensation   :: CompT   # Kahan compensation; same shape as cell_mass_rate
end

# Outer constructor: allocate zero compensation matching the rate shape.
SurfaceFluxSource(name::Symbol, rate::RateT) where {RateT} =
    SurfaceFluxSource(name, rate, _alloc_flux_comp(rate))

# Adapt hook: carry both the rate and the compensation to the device.
function Adapt.adapt_structure(to, source::SurfaceFluxSource)
    cell_mass_rate = Adapt.adapt(to, source.cell_mass_rate)
    compensation   = Adapt.adapt(to, source.compensation)
    return SurfaceFluxSource{typeof(cell_mass_rate), typeof(compensation)}(
        source.tracer_name, cell_mass_rate, compensation)
end

# Allocate a zero compensation array matching the shape of a static rate.
_alloc_flux_comp(rate::AbstractArray) = zero(rate)
_alloc_flux_comp(rates::NTuple{6})    = ntuple(p -> zero(rates[p]), Val(6))

# Allocate a zero compensation array matching ONE time-slice of a rate series
# (drops the trailing time dimension). Uses `similar` to match the source's
# array type and backend — preserves device placement for GPU-resident series.
_alloc_flux_comp_from_series(s::AbstractArray{FT}) where {FT} =
    fill!(similar(s, size(s)[1:end-1]...), zero(FT))
_alloc_flux_comp_from_series(s::NTuple{6}) =
    ntuple(p -> _alloc_flux_comp_from_series(s[p]), Val(6))

# =========================================================================
# Temporal reconstruction schemes — how a time-varying source maps its
# stored slices onto the simulation clock. Dispatch-selected; every scheme
# resolves to the SAME `(i0, i1, w0, w1)` two-slice blend the interp kernels
# consume, so adding a scheme never touches the kernels.
# =========================================================================

"""
    AbstractFluxTemporalScheme

How a [`TimeVaryingSurfaceFluxSource`](@ref) reconstructs a per-step rate
from its stored time slices at `t = current_time(meteo)` (and step `dt`).
Concrete schemes implement

    _flux_temporal_weights(scheme, times, t, dt) -> (i0, i1, w0, w1)

returning two bracketing slice indices and blend weights (`w0 + w1 == 1`)
applied as `w0·series[i0] + w1·series[i1]`. New reconstructions (e.g.
mass-flux-integral matching, higher-order in time) are added as new
subtypes + one `_flux_temporal_weights` method — no kernel or operator
change.
"""
abstract type AbstractFluxTemporalScheme end

"""
    StepwiseFlux()

Piecewise-constant in time: emit the value of the slice *block* containing
`t` (the largest slice time `≤ t`), with no blending. This holds each
3-hourly CAMS value constant across its block — the "change the flux in 3h
blocks" option.
"""
struct StepwiseFlux <: AbstractFluxTemporalScheme end

"""
    LinearInterpFlux()

Linear interpolation between the two bracketing slices, point-evaluated at
the step's `current_time(meteo)`. HEMCO-like treatment of instantaneous
3-hourly fields. Default scheme.
"""
struct LinearInterpFlux <: AbstractFluxTemporalScheme end

"""
    ConservativeMeanFlux()

Window/step integral-conserving: evaluate the linear reconstruction at the
step centre `t + dt/2`. Summed over the substeps of a met window, this
midpoint quadrature reproduces the exact trapezoidal time-integral of a
piecewise-linear flux — i.e. the emitted mass per window matches the source's
own integral, independent of the substep count.
"""
struct ConservativeMeanFlux <: AbstractFluxTemporalScheme end

"""
    _flux_temporal_weights(scheme, times, t, dt) -> (i0, i1, w0, w1)

Resolve the two-slice blend for `scheme` at time `t` (seconds since run
start) with step length `dt` (seconds). All schemes funnel through the same
`(i0, i1, w0, w1)` contract so the interp kernels are scheme-agnostic.
"""
@inline _flux_temporal_weights(::LinearInterpFlux, times, t, dt) =
    _time_interp_bracket(times, t)

@inline _flux_temporal_weights(::ConservativeMeanFlux, times, t, dt) =
    _time_interp_bracket(times, t + dt / 2)

@inline function _flux_temporal_weights(::StepwiseFlux, times, t, dt)
    n = length(times)
    n == 0 && throw(ArgumentError("_flux_temporal_weights: empty times vector"))
    t <= times[1] && return (1, 1, 1.0, 0.0)
    k = searchsortedlast(times, t)   # largest k with times[k] <= t
    return (k, k, 1.0, 0.0)
end

# -------------------------------------------------------------------------
# Exact time-integral over a step [t, t+dt]: split at every flux time knot
# inside the step so each sub-interval is integrated exactly (the source
# reconstruction is smooth — linear or constant — within a knot interval).
# Returns a vector of `(i0, i1, w0, w1, dt_frac)` segments with
# `Σ dt_frac == 1`; the operator applies the two-slice blend with `dt·dt_frac`
# per segment, so the emitted mass reproduces the scheme's intended integral
# regardless of how the step straddles the source cadence.
#
# `LinearInterpFlux` is a *point* interpolation at the step's current time, so
# it is deliberately a single full-step segment (no knot split). `StepwiseFlux`
# and `ConservativeMeanFlux` split at knots and integrate exactly (piecewise
# constant / piecewise linear, respectively).
# -------------------------------------------------------------------------
const _FluxSegment = Tuple{Int, Int, Float64, Float64, Float64}

@inline _flux_temporal_segments(::LinearInterpFlux, times, t, dt) =
    _FluxSegment[(_time_interp_bracket(times, t)..., 1.0)]

_flux_temporal_segments(s::StepwiseFlux, times, t, dt) =
    _knot_split_segments(times, t, dt, m -> _flux_temporal_weights(s, times, m, dt))

_flux_temporal_segments(::ConservativeMeanFlux, times, t, dt) =
    _knot_split_segments(times, t, dt, m -> _time_interp_bracket(times, m))

# Build the knot-split segments of [t, t+dt], evaluating `point_weights(mid)`
# (an (i0,i1,w0,w1) tuple) at each sub-interval midpoint.
function _knot_split_segments(times, t, dt, point_weights)
    t0 = Float64(t); t1 = t0 + Float64(dt)
    segs = _FluxSegment[]
    dt > 0 || (push!(segs, (point_weights(t0)..., 1.0)); return segs)
    lo = t0
    @inbounds for k in eachindex(times)
        tk = Float64(times[k])
        if tk > lo + eps(t1) && tk < t1 - eps(t1)
            mid = 0.5 * (lo + tk)
            push!(segs, (point_weights(mid)..., (tk - lo) / dt))
            lo = tk
        end
    end
    push!(segs, (point_weights(0.5 * (lo + t1))..., (t1 - lo) / dt))
    return segs
end

"""
    flux_temporal_scheme(name) -> AbstractFluxTemporalScheme

Map a config string to a temporal scheme. `"stepwise"`/`"block"` →
[`StepwiseFlux`](@ref); `"linear"`/`"interp"` → [`LinearInterpFlux`](@ref);
`"conservative"`/`"window_mean"`/`"integral"` → [`ConservativeMeanFlux`](@ref).
"""
function flux_temporal_scheme(name::AbstractString)
    s = lowercase(strip(name))
    s in ("stepwise", "block", "piecewise_constant") && return StepwiseFlux()
    s in ("linear", "interp", "interpolate") && return LinearInterpFlux()
    s in ("conservative", "window_mean", "integral", "mass_conserving") &&
        return ConservativeMeanFlux()
    throw(ArgumentError("unknown surface-flux temporal_scheme \"$(name)\"; " *
        "expected \"stepwise\", \"linear\", or \"conservative\""))
end

# =========================================================================
# TimeVaryingSurfaceFluxSource — 3-hourly (or any cadence) emission series
# =========================================================================

"""
    TimeVaryingSurfaceFluxSource{RateT, T}

A single-tracer surface source whose per-cell rate advances through a
time series of slices with the simulation clock. Used to carry the
CAMS/LMDZ 3-hourly diurnal cycle (instead of collapsing it to a monthly
mean) and to match GeosChem.

- `tracer_name :: Symbol` — matches a name in `CellState.tracer_names`.
- `cell_mass_rate_series :: RateT` — a stack of static rate slices, in
  the same per-cell model-storage units as [`SurfaceFluxSource`](@ref).
  For cubed-sphere this is an `NTuple{6}` of `(Nc, Nc, ntime)` arrays
  (one stacked panel per face). The trailing dimension is time.
- `times :: T` — slice times in **seconds since run start**, length
  `ntime`, sorted strictly ascending. Stays a **host** `Vector` even
  after Adapt-to-device so the operator can `searchsortedlast` it on the
  host before launching the interpolated kernel.
- `scheme :: S` — an [`AbstractFluxTemporalScheme`](@ref) selecting how the
  stored slices are reconstructed onto the simulation clock
  ([`StepwiseFlux`](@ref), [`LinearInterpFlux`](@ref), or
  [`ConservativeMeanFlux`](@ref)). A singleton, carried through Adapt
  unchanged.

The operator resolves `(i0, i1, w0, w1) = _flux_temporal_weights(scheme,
times, current_time(meteo), dt)` on the host, then blends the two bracketing
slices in the kernel. End slices are clamped (constant extrapolation) outside
`[times[1], times[end]]`.
"""
struct TimeVaryingSurfaceFluxSource{RateT, T <: AbstractVector{<:Real},
                                    S <: AbstractFluxTemporalScheme,
                                    CompT} <: AbstractSurfaceFluxSource
    tracer_name           :: Symbol
    cell_mass_rate_series :: RateT
    times                 :: T
    scheme                :: S
    compensation          :: CompT   # Kahan compensation; shape = one time-slice of series
end

# Outer constructors: allocate zero compensation from the series shape.
function TimeVaryingSurfaceFluxSource(name::Symbol, series::RateT,
                                      times::T, scheme::S) where {RateT, T, S}
    comp = _alloc_flux_comp_from_series(series)
    return TimeVaryingSurfaceFluxSource{RateT, T, S, typeof(comp)}(
        name, series, times, scheme, comp)
end

TimeVaryingSurfaceFluxSource(name::Symbol, series, times) =
    TimeVaryingSurfaceFluxSource(name, series, times, LinearInterpFlux())

# Adapt hook: carry series + compensation to device; keep times + scheme on host.
function Adapt.adapt_structure(to, source::TimeVaryingSurfaceFluxSource)
    series       = Adapt.adapt(to, source.cell_mass_rate_series)
    compensation = Adapt.adapt(to, source.compensation)
    return TimeVaryingSurfaceFluxSource{typeof(series), typeof(source.times),
                                        typeof(source.scheme), typeof(compensation)}(
        source.tracer_name, series, source.times, source.scheme, compensation)
end

function _check_surface_source_compatibility(state::CubedSphereState,
                                             source::TimeVaryingSurfaceFluxSource)
    tracer_index(state, source.tracer_name) === nothing &&
        throw(ArgumentError("surface source tracer $(source.tracer_name) is not present in model state"))

    series = source.cell_mass_rate_series
    series isa NTuple{6} || throw(ArgumentError(
        "cubed-sphere time-varying surface source $(source.tracer_name) must provide an NTuple{6} " *
        "of panel series, got $(typeof(series))"))

    ntime = length(source.times)
    issorted(source.times) || throw(ArgumentError(
        "time-varying surface source $(source.tracer_name) requires ascending `times`"))

    Hp = state.halo_width
    @inbounds for p in 1:6
        panel = state.air_mass[p]
        expected = (size(panel, 1) - 2Hp, size(panel, 2) - 2Hp, ntime)
        size(series[p]) == expected || throw(ArgumentError(
            "cubed-sphere time-varying surface source $(source.tracer_name) panel $p has shape " *
            "$(size(series[p])) but the expected interior series shape is $(expected)"))
    end
    return nothing
end

"""
    _time_interp_bracket(times, t) -> (i0, i1, w0, w1)

Return the two bracketing slice indices `i0 ≤ i1` and the linear
interpolation weights `w0, w1` (with `w0 + w1 == 1`) for time `t`
against the sorted `times` vector. Clamps to the end slices (constant
extrapolation) when `t` is outside `[times[1], times[end]]`:

- `t ≤ times[1]`   → `(1, 1, 1, 0)`
- `t ≥ times[end]` → `(end, end, 1, 0)`
- otherwise        → `(k, k+1, 1-frac, frac)` where `k =
  searchsortedlast(times, t)` and `frac` is the position of `t` in
  `[times[k], times[k+1]]`.
"""
@inline function _time_interp_bracket(times::AbstractVector, t::Real)
    n = length(times)
    n == 0 && throw(ArgumentError("_time_interp_bracket: empty times vector"))
    n == 1 && return (1, 1, 1.0, 0.0)
    t <= times[1]   && return (1, 1, 1.0, 0.0)
    t >= times[end] && return (n, n, 1.0, 0.0)
    k = searchsortedlast(times, t)          # times[k] <= t < times[k+1]
    t0 = Float64(times[k]); t1 = Float64(times[k + 1])
    span = t1 - t0
    frac = span > 0 ? (Float64(t) - t0) / span : 0.0
    return (k, k + 1, 1.0 - frac, frac)
end

# =========================================================================
# Surface-slice helpers — internal, used by the kernel shell and
# by the legacy `DrivenSimulation._apply_surface_sources!`.
# =========================================================================

"""
    _surface_shape(rm) -> Tuple

Return the expected shape of a surface source's `cell_mass_rate` for the
given tracer mass array `rm`. For 3D structured `(Nx, Ny, Nz)` tracers,
this is `(Nx, Ny)`; for 2D face-indexed `(Nc, Nz)` tracers, `(Nc,)`.
"""
@inline _surface_shape(rm::AbstractArray{<:Any, 3}) = (size(rm, 1), size(rm, 2))
@inline _surface_shape(rm::AbstractArray{<:Any, 2}) = (size(rm, 1),)

"""
    _check_surface_source_compatibility(state, source)

Validate that `source.tracer_name` is present in `state`, and that
`size(source.cell_mass_rate)` matches the state's surface slice shape.
Throws `ArgumentError` on mismatch. Used at DrivenSimulation construction
and at SurfaceFluxOperator construction.
"""
function _check_surface_source_compatibility(state, source::SurfaceFluxSource)
    tracer_index(state, source.tracer_name) === nothing &&
        throw(ArgumentError("surface source tracer $(source.tracer_name) is not present in model state"))
    rm = get_tracer_raw(state, source.tracer_name)
    size(source.cell_mass_rate) == _surface_shape(rm) ||
        throw(ArgumentError("surface source $(source.tracer_name) has shape $(size(source.cell_mass_rate)) but tracer surface shape is $(_surface_shape(rm))"))
    return nothing
end

function _check_surface_source_compatibility(state::CubedSphereState, source::SurfaceFluxSource)
    tracer_index(state, source.tracer_name) === nothing &&
        throw(ArgumentError("surface source tracer $(source.tracer_name) is not present in model state"))

    rates = source.cell_mass_rate
    rates isa NTuple{6} || throw(ArgumentError(
        "cubed-sphere surface source $(source.tracer_name) must provide an NTuple{6} " *
        "of panel rates, got $(typeof(rates))"))

    Hp = state.halo_width
    @inbounds for p in 1:6
        panel = state.air_mass[p]
        expected = (size(panel, 1) - 2Hp, size(panel, 2) - 2Hp)
        size(rates[p]) == expected || throw(ArgumentError(
            "cubed-sphere surface source $(source.tracer_name) panel $p has shape $(size(rates[p])) " *
            "but the interior panel shape is $(expected)"))
    end
    return nothing
end

"""
    _apply_surface_source!(rm, source, dt)

Add `source.cell_mass_rate × dt` to the surface slice of the tracer mass
array `rm`. The surface slice is `rm[:, :, Nz]` for 3D tracers and
`rm[:, Nz]` for 2D tracers — the `k = Nz = surface` convention
established by the LatLon storage layout.

Broadcasts over all surface cells; fused `.+=` is allocation-free and
GPU-friendly (KernelAbstractions dispatches to the backend of `rm`).

This is the legacy application path used by
`DrivenSimulation._apply_surface_sources!`. A unified KA-kernel version
backs the `SurfaceFluxOperator.apply!` path.
"""
function _apply_surface_source!(rm::AbstractArray{FT, 3},
                                source::SurfaceFluxSource, dt) where FT
    Nz   = size(rm, 3)
    surf = @view rm[:, :, Nz]
    comp = source.compensation
    x    = source.cell_mass_rate .* FT(dt)
    y    = x .- comp
    t    = surf .+ y
    comp .= (t .- surf) .- y
    surf .= t
    return nothing
end

function _apply_surface_source!(rm::AbstractArray{FT, 2},
                                source::SurfaceFluxSource, dt) where FT
    Nz   = size(rm, 2)
    surf = @view rm[:, Nz]
    comp = source.compensation
    x    = source.cell_mass_rate .* FT(dt)
    y    = x .- comp
    t    = surf .+ y
    comp .= (t .- surf) .- y
    surf .= t
    return nothing
end

function _apply_surface_source!(rm::NTuple{6}, source::SurfaceFluxSource, dt;
                                halo_width::Integer)
    Hp    = Int(halo_width)
    rates = source.cell_mass_rate
    comps = source.compensation
    rates isa NTuple{6} || throw(ArgumentError(
        "cubed-sphere surface source $(source.tracer_name) must provide NTuple{6} panel rates"))
    @inbounds for p in 1:6
        panel_rm = rm[p]
        Nz   = size(panel_rm, 3)
        Nc   = size(panel_rm, 1) - 2Hp
        surf = @view panel_rm[Hp + 1:Hp + Nc, Hp + 1:Hp + Nc, Nz]
        comp = comps[p]
        x    = rates[p] .* dt
        y    = x .- comp
        t    = surf .+ y
        comp .= (t .- surf) .- y
        surf .= t
    end
    return nothing
end
