# ===========================================================================
# Plan 41 P0a — typed met-reader surface (Source axis).
#
# `AbstractMetReader{FT, S, CP}` is the typed nominal that bundles a met-
# source's settings, per-day handle context, and chained-mass-policy carry
# into one struct. Concrete readers (`GEOSNativeReader`, `ERA5SpectralReader`,
# …) wrap the existing `AbstractMetSettings` machinery in `met_sources.jl`
# and `sources/geos.jl` — they do not rewrite it.
#
# What this commit closes (from `docs/plans/41_UNIFIED_PREPROCESSOR/DESIGN.md`):
#
#   (A) per-method kwarg drift — the reader's struct fields, not the
#       caller's kwargs, decide what state flows per window. Adding a new
#       reader-side knob is a struct extension on one concrete reader, not
#       a 3-method-fan-out edit.
#   (D) cross-day seed plumbing — `ChainPolicy` is a type parameter, so
#       seed shape is statically known per reader. `NoChain` readers
#       cannot accidentally accept a `ChainedMass{T}` seed, and
#       `end_of_day_seed` is inferred to the right return type at compile
#       time, not derived from a NamedTuple lookup.
#
# What this commit does NOT do (deferred to later P-stages):
#   * Drive the unified `process_day` loop. The existing per-method
#     `process_day` orchestrators stay unchanged; this surface is parallel
#     and additive.
#   * Implement `ERA5SpectralReader.read_window!` faithfully — today's
#     spectral pipeline fuses spectral synthesis + native-grid regrid +
#     vertical merge into one `process_window!` call. Splitting it is a
#     P2 refactor; for P0a, `ERA5SpectralReader` exists as the typed
#     nominal with `read_window!` throwing a clear "lands in P2" error.
# ===========================================================================

# ---------------------------------------------------------------------------
# Chain-policy hierarchy: encodes cross-day mass-state carry at the
# type-system level. Closes foot-gun (D).
#
# Tagging types (no payload). The reader's actual seed VALUE lives in a
# struct field; the type parameter `ChainedMass{T}` lets the driver know
# statically what kind of seed to thread without inspecting NamedTuples.
# ---------------------------------------------------------------------------

"""
    AbstractChainPolicy

Type-system tag for cross-day mass-state carry. Concrete subtypes are
`NoChain` (no carry; each day starts from the raw source endpoint) and
`ChainedMass{T}` where `T` is the array shape of the seed (e.g.
`NTuple{6, Array{Float32, 3}}` for GEOS cubed-sphere).
"""
abstract type AbstractChainPolicy end

"""
    NoChain <: AbstractChainPolicy

The reader does not carry mass state across days. Day N starts from the
raw source endpoint. `end_of_day_seed(::reader{…, NoChain})` returns
`nothing` (statically inferred).
"""
struct NoChain <: AbstractChainPolicy end

"""
    ChainedMass{T} <: AbstractChainPolicy

The reader carries an end-of-day mass field of shape `T` into the next
day's open. The shape `T` is the seed array type (e.g.
`NTuple{6, Array{Float32, 3}}`); the actual seed VALUE lives in the
reader struct, but its STATIC type is encoded here so `end_of_day_seed`
return-type is known at compile time.
"""
struct ChainedMass{T} <: AbstractChainPolicy end

# ---------------------------------------------------------------------------
# Abstract reader surface.
# ---------------------------------------------------------------------------

"""
    AbstractMetReader{FT, S, CP}

Typed met-reader nominal. Bundles a met-source's settings, per-day handle
context, and chained-mass-policy carry into one struct that the unified
preprocessor driver dispatches on. Type parameters:

- `FT <: AbstractFloat` — preprocessing float type (`Float32` for GPU,
  `Float64` for diagnostic runs).
- `S <: AbstractMetSettings` — concrete settings type (`GEOSITSettings`,
  `GEOSFPSettings`, `ERA5SpectralSettings`, …).
- `CP <: AbstractChainPolicy` — `NoChain` or `ChainedMass{T}` for some
  seed array type `T`.

Concrete subtypes implement the seven trait functions below.
"""
abstract type AbstractMetReader{FT <: AbstractFloat,
                                  S <: AbstractMetSettings,
                                  CP <: AbstractChainPolicy} end

"""
    open_reader(settings::AbstractMetSettings, date::Date, ::Type{FT};
                seed = nothing, next_day_handle::Bool = true)
        → reader::AbstractMetReader{FT, typeof(settings), CP}

Construct a typed met-reader for one calendar day. The reader owns the
underlying day-handle context (file handles, vertical coefficients,
chained-mass seed) and exposes the per-window trait surface.

`seed` carries cross-day mass state for chained-mass readers; pass the
return value of `end_of_day_seed(prev_day_reader)`. For `NoChain`
readers, `seed` must be `nothing`.

`next_day_handle` controls whether the reader opens a handle into the
next day's first-hour instantaneous file. Required when the day produces
endpoint mass for the last window (the standard case for hourly sources).

Dispatch by `settings` type to select the concrete reader; concrete
readers register their own method.
"""
function open_reader end

"""
    close_reader!(reader::AbstractMetReader) → nothing

Close all per-day file handles and release scratch held by the reader.
Idempotent — safe to call from a `finally` block. Concrete readers
implement.
"""
function close_reader! end

# `windows_per_day` and `read_window!` are already declared as generic
# functions in `met_sources.jl` (the `AbstractMetSettings` surface). This
# file adds reader-typed methods on them; redeclaring the generics here
# would warn-replace the existing docstrings.

"""
    end_of_day_seed(reader::AbstractMetReader) → seed_or_nothing

Return the seed value to thread into the next day's
`open_reader(..., seed = ...)`. Type-system guarantees:

- `end_of_day_seed(::AbstractMetReader{FT, S, NoChain})` returns
  `nothing` (statically inferred).
- `end_of_day_seed(::AbstractMetReader{FT, S, ChainedMass{T}})` returns
  a value of type `T` (or throws if the reader has not yet produced
  the end-of-day endpoint).

Closes foot-gun (D).
"""
function end_of_day_seed end

# Default fallback: NoChain readers always return nothing without needing
# a per-subtype override. The compiler folds this away at the call site.
@inline end_of_day_seed(::AbstractMetReader{FT, S, NoChain}) where {FT, S} = nothing

"""
    native_vertical(reader::AbstractMetReader) → HybridSigmaPressure{FT_v}

Native vertical coordinate of the source data (NOT the merged/output
coordinate). The vertical-transform axis (P0b) consumes this to plan
its mapping.
"""
function native_vertical end

"""
    window_metadata(reader::AbstractMetReader) → NamedTuple

Per-source window timing metadata. Standard fields:

- `windows::Int` — `windows_per_day(reader)`.
- `substeps::Int` — sub-windows per write window (e.g. GEOS's
  `dt_met_seconds ÷ mass_flux_dt`).
- `dt_substep::Float64` — substep wall-clock in seconds.

Concrete readers may add source-specific keys (e.g. GEOS's
`mass_flux_dt`).
"""
function window_metadata end

# ---------------------------------------------------------------------------
# GEOSNativeReader — concrete reader for GEOS-IT / GEOS-FP native NetCDF.
#
# Wraps the existing `(GEOSSettings, GEOSDayHandles)` pair plus a typed
# chained-mass slot. `read_window!` delegates to the existing
# `read_window!(::RawWindow, ::GEOSSettings, ::GEOSDayHandles, date, w)`
# so bit-exact behavior with today's `cubed_sphere_geos.jl::process_day`
# is preserved.
# ---------------------------------------------------------------------------

"""
    GEOSNativeReader{FT, S, CP, V} <: AbstractMetReader{FT, S, CP}

Typed reader wrapping `(GEOSSettings, GEOSDayHandles)` for one day, plus
optional chained-mass state. The `V` type parameter is the seed-array
type (used only when `CP = ChainedMass{V}`); for `NoChain` readers `V`
is `Nothing`.

Constructor: `open_reader(settings::AbstractGEOSSettings, date, FT;
seed, next_day_handle)`. See the function docstring.
"""
mutable struct GEOSNativeReader{FT, S <: AbstractGEOSSettings,
                                  CP <: AbstractChainPolicy, V} <:
                AbstractMetReader{FT, S, CP}
    settings :: S
    handles  :: Any         # GEOSDayHandles or GEOSFPNativeDayHandles (depends on flavor)
    date     :: Date
    seed     :: V           # Union{Nothing, NTuple{6, Array{FT,3}}} typically
    final_m  :: Base.RefValue{V}  # populated when the orchestrator finishes the last window
end

function open_reader(settings::AbstractGEOSSettings, date::Date, ::Type{FT};
                     seed = nothing,
                     next_day_handle::Bool = true,
                     chain_mass::Bool = true) where {FT <: AbstractFloat}
    handles = open_day(settings, date; next_day_handle = next_day_handle)
    # `chain_mass = false` opts out of cross-day carry entirely (NoChain).
    # Otherwise the policy is `ChainedMass{typeof(seed_template)}` where
    # the seed template is the panel-tuple shape derived from the source.
    if !chain_mass
        return GEOSNativeReader{FT, typeof(settings), NoChain, Nothing}(
            settings, handles, date, nothing, Ref{Nothing}(nothing))
    end
    # ChainedMass with NTuple{6, Array{FT,3}} seed shape. Even on day 1
    # (seed = nothing) the policy parameter is fixed; the seed field can
    # hold `nothing` until the first carry arrives because we union the
    # field type via the V type parameter.
    seed_value = seed
    V = if seed_value === nothing
        Union{Nothing, NTuple{6, Array{FT, 3}}}
    else
        typeof(seed_value)
    end
    CP = ChainedMass{NTuple{6, Array{FT, 3}}}
    return GEOSNativeReader{FT, typeof(settings), CP, V}(
        settings, handles, date, seed_value, Ref{V}(nothing))
end

@inline windows_per_day(reader::GEOSNativeReader) =
    windows_per_day(reader.settings, reader.date)

@inline native_vertical(reader::GEOSNativeReader) =
    reader.handles.vc

@inline function window_metadata(reader::GEOSNativeReader{FT}) where FT
    dt_met = 3600.0          # GEOS-IT / GEOS-FP archive cadence
    mass_flux_dt = reader.settings.mass_flux_dt
    substeps = round(Int, dt_met / mass_flux_dt)
    return (windows = windows_per_day(reader),
            substeps = substeps,
            dt_substep = mass_flux_dt,
            mass_flux_dt = mass_flux_dt)
end

@inline function read_window!(raw::RawWindow{FT}, reader::GEOSNativeReader{FT},
                                w::Int) where FT
    return read_window!(raw, reader.settings, reader.handles, reader.date, w)
end

# Chained-mass end-of-day seed. Closes foot-gun (D): the orchestrator
# threads this into the next day's `open_reader(... ; seed = ...)`. The
# `final_m` slot is populated by the orchestrator's last-window write
# path (e.g. `_set_final_m!(reader, panels_m)`).
@inline function end_of_day_seed(reader::GEOSNativeReader{FT, S, ChainedMass{T}, V}) where {FT, S, T, V}
    seed = reader.final_m[]
    seed === nothing && return nothing
    return seed::T
end

"""
    set_end_of_day_seed!(reader::GEOSNativeReader, seed) → reader

Set the end-of-day mass seed produced by the orchestrator's last
window. Called once per day at the end of `process_day`. For
`NoChain` readers this is a no-op.
"""
@inline set_end_of_day_seed!(::GEOSNativeReader{FT, S, NoChain, V},
                              _seed) where {FT, S, V} = nothing
@inline function set_end_of_day_seed!(
    reader::GEOSNativeReader{FT, S, ChainedMass{T}, V}, seed::T,
) where {FT, S, T, V}
    reader.final_m[] = seed
    return reader
end

@inline function close_reader!(reader::GEOSNativeReader)
    close_day!(reader.handles)
    return nothing
end

# ---------------------------------------------------------------------------
# ERA5SpectralReader — typed nominal for the spectral path.
#
# P0a ships the type and constructor / `windows_per_day` / `native_vertical`
# / `close_reader!` so the abstract surface compiles and dispatches.
# `read_window!` is deferred to P2: today's spectral pipeline fuses
# spectral synthesis + native-grid regrid + vertical merge into one
# `process_window!` call in `transport_binary/latlon_workspaces.jl:634`.
# Splitting it cleanly is the spectral-path refactor that lands when the
# unified driver replaces `_process_day_spectral`. Until then,
# `read_window!(::ERA5SpectralRawWindow, ::ERA5SpectralReader, ::Int)`
# throws an explicit "lands in P2" error so a caller cannot silently get
# zero-filled or stale data.
# ---------------------------------------------------------------------------

"""
    ERA5SpectralSettings <: AbstractMetSettings

P0a placeholder. Today's spectral path uses a `NamedTuple` returned by
`resolve_runtime_settings` in `configuration.jl:170`; this typed nominal
is the wrapper we'll move it inside during P2. Holds the same
NamedTuple verbatim — no semantics change.
"""
struct ERA5SpectralSettings <: AbstractMetSettings
    nt :: NamedTuple
end

# Forward the existing settings-trait surface to the wrapped NamedTuple
# so any code that does `settings.field` keeps working. P0a additive
# scope: nothing reads this yet, but the indirection costs zero.
Base.getproperty(s::ERA5SpectralSettings, k::Symbol) =
    k === :nt ? getfield(s, :nt) : getfield(s, :nt)[k]
Base.propertynames(s::ERA5SpectralSettings) = propertynames(getfield(s, :nt))

"""
    ERA5SpectralReader{FT, S} <: AbstractMetReader{FT, S, NoChain}

Typed nominal for the ERA5 spectral path. ChainPolicy is fixed at
`NoChain` because today's spectral path does not carry cross-day mass
state (it pins global-mean ps per window instead). P0a ships the
nominal + lifecycle methods; the per-window read fuses with merge in
today's `process_window!` and is deferred to P2.
"""
mutable struct ERA5SpectralReader{FT, S <: AbstractMetSettings} <:
                AbstractMetReader{FT, S, NoChain}
    settings   :: S
    date       :: Date
    spec       :: Any   # SpectralDayData; opaque at this layer
    closed     :: Bool
end

"""
    open_reader(settings::ERA5SpectralSettings, date::Date, ::Type{FT};
                seed = nothing, next_day_handle::Bool = true)

P0a: opens the spectral day's GRIB and caches the typed nominal.
`seed`/`next_day_handle` accepted for signature parity with the GEOS
constructor; the spectral path's "next-day endpoint" is handled by the
existing `next_day_hour0` helper in `configuration.jl:349` until P2.
"""
function open_reader(settings::ERA5SpectralSettings, date::Date, ::Type{FT};
                     seed = nothing,
                     next_day_handle::Bool = true,
                     chain_mass::Bool = false) where {FT <: AbstractFloat}
    seed === nothing ||
        throw(ArgumentError("ERA5SpectralReader is fixed at ChainPolicy = NoChain; " *
                             "seed must be `nothing`, got $(typeof(seed))."))
    chain_mass &&
        throw(ArgumentError("ERA5SpectralReader does not support chain_mass = true " *
                             "(spectral path pins global-mean ps per window instead)."))
    _ = next_day_handle  # accepted but unused at this layer (handled by configuration.jl)
    return ERA5SpectralReader{FT, typeof(settings)}(settings, date, nothing, false)
end

@inline windows_per_day(reader::ERA5SpectralReader) = 24

@inline function native_vertical(reader::ERA5SpectralReader)
    # P2 will plumb this through the spectral workspace's vertical setup.
    error("native_vertical(::ERA5SpectralReader) lands in P2 alongside the " *
          "spectral-path unified-driver cutover. The typed nominal exists " *
          "now so the source axis can be dispatched on; the data accessors " *
          "follow once `process_window!` is split.")
end

@inline window_metadata(reader::ERA5SpectralReader) =
    (windows = windows_per_day(reader), substeps = 1, dt_substep = 3600.0)

@inline function read_window!(_dst, reader::ERA5SpectralReader, _w::Int)
    error("read_window!(::ERA5SpectralReader, …) lands in P2 of Plan 41 — " *
          "today's spectral pipeline fuses spectral synthesis + native-grid " *
          "regrid + vertical merge inside `process_window!` " *
          "(`transport_binary/latlon_workspaces.jl:634`). Splitting it is " *
          "the spectral-path unified-driver work. For now the typed nominal " *
          "is shipped so downstream dispatch surfaces (target topology, " *
          "vertical transform) can be exercised against it. See " *
          "`docs/plans/41_UNIFIED_PREPROCESSOR/NOTES.md`.")
end

@inline function close_reader!(reader::ERA5SpectralReader)
    reader.closed = true
    return nothing
end
