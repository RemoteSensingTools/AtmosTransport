# ===========================================================================
# Plan 41 P1 — typed Axis-3 (Target topology) window-contract surface.
#
# `AbstractWindowContract{G, FT}`, `AbstractWindowWorkspace{G, FT}`, and
# `AbstractBinaryWriter{G, FT, Basis<:AbstractMassBasis}` are the typed
# nominals that close the kwarg-drift / mass-basis-runtime-check
# foot-guns (A) and (C) from `docs/plans/41_UNIFIED_PREPROCESSOR/DESIGN.md`.
#
# What this file defines (additive only; no behavior changes):
#
#   - `AbstractWindowContract{G, FT}` — typed nominal owning a
#       topology's per-window gate policy (replay tolerance, positivity
#       CFL limit, `require_substep_positivity` toggle) plus a worst-
#       window accumulator. The trait surface is `verify_window!`,
#       `update_accumulator!`, `summarize_status!`.
#
#   - `AbstractWindowWorkspace{G, FT}` — typed nominal for the per-day
#       target-shape buffers. P1 ships only the abstract type; concrete
#       subtypes land alongside the unified-driver cutover in P2.
#
#   - `AbstractBinaryWriter{G, FT, Basis}` — typed nominal for the
#       topology's streaming binary writer. Same: abstract only in P1;
#       concretes land in P2. The third type parameter is a subtype of
#       the existing `State.AbstractMassBasis` (`DryBasis`/`MoistBasis`)
#       so a writer↔reader pairing mismatch is a compile-time
#       `MethodError` rather than a post-load runtime header check —
#       closing foot-gun (C).
#
# The concrete per-topology contracts ship in:
#   * `cubed_sphere_contracts.jl`         (CubedSphereContract{FT})
#   * `latlon_contracts.jl`               (LatLonContract{FT})
#   * `reduced_gaussian_contracts.jl`     (ReducedGaussianContract{FT})
#
# Mass basis: this file deliberately does NOT redefine `AbstractMassBasis`
# / `DryBasis` / `MoistBasis`. Those tags live in `src/State/Basis.jl`
# and are already shared by `CellState`, `FluxState`, `TransportBinaryDriver`,
# and the ERA5 dry-flux builder. Preprocessing reuses them so the typed
# `AbstractBinaryWriter{G, FT, Basis}` parameter is exactly the same
# nominal the runtime reader paths dispatch on — there is no "preprocessor
# basis" vs "runtime basis" parallel hierarchy.
# ===========================================================================

"""
    mass_basis_symbol(::AbstractMassBasis) -> Symbol

Map a basis singleton (`State.DryBasis` / `State.MoistBasis`) to the
on-disk binary-header value (`:dry` / `:moist`). Inverse of
`mass_basis_from_symbol`.
"""
@inline mass_basis_symbol(::DryBasis)   = :dry
@inline mass_basis_symbol(::MoistBasis) = :moist

"""
    mass_basis_from_symbol(s::Symbol) -> AbstractMassBasis

Construct the matching basis singleton from a header `Symbol`. Throws
`ArgumentError` for unknown values.
"""
@inline function mass_basis_from_symbol(s::Symbol)
    s === :dry   && return DryBasis()
    s === :moist && return MoistBasis()
    throw(ArgumentError("Unknown mass-basis symbol $(s); expected :dry or :moist."))
end

# ---------------------------------------------------------------------------
# Abstract trait surface. Every topology that produces a transport-binary
# must register a concrete subtype of each of the three abstracts below
# (the LL/RG/CS triple is registered in this file's siblings).
# ---------------------------------------------------------------------------

"""
    AbstractWindowContract{G <: AbstractTargetGeometry, FT}

Typed nominal owning a topology's per-window gate policy and the
worst-window accumulator state. Concrete subtypes:

  - `CubedSphereContract{FT}`        (`cubed_sphere_contracts.jl`)
  - `LatLonContract{FT}`             (`latlon_contracts.jl`)
  - `ReducedGaussianContract{FT}`    (`reduced_gaussian_contracts.jl`)

A concrete contract validates its policy at construction time (so e.g.
`positivity_cfl_limit = 0.0` errors *before* any window is preprocessed)
and exposes the four-method trait surface:

```julia
verify_window!(window, contract, win_idx) -> (replay, positivity)
update_accumulator!(contract, positivity_diag, win_idx) -> nothing
summarize_status!(contract; quarantine_path) -> nothing
```

`window` is the topology-specific window payload (NamedTuple of typed
buffers today, P2-typed `ReadyWindow{G, FT}` later).

Closes foot-gun (A) from DESIGN.md: contract knobs aren't drift-prone
kwargs anymore — each topology constructs its own contract once from
config, with whatever fields IT needs.
"""
abstract type AbstractWindowContract{G <: AbstractTargetGeometry, FT} end

"""
    AbstractWindowWorkspace{G <: AbstractTargetGeometry, FT}

Typed nominal for the per-day target-shape workspace buffers. P1 ships
only the abstract type; concrete subtypes land alongside the unified
driver cutover in P2 (today's workspaces are NamedTuples constructed
inside each topology's `process_day` orchestrator).
"""
abstract type AbstractWindowWorkspace{G <: AbstractTargetGeometry, FT} end

"""
    AbstractBinaryWriter{G <: AbstractTargetGeometry, FT,
                         Basis <: AbstractMassBasis}

Typed nominal for the topology's streaming binary writer. The third
type parameter encodes the on-disk mass-basis convention (reusing
`State.AbstractMassBasis` so the same nominal flows through the runtime
reader path) so a writer↔reader pairing mismatch is a compile-time
`MethodError`.

P1 ships only the abstract type; concrete subtypes land in P2.
"""
abstract type AbstractBinaryWriter{G <: AbstractTargetGeometry, FT,
                                    Basis <: AbstractMassBasis} end

# ---------------------------------------------------------------------------
# P2b readiness/workspace skeleton.
#
# These are additive nominals for the unified driver cutover. Existing
# preprocessors still own their control flow, but they can now hand a typed
# `ReadyWindow{G, FT}` to the same `verify_window!` contract surface the P1
# NamedTuple payloads used. `ReadyWindow` deliberately forwards unknown
# property access to its payload so the P1 contract methods do not need
# duplicate overloads while the drivers migrate.
# ---------------------------------------------------------------------------

"""
    ReadyWindow{G, FT}(index, payload::NamedTuple)

Typed wrapper for a topology-specific window payload that is ready to be
verified and written. `G` is the target geometry, `FT` is the on-disk float
type, and `payload` is the existing topology NamedTuple (`m_cur`/`am`/... for
LL/CS, `m_cur`/`hflux`/... for RG).

Unknown property access is forwarded to `payload`, so existing
`verify_window!(window, contract, win_idx)` methods can accept either a raw
NamedTuple or a `ReadyWindow`.
"""
struct ReadyWindow{G <: AbstractTargetGeometry, FT, P <: NamedTuple}
    index   :: Int
    payload :: P

    function ReadyWindow{G, FT, P}(index::Integer, payload::P) where
            {G <: AbstractTargetGeometry, FT, P <: NamedTuple}
        index >= 1 || throw(ArgumentError("ReadyWindow: index = $(index); must be >= 1."))
        return new{G, FT, P}(Int(index), payload)
    end
end

ReadyWindow{G, FT}(index::Integer, payload::P) where
    {G <: AbstractTargetGeometry, FT, P <: NamedTuple} =
        ReadyWindow{G, FT, P}(index, payload)

@inline function Base.getproperty(window::ReadyWindow, name::Symbol)
    if name === :index || name === :payload
        return getfield(window, name)
    end
    return getproperty(getfield(window, :payload), name)
end

function Base.propertynames(window::ReadyWindow, private::Bool = false)
    payload_names = propertynames(getfield(window, :payload), private)
    return private ? (:index, :payload, payload_names...) :
                     (:index, payload_names...)
end

"""
    PreprocessorRunCache{G, FT}()

Small typed cache for artifacts that should be built once per preprocessing run
instead of once per day/window (for example spectral LL->CS regridders or RG
compressed Laplacians). P2b only introduces the nominal and storage; concrete
drivers decide which keys they own as they migrate.
"""
mutable struct PreprocessorRunCache{G <: AbstractTargetGeometry, FT}
    entries :: Dict{Symbol, Any}
end

PreprocessorRunCache{G, FT}() where {G <: AbstractTargetGeometry, FT} =
    PreprocessorRunCache{G, FT}(Dict{Symbol, Any}())

PreprocessorRunCache(::Type{G}, ::Type{FT}) where
    {G <: AbstractTargetGeometry, FT} = PreprocessorRunCache{G, FT}()

Base.haskey(cache::PreprocessorRunCache, key::Symbol) = haskey(cache.entries, key)
Base.getindex(cache::PreprocessorRunCache, key::Symbol) = cache.entries[key]
Base.get(cache::PreprocessorRunCache, key::Symbol, default) = get(cache.entries, key, default)
Base.setindex!(cache::PreprocessorRunCache, value, key::Symbol) =
    setindex!(cache.entries, value, key)

# Generic trait functions. The four below are the canonical contract
# surface that every concrete `AbstractWindowContract{G, FT}` registers
# a method on. Declared here (as bare generic functions) so the per-
# topology contract files only need to add methods.

"""
    verify_window!(window, contract::AbstractWindowContract, win_idx::Int)
        -> (; replay, positivity)

Run the contract's replay and positivity gates for one window. The
replay gate throws on violation; the positivity gate is non-fatal at
this layer (the run-level accumulator + `summarize_status!` decides
fatal-vs-warn based on `require_substep_positivity`).

`window` is the topology's per-window payload (a NamedTuple in P1, a
typed `ReadyWindow{G, FT}` in P2).
"""
function verify_window! end

"""
    update_accumulator!(contract::AbstractWindowContract,
                         positivity_diag, win_idx::Int) -> nothing

Fold one window's positivity diagnostic into the contract's worst-
window accumulator. Concrete contracts mutate their internal state and
return `nothing`. Idempotent if `positivity_diag.ratio` does not exceed
the current worst.
"""
function update_accumulator! end

"""
    summarize_status!(contract::AbstractWindowContract;
                       quarantine_path::Union{Nothing, AbstractString} = nothing)
        -> nothing

Post-loop summary helper. Logs the worst-window outcome; if the
contract's policy demands and the worst exceeds the gate, deletes
`quarantine_path` (if given) and errors. Otherwise warns when the
gate is violated but the policy is set to record-and-continue.
"""
function summarize_status! end

# Optional accessors that concrete contracts may override. The defaults
# match `CubedSphereContract`'s current behavior so siblings can share
# them when their policy matches; topology-specific contracts that need
# different defaults register their own method.

"""
    contract_replay_tolerance(contract::AbstractWindowContract) -> Float64

Relative replay tolerance the contract's `verify_window!` uses.
Concrete contracts return their stored policy value.
"""
function contract_replay_tolerance end

"""
    contract_cfl_limit(contract::AbstractWindowContract) -> Float64

Per-substep positivity CFL gate the contract enforces.
"""
function contract_cfl_limit end

"""
    contract_require_positivity(contract::AbstractWindowContract) -> Bool

Whether `summarize_status!` errors (`true`) or warns (`false`) on a
positivity violation. Closes-the-escape-hatch toggle from CS round-2.
"""
function contract_require_positivity end

"""
    allocate_window_workspace(grid, settings, vertical, ::Type{FT}; cache)

Construct the topology-specific `AbstractWindowWorkspace{G, FT}` for one
preprocessing day. Concrete methods land as production drivers migrate.
"""
function allocate_window_workspace end

"""
    reset_workspace!(workspace, day_state) -> workspace

Reset a reusable workspace before ingesting a new day/source stream.
"""
function reset_workspace! end

"""
    ingest_window!(workspace, raw_window, reader, vertical, contract; cache) -> nothing

Consume one source/met window into the topology workspace. Ready windows are
exposed by `drain_ready_windows!`.
"""
function ingest_window! end

"""
    drain_ready_windows!(workspace) -> iterator of `ReadyWindow`

Return the windows that became write-ready after the last ingest.
"""
function drain_ready_windows! end

"""
    flush_final_windows!(workspace, reader, vertical, contract; cache)

Emit any final cross-day/zero-tendency windows once the source stream is
exhausted.
"""
function flush_final_windows! end

# ---------------------------------------------------------------------------
# Shared parameter-validation helpers used at the entry of every exported
# replay / positivity / window-contract wrapper.
#
# Codex review round 3 of `f5224a6` flagged that the `verify_*_window_contract!`
# direct surfaces accepted `replay_tol = Inf` / `NaN` and
# `positivity_cfl_limit = 0.0` / `>1` without the constructor guards —
# `Inf` would silently disable replay and `NaN` would fail every window
# late. The earlier round-2 fix for `boundary_stub_tol` followed the
# same pattern. These helpers route every exported wrapper through one
# place so a future contract knob inherits the validation automatically.
#
# Caller passes its own name string so the error attributes the
# violation to the surface the user actually called (helper vs wrapper
# vs constructor). The helpers are `@inline` so the validation is free
# in the compiled code.
# ---------------------------------------------------------------------------

@inline function _validate_replay_tol(replay_tol::Real, caller::AbstractString)
    isfinite(replay_tol) && replay_tol > 0 ||
        error("$(caller): replay_tol = $(replay_tol); must be finite and " *
              "> 0 (Inf silently disables replay; NaN would fail every " *
              "window late).")
    return nothing
end

@inline function _validate_cfl_limit(cfl_limit::Real, caller::AbstractString)
    isfinite(cfl_limit) && 0 < cfl_limit ≤ 1 ||
        error("$(caller): cfl_limit = $(cfl_limit); must be finite and " *
              "in (0, 1] (Inf/NaN silently disable the gate).")
    return nothing
end
