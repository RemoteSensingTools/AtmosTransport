"""
    AbstractSurfaceFluxOperator

Top of the surface emission operator hierarchy. Concrete subtypes in
plan 17 Commit 3:

- [`NoSurfaceFlux`](@ref) — identity (default).
- [`SurfaceFluxOperator`](@ref) — wraps a
  [`PerTracerFluxMap`](@ref) and applies each tracer's surface flux
  to `k = Nz` of the tracer mass array.

Every concrete subtype implements two entry points:

- State-level `apply!(state, meteo, grid, op, dt; workspace)` →
  mutates `state.tracers_raw`.
- Array-level `apply_surface_flux!(q_raw, op, ws, dt, meteo, grid;
  tracer_names)` → mutates a raw 4D tracer buffer directly. Used by
  the palindrome integration (Commit 5) on whichever ping-pong
  buffer currently holds the tracer state.
"""
abstract type AbstractSurfaceFluxOperator end

"""
    NoSurfaceFlux()

Identity operator — `apply!` is a no-op. Default for configurations
without surface emissions, and the value `strang_split_mt!` sees when
the palindrome's S position is unoccupied (Commit 5).

`NoSurfaceFlux`'s `apply!` is literally `= state` (and the array-level
`apply_surface_flux!` is `= nothing`). Julia's dispatch turns the call
site into a dead branch — zero floating-point work, bit-exact
backward compatibility.
"""
struct NoSurfaceFlux <: AbstractSurfaceFluxOperator end

"""
    SurfaceFluxOperator(flux_map::PerTracerFluxMap)
    SurfaceFluxOperator(sources::SurfaceFluxSource...)

Applies a `PerTracerFluxMap` of surface sources to the `k = Nz` slab
of the tracer mass array.

For every tracer named in the flux map, the operator launches the
`_surface_flux_kernel!` kernel over `(Nx, Ny)` and adds
`rate[i, j] * dt` to `tracers_raw[i, j, Nz, tracer_idx]`, where
`tracer_idx` is resolved on the host from `state.tracer_names`.
Tracers absent from the map are untouched.

Per plan 17 Decision 1, the rate is in kg/s per cell (already area-
integrated); no cell-area multiplier appears in the kernel. This
preserves the pre-17 `_apply_surface_source!` semantics and matches
the way legacy `SurfaceFluxSource` callers supply their arrays.

# `apply!` contract

    apply!(state, meteo, grid, op::SurfaceFluxOperator, dt; workspace)

- Walks `op.flux_map` in storage order.
- For each source, skips tracers not present in `state`.
- Dispatches to the array-level entry point with `tracer_names`
  pulled from the state.
- Synchronises the backend once at the end.

# Fields

- `flux_map :: M` — a [`PerTracerFluxMap`](@ref) of
  `SurfaceFluxSource`s. Emitting tracers are exactly those named in
  the map.
"""
struct SurfaceFluxOperator{M <: PerTracerFluxMap} <: AbstractSurfaceFluxOperator
    flux_map :: M

    SurfaceFluxOperator(flux_map::M) where {M <: PerTracerFluxMap} = new{M}(flux_map)
end

"""
    SurfaceFluxOperator(sources::SurfaceFluxSource...)

Convenience constructor: wraps a variadic list of sources in a
`PerTracerFluxMap` first. Empty variadic list is allowed (produces an
empty-map operator; equivalent to `NoSurfaceFlux` except type-
distinguishable).
"""
SurfaceFluxOperator(sources::SurfaceFluxSource...) =
    SurfaceFluxOperator(PerTracerFluxMap(sources...))

# =========================================================================
# Array-level entry point (for the palindrome, Commit 5)
# =========================================================================

"""
    apply_surface_flux!(q_raw, op::AbstractSurfaceFluxOperator, workspace, dt, meteo, grid;
                        tracer_names)

Array-level surface-flux application. Writes directly to the supplied
4D tracer buffer `q_raw` of shape `(Nx, Ny, Nz, Nt)`, adding each
source's `rate × dt` contribution to the surface slab
`q_raw[:, :, Nz, tracer_idx]`.

`tracer_names::NTuple{Nt, Symbol}` is required as a keyword so the
function can resolve each source's name to a slab index without
reaching back into the caller's `CellState`. This lets the palindrome
integration (Commit 5) point the operator at either the caller's
`state.tracers_raw` or the workspace's ping-pong buffer — whichever
currently holds the post-Z-sweep tracer state.

`workspace`, `meteo`, and `grid` are accepted but currently unused;
they are in the signature to match the operator-interface convention
(§"Workflow: Adding a new physics operator" in CLAUDE.md) and to leave
room for future extensions (e.g. meteorology-dependent emissions).

Returns `nothing` on success. For `NoSurfaceFlux`, returns `nothing`
immediately (zero floating-point work).
"""
function apply_surface_flux! end

# NoSurfaceFlux dead branch — dispatched away on the default config
apply_surface_flux!(::AbstractArray{<:Any, 4}, ::NoSurfaceFlux, ws, dt,
                     meteo, grid; tracer_names) = nothing

function apply_surface_flux!(q_raw::AbstractArray{FT, 4},
                             op::SurfaceFluxOperator,
                             workspace,
                             dt::Real,
                             meteo,
                             grid;
                             tracer_names::Tuple) where FT
    Nx, Ny, Nz, _ = size(q_raw)
    backend = get_backend(q_raw)
    dt_FT   = FT(dt)

    for src in op.flux_map.sources
        t_idx = findfirst(==(src.tracer_name), tracer_names)
        t_idx === nothing && continue   # tracer not in this state; skip
        kernel = _surface_flux_kernel!(backend, (16, 16))
        kernel(q_raw, src.cell_mass_rate, dt_FT, t_idx, Nz;
               ndrange = (Nx, Ny))
    end

    synchronize(backend)
    return nothing
end

# =========================================================================
# State-level entry point
# =========================================================================

# NoSurfaceFlux: identity
function apply!(state::CellState, meteo, grid, ::NoSurfaceFlux, dt;
                workspace = nothing)
    return state
end

function apply!(state::CellState, meteo, grid, op::SurfaceFluxOperator, dt;
                workspace = nothing)
    apply_surface_flux!(state.tracers_raw, op, workspace, dt, meteo, grid;
                        tracer_names = state.tracer_names)
    return state
end

# =========================================================================
# Helpers
# =========================================================================

"""
    emitting_tracer_indices(op::SurfaceFluxOperator, state::CellState) -> NTuple

Ordered tuple of tracer indices in `state.tracer_names` for the
tracers present in `op.flux_map`. Tracers in the map but missing
from the state are skipped (returned as `nothing` slots). Useful for
testing / introspection and unchanged by `apply!`.
"""
function emitting_tracer_indices(op::SurfaceFluxOperator, state::CellState)
    return map(src -> findfirst(==(src.tracer_name), state.tracer_names),
               op.flux_map.sources)
end

emitting_tracer_indices(::NoSurfaceFlux, ::CellState) = ()

# =========================================================================
# Adapt — GPU dispatch
# =========================================================================

Adapt.adapt_structure(to, op::NoSurfaceFlux) = op   # bits-stable singleton

Adapt.adapt_structure(to, op::SurfaceFluxOperator) =
    SurfaceFluxOperator(Adapt.adapt(to, op.flux_map))
