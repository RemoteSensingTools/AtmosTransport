"""
    SurfaceFluxSource{RateT}

A single-tracer surface source: a `tracer_name` plus a `cell_mass_rate`
array supplying mass added **per cell per second** to the surface layer.

- `tracer_name :: Symbol` — matches a name in `CellState.tracer_names`.
- `cell_mass_rate :: RateT` — either a 3D `(Nx, Ny, Nz)` array for
  structured grids (only the `k = Nz` surface slice is read) or a 2D
  `(Nc, Nz)` array for face-indexed grids (surface slice `[:, Nz]`).
  The **units are kg/s per cell** — already area-integrated. The surface
  flux kernel applies `rm_surface += rate × dt` without multiplying by
  cell area.

# Why kg/s per cell (not kg/m²/s)

Plan 17 Decision 1 (user-approved 2026-04-19): the prognostic tracer is
mass (kg per cell), so a per-cell rate × dt is the natural unit and
matches the legacy `DrivenSimulation._apply_surface_source!` signature
that shipped before plan 17. A per-area (kg/m²/s) variant that multiplies
by `cell_area` is deferred to a follow-up plan.

# History

Originally introduced in `src/Models/DrivenSimulation.jl` before plan 17.
Migrated to `src/Operators/SurfaceFlux/` in plan 17 Commit 2 so that
`SurfaceFluxOperator` (Commit 3) can consume it; the name is still
re-exported from `AtmosTransport` for backward compat with external
callers that imported it by fully-qualified name.

# Fields
- `tracer_name :: Symbol`
- `cell_mass_rate :: RateT` — backend-agnostic; `Adapt.adapt` converts
  the array between host and device transparently via
  [`Adapt.adapt_structure`](@ref).
"""
struct SurfaceFluxSource{RateT}
    tracer_name    :: Symbol
    cell_mass_rate :: RateT
end

# Adapt hook: carries the rate array to the device without disturbing
# the tracer name.
function Adapt.adapt_structure(to, source::SurfaceFluxSource)
    cell_mass_rate = Adapt.adapt(to, source.cell_mass_rate)
    return SurfaceFluxSource{typeof(cell_mass_rate)}(source.tracer_name,
                                                      cell_mass_rate)
end

# =========================================================================
# Surface-slice helpers — internal, used by the Commit 3 kernel shell and
# by the legacy `DrivenSimulation._apply_surface_sources!` until plan 17
# Commit 6 moves the call site into the palindrome.
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
and (in Commit 3) at SurfaceFluxOperator construction.
"""
function _check_surface_source_compatibility(state, source::SurfaceFluxSource)
    tracer_index(state, source.tracer_name) === nothing &&
        throw(ArgumentError("surface source tracer $(source.tracer_name) is not present in model state"))
    rm = get_tracer(state, source.tracer_name)
    size(source.cell_mass_rate) == _surface_shape(rm) ||
        throw(ArgumentError("surface source $(source.tracer_name) has shape $(size(source.cell_mass_rate)) but tracer surface shape is $(_surface_shape(rm))"))
    return nothing
end

"""
    _apply_surface_source!(rm, source, dt)

Add `source.cell_mass_rate × dt` to the surface slice of the tracer mass
array `rm`. The surface slice is `rm[:, :, Nz]` for 3D tracers and
`rm[:, Nz]` for 2D tracers — the `k = Nz = surface` convention
established by the LatLon storage layout (plan 17 Decision 2).

Broadcasts over all surface cells; fused `.+=` is allocation-free and
GPU-friendly (KernelAbstractions dispatches to the backend of `rm`).

This is the legacy application path used by
`DrivenSimulation._apply_surface_sources!`. Plan 17 Commit 3 ships a
unified KA-kernel version that the `SurfaceFluxOperator.apply!` path
launches; Commit 5 moves the call site into the palindrome. Until
Commit 6, DrivenSimulation continues to call this helper at the sim
level.
"""
function _apply_surface_source!(rm::AbstractArray{FT, 3},
                                source::SurfaceFluxSource, dt) where FT
    Nz = size(rm, 3)
    @views rm[:, :, Nz] .+= source.cell_mass_rate .* dt
    return nothing
end

function _apply_surface_source!(rm::AbstractArray{FT, 2},
                                source::SurfaceFluxSource, dt) where FT
    Nz = size(rm, 2)
    @views rm[:, Nz] .+= source.cell_mass_rate .* dt
    return nothing
end
