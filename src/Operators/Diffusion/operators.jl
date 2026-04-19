"""
    AbstractDiffusionOperator

Top of the diffusion operator hierarchy. Concrete subtypes in this
commit: [`NoDiffusion`](@ref), [`ImplicitVerticalDiffusion`](@ref).
Plan 16b+ can add non-local / counter-gradient variants as sibling
concrete types.

Every concrete subtype implements

    apply!(state::CellState, meteo, grid, op, dt; workspace)

mutating `state.tracers_raw` in place and returning `state`.
"""
abstract type AbstractDiffusionOperator end

"""
    NoDiffusion()

Identity operator — `apply!` is a no-op. Default for configurations
without active vertical mixing, and the value `strang_split_mt!` sees
when the palindrome's V position is unoccupied (Commit 4).
"""
struct NoDiffusion <: AbstractDiffusionOperator end

"""
    ImplicitVerticalDiffusion(; kz_field)

Backward-Euler vertical diffusion driven by a rank-3 Kz field. Any
`AbstractTimeVaryingField{FT, 3}` concrete type works:
[`ConstantField{FT, 3}`](@ref) for a uniform Kz,
[`ProfileKzField{FT}`](@ref) for a vertical profile,
[`PreComputedKzField{FT, A}`](@ref) for a 3D snapshot, or
[`DerivedKzField`](@ref) for meteorology-driven Beljaars-Viterbo.

# `apply!` contract

    apply!(state, meteo, grid, op::ImplicitVerticalDiffusion, dt; workspace)

- Refreshes the Kz cache: `update_field!(op.kz_field, t)` with
  `t` drawn from the meteorology where available; plan 16b currently
  passes `zero(FT)` as a placeholder (chemistry-style, mirrors plan 15's
  deferred `current_time(meteo)` accessor).
- Reads `workspace.dz_scratch` as the current layer thicknesses [m].
  The caller is responsible for filling this array before calling
  `apply!` — typically from a hydrostatic integration of the current
  `delp` and surface temperature.
- Uses `workspace.w_scratch` as Thomas-forward-elimination storage.
- Launches [`_vertical_diffusion_kernel!`](@ref) over
  `(Nx, Ny, Nt)`, one thread per column per tracer.

The operator is linear (Kz does not depend on tracer values), so
a single `apply!(dt)` at the palindrome center is equivalent to two
half-steps — see plan 16b §4.3 Decision 8. Commit 4 performs the
palindrome integration.

# Fields
- `kz_field::KzF` — any `AbstractTimeVaryingField{FT, 3}` providing
  cell-centered Kz values [m²/s geometric].
"""
struct ImplicitVerticalDiffusion{FT, KzF} <: AbstractDiffusionOperator
    kz_field :: KzF

    function ImplicitVerticalDiffusion{FT, KzF}(kz_field::KzF) where {FT, KzF}
        KzF <: AbstractTimeVaryingField{FT, 3} ||
            throw(ArgumentError("ImplicitVerticalDiffusion: kz_field must be an " *
                "AbstractTimeVaryingField{$FT, 3}, got $KzF"))
        new{FT, KzF}(kz_field)
    end
end

"""
    ImplicitVerticalDiffusion(; kz_field::AbstractTimeVaryingField{FT, 3})

Keyword constructor. `FT` is inferred from `kz_field`.
"""
function ImplicitVerticalDiffusion(; kz_field::AbstractTimeVaryingField{FT, 3}) where FT
    ImplicitVerticalDiffusion{FT, typeof(kz_field)}(kz_field)
end

# =========================================================================
# apply!
# =========================================================================

"""
    apply!(state::CellState, meteo, grid, op::NoDiffusion, dt; workspace=nothing)

No-op; returns `state` unchanged.
"""
function apply!(state::CellState, meteo, grid, ::NoDiffusion, dt;
                workspace = nothing)
    return state
end

"""
    apply!(state::CellState, meteo, grid, op::ImplicitVerticalDiffusion, dt;
           workspace)

Apply one Backward-Euler implicit diffusion step to every tracer in
`state.tracers_raw` using the column Kz field `op.kz_field` and the
dz stored in `workspace.dz_scratch` (caller-filled).

Throws if `workspace` is not supplied (diffusion cannot run without
`w_scratch` / `dz_scratch` buffers) or if the workspace's `dz_scratch`
is 0-sized (face-indexed grids; diffusion is structured-grid-only
in this commit).
"""
function apply!(state::CellState, meteo, grid,
                op::ImplicitVerticalDiffusion{FT}, dt;
                workspace) where FT
    workspace === nothing && throw(ArgumentError(
        "ImplicitVerticalDiffusion.apply!: workspace is required " *
        "(w_scratch and dz_scratch must be supplied)"))

    raw = state.tracers_raw
    ndims(raw) == 4 || throw(ArgumentError(
        "ImplicitVerticalDiffusion.apply!: tracers_raw must be rank-4, got ndims=$(ndims(raw))"))

    w_scratch  = workspace.w_scratch
    dz_scratch = workspace.dz_scratch
    length(dz_scratch) == 0 && throw(ArgumentError(
        "ImplicitVerticalDiffusion.apply!: workspace.dz_scratch is 0-sized — " *
        "diffusion is only supported on structured 3D grids in this commit"))
    size(dz_scratch) == size(w_scratch) ||
        throw(DimensionMismatch("w_scratch and dz_scratch sizes must match"))

    Nx, Ny, Nz, Nt = size(raw)
    (size(dz_scratch) == (Nx, Ny, Nz)) || throw(DimensionMismatch(
        "workspace scratch arrays are $(size(dz_scratch)) but tracers_raw " *
        "spatial shape is ($Nx, $Ny, $Nz)"))

    # Refresh Kz cache for the current time. Placeholder `t = zero(FT)`
    # matches the plan-15 chemistry convention until the meteorology
    # ships a `current_time` accessor (deferred; see plan 16b NOTES.md).
    t = zero(FT)
    update_field!(op.kz_field, t)

    backend = get_backend(raw)
    kernel = _vertical_diffusion_kernel!(backend, (8, 8, 1))
    kernel(raw, op.kz_field, dz_scratch, w_scratch, FT(dt), Nz;
           ndrange = (Nx, Ny, Nt))
    synchronize(backend)
    return state
end
