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

Backward-Euler vertical diffusion driven by a cell-centered Kz field.
Two spatial layouts are supported:

- structured: `AbstractTimeVaryingField{FT, 3}` over `(Nx, Ny, Nz)`
- face-indexed: `AbstractTimeVaryingField{FT, 2}` over `(ncells, Nz)`

Concrete examples:
- [`ConstantField{FT, 3}`](@ref) / [`ConstantField{FT, 2}`](@ref)
- [`ProfileKzField{FT}`](@ref) with default rank 3 or
  `ProfileKzField(profile; spatial_rank = 2)`
- [`PreComputedKzField{FT, A}`](@ref) wrapping 3D or 2D storage
- [`DerivedKzField`](@ref) for meteorology-driven Beljaars-Viterbo on
  structured grids

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
- Launches a layout-specific diffusion kernel:
  - structured: [`_vertical_diffusion_kernel!`](@ref) over `(Nx, Ny, Nt)`
  - face-indexed: `_vertical_diffusion_face_kernel!` over `(ncells, Nt)`

The operator is linear (Kz does not depend on tracer values), so
a single `apply!(dt)` at the palindrome center is equivalent to two
half-steps — see plan 16b §4.3 Decision 8. Commit 4 performs the
palindrome integration.

# Fields
- `kz_field::KzF` — any `AbstractTimeVaryingField{FT, 2}` or
  `AbstractTimeVaryingField{FT, 3}` providing cell-centered Kz values
  [m²/s geometric].
"""
struct ImplicitVerticalDiffusion{FT, KzF} <: AbstractDiffusionOperator
    kz_field :: KzF

    function ImplicitVerticalDiffusion{FT, KzF}(kz_field::KzF) where {FT, KzF}
        (KzF <: AbstractTimeVaryingField{FT, 2} ||
         KzF <: AbstractTimeVaryingField{FT, 3}) ||
            throw(ArgumentError("ImplicitVerticalDiffusion: kz_field must be an " *
                "AbstractTimeVaryingField{$FT, 2} or AbstractTimeVaryingField{$FT, 3}, got $KzF"))
        new{FT, KzF}(kz_field)
    end
end

"""
    ImplicitVerticalDiffusion(; kz_field::AbstractTimeVaryingField{FT, N})

Keyword constructor. `FT` is inferred from `kz_field`.
"""
function ImplicitVerticalDiffusion(; kz_field::AbstractTimeVaryingField{FT, N}) where {FT, N}
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
`state.tracers_raw` using the column Kz field `op.kz_field` and the dz
stored in `workspace.dz_scratch` (caller-filled). Delegates to
[`apply_vertical_diffusion!`](@ref), which is the lower-level entry
point consumed by both the structured multi-tracer palindrome and the
face-indexed reduced-Gaussian transport block.

Throws if `workspace` is not supplied or if its `dz_scratch` shape
doesn't match `state.tracers_raw`.
"""
function apply!(state::CellState, meteo, grid,
                op::ImplicitVerticalDiffusion{FT}, dt;
                workspace) where FT
    workspace === nothing && throw(ArgumentError(
        "ImplicitVerticalDiffusion.apply!: workspace is required " *
        "(w_scratch and dz_scratch must be supplied)"))
    apply_vertical_diffusion!(state.tracers_raw, op, workspace, dt, meteo)
    return state
end

# =========================================================================
# Lower-level apply_vertical_diffusion! — array-level entry point
# =========================================================================

"""
    apply_vertical_diffusion!(q_raw, op, workspace, dt, meteo = nothing) -> nothing

Low-level entry point. Applies one Backward-Euler diffusion step to a
raw tracer buffer in any of the supported layouts:

- structured packed tracers: `q_raw :: (Nx, Ny, Nz, Nt)`
- face-indexed packed tracers: `q_raw :: (ncells, Nz, Nt)`
- face-indexed single-tracer slice: `q_raw :: (ncells, Nz)`

This is the function `strang_split_mt!` calls at the palindrome
center (plan 16b Commit 4). The face-indexed reduced-Gaussian path also
uses it at its H → V → D → V → H center slot.

`meteo` is threaded through to `update_field!(op.kz_field, t)` as
`t = FT(current_time(meteo))` (or `zero(FT)` if `meteo === nothing`).
Plan 17 Commit 4: `meteo` defaults to `nothing` so pre-17 palindrome
call sites (`apply_vertical_diffusion!(rm, op, ws, dt)`) continue
to work unchanged; Commit 5 threads `meteo` through the palindrome.

`NoDiffusion` is a no-op: the method is `= nothing` so Julia's
dispatch reduces the call site to a dead branch when
`diffusion_op isa NoDiffusion`. This is what makes the Commit 4
palindrome integration bit-exact backward-compatible.
"""
function apply_vertical_diffusion! end

apply_vertical_diffusion!(q_raw::AbstractArray{<:Any, 4},
                          ::NoDiffusion, workspace, dt,
                          meteo = nothing) = nothing
apply_vertical_diffusion!(q_raw::AbstractArray{<:Any, 3},
                          ::NoDiffusion, workspace, dt,
                          meteo = nothing) = nothing
apply_vertical_diffusion!(q_raw::AbstractArray{<:Any, 2},
                          ::NoDiffusion, workspace, dt,
                          meteo = nothing) = nothing

@inline function _diffusion_time(::Type{FT}, meteo) where FT
    return meteo === nothing ? zero(FT) : FT(current_time(meteo))
end

@inline function _check_diffusion_workspace_shape(dz_scratch, w_scratch,
                                                  expected_shape, shape_label)
    size(dz_scratch) == size(w_scratch) ||
        throw(DimensionMismatch("w_scratch and dz_scratch sizes must match"))
    size(dz_scratch) == expected_shape || throw(DimensionMismatch(
        "workspace scratch arrays are $(size(dz_scratch)) but q_raw " *
        "$shape_label shape is $(expected_shape)"))
    return nothing
end

function apply_vertical_diffusion!(q_raw::AbstractArray{FT, 4},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace, dt,
                                   meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 3}}
    w_scratch  = workspace.w_scratch
    dz_scratch = workspace.dz_scratch

    Nx, Ny, Nz, Nt = size(q_raw)
    _check_diffusion_workspace_shape(dz_scratch, w_scratch, (Nx, Ny, Nz),
                                     "spatial")

    update_field!(op.kz_field, _diffusion_time(FT, meteo))

    backend = get_backend(q_raw)
    kernel = _vertical_diffusion_kernel!(backend, (8, 8, 1))
    kernel(q_raw, op.kz_field, dz_scratch, w_scratch, FT(dt), Nz;
           ndrange = (Nx, Ny, Nt))
    synchronize(backend)
    return nothing
end

function apply_vertical_diffusion!(q_raw::AbstractArray{FT, 3},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace, dt,
                                   meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 2}}
    w_scratch  = workspace.w_scratch
    dz_scratch = workspace.dz_scratch

    ncells, Nz, Nt = size(q_raw)
    _check_diffusion_workspace_shape(dz_scratch, w_scratch, (ncells, Nz),
                                     "face-indexed")

    update_field!(op.kz_field, _diffusion_time(FT, meteo))

    backend = get_backend(q_raw)
    kernel = _vertical_diffusion_face_kernel!(backend, 256)
    kernel(q_raw, op.kz_field, dz_scratch, w_scratch, FT(dt), Nz;
           ndrange = (ncells, Nt))
    synchronize(backend)
    return nothing
end

function apply_vertical_diffusion!(q_raw::AbstractArray{FT, 2},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace, dt,
                                   meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 2}}
    w_scratch  = workspace.w_scratch
    dz_scratch = workspace.dz_scratch

    ncells, Nz = size(q_raw)
    _check_diffusion_workspace_shape(dz_scratch, w_scratch, (ncells, Nz),
                                     "face-indexed")

    update_field!(op.kz_field, _diffusion_time(FT, meteo))

    backend = get_backend(q_raw)
    kernel = _vertical_diffusion_face_single_kernel!(backend, 256)
    kernel(q_raw, op.kz_field, dz_scratch, w_scratch, FT(dt), Nz;
           ndrange = ncells)
    synchronize(backend)
    return nothing
end

function apply_vertical_diffusion!(q_raw::AbstractArray{FT, 4},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace, dt,
                                   meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 2}}
    throw(ArgumentError(
        "apply_vertical_diffusion!: rank-2 kz_field is incompatible with " *
        "structured q_raw shape $(size(q_raw)); use a rank-3 field on " *
        "(Nx, Ny, Nz) grids"))
end

function apply_vertical_diffusion!(q_raw::AbstractArray{FT, 3},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace, dt,
                                   meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 3}}
    throw(ArgumentError(
        "apply_vertical_diffusion!: rank-3 kz_field is incompatible with " *
        "face-indexed q_raw shape $(size(q_raw)); use a rank-2 field on " *
        "(ncells, Nz, Nt) grids"))
end

function apply_vertical_diffusion!(q_raw::AbstractArray{FT, 2},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace, dt,
                                   meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 3}}
    throw(ArgumentError(
        "apply_vertical_diffusion!: rank-3 kz_field is incompatible with " *
        "face-indexed q_raw shape $(size(q_raw)); use a rank-2 field on " *
        "(ncells, Nz) tracer slices"))
end
