# `AbstractDiffusion` is the global root declared in
# `src/Operators/AbstractOperators.jl`. Concrete subtypes here:
# [`NoDiffusion`](@ref), [`ImplicitVerticalDiffusion`](@ref). Plan 16b+
# can add non-local / counter-gradient variants as sibling concrete types.
# Every concrete subtype implements
#
#     apply!(state::CellState, meteo, grid, op, dt; workspace)
#
# mutating `state.tracers_raw` in place and returning `state`.

"""
    AbstractSurfaceFluxCoupling

Typed policy for where configured surface fluxes enter relative to a
vertical diffusion/mixing operator.

- `SplitSurfaceFluxCoupling`: existing Strang-center composition,
  `V(dt/2) -> S(dt) -> V(dt/2)`.
- `DiffusiveSurfaceFluxBoundary`: GCHP/VDIFF-style lower-boundary
  placement, `S(dt) -> V(dt)`, so fresh surface flux is included in
  the implicit vertical mixing solve.
"""
abstract type AbstractSurfaceFluxCoupling end

struct SplitSurfaceFluxCoupling <: AbstractSurfaceFluxCoupling end
struct DiffusiveSurfaceFluxBoundary <: AbstractSurfaceFluxCoupling end

"""
    NoDiffusion()

Identity operator — `apply!` is a no-op. Default for configurations
without active vertical mixing, and the value `strang_split_mt!` sees
when the palindrome's V position is unoccupied (Commit 4).
"""
struct NoDiffusion <: AbstractDiffusion end

Adapt.adapt_structure(_to, op::NoDiffusion) = op

@inline uses_diffusive_surface_flux_boundary(::NoDiffusion) = false

"""
    ImplicitVerticalDiffusion(; kz_field)

Backward-Euler vertical diffusion driven by a cell-centered Kz field.
Two spatial layouts are supported:

- structured: `AbstractTimeVaryingField{FT, 3}` over `(Nx, Ny, Nz)`
- face-indexed: `AbstractTimeVaryingField{FT, 2}` over `(ncells, Nz)`

Concrete examples:
- `ConstantField{FT, 3}` / `ConstantField{FT, 2}`
- `ProfileKzField{FT}` with default rank 3 or
  `ProfileKzField(profile; spatial_rank = 2)`
- `PreComputedKzField{FT, A}` wrapping 3D or 2D storage
- `DerivedKzField` for meteorology-driven Beljaars-Viterbo on
  structured grids
- `CubedSphereField` wrapping one structured rank-3 field per panel

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
  - structured: `_vertical_diffusion_kernel!` over `(Nx, Ny, Nt)`
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
struct ImplicitVerticalDiffusion{FT, KzF, SFC <: AbstractSurfaceFluxCoupling} <: AbstractDiffusion
    kz_field              :: KzF
    surface_flux_coupling :: SFC

    function ImplicitVerticalDiffusion{FT, KzF, SFC}(kz_field::KzF,
                                                     surface_flux_coupling::SFC) where {
                                                     FT, KzF, SFC <: AbstractSurfaceFluxCoupling}
        (KzF <: AbstractTimeVaryingField{FT, 2} ||
         KzF <: AbstractTimeVaryingField{FT, 3} ||
         KzF <: AbstractCubedSphereField{FT}) ||
            throw(ArgumentError("ImplicitVerticalDiffusion: kz_field must be an " *
                "AbstractTimeVaryingField{$FT, 2}, AbstractTimeVaryingField{$FT, 3}, " *
                "or AbstractCubedSphereField{$FT}; got $KzF"))
        new{FT, KzF, SFC}(kz_field, surface_flux_coupling)
    end
end

"""
    ImplicitVerticalDiffusion(; kz_field)

Keyword constructor. `FT` is inferred from `kz_field`.
"""
@inline _diffusion_field_eltype(::AbstractTimeVaryingField{FT}) where FT = FT
@inline _diffusion_field_eltype(::AbstractCubedSphereField{FT}) where FT = FT

@inline function _diffusion_field_eltype(kz_field)
    throw(ArgumentError("ImplicitVerticalDiffusion: kz_field must be an " *
        "AbstractTimeVaryingField or AbstractCubedSphereField; got $(typeof(kz_field))"))
end

function ImplicitVerticalDiffusion(; kz_field,
                                   surface_flux_coupling::AbstractSurfaceFluxCoupling =
                                       SplitSurfaceFluxCoupling())
    FT = _diffusion_field_eltype(kz_field)
    return ImplicitVerticalDiffusion{FT, typeof(kz_field),
                                     typeof(surface_flux_coupling)}(
        kz_field, surface_flux_coupling)
end

Adapt.adapt_structure(to, op::ImplicitVerticalDiffusion) =
    ImplicitVerticalDiffusion(; kz_field = Adapt.adapt(to, op.kz_field),
                              surface_flux_coupling = op.surface_flux_coupling)

@inline uses_diffusive_surface_flux_boundary(op::ImplicitVerticalDiffusion) =
    op.surface_flux_coupling isa DiffusiveSurfaceFluxBoundary

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

function apply!(state::CubedSphereState, meteo, grid, ::NoDiffusion, dt;
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

function apply!(state::CubedSphereState, meteo, grid,
                op::ImplicitVerticalDiffusion{FT, KzF}, dt;
                workspace) where {FT, KzF <: AbstractCubedSphereField{FT}}
    workspace === nothing && throw(ArgumentError(
        "ImplicitVerticalDiffusion.apply!: workspace is required " *
        "(cubed-sphere diffusion needs panel-native w_scratch and dz_scratch)"))
    apply_vertical_diffusion_vmr!(state.tracers_raw, state.air_mass, op,
                                  workspace, dt, meteo;
                                  halo_width = state.halo_width)
    return state
end

function apply!(state::CubedSphereState, meteo, grid,
                op::ImplicitVerticalDiffusion, dt;
                workspace) 
    throw(ArgumentError(
        "CubedSphereState diffusion requires a panel-native kz_field. " *
        "Wrap six structured rank-3 fields in CubedSphereField(...) before " *
        "constructing ImplicitVerticalDiffusion."))
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
function apply_vertical_diffusion_vmr! end

apply_vertical_diffusion!(q_raw::AbstractArray{<:Any, 4},
                          ::NoDiffusion, workspace, dt,
                          meteo = nothing) = nothing
apply_vertical_diffusion!(q_raw::AbstractArray{<:Any, 3},
                          ::NoDiffusion, workspace, dt,
                          meteo = nothing) = nothing
apply_vertical_diffusion!(q_raw::AbstractArray{<:Any, 2},
                          ::NoDiffusion, workspace, dt,
                          meteo = nothing) = nothing
apply_vertical_diffusion!(q_raw::NTuple{6}, ::NoDiffusion, workspace, dt,
                          meteo = nothing; halo_width = 0) = nothing
apply_vertical_diffusion_vmr!(q_raw::NTuple{6}, air_mass::NTuple{6},
                              ::NoDiffusion, workspace, dt,
                              meteo = nothing; halo_width = 0) = nothing

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

@inline function _check_cs_diffusion_workspace_shape(dz_scratch, w_scratch,
                                                     expected_shape, panel)
    size(dz_scratch) == size(w_scratch) ||
        throw(DimensionMismatch("cubed-sphere w_scratch and dz_scratch sizes must match on panel $panel"))
    size(dz_scratch) == expected_shape || throw(DimensionMismatch(
        "cubed-sphere workspace scratch arrays on panel $panel are $(size(dz_scratch)) " *
        "but the interior panel shape is $(expected_shape)"))
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

function apply_vertical_diffusion!(q_raw::NTuple{6, A},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace, dt,
                                   meteo = nothing;
                                   halo_width::Integer) where {FT, A <: AbstractArray{FT, 3},
                                                                KzF <: AbstractCubedSphereField{FT}}
    hasproperty(workspace, :w_scratch) && hasproperty(workspace, :dz_scratch) ||
        throw(ArgumentError(
            "cubed-sphere diffusion requires a workspace with panel-native " *
            "`w_scratch` and `dz_scratch` tuples"))

    w_scratch = getproperty(workspace, :w_scratch)
    dz_scratch = getproperty(workspace, :dz_scratch)
    length(w_scratch) == 6 && length(dz_scratch) == 6 ||
        throw(DimensionMismatch("cubed-sphere diffusion workspace must provide 6 panel scratch arrays"))

    update_field!(op.kz_field, _diffusion_time(FT, meteo))

    Hp = Int(halo_width)
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        _check_cs_diffusion_workspace_shape(dz_scratch[p], w_scratch[p], (Nc, Ny, Nz), p)
        panel_kz = panel_field(op.kz_field, p)
        backend = get_backend(panel_q)
        kernel = _vertical_diffusion_cs_single_kernel!(backend, (8, 8))
        kernel(panel_q, panel_kz, dz_scratch[p], w_scratch[p], FT(dt), Nz, Hp;
               ndrange = (Nc, Ny))
        synchronize(backend)
    end
    return nothing
end

function apply_vertical_diffusion!(q_raw::NTuple{6, A},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace, dt,
                                   meteo = nothing;
                                   halo_width::Integer) where {FT, A <: AbstractArray{FT, 4},
                                                                KzF <: AbstractCubedSphereField{FT}}
    hasproperty(workspace, :w_scratch) && hasproperty(workspace, :dz_scratch) ||
        throw(ArgumentError(
            "cubed-sphere diffusion requires a workspace with panel-native " *
            "`w_scratch` and `dz_scratch` tuples"))

    w_scratch = getproperty(workspace, :w_scratch)
    dz_scratch = getproperty(workspace, :dz_scratch)
    length(w_scratch) == 6 && length(dz_scratch) == 6 ||
        throw(DimensionMismatch("cubed-sphere diffusion workspace must provide 6 panel scratch arrays"))

    update_field!(op.kz_field, _diffusion_time(FT, meteo))

    Hp = Int(halo_width)
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        Nt = size(panel_q, 4)
        _check_cs_diffusion_workspace_shape(dz_scratch[p], w_scratch[p], (Nc, Ny, Nz), p)
        panel_kz = panel_field(op.kz_field, p)
        backend = get_backend(panel_q)
        kernel = _vertical_diffusion_cs_kernel!(backend, (8, 8, 1))
        kernel(panel_q, panel_kz, dz_scratch[p], w_scratch[p], FT(dt), Nz, Hp;
               ndrange = (Nc, Ny, Nt))
        synchronize(backend)
    end
    return nothing
end

function _cs_scale_tracer_mass_to_vmr!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       halo_width::Integer) where {A <: AbstractArray}
    Hp = Int(halo_width)
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        panel_m = air_mass[p]
        size(panel_q) == size(panel_m) || throw(DimensionMismatch(
            "cubed-sphere tracer panel $p shape $(size(panel_q)) does not match " *
            "air_mass shape $(size(panel_m))"))
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        Nc > 0 && Ny > 0 || throw(DimensionMismatch(
            "cubed-sphere panel $p shape $(size(panel_q)) cannot provide an " *
            "interior with halo_width=$Hp"))
        backend = get_backend(panel_q)
        kernel = _cs_tracer_mass_to_vmr_kernel!(backend, (8, 8, 1))
        kernel(panel_q, panel_m, Hp; ndrange = (Nc, Ny, Nz))
        synchronize(backend)
    end
    return q_raw
end

function _cs_scale_tracer_mass_to_vmr!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       halo_width::Integer) where {A <: AbstractArray{<:Any, 4}}
    Hp = Int(halo_width)
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        panel_m = air_mass[p]
        size(panel_q)[1:3] == size(panel_m) || throw(DimensionMismatch(
            "cubed-sphere packed tracer panel $p spatial shape $(size(panel_q)[1:3]) " *
            "does not match air_mass shape $(size(panel_m))"))
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        Nt = size(panel_q, 4)
        Nc > 0 && Ny > 0 || throw(DimensionMismatch(
            "cubed-sphere panel $p shape $(size(panel_q)) cannot provide an " *
            "interior with halo_width=$Hp"))
        backend = get_backend(panel_q)
        kernel = _cs_tracer_mass_to_vmr_4d_kernel!(backend, (8, 8, 1))
        kernel(panel_q, panel_m, Hp; ndrange = (Nc, Ny, Nz, Nt))
        synchronize(backend)
    end
    return q_raw
end

function _cs_scale_vmr_to_tracer_mass!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       halo_width::Integer) where {A <: AbstractArray}
    Hp = Int(halo_width)
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        panel_m = air_mass[p]
        size(panel_q) == size(panel_m) || throw(DimensionMismatch(
            "cubed-sphere tracer panel $p shape $(size(panel_q)) does not match " *
            "air_mass shape $(size(panel_m))"))
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        Nc > 0 && Ny > 0 || throw(DimensionMismatch(
            "cubed-sphere panel $p shape $(size(panel_q)) cannot provide an " *
            "interior with halo_width=$Hp"))
        backend = get_backend(panel_q)
        kernel = _cs_vmr_to_tracer_mass_kernel!(backend, (8, 8, 1))
        kernel(panel_q, panel_m, Hp; ndrange = (Nc, Ny, Nz))
        synchronize(backend)
    end
    return q_raw
end

function _cs_scale_vmr_to_tracer_mass!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       halo_width::Integer) where {A <: AbstractArray{<:Any, 4}}
    Hp = Int(halo_width)
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        panel_m = air_mass[p]
        size(panel_q)[1:3] == size(panel_m) || throw(DimensionMismatch(
            "cubed-sphere packed tracer panel $p spatial shape $(size(panel_q)[1:3]) " *
            "does not match air_mass shape $(size(panel_m))"))
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        Nt = size(panel_q, 4)
        Nc > 0 && Ny > 0 || throw(DimensionMismatch(
            "cubed-sphere panel $p shape $(size(panel_q)) cannot provide an " *
            "interior with halo_width=$Hp"))
        backend = get_backend(panel_q)
        kernel = _cs_vmr_to_tracer_mass_4d_kernel!(backend, (8, 8, 1))
        kernel(panel_q, panel_m, Hp; ndrange = (Nc, Ny, Nz, Nt))
        synchronize(backend)
    end
    return q_raw
end

"""
    apply_vertical_diffusion_vmr!(rm, air_mass, op, workspace, dt, meteo; halo_width)

Cubed-sphere helper for state variables stored as tracer mass. The implicit
vertical solver acts on mixing ratio; this wrapper converts tracer mass to VMR
using the current dry air mass, applies the existing column solve, then restores
tracer mass before advection resumes.
"""
function apply_vertical_diffusion_vmr!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace, dt,
                                       meteo = nothing;
                                       halo_width::Integer) where {FT, A <: AbstractArray{FT, 3},
                                                                    KzF <: AbstractCubedSphereField{FT}}
    _cs_scale_tracer_mass_to_vmr!(q_raw, air_mass, halo_width)
    apply_vertical_diffusion!(q_raw, op, workspace, dt, meteo;
                              halo_width = halo_width)
    _cs_scale_vmr_to_tracer_mass!(q_raw, air_mass, halo_width)
    return nothing
end

function apply_vertical_diffusion_vmr!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace, dt,
                                       meteo = nothing;
                                       halo_width::Integer) where {FT, A <: AbstractArray{FT, 4},
                                                                    KzF <: AbstractCubedSphereField{FT}}
    _cs_scale_tracer_mass_to_vmr!(q_raw, air_mass, halo_width)
    apply_vertical_diffusion!(q_raw, op, workspace, dt, meteo;
                              halo_width = halo_width)
    _cs_scale_vmr_to_tracer_mass!(q_raw, air_mass, halo_width)
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
