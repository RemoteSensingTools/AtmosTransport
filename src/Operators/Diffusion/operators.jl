# `AbstractDiffusion` is the global root declared in
# `src/Operators/AbstractOperators.jl`. Concrete subtypes here:
# [`NoDiffusion`](@ref), [`ImplicitVerticalDiffusion`](@ref). Non-local /
# counter-gradient variants can be added as sibling concrete types.
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
when the palindrome's V position is unoccupied.
"""
struct NoDiffusion <: AbstractDiffusion end

Adapt.adapt_structure(_to, op::NoDiffusion) = op

@inline uses_diffusive_surface_flux_boundary(::NoDiffusion) = false

"""
    ImplicitVerticalDiffusion(; kz_field)

Backward-Euler vertical diffusion driven by either a cell-centered Kz field or
an exact precomputed TM5 interface `dkg` field.
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
  `t` drawn from the meteorology where available; currently
  passes `zero(FT)` as a placeholder (chemistry-style, mirroring the
  deferred `current_time(meteo)` accessor).
- For Kz fields, reads `workspace.dz_scratch` as the current layer thicknesses [m].
  The caller is responsible for filling this array before calling
  `apply!` — typically from a hydrostatic integration of the current
  `delp` and surface temperature.
- `PrecomputedCSDkgField` bypasses Kz/geometry reconstruction and does not read
  `dz_scratch`; its interface exchange [kg s⁻¹] is already complete.
- Uses `workspace.w_scratch` as Thomas-forward-elimination storage.
- Launches a layout-specific diffusion kernel:
  - structured: `_vertical_diffusion_kernel!` over `(Nx, Ny, Nt)`
  - face-indexed: `_vertical_diffusion_face_kernel!` over `(ncells, Nt)`

The operator is linear (Kz does not depend on tracer values), so
a single `apply!(dt)` at the palindrome center is equivalent to two
half-steps.

# Fields
- `kz_field::KzF` — any `AbstractTimeVaryingField{FT, 2}` or
  `AbstractTimeVaryingField{FT, 3}` providing cell-centered Kz values
  [m²/s geometric], or a `PrecomputedCSDkgField`. The field name is retained
  for source compatibility with existing operator construction.
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
    # LL packed + RG face-indexed now go through the mass-flux VMR
    # wrapper: pre-scale tracer_mass → VMR, solve with mass-flux
    # coefficients, post-scale VMR → tracer_mass. Preserves `Σ m·q` to
    # roundoff for inert tracers, matching the CS path's conservation
    # contract. See `memory/diffusion_full_pipeline_audit_2026_05_25.md`.
    apply_vertical_diffusion_vmr!(state.tracers_raw, state.air_mass,
                                   op, workspace, dt, meteo)
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
center. The face-indexed reduced-Gaussian path also
uses it at its H → V → D → V → H center slot.

`meteo` is threaded through to `update_field!(op.kz_field, t)` as
`t = FT(current_time(meteo))` (or `zero(FT)` if `meteo === nothing`).
`meteo` defaults to `nothing` so call sites that omit it
(`apply_vertical_diffusion!(rm, op, ws, dt)`) continue to work unchanged.

`NoDiffusion` is a no-op: the method is `= nothing` so Julia's
dispatch reduces the call site to a dead branch when
`diffusion_op isa NoDiffusion`. This makes the palindrome
integration bit-exact backward-compatible.
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
# CS-only mass-flux variant takes air_mass too; the no-op overload covers
# the new dispatch.
apply_vertical_diffusion!(q_raw::NTuple{6}, air_mass::NTuple{6},
                          ::NoDiffusion, workspace, dt,
                          meteo = nothing; halo_width = 0) = nothing
apply_vertical_diffusion_vmr!(q_raw::NTuple{6}, air_mass::NTuple{6},
                              ::NoDiffusion, workspace, dt,
                              meteo = nothing; halo_width = 0) = nothing
# LL/RG mass-flux NoDiffusion stubs (new VMR wrappers and the bare
# air_mass-bearing apply variants must also no-op).
apply_vertical_diffusion!(q_raw::AbstractArray{<:Any, 4},
                          air_mass::AbstractArray{<:Any, 3},
                          ::NoDiffusion, workspace, dt,
                          meteo = nothing) = nothing
apply_vertical_diffusion!(q_raw::AbstractArray{<:Any, 3},
                          air_mass::AbstractArray{<:Any, 2},
                          ::NoDiffusion, workspace, dt,
                          meteo = nothing) = nothing
apply_vertical_diffusion!(q_raw::AbstractArray{<:Any, 2},
                          air_mass::AbstractArray{<:Any, 2},
                          ::NoDiffusion, workspace, dt,
                          meteo = nothing) = nothing
apply_vertical_diffusion_vmr!(q_raw::AbstractArray{<:Any, 4},
                              air_mass::AbstractArray{<:Any, 3},
                              ::NoDiffusion, workspace, dt,
                              meteo = nothing) = nothing
apply_vertical_diffusion_vmr!(q_raw::AbstractArray{<:Any, 3},
                              air_mass::AbstractArray{<:Any, 2},
                              ::NoDiffusion, workspace, dt,
                              meteo = nothing) = nothing
apply_vertical_diffusion_vmr!(q_raw::AbstractArray{<:Any, 2},
                              air_mass::AbstractArray{<:Any, 2},
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
                                   air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace, dt,
                                   meteo = nothing;
                                   halo_width::Integer) where {FT, A <: AbstractArray{FT, 3},
                                                                KzF <: PrecomputedCSDkgField{FT}}
    w_scratch = getproperty(workspace, :w_scratch)
    update_field!(op.kz_field, _diffusion_time(FT, meteo))
    Hp = Int(halo_width)
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        panel_m = air_mass[p]
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        size(w_scratch[p]) == (Nc, Ny, Nz) || throw(DimensionMismatch(
            "cubed-sphere dkg workspace panel $p has shape $(size(w_scratch[p])); expected $((Nc, Ny, Nz))"))
        panel_dkg = panel_field(op.kz_field, p)
        backend = get_backend(panel_q)
        kernel = _vertical_diffusion_cs_single_dkg_kernel!(backend, (8, 8))
        kernel(panel_q, panel_m, panel_dkg, w_scratch[p], FT(dt), Nz, Hp;
               ndrange = (Nc, Ny))
        synchronize(backend)
    end
    return nothing
end

function apply_vertical_diffusion!(q_raw::NTuple{6, A},
                                   air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace, dt,
                                   meteo = nothing;
                                   halo_width::Integer) where {FT, A <: AbstractArray{FT, 4},
                                                                KzF <: PrecomputedCSDkgField{FT}}
    w_scratch = getproperty(workspace, :w_scratch)
    reference_scratch = getproperty(workspace, :diffusion_reference)
    length(w_scratch) == 6 && length(reference_scratch) == 6 ||
        throw(DimensionMismatch(
            "cubed-sphere dkg workspace must provide 6 factor and reference panels"))
    update_field!(op.kz_field, _diffusion_time(FT, meteo))
    Hp = Int(halo_width)
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        panel_m = air_mass[p]
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        Nt = size(panel_q, 4)
        size(w_scratch[p]) == (Nc, Ny, Nz) || throw(DimensionMismatch(
            "cubed-sphere dkg workspace panel $p has shape $(size(w_scratch[p])); expected $((Nc, Ny, Nz))"))
        size(reference_scratch[p]) == (Nc, Ny, Nt) || throw(DimensionMismatch(
            "cubed-sphere dkg reference panel $p has shape $(size(reference_scratch[p])); expected $((Nc, Ny, Nt))"))
        panel_dkg = panel_field(op.kz_field, p)
        backend = get_backend(panel_q)
        kernel = _vertical_diffusion_cs_dkg_kernel!(backend, (8, 8))
        kernel(panel_q, panel_m, panel_dkg, w_scratch[p], reference_scratch[p],
               FT(dt), Nz, Nt, Hp; ndrange = (Nc, Ny))
        synchronize(backend)
    end
    return nothing
end

function apply_vertical_diffusion!(q_raw::NTuple{6, A},
                                   air_mass::NTuple{6, <:AbstractArray{FT, 3}},
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
        throw(DimensionMismatch(
            "cubed-sphere diffusion workspace must provide 6 factor and geometry panels"))

    update_field!(op.kz_field, _diffusion_time(FT, meteo))

    Hp = Int(halo_width)
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        panel_m = air_mass[p]
        size(panel_q) == size(panel_m) || throw(DimensionMismatch(
            "cubed-sphere diffusion panel $p: q shape $(size(panel_q)) does not match " *
            "air_mass shape $(size(panel_m))"))
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        _check_cs_diffusion_workspace_shape(dz_scratch[p], w_scratch[p], (Nc, Ny, Nz), p)
        panel_kz = panel_field(op.kz_field, p)
        backend = get_backend(panel_q)
        kernel = _vertical_diffusion_cs_single_kernel!(backend, (8, 8))
        kernel(panel_q, panel_m, panel_kz, dz_scratch[p], w_scratch[p],
               FT(dt), Nz, Hp;
               ndrange = (Nc, Ny))
        synchronize(backend)
    end
    return nothing
end

function apply_vertical_diffusion!(q_raw::NTuple{6, A},
                                   air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace, dt,
                                   meteo = nothing;
                                   halo_width::Integer) where {FT, A <: AbstractArray{FT, 4},
                                                                KzF <: AbstractCubedSphereField{FT}}
    hasproperty(workspace, :w_scratch) && hasproperty(workspace, :dz_scratch) &&
        hasproperty(workspace, :diffusion_reference) ||
        throw(ArgumentError(
            "cubed-sphere diffusion requires a workspace with panel-native " *
            "`w_scratch`, `dz_scratch`, and `diffusion_reference` tuples"))

    w_scratch = getproperty(workspace, :w_scratch)
    dz_scratch = getproperty(workspace, :dz_scratch)
    reference_scratch = getproperty(workspace, :diffusion_reference)
    length(w_scratch) == 6 && length(dz_scratch) == 6 &&
        length(reference_scratch) == 6 || throw(DimensionMismatch(
            "cubed-sphere diffusion workspace must provide 6 factor, geometry, and reference panels"))

    update_field!(op.kz_field, _diffusion_time(FT, meteo))

    Hp = Int(halo_width)
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        panel_m = air_mass[p]
        size(panel_q)[1:3] == size(panel_m) || throw(DimensionMismatch(
            "cubed-sphere packed diffusion panel $p: q spatial shape " *
            "$(size(panel_q)[1:3]) does not match air_mass shape $(size(panel_m))"))
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        Nt = size(panel_q, 4)
        _check_cs_diffusion_workspace_shape(dz_scratch[p], w_scratch[p], (Nc, Ny, Nz), p)
        size(reference_scratch[p]) == (Nc, Ny, Nt) || throw(DimensionMismatch(
            "cubed-sphere diffusion reference panel $p has shape $(size(reference_scratch[p])); expected $((Nc, Ny, Nt))"))
        panel_kz = panel_field(op.kz_field, p)
        backend = get_backend(panel_q)
        kernel = _vertical_diffusion_cs_kernel!(backend, (8, 8))
        kernel(panel_q, panel_m, panel_kz, dz_scratch[p], w_scratch[p],
               reference_scratch[p], FT(dt), Nz, Nt, Hp;
               ndrange = (Nc, Ny))
        synchronize(backend)
    end
    return nothing
end

function _cs_scale_tracer_mass_to_vmr!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       halo_width::Integer) where {A <: AbstractArray}
    # Single barrier at the end: the six CS panels are spatially
    # independent within this scale step, so the six kernel launches
    # can stream through the backend's queue and we only need one
    # synchronize() before the caller observes the result. Saves five
    # device-host round-trips per call.
    Hp = Int(halo_width)
    backend = get_backend(q_raw[1])
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
        kernel = _cs_tracer_mass_to_vmr_kernel!(backend, (8, 8, 1))
        kernel(panel_q, panel_m, Hp; ndrange = (Nc, Ny, Nz))
    end
    synchronize(backend)
    return q_raw
end

function _cs_scale_tracer_mass_to_vmr!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       halo_width::Integer) where {A <: AbstractArray{<:Any, 4}}
    Hp = Int(halo_width)
    backend = get_backend(q_raw[1])
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
        kernel = _cs_tracer_mass_to_vmr_4d_kernel!(backend, (8, 8, 1))
        kernel(panel_q, panel_m, Hp; ndrange = (Nc, Ny, Nz, Nt))
    end
    synchronize(backend)
    return q_raw
end

function _cs_scale_vmr_to_tracer_mass!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       halo_width::Integer) where {A <: AbstractArray}
    Hp = Int(halo_width)
    backend = get_backend(q_raw[1])
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
        kernel = _cs_vmr_to_tracer_mass_kernel!(backend, (8, 8, 1))
        kernel(panel_q, panel_m, Hp; ndrange = (Nc, Ny, Nz))
    end
    synchronize(backend)
    return q_raw
end

function _cs_scale_vmr_to_tracer_mass!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       halo_width::Integer) where {A <: AbstractArray{<:Any, 4}}
    Hp = Int(halo_width)
    backend = get_backend(q_raw[1])
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
        kernel = _cs_vmr_to_tracer_mass_4d_kernel!(backend, (8, 8, 1))
        kernel(panel_q, panel_m, Hp; ndrange = (Nc, Ny, Nz, Nt))
    end
    synchronize(backend)
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
    # Mass-flux kernel needs `air_mass` for coefficient construction.
    apply_vertical_diffusion!(q_raw, air_mass, op, workspace, dt, meteo;
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
    # Mass-flux kernel needs `air_mass` for coefficient construction.
    apply_vertical_diffusion!(q_raw, air_mass, op, workspace, dt, meteo;
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

# ---------------------------------------------------------------------------
# Mass-flux LL packed + face-indexed RG paths — opt-in via the
# `apply_vertical_diffusion_vmr!` wrapper. These methods take `air_mass`
# explicitly and dispatch to the mass-flux kernel variants. The legacy
# geometric `apply_vertical_diffusion!` API (no `air_mass`) is preserved
# above for backward compatibility — callers that haven't been
# updated continue to get the leaky geometric behavior.
# ---------------------------------------------------------------------------

function apply_vertical_diffusion!(q_raw::AbstractArray{FT, 4},
                                   air_mass::AbstractArray{FT, 3},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace, dt,
                                   meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 3}}
    w_scratch  = workspace.w_scratch
    dz_scratch = workspace.dz_scratch
    Nx, Ny, Nz, Nt = size(q_raw)
    _check_diffusion_workspace_shape(dz_scratch, w_scratch, (Nx, Ny, Nz),
                                     "spatial")
    size(air_mass) == (Nx, Ny, Nz) || throw(DimensionMismatch(
        "LL mass-flux diffusion: air_mass shape $(size(air_mass)) does not " *
        "match q_raw spatial shape $((Nx, Ny, Nz))"))
    update_field!(op.kz_field, _diffusion_time(FT, meteo))
    backend = get_backend(q_raw)
    kernel = _vertical_diffusion_kernel_mass_flux!(backend, (8, 8))
    kernel(q_raw, air_mass, op.kz_field, dz_scratch, w_scratch, FT(dt), Nz, Nt;
           ndrange = (Nx, Ny))
    synchronize(backend)
    return nothing
end

function apply_vertical_diffusion!(q_raw::AbstractArray{FT, 3},
                                   air_mass::AbstractArray{FT, 2},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace, dt,
                                   meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 2}}
    w_scratch  = workspace.w_scratch
    dz_scratch = workspace.dz_scratch
    ncells, Nz, Nt = size(q_raw)
    _check_diffusion_workspace_shape(dz_scratch, w_scratch, (ncells, Nz),
                                     "face-indexed")
    size(air_mass) == (ncells, Nz) || throw(DimensionMismatch(
        "RG mass-flux diffusion: air_mass shape $(size(air_mass)) does not " *
        "match q_raw spatial shape $((ncells, Nz))"))
    update_field!(op.kz_field, _diffusion_time(FT, meteo))
    backend = get_backend(q_raw)
    kernel = _vertical_diffusion_face_kernel_mass_flux!(backend, 256)
    kernel(q_raw, air_mass, op.kz_field, dz_scratch, w_scratch, FT(dt), Nz, Nt;
           ndrange = ncells)
    synchronize(backend)
    return nothing
end

function apply_vertical_diffusion!(q_raw::AbstractArray{FT, 2},
                                   air_mass::AbstractArray{FT, 2},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace, dt,
                                   meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 2}}
    w_scratch  = workspace.w_scratch
    dz_scratch = workspace.dz_scratch
    ncells, Nz = size(q_raw)
    _check_diffusion_workspace_shape(dz_scratch, w_scratch, (ncells, Nz),
                                     "face-indexed")
    size(air_mass) == (ncells, Nz) || throw(DimensionMismatch(
        "RG mass-flux diffusion: air_mass shape $(size(air_mass)) does not " *
        "match q_raw shape $((ncells, Nz))"))
    update_field!(op.kz_field, _diffusion_time(FT, meteo))
    backend = get_backend(q_raw)
    kernel = _vertical_diffusion_face_single_kernel_mass_flux!(backend, 256)
    kernel(q_raw, air_mass, op.kz_field, dz_scratch, w_scratch, FT(dt), Nz;
           ndrange = ncells)
    synchronize(backend)
    return nothing
end

# Mass-VMR scaling helpers for the LL + RG arrays.

function _ll_scale_tracer_mass_to_vmr!(q_raw::AbstractArray{FT, 4},
                                       air_mass::AbstractArray{FT, 3}) where {FT}
    Nx, Ny, Nz, Nt = size(q_raw)
    backend = get_backend(q_raw)
    kernel = _ll_tracer_mass_to_vmr_kernel!(backend, (8, 8, 1, 1))
    kernel(q_raw, air_mass; ndrange = (Nx, Ny, Nz, Nt))
    synchronize(backend)
    return q_raw
end

function _ll_scale_vmr_to_tracer_mass!(q_raw::AbstractArray{FT, 4},
                                       air_mass::AbstractArray{FT, 3}) where {FT}
    Nx, Ny, Nz, Nt = size(q_raw)
    backend = get_backend(q_raw)
    kernel = _ll_vmr_to_tracer_mass_kernel!(backend, (8, 8, 1, 1))
    kernel(q_raw, air_mass; ndrange = (Nx, Ny, Nz, Nt))
    synchronize(backend)
    return q_raw
end

function _face_scale_tracer_mass_to_vmr!(q_raw::AbstractArray{FT, 3},
                                         air_mass::AbstractArray{FT, 2}) where {FT}
    ncells, Nz, Nt = size(q_raw)
    backend = get_backend(q_raw)
    kernel = _face_tracer_mass_to_vmr_kernel!(backend, (256, 1, 1))
    kernel(q_raw, air_mass; ndrange = (ncells, Nz, Nt))
    synchronize(backend)
    return q_raw
end

function _face_scale_vmr_to_tracer_mass!(q_raw::AbstractArray{FT, 3},
                                         air_mass::AbstractArray{FT, 2}) where {FT}
    ncells, Nz, Nt = size(q_raw)
    backend = get_backend(q_raw)
    kernel = _face_vmr_to_tracer_mass_kernel!(backend, (256, 1, 1))
    kernel(q_raw, air_mass; ndrange = (ncells, Nz, Nt))
    synchronize(backend)
    return q_raw
end

function _face_scale_tracer_mass_to_vmr!(q_raw::AbstractArray{FT, 2},
                                         air_mass::AbstractArray{FT, 2}) where {FT}
    ncells, Nz = size(q_raw)
    backend = get_backend(q_raw)
    kernel = _face_single_tracer_mass_to_vmr_kernel!(backend, (256, 1))
    kernel(q_raw, air_mass; ndrange = (ncells, Nz))
    synchronize(backend)
    return q_raw
end

function _face_scale_vmr_to_tracer_mass!(q_raw::AbstractArray{FT, 2},
                                         air_mass::AbstractArray{FT, 2}) where {FT}
    ncells, Nz = size(q_raw)
    backend = get_backend(q_raw)
    kernel = _face_single_vmr_to_tracer_mass_kernel!(backend, (256, 1))
    kernel(q_raw, air_mass; ndrange = (ncells, Nz))
    synchronize(backend)
    return q_raw
end

# LL packed VMR wrapper: pre-scale tracer_mass → VMR, mass-flux solve,
# post-scale VMR → tracer_mass.
function apply_vertical_diffusion_vmr!(q_raw::AbstractArray{FT, 4},
                                       air_mass::AbstractArray{FT, 3},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace, dt,
                                       meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 3}}
    _ll_scale_tracer_mass_to_vmr!(q_raw, air_mass)
    apply_vertical_diffusion!(q_raw, air_mass, op, workspace, dt, meteo)
    _ll_scale_vmr_to_tracer_mass!(q_raw, air_mass)
    return nothing
end

# Face-indexed RG packed VMR wrapper.
function apply_vertical_diffusion_vmr!(q_raw::AbstractArray{FT, 3},
                                       air_mass::AbstractArray{FT, 2},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace, dt,
                                       meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 2}}
    _face_scale_tracer_mass_to_vmr!(q_raw, air_mass)
    apply_vertical_diffusion!(q_raw, air_mass, op, workspace, dt, meteo)
    _face_scale_vmr_to_tracer_mass!(q_raw, air_mass)
    return nothing
end

# Face-indexed RG single-tracer VMR wrapper.
function apply_vertical_diffusion_vmr!(q_raw::AbstractArray{FT, 2},
                                       air_mass::AbstractArray{FT, 2},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace, dt,
                                       meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 2}}
    _face_scale_tracer_mass_to_vmr!(q_raw, air_mass)
    apply_vertical_diffusion!(q_raw, air_mass, op, workspace, dt, meteo)
    _face_scale_vmr_to_tracer_mass!(q_raw, air_mass)
    return nothing
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
