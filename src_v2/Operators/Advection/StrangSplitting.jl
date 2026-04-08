# ---------------------------------------------------------------------------
# Strang splitting orchestrator for structured grids
#
# Performs X→Y→Z→Z→Y→X operator splitting using the v2 state types.
# Each directional sweep uses the kernels from the advection scheme
# (RussellLernerAdvection, PPMAdvection) via dispatch.
# ---------------------------------------------------------------------------

using KernelAbstractions: get_backend, synchronize

"""
    AdvectionWorkspace{FT, A3, V1}

Pre-allocated buffers for mass-flux Strang splitting.
Eliminates all array allocations from the inner time-stepping loop.

# Fields
- `rm_buf`, `m_buf` — double-buffer targets for tracer and air mass
- `cluster_sizes` — per-latitude reduced-grid clustering (1 = uniform)
"""
struct AdvectionWorkspace{FT, A3 <: AbstractArray{FT,3}, V1 <: AbstractVector{Int32}}
    rm_buf        :: A3
    m_buf         :: A3
    cluster_sizes :: V1
end

function AdvectionWorkspace(m::AbstractArray{FT,3};
                            cluster_sizes_cpu::Union{Nothing, Vector{Int32}} = nothing) where FT
    Nx, Ny, Nz = size(m)
    cs_cpu = cluster_sizes_cpu !== nothing ? cluster_sizes_cpu : ones(Int32, Ny)
    cs_dev = similar(m, Int32, Ny)
    copyto!(cs_dev, cs_cpu)
    AdvectionWorkspace{FT, typeof(m), typeof(cs_dev)}(
        similar(m), similar(m), cs_dev)
end

# ---------------------------------------------------------------------------
# Directional sweep dispatch
# ---------------------------------------------------------------------------

"""
    sweep_x!(rm, m, am, scheme, ws)

Single x-sweep: updates rm and m in-place using double buffering.
"""
function sweep_x!(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                  am::AbstractArray{FT,3},
                  scheme::RussellLernerAdvection,
                  ws::AdvectionWorkspace{FT}) where FT
    backend = get_backend(m)
    Nx = size(m, 1)
    kernel! = _rl_x_kernel!(backend, 256)
    kernel!(ws.rm_buf, rm, ws.m_buf, m, am,
            Int32(Nx), ws.cluster_sizes, scheme.use_limiter;
            ndrange=size(m))
    synchronize(backend)
    copyto!(rm, ws.rm_buf)
    copyto!(m,  ws.m_buf)
    return nothing
end

"""
    sweep_y!(rm, m, bm, scheme, ws)

Single y-sweep: updates rm and m in-place using double buffering.
"""
function sweep_y!(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                  bm::AbstractArray{FT,3},
                  scheme::RussellLernerAdvection,
                  ws::AdvectionWorkspace{FT}) where FT
    backend = get_backend(m)
    Ny = size(m, 2)
    kernel! = _rl_y_kernel!(backend, 256)
    kernel!(ws.rm_buf, rm, ws.m_buf, m, bm,
            Int32(Ny), scheme.use_limiter;
            ndrange=size(m))
    synchronize(backend)
    copyto!(rm, ws.rm_buf)
    copyto!(m,  ws.m_buf)
    return nothing
end

"""
    sweep_z!(rm, m, cm, scheme, ws)

Single z-sweep: updates rm and m in-place using double buffering.
"""
function sweep_z!(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                  cm::AbstractArray{FT,3},
                  scheme::RussellLernerAdvection,
                  ws::AdvectionWorkspace{FT}) where FT
    backend = get_backend(m)
    Nz = size(m, 3)
    kernel! = _rl_z_kernel!(backend, 256)
    kernel!(ws.rm_buf, rm, ws.m_buf, m, cm,
            Int32(Nz), scheme.use_limiter;
            ndrange=size(m))
    synchronize(backend)
    copyto!(rm, ws.rm_buf)
    copyto!(m,  ws.m_buf)
    return nothing
end

# ---------------------------------------------------------------------------
# Strang splitting: X→Y→Z→Z→Y→X
# ---------------------------------------------------------------------------

"""
    strang_split!(state::CellState, fluxes::StructuredFaceFluxState{DryMassFluxBasis},
                  grid::AtmosGrid{<:AbstractStructuredMesh},
                  scheme::AbstractAdvection;
                  workspace::AdvectionWorkspace)

Perform one full Strang-split advection step on a structured mesh.

Operates on each tracer independently, restoring air mass between tracers.
The X→Y→Z→Z→Y→X sequence is the TM5-standard Strang splitting.

Only accepts `DryMassFluxBasis` fluxes — passing moist fluxes is a type error.
"""
function strang_split!(state::CellState, fluxes::StructuredFaceFluxState{DryMassFluxBasis},
                       grid::AtmosGrid{<:AbstractStructuredMesh},
                       scheme::AbstractAdvection;
                       workspace::AdvectionWorkspace)
    m = state.air_dry_mass
    am, bm, cm = fluxes.am, fluxes.bm, fluxes.cm

    n_tr = length(state.tracers)
    m_save = n_tr > 1 ? similar(m) : m
    if n_tr > 1
        copyto!(m_save, m)
    end

    for (idx, (name, rm_tracer)) in enumerate(pairs(state.tracers))
        if idx > 1
            copyto!(m, m_save)
        end

        # X → Y → Z → Z → Y → X
        sweep_x!(rm_tracer, m, am, scheme, workspace)
        sweep_y!(rm_tracer, m, bm, scheme, workspace)
        sweep_z!(rm_tracer, m, cm, scheme, workspace)
        sweep_z!(rm_tracer, m, cm, scheme, workspace)
        sweep_y!(rm_tracer, m, bm, scheme, workspace)
        sweep_x!(rm_tracer, m, am, scheme, workspace)
    end

    return nothing
end

"""
    apply!(state::CellState, fluxes::StructuredFaceFluxState{DryMassFluxBasis},
           grid::AtmosGrid{<:AbstractStructuredMesh},
           scheme::AbstractAdvection, dt;
           workspace::AdvectionWorkspace)

Structured-mesh advection entry point. Delegates to `strang_split!`.
`dt` is informational (fluxes are already scaled to the half-timestep).

Only accepts `DryMassFluxBasis` fluxes — passing moist fluxes is a type error.
"""
function apply!(state::CellState, fluxes::StructuredFaceFluxState{DryMassFluxBasis},
                grid::AtmosGrid{<:AbstractStructuredMesh},
                scheme::AbstractAdvection, dt;
                workspace::AdvectionWorkspace)
    strang_split!(state, fluxes, grid, scheme; workspace=workspace)
    return nothing
end

export AdvectionWorkspace, strang_split!, sweep_x!, sweep_y!, sweep_z!
