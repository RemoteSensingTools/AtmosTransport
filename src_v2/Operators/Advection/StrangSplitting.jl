# ---------------------------------------------------------------------------
# Strang splitting orchestrator for structured grids
#
# Performs X→Y→Z→Z→Y→X operator splitting using the v2 state types.
# Each directional sweep uses the kernels from the advection scheme
# (RussellLernerAdvection, PPMAdvection) via dispatch.
# ---------------------------------------------------------------------------

using KernelAbstractions: get_backend, synchronize

"""
    AdvectionWorkspace{FT, A, V1}

Pre-allocated buffers for mass-flux Strang splitting.
Eliminates all array allocations from the inner time-stepping loop.

# Fields
- `rm_buf`, `m_buf` — double-buffer targets for tracer and air mass
- `cluster_sizes` — per-latitude reduced-grid clustering (1 = uniform) for
  structured meshes; empty for face-connected meshes
"""
struct AdvectionWorkspace{FT, A <: AbstractArray{FT}, V1 <: AbstractVector{Int32}}
    rm_buf        :: A
    m_buf         :: A
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

function AdvectionWorkspace(m::AbstractArray{FT,2};
                            cluster_sizes_cpu::Union{Nothing, Vector{Int32}} = nothing) where FT
    cs_dev = Int32[]
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
                  scheme::UpwindAdvection,
                  ws::AdvectionWorkspace{FT}) where FT
    backend = get_backend(m)
    Nx = size(m, 1)
    kernel! = _upwind_x_kernel!(backend, 256)
    kernel!(ws.rm_buf, rm, ws.m_buf, m, am,
            Int32(Nx), ws.cluster_sizes;
            ndrange=size(m))
    synchronize(backend)
    copyto!(rm, ws.rm_buf)
    copyto!(m,  ws.m_buf)
    return nothing
end

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
                  scheme::UpwindAdvection,
                  ws::AdvectionWorkspace{FT}) where FT
    backend = get_backend(m)
    Ny = size(m, 2)
    kernel! = _upwind_y_kernel!(backend, 256)
    kernel!(ws.rm_buf, rm, ws.m_buf, m, bm,
            Int32(Ny);
            ndrange=size(m))
    synchronize(backend)
    copyto!(rm, ws.rm_buf)
    copyto!(m,  ws.m_buf)
    return nothing
end

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
                  scheme::UpwindAdvection,
                  ws::AdvectionWorkspace{FT}) where FT
    backend = get_backend(m)
    Nz = size(m, 3)
    kernel! = _upwind_z_kernel!(backend, 256)
    kernel!(ws.rm_buf, rm, ws.m_buf, m, cm,
            Int32(Nz);
            ndrange=size(m))
    synchronize(backend)
    copyto!(rm, ws.rm_buf)
    copyto!(m,  ws.m_buf)
    return nothing
end

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

@inline function _horizontal_face_tendency!(rm_new::AbstractArray{FT,2},
                                            rm::AbstractArray{FT,2},
                                            m_new::AbstractArray{FT,2},
                                            m::AbstractArray{FT,2},
                                            horizontal_flux::AbstractArray{FT,2},
                                            mesh::AbstractHorizontalMesh) where FT
    copyto!(rm_new, rm)
    copyto!(m_new, m)
    m_floor = eps(FT)
    nface = nfaces(mesh)
    Nz = size(m, 2)

    @inbounds for k in 1:Nz
        for f in 1:nface
            left, right = face_cells(mesh, f)
            if left > 0 && right > 0
                flux = horizontal_flux[f, k]
                c_left = rm[left, k] / max(m[left, k], m_floor)
                c_right = rm[right, k] / max(m[right, k], m_floor)
                tracer_flux = _upwind_face_flux(flux, c_left, c_right)

                rm_new[left,  k] -= tracer_flux
                rm_new[right, k] += tracer_flux
                m_new[left,   k] -= flux
                m_new[right,  k] += flux
            end
        end
    end

    return nothing
end

@inline function _vertical_column_tendency!(rm_new::AbstractArray{FT,2},
                                            rm::AbstractArray{FT,2},
                                            m_new::AbstractArray{FT,2},
                                            m::AbstractArray{FT,2},
                                            cm::AbstractArray{FT,2}) where FT
    copyto!(rm_new, rm)
    copyto!(m_new, m)
    m_floor = eps(FT)
    nc = size(m, 1)
    Nz = size(m, 2)

    @inbounds for k in 1:Nz
        for c in 1:nc
            flux_t = ifelse(k > 1,
                            _upwind_face_flux(cm[c, k],
                                              rm[c, k - 1] / max(m[c, k - 1], m_floor),
                                              rm[c, k]     / max(m[c, k],     m_floor)),
                            zero(FT))

            flux_b = ifelse(k < Nz,
                            _upwind_face_flux(cm[c, k + 1],
                                              rm[c, k]     / max(m[c, k],     m_floor),
                                              rm[c, k + 1] / max(m[c, k + 1], m_floor)),
                            zero(FT))

            rm_new[c, k] = rm[c, k] + flux_t - flux_b
            m_new[c, k] = m[c, k] + cm[c, k] - cm[c, k + 1]
        end
    end

    return nothing
end

function sweep_horizontal!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                           horizontal_flux::AbstractArray{FT,2},
                           mesh::AbstractHorizontalMesh,
                           scheme::UpwindAdvection,
                           ws::AdvectionWorkspace{FT}) where FT
    _horizontal_face_tendency!(ws.rm_buf, rm, ws.m_buf, m, horizontal_flux, mesh)
    copyto!(rm, ws.rm_buf)
    copyto!(m, ws.m_buf)
    return nothing
end

function sweep_vertical!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                         cm::AbstractArray{FT,2},
                         scheme::UpwindAdvection,
                         ws::AdvectionWorkspace{FT}) where FT
    _vertical_column_tendency!(ws.rm_buf, rm, ws.m_buf, m, cm)
    copyto!(rm, ws.rm_buf)
    copyto!(m, ws.m_buf)
    return nothing
end

# ---------------------------------------------------------------------------
# Strang splitting: X→Y→Z→Z→Y→X
# ---------------------------------------------------------------------------

"""
    strang_split!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
                  grid::AtmosGrid{<:AbstractStructuredMesh},
                  scheme::AbstractAdvection;
                  workspace::AdvectionWorkspace)

Perform one full Strang-split advection step on a structured mesh.

Operates on each tracer independently, restoring air mass between tracers.
The X→Y→Z→Z→Y→X sequence is the default structured Strang splitting.
"""
function strang_split!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
                       grid::AtmosGrid{<:AbstractStructuredMesh},
                       scheme::AbstractAdvection;
                       workspace::AdvectionWorkspace) where {B <: AbstractMassBasis}
    m = state.air_mass
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
    apply!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
           grid::AtmosGrid{<:AbstractStructuredMesh},
           scheme::AbstractAdvection, dt;
           workspace::AdvectionWorkspace)

Structured-mesh advection entry point. Delegates to `strang_split!`.
`dt` is informational (fluxes are already scaled to the half-timestep).
"""
function apply!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
                grid::AtmosGrid{<:AbstractStructuredMesh},
                scheme::AbstractAdvection, dt;
                workspace::AdvectionWorkspace) where {B <: AbstractMassBasis}
    strang_split!(state, fluxes, grid, scheme; workspace=workspace)
    return nothing
end

function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                grid::AtmosGrid{<:AbstractHorizontalMesh},
                scheme::FirstOrderUpwindAdvection, dt;
                workspace::AdvectionWorkspace) where {B <: AbstractMassBasis}
    m = state.air_mass
    hflux, cm = fluxes.horizontal_flux, fluxes.cm

    n_tr = length(state.tracers)
    m_save = n_tr > 1 ? similar(m) : m
    if n_tr > 1
        copyto!(m_save, m)
    end

    for (idx, (_, rm_tracer)) in enumerate(pairs(state.tracers))
        if idx > 1
            copyto!(m, m_save)
        end

        sweep_horizontal!(rm_tracer, m, hflux, grid.horizontal, scheme, workspace)
        sweep_vertical!(rm_tracer, m, cm, scheme, workspace)
        sweep_vertical!(rm_tracer, m, cm, scheme, workspace)
        sweep_horizontal!(rm_tracer, m, hflux, grid.horizontal, scheme, workspace)
    end

    return nothing
end

function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                grid::AtmosGrid{<:AbstractHorizontalMesh},
                scheme::AbstractLinearReconstruction, dt; kwargs...) where {B <: AbstractMassBasis}
    throw(ArgumentError("Face-connected advection does not implement linear-reconstruction schemes yet for $(typeof(grid.horizontal))"))
end

function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                grid::AtmosGrid{<:AbstractHorizontalMesh},
                scheme::AbstractQuadraticReconstruction, dt; kwargs...) where {B <: AbstractMassBasis}
    throw(ArgumentError("Face-connected advection does not implement quadratic-reconstruction schemes yet for $(typeof(grid.horizontal))"))
end

function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                grid::AtmosGrid{<:AbstractHorizontalMesh},
                scheme::AbstractAdvection, dt; kwargs...) where {B <: AbstractMassBasis}
    throw(ArgumentError("Face-connected advection currently implements only constant-reconstruction upwind ($(UpwindAdvection)) for $(typeof(grid.horizontal))"))
end

export AdvectionWorkspace, strang_split!
