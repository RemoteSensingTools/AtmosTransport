# ---------------------------------------------------------------------------
# Strang splitting orchestrator for structured grids
#
# Performs dimensionally-split advection using the Strang (1968) symmetric
# splitting sequence:  X → Y → Z → Z → Y → X
#
# This second-order splitting eliminates the first-order directional bias
# that would arise from a simple X → Y → Z sequence.  Each half of the
# palindrome advances the state by half a timestep in each direction,
# yielding a full timestep with O(Δt²) splitting error.
#
# Directional sweep methods are generated via @eval to eliminate
# per-direction boilerplate — the only things that vary between x/y/z
# sweeps are the kernel function and the dimension index for N.
#
# References
# ----------
# - Strang (1968), "On the construction and comparison of difference
#   schemes", SIAM J. Numer. Anal., 5:506–517.
# - Russell & Lerner (1981), "A new finite-differencing scheme for the
#   tracer transport equation", J. Appl. Meteor., 20:1483–1498.
#   → Strang splitting of slopes advection in TM5.
# ---------------------------------------------------------------------------

using KernelAbstractions: get_backend, synchronize, @kernel, @index, @Const, @atomic

# =========================================================================
# AdvectionWorkspace — pre-allocated double buffers
# =========================================================================

"""
    AdvectionWorkspace{FT, A, V1, A4}

Pre-allocated buffers for mass-flux Strang splitting.  Eliminates all
array allocations from the inner time-stepping loop.

The workspace provides TWO complete buffer pairs (`(rm_A, m_A)` and
`(rm_B, m_B)`) used as ping-pong source/destination by
[`strang_split!`](@ref). Each directional sweep reads from one pair and
writes to the other; the palindrome's six sweeps flip parity an even
number of times, so the caller's home arrays receive the final result
naturally.

Kernels always write to a DIFFERENT array than they read from — this is
the double-buffer contract, and in-place updates would violate the
stencil's read-before-write assumption and break mass conservation by
~10% per step.

# Fields
- `rm_A::A`, `rm_B::A` — 3D tracer-mass ping-pong pair (same size as `rm`)
- `m_A::A`, `m_B::A`   — 3D air-mass ping-pong pair (same size as `m`)
- `m_save::A` — backup of `m` for the per-tracer multi-tracer protocol
- `cfl_scratch_m::A`, `cfl_scratch_rm::A` — preserved scratch pair from
  the pre-plan-13 evolving-mass CPU CFL pilot. No longer referenced by
  the unified static pilot and candidate for removal in a follow-up.
- `cluster_sizes::V1` — per-latitude clustering factors for reduced grids
  (`Int32[Ny]`; all ones for uniform grids; empty for face-indexed meshes)
- `face_left::V1`, `face_right::V1` — face connectivity for face-indexed meshes
- `rm_4d_A::A4`, `rm_4d_B::A4` — 4D tracer-mass ping-pong pair for the
  multi-tracer fused path (`(Nx, Ny, Nz, Nt)`). Both are allocated to
  size 0×0×0×0 when `n_tracers == 0`.

# Constructors

    AdvectionWorkspace(m::AbstractArray{FT,3}; cluster_sizes_cpu=nothing, n_tracers=0)

Create workspace for a 3D structured grid, allocating both buffer pairs
matching the size of `m`.  If `cluster_sizes_cpu` is nothing, defaults to
uniform (all ones).

    AdvectionWorkspace(m::AbstractArray{FT,2}; cluster_sizes_cpu=nothing, mesh=nothing)

Create workspace for a 2D face-indexed grid (cell × level layout).
"""
struct AdvectionWorkspace{FT, A <: AbstractArray{FT}, V1 <: AbstractVector{Int32}, A4}
    rm_A           :: A
    m_A            :: A
    rm_B           :: A
    m_B            :: A
    m_save         :: A       # backup of m for multi-tracer strang_split!
    cfl_scratch_m  :: A       # dead: pre-plan-13 evolving-mass pilot scratch
    cfl_scratch_rm :: A       # dead: pre-plan-13 evolving-mass pilot scratch
    cluster_sizes  :: V1
    face_left      :: V1
    face_right     :: V1
    rm_4d_A        :: A4
    rm_4d_B        :: A4
end

function _face_connectivity_vectors(mesh::AbstractHorizontalMesh)
    left = Vector{Int32}(undef, nfaces(mesh))
    right = Vector{Int32}(undef, nfaces(mesh))
    @inbounds for f in eachindex(left)
        l, r = face_cells(mesh, f)
        left[f] = Int32(l)
        right[f] = Int32(r)
    end
    return left, right
end

function AdvectionWorkspace(m::AbstractArray{FT,3};
                            cluster_sizes_cpu::Union{Nothing, Vector{Int32}} = nothing,
                            n_tracers::Int = 0) where FT
    Nx, Ny, Nz = size(m)
    cs_cpu = cluster_sizes_cpu !== nothing ? cluster_sizes_cpu : ones(Int32, Ny)
    cs_dev = similar(m, Int32, Ny)
    copyto!(cs_dev, cs_cpu)
    face_left = similar(m, Int32, 0)
    face_right = similar(m, Int32, 0)
    rm_4d_A = n_tracers > 0 ? similar(m, Nx, Ny, Nz, n_tracers) : similar(m, 0, 0, 0, 0)
    rm_4d_B = n_tracers > 0 ? similar(m, Nx, Ny, Nz, n_tracers) : similar(m, 0, 0, 0, 0)
    AdvectionWorkspace{FT, typeof(m), typeof(cs_dev), typeof(rm_4d_A)}(
        similar(m), similar(m),                       # rm_A, m_A
        similar(m), similar(m),                       # rm_B, m_B
        similar(m),                                   # m_save
        similar(m), similar(m),                       # cfl_scratch_m, cfl_scratch_rm
        cs_dev, face_left, face_right,
        rm_4d_A, rm_4d_B)
end

function AdvectionWorkspace(m::AbstractArray{FT,2};
                            cluster_sizes_cpu::Union{Nothing, Vector{Int32}} = nothing,
                            mesh::Union{Nothing, AbstractHorizontalMesh} = nothing) where FT
    cs_dev = similar(m, Int32, 0)
    if mesh === nothing
        face_left = similar(m, Int32, 0)
        face_right = similar(m, Int32, 0)
    else
        left_cpu, right_cpu = _face_connectivity_vectors(mesh)
        face_left = similar(m, Int32, length(left_cpu))
        face_right = similar(m, Int32, length(right_cpu))
        copyto!(face_left, left_cpu)
        copyto!(face_right, right_cpu)
    end
    rm_4d_A = similar(m, 0, 0)
    rm_4d_B = similar(m, 0, 0)
    AdvectionWorkspace{FT, typeof(m), typeof(cs_dev), typeof(rm_4d_A)}(
        similar(m), similar(m),                       # rm_A, m_A
        similar(m), similar(m),                       # rm_B, m_B
        similar(m),                                   # m_save
        similar(m), similar(m),                       # cfl_scratch_m, cfl_scratch_rm
        cs_dev, face_left, face_right,
        rm_4d_A, rm_4d_B)
end

function Adapt.adapt_structure(to, ws::AdvectionWorkspace{FT}) where {FT}
    rm_A           = Adapt.adapt(to, getfield(ws, :rm_A))
    m_A            = Adapt.adapt(to, getfield(ws, :m_A))
    rm_B           = Adapt.adapt(to, getfield(ws, :rm_B))
    m_B            = Adapt.adapt(to, getfield(ws, :m_B))
    m_save         = Adapt.adapt(to, getfield(ws, :m_save))
    cfl_scratch_m  = Adapt.adapt(to, getfield(ws, :cfl_scratch_m))
    cfl_scratch_rm = Adapt.adapt(to, getfield(ws, :cfl_scratch_rm))
    cluster_sizes  = Adapt.adapt(to, getfield(ws, :cluster_sizes))
    face_left      = Adapt.adapt(to, getfield(ws, :face_left))
    face_right     = Adapt.adapt(to, getfield(ws, :face_right))
    rm_4d_A        = Adapt.adapt(to, getfield(ws, :rm_4d_A))
    rm_4d_B        = Adapt.adapt(to, getfield(ws, :rm_4d_B))
    return AdvectionWorkspace{FT, typeof(rm_A), typeof(cluster_sizes), typeof(rm_4d_A)}(
        rm_A, m_A, rm_B, m_B, m_save,
        cfl_scratch_m, cfl_scratch_rm,
        cluster_sizes, face_left, face_right,
        rm_4d_A, rm_4d_B)
end

# =========================================================================
# Generic directional sweeps — generated via @eval
# =========================================================================
#
# Ping-pong contract (post-refactor):
#   - The 7-arg form `sweep_x!(rm_in, rm_out, m_in, m_out, flux, scheme, ws)`
#     is the KERNEL entry point. It reads from (rm_in, m_in), writes to
#     (rm_out, m_out), synchronizes the backend, and returns. NO copyto!.
#     `strang_split!` uses this form and tracks parity so the output of
#     one sweep becomes the input of the next.
#   - The 5-arg form `sweep_x!(rm, m, flux, scheme, ws)` is a backward-
#     compat wrapper used by callers that bypass the orchestrator (LinRood,
#     CubedSphereStrang, direct tests). It kernels into `(ws.rm_B, ws.m_B)`
#     then copies back to `(rm, m)`, preserving the pre-refactor semantics.
#   - The 6-arg / 8-arg `flux_scale` variants follow the same pattern.
#
# The three sweeps differ ONLY in:
#   - Which kernel function to call (_xsweep_kernel!, etc.)
#   - Which dimension to use for N (1=Nx, 2=Ny, 3=Nz)

for (sweep_fn, kernel_fn, dim) in (
    (:sweep_x!, :_xsweep_kernel!, 1),
    (:sweep_y!, :_ysweep_kernel!, 2),
    (:sweep_z!, :_zsweep_kernel!, 3),
)
    @eval begin
        # Ping-pong entry point: writes (rm_out, m_out), reads (rm_in, m_in).
        function $sweep_fn(rm_in::AbstractArray{FT,3},  rm_out::AbstractArray{FT,3},
                            m_in::AbstractArray{FT,3},   m_out::AbstractArray{FT,3},
                            flux::AbstractArray{FT,3},
                            scheme::AbstractAdvectionScheme,
                            ws::AdvectionWorkspace{FT}) where FT
            backend = get_backend(m_in)
            kernel! = $kernel_fn(backend, 256)
            kernel!(rm_out, rm_in, m_out, m_in, flux, scheme,
                    Int32(size(m_in, $dim)), one(FT);
                    ndrange=size(m_in))
            synchronize(backend)
            return nothing
        end

        """
            $($sweep_fn)(rm, m, flux, scheme, ws)

        Backward-compat wrapper: write into the workspace B pair, then
        copy back to `rm` and `m`. New callers should use the 7-arg
        ping-pong form instead.
        """
        function $sweep_fn(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                           flux::AbstractArray{FT,3},
                           scheme::AbstractAdvectionScheme,
                           ws::AdvectionWorkspace{FT}) where FT
            $sweep_fn(rm, ws.rm_B, m, ws.m_B, flux, scheme, ws)
            copyto!(rm, ws.rm_B)
            copyto!(m,  ws.m_B)
            return nothing
        end
    end
end

"""
    _horizontal_face_atomic_kernel!(rm_new, rm, m_new, m, horizontal_flux,
                                     face_left, face_right, scheme, flux_scale)

Face-indexed horizontal advection kernel for **unstructured meshes** (e.g.
reduced-Gaussian). Each work item processes one face `f` at level `k`.

The face topology is defined by `face_left[f]` and `face_right[f]`:
- Both > 0: interior face connecting two cells. Flux is accumulated via
  `@atomic` additions to rm_new/m_new at both left and right cells
  (race-safe for GPU workgroups that share cells across faces).
- `face_left[f] == 0`: south pole boundary stub (only right cell exists).
  **Skipped** — no mass enters or leaves through the pole singularity.
- `face_right[f] == 0`: north pole boundary stub. **Skipped**.

Flux sign convention: positive `horizontal_flux[f, k]` = mass moving from
`left` to `right`. The tracer flux `_hface_tracer_flux` uses the donor
cell's mixing ratio (upwind) or higher-order reconstruction.
"""
@kernel function _horizontal_face_atomic_kernel!(rm_new, @Const(rm), m_new, @Const(m),
                                                 @Const(horizontal_flux),
                                                 @Const(face_left), @Const(face_right),
                                                 scheme, flux_scale)
    f, k = @index(Global, NTuple)
    @inbounds begin
        left = Int(face_left[f])
        right = Int(face_right[f])
        # Skip boundary stubs: left=0 (south pole) or right=0 (north pole)
        if left > 0 && right > 0
            flux = flux_scale * horizontal_flux[f, k]
            tracer_flux = _hface_tracer_flux(rm, m, flux, left, right, k, scheme)
            # Atomic accumulation: multiple faces share the same cell
            @atomic rm_new[left,  k] += -tracer_flux   # tracer leaving left cell
            @atomic rm_new[right, k] +=  tracer_flux   # tracer entering right cell
            @atomic m_new[left,   k] += -flux           # mass leaving left cell
            @atomic m_new[right,  k] +=  flux           # mass entering right cell
        end
    end
end

"""
    _vertical_face_kernel!(rm_new, rm, m_new, m, cm, scheme, flux_scale, Nz)

Vertical advection kernel for face-indexed meshes. Each work item processes
one cell `c` at level `k`.

Vertical boundaries: `k=1` is TOA (no flux above), `k=Nz` is deepest level
(no flux below). `cm[c, k]` is the vertical mass flux through the TOP face
of cell `(c, k)`. Positive cm = downward (toward surface).
"""
@kernel function _vertical_face_kernel!(rm_new, @Const(rm), m_new, @Const(m),
                                        @Const(cm), scheme, flux_scale, Nz)
    c, k = @index(Global, NTuple)
    FT = eltype(rm)
    @inbounds begin
        # NOTE: `?:` (branch) is used here intentionally, NOT `ifelse`.
        # `ifelse` evaluates BOTH branches, but `_vface_tracer_flux` reads
        # `rm[c, k±1]` which is out-of-bounds at the boundaries:
        #   k=1:  k-1 = 0     → rm[c, 0] is OOB
        #   k=Nz: k+1 = Nz+1 → rm[c, Nz+1] is OOB (rm is nc×Nz, not nc×Nz+1)
        # The `?:` branch avoids evaluating the OOB branch entirely.
        # Warp divergence is not a concern because `k` is typically constant
        # within a warp (KA maps (c, k) with c as the fast dimension).
        # k=1: TOA boundary → flux_t = 0 (no flux above top level)
        flux_t = k > 1  ? _vface_tracer_flux(rm, m, flux_scale * cm[c, k],     c, k - 1, k,     scheme) : zero(FT)
        # k=Nz: surface boundary → flux_b = 0 (no flux below bottom level)
        flux_b = k < Nz ? _vface_tracer_flux(rm, m, flux_scale * cm[c, k + 1], c, k,     k + 1, scheme) : zero(FT)
        rm_new[c, k] = rm[c, k] + flux_t - flux_b
        m_new[c, k]  = m[c, k]  + flux_scale * cm[c, k] - flux_scale * cm[c, k + 1]
    end
end

function _sweep_horizontal_face_gpu!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                     horizontal_flux::AbstractArray{FT,2},
                                     scheme::UpwindScheme,
                                     ws::AdvectionWorkspace{FT},
                                     flux_scale::FT) where FT
    isempty(ws.face_left) &&
        throw(ArgumentError("face-indexed GPU sweep requires mesh connectivity in AdvectionWorkspace"))
    backend = get_backend(rm)
    copyto!(ws.rm_A, rm)
    copyto!(ws.m_A, m)
    kernel! = _horizontal_face_atomic_kernel!(backend, 256)
    kernel!(ws.rm_A, rm, ws.m_A, m, horizontal_flux, ws.face_left, ws.face_right, scheme, flux_scale;
            ndrange=size(horizontal_flux))
    synchronize(backend)
    copyto!(rm, ws.rm_A)
    copyto!(m, ws.m_A)
    return nothing
end

function _sweep_vertical_face_gpu!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                   cm::AbstractArray{FT,2},
                                   scheme::UpwindScheme,
                                   ws::AdvectionWorkspace{FT},
                                   flux_scale::FT) where FT
    backend = get_backend(rm)
    kernel! = _vertical_face_kernel!(backend, 256)
    kernel!(ws.rm_A, rm, ws.m_A, m, cm, scheme, flux_scale, Int32(size(m, 2));
            ndrange=size(m))
    synchronize(backend)
    copyto!(rm, ws.rm_A)
    copyto!(m, ws.m_A)
    return nothing
end

# Additional structured sweep overloads with explicit flux scaling.
# These are used by the CFL-based subcycling wrappers to reapply the same
# directional forcing in smaller conservative pieces. Same ping-pong
# contract as the one(FT) variants above: the 8-arg form does the kernel
# launch only (no copyto!); the 6-arg form is a backward-compat wrapper.
for (sweep_fn, kernel_fn, dim) in (
    (:sweep_x!, :_xsweep_kernel!, 1),
    (:sweep_y!, :_ysweep_kernel!, 2),
    (:sweep_z!, :_zsweep_kernel!, 3),
)
    @eval begin
        # Ping-pong entry point with explicit flux scale.
        function $sweep_fn(rm_in::AbstractArray{FT,3},  rm_out::AbstractArray{FT,3},
                            m_in::AbstractArray{FT,3},   m_out::AbstractArray{FT,3},
                            flux::AbstractArray{FT,3},
                            scheme::AbstractAdvectionScheme,
                            ws::AdvectionWorkspace{FT},
                            flux_scale::FT) where FT
            backend = get_backend(m_in)
            kernel! = $kernel_fn(backend, 256)
            kernel!(rm_out, rm_in, m_out, m_in, flux, scheme,
                    Int32(size(m_in, $dim)), flux_scale;
                    ndrange=size(m_in))
            synchronize(backend)
            return nothing
        end

        # Backward-compat 6-arg wrapper.
        function $sweep_fn(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                           flux::AbstractArray{FT,3},
                           scheme::AbstractAdvectionScheme,
                           ws::AdvectionWorkspace{FT},
                           flux_scale::FT) where FT
            $sweep_fn(rm, ws.rm_B, m, ws.m_B, flux, scheme, ws, flux_scale)
            copyto!(rm, ws.rm_B)
            copyto!(m,  ws.m_B)
            return nothing
        end
    end
end

function sweep_horizontal!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                           horizontal_flux::AbstractArray{FT,2},
                           mesh::AbstractHorizontalMesh,
                           scheme::UpwindScheme,
                           ws::AdvectionWorkspace{FT}) where FT
    if rm isa Array
        _horizontal_face_tendency!(ws.rm_A, rm, ws.m_A, m, horizontal_flux, mesh, scheme, one(FT))
        copyto!(rm, ws.rm_A)
        copyto!(m, ws.m_A)
    else
        _sweep_horizontal_face_gpu!(rm, m, horizontal_flux, scheme, ws, one(FT))
    end
    return nothing
end

function sweep_horizontal!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                           horizontal_flux::AbstractArray{FT,2},
                           mesh::AbstractHorizontalMesh,
                           scheme::UpwindScheme,
                           ws::AdvectionWorkspace{FT},
                           flux_scale::FT) where FT
    if rm isa Array
        _horizontal_face_tendency!(ws.rm_A, rm, ws.m_A, m, horizontal_flux, mesh, scheme, flux_scale)
        copyto!(rm, ws.rm_A)
        copyto!(m, ws.m_A)
    else
        _sweep_horizontal_face_gpu!(rm, m, horizontal_flux, scheme, ws, flux_scale)
    end
    return nothing
end

function sweep_vertical!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                         cm::AbstractArray{FT,2},
                         scheme::UpwindScheme,
                         ws::AdvectionWorkspace{FT}) where FT
    if rm isa Array
        _vertical_column_tendency!(ws.rm_A, rm, ws.m_A, m, cm, scheme, one(FT))
        copyto!(rm, ws.rm_A)
        copyto!(m, ws.m_A)
    else
        _sweep_vertical_face_gpu!(rm, m, cm, scheme, ws, one(FT))
    end
    return nothing
end

function sweep_vertical!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                         cm::AbstractArray{FT,2},
                         scheme::UpwindScheme,
                         ws::AdvectionWorkspace{FT},
                         flux_scale::FT) where FT
    if rm isa Array
        _vertical_column_tendency!(ws.rm_A, rm, ws.m_A, m, cm, scheme, flux_scale)
        copyto!(rm, ws.rm_A)
        copyto!(m, ws.m_A)
    else
        _sweep_vertical_face_gpu!(rm, m, cm, scheme, ws, flux_scale)
    end
    return nothing
end

# =========================================================================
# Face-indexed tendency functions
# =========================================================================
#
# For unstructured grids (face-connected meshes like ReducedGaussian),
# advection operates on 2D arrays (cell, level) with face connectivity
# provided by the mesh object.
#
# Two versions exist:
# - New hierarchy: accepts a `scheme::AbstractAdvectionScheme` argument
# - Legacy: no scheme argument, uses hardcoded upwind flux

"""
    _horizontal_face_tendency!(rm_new, rm, m_new, m, horizontal_flux, mesh, scheme)

Compute horizontal advection tendency on a face-indexed mesh using
the new scheme dispatch.

Iterates over all faces in the mesh, computing the face tracer flux
via `_hface_tracer_flux` (dispatches on `scheme` type) and accumulating
the flux divergence into `rm_new` and `m_new`.

# Mass conservation
The flux through each face is added to one cell and subtracted from
the other, so ``\\sum_c r_{m,\\text{new}}[c,k] = \\sum_c r_m[c,k]``
holds exactly for each level `k`.
"""
@inline function _horizontal_face_tendency!(rm_new::AbstractArray{FT,2},
                                            rm::AbstractArray{FT,2},
                                            m_new::AbstractArray{FT,2},
                                            m::AbstractArray{FT,2},
                                            horizontal_flux::AbstractArray{FT,2},
                                            mesh::AbstractHorizontalMesh,
                                            scheme::AbstractAdvectionScheme,
                                            flux_scale::FT = one(FT)) where FT
    copyto!(rm_new, rm)
    copyto!(m_new, m)
    nface = nfaces(mesh)
    Nz = size(m, 2)

    @inbounds for k in 1:Nz
        for f in 1:nface
            left, right = face_cells(mesh, f)
            if left > 0 && right > 0
                flux = flux_scale * horizontal_flux[f, k]
                tracer_flux = _hface_tracer_flux(rm, m, flux, left, right, k, scheme)
                rm_new[left,  k] -= tracer_flux
                rm_new[right, k] += tracer_flux
                m_new[left,   k] -= flux
                m_new[right,  k] += flux
            end
        end
    end

    return nothing
end

"""
    _vertical_column_tendency!(rm_new, rm, m_new, m, cm, scheme)

Compute vertical advection tendency for face-indexed grids using
the new scheme dispatch.

Iterates level by level, computing the vertical face tracer flux via
`_vface_tracer_flux` (dispatches on `scheme` type).  Closed boundaries
at TOA (k=1) and surface (k=Nz) are enforced explicitly with branch guards.
"""
@inline function _vertical_column_tendency!(rm_new::AbstractArray{FT,2},
                                            rm::AbstractArray{FT,2},
                                            m_new::AbstractArray{FT,2},
                                            m::AbstractArray{FT,2},
                                            cm::AbstractArray{FT,2},
                                            scheme::AbstractAdvectionScheme,
                                            flux_scale::FT = one(FT)) where FT
    copyto!(rm_new, rm)
    copyto!(m_new, m)
    nc = size(m, 1)
    Nz = size(m, 2)

    @inbounds for k in 1:Nz
        for c in 1:nc
            flux_t = k > 1 ? _vface_tracer_flux(rm, m, flux_scale * cm[c, k], c, k - 1, k, scheme) : zero(FT)
            flux_b = k < Nz ? _vface_tracer_flux(rm, m, flux_scale * cm[c, k + 1], c, k, k + 1, scheme) : zero(FT)
            rm_new[c, k] = rm[c, k] + flux_t - flux_b
            m_new[c, k]  = m[c, k]  + flux_scale * cm[c, k] - flux_scale * cm[c, k + 1]
        end
    end

    return nothing
end

# =========================================================================
# Face-indexed sweep helpers — generated via @eval
# =========================================================================

for (scheme_type, h_args, v_args) in (
    (:AbstractConstantScheme, (:mesh, :scheme), (:scheme,)),
)
    @eval begin
        """
            sweep_horizontal!(rm, m, horizontal_flux, mesh, scheme::$($scheme_type), ws)

        Horizontal advection sweep for face-indexed grids.  Computes the
        face-flux tendency and copies the result from the workspace buffers
        back to `rm` and `m`.
        """
        function sweep_horizontal!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                         horizontal_flux::AbstractArray{FT,2},
                                         mesh::AbstractHorizontalMesh,
                                         scheme::$scheme_type,
                                         ws::AdvectionWorkspace{FT}) where FT
            _horizontal_face_tendency!(ws.rm_A, rm, ws.m_A, m, horizontal_flux, $(h_args...), one(FT))
            copyto!(rm, ws.rm_A)
            copyto!(m, ws.m_A)
            return nothing
        end

        function sweep_horizontal!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                         horizontal_flux::AbstractArray{FT,2},
                                         mesh::AbstractHorizontalMesh,
                                         scheme::$scheme_type,
                                         ws::AdvectionWorkspace{FT},
                                         flux_scale::FT) where FT
            _horizontal_face_tendency!(ws.rm_A, rm, ws.m_A, m, horizontal_flux, $(h_args...), flux_scale)
            copyto!(rm, ws.rm_A)
            copyto!(m, ws.m_A)
            return nothing
        end

        """
            sweep_vertical!(rm, m, cm, scheme::$($scheme_type), ws)

        Vertical advection sweep for face-indexed grids.  Computes the
        vertical face-flux tendency and copies results back.
        """
        function sweep_vertical!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                       cm::AbstractArray{FT,2},
                                       scheme::$scheme_type,
                                       ws::AdvectionWorkspace{FT}) where FT
            _vertical_column_tendency!(ws.rm_A, rm, ws.m_A, m, cm, $(v_args...), one(FT))
            copyto!(rm, ws.rm_A)
            copyto!(m, ws.m_A)
            return nothing
        end

        function sweep_vertical!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                       cm::AbstractArray{FT,2},
                                       scheme::$scheme_type,
                                       ws::AdvectionWorkspace{FT},
                                       flux_scale::FT) where FT
            _vertical_column_tendency!(ws.rm_A, rm, ws.m_A, m, cm, $(v_args...), flux_scale)
            copyto!(rm, ws.rm_A)
            copyto!(m, ws.m_A)
            return nothing
        end
    end
end

# =========================================================================
# Structured-grid CFL subcycling helpers
# =========================================================================

@inline function _subcycling_pass_count(max_cfl::FT, cfl_limit::FT) where FT
    cfl_limit > zero(FT) || throw(ArgumentError("structured advection requires cfl_limit > 0, got $(cfl_limit)"))
    return max(1, ceil(Int, max_cfl / cfl_limit))
end

function _horizontal_face_outgoing_ratio(horizontal_flux::AbstractArray{FT,2},
                                         m::AbstractArray{FT,2},
                                         mesh::AbstractHorizontalMesh) where FT
    nc, Nz = size(m)
    outgoing = zeros(FT, nc)
    max_ratio = zero(FT)

    @inbounds for k in 1:Nz
        outgoing .= zero(FT)
        for f in 1:nfaces(mesh)
            left, right = face_cells(mesh, f)
            if left > 0 && right > 0
                flux = horizontal_flux[f, k]
                if flux >= zero(FT)
                    outgoing[left] += flux
                else
                    outgoing[right] -= flux
                end
            end
        end
        max_ratio = max(max_ratio, maximum(outgoing ./ max.(m[:, k], eps(FT))))
    end

    return max_ratio
end

function _vertical_face_outgoing_ratio(cm::AbstractArray{FT,2},
                                       m::AbstractArray{FT,2}) where FT
    # cm is (nc, Nz+1). `cm[:, k]` is the flux through the TOP face of cell
    # (:, k) (positive = downward = inflow from above). `cm[:, k+1]` is the
    # flux through the BOTTOM face (positive = downward = outflow to below).
    # So per-cell OUTFLOW = upward through top (max(-cm[:,k], 0)) + downward
    # through bottom (max(cm[:,k+1], 0)). Single broadcast over all (:, :)
    # stays on device for GPU callers. Correction of the pre-plan-13 bug
    # that summed inflow, not outflow (GPU and CPU saw CFL half the true
    # value in flows where inflow ≠ outflow).
    Nz = size(m, 2)
    out = max.(.- @view(cm[:, 1:Nz]), zero(FT)) .+ max.(@view(cm[:, 2:Nz+1]), zero(FT))
    return maximum(out ./ max.(m, eps(FT)))
end

# Face-indexed pilots — unified static algorithm (plan 13).
#
# The horizontal path requires mesh connectivity (face_cells) which lives on
# CPU, so device arrays are materialized via Array(...) before the static
# ratio is computed. For realistic problem sizes the transfer is ~1–10 MB and
# happens once per sweep — far cheaper than the old evolving-mass iteration
# that transferred on every pilot pass.
#
# The vertical path is a pure broadcast reduction — stays on device.
function _horizontal_face_subcycling_pass_count(horizontal_flux::AbstractArray{FT,2},
                                                m::AbstractArray{FT,2},
                                                mesh::AbstractHorizontalMesh,
                                                ws::AdvectionWorkspace{FT},
                                                cfl_limit::FT; max_n_sub::Int = 4096) where FT
    isinf(cfl_limit) && return 1
    static_cfl = m isa Array ?
        _horizontal_face_outgoing_ratio(horizontal_flux, m, mesh) :
        _horizontal_face_outgoing_ratio(Array(horizontal_flux), Array(m), mesh)
    n_sub = _subcycling_pass_count(static_cfl, cfl_limit)
    n_sub <= max_n_sub || throw(ArgumentError("face-indexed horizontal subcycling exceeded max_n_sub=$(max_n_sub)"))
    return n_sub
end

function _vertical_face_subcycling_pass_count(cm::AbstractArray{FT,2},
                                              m::AbstractArray{FT,2},
                                              ws::AdvectionWorkspace{FT},
                                              cfl_limit::FT; max_n_sub::Int = 4096) where FT
    isinf(cfl_limit) && return 1
    static_cfl = _vertical_face_outgoing_ratio(cm, m)
    n_sub = _subcycling_pass_count(static_cfl, cfl_limit)
    n_sub <= max_n_sub || throw(ArgumentError("face-indexed vertical subcycling exceeded max_n_sub=$(max_n_sub)"))
    return n_sub
end

@inline function _sweep_horizontal_face_subcycled!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                                   horizontal_flux::AbstractArray{FT,2},
                                                   mesh::AbstractHorizontalMesh,
                                                   scheme::Union{AbstractAdvection, AbstractAdvectionScheme},
                                                   ws::AdvectionWorkspace{FT},
                                                   cfl_limit::FT) where FT
    n_sub = _horizontal_face_subcycling_pass_count(horizontal_flux, m, mesh, ws, cfl_limit)
    if n_sub == 1
        sweep_horizontal!(rm, m, horizontal_flux, mesh, scheme, ws)
        return 1
    end
    flux_scale = inv(FT(n_sub))
    for _ in 1:n_sub
        sweep_horizontal!(rm, m, horizontal_flux, mesh, scheme, ws, flux_scale)
    end
    return n_sub
end

@inline function _sweep_vertical_face_subcycled!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                                 cm::AbstractArray{FT,2},
                                                 scheme::Union{AbstractAdvection, AbstractAdvectionScheme},
                                                 ws::AdvectionWorkspace{FT},
                                                 cfl_limit::FT) where FT
    n_sub = _vertical_face_subcycling_pass_count(cm, m, ws, cfl_limit)
    if n_sub == 1
        sweep_vertical!(rm, m, cm, scheme, ws)
        return 1
    end
    flux_scale = inv(FT(n_sub))
    for _ in 1:n_sub
        sweep_vertical!(rm, m, cm, scheme, ws, flux_scale)
    end
    return n_sub
end

# Unified CFL pilot — single static algorithm for CPU and GPU (plan 13).
#
# For each cell (i,j,k), CFL = total_outflow / cell_mass. Total outflow is
# the sum of flux leaving the cell through its two faces in the active
# direction; the maximum over all cells determines n_sub.
#
# For the structured x direction, face flux `am[face, j, k]` lives between
# cells (face-1, j, k) and (face, j, k):
#   positive am[i]   = rightward flow   → leaves cell i-1 through its right face
#   negative am[i]   = leftward flow    → leaves cell i   through its left face
# so outflow(cell i) = max(-am[i], 0) + max(am[i+1], 0). y and z are analogous
# (positive cm = downward; cell k's outflow is upward through cm[:,:,k] and
# downward through cm[:,:,k+1]).
#
# Plan 13 Commit 2 replaces the pre-existing dual-path (CPU evolving-mass,
# GPU static-inflow) with one algorithm that (a) is backend-agnostic via
# pure broadcast, (b) computes OUTFLOW correctly (the prior GPU path
# summed inflow — a sign bug that under-estimated CFL on device).
function _x_subcycling_pass_count(am::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                                  ws::AdvectionWorkspace{FT},
                                  cfl_limit::FT; max_n_sub::Int = 4096) where FT
    isinf(cfl_limit) && return 1
    Nx = size(m, 1)
    out = max.(.- @view(am[1:Nx, :, :]), zero(FT)) .+ max.(@view(am[2:Nx+1, :, :]), zero(FT))
    static_cfl = maximum(out ./ max.(m, eps(FT)))
    n_sub = _subcycling_pass_count(static_cfl, cfl_limit)
    n_sub <= max_n_sub || throw(ArgumentError("x-direction subcycling exceeded max_n_sub=$(max_n_sub)"))
    return n_sub
end

function _y_subcycling_pass_count(bm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                                  ws::AdvectionWorkspace{FT},
                                  cfl_limit::FT; max_n_sub::Int = 4096) where FT
    isinf(cfl_limit) && return 1
    Ny = size(m, 2)
    out = max.(.- @view(bm[:, 1:Ny, :]), zero(FT)) .+ max.(@view(bm[:, 2:Ny+1, :]), zero(FT))
    static_cfl = maximum(out ./ max.(m, eps(FT)))
    n_sub = _subcycling_pass_count(static_cfl, cfl_limit)
    n_sub <= max_n_sub || throw(ArgumentError("y-direction subcycling exceeded max_n_sub=$(max_n_sub)"))
    return n_sub
end

function _z_subcycling_pass_count(cm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                                  ws::AdvectionWorkspace{FT},
                                  cfl_limit::FT; max_n_sub::Int = 4096) where FT
    isinf(cfl_limit) && return 1
    Nz = size(m, 3)
    out = max.(.- @view(cm[:, :, 1:Nz]), zero(FT)) .+ max.(@view(cm[:, :, 2:Nz+1]), zero(FT))
    static_cfl = maximum(out ./ max.(m, eps(FT)))
    n_sub = _subcycling_pass_count(static_cfl, cfl_limit)
    n_sub <= max_n_sub || throw(ArgumentError("z-direction subcycling exceeded max_n_sub=$(max_n_sub)"))
    return n_sub
end

@inline function _sweep_x_subcycled!(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                                     am::AbstractArray{FT,3},
                                     scheme::Union{AbstractAdvection, AbstractAdvectionScheme},
                                     ws::AdvectionWorkspace{FT},
                                     cfl_limit::FT) where FT
    n_sub = _x_subcycling_pass_count(am, m, ws, cfl_limit)
    if n_sub == 1
        sweep_x!(rm, m, am, scheme, ws)
        return 1
    end
    flux_scale = inv(FT(n_sub))
    for _ in 1:n_sub
        sweep_x!(rm, m, am, scheme, ws, flux_scale)
    end
    return n_sub
end

@inline function _sweep_y_subcycled!(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                                     bm::AbstractArray{FT,3},
                                     scheme::Union{AbstractAdvection, AbstractAdvectionScheme},
                                     ws::AdvectionWorkspace{FT},
                                     cfl_limit::FT) where FT
    n_sub = _y_subcycling_pass_count(bm, m, ws, cfl_limit)
    if n_sub == 1
        sweep_y!(rm, m, bm, scheme, ws)
        return 1
    end
    flux_scale = inv(FT(n_sub))
    for _ in 1:n_sub
        sweep_y!(rm, m, bm, scheme, ws, flux_scale)
    end
    return n_sub
end

@inline function _sweep_z_subcycled!(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                                     cm::AbstractArray{FT,3},
                                     scheme::Union{AbstractAdvection, AbstractAdvectionScheme},
                                     ws::AdvectionWorkspace{FT},
                                     cfl_limit::FT) where FT
    n_sub = _z_subcycling_pass_count(cm, m, ws, cfl_limit)
    if n_sub == 1
        sweep_z!(rm, m, cm, scheme, ws)
        return 1
    end
    flux_scale = inv(FT(n_sub))
    for _ in 1:n_sub
        sweep_z!(rm, m, cm, scheme, ws, flux_scale)
    end
    return n_sub
end

# -------------------------------------------------------------------------
# Ping-pong subcycled sweeps — used by strang_split! to eliminate the
# per-sweep copyto!. Each helper reads from (rm_in, m_in) and — after
# n_sub internal passes — leaves the result either in (rm_out, m_out)
# when n_sub is odd, or back in (rm_in, m_in) when n_sub is even.
# The caller rebinds `cur` / `alt` locally based on `isodd(n_sub)`.
# -------------------------------------------------------------------------

@inline function _sweep_x_pp_subcycled!(rm_in::AbstractArray{FT,3},  rm_out::AbstractArray{FT,3},
                                        m_in::AbstractArray{FT,3},   m_out::AbstractArray{FT,3},
                                        am::AbstractArray{FT,3},
                                        scheme::Union{AbstractAdvection, AbstractAdvectionScheme},
                                        ws::AdvectionWorkspace{FT},
                                        cfl_limit::FT) where FT
    n_sub = _x_subcycling_pass_count(am, m_in, ws, cfl_limit)
    if n_sub == 1
        sweep_x!(rm_in, rm_out, m_in, m_out, am, scheme, ws)
        return n_sub
    end
    flux_scale = inv(FT(n_sub))
    @inbounds for pass in 1:n_sub
        if isodd(pass)
            sweep_x!(rm_in,  rm_out, m_in,  m_out, am, scheme, ws, flux_scale)
        else
            sweep_x!(rm_out, rm_in,  m_out, m_in,  am, scheme, ws, flux_scale)
        end
    end
    return n_sub
end

@inline function _sweep_y_pp_subcycled!(rm_in::AbstractArray{FT,3},  rm_out::AbstractArray{FT,3},
                                        m_in::AbstractArray{FT,3},   m_out::AbstractArray{FT,3},
                                        bm::AbstractArray{FT,3},
                                        scheme::Union{AbstractAdvection, AbstractAdvectionScheme},
                                        ws::AdvectionWorkspace{FT},
                                        cfl_limit::FT) where FT
    n_sub = _y_subcycling_pass_count(bm, m_in, ws, cfl_limit)
    if n_sub == 1
        sweep_y!(rm_in, rm_out, m_in, m_out, bm, scheme, ws)
        return n_sub
    end
    flux_scale = inv(FT(n_sub))
    @inbounds for pass in 1:n_sub
        if isodd(pass)
            sweep_y!(rm_in,  rm_out, m_in,  m_out, bm, scheme, ws, flux_scale)
        else
            sweep_y!(rm_out, rm_in,  m_out, m_in,  bm, scheme, ws, flux_scale)
        end
    end
    return n_sub
end

@inline function _sweep_z_pp_subcycled!(rm_in::AbstractArray{FT,3},  rm_out::AbstractArray{FT,3},
                                        m_in::AbstractArray{FT,3},   m_out::AbstractArray{FT,3},
                                        cm::AbstractArray{FT,3},
                                        scheme::Union{AbstractAdvection, AbstractAdvectionScheme},
                                        ws::AdvectionWorkspace{FT},
                                        cfl_limit::FT) where FT
    n_sub = _z_subcycling_pass_count(cm, m_in, ws, cfl_limit)
    if n_sub == 1
        sweep_z!(rm_in, rm_out, m_in, m_out, cm, scheme, ws)
        return n_sub
    end
    flux_scale = inv(FT(n_sub))
    @inbounds for pass in 1:n_sub
        if isodd(pass)
            sweep_z!(rm_in,  rm_out, m_in,  m_out, cm, scheme, ws, flux_scale)
        else
            sweep_z!(rm_out, rm_in,  m_out, m_in,  cm, scheme, ws, flux_scale)
        end
    end
    return n_sub
end

# =========================================================================
# Strang splitting: X → Y → Z → Z → Y → X
# =========================================================================
#
# The Strang (1968) symmetric splitting sequence achieves second-order
# accuracy in the splitting error while using directional sweeps in the
# palindrome X → Y → Z → Z → Y → X.
#
# In `src`, the kernels consume a fully prepared mass-flux state for the
# active substep.  Any time interpolation or window-to-substep conversion is
# handled upstream by the driver/runtime layer before these sweeps are called.
#
# Multi-tracer handling: each tracer is advected independently through
# the full Strang sequence.  Air mass is saved before the first tracer
# and restored before subsequent tracers, so that each tracer sees the
# same initial mass field (mass changes from one tracer's advection
# must not leak into another tracer's transport).

"""
    strang_split!(state, fluxes, grid, scheme; workspace)

Perform one full Strang-split advection step on a structured mesh.

# Splitting sequence
```
  X → Y → Z → Z → Y → X
  ─────────────────────────
  half    half   full  half    half
```

Each directional sweep updates both tracer mass (`rm`) and air mass (`m`)
using the divergence of the prepared substep face mass flux.  The kernels do
not perform any time interpolation or vertical-flux closure.

# Multi-tracer protocol
When `state.tracers` contains multiple tracers:
1. Save the initial air mass (`m_save = copy(m)`)
2. For each tracer: restore `m` from `m_save`, then run the full
   X-Y-Z-Z-Y-X sequence

This ensures each tracer sees the same initial air mass distribution.

# Arguments
- `state::CellState` — cell state containing `air_mass` and `tracers`
- `fluxes::StructuredFaceFluxState` — mass fluxes (am, bm, cm)
- `grid::AtmosGrid{<:LatLonMesh}` — structured lat-lon grid
- `scheme` — advection scheme (new `AbstractAdvectionScheme` or legacy `AbstractAdvection`)
- `workspace::AdvectionWorkspace` — pre-allocated double buffers
"""
function strang_split!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
                       grid::AtmosGrid{<:LatLonMesh},
                       scheme::Union{AbstractAdvection, AbstractAdvectionScheme};
                       workspace::AdvectionWorkspace,
                       cfl_limit::Real = one(eltype(state.air_mass))) where {B <: AbstractMassBasis}
    m = state.air_mass
    am, bm, cm = fluxes.am, fluxes.bm, fluxes.cm

    cfl_limit_ft = convert(eltype(m), cfl_limit)

    n_tr = length(state.tracers)
    m_save = workspace.m_save
    if n_tr > 1
        copyto!(m_save, m)
    end

    # Ping-pong alternate buffers used as the "other side" of each sweep.
    rm_alt0 = workspace.rm_B
    m_alt0  = workspace.m_B

    for (idx, (name, rm_tracer)) in enumerate(pairs(state.tracers))
        if idx > 1
            copyto!(m, m_save)
        end

        # Local ping-pong state: (rm_cur, m_cur) holds the current values,
        # (rm_alt, m_alt) is the scratch destination for the next sweep.
        # At entry, (rm_cur, m_cur) = caller's (rm_tracer, m). After each
        # sweep we rebind so the output becomes input for the next direction.
        rm_cur, m_cur = rm_tracer, m
        rm_alt, m_alt = rm_alt0,   m_alt0

        # X → Y → Z → Z → Y → X palindrome, six sweeps. The helper returns
        # n_sub for the sweep; when n_sub is odd, the result is in (alt),
        # so we flip the cur/alt bindings.
        n = _sweep_x_pp_subcycled!(rm_cur, rm_alt, m_cur, m_alt, am, scheme, workspace, cfl_limit_ft)
        if isodd(n); rm_cur, rm_alt = rm_alt, rm_cur; m_cur, m_alt = m_alt, m_cur; end

        n = _sweep_y_pp_subcycled!(rm_cur, rm_alt, m_cur, m_alt, bm, scheme, workspace, cfl_limit_ft)
        if isodd(n); rm_cur, rm_alt = rm_alt, rm_cur; m_cur, m_alt = m_alt, m_cur; end

        n = _sweep_z_pp_subcycled!(rm_cur, rm_alt, m_cur, m_alt, cm, scheme, workspace, cfl_limit_ft)
        if isodd(n); rm_cur, rm_alt = rm_alt, rm_cur; m_cur, m_alt = m_alt, m_cur; end

        n = _sweep_z_pp_subcycled!(rm_cur, rm_alt, m_cur, m_alt, cm, scheme, workspace, cfl_limit_ft)
        if isodd(n); rm_cur, rm_alt = rm_alt, rm_cur; m_cur, m_alt = m_alt, m_cur; end

        n = _sweep_y_pp_subcycled!(rm_cur, rm_alt, m_cur, m_alt, bm, scheme, workspace, cfl_limit_ft)
        if isodd(n); rm_cur, rm_alt = rm_alt, rm_cur; m_cur, m_alt = m_alt, m_cur; end

        n = _sweep_x_pp_subcycled!(rm_cur, rm_alt, m_cur, m_alt, am, scheme, workspace, cfl_limit_ft)
        if isodd(n); rm_cur, rm_alt = rm_alt, rm_cur; m_cur, m_alt = m_alt, m_cur; end

        # If total parity wound up in the alternate buffer, copy it back
        # to the caller-provided arrays. In the typical case where all six
        # n_sub values are 1 (or all even), the final parity lands back in
        # (rm_tracer, m) and NO copies happen — that is the ping-pong win.
        if rm_cur !== rm_tracer
            copyto!(rm_tracer, rm_cur)
            copyto!(m,         m_cur)
        end
    end

    return nothing
end

function strang_split!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
                       grid::AtmosGrid{<:CubedSphereMesh},
                       scheme::Union{AbstractAdvection, AbstractAdvectionScheme};
                       workspace::AdvectionWorkspace) where {B <: AbstractMassBasis}
    throw(ArgumentError("CubedSphereMesh remains metadata-only in src; structured advection is only supported on LatLonMesh until cubed-sphere geometry/connectivity are implemented"))
end

# =========================================================================
# apply! entry points
# =========================================================================

"""
    apply!(state, fluxes, grid, scheme, dt; workspace)

Structured-mesh advection entry point.  Delegates to [`strang_split!`](@ref).

The `dt` argument is not used inside the kernels. Callers are responsible
for preparing `fluxes` so they already represent the intended substep forcing.
"""
function apply!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
                grid::AtmosGrid{<:LatLonMesh},
                scheme::Union{AbstractAdvection, AbstractAdvectionScheme}, dt;
                workspace::AdvectionWorkspace,
                cfl_limit::Real = one(eltype(state.air_mass))) where {B <: AbstractMassBasis}
    strang_split!(state, fluxes, grid, scheme; workspace=workspace, cfl_limit=cfl_limit)
    return nothing
end

function apply!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
                grid::AtmosGrid{<:CubedSphereMesh},
                scheme::Union{AbstractAdvection, AbstractAdvectionScheme}, dt;
                workspace::AdvectionWorkspace) where {B <: AbstractMassBasis}
    throw(ArgumentError("CubedSphereMesh remains metadata-only in src; structured advection is only supported on LatLonMesh until cubed-sphere geometry/connectivity are implemented"))
end

# Face-indexed apply! — shared tracer loop, generated for each topology
for (scheme_type, h_sweep, v_sweep) in (
    (:AbstractConstantScheme,    :sweep_horizontal!, :sweep_vertical!),
)
    @eval function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                          grid::AtmosGrid{<:AbstractHorizontalMesh},
                          scheme::$scheme_type, dt;
                          workspace::AdvectionWorkspace,
                          cfl_limit::Real = one(eltype(state.air_mass))) where {B <: AbstractMassBasis}
        m = state.air_mass
        hflux, cm = fluxes.horizontal_flux, fluxes.cm
        cfl_limit_ft = convert(eltype(m), cfl_limit)

        n_tr = length(state.tracers)
        m_save = n_tr > 1 ? similar(m) : m
        if n_tr > 1
            copyto!(m_save, m)
        end

        for (idx, (_, rm_tracer)) in enumerate(pairs(state.tracers))
            if idx > 1
                copyto!(m, m_save)
            end

            _sweep_horizontal_face_subcycled!(rm_tracer, m, hflux, grid.horizontal, scheme, workspace, cfl_limit_ft)
            _sweep_vertical_face_subcycled!(rm_tracer, m, cm, scheme, workspace, cfl_limit_ft)
            _sweep_vertical_face_subcycled!(rm_tracer, m, cm, scheme, workspace, cfl_limit_ft)
            _sweep_horizontal_face_subcycled!(rm_tracer, m, hflux, grid.horizontal, scheme, workspace, cfl_limit_ft)
        end

        return nothing
    end
end

# Error stubs for unsupported face-indexed scheme families
for (scheme_type, label) in (
    (:AbstractLinearScheme,            "linear-reconstruction"),
    (:AbstractQuadraticScheme,         "quadratic-reconstruction"),
)
    @eval function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                          grid::AtmosGrid{<:AbstractHorizontalMesh},
                          scheme::$scheme_type, dt; kwargs...) where {B <: AbstractMassBasis}
        throw(ArgumentError("Face-connected " * $label * " schemes not yet implemented for $(typeof(grid.horizontal))"))
    end
end

# =========================================================================
# Multi-tracer directional sweeps — generated via @eval
# =========================================================================
#
# These operate on 4D tracer arrays (Nx, Ny, Nz, Nt) and use the
# multi-tracer kernel shells from multitracer_kernels.jl.  The mass
# update is computed ONCE per cell, shared across all tracers.

for (sweep_fn, kernel_fn, dim) in (
    (:sweep_x_mt!, :_xsweep_mt_kernel!, 1),
    (:sweep_y_mt!, :_ysweep_mt_kernel!, 2),
    (:sweep_z_mt!, :_zsweep_mt_kernel!, 3),
)
    @eval begin
        # Ping-pong entry point: writes (rm4d_out, m_out), reads (rm4d_in, m_in).
        function $sweep_fn(rm4d_in::AbstractArray{FT,4},  rm4d_out::AbstractArray{FT,4},
                           m_in::AbstractArray{FT,3},     m_out::AbstractArray{FT,3},
                           flux::AbstractArray{FT,3},
                           scheme::AbstractAdvectionScheme,
                           ws::AdvectionWorkspace{FT},
                           flux_scale::FT = one(FT)) where FT
            backend = get_backend(m_in)
            Nt = Int32(size(rm4d_in, 4))
            kernel! = $kernel_fn(backend, 256)
            kernel!(rm4d_out, rm4d_in, m_out, m_in, flux, scheme,
                    Int32(size(m_in, $dim)), Nt, flux_scale;
                    ndrange=size(m_in))
            synchronize(backend)
            return nothing
        end

        """
            $($sweep_fn)(rm_4d, m, flux, scheme, ws[, flux_scale])

        Backward-compat wrapper: write into workspace B buffers, then
        copy back. New callers should use the 7-arg / 8-arg ping-pong
        form instead.
        """
        function $sweep_fn(rm_4d::AbstractArray{FT,4}, m::AbstractArray{FT,3},
                           flux::AbstractArray{FT,3},
                           scheme::AbstractAdvectionScheme,
                           ws::AdvectionWorkspace{FT},
                           flux_scale::FT = one(FT)) where FT
            $sweep_fn(rm_4d, ws.rm_4d_B, m, ws.m_B, flux, scheme, ws, flux_scale)
            copyto!(rm_4d, ws.rm_4d_B)
            copyto!(m,     ws.m_B)
            return nothing
        end
    end
end

# =========================================================================
# Multi-tracer Strang splitting: X → Y → Z → Z → Y → X
# =========================================================================

"""
    strang_split_mt!(rm_4d, m, am, bm, cm, scheme, ws)

Multi-tracer Strang-split advection on a packed 4D tracer array.

This is the performance-optimized path: all `Nt = size(rm_4d, 4)` tracers
are processed in a SINGLE kernel launch per sweep direction (6 total),
rather than `6 × Nt` launches in the per-tracer path.

The mass update (m_new = m + flux_in - flux_out) is computed ONCE per
cell per sweep, shared across all tracers.

# Arguments
- `rm_4d::AbstractArray{FT,4}` — tracer mass `(Nx, Ny, Nz, Nt)`, mutated
- `m::AbstractArray{FT,3}` — air mass `(Nx, Ny, Nz)`, mutated
- `am, bm, cm` — mass fluxes (x, y, z directions)
- `scheme::AbstractAdvectionScheme` — advection scheme
- `ws::AdvectionWorkspace` — workspace with 4D buffer allocated
"""
function strang_split_mt!(rm_4d::AbstractArray{FT,4}, m::AbstractArray{FT,3},
                          am::AbstractArray{FT,3}, bm::AbstractArray{FT,3},
                          cm::AbstractArray{FT,3},
                          scheme::AbstractAdvectionScheme,
                          ws::AdvectionWorkspace{FT};
                          cfl_limit::Real = one(FT)) where FT
    cfl_ft = convert(FT, cfl_limit)

    # CFL subcycling per direction (reuse single-tracer pilot on the 3D mass)
    n_x = _x_subcycling_pass_count(am, m, ws, cfl_ft)
    n_y = _y_subcycling_pass_count(bm, m, ws, cfl_ft)
    n_z = _z_subcycling_pass_count(cm, m, ws, cfl_ft)

    fs_x = inv(FT(n_x))
    fs_y = inv(FT(n_y))
    fs_z = inv(FT(n_z))

    # Ping-pong state. (rm_cur, m_cur) starts as the caller's buffers;
    # (rm_alt, m_alt) is the workspace B pair. Each kernel launch reads
    # from cur and writes to alt; we then rebind cur ↔ alt so the next
    # launch continues the chain with zero inter-sweep copies.
    rm_cur, m_cur = rm_4d,       m
    rm_alt, m_alt = ws.rm_4d_B,  ws.m_B

    # Inline one palindrome direction at a time. The helper does the
    # rebinding to avoid @eval-ing a parametric loop body.
    @inline function _pass!(sweep_fn, rm_cur_, rm_alt_, m_cur_, m_alt_, flux, fs)
        sweep_fn(rm_cur_, rm_alt_, m_cur_, m_alt_, flux, scheme, ws, fs)
        return rm_alt_, rm_cur_, m_alt_, m_cur_
    end

    for _ in 1:n_x; rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_x_mt!, rm_cur, rm_alt, m_cur, m_alt, am, fs_x); end
    for _ in 1:n_y; rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_y_mt!, rm_cur, rm_alt, m_cur, m_alt, bm, fs_y); end
    for _ in 1:n_z; rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_z_mt!, rm_cur, rm_alt, m_cur, m_alt, cm, fs_z); end
    for _ in 1:n_z; rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_z_mt!, rm_cur, rm_alt, m_cur, m_alt, cm, fs_z); end
    for _ in 1:n_y; rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_y_mt!, rm_cur, rm_alt, m_cur, m_alt, bm, fs_y); end
    for _ in 1:n_x; rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_x_mt!, rm_cur, rm_alt, m_cur, m_alt, am, fs_x); end

    # If total parity wound up in the alternate (B) buffer, copy back.
    # When (n_x + n_y + n_z + n_z + n_y + n_x) is even — the common case,
    # e.g. all-ones — the final result lands in the caller's arrays with
    # zero copyto! calls.
    if rm_cur !== rm_4d
        copyto!(rm_4d, rm_cur)
        copyto!(m,     m_cur)
    end
    return nothing
end

export AdvectionWorkspace, strang_split!, strang_split_mt!
