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

using KernelAbstractions: get_backend, synchronize

# =========================================================================
# AdvectionWorkspace — pre-allocated double buffers
# =========================================================================

"""
    AdvectionWorkspace{FT, A, V1}

Pre-allocated buffers for mass-flux Strang splitting.  Eliminates all
array allocations from the inner time-stepping loop.

The workspace provides two buffer arrays (`rm_buf`, `m_buf`) that serve
as output targets for the advection kernels (double-buffering pattern).
The kernel writes to the buffers while reading from the original arrays,
then the sweep function copies the buffers back.  This is ESSENTIAL for
correctness — in-place kernel updates would violate the stencil's
read-before-write contract and break mass conservation by ~10% per step.

# Fields
- `rm_buf::A` — double-buffer target for tracer mass (same size as `rm`)
- `m_buf::A` — double-buffer target for air mass (same size as `m`)
- `cluster_sizes::V1` — per-latitude clustering factors for reduced grids
  (`Int32[Ny]`; all ones for uniform grids; empty for face-indexed meshes)

# Constructors

    AdvectionWorkspace(m::AbstractArray{FT,3}; cluster_sizes_cpu=nothing)

Create workspace for a 3D structured grid, allocating buffers matching
the size of `m`.  If `cluster_sizes_cpu` is nothing, defaults to uniform
(all ones).

    AdvectionWorkspace(m::AbstractArray{FT,2}; cluster_sizes_cpu=nothing)

Create workspace for a 2D face-indexed grid (cell × level layout).
"""
struct AdvectionWorkspace{FT, A <: AbstractArray{FT}, V1 <: AbstractVector{Int32}, A4}
    rm_buf        :: A
    m_buf         :: A
    cluster_sizes :: V1
    rm_4d_buf     :: A4
end

function AdvectionWorkspace(m::AbstractArray{FT,3};
                            cluster_sizes_cpu::Union{Nothing, Vector{Int32}} = nothing,
                            n_tracers::Int = 0) where FT
    Nx, Ny, Nz = size(m)
    cs_cpu = cluster_sizes_cpu !== nothing ? cluster_sizes_cpu : ones(Int32, Ny)
    cs_dev = similar(m, Int32, Ny)
    copyto!(cs_dev, cs_cpu)
    rm_4d = n_tracers > 0 ? similar(m, Nx, Ny, Nz, n_tracers) : similar(m, 0, 0, 0, 0)
    AdvectionWorkspace{FT, typeof(m), typeof(cs_dev), typeof(rm_4d)}(
        similar(m), similar(m), cs_dev, rm_4d)
end

function AdvectionWorkspace(m::AbstractArray{FT,2};
                            cluster_sizes_cpu::Union{Nothing, Vector{Int32}} = nothing) where FT
    cs_dev = Int32[]
    rm_4d = similar(m, 0, 0)
    AdvectionWorkspace{FT, typeof(m), typeof(cs_dev), typeof(rm_4d)}(
        similar(m), similar(m), cs_dev, rm_4d)
end

function Adapt.adapt_structure(to, ws::AdvectionWorkspace{FT}) where {FT}
    rm_buf = Adapt.adapt(to, ws.rm_buf)
    m_buf = Adapt.adapt(to, ws.m_buf)
    cluster_sizes = Adapt.adapt(to, ws.cluster_sizes)
    rm_4d_buf = Adapt.adapt(to, ws.rm_4d_buf)
    return AdvectionWorkspace{FT, typeof(rm_buf), typeof(cluster_sizes), typeof(rm_4d_buf)}(
        rm_buf, m_buf, cluster_sizes, rm_4d_buf)
end

# =========================================================================
# Generic directional sweeps — generated via @eval
# =========================================================================
#
# Pattern (from CUDA.jl wrappers.jl):
#   for (symbol_tuple...) in table
#       @eval function $sweep_fn(...) ... $kernel_fn ... end
#   end
#
# Each sweep:
# 1. Compiles the appropriate KA kernel for the current backend
# 2. Launches it with ndrange = size(m), writing to ws.rm_buf / ws.m_buf
# 3. Synchronizes (GPU fence)
# 4. Copies results back to rm / m (completing the double-buffer swap)
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
        """
            $($sweep_fn)(rm, m, flux, scheme, ws)

        One directional advection sweep using the new scheme hierarchy.

        Launches `$($kernel_fn)` on the current backend, writing into
        workspace buffers, then copies results back to `rm` and `m`.

        # Arguments
        - `rm::AbstractArray{FT,3}` — tracer mass (mutated in place)
        - `m::AbstractArray{FT,3}` — air mass (mutated in place)
        - `flux::AbstractArray{FT,3}` — staggered-grid mass flux for this direction
        - `scheme::AbstractAdvectionScheme` — determines the face-flux function
        - `ws::AdvectionWorkspace{FT}` — pre-allocated double buffers
        """
        function $sweep_fn(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                                  flux::AbstractArray{FT,3},
                                  scheme::AbstractAdvectionScheme,
                                  ws::AdvectionWorkspace{FT}) where FT
            backend = get_backend(m)
            kernel! = $kernel_fn(backend, 256)
            kernel!(ws.rm_buf, rm, ws.m_buf, m, flux, scheme, Int32(size(m, $dim)), one(FT);
                    ndrange=size(m))
            synchronize(backend)
            copyto!(rm, ws.rm_buf)
            copyto!(m,  ws.m_buf)
            return nothing
        end
    end
end

# =========================================================================
# Legacy per-scheme directional sweeps — also generated via @eval
# =========================================================================
#
# These use the old monolithic kernels from Upwind.jl / RussellLerner.jl.
# They exist for backward compatibility and equivalence testing.
#
# The `extra_args` tuple captures per-direction / per-scheme extra kernel
# arguments (e.g., cluster_sizes for legacy x-kernels, use_limiter for
# the Russell–Lerner scheme).

for (sweep_fn, scheme_type, kernel_fn, dim, extra_args) in (
    (:sweep_x!, :UpwindAdvection,        :_upwind_x_kernel!, 1, (:(ws.cluster_sizes),)),
    (:sweep_y!, :UpwindAdvection,        :_upwind_y_kernel!, 2, ()),
    (:sweep_z!, :UpwindAdvection,        :_upwind_z_kernel!, 3, ()),
    (:sweep_x!, :RussellLernerAdvection, :_rl_x_kernel!,     1, (:(ws.cluster_sizes), :(scheme.use_limiter))),
    (:sweep_y!, :RussellLernerAdvection, :_rl_y_kernel!,     2, (:(scheme.use_limiter),)),
    (:sweep_z!, :RussellLernerAdvection, :_rl_z_kernel!,     3, (:(scheme.use_limiter),)),
)
    @eval function $sweep_fn(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                              flux::AbstractArray{FT,3},
                              scheme::$scheme_type,
                              ws::AdvectionWorkspace{FT}) where FT
        backend = get_backend(m)
        kernel! = $kernel_fn(backend, 256)
        kernel!(ws.rm_buf, rm, ws.m_buf, m, flux,
                Int32(size(m, $dim)), $(extra_args...), one(FT);
                ndrange=size(m))
        synchronize(backend)
        copyto!(rm, ws.rm_buf)
        copyto!(m,  ws.m_buf)
        return nothing
    end
end

# Additional structured sweep overloads with explicit flux scaling.
# These are used by the CFL-based subcycling wrappers to reapply the same
# directional forcing in smaller conservative pieces.
for (sweep_fn, kernel_fn, dim) in (
    (:sweep_x!, :_xsweep_kernel!, 1),
    (:sweep_y!, :_ysweep_kernel!, 2),
    (:sweep_z!, :_zsweep_kernel!, 3),
)
    @eval function $sweep_fn(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                             flux::AbstractArray{FT,3},
                             scheme::AbstractAdvectionScheme,
                             ws::AdvectionWorkspace{FT},
                             flux_scale::FT) where FT
        backend = get_backend(m)
        kernel! = $kernel_fn(backend, 256)
        kernel!(ws.rm_buf, rm, ws.m_buf, m, flux, scheme, Int32(size(m, $dim)), flux_scale;
                ndrange=size(m))
        synchronize(backend)
        copyto!(rm, ws.rm_buf)
        copyto!(m,  ws.m_buf)
        return nothing
    end
end

for (sweep_fn, scheme_type, kernel_fn, dim, extra_args) in (
    (:sweep_x!, :UpwindAdvection,        :_upwind_x_kernel!, 1, (:(ws.cluster_sizes),)),
    (:sweep_y!, :UpwindAdvection,        :_upwind_y_kernel!, 2, ()),
    (:sweep_z!, :UpwindAdvection,        :_upwind_z_kernel!, 3, ()),
    (:sweep_x!, :RussellLernerAdvection, :_rl_x_kernel!,     1, (:(ws.cluster_sizes), :(scheme.use_limiter))),
    (:sweep_y!, :RussellLernerAdvection, :_rl_y_kernel!,     2, (:(scheme.use_limiter),)),
    (:sweep_z!, :RussellLernerAdvection, :_rl_z_kernel!,     3, (:(scheme.use_limiter),)),
)
    @eval function $sweep_fn(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                             flux::AbstractArray{FT,3},
                             scheme::$scheme_type,
                             ws::AdvectionWorkspace{FT},
                             flux_scale::FT) where FT
        backend = get_backend(m)
        kernel! = $kernel_fn(backend, 256)
        kernel!(ws.rm_buf, rm, ws.m_buf, m, flux,
                Int32(size(m, $dim)), $(extra_args...), flux_scale;
                ndrange=size(m))
        synchronize(backend)
        copyto!(rm, ws.rm_buf)
        copyto!(m,  ws.m_buf)
        return nothing
    end
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
                                            scheme::AbstractAdvectionScheme) where FT
    copyto!(rm_new, rm)
    copyto!(m_new, m)
    nface = nfaces(mesh)
    Nz = size(m, 2)

    @inbounds for k in 1:Nz
        for f in 1:nface
            left, right = face_cells(mesh, f)
            if left > 0 && right > 0
                flux = horizontal_flux[f, k]
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
                                            scheme::AbstractAdvectionScheme) where FT
    copyto!(rm_new, rm)
    copyto!(m_new, m)
    nc = size(m, 1)
    Nz = size(m, 2)

    @inbounds for k in 1:Nz
        for c in 1:nc
            flux_t = k > 1 ? _vface_tracer_flux(rm, m, cm[c, k], c, k - 1, k, scheme) : zero(FT)
            flux_b = k < Nz ? _vface_tracer_flux(rm, m, cm[c, k + 1], c, k, k + 1, scheme) : zero(FT)
            rm_new[c, k] = rm[c, k] + flux_t - flux_b
            m_new[c, k]  = m[c, k]  + cm[c, k] - cm[c, k + 1]
        end
    end

    return nothing
end

# ---- Legacy overloads (no scheme argument) for old UpwindAdvection path ----

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
            flux_t = k > 1 ? _upwind_face_flux(cm[c, k],
                                               rm[c, k - 1] / max(m[c, k - 1], m_floor),
                                               rm[c, k]     / max(m[c, k],     m_floor)) : zero(FT)
            flux_b = k < Nz ? _upwind_face_flux(cm[c, k + 1],
                                               rm[c, k]     / max(m[c, k],     m_floor),
                                               rm[c, k + 1] / max(m[c, k + 1], m_floor)) : zero(FT)
            rm_new[c, k] = rm[c, k] + flux_t - flux_b
            m_new[c, k]  = m[c, k]  + cm[c, k] - cm[c, k + 1]
        end
    end

    return nothing
end

# =========================================================================
# Face-indexed sweep helpers — generated via @eval for both legacy and new
# =========================================================================

for (scheme_type, h_args, v_args) in (
    (:UpwindAdvection,    (:mesh,), ()),
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
            _horizontal_face_tendency!(ws.rm_buf, rm, ws.m_buf, m, horizontal_flux, $(h_args...))
            copyto!(rm, ws.rm_buf)
            copyto!(m, ws.m_buf)
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
            _vertical_column_tendency!(ws.rm_buf, rm, ws.m_buf, m, cm, $(v_args...))
            copyto!(rm, ws.rm_buf)
            copyto!(m, ws.m_buf)
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

function _x_subcycling_pass_count(am::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                                  ws::AdvectionWorkspace{FT},
                                  cfl_limit::FT; max_n_sub::Int = 4096) where FT
    # Device-side CFL pilots are not implemented yet. Keep GPU and other
    # non-Array backends on the existing single-pass path until a proper
    # reduction-based pilot is added.
    m isa Array || return 1
    Nx, Ny, Nz = size(m)
    mx = ws.m_buf
    mx_next = ws.rm_buf
    n_sub = _subcycling_pass_count(max_cfl_x(am, m, ws.cluster_sizes), cfl_limit)

    while true
        copyto!(mx, m)
        flux_scale = inv(FT(n_sub))
        cfl_ok = true

        for pass in 1:n_sub
            @inbounds for k in 1:Nz, j in 1:Ny, face in 1:(Nx + 1)
                flux = flux_scale * am[face, j, k]
                donor = flux >= zero(FT) ? (face == 1 ? Nx : face - 1) : (face > Nx ? 1 : face)
                if abs(flux) >= cfl_limit * max(mx[donor, j, k], eps(FT))
                    cfl_ok = false
                    break
                end
            end
            cfl_ok || break

            if pass != n_sub
                @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
                    mx_next[i, j, k] = mx[i, j, k] + flux_scale * am[i, j, k] - flux_scale * am[i + 1, j, k]
                end
                mx, mx_next = mx_next, mx
            end
        end

        cfl_ok && return n_sub
        n_sub += 1
        n_sub <= max_n_sub || throw(ArgumentError("x-direction subcycling exceeded max_n_sub=$(max_n_sub)"))
    end
end

function _y_subcycling_pass_count(bm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                                  ws::AdvectionWorkspace{FT},
                                  cfl_limit::FT; max_n_sub::Int = 4096) where FT
    # Device-side CFL pilots are not implemented yet. Keep GPU and other
    # non-Array backends on the existing single-pass path until a proper
    # reduction-based pilot is added.
    m isa Array || return 1
    Nx, Ny, Nz = size(m)
    mx = ws.m_buf
    mx_next = ws.rm_buf
    n_sub = _subcycling_pass_count(max_cfl_y(bm, m), cfl_limit)

    while true
        copyto!(mx, m)
        flux_scale = inv(FT(n_sub))
        cfl_ok = true

        for pass in 1:n_sub
            @inbounds for k in 1:Nz, j in 1:(Ny + 1), i in 1:Nx
                flux = flux_scale * bm[i, j, k]
                donor = flux >= zero(FT) ? max(j - 1, 1) : min(j, Ny)
                if abs(flux) >= cfl_limit * max(mx[i, donor, k], eps(FT))
                    cfl_ok = false
                    break
                end
            end
            cfl_ok || break

            if pass != n_sub
                @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
                    mx_next[i, j, k] = mx[i, j, k] + flux_scale * bm[i, j, k] - flux_scale * bm[i, j + 1, k]
                end
                mx, mx_next = mx_next, mx
            end
        end

        cfl_ok && return n_sub
        n_sub += 1
        n_sub <= max_n_sub || throw(ArgumentError("y-direction subcycling exceeded max_n_sub=$(max_n_sub)"))
    end
end

function _z_subcycling_pass_count(cm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                                  ws::AdvectionWorkspace{FT},
                                  cfl_limit::FT; max_n_sub::Int = 4096) where FT
    # Device-side CFL pilots are not implemented yet. Keep GPU and other
    # non-Array backends on the existing single-pass path until a proper
    # reduction-based pilot is added.
    m isa Array || return 1
    Nx, Ny, Nz = size(m)
    mx = ws.m_buf
    mx_next = ws.rm_buf
    n_sub = _subcycling_pass_count(max_cfl_z(cm, m), cfl_limit)

    while true
        copyto!(mx, m)
        flux_scale = inv(FT(n_sub))
        cfl_ok = true

        for pass in 1:n_sub
            @inbounds for k in 1:(Nz + 1), j in 1:Ny, i in 1:Nx
                flux = flux_scale * cm[i, j, k]
                donor = flux >= zero(FT) ? max(k - 1, 1) : min(k, Nz)
                if abs(flux) >= cfl_limit * max(mx[i, j, donor], eps(FT))
                    cfl_ok = false
                    break
                end
            end
            cfl_ok || break

            if pass != n_sub
                @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
                    mx_next[i, j, k] = mx[i, j, k] + flux_scale * cm[i, j, k] - flux_scale * cm[i, j, k + 1]
                end
                mx, mx_next = mx_next, mx
            end
        end

        cfl_ok && return n_sub
        n_sub += 1
        n_sub <= max_n_sub || throw(ArgumentError("z-direction subcycling exceeded max_n_sub=$(max_n_sub)"))
    end
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

# =========================================================================
# Strang splitting: X → Y → Z → Z → Y → X
# =========================================================================
#
# The Strang (1968) symmetric splitting sequence achieves second-order
# accuracy in the splitting error while using directional sweeps in the
# palindrome X → Y → Z → Z → Y → X.
#
# In `src_v2`, the kernels consume a fully prepared mass-flux state for the
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
    m_save = n_tr > 1 ? similar(m) : m
    if n_tr > 1
        copyto!(m_save, m)
    end

    for (idx, (name, rm_tracer)) in enumerate(pairs(state.tracers))
        if idx > 1
            copyto!(m, m_save)
        end

        _sweep_x_subcycled!(rm_tracer, m, am, scheme, workspace, cfl_limit_ft)
        _sweep_y_subcycled!(rm_tracer, m, bm, scheme, workspace, cfl_limit_ft)
        _sweep_z_subcycled!(rm_tracer, m, cm, scheme, workspace, cfl_limit_ft)
        _sweep_z_subcycled!(rm_tracer, m, cm, scheme, workspace, cfl_limit_ft)
        _sweep_y_subcycled!(rm_tracer, m, bm, scheme, workspace, cfl_limit_ft)
        _sweep_x_subcycled!(rm_tracer, m, am, scheme, workspace, cfl_limit_ft)
    end

    return nothing
end

function strang_split!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
                       grid::AtmosGrid{<:CubedSphereMesh},
                       scheme::Union{AbstractAdvection, AbstractAdvectionScheme};
                       workspace::AdvectionWorkspace) where {B <: AbstractMassBasis}
    throw(ArgumentError("CubedSphereMesh remains metadata-only in src_v2; structured advection is only supported on LatLonMesh until cubed-sphere geometry/connectivity are implemented"))
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
    throw(ArgumentError("CubedSphereMesh remains metadata-only in src_v2; structured advection is only supported on LatLonMesh until cubed-sphere geometry/connectivity are implemented"))
end

# Face-indexed apply! — shared tracer loop, generated for each topology
for (scheme_type, h_sweep, v_sweep) in (
    (:FirstOrderUpwindAdvection, :sweep_horizontal!, :sweep_vertical!),
    (:AbstractConstantScheme,    :sweep_horizontal!, :sweep_vertical!),
)
    @eval function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                          grid::AtmosGrid{<:AbstractHorizontalMesh},
                          scheme::$scheme_type, dt;
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

            $h_sweep(rm_tracer, m, hflux, grid.horizontal, scheme, workspace)
            $v_sweep(rm_tracer, m, cm, scheme, workspace)
            $v_sweep(rm_tracer, m, cm, scheme, workspace)
            $h_sweep(rm_tracer, m, hflux, grid.horizontal, scheme, workspace)
        end

        return nothing
    end
end

# Error stubs for unsupported face-indexed scheme families
for (scheme_type, label) in (
    (:AbstractLinearReconstruction,    "linear-reconstruction"),
    (:AbstractQuadraticReconstruction, "quadratic-reconstruction"),
    (:AbstractAdvection,               "this"),
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
        """
            $($sweep_fn)(rm_4d, m, flux, scheme, ws)

        Multi-tracer directional sweep operating on a 4D tracer array
        `rm_4d[Nx, Ny, Nz, Nt]`.  Launches a single kernel that processes
        all tracers, sharing the mass update computation.

        Uses `ws.rm_4d_buf` as the double buffer for the 4D tracer array
        and `ws.m_buf` for the 3D mass array.
        """
        function $sweep_fn(rm_4d::AbstractArray{FT,4}, m::AbstractArray{FT,3},
                           flux::AbstractArray{FT,3},
                           scheme::AbstractAdvectionScheme,
                           ws::AdvectionWorkspace{FT}) where FT
            backend = get_backend(m)
            Nt = Int32(size(rm_4d, 4))
            kernel! = $kernel_fn(backend, 256)
            kernel!(ws.rm_4d_buf, rm_4d, ws.m_buf, m, flux, scheme,
                    Int32(size(m, $dim)), Nt, one(FT);
                    ndrange=size(m))
            synchronize(backend)
            copyto!(rm_4d, ws.rm_4d_buf)
            copyto!(m, ws.m_buf)
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
                          ws::AdvectionWorkspace{FT}) where FT
    sweep_x_mt!(rm_4d, m, am, scheme, ws)
    sweep_y_mt!(rm_4d, m, bm, scheme, ws)
    sweep_z_mt!(rm_4d, m, cm, scheme, ws)
    sweep_z_mt!(rm_4d, m, cm, scheme, ws)
    sweep_y_mt!(rm_4d, m, bm, scheme, ws)
    sweep_x_mt!(rm_4d, m, am, scheme, ws)
    return nothing
end

export AdvectionWorkspace, strang_split!, strang_split_mt!
