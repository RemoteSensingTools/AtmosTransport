# ---------------------------------------------------------------------------
# Lin-Rood adjoint kernels (Plan 25 — LinRood CS adjoint).
#
# Reverse-mode kernels for the LinRood cubed-sphere PPM path. Each kernel
# here is the discrete transpose of the matching forward kernel in
# `LinRood.jl` for fixed velocities (`am`, `bm`) — the tracer state and
# face mixing ratios are the differentiated inputs, the velocity is
# treated as a parameter from the meteo tape.
#
# The naming convention is `<forward_kernel>_adjoint!`. Each adjoint
# kernel takes adjoint arrays sized identically to the forward output
# (read-only) plus adjoint accumulators for each forward input
# (atomically incremented). The forward velocity fields are passed
# through unchanged.
#
# References:
#   docs/plans/25_LINROOD_ADJOINT/NOTES.md  — staged plan and derivations
#   docs/src/theory/adjoint_status.md       — shipped adjoint surface
# ---------------------------------------------------------------------------

# LinRood.jl imports `@kernel, @index, @Const, synchronize, get_backend`
# but not `@atomic` — adjoint face accumulators need it, so we pull it
# in alongside.
using KernelAbstractions: @atomic

# ---------------------------------------------------------------------------
# Adjoint of `_linrood_update_kernel!` (LinRood.jl)
#
# Forward (mass-space):
#   rm_new[ii, jj, k] = rm[ii, jj, k]
#                     + am_w * (fx_in[i,   j, k] + fx_out[i,   j, k]) / 2
#                     - am_e * (fx_in[i+1, j, k] + fx_out[i+1, j, k]) / 2
#                     + bm_s * (fy_in[i, j,   k] + fy_out[i, j,   k]) / 2
#                     - bm_n * (fy_in[i, j+1, k] + fy_out[i, j+1, k]) / 2
#   m_new [ii, jj, k] = m [ii, jj, k] + (am_w - am_e) + (bm_s - bm_n)
#
# For fixed velocities `(am, bm)`, the forward map
#   F :  (rm, m, fx_in, fx_out, fy_in, fy_out) -> (rm_new, m_new)
# is linear. Its transpose, applied to (lambda_rm_new, lambda_m_new),
# accumulates atomically into the six adjoint inputs.
#
# Coefficients (with `half = 0.5`):
#   ∂rm_new/∂rm[ii,jj,k]           = 1
#   ∂rm_new/∂fx_in [i,   j, k]     = +half · am_w
#   ∂rm_new/∂fx_out[i,   j, k]     = +half · am_w
#   ∂rm_new/∂fx_in [i+1, j, k]     = -half · am_e
#   ∂rm_new/∂fx_out[i+1, j, k]     = -half · am_e
#   ∂rm_new/∂fy_in [i, j,   k]     = +half · bm_s
#   ∂rm_new/∂fy_out[i, j,   k]     = +half · bm_s
#   ∂rm_new/∂fy_in [i, j+1, k]     = -half · bm_n
#   ∂rm_new/∂fy_out[i, j+1, k]     = -half · bm_n
#   ∂m_new /∂m [ii,jj,k]           = 1
# All other partials are zero. `m_new` has no tracer dependence; `rm_new`
# has no air-mass dependence. Face writes share neighbouring cells, so
# face accumulations use `@atomic`.
# ---------------------------------------------------------------------------

@kernel function _linrood_update_kernel_adjoint!(
    lambda_rm, lambda_m,
    lambda_fx_in, lambda_fx_out, lambda_fy_in, lambda_fy_out,
    @Const(lambda_rm_new), @Const(lambda_m_new),
    @Const(am), @Const(bm), Hp,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(lambda_rm_new)
        half = FT(0.5)

        bar_rm = lambda_rm_new[ii, jj, k]
        bar_m  = lambda_m_new[ii, jj, k]

        # rm[ii,jj,k] receives bar_rm; m[ii,jj,k] receives bar_m.
        # Each interior cell is touched by exactly one thread, so no
        # atomic needed for the cell-centre accumulations.
        lambda_rm[ii, jj, k] += bar_rm
        lambda_m[ii, jj, k]  += bar_m

        am_w = am[i, j, k]
        am_e = am[i + 1, j, k]
        bm_s = bm[i, j, k]
        bm_n = bm[i, j + 1, k]

        wx_w =  half * am_w * bar_rm
        wx_e = -half * am_e * bar_rm
        wy_s =  half * bm_s * bar_rm
        wy_n = -half * bm_n * bar_rm

        # Face writes: each face index can be touched by two interior
        # cells (left/right or below/above), hence the atomic adds.
        @atomic lambda_fx_in[i, j, k]      += wx_w
        @atomic lambda_fx_out[i, j, k]     += wx_w
        @atomic lambda_fx_in[i + 1, j, k]  += wx_e
        @atomic lambda_fx_out[i + 1, j, k] += wx_e
        @atomic lambda_fy_in[i, j, k]      += wy_s
        @atomic lambda_fy_out[i, j, k]     += wy_s
        @atomic lambda_fy_in[i, j + 1, k]  += wy_n
        @atomic lambda_fy_out[i, j + 1, k] += wy_n
    end
end

"""
    apply_linrood_update_adjoint!(lambda_rm, lambda_m,
                                   lambda_fx_in, lambda_fx_out,
                                   lambda_fy_in, lambda_fy_out,
                                   lambda_rm_new, lambda_m_new,
                                   am, bm, mesh)

Apply the discrete transpose of `_linrood_update_kernel!` for one panel:
accumulate the adjoint of `(rm_new, m_new)` into the adjoint inputs
`(rm, m, fx_in, fx_out, fy_in, fy_out)` for fixed velocities `(am, bm)`.

All `lambda_*` adjoint accumulators are read-and-modified (atomically for
face arrays) — callers are responsible for initialising them to zero
before the call. The kernel only touches interior `(i, j)` indices
`1..Nc`; halo cells of `lambda_rm`/`lambda_m` and face cells outside
`1..Nc+1` are left untouched.
"""
function apply_linrood_update_adjoint!(lambda_rm, lambda_m,
                                       lambda_fx_in, lambda_fx_out,
                                       lambda_fy_in, lambda_fy_out,
                                       lambda_rm_new, lambda_m_new,
                                       am, bm,
                                       mesh::CubedSphereMesh)
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(lambda_rm_new, 3)
    backend = get_backend(lambda_rm)
    k! = _linrood_update_kernel_adjoint!(backend, 256)
    k!(lambda_rm, lambda_m,
       lambda_fx_in, lambda_fx_out, lambda_fy_in, lambda_fy_out,
       lambda_rm_new, lambda_m_new, am, bm, Hp;
       ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    return nothing
end

# ---------------------------------------------------------------------------
# Adjoint of `_pre_advect_y_kernel!` (LinRood.jl:364).
#
# Forward:
#   bm_s   = bm[i, j, k]
#   bm_n   = bm[i, j+1, k]
#   rm_new = rm[ii, jj, k] + bm_s · fy_face[i, j, k]
#                          - bm_n · fy_face[i, j+1, k]
#   m_new  = m [ii, jj, k] + bm_s - bm_n
#   q_i[ii, jj, k] = m_new > thresh ? rm_new / m_new : 0
#
# where `thresh = 100 · eps(FT)` mirrors `_safe_mixing_ratio`. For
# `m_new > thresh` the operator is smooth in `(rm, m, fy_face)`; below
# threshold the output is exactly zero, so all adjoint contributions are
# zero. The adjoint then is:
#
#   inv_m_new = 1 / m_new      (when m_new > thresh, else 0)
#   lambda_rm     [ii, jj, k]   += lambda_q_i · inv_m_new
#   lambda_fy_face[i, j, k]     += lambda_q_i · bm_s · inv_m_new
#   lambda_fy_face[i, j+1, k]   += lambda_q_i · (-bm_n) · inv_m_new
#   lambda_m      [ii, jj, k]   += lambda_q_i · (-q_i) · inv_m_new
#
# Face writes share neighbour cells along the Y direction, hence the
# `@atomic` accumulation on `lambda_fy_face`. Cell-centred writes are
# unique per thread.
# ---------------------------------------------------------------------------

@kernel function _pre_advect_y_kernel_adjoint!(
    lambda_rm, lambda_m, lambda_fy_face,
    @Const(lambda_q_i), @Const(rm), @Const(m), @Const(bm), @Const(fy_face), Hp,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(lambda_q_i)
        thresh = FT(100) * eps(FT)

        bm_s = bm[i, j,     k]
        bm_n = bm[i, j + 1, k]
        m_new = m[ii, jj, k] + bm_s - bm_n

        if m_new > thresh
            rm_new = rm[ii, jj, k] +
                     bm_s * fy_face[i, j, k] - bm_n * fy_face[i, j + 1, k]
            inv_m_new = one(FT) / m_new
            q_i = rm_new * inv_m_new

            bar = lambda_q_i[ii, jj, k]
            scaled = bar * inv_m_new

            lambda_rm[ii, jj, k] += scaled
            lambda_m[ii, jj, k]  += -q_i * scaled

            @atomic lambda_fy_face[i, j,     k] +=  bm_s * scaled
            @atomic lambda_fy_face[i, j + 1, k] += -bm_n * scaled
        end
        # m_new <= thresh: q_i = 0 deterministically, so gradient = 0.
    end
end

"""
    apply_pre_advect_y_adjoint!(lambda_rm, lambda_m, lambda_fy_face,
                                  lambda_q_i, rm, m, bm, fy_face, mesh)

Discrete transpose of `_pre_advect_y_kernel!` for one panel: accumulate
`lambda_q_i` into the adjoint accumulators of `(rm, m, fy_face)` for
fixed velocity `bm`. The small-`m_new` zeroing exactly mirrors
`_safe_mixing_ratio` (LinRood-style 100·eps threshold).

All `lambda_*` accumulators are read-modified; callers initialise them
to zero before the call. Face writes use `@atomic` for shared
neighbour-cell accumulation along Y.
"""
function apply_pre_advect_y_adjoint!(lambda_rm, lambda_m, lambda_fy_face,
                                     lambda_q_i, rm, m, bm, fy_face,
                                     mesh::CubedSphereMesh)
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(lambda_q_i, 3)
    backend = get_backend(lambda_rm)
    k! = _pre_advect_y_kernel_adjoint!(backend, 256)
    k!(lambda_rm, lambda_m, lambda_fy_face,
       lambda_q_i, rm, m, bm, fy_face, Hp;
       ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    return nothing
end

# ---------------------------------------------------------------------------
# Adjoint of `_pre_advect_x_kernel!` (LinRood.jl:377).
#
# Identical structure to `_pre_advect_y_kernel!` with the directions
# transposed: `am`/`fx_face` replace `bm`/`fy_face`, and the face
# neighbour is `(i+1, j, k)` rather than `(i, j+1, k)`.
# ---------------------------------------------------------------------------

@kernel function _pre_advect_x_kernel_adjoint!(
    lambda_rm, lambda_m, lambda_fx_face,
    @Const(lambda_q_j), @Const(rm), @Const(m), @Const(am), @Const(fx_face), Hp,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(lambda_q_j)
        thresh = FT(100) * eps(FT)

        am_w = am[i,     j, k]
        am_e = am[i + 1, j, k]
        m_new = m[ii, jj, k] + am_w - am_e

        if m_new > thresh
            rm_new = rm[ii, jj, k] +
                     am_w * fx_face[i, j, k] - am_e * fx_face[i + 1, j, k]
            inv_m_new = one(FT) / m_new
            q_j = rm_new * inv_m_new

            bar = lambda_q_j[ii, jj, k]
            scaled = bar * inv_m_new

            lambda_rm[ii, jj, k] += scaled
            lambda_m[ii, jj, k]  += -q_j * scaled

            @atomic lambda_fx_face[i,     j, k] +=  am_w * scaled
            @atomic lambda_fx_face[i + 1, j, k] += -am_e * scaled
        end
    end
end

"""
    apply_pre_advect_x_adjoint!(lambda_rm, lambda_m, lambda_fx_face,
                                  lambda_q_j, rm, m, am, fx_face, mesh)

Discrete transpose of `_pre_advect_x_kernel!` for one panel. See
`apply_pre_advect_y_adjoint!` for the contract — same structure with X
substituted for Y.
"""
function apply_pre_advect_x_adjoint!(lambda_rm, lambda_m, lambda_fx_face,
                                     lambda_q_j, rm, m, am, fx_face,
                                     mesh::CubedSphereMesh)
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(lambda_q_j, 3)
    backend = get_backend(lambda_rm)
    k! = _pre_advect_x_kernel_adjoint!(backend, 256)
    k!(lambda_rm, lambda_m, lambda_fx_face,
       lambda_q_j, rm, m, am, fx_face, Hp;
       ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    return nothing
end
