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
