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

# ===========================================================================
# Plan 25 Commit 3 — Adjoints of the two `_from_q` PPM face kernels (ORD=5)
#
# The forward kernels `_ppm_x_face_from_q_kernel!` and
# `_ppm_y_face_from_q_kernel!` (LinRood.jl:299, 325) compute a PPM
# parabolic-integral face value from a 6-cell q stencil at fixed
# velocity (am/bm). Both share the chain
#   _ppm_edge_values → _apply_monotonicity → _ppm_face_value
# where each step is piecewise-smooth in q. The composition is rational
# and branch-rich, so we differentiate via a small 6-cell forward-AD
# wrapper `D6{FT}` and propagate `(value, ∂/∂q_n)` pairs through the
# chain. The resulting 6-tuple of face partials is then multiplied by
# the adjoint seed `lambda_face` and atomically accumulated into
# `lambda_q` at the six stencil cells.
# ===========================================================================

# D6 value-tangent pair: `value` plus a 6-component gradient w.r.t. the
# six PPM stencil cells. Immutable + tuple-of-FT, so it works
# unchanged on CPU and CUDA backends.
struct D6{FT}
    v :: FT
    g :: NTuple{6, FT}
end

@inline _d6_const(::Type{FT}, x::FT) where {FT} =
    D6{FT}(x, ntuple(_ -> zero(FT), Val(6)))
@inline _d6_var(x::FT, ::Val{N}) where {FT, N} =
    D6{FT}(x, ntuple(i -> i == N ? one(FT) : zero(FT), Val(6)))
@inline _d6_pack(v::FT, g::NTuple{6, FT}) where {FT} = D6{FT}(v, g)

@inline Base.:+(a::D6{FT}, b::D6{FT}) where {FT} = _d6_pack(a.v + b.v, a.g .+ b.g)
@inline Base.:+(a::D6{FT}, b::FT) where {FT}     = _d6_pack(a.v + b, a.g)
@inline Base.:+(a::FT, b::D6{FT}) where {FT}     = _d6_pack(a + b.v, b.g)
@inline Base.:-(a::D6{FT}, b::D6{FT}) where {FT} = _d6_pack(a.v - b.v, a.g .- b.g)
@inline Base.:-(a::D6{FT}, b::FT) where {FT}     = _d6_pack(a.v - b, a.g)
@inline Base.:-(a::FT, b::D6{FT}) where {FT}     = _d6_pack(a - b.v, .-b.g)
@inline Base.:-(a::D6{FT}) where {FT}            = _d6_pack(-a.v, .-a.g)
@inline Base.:*(a::D6{FT}, b::FT) where {FT}     = _d6_pack(a.v * b, a.g .* b)
@inline Base.:*(a::FT, b::D6{FT}) where {FT}     = _d6_pack(a * b.v, b.g .* a)
@inline Base.:*(a::D6{FT}, b::D6{FT}) where {FT} =
    _d6_pack(a.v * b.v, a.v .* b.g .+ b.v .* a.g)
@inline Base.:/(a::D6{FT}, b::FT) where {FT}     = _d6_pack(a.v / b, a.g ./ b)
@inline function Base.:/(a::D6{FT}, b::D6{FT}) where {FT}
    inv_bv = one(FT) / b.v
    qv = a.v * inv_bv
    qg = inv_bv .* (a.g .- qv .* b.g)
    return _d6_pack(qv, qg)
end
@inline Base.abs(a::D6{FT}) where {FT} =
    a.v >= zero(FT) ? a : _d6_pack(-a.v, .-a.g)

# Forward-only comparisons: branches are taken on `.v`. Used by
# `huynh_second_constraint_d6`, `apply_monotonicity_d6`,
# `ppm_face_value_d6` to mirror the forward branch decisions.

# ---------------------------------------------------------------------------
# d6-AD versions of the LinRood forward chain helpers.
#
# Each function below has the same name as its forward counterpart with
# a `_d6` suffix and accepts `D6{FT}` arguments. Numerical thresholds
# (`10·eps(FT)`, `100·eps(FT)`) come from the forward implementations
# in `ppm_subgrid_distributions.jl` and `LinRood.jl`.
# ---------------------------------------------------------------------------

@inline function _huynh_second_constraint_d6(
    q_l::D6{FT}, q_c::D6{FT}, q_r::D6{FT},
    q_LL::D6{FT}, q_RR::D6{FT},
) where {FT}
    _ = (q_LL, q_RR)  # kept for forward-signature parity; not used by the formula
    denom = q_r - q_l
    if abs(denom.v) < FT(10) * eps(FT)
        return _d6_const(FT, zero(FT))
    end
    # q_6 = 3 · (q_c − (2·q_l + q_r) / 3)
    q6 = FT(3) * (q_c - (FT(2) * q_l + q_r) * (one(FT) / FT(3)))
    abs_denom = abs(denom)               # D6: value = |denom.v|, grad = sign(denom.v)·denom.g
    if q6.v > abs_denom.v
        # Clamp triggered at +mag. `mag = abs(denom)` carries the q_r/q_l
        # gradient.
        return abs_denom
    elseif q6.v < -abs_denom.v
        return -abs_denom
    else
        return q6
    end
end

@inline function _ppm_edge_values_ord5_d6(
    q_imm::D6{FT}, q_im::D6{FT}, q_i::D6{FT}, q_ip::D6{FT}, q_ipp::D6{FT},
) where {FT}
    s_im = _huynh_second_constraint_d6(q_im, q_i, q_i, q_imm, q_ip)
    s_i  = _huynh_second_constraint_d6(q_i, q_ip, q_ip, q_im, q_ipp)
    half = one(FT) / FT(2)
    q_L = q_i - s_im * half
    q_R = q_i + s_i  * half
    return (q_L, q_R)
end

@inline function _apply_monotonicity_d6(
    q_L::D6{FT}, q_R::D6{FT}, c::D6{FT},
) where {FT}
    diff_R = q_R - c
    diff_L = c - q_L
    if (diff_R.v * diff_L.v) <= zero(FT)
        return (c, c)
    end
    return (q_L, q_R)
end

# `_ppm_face_value` (LinRood.jl:215) with the donor-mass denominator held
# constant. For LinRood adjoint Commit 3 the velocity tape supplies
# fixed `(F, m_lo, m_hi)`; the d6 tangent only propagates the q-stencil
# sensitivities.
@inline function _ppm_face_value_d6(
    F::FT, m_lo::FT, m_hi::FT,
    c_lo::D6{FT}, c_hi::D6{FT},
    q_L_lo::D6{FT}, q_R_lo::D6{FT},
    q_L_hi::D6{FT}, q_R_hi::D6{FT},
) where {FT}
    m_floor = FT(100) * eps(FT)
    if F >= zero(FT)
        alpha = m_lo > m_floor ? F / m_lo : zero(FT)
        bl = q_L_lo - c_lo
        br = q_R_lo - c_lo
        b0 = bl + br
        return c_lo + (one(FT) - alpha) * (br - alpha * b0)
    else
        alpha = m_hi > m_floor ? F / m_hi : zero(FT)
        bl = q_L_hi - c_hi
        br = q_R_hi - c_hi
        b0 = bl + br
        return c_hi + (one(FT) + alpha) * (bl + alpha * b0)
    end
end

# Full chain on a 6-cell stencil of q values. Returns the 6-component
# gradient `∂face/∂q_n` for n = -3, -2, -1, 0, +1, +2.
@inline function _linrood_ppm_face_from_q_grad_ord5(
    F::FT, m_l::FT, m_r::FT,
    q_m3::FT, q_m2::FT, q_m1::FT, q_0::FT, q_p1::FT, q_p2::FT,
) where {FT}
    c_m3 = _d6_var(q_m3, Val(1))
    c_m2 = _d6_var(q_m2, Val(2))
    c_m1 = _d6_var(q_m1, Val(3))
    c_0  = _d6_var(q_0,  Val(4))
    c_p1 = _d6_var(q_p1, Val(5))
    c_p2 = _d6_var(q_p2, Val(6))

    q_L_m, q_R_m = _ppm_edge_values_ord5_d6(c_m3, c_m2, c_m1, c_0, c_p1)
    q_L_0, q_R_0 = _ppm_edge_values_ord5_d6(c_m2, c_m1, c_0, c_p1, c_p2)
    q_L_m, q_R_m = _apply_monotonicity_d6(q_L_m, q_R_m, c_m1)
    q_L_0, q_R_0 = _apply_monotonicity_d6(q_L_0, q_R_0, c_0)

    face = _ppm_face_value_d6(F, m_l, m_r, c_m1, c_0,
                              q_L_m, q_R_m, q_L_0, q_R_0)
    return face.g  # NTuple{6, FT} = (∂f/∂q_m3, ..., ∂f/∂q_p2)
end

# ---------------------------------------------------------------------------
# Adjoint of `_ppm_x_face_from_q_kernel!` (LinRood.jl:299) for ORD=5.
#
# Forward: `fx_face[iif, j, k] = _ppm_face_value(am[iif, j, k], …,
# q[ii_l-2..ii_r+2, jj, k])`. The stencil spans 6 cells in i at fixed j.
# Adjoint accumulates lambda_fx_face[iif, j, k] into lambda_q at those
# six stencil cells via `@atomic` (multiple faces can write to the same
# cell).
# ---------------------------------------------------------------------------

@kernel function _ppm_x_face_from_q_kernel_adjoint_ord5!(
    lambda_q,
    @Const(lambda_fx_face), @Const(q), @Const(am), @Const(m),
    Hp, Nc,
)
    iif, j, k = @index(Global, NTuple)
    _ = Nc  # signature parity with forward kernels — only used at dispatch
    @inbounds begin
        jj   = Hp + j
        ii_l = Hp + iif - 1
        ii_r = Hp + iif

        q_m3 = q[ii_l - 2, jj, k]
        q_m2 = q[ii_l - 1, jj, k]
        q_m1 = q[ii_l,     jj, k]
        q_0  = q[ii_r,     jj, k]
        q_p1 = q[ii_r + 1, jj, k]
        q_p2 = q[ii_r + 2, jj, k]

        F   = am[iif, j, k]
        m_l = m[ii_l, jj, k]
        m_r = m[ii_r, jj, k]

        grad = _linrood_ppm_face_from_q_grad_ord5(F, m_l, m_r,
                                                  q_m3, q_m2, q_m1, q_0, q_p1, q_p2)
        bar = lambda_fx_face[iif, j, k]

        @atomic lambda_q[ii_l - 2, jj, k] += bar * grad[1]
        @atomic lambda_q[ii_l - 1, jj, k] += bar * grad[2]
        @atomic lambda_q[ii_l,     jj, k] += bar * grad[3]
        @atomic lambda_q[ii_r,     jj, k] += bar * grad[4]
        @atomic lambda_q[ii_r + 1, jj, k] += bar * grad[5]
        @atomic lambda_q[ii_r + 2, jj, k] += bar * grad[6]
    end
end

"""
    apply_ppm_x_face_from_q_adjoint!(lambda_q, lambda_fx_face, q, am, m,
                                       mesh, ::Val{ORD})

Discrete transpose of `_ppm_x_face_from_q_kernel!` for one panel at
ORD=5 (LinRoodPPMScheme default). The donor-mass denominator
`m_l`/`m_r` in `_ppm_face_value` and the velocity `am` are treated as
fixed parameters from the tape — the adjoint propagates only the
q-stencil sensitivity. Atomic writes on `lambda_q` because multiple
faces share each cell.
"""
function apply_ppm_x_face_from_q_adjoint!(lambda_q, lambda_fx_face,
                                          q, am, m,
                                          mesh::CubedSphereMesh,
                                          ::Val{ORD}=Val(5)) where {ORD}
    ORD == 5 || throw(ArgumentError(
        "Plan-25 Commit 3 implements ORD=5 only; ORD=$ORD lands in Commit 3b"))
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(q, 3)
    backend = get_backend(lambda_q)
    k! = _ppm_x_face_from_q_kernel_adjoint_ord5!(backend, 256)
    k!(lambda_q, lambda_fx_face, q, am, m, Hp, Nc;
       ndrange=(Nc + 1, Nc, Nz))
    synchronize(backend)
    return nothing
end

# ---------------------------------------------------------------------------
# Adjoint of `_ppm_y_face_from_q_kernel!` (LinRood.jl:325) for ORD=5.
#
# Same chain as the X variant with the stencil running along j at fixed
# i.
# ---------------------------------------------------------------------------

@kernel function _ppm_y_face_from_q_kernel_adjoint_ord5!(
    lambda_q,
    @Const(lambda_fy_face), @Const(q), @Const(bm), @Const(m),
    Hp, Nc,
)
    i, jf, k = @index(Global, NTuple)
    _ = Nc
    @inbounds begin
        ii   = Hp + i
        jj_b = Hp + jf - 1
        jj_a = Hp + jf

        q_m3 = q[ii, jj_b - 2, k]
        q_m2 = q[ii, jj_b - 1, k]
        q_m1 = q[ii, jj_b,     k]
        q_0  = q[ii, jj_a,     k]
        q_p1 = q[ii, jj_a + 1, k]
        q_p2 = q[ii, jj_a + 2, k]

        F   = bm[i, jf, k]
        m_l = m[ii, jj_b, k]
        m_r = m[ii, jj_a, k]

        grad = _linrood_ppm_face_from_q_grad_ord5(F, m_l, m_r,
                                                  q_m3, q_m2, q_m1, q_0, q_p1, q_p2)
        bar = lambda_fy_face[i, jf, k]

        @atomic lambda_q[ii, jj_b - 2, k] += bar * grad[1]
        @atomic lambda_q[ii, jj_b - 1, k] += bar * grad[2]
        @atomic lambda_q[ii, jj_b,     k] += bar * grad[3]
        @atomic lambda_q[ii, jj_a,     k] += bar * grad[4]
        @atomic lambda_q[ii, jj_a + 1, k] += bar * grad[5]
        @atomic lambda_q[ii, jj_a + 2, k] += bar * grad[6]
    end
end

"""
    apply_ppm_y_face_from_q_adjoint!(lambda_q, lambda_fy_face, q, bm, m,
                                       mesh, ::Val{ORD})

Discrete transpose of `_ppm_y_face_from_q_kernel!` for one panel at
ORD=5. See `apply_ppm_x_face_from_q_adjoint!` for the contract.
"""
function apply_ppm_y_face_from_q_adjoint!(lambda_q, lambda_fy_face,
                                          q, bm, m,
                                          mesh::CubedSphereMesh,
                                          ::Val{ORD}=Val(5)) where {ORD}
    ORD == 5 || throw(ArgumentError(
        "Plan-25 Commit 3 implements ORD=5 only; ORD=$ORD lands in Commit 3b"))
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(q, 3)
    backend = get_backend(lambda_q)
    k! = _ppm_y_face_from_q_kernel_adjoint_ord5!(backend, 256)
    k!(lambda_q, lambda_fy_face, q, bm, m, Hp, Nc;
       ndrange=(Nc, Nc + 1, Nz))
    synchronize(backend)
    return nothing
end

# ===========================================================================
# Plan 25 Commit 3b — Adjoints of the rm-input PPM face kernels (ORD=5)
#
# Forward kernels `_ppm_x_face_kernel!` and `_ppm_y_face_kernel!`
# (LinRood.jl:241, 270) at ORD=5 fold `_safe_mixing_ratio` into the
# face computation: `c_n = rm_n / m_n` (zero below the
# `100·eps(FT)` threshold) feeds the same downstream
# `_ppm_edge_values_ord5 → _apply_monotonicity → _ppm_face_value`
# chain as the `_from_q` variants. The adjoint therefore needs to
# distribute the face seed into BOTH `lambda_rm` and `lambda_m` at
# the six stencil cells, with the donor-cell `m_donor` additionally
# feeding `_ppm_face_value` directly via `α = F / m_donor`.
#
# Strategy: run the d6 chain TWICE, once with the rm tangent
# `dc_n = 1/m_n · e_n` and once with the m tangent
# `dc_n = -rm_n / m_n² · e_n`. The first run returns `∂f/∂rm_n`; the
# second returns the chain-rule part of `∂f/∂m_n` (i.e., the
# `c = rm/m` coupling). The donor-cell m_donor additionally
# contributes `∂α/∂m_donor = -F / m_donor²` (when above threshold)
# which we add analytically.
# ===========================================================================

# d6-AD safe-mixing-ratio: returns the D6{FT} value `rm_n / m_n` with the
# requested tangent. Mirrors the forward `_safe_mixing_ratio`
# 100·eps threshold by returning a zero-gradient D6 when m_n is too
# small.
@inline function _safe_mixing_ratio_d6(rm_n::FT, m_n::FT,
                                        tangent::NTuple{6, FT}) where {FT}
    if m_n > FT(100) * eps(FT)
        return _d6_pack(rm_n / m_n, tangent)
    else
        return _d6_const(FT, zero(FT))
    end
end

# Pre-compute the rm-input chain once at a state; return the 6-tuple
# face partials w.r.t. one cell-attribute (either rm or m), driven by
# the input `tangents` (one length-6 tuple per stencil cell).
@inline function _linrood_ppm_face_chain_rm_ord5(
    F::FT, m_l::FT, m_r::FT,
    rm_m3::FT, rm_m2::FT, rm_m1::FT, rm_0::FT, rm_p1::FT, rm_p2::FT,
    m_m3::FT,  m_m2::FT,  m_m1::FT,  m_0::FT,  m_p1::FT,  m_p2::FT,
    tan_m3::NTuple{6, FT}, tan_m2::NTuple{6, FT}, tan_m1::NTuple{6, FT},
    tan_0::NTuple{6, FT},  tan_p1::NTuple{6, FT}, tan_p2::NTuple{6, FT},
) where {FT}
    c_m3 = _safe_mixing_ratio_d6(rm_m3, m_m3, tan_m3)
    c_m2 = _safe_mixing_ratio_d6(rm_m2, m_m2, tan_m2)
    c_m1 = _safe_mixing_ratio_d6(rm_m1, m_m1, tan_m1)
    c_0  = _safe_mixing_ratio_d6(rm_0,  m_0,  tan_0)
    c_p1 = _safe_mixing_ratio_d6(rm_p1, m_p1, tan_p1)
    c_p2 = _safe_mixing_ratio_d6(rm_p2, m_p2, tan_p2)

    q_L_m, q_R_m = _ppm_edge_values_ord5_d6(c_m3, c_m2, c_m1, c_0, c_p1)
    q_L_0, q_R_0 = _ppm_edge_values_ord5_d6(c_m2, c_m1, c_0, c_p1, c_p2)
    q_L_m, q_R_m = _apply_monotonicity_d6(q_L_m, q_R_m, c_m1)
    q_L_0, q_R_0 = _apply_monotonicity_d6(q_L_0, q_R_0, c_0)
    face = _ppm_face_value_d6(F, m_l, m_r, c_m1, c_0,
                              q_L_m, q_R_m, q_L_0, q_R_0)
    return face.g
end

# Full rm-input face Jacobian: returns
# `(∂f/∂rm_-3..p2, ∂f/∂m_-3..p2)` for the six stencil cells.
@inline function _linrood_ppm_face_from_rm_grad_ord5(
    F::FT, m_l::FT, m_r::FT,
    rm_m3::FT, rm_m2::FT, rm_m1::FT, rm_0::FT, rm_p1::FT, rm_p2::FT,
    m_m3::FT,  m_m2::FT,  m_m1::FT,  m_0::FT,  m_p1::FT,  m_p2::FT,
) where {FT}
    floor_thresh = FT(100) * eps(FT)
    # rm pass: dc_n = (1/m_n) · e_n   (zero if m_n below threshold)
    inv_m_m3 = m_m3 > floor_thresh ? one(FT) / m_m3 : zero(FT)
    inv_m_m2 = m_m2 > floor_thresh ? one(FT) / m_m2 : zero(FT)
    inv_m_m1 = m_m1 > floor_thresh ? one(FT) / m_m1 : zero(FT)
    inv_m_0  = m_0  > floor_thresh ? one(FT) / m_0  : zero(FT)
    inv_m_p1 = m_p1 > floor_thresh ? one(FT) / m_p1 : zero(FT)
    inv_m_p2 = m_p2 > floor_thresh ? one(FT) / m_p2 : zero(FT)

    tan_rm_m3 = ntuple(i -> i == 1 ? inv_m_m3 : zero(FT), Val(6))
    tan_rm_m2 = ntuple(i -> i == 2 ? inv_m_m2 : zero(FT), Val(6))
    tan_rm_m1 = ntuple(i -> i == 3 ? inv_m_m1 : zero(FT), Val(6))
    tan_rm_0  = ntuple(i -> i == 4 ? inv_m_0  : zero(FT), Val(6))
    tan_rm_p1 = ntuple(i -> i == 5 ? inv_m_p1 : zero(FT), Val(6))
    tan_rm_p2 = ntuple(i -> i == 6 ? inv_m_p2 : zero(FT), Val(6))

    grad_rm = _linrood_ppm_face_chain_rm_ord5(
        F, m_l, m_r,
        rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
        m_m3, m_m2, m_m1, m_0, m_p1, m_p2,
        tan_rm_m3, tan_rm_m2, tan_rm_m1, tan_rm_0, tan_rm_p1, tan_rm_p2)

    # m pass: dc_n = (-rm_n / m_n²) · e_n
    neg_rm_over_m2_m3 = m_m3 > floor_thresh ? -rm_m3 / (m_m3 * m_m3) : zero(FT)
    neg_rm_over_m2_m2 = m_m2 > floor_thresh ? -rm_m2 / (m_m2 * m_m2) : zero(FT)
    neg_rm_over_m2_m1 = m_m1 > floor_thresh ? -rm_m1 / (m_m1 * m_m1) : zero(FT)
    neg_rm_over_m2_0  = m_0  > floor_thresh ? -rm_0  / (m_0  * m_0)  : zero(FT)
    neg_rm_over_m2_p1 = m_p1 > floor_thresh ? -rm_p1 / (m_p1 * m_p1) : zero(FT)
    neg_rm_over_m2_p2 = m_p2 > floor_thresh ? -rm_p2 / (m_p2 * m_p2) : zero(FT)

    tan_m_m3 = ntuple(i -> i == 1 ? neg_rm_over_m2_m3 : zero(FT), Val(6))
    tan_m_m2 = ntuple(i -> i == 2 ? neg_rm_over_m2_m2 : zero(FT), Val(6))
    tan_m_m1 = ntuple(i -> i == 3 ? neg_rm_over_m2_m1 : zero(FT), Val(6))
    tan_m_0  = ntuple(i -> i == 4 ? neg_rm_over_m2_0  : zero(FT), Val(6))
    tan_m_p1 = ntuple(i -> i == 5 ? neg_rm_over_m2_p1 : zero(FT), Val(6))
    tan_m_p2 = ntuple(i -> i == 6 ? neg_rm_over_m2_p2 : zero(FT), Val(6))

    grad_m_chain = _linrood_ppm_face_chain_rm_ord5(
        F, m_l, m_r,
        rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
        m_m3, m_m2, m_m1, m_0, m_p1, m_p2,
        tan_m_m3, tan_m_m2, tan_m_m1, tan_m_0, tan_m_p1, tan_m_p2)

    # Donor-mass alpha contribution. Forward `_ppm_face_value` uses
    #   α = F / m_donor   (above threshold; else 0)
    # where m_donor = m_l when F ≥ 0 (donor is the cell to the "lo"
    # side of the face, i.e. stencil position 3 = c_m1) and m_donor =
    # m_r when F < 0 (donor = stencil position 4 = c_0). The chain
    # rule contribution
    #   ∂face/∂m_donor |_{α-only} = (∂face/∂α) · (∂α/∂m_donor)
    # adds to the corresponding cell's `∂f/∂m`. We compute
    # ∂face/∂α analytically by differentiating the parabolic-integral
    # form once more.
    # Recompute the limited (q_L, q_R) at the donor cell using the
    # forward chain on plain FT values so we can build the α
    # contribution exactly.
    extra_m_m1 = zero(FT)
    extra_m_0  = zero(FT)
    if F >= zero(FT)
        if m_m1 > floor_thresh
            bl_lo, br_lo, b0_lo, _ = _ppm_face_value_donor_state_lo(
                F, rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
                m_m3, m_m2, m_m1, m_0, m_p1, m_p2)
            _ = bl_lo  # forward chain reads bl, br, b0; ∂face/∂α only uses br and b0
            alpha = F / m_m1
            # face = c_lo + (1 - α)(br_lo - α·b0_lo)
            #       ∂face/∂α = -(br_lo - α·b0_lo) + (1 - α)·(-b0_lo)
            #                = -br_lo + (2α - 1)·b0_lo
            dface_dalpha = -br_lo + (FT(2) * alpha - one(FT)) * b0_lo
            dalpha_dm   = -F / (m_m1 * m_m1)
            extra_m_m1 = dface_dalpha * dalpha_dm
        end
    else
        if m_0 > floor_thresh
            bl_hi, _, b0_hi, _ = _ppm_face_value_donor_state_hi(
                F, rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
                m_m3, m_m2, m_m1, m_0, m_p1, m_p2)
            alpha = F / m_0
            # face = c_hi + (1 + α)(bl_hi + α·b0_hi)
            #       ∂face/∂α = (bl_hi + α·b0_hi) + (1 + α)·b0_hi
            #                = bl_hi + b0_hi + 2·α·b0_hi
            dface_dalpha = bl_hi + b0_hi + FT(2) * alpha * b0_hi
            dalpha_dm   = -F / (m_0 * m_0)
            extra_m_0 = dface_dalpha * dalpha_dm
        end
    end

    grad_m = (grad_m_chain[1], grad_m_chain[2],
              grad_m_chain[3] + extra_m_m1,
              grad_m_chain[4] + extra_m_0,
              grad_m_chain[5], grad_m_chain[6])

    return (grad_rm, grad_m)
end

# Helper: compute the forward parabolic-integral coefficients
# `(bl, br, b0, c)` at the donor cell for the F >= 0 branch.
# Mirrors `_ppm_face_value` exactly (no d6 propagation).
@inline function _ppm_face_value_donor_state_lo(
    F::FT,
    rm_m3::FT, rm_m2::FT, rm_m1::FT, rm_0::FT, rm_p1::FT, rm_p2::FT,
    m_m3::FT,  m_m2::FT,  m_m1::FT,  m_0::FT,  m_p1::FT,  m_p2::FT,
) where {FT}
    _ = F
    _ = (rm_p2, m_p2)  # not part of the q_L_m/q_R_m stencil
    floor_thresh = FT(100) * eps(FT)
    c_m3v = m_m3 > floor_thresh ? rm_m3 / m_m3 : zero(FT)
    c_m2v = m_m2 > floor_thresh ? rm_m2 / m_m2 : zero(FT)
    c_m1v = m_m1 > floor_thresh ? rm_m1 / m_m1 : zero(FT)
    c_0v  = m_0  > floor_thresh ? rm_0  / m_0  : zero(FT)
    c_p1v = m_p1 > floor_thresh ? rm_p1 / m_p1 : zero(FT)
    q_L_m_v, q_R_m_v = _ppm_edge_values(c_m3v, c_m2v, c_m1v, c_0v, c_p1v, Val(5))
    q_L_m_v, q_R_m_v = _apply_monotonicity(q_L_m_v, q_R_m_v, c_m1v)
    bl = q_L_m_v - c_m1v
    br = q_R_m_v - c_m1v
    b0 = bl + br
    return (bl, br, b0, c_m1v)
end

@inline function _ppm_face_value_donor_state_hi(
    F::FT,
    rm_m3::FT, rm_m2::FT, rm_m1::FT, rm_0::FT, rm_p1::FT, rm_p2::FT,
    m_m3::FT,  m_m2::FT,  m_m1::FT,  m_0::FT,  m_p1::FT,  m_p2::FT,
) where {FT}
    _ = F
    _ = (rm_m3, m_m3)  # not part of the q_L_0/q_R_0 stencil
    floor_thresh = FT(100) * eps(FT)
    c_m2v = m_m2 > floor_thresh ? rm_m2 / m_m2 : zero(FT)
    c_m1v = m_m1 > floor_thresh ? rm_m1 / m_m1 : zero(FT)
    c_0v  = m_0  > floor_thresh ? rm_0  / m_0  : zero(FT)
    c_p1v = m_p1 > floor_thresh ? rm_p1 / m_p1 : zero(FT)
    c_p2v = m_p2 > floor_thresh ? rm_p2 / m_p2 : zero(FT)
    q_L_0_v, q_R_0_v = _ppm_edge_values(c_m2v, c_m1v, c_0v, c_p1v, c_p2v, Val(5))
    q_L_0_v, q_R_0_v = _apply_monotonicity(q_L_0_v, q_R_0_v, c_0v)
    bl = q_L_0_v - c_0v
    br = q_R_0_v - c_0v
    b0 = bl + br
    return (bl, br, b0, c_0v)
end

# ---------------------------------------------------------------------------
# rm-input face kernels (X and Y, ORD=5).
# ---------------------------------------------------------------------------

@kernel function _ppm_x_face_kernel_adjoint_ord5!(
    lambda_rm, lambda_m,
    @Const(lambda_fx_face), @Const(rm), @Const(m), @Const(am),
    Hp, Nc,
)
    iif, j, k = @index(Global, NTuple)
    _ = Nc
    @inbounds begin
        jj   = Hp + j
        ii_l = Hp + iif - 1
        ii_r = Hp + iif

        rm_m3 = rm[ii_l - 2, jj, k]; m_m3 = m[ii_l - 2, jj, k]
        rm_m2 = rm[ii_l - 1, jj, k]; m_m2 = m[ii_l - 1, jj, k]
        rm_m1 = rm[ii_l,     jj, k]; m_m1 = m[ii_l,     jj, k]
        rm_0  = rm[ii_r,     jj, k]; m_0  = m[ii_r,     jj, k]
        rm_p1 = rm[ii_r + 1, jj, k]; m_p1 = m[ii_r + 1, jj, k]
        rm_p2 = rm[ii_r + 2, jj, k]; m_p2 = m[ii_r + 2, jj, k]

        F = am[iif, j, k]
        grad_rm, grad_m = _linrood_ppm_face_from_rm_grad_ord5(
            F, m_m1, m_0,
            rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
            m_m3,  m_m2,  m_m1,  m_0,  m_p1,  m_p2,
        )

        bar = lambda_fx_face[iif, j, k]
        @atomic lambda_rm[ii_l - 2, jj, k] += bar * grad_rm[1]
        @atomic lambda_rm[ii_l - 1, jj, k] += bar * grad_rm[2]
        @atomic lambda_rm[ii_l,     jj, k] += bar * grad_rm[3]
        @atomic lambda_rm[ii_r,     jj, k] += bar * grad_rm[4]
        @atomic lambda_rm[ii_r + 1, jj, k] += bar * grad_rm[5]
        @atomic lambda_rm[ii_r + 2, jj, k] += bar * grad_rm[6]
        @atomic lambda_m[ii_l - 2, jj, k]  += bar * grad_m[1]
        @atomic lambda_m[ii_l - 1, jj, k]  += bar * grad_m[2]
        @atomic lambda_m[ii_l,     jj, k]  += bar * grad_m[3]
        @atomic lambda_m[ii_r,     jj, k]  += bar * grad_m[4]
        @atomic lambda_m[ii_r + 1, jj, k]  += bar * grad_m[5]
        @atomic lambda_m[ii_r + 2, jj, k]  += bar * grad_m[6]
    end
end

"""
    apply_ppm_x_face_adjoint!(lambda_rm, lambda_m, lambda_fx_face, rm, m, am,
                                mesh, ::Val{ORD})

Discrete transpose of `_ppm_x_face_kernel!` (LinRood.jl:270) at ORD=5.
Folds `_safe_mixing_ratio` into the d6-AD chain and includes the
donor-mass `α = F / m_donor` contribution. Atomic accumulation on
shared cells.
"""
function apply_ppm_x_face_adjoint!(lambda_rm, lambda_m, lambda_fx_face,
                                   rm, m, am,
                                   mesh::CubedSphereMesh,
                                   ::Val{ORD}=Val(5)) where {ORD}
    ORD == 5 || throw(ArgumentError(
        "Plan-25 Commit 3b implements ORD=5 only; ORD=$ORD is future work"))
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(rm, 3)
    backend = get_backend(lambda_rm)
    k! = _ppm_x_face_kernel_adjoint_ord5!(backend, 256)
    k!(lambda_rm, lambda_m, lambda_fx_face, rm, m, am, Hp, Nc;
       ndrange=(Nc + 1, Nc, Nz))
    synchronize(backend)
    return nothing
end

@kernel function _ppm_y_face_kernel_adjoint_ord5!(
    lambda_rm, lambda_m,
    @Const(lambda_fy_face), @Const(rm), @Const(m), @Const(bm),
    Hp, Nc,
)
    i, jf, k = @index(Global, NTuple)
    _ = Nc
    @inbounds begin
        ii   = Hp + i
        jj_b = Hp + jf - 1
        jj_a = Hp + jf

        rm_m3 = rm[ii, jj_b - 2, k]; m_m3 = m[ii, jj_b - 2, k]
        rm_m2 = rm[ii, jj_b - 1, k]; m_m2 = m[ii, jj_b - 1, k]
        rm_m1 = rm[ii, jj_b,     k]; m_m1 = m[ii, jj_b,     k]
        rm_0  = rm[ii, jj_a,     k]; m_0  = m[ii, jj_a,     k]
        rm_p1 = rm[ii, jj_a + 1, k]; m_p1 = m[ii, jj_a + 1, k]
        rm_p2 = rm[ii, jj_a + 2, k]; m_p2 = m[ii, jj_a + 2, k]

        F = bm[i, jf, k]
        grad_rm, grad_m = _linrood_ppm_face_from_rm_grad_ord5(
            F, m_m1, m_0,
            rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
            m_m3,  m_m2,  m_m1,  m_0,  m_p1,  m_p2,
        )

        bar = lambda_fy_face[i, jf, k]
        @atomic lambda_rm[ii, jj_b - 2, k] += bar * grad_rm[1]
        @atomic lambda_rm[ii, jj_b - 1, k] += bar * grad_rm[2]
        @atomic lambda_rm[ii, jj_b,     k] += bar * grad_rm[3]
        @atomic lambda_rm[ii, jj_a,     k] += bar * grad_rm[4]
        @atomic lambda_rm[ii, jj_a + 1, k] += bar * grad_rm[5]
        @atomic lambda_rm[ii, jj_a + 2, k] += bar * grad_rm[6]
        @atomic lambda_m[ii, jj_b - 2, k]  += bar * grad_m[1]
        @atomic lambda_m[ii, jj_b - 1, k]  += bar * grad_m[2]
        @atomic lambda_m[ii, jj_b,     k]  += bar * grad_m[3]
        @atomic lambda_m[ii, jj_a,     k]  += bar * grad_m[4]
        @atomic lambda_m[ii, jj_a + 1, k]  += bar * grad_m[5]
        @atomic lambda_m[ii, jj_a + 2, k]  += bar * grad_m[6]
    end
end

"""
    apply_ppm_y_face_adjoint!(lambda_rm, lambda_m, lambda_fy_face, rm, m, bm,
                                mesh, ::Val{ORD})

Discrete transpose of `_ppm_y_face_kernel!` (LinRood.jl:241) at ORD=5.
"""
function apply_ppm_y_face_adjoint!(lambda_rm, lambda_m, lambda_fy_face,
                                   rm, m, bm,
                                   mesh::CubedSphereMesh,
                                   ::Val{ORD}=Val(5)) where {ORD}
    ORD == 5 || throw(ArgumentError(
        "Plan-25 Commit 3b implements ORD=5 only; ORD=$ORD is future work"))
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(rm, 3)
    backend = get_backend(lambda_rm)
    k! = _ppm_y_face_kernel_adjoint_ord5!(backend, 256)
    k!(lambda_rm, lambda_m, lambda_fy_face, rm, m, bm, Hp, Nc;
       ndrange=(Nc, Nc + 1, Nz))
    synchronize(backend)
    return nothing
end
