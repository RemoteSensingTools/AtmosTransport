# ---------------------------------------------------------------------------
# PPM Subgrid Distribution Helpers
#
# Inline functions for computing parabolic edge values at cell faces,
# dispatched on ORD via compile-time Val{ORD} parameter.
#
# Each variant reads a 5-point stencil (q_{i-2}, q_{i-1}, q_i, q_{i+1}, q_{i+2})
# and returns left and right face values (q_L, q_R) for the parabolic profile.
#
# Reference: Putman & Lin (2007), "Finite-volume transport on various cubed-sphere grids"
#
# Line-for-line port of the legacy Advection PPM helpers (git commit
# ec2d2c0, path src_legacy/Advection/ppm_subgrid_distributions.jl).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Safe mixing ratio (rm / m with zero guard)
# ---------------------------------------------------------------------------

"""Extract mixing ratio, returning zero if air mass too small."""
@inline function _safe_mixing_ratio(rm::FT, m::FT) where FT
    return m > 100 * eps(FT) ? rm / m : zero(FT)
end

# ---------------------------------------------------------------------------
# Minmod flux limiter (generic 3-argument version)
# ---------------------------------------------------------------------------

"""
    minmod_ppm(a, b, c)

Minmod limiter for PPM: returns the value with smallest magnitude if all have
the same sign, otherwise zero. Used in ORD=4 and ORD=5.
"""
@inline function minmod_ppm(a::FT, b::FT, c::FT) where FT
    if a > zero(FT) && b > zero(FT) && c > zero(FT)
        return min(a, b, c)
    elseif a < zero(FT) && b < zero(FT) && c < zero(FT)
        return max(a, b, c)
    else
        return zero(FT)
    end
end

# ---------------------------------------------------------------------------
# Huynh's second constraint (monotonicity)
#
# Huynh (1996), "Schemes and constraints for advection"
# Eq. 21 from Putman & Lin (2007) Appendix A
# ---------------------------------------------------------------------------

"""
    huynh_second_constraint(q_l, q_c, q_r, q_LL, q_RR)

Huynh's second constraint: ensures monotonicity without over-limiting minmod.
More accurate than minmod for smooth flows.

Formula (Putman & Lin Appendix A):
q_6 = 3(q_c - (2q_L + q_R)/3) is clamped to lie in [-|q_R - q_L|, |q_R - q_L|]
"""
@inline function huynh_second_constraint(q_l, q_c, q_r, q_LL, q_RR)
    FT = typeof(q_l)
    denom = q_r - q_l

    # If edge values are identical, curvature is zero
    if abs(denom) < 10 * eps(FT)
        return zero(FT)
    end

    # Curvature coefficient in the parabolic profile
    q_6 = 3 * (q_c - (2 * q_l + q_r) / 3)

    # Clamp to magnitude of edge difference
    mag = abs(denom)
    return clamp(q_6, -mag, mag)
end

# ---------------------------------------------------------------------------
# ORD=4: Optimized PPM (LR96 PPM + minmod limiter)
#
# Putman & Lin Sec. 4, "ORD=4 scheme"
# Collela & Woodward (1984), "The piecewise parabolic method"
# ---------------------------------------------------------------------------

"""
    _ppm_edge_values_ord4(q_imm, q_im, q_i, q_ip, q_ipp)

Compute PPM edge values (q_L, q_R) using ORD=4 (LR96 PPM + minmod).
Uses a 5-point stencil with minmod slope limiter.
"""
@inline function _ppm_edge_values_ord4(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT) where FT
    dq_imm = q_im - q_imm
    dq_im = q_i - q_im
    dq_i = q_ip - q_i
    dq_ip = q_ipp - q_ip

    s_im = minmod_ppm(dq_im, dq_imm, dq_i)
    s_i = minmod_ppm(dq_i, dq_ip, dq_im)

    q_L = q_i - s_im / 2
    q_R = q_i + s_i / 2

    return (q_L, q_R)
end

# ---------------------------------------------------------------------------
# ORD=5: PPM with Huynh's second constraint
#
# Putman & Lin Sec. 4, "ORD=5 scheme"
# Huynh (1996), "Schemes and constraints for advection"
# ---------------------------------------------------------------------------

"""
    _ppm_edge_values_ord5(q_imm, q_im, q_i, q_ip, q_ipp)

Compute PPM edge values (q_L, q_R) using ORD=5 (Huynh's second constraint).
Provides better accuracy than minmod while maintaining quasi-monotonicity.
"""
@inline function _ppm_edge_values_ord5(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT) where FT
    s_im = huynh_second_constraint(q_im, q_i, q_i, q_imm, q_ip)
    s_i = huynh_second_constraint(q_i, q_ip, q_ip, q_im, q_ipp)

    q_L = q_i - s_im / 2
    q_R = q_i + s_i / 2

    return (q_L, q_R)
end

# ---------------------------------------------------------------------------
# ORD=6: Quasi-5th order (non-monotonic, best pointwise errors)
#
# Putman & Lin Appendix B, "ORD=6 scheme"
# Suresh & Huynh (1997)
# ---------------------------------------------------------------------------

"""
    _ppm_edge_values_ord6(q_imm, q_im, q_i, q_ip, q_ipp)

Compute PPM edge values (q_L, q_R) using ORD=6 (quasi-5th order, non-monotonic).
Best pointwise error (L-inf) at cost of ~1% negative excursions.
"""
@inline function _ppm_edge_values_ord6(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT) where FT
    a1 = FT(1/30)
    a2 = FT(13/60)
    a3 = FT(13/60)
    a4 = FT(9/20)
    a5 = FT(1/20)

    q_R = a1 * q_imm + a2 * q_im + a3 * q_i + a4 * q_ip + a5 * q_ipp
    q_L = a5 * q_imm + a4 * q_im + a3 * q_i + a2 * q_ip + a1 * q_ipp

    return (q_L, q_R)
end

# ---------------------------------------------------------------------------
# ORD=7: Special gnomonic face discontinuity treatment
#
# Putman & Lin Appendix C, Eq. 47
# Averages two one-sided second-order extrapolations at CS face boundaries.
# ---------------------------------------------------------------------------

"""
    _ppm_face_edge_value_ord7_discontinuous(q_left_0, q_left_1, q_right_0, q_right_1)

Compute edge value at a gnomonic CS face discontinuity. Averages two one-sided
second-order extrapolations (Putman & Lin Appendix C, Eq. 47).
"""
@inline function _ppm_face_edge_value_ord7_discontinuous(
    q_left_0::FT, q_left_1::FT,
    q_right_0::FT, q_right_1::FT
) where FT
    extrap_left = FT(3/2) * q_left_0 - q_left_1 / FT(2)
    extrap_right = FT(3/2) * q_right_0 - q_right_1 / FT(2)
    return (extrap_left + extrap_right) / 2
end

# ---------------------------------------------------------------------------
# Dispatcher for all ORD variants
# ---------------------------------------------------------------------------

"""
    _ppm_edge_values(q_imm, q_im, q_i, q_ip, q_ipp, ::Val{ORD})

Dispatch to the appropriate PPM subgrid distribution for the given ORD.
Returns (q_L, q_R) edge values for a parabolic flux calculation.
"""
@inline function _ppm_edge_values(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT, ::Val{4}) where FT
    return _ppm_edge_values_ord4(q_imm, q_im, q_i, q_ip, q_ipp)
end

@inline function _ppm_edge_values(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT, ::Val{5}) where FT
    return _ppm_edge_values_ord5(q_imm, q_im, q_i, q_ip, q_ipp)
end

@inline function _ppm_edge_values(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT, ::Val{6}) where FT
    return _ppm_edge_values_ord6(q_imm, q_im, q_i, q_ip, q_ipp)
end

@inline function _ppm_edge_values(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT, ::Val{7}) where FT
    # ORD=7 uses Huynh constraint same as ORD=5; special boundary handling
    # is done at the kernel level via _apply_ord7_boundary()
    return _ppm_edge_values_ord5(q_imm, q_im, q_i, q_ip, q_ipp)
end
