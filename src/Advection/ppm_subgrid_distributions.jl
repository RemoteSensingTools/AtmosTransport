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
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Minmod flux limiter (generic 3-argument version)
# ---------------------------------------------------------------------------

"""
    minmod_ppm(a, b, c)

Minmod limiter for PPM: returns the value with smallest magnitude if all have
the same sign, otherwise zero. Used in ORD=4 and ORD=5.

```
minmod(a, b, c) = {
    min(a, b, c)   if all > 0
    max(a, b, c)   if all < 0
    0              otherwise
}
```
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

@inline function huynh_second_constraint(q_l, q_c, q_r, q_LL, q_RR)
    """
    Huynh's second constraint: ensures monotonicity without over-limiting minmod.
    More accurate than minmod for smooth flows.

    Formula (Putman & Lin Appendix A):
    q_6 = 3(q_c - (2q_L + q_R)/3) is clamped to lie in [-|q_R - q_L|, |q_R - q_L|]
    """
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

Returns a tuple (q_L, q_R) of face values for a parabolic subgrid distribution.
Uses a 5-point stencil with minmod slope limiter.
"""
@inline function _ppm_edge_values_ord4(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT) where FT
    # Differences between adjacent cells
    dq_imm = q_im - q_imm      # q_{i-1} - q_{i-2}
    dq_im = q_i - q_im         # q_i - q_{i-1}
    dq_i = q_ip - q_i          # q_{i+1} - q_i
    dq_ip = q_ipp - q_ip       # q_{i+2} - q_{i+1}

    # Compute limited slopes via minmod at cell interfaces
    # These represent the slope of a linear reconstruction within the cell
    s_im = minmod_ppm(dq_im, dq_imm, dq_i)    # slope affecting left edge (i-1/2)
    s_i = minmod_ppm(dq_i, dq_ip, dq_im)      # slope affecting right edge (i+1/2)

    # PPM edge values (parabolic reconstruction at cell boundaries)
    # For a cell of unit width, edge values are:
    q_L = q_i - s_im / 2
    q_R = q_i + s_i / 2

    return (q_L, q_R)
end

# ---------------------------------------------------------------------------
# ORD=5: PPM with Huynh's second constraint
#
# Putman & Lin Sec. 4, "ORD=5 scheme"
# Huynh (1996), "Schemes and constraints for advection"
# Produces smaller L1 and L2 errors than ORD=4 while maintaining quasi-monotonicity
# ---------------------------------------------------------------------------

"""
    _ppm_edge_values_ord5(q_imm, q_im, q_i, q_ip, q_ipp)

Compute PPM edge values (q_L, q_R) using ORD=5 (Huynh's second constraint).

Provides better accuracy than minmod while maintaining quasi-monotonicity.
Reference: Huynh (1996), "Schemes and constraints for advection"
"""
@inline function _ppm_edge_values_ord5(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT) where FT
    # Use Huynh's second constraint for slope limitation instead of minmod
    # This provides better L1/L2 error metrics while maintaining quasi-monotonicity
    s_im = huynh_second_constraint(q_im, q_i, q_i, q_imm, q_ip)
    s_i = huynh_second_constraint(q_i, q_ip, q_ip, q_im, q_ipp)

    # Parabolic edge values
    q_L = q_i - s_im / 2
    q_R = q_i + s_i / 2

    return (q_L, q_R)
end

# ---------------------------------------------------------------------------
# ORD=6: Quasi-5th order (non-monotonic, best pointwise errors)
#
# Putman & Lin Appendix B, "ORD=6 scheme"
# Suresh & Huynh (1997), "Accurate monotonicity-preserving schemes with Runge-Kutta time stepping"
# Produces smaller L∞ errors but allows ~1% negative excursions
# ---------------------------------------------------------------------------

"""
    _ppm_edge_values_ord6(q_imm, q_im, q_i, q_ip, q_ipp)

Compute PPM edge values (q_L, q_R) using ORD=6 (quasi-5th order, non-monotonic).

Provides better pointwise error metrics (L∞) at the cost of allowing
occasional negative values (~1% in test cases).
"""
@inline function _ppm_edge_values_ord6(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT) where FT
    # Quasi-5th order edge values using Suresh & Huynh stencil
    # (Putman & Lin Appendix B, Eq. 41-42)

    # Coefficients for 5th order interpolation
    a1 = FT(1/30)
    a2 = FT(13/60)
    a3 = FT(13/60)
    a4 = FT(9/20)  # 0.45
    a5 = FT(1/20)  # 0.05

    # Right edge value (departure from mean): linear combination of 5 points
    # Putman & Lin Eq. 41
    q_R = a1 * q_imm + a2 * q_im + a3 * q_i + a4 * q_ip + a5 * q_ipp

    # Left edge value: mirror reflection
    # Putman & Lin Eq. 42
    q_L = a5 * q_imm + a4 * q_im + a3 * q_i + a2 * q_ip + a1 * q_ipp

    return (q_L, q_R)
end

# ---------------------------------------------------------------------------
# ORD=7: Special gnomonic face discontinuity treatment
#
# Putman & Lin Appendix C, "Discontinuous treatment of the 12 face edges on the cubed-sphere"
# When flow crosses a CS face boundary, treat the discontinuity by averaging
# two one-sided extrapolations instead of naive PPM across the boundary.
#
# Only used for ORD=7; ORD=4,5,6 use standard edge values everywhere.
# ---------------------------------------------------------------------------

"""
    _ppm_face_edge_value_ord7_discontinuous(
        q_left_0, q_left_1,
        q_right_0, q_right_1,
        orient::Int
    )

Compute edge value at a gnomonic CS face discontinuity using Appendix C of Putman & Lin.

For ORD=7, when computing edge values **at** a CS face boundary, we average
two one-sided second-order extrapolations to handle the coordinate discontinuity.

Returns the shared edge value used by both panels.

# Arguments
- `q_left_0, q_left_1`: interior values (0 and 1 cells from face) on left panel
- `q_right_0, q_right_1`: interior values (0 and 1 cells from face) on right panel
"""
@inline function _ppm_face_edge_value_ord7_discontinuous(
    q_left_0::FT, q_left_1::FT,
    q_right_0::FT, q_right_1::FT
) where FT
    # Putman & Lin Appendix C, Eq. 47
    # Average of two one-sided second-order extrapolations

    # Left panel extrapolation to boundary
    extrap_left = FT(3/2) * q_left_0 - q_left_1 / FT(2)

    # Right panel extrapolation to boundary
    extrap_right = FT(3/2) * q_right_0 - q_right_1 / FT(2)

    # Average the two extrapolations
    q_edge = (extrap_left + extrap_right) / 2

    # Optional: enforce positive-definiteness for tracers
    # q_edge = max(zero(FT), q_edge)  # Eq. 48

    return q_edge
end

# ---------------------------------------------------------------------------
# Dispatcher for all ORD variants
#
# This is called from the kernel to get edge values. The compiler will
# specialize the function for each distinct ORD via Val{ORD}.
# ---------------------------------------------------------------------------

"""
    _ppm_edge_values(q_imm, q_im, q_i, q_ip, q_ipp, ::Val{ORD}) where ORD

Dispatch to the appropriate PPM subgrid distribution for the given ORD variant.

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
    # ORD=7 uses Huynh constraint same as ORD=5, but special handling at CS boundaries
    # (done at kernel level, not here)
    return _ppm_edge_values_ord5(q_imm, q_im, q_i, q_ip, q_ipp)
end
