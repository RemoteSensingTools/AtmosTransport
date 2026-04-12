# ---------------------------------------------------------------------------
# Face reconstruction helpers
#
# Common reconstruction primitives used by multiple advection schemes.
# These are face-oriented operations: given cell-centered values, compute
# the face-reconstructed value for upwind flux evaluation.
# ---------------------------------------------------------------------------

"""
    minmod(a, b, c)

Three-argument minmod limiter. Returns zero if arguments have mixed signs.
"""
@inline function minmod(a, b, c)
    if a > 0 && b > 0 && c > 0
        return min(a, b, c)
    elseif a < 0 && b < 0 && c < 0
        return max(a, b, c)
    else
        return zero(a)
    end
end

"""
    van_leer_slope(c_left, c_center, c_right; limited=true)

Compute the van Leer slope at cell center given three cell values.
"""
@inline function van_leer_slope(c_left, c_center, c_right; limited::Bool=true)
    s = (c_right - c_left) / 2
    if limited
        s = minmod(s, 2*(c_right - c_center), 2*(c_center - c_left))
    end
    return s
end

export minmod, van_leer_slope
