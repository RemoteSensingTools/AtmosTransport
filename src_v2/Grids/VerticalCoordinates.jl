# ---------------------------------------------------------------------------
# Vertical coordinates — lifted from src/Grids/vertical_coordinates.jl
#
# HybridSigmaPressure is the primary vertical coordinate for ERA5, GEOS-FP,
# GEOS-IT, and TM5.  p(k) = A(k) + B(k) * p_surface
# ---------------------------------------------------------------------------

"""
    HybridSigmaPressure{FT} <: AbstractVerticalCoordinate{FT}

Hybrid sigma-pressure coordinate: `p(k) = A(k) + B(k) * p_surface`.

Vectors `A` and `B` have length `Nz + 1` (level interfaces / half-levels).
"""
struct HybridSigmaPressure{FT} <: AbstractVerticalCoordinate{FT}
    A :: Vector{FT}
    B :: Vector{FT}

    function HybridSigmaPressure(A::Vector{FT}, B::Vector{FT}) where {FT}
        length(A) == length(B) ||
            throw(DimensionMismatch("A and B must have the same length (Nz+1 interfaces)"))
        return new{FT}(A, B)
    end
end

n_levels(vc::HybridSigmaPressure) = length(vc.A) - 1

pressure_at_interface(vc::HybridSigmaPressure, k, p_s) = vc.A[k] + vc.B[k] * p_s

function pressure_at_level(vc::HybridSigmaPressure, k, p_s)
    p_top = pressure_at_interface(vc, k, p_s)
    p_bot = pressure_at_interface(vc, k + 1, p_s)
    return (p_top + p_bot) / 2
end

function level_thickness(vc::HybridSigmaPressure, k, p_s)
    return pressure_at_interface(vc, k + 1, p_s) - pressure_at_interface(vc, k, p_s)
end

"""
    b_sum(vc::HybridSigmaPressure, k)

Difference of B coefficients across level k: B[k+1] - B[k].
Used for distributing surface pressure tendency across levels.
"""
b_diff(vc::HybridSigmaPressure, k) = vc.B[k + 1] - vc.B[k]

export HybridSigmaPressure
export n_levels, pressure_at_interface, pressure_at_level, level_thickness, b_diff
