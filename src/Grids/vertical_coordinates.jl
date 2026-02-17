# ---------------------------------------------------------------------------
# Vertical coordinate types
#
# Vertical coordinates define how model levels map to physical height/pressure.
# The hybrid sigma-pressure system is used by ERA5 (137 levels),
# MERRA-2 (72 levels), and TM5 (25-60 levels).
#
# Pressure at level k: p(k) = A(k) + B(k) * p_surface
# ---------------------------------------------------------------------------

"""
    AbstractVerticalCoordinate{FT}

Supertype for vertical coordinate systems. Parametric on float type `FT`.

# Interface contract

Any subtype must implement:
- `n_levels(vc)` — number of vertical levels
- `pressure_at_level(vc, k, p_surface)` — pressure at level k given surface pressure
- `level_thickness(vc, k, p_surface)` — thickness (in Pa) of level k
"""
abstract type AbstractVerticalCoordinate{FT} end

"""
    HybridSigmaPressure{FT} <: AbstractVerticalCoordinate{FT}

Hybrid sigma-pressure coordinate: `p(k) = A(k) + B(k) * p_surface`.

Vectors `A` and `B` have length `Nz + 1` (level interfaces / half-levels).
Level centers are at the midpoint of adjacent interfaces.

# Fields
- `A :: Vector{FT}` — pressure coefficient at each interface [Pa]
- `B :: Vector{FT}` — sigma coefficient at each interface [dimensionless]
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

"""Number of vertical levels (one less than the number of interfaces)."""
n_levels(vc::HybridSigmaPressure) = length(vc.A) - 1

"""Pressure at interface `k` given surface pressure `p_s`."""
pressure_at_interface(vc::HybridSigmaPressure, k, p_s) = vc.A[k] + vc.B[k] * p_s

"""Pressure at level center `k` (average of bounding interfaces)."""
function pressure_at_level(vc::HybridSigmaPressure, k, p_s)
    p_top = pressure_at_interface(vc, k, p_s)
    p_bot = pressure_at_interface(vc, k + 1, p_s)
    return (p_top + p_bot) / 2
end

"""Thickness of level `k` in Pascals."""
function level_thickness(vc::HybridSigmaPressure, k, p_s)
    return pressure_at_interface(vc, k + 1, p_s) - pressure_at_interface(vc, k, p_s)
end

export n_levels, pressure_at_interface, pressure_at_level, level_thickness
