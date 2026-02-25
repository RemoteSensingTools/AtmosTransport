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
$(TYPEDEF)

Supertype for vertical coordinate systems. Parametric on float type `FT`.

# Interface contract

Any subtype must implement:
- `n_levels(vc)` — number of vertical levels
- `pressure_at_level(vc, k, p_surface)` — pressure at level k given surface pressure
- `level_thickness(vc, k, p_surface)` — thickness (in Pa) of level k
"""
abstract type AbstractVerticalCoordinate{FT} end

"""
$(TYPEDEF)

Hybrid sigma-pressure coordinate: `p(k) = A(k) + B(k) * p_surface`.

Vectors `A` and `B` have length `Nz + 1` (level interfaces / half-levels).
Level centers are at the midpoint of adjacent interfaces.

$(FIELDS)
"""
struct HybridSigmaPressure{FT} <: AbstractVerticalCoordinate{FT}
    "pressure coefficient at each interface [Pa]"
    A :: Vector{FT}
    "sigma coefficient at each interface [dimensionless]"
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

"""
    merge_upper_levels(vc, merge_above_Pa; min_thickness_Pa=500, p_surface=101325)

Merge thin upper-atmosphere levels above `merge_above_Pa` into coarser layers
with a minimum thickness of `min_thickness_Pa`. Levels below the threshold are
kept unchanged.

Returns `(merged_vc, merge_map)` where:
- `merged_vc`: new `HybridSigmaPressure` with fewer levels
- `merge_map`: `Vector{Int}` of length `n_levels(vc)` mapping each native
  level index to its merged level index (for use in met data regridding)
"""
function merge_upper_levels(vc::HybridSigmaPressure{FT},
                            merge_above_Pa::Real;
                            min_thickness_Pa::Real = FT(500),
                            p_surface::Real = FT(101325)) where FT
    Nz = n_levels(vc)
    ps = FT(p_surface)

    # Find the first interface whose pressure exceeds merge_above_Pa
    # (everything above that is a candidate for merging)
    k_threshold = 1
    for k in 1:(Nz + 1)
        if pressure_at_interface(vc, k, ps) > FT(merge_above_Pa)
            k_threshold = k
            break
        end
    end

    # Accumulate thin layers into merged groups above the threshold
    # keep_interfaces: the A/B interface indices we keep in the merged grid
    keep_interfaces = Int[1]  # always keep the top boundary
    accumulated_dp = zero(FT)
    for k in 1:(k_threshold - 1)
        dp = pressure_at_interface(vc, k + 1, ps) - pressure_at_interface(vc, k, ps)
        accumulated_dp += dp
        if accumulated_dp >= FT(min_thickness_Pa)
            push!(keep_interfaces, k + 1)
            accumulated_dp = zero(FT)
        end
    end
    # Close any remaining partial group at the threshold boundary
    if keep_interfaces[end] != k_threshold
        push!(keep_interfaces, k_threshold)
    end
    # Append all interfaces below the threshold (kept as-is)
    for k in (k_threshold + 1):(Nz + 1)
        push!(keep_interfaces, k)
    end

    # Build merged A/B vectors
    A_merged = FT[vc.A[k] for k in keep_interfaces]
    B_merged = FT[vc.B[k] for k in keep_interfaces]
    merged_vc = HybridSigmaPressure(A_merged, B_merged)

    # Build merge_map: native level index → merged level index
    Nz_merged = n_levels(merged_vc)
    merge_map = Vector{Int}(undef, Nz)
    km = 1  # pointer into merged levels
    for k_native in 1:Nz
        # Advance merged pointer when native interface exceeds the merged boundary
        while km < Nz_merged && keep_interfaces[km + 1] <= k_native
            km += 1
        end
        merge_map[k_native] = km
    end

    @info "Merged vertical levels: $(Nz) → $(Nz_merged) " *
          "($(k_threshold - 1) levels above $(merge_above_Pa) Pa → " *
          "$(count(k -> k < k_threshold, keep_interfaces) - 1) merged)"

    return merged_vc, merge_map
end

export n_levels, pressure_at_interface, pressure_at_level, level_thickness
export merge_upper_levels
