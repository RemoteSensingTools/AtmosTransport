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

"""
    merge_thin_levels(vc; min_thickness_Pa=1000, p_surface=101325)

Merge consecutive levels thinner than `min_thickness_Pa` (evaluated at
`p_surface`) into coarser layers.  Works inward from both ends:

1. **Top-down pass**: accumulates thin levels from TOA until each group
   reaches `min_thickness_Pa`.
2. **Bottom-up pass**: same from the surface upward.
3. The two regions are joined at the level where they meet.

Returns `(merged_vc, merge_map)` where:
- `merged_vc`: new `HybridSigmaPressure` with fewer levels
- `merge_map`: `Vector{Int}` mapping each native level to its merged level
"""
function merge_thin_levels(vc::HybridSigmaPressure{FT};
                            min_thickness_Pa::Real = FT(1000),
                            p_surface::Real = FT(101325)) where FT
    Nz = n_levels(vc)
    ps = FT(p_surface)
    min_dp = FT(min_thickness_Pa)

    dp = [level_thickness(vc, k, ps) for k in 1:Nz]

    # Top-down: collect merge boundaries (interface indices) from k=1 downward
    top_ifaces = Int[1]  # always keep TOA
    acc = zero(FT)
    for k in 1:Nz
        acc += dp[k]
        if acc >= min_dp
            push!(top_ifaces, k + 1)
            acc = zero(FT)
        end
    end

    # Bottom-up: collect merge boundaries from k=Nz upward
    bot_ifaces = Int[Nz + 1]  # always keep surface
    acc = zero(FT)
    for k in Nz:-1:1
        acc += dp[k]
        if acc >= min_dp
            pushfirst!(bot_ifaces, k)
            acc = zero(FT)
        end
    end

    # Merge the two sets: use top-down interfaces up to some meeting point,
    # then switch to bottom-up interfaces for the rest.
    # Meeting point: first top-down interface that overlaps with bottom-up.
    # We pick the cleanest join: use top_ifaces as long as they are ≤ first bot_iface.
    first_bot = bot_ifaces[1]
    keep_interfaces = Int[]
    for iface in top_ifaces
        if iface <= first_bot
            push!(keep_interfaces, iface)
        end
    end
    # Close any gap: if last top interface < first bot interface, add first_bot
    if isempty(keep_interfaces) || keep_interfaces[end] < first_bot
        push!(keep_interfaces, first_bot)
    end
    # Add remaining bot interfaces (skip first if already added)
    for iface in bot_ifaces
        if iface > keep_interfaces[end]
            push!(keep_interfaces, iface)
        end
    end

    # Build merged A/B
    A_merged = FT[vc.A[k] for k in keep_interfaces]
    B_merged = FT[vc.B[k] for k in keep_interfaces]
    merged_vc = HybridSigmaPressure(A_merged, B_merged)
    Nz_merged = n_levels(merged_vc)

    # Build merge_map: native level → merged level
    merge_map = Vector{Int}(undef, Nz)
    km = 1
    for k_native in 1:Nz
        while km < Nz_merged && keep_interfaces[km + 1] <= k_native
            km += 1
        end
        merge_map[k_native] = km
    end

    # Summary
    n_top_merged = 0
    n_bot_merged = 0
    for k in 1:Nz
        if dp[k] < min_dp
            p_mid = pressure_at_level(vc, k, ps)
            if p_mid < ps / 2
                n_top_merged += 1
            else
                n_bot_merged += 1
            end
        end
    end

    @info "Merged thin levels: $(Nz) → $(Nz_merged) " *
          "(min_dp=$(min_thickness_Pa) Pa; " *
          "$(n_top_merged) thin upper + $(n_bot_merged) thin lower levels merged)"

    # Print merged grid summary
    for km in 1:Nz_merged
        native_levels = findall(==(km), merge_map)
        dp_m = level_thickness(merged_vc, km, ps)
        p_mid = pressure_at_level(merged_vc, km, ps)
        n = length(native_levels)
        if n > 1
            @info "  merged level $(km): $(n) native levels " *
                  "(k=$(first(native_levels))..$(last(native_levels))), " *
                  "dp=$(round(dp_m; digits=0)) Pa, p=$(round(p_mid; digits=0)) Pa"
        end
    end

    return merged_vc, merge_map
end

export n_levels, pressure_at_interface, pressure_at_level, level_thickness
export merge_upper_levels, merge_thin_levels
