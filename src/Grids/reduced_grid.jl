# ---------------------------------------------------------------------------
# Reduced Grid for Polar CFL Stability
#
# At high latitudes on a regular lat-lon grid, zonal grid cells become very
# narrow (Δx = R·cos(φ)·Δλ → 0 as φ → ±90°), causing CFL violations for
# any practical time step.
#
# TM5 handles this by clustering adjacent zonal cells at high latitudes
# before x-advection, advecting on the coarser reduced row, then
# redistributing back.  This keeps the effective zonal spacing roughly
# constant (~Δx at the equator) everywhere.
#
# Reference: TM5 deps/tm5/base/src/advectm_cfl.F90 (uni2red, red2uni)
# ---------------------------------------------------------------------------

export ReducedGridSpec, compute_reduced_grid
export reduce_row!, expand_row!, reduce_velocity_row!

"""
    ReducedGridSpec

Per-latitude specification of zonal cell clustering for CFL stability.

# Fields
- `Nx`             — number of uniform-grid zonal cells
- `cluster_sizes`  — length-Ny vector; how many fine cells form one reduced cell
- `reduced_counts` — length-Ny vector; effective number of zonal cells (Nx / cluster_size)
"""
struct ReducedGridSpec
    Nx             :: Int
    cluster_sizes  :: Vector{Int}
    reduced_counts :: Vector{Int}
end

"""
    compute_reduced_grid(Nx, φᶜ; max_cluster=nothing)

Auto-compute per-latitude cluster sizes so that the effective zonal spacing
Δx_eff ≈ R·Δλ (equatorial value) at every latitude.

Cluster sizes are constrained to be divisors of `Nx` so that cells divide
evenly.  `max_cluster` caps the maximum cluster size (defaults to Nx).

Returns `nothing` if no reduction is needed (all cluster sizes are 1).
"""
function compute_reduced_grid(Nx::Int, φᶜ::AbstractVector;
                              max_cluster::Union{Nothing,Int} = nothing)
    Ny = length(φᶜ)
    max_r = something(max_cluster, Nx)

    divs = _divisors(Nx)

    cluster_sizes  = ones(Int, Ny)
    reduced_counts = fill(Nx, Ny)
    any_reduced = false

    for j in 1:Ny
        cos_phi = abs(cosd(Float64(φᶜ[j])))
        cos_phi = max(cos_phi, 1e-10)
        ideal = 1.0 / cos_phi
        # Only reduce when the CFL penalty is significant (>50% worse than equator)
        if ideal < 1.5
            continue
        end
        r = _nearest_divisor_geq(divs, ceil(Int, ideal))
        r = min(r, max_r)
        cluster_sizes[j]  = r
        reduced_counts[j] = Nx ÷ r
        if r > 1
            any_reduced = true
        end
    end

    any_reduced || return nothing
    return ReducedGridSpec(Nx, cluster_sizes, reduced_counts)
end

"""Sorted divisors of n."""
function _divisors(n::Int)
    d = Int[]
    for i in 1:isqrt(n)
        if n % i == 0
            push!(d, i)
            if i != n ÷ i
                push!(d, n ÷ i)
            end
        end
    end
    return sort!(d)
end

"""Smallest divisor in `divs` that is >= `target`."""
function _nearest_divisor_geq(divs::Vector{Int}, target::Int)
    idx = searchsortedfirst(divs, target)
    return idx <= length(divs) ? divs[idx] : divs[end]
end

# ---------------------------------------------------------------------------
# Row-level reduction / expansion helpers
# ---------------------------------------------------------------------------

"""
    reduce_row!(c_red, c, j, k, r, Nx)

Average `r` adjacent fine cells into one reduced cell for latitude `j`, level `k`.
`c_red` must have length `Nx ÷ r`.
"""
function reduce_row!(c_red::AbstractVector, c::AbstractArray, j::Int, k::Int,
                     r::Int, Nx::Int)
    Nx_red = Nx ÷ r
    @inbounds for i_red in 1:Nx_red
        s = zero(eltype(c))
        i_start = (i_red - 1) * r + 1
        for m in 0:r-1
            s += c[i_start + m, j, k]
        end
        c_red[i_red] = s / r
    end
    return nothing
end

"""
    expand_row!(c, c_red_new, c_red_old, j, k, r, Nx)

Distribute the change (c_red_new - c_red_old) uniformly back to the fine cells.
Preserves sub-grid structure while conserving the volume-averaged concentration.
"""
function expand_row!(c::AbstractArray, c_red_new::AbstractVector,
                     c_red_old::AbstractVector, j::Int, k::Int,
                     r::Int, Nx::Int)
    Nx_red = Nx ÷ r
    @inbounds for i_red in 1:Nx_red
        delta = c_red_new[i_red] - c_red_old[i_red]
        i_start = (i_red - 1) * r + 1
        for m in 0:r-1
            c[i_start + m, j, k] += delta
        end
    end
    return nothing
end

"""
    reduce_velocity_row!(u_red, u, j, k, r, Nx)

Pick face velocities at reduced-grid cell boundaries.
`u` has shape (Nx+1, Ny, Nz); `u_red` has length `Nx_red + 1`.
Face i_red corresponds to fine face `(i_red - 1) * r + 1`.
"""
function reduce_velocity_row!(u_red::AbstractVector, u::AbstractArray,
                              j::Int, k::Int, r::Int, Nx::Int)
    Nx_red = Nx ÷ r
    @inbounds for i_red in 1:Nx_red
        u_red[i_red] = u[(i_red - 1) * r + 1, j, k]
    end
    u_red[Nx_red + 1] = u_red[1]  # periodic
    return nothing
end
