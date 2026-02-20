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
export reduce_row_mass!, reduce_am_row!, expand_row_mass!

"""
$(TYPEDEF)

Per-latitude specification of zonal cell clustering for CFL stability.

$(FIELDS)
"""
struct ReducedGridSpec
    "number of uniform-grid zonal cells"
    Nx             :: Int
    "length-Ny vector; how many fine cells form one reduced cell"
    cluster_sizes  :: Vector{Int}
    "length-Ny vector; effective number of zonal cells (Nx / cluster_size)"
    reduced_counts :: Vector{Int}
end

"""
$(TYPEDSIGNATURES)

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
$(SIGNATURES)

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
$(SIGNATURES)

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
$(SIGNATURES)

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

# ---------------------------------------------------------------------------
# Mass-quantity reduce / expand for mass-flux advection
#
# Unlike concentration (intensive → average), mass is extensive → SUM.
# Mass flux am is also extensive → pick at reduced-cell boundaries.
# ---------------------------------------------------------------------------

"""
$(SIGNATURES)

Sum `r` adjacent fine cells of an extensive field (rm or m) into one
reduced cell for latitude `j`, level `k`.  `q_red` must have length `Nx ÷ r`.
"""
function reduce_row_mass!(q_red::AbstractVector, q::AbstractArray,
                          j::Int, k::Int, r::Int, Nx::Int)
    Nx_red = Nx ÷ r
    @inbounds for i_red in 1:Nx_red
        s = zero(eltype(q))
        i_start = (i_red - 1) * r + 1
        for off in 0:r-1
            s += q[i_start + off, j, k]
        end
        q_red[i_red] = s
    end
    return nothing
end

"""
$(SIGNATURES)

Pick mass flux `am` at reduced-grid cell boundaries.
`am` has shape (Nx+1, Ny, Nz); `am_red` has length `Nx_red + 1`.
The face between reduced cells `i_red-1` and `i_red` corresponds to fine
face `(i_red - 1) * r + 1`.
"""
function reduce_am_row!(am_red::AbstractVector, am::AbstractArray,
                        j::Int, k::Int, r::Int, Nx::Int)
    Nx_red = Nx ÷ r
    @inbounds for i_red in 1:Nx_red
        am_red[i_red] = am[(i_red - 1) * r + 1, j, k]
    end
    am_red[Nx_red + 1] = am_red[1]  # periodic
    return nothing
end

"""
$(SIGNATURES)

Distribute mass changes from a reduced-grid advection step back to fine cells.
Changes in `rm` and `m` are distributed proportionally to each fine cell's
original mass fraction within the reduced cell, guaranteeing exact conservation
of both tracer mass and air mass.

  rm_fine[i] += (rm_red_new[i_r] - rm_red_old[i_r]) × (rm_old[i] / rm_red_old[i_r])
  m_fine[i]  += (m_red_new[i_r]  - m_red_old[i_r])  × (m_old[i]  / m_red_old[i_r])
"""
function expand_row_mass!(rm::AbstractArray, m::AbstractArray,
                          rm_red_new::AbstractVector, rm_red_old::AbstractVector,
                          m_red_new::AbstractVector, m_red_old::AbstractVector,
                          j::Int, k::Int, r::Int, Nx::Int)
    FT = eltype(rm)
    Nx_red = Nx ÷ r
    @inbounds for i_red in 1:Nx_red
        i_start = (i_red - 1) * r + 1
        delta_rm = rm_red_new[i_red] - rm_red_old[i_red]
        delta_m  = m_red_new[i_red]  - m_red_old[i_red]
        rm_sum = rm_red_old[i_red]
        m_sum  = m_red_old[i_red]
        for off in 0:r-1
            i = i_start + off
            if abs(rm_sum) > eps(FT)
                rm[i, j, k] += delta_rm * (rm[i, j, k] / rm_sum)
            else
                rm[i, j, k] += delta_rm / FT(r)
            end
            if abs(m_sum) > eps(FT)
                m[i, j, k] += delta_m * (m[i, j, k] / m_sum)
            else
                m[i, j, k] += delta_m / FT(r)
            end
        end
    end
    return nothing
end
