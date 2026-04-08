# ---------------------------------------------------------------------------
# Mass-flux CFL utilities
#
# CFL number for mass-flux advection: |flux| / donor_mass.
# These are used by the Strang splitting orchestrator to decide subcycling.
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, get_backend, synchronize

"""
    max_cfl_x(am, m, cluster_sizes) -> FT

Maximum CFL in the x-direction: max |am[i,j,k]| / m[donor,j,k].
Respects reduced grid clustering via `cluster_sizes`.
"""
function max_cfl_x(am::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                   cluster_sizes::AbstractVector{<:Integer}) where FT
    Nx, Ny, Nz = size(m)
    max_cfl = zero(FT)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx+1
        r = Int(cluster_sizes[j])
        if r == 1
            donor = am[i, j, k] >= zero(FT) ? max(i - 1, 1) : min(i, Nx)
            donor = donor == 0 ? Nx : donor
            cfl_val = abs(am[i, j, k]) / max(m[donor, j, k], eps(FT))
        else
            Nx_red = Nx ÷ r
            ic = (min(i, Nx) - 1) ÷ r + 1
            m_cluster = zero(FT)
            i_start = (ic - 1) * r + 1
            for off in 0:r-1
                m_cluster += m[i_start + off, j, k]
            end
            cfl_val = abs(am[i, j, k]) / max(m_cluster, eps(FT))
        end
        max_cfl = max(max_cfl, cfl_val)
    end
    return max_cfl
end

"""
    max_cfl_y(bm, m) -> FT

Maximum CFL in the y-direction: max |bm[i,j,k]| / m[i,donor,k].
"""
function max_cfl_y(bm::AbstractArray{FT,3}, m::AbstractArray{FT,3}) where FT
    Nx, Ny, Nz = size(m)
    max_cfl = zero(FT)
    @inbounds for k in 1:Nz, j in 1:Ny+1, i in 1:Nx
        donor = bm[i, j, k] >= zero(FT) ? max(j - 1, 1) : min(j, Ny)
        cfl_val = abs(bm[i, j, k]) / max(m[i, donor, k], eps(FT))
        max_cfl = max(max_cfl, cfl_val)
    end
    return max_cfl
end

"""
    max_cfl_z(cm, m) -> FT

Maximum CFL in the z-direction: max |cm[i,j,k]| / m[i,j,donor].
"""
function max_cfl_z(cm::AbstractArray{FT,3}, m::AbstractArray{FT,3}) where FT
    Nx, Ny, Nz = size(m)
    max_cfl = zero(FT)
    @inbounds for k in 1:Nz+1, j in 1:Ny, i in 1:Nx
        donor = cm[i, j, k] >= zero(FT) ? max(k - 1, 1) : min(k, Nz)
        cfl_val = abs(cm[i, j, k]) / max(m[i, j, donor], eps(FT))
        max_cfl = max(max_cfl, cfl_val)
    end
    return max_cfl
end

export max_cfl_x, max_cfl_y, max_cfl_z
