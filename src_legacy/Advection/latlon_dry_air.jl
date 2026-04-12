# ---------------------------------------------------------------------------
# Lat-Lon Dry-Air Correction Kernels
#
# Converts wet (total moist) DELP, mass fluxes, and convective fluxes to dry
# basis using specific humidity QV for lat-lon grids.  Mirror of the CS panel
# kernels in cubed_sphere_mass_flux.jl but without halo offsets.
#
#   m_ref_dry = m_ref_wet × (1 − qv)
#   am_dry    = am_wet    × (1 − qv_face_x)    periodic wrapping in x
#   bm_dry    = bm_wet    × (1 − qv_face_y)    pole boundary in y
#   cmfmc_dry = cmfmc_wet × (1 − qv_interface)  vertical interface avg
#   dtrain_dry= dtrain_wet× (1 − qv)            layer center
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend

# ── m_ref (DELP-derived air mass) ──────────────────────────────────────

@kernel function _dry_mref_ll_kernel!(m_ref, @Const(qv))
    i, j, k = @index(Global, NTuple)
    @inbounds m_ref[i, j, k] *= (1 - qv[i, j, k])
end

"""Apply dry-air correction to lat-lon air mass: `m_ref *= (1 - qv)`."""
function apply_dry_mref_ll!(m_ref::AbstractArray{FT,3},
                             qv::AbstractArray{FT,3},
                             Nx::Int, Ny::Int, Nz::Int) where FT
    backend = get_backend(m_ref)
    k! = _dry_mref_ll_kernel!(backend, 256)
    k!(m_ref, qv; ndrange=(Nx, Ny, Nz))
    return nothing
end

# ── am (x-direction mass flux, periodic) ───────────────────────────────

@kernel function _dry_am_ll_kernel!(am, @Const(qv), Nx)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        # Face between cell (i-1) and cell (i), periodic wrapping
        i_left = ifelse(i == 1, Nx, i - 1)
        i_right = ifelse(i == Nx + 1, 1, i)
        qv_face = eltype(am)(0.5) * (qv[i_left, j, k] + qv[i_right, j, k])
        am[i, j, k] *= (1 - qv_face)
    end
end

"""Apply dry-air correction to x-direction mass flux with periodic wrapping.
`am` is `(Nx+1, Ny, Nz)`, `qv` is `(Nx, Ny, Nz)`."""
function apply_dry_am_ll!(am::AbstractArray{FT,3},
                           qv::AbstractArray{FT,3},
                           Nx::Int, Ny::Int, Nz::Int) where FT
    backend = get_backend(am)
    k! = _dry_am_ll_kernel!(backend, 256)
    k!(am, qv, Nx; ndrange=(Nx + 1, Ny, Nz))
    return nothing
end

# ── bm (y-direction mass flux, pole boundaries) ───────────────────────

@kernel function _dry_bm_ll_kernel!(bm, @Const(qv), Ny)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        # Face between cell (j-1) and cell (j); at poles (j=1 or Ny+1),
        # flux is zero so correction is irrelevant, but use nearest cell QV
        j_south = max(j - 1, 1)
        j_north = min(j, Ny)
        qv_face = eltype(bm)(0.5) * (qv[i, j_south, k] + qv[i, j_north, k])
        bm[i, j, k] *= (1 - qv_face)
    end
end

"""Apply dry-air correction to y-direction mass flux with pole boundaries.
`bm` is `(Nx, Ny+1, Nz)`, `qv` is `(Nx, Ny, Nz)`."""
function apply_dry_bm_ll!(bm::AbstractArray{FT,3},
                           qv::AbstractArray{FT,3},
                           Nx::Int, Ny::Int, Nz::Int) where FT
    backend = get_backend(bm)
    k! = _dry_bm_ll_kernel!(backend, 256)
    k!(bm, qv, Ny; ndrange=(Nx, Ny + 1, Nz))
    return nothing
end

# ── CMFMC (convective mass flux at interfaces) ────────────────────────

@kernel function _dry_cmfmc_ll_kernel!(cmfmc, @Const(qv), Nz)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        qv_iface = if k == 1
            qv[i, j, 1]
        elseif k == Nz + 1
            qv[i, j, Nz]
        else
            eltype(cmfmc)(0.5) * (qv[i, j, k - 1] + qv[i, j, k])
        end
        cmfmc[i, j, k] *= (1 - qv_iface)
    end
end

"""Apply dry-air correction to convective mass flux using interface-averaged QV.
`cmfmc` is `(Nx, Ny, Nz+1)`, `qv` is `(Nx, Ny, Nz)`."""
function apply_dry_cmfmc_ll!(cmfmc::AbstractArray{FT,3},
                              qv::AbstractArray{FT,3},
                              Nx::Int, Ny::Int, Nz::Int) where FT
    backend = get_backend(cmfmc)
    k! = _dry_cmfmc_ll_kernel!(backend, 256)
    k!(cmfmc, qv, Nz; ndrange=(Nx, Ny, Nz + 1))
    return nothing
end

# ── DTRAIN (detraining mass flux at layer centers) ─────────────────────

"""Apply dry-air correction to detraining mass flux: `dtrain *= (1 - qv)`.
`dtrain` and `qv` are both `(Nx, Ny, Nz)`.  Reuses the m_ref kernel."""
function apply_dry_dtrain_ll!(dtrain::AbstractArray{FT,3},
                               qv::AbstractArray{FT,3},
                               Nx::Int, Ny::Int, Nz::Int) where FT
    apply_dry_mref_ll!(dtrain, qv, Nx, Ny, Nz)   # same pointwise op
end

# ── Recompute cm from dry am/bm ───────────────────────────────────────

"""Recompute vertical mass flux `cm` from (dry-corrected) `am` and `bm`
via the continuity-equation column sweep.  Uses `_cm_column_kernel!`
(defined in mass_flux_advection.jl, same Advection module scope)."""
function recompute_cm_ll!(cm::AbstractArray{FT,3},
                           am::AbstractArray{FT,3},
                           bm::AbstractArray{FT,3},
                           bt::AbstractVector{FT},
                           Nx::Int, Ny::Int, Nz::Int) where FT
    fill!(cm, zero(FT))
    backend = get_backend(cm)
    k! = _cm_column_kernel!(backend, 256)
    k!(cm, am, bm, bt, Nz; ndrange=(Nx, Ny))
    synchronize(backend)
    return nothing
end
