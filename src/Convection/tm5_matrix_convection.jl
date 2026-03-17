# ---------------------------------------------------------------------------
# TM5 matrix convection scheme — forward
#
# Faithful port of TM5's convection algorithm from:
#   deps/tm5/base/src/tm5_conv.F90      (TM5_Conv_Matrix, TM5_Conv_Apply)
#   deps/tm5/base/src/convection.F90    (driver)
#   deps/tm5/base/src/tmphys_convec.F90 (ConvCloudDim)
#
# Key differences from our TiedtkeConvection:
#   - Uses 4 separate fields (entu, detu, entd, detd) not 1 (CMFMC)
#   - Builds a full Nz×Nz transfer matrix per column
#   - Applies via implicit LU solve (unconditionally stable)
#   - Tracks updraft/downdraft separately with retention fractions
#
# Level convention:
#   TM5 internal: bottom-to-top (k=1=surface, k=lm=TOA)
#   Our data:     top-to-bottom (k=1=TOA, k=Nz=surface)
#   We reverse columns at the interface to use TM5's algorithm verbatim.
#
# The implicit scheme solves:  conv1 * rm_new = rm_old
# where conv1 = I - dt * D  (D = flux divergence operator)
# This is backward Euler: rm_new = rm_old + dt * D * rm_new
# ---------------------------------------------------------------------------

using ..Fields: interior, AbstractField
using ..Grids: grid_size, Δz, floattype, LatitudeLongitudeGrid, CubedSphereGrid
using ..Parameters: PlanetParameters
using LinearAlgebra: lu!, ldiv!

"""
Return the modifiable 3D array for a tracer (Field or raw array).
"""
_tm5_tracer_data(t) = t isa AbstractField ? interior(t) : t

# =====================================================================
# Pure matrix builder — TM5 level convention (k=1=surface, k=lm=TOA)
#
# Faithful port of TM5_Conv_Matrix from tm5_conv.F90 lines 32-186.
# =====================================================================

"""
    tm5_conv_matrix!(conv1, m, entu, detu, entd, detd, lmx, li, ld, dt)

Build the TM5 convection transfer matrix for a single column.

All input arrays use TM5's bottom-to-top convention (k=1=surface).

# Arguments
- `conv1::Matrix{FT}`: output matrix (lmx × lmx), overwritten
- `m::Vector{FT}`: air mass per level [kg/m²]
- `entu, detu`: updraft entrainment/detrainment [kg/m²/s]
- `entd, detd`: downdraft entrainment/detrainment [kg/m²/s]
- `lmx::Int`: number of levels
- `li::Int`: cloud top level (updraft stops here), 0 if no updraft
- `ld::Int`: level of free sinking (downdraft starts here), 0 if no downdraft
- `dt::FT`: timestep [s]

# Returns
- `lmc::Int`: highest active convection level (0 = no convection)
"""
function tm5_conv_matrix!(conv1::AbstractMatrix{FT},
                          m::AbstractVector{FT},
                          entu::AbstractVector{FT}, detu::AbstractVector{FT},
                          entd::AbstractVector{FT}, detd::AbstractVector{FT},
                          lmx::Int, li::Int, ld::Int, dt::FT) where FT

    # Work arrays — stack-allocated for small lmx
    # f(k,kk):  total tracer fraction flux through interface k from source level kk
    # fu(k,kk): updraft component only
    # Index k=0 is the surface boundary (zero flux through ground)
    # Using 1-based indexing with offset: f_arr[k+1, kk] = f(k, kk)
    f  = zeros(FT, lmx + 1, lmx)   # f[1,:] = f(0,:) = surface boundary
    fu = zeros(FT, lmx + 1, lmx)
    amu = zeros(FT, lmx + 1)        # amu[1] = amu(0) = 0
    amd = zeros(FT, lmx + 1)

    # === Updraft: amu is positive upward ===
    # Loop from surface (k=1) to cloud top (k=li)
    @inbounds for k in 1:li
        amu[k+1] = amu[k] + entu[k] - detu[k]
        if amu[k+1] > zero(FT)
            # Retention fraction: fraction of updraft passing through level k
            # Limited from below at 0 for inconsistent en/detrainment rates
            zxi = max(zero(FT), one(FT) - detu[k] / (amu[k] + entu[k]))
        else
            amu[k+1] = zero(FT)
            zxi = zero(FT)
        end
        # Propagate tracer fractions from previous levels through this interface
        for kk in 1:k-1
            fu[k+1, kk] = fu[k, kk] * zxi
        end
        # Entrainment from level k itself
        # fu(k-1,k) = 0 by construction (see Heimann manual)
        fu[k+1, k] = entu[k] / m[k] * zxi
    end

    # === Downdraft: amd is negative downward ===
    # Loop from level of free sinking (k=ld) down to k=2
    @inbounds for k in ld:-1:2
        amd[k] = amd[k+1] - entd[k] + detd[k]   # amd[k] = amd(k-1) in Fortran
        if amd[k] < zero(FT)
            zxi = max(zero(FT), one(FT) + detd[k] / (amd[k+1] - entd[k]))
        else
            amd[k] = zero(FT)
            zxi = zero(FT)
        end
        # Propagate downdraft tracer fractions
        for kk in k+1:ld
            f[k, kk] = f[k+1, kk] * zxi   # f[k] = f(k-1) in Fortran
        end
        # Entrainment from level k into downdraft (note negative sign)
        f[k, k] = -entd[k] / m[k] * zxi
    end

    # === Combine updraft + downdraft + subsidence ===
    @inbounds for k in 1:lmx-1
        for kk in 1:lmx
            f[k+1, kk] = fu[k+1, kk] + f[k+1, kk]   # merge updraft and downdraft
        end
        # Compensating subsidence from updraft: pulls air from level k+1
        f[k+1, k+1] = f[k+1, k+1] - amu[k+1] / m[k+1]
        # Compensating ascent from downdraft: pushes air into level k
        # (Sander Houweling fix: separate updraft/downdraft subsidence)
        f[k+1, k] = f[k+1, k] - amd[k+1] / m[k]
    end

    # === Assemble forward matrix ===
    # conv1 = I - dt * D  where D(k,kk) = f(k-1,kk) - f(k,kk)
    # Solved as: conv1 * rm_new = rm_old  (implicit backward Euler)
    lmc = 0
    fill!(conv1, zero(FT))
    @inbounds for k in 1:lmx
        for kk in 1:lmx
            # f[k, kk] = f(k-1, kk), f[k+1, kk] = f(k, kk) in Fortran convention
            fk_below = f[k, kk]     # flux through interface below level k
            fk_above = f[k+1, kk]   # flux through interface above level k
            if fk_below != fk_above
                conv1[k, kk] = -dt * (fk_below - fk_above)
                lmc = max(lmc, max(k, kk))
            end
        end
        conv1[k, k] += one(FT)
    end

    return lmc
end

# =====================================================================
# Cloud dimension diagnostics — port of ConvCloudDim
# =====================================================================

"""
    _conv_cloud_dim(detu, entd, lmx)

Compute cloud top (li) and level of free sinking (ld) from detrainment/entrainment.
Uses TM5 bottom-to-top convention (k=1=surface).

Returns `(li, ld)`.
"""
function _conv_cloud_dim(detu::AbstractVector{FT}, entd::AbstractVector{FT},
                         lmx::Int) where FT
    # cloud_top (li): highest level with detu > 0 (scan from top down)
    li = 0
    @inbounds for k in lmx:-1:1
        if detu[k] > zero(FT)
            li = k
            break
        end
    end

    # cloud_lfs (ld): highest level with entd > 0 (scan from top down)
    ld = 0
    @inbounds for k in lmx:-1:1
        if entd[k] > zero(FT)
            ld = k
            break
        end
    end

    return li, ld
end

# =====================================================================
# Dispatch: LatitudeLongitudeGrid (CPU path)
# =====================================================================

"""
    convect!(tracers, tm5conv_data, delp, conv::TM5MatrixConvection,
             grid::LatitudeLongitudeGrid, dt, planet; kwargs...)

Apply TM5 matrix convection to lat-lon tracers.

`tm5conv_data` is a NamedTuple with fields `entu`, `detu`, `entd`, `detd`,
each of size `(Nx, Ny, Nz)` in our top-to-bottom convention.
`delp` is `(Nx, Ny, Nz)` pressure thickness per layer [Pa].

For each (i,j) column:
1. Reverse levels to TM5 bottom-to-top convention
2. Compute cloud dimensions (li, ld)
3. Build the Nz×Nz transfer matrix
4. Solve `conv1 * rm_new = rm_old` via LU decomposition
5. Reverse back and update tracers
"""
function convect!(tracers::NamedTuple, tm5conv_data::NamedTuple, delp,
                   conv::TM5MatrixConvection, grid::LatitudeLongitudeGrid, dt,
                   planet::PlanetParameters;
                   dtrain_panels=nothing, workspace=nothing)
    FT = floattype(grid)
    Nx, Ny, Nz = size(delp)
    grav = FT(planet.gravity)
    dt_FT = FT(dt)

    lmax = conv.lmax_conv > 0 ? min(conv.lmax_conv, Nz) : Nz

    entu_data = tm5conv_data.entu
    detu_data = tm5conv_data.detu
    entd_data = tm5conv_data.entd
    detd_data = tm5conv_data.detd

    ntr = length(tracers)

    # Process each column independently (parallelizable)
    Threads.@threads for idx in 1:Nx*Ny
        j = div(idx - 1, Nx) + 1
        i = mod(idx - 1, Nx) + 1

        # Column work arrays (TM5 bottom-to-top convention)
        m_col     = Vector{FT}(undef, lmax)
        entu_col  = Vector{FT}(undef, lmax)
        detu_col  = Vector{FT}(undef, lmax)
        entd_col  = Vector{FT}(undef, lmax)
        detd_col  = Vector{FT}(undef, lmax)
        conv1     = Matrix{FT}(undef, lmax, lmax)

        # Extract and reverse: our k=1=TOA → TM5 k=1=surface
        @inbounds for k in 1:lmax
            k_rev = Nz + 1 - k   # map TM5 level k to our level index
            m_col[k]    = delp[i, j, k_rev] / grav   # air mass [kg/m²]
            entu_col[k] = entu_data[i, j, k_rev]
            detu_col[k] = detu_data[i, j, k_rev]
            entd_col[k] = entd_data[i, j, k_rev]
            detd_col[k] = detd_data[i, j, k_rev]
        end

        # Compute cloud dimensions
        li, ld = _conv_cloud_dim(detu_col, entd_col, lmax)

        # Skip if no convection in this column
        (li == 0 && ld == 0) && continue

        # Build the transfer matrix
        lmc = tm5_conv_matrix!(conv1, m_col, entu_col, detu_col,
                               entd_col, detd_col, lmax, li, ld, dt_FT)

        # Skip if matrix has no active levels
        lmc == 0 && continue

        # LU factorize the active submatrix
        conv1_sub = @view conv1[1:lmc, 1:lmc]
        F = lu!(conv1_sub)

        # Apply to each tracer
        for tracer in values(tracers)
            arr = _tm5_tracer_data(tracer)

            # Extract column, convert mixing ratio → tracer mass, reverse
            rm_col = Vector{FT}(undef, lmc)
            @inbounds for k in 1:lmc
                k_rev = Nz + 1 - k
                rm_col[k] = arr[i, j, k_rev] * m_col[k]
            end

            # Solve: conv1 * rm_new = rm_old (overwrites rm_col with rm_new)
            ldiv!(F, rm_col)

            # Convert back: tracer mass → mixing ratio, reverse to our convention
            @inbounds for k in 1:lmc
                k_rev = Nz + 1 - k
                arr[i, j, k_rev] = rm_col[k] / m_col[k]
            end
        end
    end

    return nothing
end
