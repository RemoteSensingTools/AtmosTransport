# ---------------------------------------------------------------------------
# TM5-faithful mass-flux advection (Russell & Lerner, 1981)
#
# This module implements TM5's native formulation where tracer mass (rm) and
# air mass (m) are co-advected.  The key equation is:
#
#   alpha   = am / m_donor           (mass-based Courant number)
#   f       = alpha * (rm + (1 - alpha) * rxm)   (tracer mass flux, positive flow)
#   rm_new  = rm + f_left - f_right              (tracer mass update)
#   m_new   = m  + am_left - am_right            (air mass update)
#   c       = rm_new / m_new                     (recover concentration)
#
# Advantages over concentration-based advection:
# - Division by m_new naturally handles mass divergence from operator splitting
# - Exact mass conservation (sum of rm is invariant)
# - No need for post-hoc mass correction
#
# Reference: TM5 advectx.F90, advecty.F90, advectz.F90 (dynamw_1d)
# ---------------------------------------------------------------------------

# =====================================================================
# Helpers: air mass and mass flux computation
# =====================================================================

"""
$(SIGNATURES)

Compute 3D air mass array from pressure thickness and grid geometry.

Returns `m[i,j,k] = Δp[i,j,k] * cell_area(i,j,grid) / g` in kg.
"""
function compute_air_mass(Δp::AbstractArray{FT,3}, grid) where FT
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    g = FT(grid.gravity)
    m = Array{FT}(undef, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny
        area_j = FT(cell_area(1, j, grid))
        for i in 1:Nx
            m[i, j, k] = Δp[i, j, k] * area_j / g
        end
    end
    return m
end

"""
$(SIGNATURES)

Compute mass fluxes `am`, `bm`, `cm` from staggered velocities, pressure
thickness, and half-timestep.  Follows TM5's `dynam0` approach.

`u` is size `(Nx+1, Ny, Nz)`, `v` is `(Nx, Ny+1, Nz)`.

Returns a `NamedTuple` `(; am, bm, cm)`:
- `am[i,j,k]`: eastward mass flux at x-face `i` (kg per half-timestep).
  Size `(Nx+1, Ny, Nz)`.
- `bm[i,j,k]`: northward mass flux at y-face `j` (kg per half-timestep).
  Size `(Nx, Ny+1, Nz)`.
- `cm[i,j,k]`: downward mass flux at z-interface `k` (kg per half-timestep).
  Size `(Nx, Ny, Nz+1)`.  Derived from horizontal convergence via the
  continuity equation to ensure column mass conservation.
"""
function compute_mass_fluxes(u, v, grid, Δp::AbstractArray{FT,3}, half_dt) where FT
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    g = FT(grid.gravity)
    vc = grid.vertical

    # --- Horizontal mass fluxes ---
    am = Array{FT}(undef, Nx + 1, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny
        dy_j = FT(Δy(1, j, grid))
        for i in 1:Nx+1
            i_left  = i == 1 ? Nx : i - 1
            i_right = i > Nx ? 1  : i
            Δp_face = (Δp[i_left, j, k] + Δp[i_right, j, k]) / 2
            am[i, j, k] = FT(half_dt) * u[i, j, k] * Δp_face * dy_j / g
        end
    end

    bm = Array{FT}(undef, Nx, Ny + 1, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny+1, i in 1:Nx
        j_below = max(j - 1, 1)
        j_above = min(j, Ny)
        Δp_face = (Δp[i, j_below, k] + Δp[i, j_above, k]) / 2
        φ_face = if j == 1
            FT(-90.0)
        elseif j == Ny + 1
            FT(90.0)
        else
            (grid.φᶠ isa Array ? grid.φᶠ[j] : Array(grid.φᶠ)[j])
        end
        dx_face = FT(grid.radius) * cosd(φ_face) * deg2rad(FT(grid.Δλ))
        bm[i, j, k] = FT(half_dt) * v[i, j, k] * Δp_face * abs(dx_face) / g
    end

    # --- Vertical mass flux from continuity (TM5 dynam0) ---
    # Horizontal convergence into each cell
    conv = Array{FT}(undef, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        conv[i, j, k] = am[i, j, k] - am[i + 1, j, k] +
                         bm[i, j, k] - bm[i, j + 1, k]
    end

    # Column-integrated convergence
    pit = zeros(FT, Nx, Ny)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        pit[i, j] += conv[i, j, k]
    end

    # Normalized B coefficient differences (fraction of sp-tendency per layer)
    ΔB = Vector{FT}(undef, Nz)
    @inbounds for k in 1:Nz
        ΔB[k] = vc.B[k + 1] - vc.B[k]
    end
    ΔB_total = vc.B[Nz + 1] - vc.B[1]
    bt = abs(ΔB_total) > eps(FT) ? ΔB ./ ΔB_total : zeros(FT, Nz)

    cm = zeros(FT, Nx, Ny, Nz + 1)
    # cm[:,,:,1] = 0 (top of atmosphere, no flux)
    # cm[:,:,Nz+1] = 0 (surface, no flux)
    # Accumulate from top downward:
    #   cm[k+1] = cm[k] + conv[k] - bt[k] * pit
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        cm[i, j, k + 1] = cm[i, j, k] + conv[i, j, k] - bt[k] * pit[i, j]
    end

    return (; am, bm, cm)
end

# =====================================================================
# X-direction mass-flux advection
# =====================================================================

"""
$(SIGNATURES)

TM5-faithful slopes advection in the x-direction (longitude) using mass fluxes.

Modifies `rm_tracers` (NamedTuple of tracer mass arrays) and `m` (air mass) in
place.  Periodic boundary conditions in longitude.

`am` is the eastward mass flux at x-faces, size `(Nx+1, Ny, Nz)`.

The scheme:
1. Computes slopes of `rm` (centered difference, with optional limiter).
2. Computes tracer mass flux `f` using `alpha = am / m_donor`.
3. Updates `rm = rm + f_in - f_out`.
4. Updates `m = m + am_in - am_out`.
"""
function advect_x_massflux!(rm_tracers::NamedTuple,
                             m::AbstractArray{FT,3},
                             am::AbstractArray{FT,3},
                             grid::LatitudeLongitudeGrid,
                             use_limiter::Bool) where FT
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    f  = Vector{FT}(undef, Nx + 1)
    sx = Vector{FT}(undef, Nx)

    @inbounds for k in 1:Nz, j in 1:Ny

        for (_, rm) in pairs(rm_tracers)

            # Slopes of concentration (c = rm/m), then scale by m.
            # Computing slopes from c rather than rm ensures that a spatially
            # uniform concentration field is exactly preserved.
            for i in 1:Nx
                i_prev = i == 1 ? Nx : i - 1
                i_next = i == Nx ? 1 : i + 1
                c_prev = rm[i_prev, j, k] / m[i_prev, j, k]
                c_this = rm[i, j, k] / m[i, j, k]
                c_next = rm[i_next, j, k] / m[i_next, j, k]
                sc = (c_next - c_prev) / 2
                if use_limiter
                    sc = minmod(sc, 2 * (c_next - c_this), 2 * (c_this - c_prev))
                end
                sx[i] = m[i, j, k] * sc
                if use_limiter
                    sx[i] = max(min(sx[i], rm[i, j, k]), -rm[i, j, k])
                end
            end

            # Fluxes at each x-face
            for i in 1:Nx+1
                i_left  = i == 1 ? Nx : i - 1
                i_right = i > Nx ? 1  : i

                if am[i, j, k] >= zero(FT)
                    alpha = am[i, j, k] / m[i_left, j, k]
                    f[i] = alpha * (rm[i_left, j, k] + (one(FT) - alpha) * sx[i_left])
                else
                    alpha = am[i, j, k] / m[i_right, j, k]
                    f[i] = alpha * (rm[i_right, j, k] - (one(FT) + alpha) * sx[i_right])
                end
            end
            # Periodic: face Nx+1 == face 1
            f[Nx + 1] = f[1]

            # Update rm
            for i in 1:Nx
                rm[i, j, k] += f[i] - f[i + 1]
            end
        end

        # --- Update air mass m ---
        for i in 1:Nx
            m[i, j, k] += am[i, j, k] - am[i + 1, j, k]
        end
    end
    return nothing
end

# =====================================================================
# Y-direction mass-flux advection
# =====================================================================

"""
$(SIGNATURES)

TM5-faithful slopes advection in the y-direction (latitude) using mass fluxes.

Modifies `rm_tracers` and `m` in place.  Zero-flux walls at poles.

`bm` is the northward mass flux at y-faces, size `(Nx, Ny+1, Nz)`.

Poles (j=1 and j=Ny) receive first-order treatment (no meridional slope)
following TM5's convention.
"""
function advect_y_massflux!(rm_tracers::NamedTuple,
                             m::AbstractArray{FT,3},
                             bm::AbstractArray{FT,3},
                             grid::LatitudeLongitudeGrid,
                             use_limiter::Bool) where FT
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    f  = Vector{FT}(undef, Ny + 1)
    sy = Vector{FT}(undef, Ny)

    @inbounds for k in 1:Nz, i in 1:Nx

        for (_, rm) in pairs(rm_tracers)

            # Slopes of concentration, scaled by m (preserves uniform fields)
            for j in 1:Ny
                if j > 1 && j < Ny
                    c_prev = rm[i, j - 1, k] / m[i, j - 1, k]
                    c_this = rm[i, j, k] / m[i, j, k]
                    c_next = rm[i, j + 1, k] / m[i, j + 1, k]
                    sc = (c_next - c_prev) / 2
                    if use_limiter
                        sc = minmod(sc, 2 * (c_next - c_this), 2 * (c_this - c_prev))
                    end
                    sy[j] = m[i, j, k] * sc
                    if use_limiter
                        sy[j] = max(min(sy[j], rm[i, j, k]), -rm[i, j, k])
                    end
                else
                    sy[j] = zero(FT)
                end
            end

            # Fluxes at each y-face
            # Face 1 = south boundary (zero flux)
            # Face Ny+1 = north boundary (zero flux)
            f[1] = zero(FT)
            f[Ny + 1] = zero(FT)

            for j in 2:Ny
                j_south = j - 1
                j_north = j

                if bm[i, j, k] >= zero(FT)
                    beta = bm[i, j, k] / m[i, j_south, k]
                    if j_south == 1
                        # South pole: first-order (no slope)
                        f[j] = beta * rm[i, j_south, k]
                    else
                        f[j] = beta * (rm[i, j_south, k] + (one(FT) - beta) * sy[j_south])
                    end
                else
                    beta = bm[i, j, k] / m[i, j_north, k]
                    if j_north == Ny
                        # North pole: first-order (no slope)
                        f[j] = beta * rm[i, j_north, k]
                    else
                        f[j] = beta * (rm[i, j_north, k] - (one(FT) + beta) * sy[j_north])
                    end
                end
            end

            # Update rm:  bm > 0 = northward, so mass enters j from south (face j)
            # and leaves j to north (face j+1).
            for j in 1:Ny
                rm[i, j, k] += f[j] - f[j + 1]
            end
        end

        # Update air mass m
        for j in 1:Ny
            m[i, j, k] += bm[i, j, k] - bm[i, j + 1, k]
        end
    end
    return nothing
end

# =====================================================================
# Z-direction mass-flux advection (TM5 dynamw_1d)
# =====================================================================

"""
$(SIGNATURES)

TM5-faithful slopes advection in the z-direction (vertical) using mass fluxes.

Modifies `rm_tracers` and `m` in place.

`cm` is the downward mass flux at z-interfaces, size `(Nx, Ny, Nz+1)`.
`cm > 0` means downward (from layer k toward layer k+1).
`cm[:,:,1] = 0` (top) and `cm[:,:,Nz+1] = 0` (surface).
"""
function advect_z_massflux!(rm_tracers::NamedTuple,
                             m::AbstractArray{FT,3},
                             cm::AbstractArray{FT,3},
                             use_limiter::Bool) where FT
    Nx, Ny, Nz = Base.size(m)

    f  = Vector{FT}(undef, Nz + 1)
    sz = Vector{FT}(undef, Nz)
    mnew = Vector{FT}(undef, Nz)

    @inbounds for j in 1:Ny, i in 1:Nx

        # New air mass distribution for this column
        for k in 1:Nz
            mnew[k] = m[i, j, k] + cm[i, j, k] - cm[i, j, k + 1]
        end

        for (_, rm) in pairs(rm_tracers)

            # Slopes of concentration, scaled by m (preserves uniform fields)
            for k in 1:Nz
                if k > 1 && k < Nz
                    c_prev = rm[i, j, k - 1] / m[i, j, k - 1]
                    c_this = rm[i, j, k] / m[i, j, k]
                    c_next = rm[i, j, k + 1] / m[i, j, k + 1]
                    sc = (c_next - c_prev) / 2
                    if use_limiter
                        sc = minmod(sc, 2 * (c_next - c_this), 2 * (c_this - c_prev))
                    end
                    sz[k] = m[i, j, k] * sc
                    if use_limiter
                        sz[k] = max(min(sz[k], rm[i, j, k]), -rm[i, j, k])
                    end
                else
                    sz[k] = zero(FT)
                end
            end

            # Fluxes at each z-interface
            # Interface 1 = top of atmosphere (zero flux)
            # Interface Nz+1 = surface (zero flux)
            f[1] = zero(FT)
            f[Nz + 1] = zero(FT)

            for k in 2:Nz
                k_above = k - 1
                k_below = k

                if cm[i, j, k] > zero(FT)
                    # Downward flow: donor = layer above (k_above)
                    gamma = cm[i, j, k] / m[i, j, k_above]
                    f[k] = gamma * (rm[i, j, k_above] + (one(FT) - gamma) * sz[k_above])
                elseif cm[i, j, k] < zero(FT)
                    # Upward flow: donor = layer below (k_below)
                    gamma = cm[i, j, k] / m[i, j, k_below]
                    f[k] = gamma * (rm[i, j, k_below] - (one(FT) + gamma) * sz[k_below])
                else
                    f[k] = zero(FT)
                end
            end

            # Update rm: cm > 0 = downward, so mass enters k from above (face k)
            # and leaves k downward (face k+1).
            for k in 1:Nz
                rm[i, j, k] += f[k] - f[k + 1]
            end
        end

        # Update m
        for k in 1:Nz
            m[i, j, k] = mnew[k]
        end
    end
    return nothing
end

# =====================================================================
# CFL-based mass-flux Courant number check
# =====================================================================

"""
$(SIGNATURES)

Maximum mass-based Courant number for x-direction mass fluxes.
"""
function max_cfl_massflux_x(am::AbstractArray{FT,3}, m::AbstractArray{FT,3}) where FT
    Nx, Ny, Nz = Base.size(m)
    cfl_max = zero(FT)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx+1
        i_left  = i == 1 ? Nx : i - 1
        i_right = i > Nx ? 1  : i
        m_donor = am[i, j, k] >= zero(FT) ? m[i_left, j, k] : m[i_right, j, k]
        cfl_max = max(cfl_max, abs(am[i, j, k]) / m_donor)
    end
    return cfl_max
end

"""
$(SIGNATURES)

Maximum mass-based Courant number for y-direction mass fluxes.
"""
function max_cfl_massflux_y(bm::AbstractArray{FT,3}, m::AbstractArray{FT,3}) where FT
    Nx, Ny, Nz = Base.size(m)
    cfl_max = zero(FT)
    @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
        j_south = j - 1
        j_north = j
        m_donor = bm[i, j, k] >= zero(FT) ? m[i, j_south, k] : m[i, j_north, k]
        cfl_max = max(cfl_max, abs(bm[i, j, k]) / m_donor)
    end
    return cfl_max
end

"""
$(SIGNATURES)

Maximum mass-based Courant number for z-direction mass fluxes.
"""
function max_cfl_massflux_z(cm::AbstractArray{FT,3}, m::AbstractArray{FT,3}) where FT
    Nx, Ny, Nz = Base.size(m)
    cfl_max = zero(FT)
    @inbounds for k in 2:Nz, j in 1:Ny, i in 1:Nx
        k_above = k - 1
        k_below = k
        m_donor = cm[i, j, k] > zero(FT) ? m[i, j, k_above] : m[i, j, k_below]
        if m_donor > zero(FT)
            cfl_max = max(cfl_max, abs(cm[i, j, k]) / m_donor)
        end
    end
    return cfl_max
end

# =====================================================================
# Subcycled mass-flux advection
# =====================================================================

"""
$(SIGNATURES)

CFL-adaptive subcycled x-advection in mass-flux form.  When the maximum
Courant number exceeds `cfl_limit`, the mass fluxes are divided into `n_sub`
equal sub-steps (following TM5's `advectx_get_nloop`).
"""
function advect_x_massflux_subcycled!(rm_tracers, m::AbstractArray{FT,3}, am,
                                       grid, use_limiter;
                                       cfl_limit = FT(0.95)) where FT
    cfl = max_cfl_massflux_x(am, m)
    n_sub = max(1, ceil(Int, cfl / cfl_limit))
    if n_sub > 1
        am_sub = am ./ FT(n_sub)
        for _ in 1:n_sub
            advect_x_massflux!(rm_tracers, m, am_sub, grid, use_limiter)
        end
    else
        advect_x_massflux!(rm_tracers, m, am, grid, use_limiter)
    end
    return n_sub
end

"""
$(SIGNATURES)

CFL-adaptive subcycled y-advection in mass-flux form.
"""
function advect_y_massflux_subcycled!(rm_tracers, m::AbstractArray{FT,3}, bm,
                                       grid, use_limiter;
                                       cfl_limit = FT(0.95)) where FT
    cfl = max_cfl_massflux_y(bm, m)
    n_sub = max(1, ceil(Int, cfl / cfl_limit))
    if n_sub > 1
        bm_sub = bm ./ FT(n_sub)
        for _ in 1:n_sub
            advect_y_massflux!(rm_tracers, m, bm_sub, grid, use_limiter)
        end
    else
        advect_y_massflux!(rm_tracers, m, bm, grid, use_limiter)
    end
    return n_sub
end

"""
$(SIGNATURES)

CFL-adaptive subcycled z-advection in mass-flux form.
"""
function advect_z_massflux_subcycled!(rm_tracers, m::AbstractArray{FT,3}, cm,
                                       use_limiter;
                                       cfl_limit = FT(0.95)) where FT
    cfl = max_cfl_massflux_z(cm, m)
    n_sub = max(1, ceil(Int, cfl / cfl_limit))
    if n_sub > 1
        cm_sub = cm ./ FT(n_sub)
        for _ in 1:n_sub
            advect_z_massflux!(rm_tracers, m, cm_sub, use_limiter)
        end
    else
        advect_z_massflux!(rm_tracers, m, cm, use_limiter)
    end
    return n_sub
end

# =====================================================================
# Convenience: full Strang-split mass-flux advection step
# =====================================================================

"""
$(SIGNATURES)

Perform a full Strang-split advection step (X-Y-Z-Z-Y-X) using TM5-style
mass-flux advection.

Converts concentration `tracers` to tracer mass using `m`, performs the split,
then converts back.  `m` is updated in-place to track air mass through the
split (NOT reset between directional steps — this is critical for consistency).

Arguments:
- `tracers`: NamedTuple of 3D concentration arrays (modified in-place)
- `m`: 3D air mass array (modified in-place; tracks continuously)
- `am, bm, cm`: pre-computed mass fluxes for one half-timestep
- `grid`: LatitudeLongitudeGrid
- `use_limiter`: enable slope limiter
- `cfl_limit`: CFL threshold for subcycling
"""
function strang_split_massflux!(tracers::NamedTuple,
                                 m::AbstractArray{FT,3},
                                 am, bm, cm,
                                 grid::LatitudeLongitudeGrid,
                                 use_limiter::Bool;
                                 cfl_limit::FT = FT(0.95)) where FT
    # Convert concentration → tracer mass
    rm_tracers = NamedTuple{keys(tracers)}(
        Tuple(m .* c for c in values(tracers))
    )

    # Strang split: X → Y → Z → Z → Y → X
    advect_x_massflux_subcycled!(rm_tracers, m, am, grid, use_limiter; cfl_limit)
    advect_y_massflux_subcycled!(rm_tracers, m, bm, grid, use_limiter; cfl_limit)
    advect_z_massflux_subcycled!(rm_tracers, m, cm, use_limiter; cfl_limit)
    advect_z_massflux_subcycled!(rm_tracers, m, cm, use_limiter; cfl_limit)
    advect_y_massflux_subcycled!(rm_tracers, m, bm, grid, use_limiter; cfl_limit)
    advect_x_massflux_subcycled!(rm_tracers, m, am, grid, use_limiter; cfl_limit)

    # Convert tracer mass → concentration
    for (name, c) in pairs(tracers)
        rm = rm_tracers[name]
        @inbounds @simd for idx in eachindex(c)
            c[idx] = rm[idx] / m[idx]
        end
    end

    return nothing
end
