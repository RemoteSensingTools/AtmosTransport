# ---------------------------------------------------------------------------
# DEPRECATED: Post-hoc mass correction (pressure fixer)
#
# This approach is superseded by mass_flux_advection.jl, which implements
# TM5's native mass-flux formulation (co-advection of tracer mass and air
# mass).  The post-hoc correction here is retained for backward compatibility
# but should NOT be used for production runs.
#
# See mass_flux_advection.jl for the correct implementation.
# ---------------------------------------------------------------------------

"""
$(SIGNATURES)

Update pressure thickness `Δp` in the x-direction using first-order upwind
mass-flux divergence.  Periodic boundary conditions in longitude.

Modifies `Δp` in place.
"""
function update_pressure_x!(Δp::AbstractArray{FT,3}, u, grid::LatitudeLongitudeGrid,
                             Δt) where FT
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δp_new = similar(Δp)

    @inbounds for k in 1:Nz, j in 1:Ny
        Δx_j = FT(Δx(1, j, grid))
        for i in 1:Nx
            i_prev = i == 1 ? Nx : i - 1
            i_next = i == Nx ? 1 : i + 1

            u_R = u[i + 1, j, k]
            u_L = u[i, j, k]

            Δp_donor_R = u_R >= 0 ? Δp[i, j, k]      : Δp[i_next, j, k]
            Δp_donor_L = u_L >= 0 ? Δp[i_prev, j, k]  : Δp[i, j, k]

            Δp_new[i, j, k] = Δp[i, j, k] - FT(Δt) / Δx_j * (u_R * Δp_donor_R - u_L * Δp_donor_L)
        end
    end
    copyto!(Δp, Δp_new)
    return nothing
end

"""
$(SIGNATURES)

Update pressure thickness `Δp` in the y-direction using first-order upwind
mass-flux divergence.  Zero-flux (wall) boundary at poles.

Modifies `Δp` in place.
"""
function update_pressure_y!(Δp::AbstractArray{FT,3}, v, grid::LatitudeLongitudeGrid,
                             Δt) where FT
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δp_new = similar(Δp)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Δy_j = FT(Δy(i, j, grid))

        v_N = v[i, j + 1, k]
        v_S = v[i, j, k]

        Δp_donor_N = if j < Ny
            v_N >= 0 ? Δp[i, j, k] : Δp[i, j + 1, k]
        else
            v_N >= 0 ? Δp[i, j, k] : Δp[i, j, k]
        end

        Δp_donor_S = if j > 1
            v_S >= 0 ? Δp[i, j - 1, k] : Δp[i, j, k]
        else
            v_S >= 0 ? Δp[i, j, k] : Δp[i, j, k]
        end

        Δp_new[i, j, k] = Δp[i, j, k] - FT(Δt) / Δy_j * (v_N * Δp_donor_N - v_S * Δp_donor_S)
    end
    copyto!(Δp, Δp_new)
    return nothing
end

"""
$(SIGNATURES)

Update pressure thickness `Δp` in the z-direction using the continuity equation.
`w` is omega at z-interfaces (Nx, Ny, Nz+1) with w > 0 = downward.

Zero flux at the model top (k=1) and surface (k=Nz+1).

Modifies `Δp` in place.
"""
function update_pressure_z!(Δp::AbstractArray{FT,3}, w, Δt) where FT
    Nx, Ny, Nz = Base.size(Δp)
    Δp_new = similar(Δp)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        w_top = w[i, j, k]
        w_bot = w[i, j, k + 1]
        Δp_new[i, j, k] = Δp[i, j, k] - FT(Δt) * (w_bot - w_top)
    end
    copyto!(Δp, Δp_new)
    return nothing
end

"""
$(SIGNATURES)

Apply the mass correction `c *= Δp_old / Δp_new` to all tracers.
"""
function apply_mass_correction!(tracers::NamedTuple,
                                 Δp_old::AbstractArray{FT,3},
                                 Δp_new::AbstractArray{FT,3}) where FT
    for (_, c) in pairs(tracers)
        @inbounds @simd for idx in eachindex(c)
            c[idx] *= Δp_old[idx] / Δp_new[idx]
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Mass-corrected advection wrappers
# ---------------------------------------------------------------------------

"""
$(SIGNATURES)

Advect tracers in x with TM5-style mass correction.

1. Save `Δp` before advection.
2. Run concentration-based `advect_x!`.
3. Update `Δp` via 1D mass-flux divergence.
4. Rescale: `c *= Δp_old / Δp_new`.

`Δp` is modified **in place** to reflect the post-advection air mass.
"""
function advect_x_mass_corrected!(tracers, velocities, grid::LatitudeLongitudeGrid,
                                   scheme::AbstractAdvectionScheme, Δt,
                                   Δp::AbstractArray{FT,3}) where FT
    Δp_old = copy(Δp)
    advect_x!(tracers, velocities, grid, scheme, Δt)
    update_pressure_x!(Δp, velocities.u, grid, Δt)
    apply_mass_correction!(tracers, Δp_old, Δp)
    return nothing
end

"""
$(SIGNATURES)

Advect tracers in y with TM5-style mass correction.
"""
function advect_y_mass_corrected!(tracers, velocities, grid::LatitudeLongitudeGrid,
                                   scheme::AbstractAdvectionScheme, Δt,
                                   Δp::AbstractArray{FT,3}) where FT
    Δp_old = copy(Δp)
    advect_y!(tracers, velocities, grid, scheme, Δt)
    update_pressure_y!(Δp, velocities.v, grid, Δt)
    apply_mass_correction!(tracers, Δp_old, Δp)
    return nothing
end

"""
$(SIGNATURES)

Advect tracers in z with TM5-style mass correction.
"""
function advect_z_mass_corrected!(tracers, velocities, grid::LatitudeLongitudeGrid,
                                   scheme::AbstractAdvectionScheme, Δt,
                                   Δp::AbstractArray{FT,3}) where FT
    Δp_old = copy(Δp)
    advect_z!(tracers, velocities, grid, scheme, Δt)
    update_pressure_z!(Δp, velocities.w, Δt)
    apply_mass_correction!(tracers, Δp_old, Δp)
    return nothing
end

# ---------------------------------------------------------------------------
# Subcycled mass-corrected advection
# ---------------------------------------------------------------------------

"""
$(SIGNATURES)

CFL-adaptive subcycled x-advection with mass correction applied at every
sub-step.
"""
function advect_x_mass_corrected_subcycled!(tracers, vel, grid, scheme, dt,
                                             Δp::AbstractArray{FT,3};
                                             n_sub=nothing, cfl_limit=0.95) where FT
    if n_sub === nothing
        cfl = max_cfl_x(vel, grid, dt)
        n_sub = max(1, ceil(Int, cfl / cfl_limit))
    end
    dt_sub = dt / n_sub
    for _ in 1:n_sub
        advect_x_mass_corrected!(tracers, vel, grid, scheme, dt_sub, Δp)
    end
    return n_sub
end

"""
$(SIGNATURES)

CFL-adaptive subcycled y-advection with mass correction applied at every
sub-step.
"""
function advect_y_mass_corrected_subcycled!(tracers, vel, grid, scheme, dt,
                                             Δp::AbstractArray{FT,3};
                                             n_sub=nothing, cfl_limit=0.95) where FT
    if n_sub === nothing
        cfl = max_cfl_y(vel, grid, dt)
        n_sub = max(1, ceil(Int, cfl / cfl_limit))
    end
    dt_sub = dt / n_sub
    for _ in 1:n_sub
        advect_y_mass_corrected!(tracers, vel, grid, scheme, dt_sub, Δp)
    end
    return n_sub
end

"""
$(SIGNATURES)

CFL-adaptive subcycled z-advection with mass correction applied at every
sub-step.
"""
function advect_z_mass_corrected_subcycled!(tracers, vel, grid, scheme, dt,
                                             Δp::AbstractArray{FT,3};
                                             n_sub=nothing, cfl_limit=0.95) where FT
    if n_sub === nothing
        cfl = max_cfl_z(vel, grid, dt)
        n_sub = max(1, ceil(Int, cfl / cfl_limit))
    end
    dt_sub = dt / n_sub
    for _ in 1:n_sub
        advect_z_mass_corrected!(tracers, vel, grid, scheme, dt_sub, Δp)
    end
    return n_sub
end
