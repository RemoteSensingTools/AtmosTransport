# ---------------------------------------------------------------------------
# Met-to-physics bridge
#
# Converts MetDataSource buffers (canonical variable names, cell-center)
# into the staggered arrays expected by the physics operators:
#   - u at x-faces: (Nx+1, Ny, Nz)     [periodic in x]
#   - v at y-faces: (Nx, Ny+1, Nz)     [zero at boundaries]
#   - w at z-interfaces: (Nx, Ny, Nz+1) [zero at top/bottom]
#   - conv_mass_flux: (Nx, Ny, Nz+1)    [net convective mass flux]
#   - diffusivity: (Nx, Ny, Nz)         [vertical diffusivity Kz]
# ---------------------------------------------------------------------------

using ..Grids: AbstractGrid, LatitudeLongitudeGrid, grid_size,
               cell_area, level_thickness, pressure_at_interface

"""
$(SIGNATURES)

Convert the cell-center met data fields into the staggered arrays
required by the physics operators. Returns a NamedTuple with:

- `u`, `v`, `w` — staggered velocity components for advection
- `conv_mass_flux` — net convective mass flux at interfaces (if available)
- `diffusivity` — vertical diffusivity (if available)

This function is the bridge between the I/O system (canonical variables at
cell centers) and the physics operators (staggered grids).

# Keyword arguments
- `use_continuity_omega`: if `true` (default for `LatitudeLongitudeGrid`),
  compute `w` from horizontal wind divergence via the continuity equation
  rather than staggering raw omega.  This eliminates extreme values caused by
  inconsistent wind divergence and is the TM5 approach.
- `p_surface`: optional per-column surface pressure `(Nx, Ny)` array.
"""
function prepare_met_for_physics(met::MetDataSource{FT}, grid::AbstractGrid;
                                 use_continuity_omega::Bool = grid isa LatitudeLongitudeGrid,
                                 p_surface::Union{AbstractMatrix{FT}, Nothing} = nothing) where {FT}
    gs = grid_size(grid)
    Nx, Ny, Nz = gs.Nx, gs.Ny, gs.Nz

    # --- Horizontal velocities: cell centers → face values ---
    u_cc = get_field(met, :u_wind)
    v_cc = get_field(met, :v_wind)

    # u at x-faces: (Nx+1, Ny, Nz), periodic
    u = zeros(FT, Nx + 1, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny
        for i in 1:Nx
            ip = i == Nx ? 1 : i + 1
            u[i, j, k] = (u_cc[ip, j, k] + u_cc[i, j, k]) / 2
        end
        u[Nx + 1, j, k] = u[1, j, k]  # periodic wrap
    end

    # v at y-faces: (Nx, Ny+1, Nz), zero at boundaries
    v = zeros(FT, Nx, Ny + 1, Nz)
    @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
        v[i, j, k] = (v_cc[i, j - 1, k] + v_cc[i, j, k]) / 2
    end

    # --- Vertical velocity ---
    w = if use_continuity_omega && grid isa LatitudeLongitudeGrid
        compute_continuity_omega(u, v, grid; p_surface)
    else
        # Fallback: stagger raw omega
        _w = zeros(FT, Nx, Ny, Nz + 1)
        if has_variable(met, :w_wind) && haskey(met.buffers, :w_wind)
            omega = get_field(met, :w_wind)
            @inbounds for k in 2:Nz, j in 1:Ny, i in 1:Nx
                _w[i, j, k] = (omega[i, j, k - 1] + omega[i, j, k]) / 2
            end
        end
        _w
    end

    result = p_surface !== nothing ? (; u, v, w, p_surface) : (; u, v, w)

    # --- Convective mass flux (net: up - down) at interfaces ---
    has_up = has_variable(met, :conv_mass_flux_up) &&
             haskey(met.buffers, :conv_mass_flux_up)
    has_down = has_variable(met, :conv_mass_flux_down) &&
               haskey(met.buffers, :conv_mass_flux_down)

    if has_up
        mf_up = get_field(met, :conv_mass_flux_up)
        mf_net = if has_down
            mf_down = get_field(met, :conv_mass_flux_down)
            FT.(mf_up) .- FT.(mf_down)
        else
            FT.(mf_up)
        end
        result = merge(result, (; conv_mass_flux = mf_net))
    end

    # --- Vertical diffusivity ---
    if has_variable(met, :diffusivity) && haskey(met.buffers, :diffusivity)
        Kz = FT.(get_field(met, :diffusivity))
        result = merge(result, (; diffusivity = Kz))
    end

    return result
end

"""
$(SIGNATURES)

Compute continuity-consistent vertical velocity (omega, Pa/s) from staggered
horizontal winds on a `LatitudeLongitudeGrid`.  This follows the TM5 approach
(Bregman et al. 2003): integrate the horizontal mass flux divergence downward
from the model top, then apply a correction so that omega vanishes at both the
top and bottom boundaries.

The result satisfies the discrete continuity equation column-by-column, which
eliminates the spurious convergence/divergence extremes that arise when using
raw ERA5 omega with a fixed reference-pressure grid spacing.

# Arguments
- `u`:  zonal wind on x-faces, size `(Nx+1, Ny, Nz)` [m/s]
- `v`:  meridional wind on y-faces, size `(Nx, Ny+1, Nz)` [m/s]
- `grid`: the `LatitudeLongitudeGrid` providing geometry and vertical coordinate

# Keyword arguments
- `p_surface`:  per-column surface pressure `(Nx, Ny)` array.  When `nothing`
  (default) the grid's reference pressure is used uniformly.

# Returns
- `ω`:  vertical velocity at z-interfaces, size `(Nx, Ny, Nz+1)` [Pa/s].
  Sign convention: `ω > 0` = downward (toward higher pressure / increasing k).
  Boundary values `ω[:,:,1] = 0` (model top) and `ω[:,:,Nz+1] = 0` (surface)
  are guaranteed by construction.
"""
function compute_continuity_omega(
        u::AbstractArray{FT, 3},
        v::AbstractArray{FT, 3},
        grid::LatitudeLongitudeGrid{FT};
        p_surface::Union{AbstractMatrix{FT}, Nothing} = nothing) where FT

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    R  = grid.radius
    Δλ = FT(deg2rad(grid.Δλ))
    Δφ = FT(deg2rad(grid.Δφ))

    ω = zeros(FT, Nx, Ny, Nz + 1)

    @inbounds for k in 1:Nz
        for j in 1:Ny
            A_j  = cell_area(1, j, grid)
            Ly   = R * Δφ                                          # y-edge length (uniform)
            Lx_n = R * FT(cosd(Array(grid.φᶠ)[j + 1])) * Δλ      # north x-edge
            Lx_s = R * FT(cosd(Array(grid.φᶠ)[j]))     * Δλ      # south x-edge

            for i in 1:Nx
                ps_ij = p_surface !== nothing ? p_surface[i, j] : grid.reference_pressure
                Δp_k  = level_thickness(grid.vertical, k, ps_ij)

                u_E = u[i + 1, j, k]
                u_W = u[i,     j, k]
                v_N = v[i, j + 1, k]
                v_S = v[i, j,     k]

                # Finite-volume horizontal wind divergence [1/s]
                div_h = (Ly * (u_E - u_W) + Lx_n * v_N - Lx_s * v_S) / A_j

                ω[i, j, k + 1] = ω[i, j, k] - div_h * Δp_k
            end
        end
    end

    # --- Correction: distribute bottom residual so ω(Nz+1) = 0 ---
    @inbounds for j in 1:Ny, i in 1:Nx
        ps_ij      = p_surface !== nothing ? p_surface[i, j] : grid.reference_pressure
        ω_residual = ω[i, j, Nz + 1]
        p_top      = pressure_at_interface(grid.vertical, 1, ps_ij)
        p_total    = pressure_at_interface(grid.vertical, Nz + 1, ps_ij) - p_top
        if abs(p_total) < eps(FT)
            continue
        end
        cumul = zero(FT)
        for k in 2:Nz + 1
            cumul += level_thickness(grid.vertical, k - 1, ps_ij)
            ω[i, j, k] -= ω_residual * cumul / p_total
        end
    end

    return ω
end

export prepare_met_for_physics, compute_continuity_omega
