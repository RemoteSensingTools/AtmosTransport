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

using ..Grids: AbstractGrid, grid_size

"""
$(SIGNATURES)

Convert the cell-center met data fields into the staggered arrays
required by the physics operators. Returns a NamedTuple with:

- `u`, `v`, `w` — staggered velocity components for advection
- `conv_mass_flux` — net convective mass flux at interfaces (if available)
- `diffusivity` — vertical diffusivity (if available)

This function is the bridge between the I/O system (canonical variables at
cell centers) and the physics operators (staggered grids).
"""
function prepare_met_for_physics(met::MetDataSource{FT}, grid::AbstractGrid) where {FT}
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
    # v[:, 1, :] = 0  and  v[:, Ny+1, :] = 0  (already zeros)

    # --- Vertical velocity: omega (Pa/s) → w at interfaces ---
    w = zeros(FT, Nx, Ny, Nz + 1)
    if has_variable(met, :w_wind) && haskey(met.buffers, :w_wind)
        omega = get_field(met, :w_wind)  # Pa/s, positive downward
        @inbounds for k in 2:Nz, j in 1:Ny, i in 1:Nx
            # Average adjacent level centers to interface,
            # negate because omega > 0 is downward, our w > 0 is upward
            w[i, j, k] = -(omega[i, j, k - 1] + omega[i, j, k]) / 2
        end
    end
    # w[:, :, 1] = 0 (model top) and w[:, :, Nz+1] = 0 (surface)

    result = (; u, v, w)

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

export prepare_met_for_physics
