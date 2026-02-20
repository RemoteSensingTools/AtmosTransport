"""
    Advection

Advection schemes for tracer transport, each with a paired discrete adjoint.

Every advection scheme dispatches on both the scheme type AND the grid type,
so that grid-specific stencils (e.g. cubed-sphere panel boundaries) are
handled transparently.

# Interface contract for new schemes

    advect!(tracers, velocities, grid, scheme::YourScheme, Δt)
    adjoint_advect!(adj_tracers, velocities, grid, scheme::YourScheme, Δt)

Optional (has defaults):

    advection_cfl(grid, velocities, scheme::YourScheme)
"""
module Advection

using DocStringExtensions

using ..Grids: AbstractGrid, AbstractStructuredGrid, LatitudeLongitudeGrid, CubedSphereGrid
using ..Grids: Δx, Δy, Δz, floattype, level_thickness, cell_area
using ..Grids: ReducedGridSpec, reduce_row!, expand_row!, reduce_velocity_row!
using ..Grids: reduce_row_mass!, reduce_am_row!, expand_row_mass!
using ..Grids: fill_panel_halos!, allocate_cubed_sphere_field
using ..Fields: AbstractField
using ..Architectures: architecture, device, array_type, CPU, GPU

"""
Extract `p_surface` from velocities if present, otherwise `nothing`.
"""
_get_p_surface(vel) = hasproperty(vel, :p_surface) ? vel.p_surface : nothing

"""
Build a 3D layer-thickness array `(Nx, Ny, Nz)` using per-column surface pressure
when available, falling back to `grid.reference_pressure`.
"""
function _build_Δz_3d(grid, p_surface)
    FT = floattype(grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δz_3d = Array{FT}(undef, Nx, Ny, Nz)
    _build_Δz_3d!(Δz_3d, grid, p_surface)
    return Δz_3d
end

function _build_Δz_3d!(Δz_3d, grid, p_surface)
    FT = floattype(grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ps = p_surface !== nothing ? FT(p_surface[i, j]) : grid.reference_pressure
        Δz_3d[i, j, k] = level_thickness(grid.vertical, k, ps)
    end
    return nothing
end

export AbstractAdvectionScheme
export SlopesAdvection, UpwindAdvection
export advect!, adjoint_advect!
export advect_x!, advect_y!, advect_z!
export adjoint_advect_x!, adjoint_advect_y!, adjoint_advect_z!
export advection_cfl
export max_cfl_x, max_cfl_y, max_cfl_z, subcycling_counts
export advect_x_subcycled!, advect_y_subcycled!, advect_z_subcycled!
export update_pressure_x!, update_pressure_y!, update_pressure_z!
export apply_mass_correction!
export advect_x_mass_corrected!, advect_y_mass_corrected!, advect_z_mass_corrected!
export advect_x_mass_corrected_subcycled!, advect_y_mass_corrected_subcycled!, advect_z_mass_corrected_subcycled!
export compute_air_mass, compute_air_mass!, compute_mass_fluxes, compute_mass_fluxes!
export GridGeometryCache, build_geometry_cache
export MassFluxWorkspace, allocate_massflux_workspace
export advect_x_massflux!, advect_y_massflux!, advect_z_massflux!
export advect_x_massflux_subcycled!, advect_y_massflux_subcycled!, advect_z_massflux_subcycled!
export max_cfl_massflux_x, max_cfl_massflux_y, max_cfl_massflux_z
export strang_split_massflux!
export advect_x_massflux_reduced!

include("abstract_advection.jl")
include("slopes_advection_kernels.jl")
include("slopes_advection.jl")
include("slopes_advection_adjoint.jl")
include("upwind_advection.jl")
include("upwind_advection_adjoint.jl")
include("cubed_sphere_advection.jl")
include("subcycling.jl")
include("mass_correction.jl")
include("mass_flux_advection.jl")

end # module Advection
