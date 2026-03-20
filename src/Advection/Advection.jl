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
using ..Grids: fill_panel_halos!, copy_corners!, allocate_cubed_sphere_field
using ..Fields: AbstractField
using ..Architectures: architecture, device, array_type, CPU, GPU
using KernelAbstractions: get_backend, synchronize

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
export AbstractPPMScheme, PPMAdvection
export PratherAdvection, PratherWorkspace, allocate_prather_workspace
export CSPratherWorkspace, allocate_cs_prather_workspace, allocate_cs_prather_workspaces
export strang_split_prather_cs!
export strang_split_prather!
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
export strang_split_massflux!, strang_split_massflux_ppm!
export advect_x_massflux_reduced!
export CubedSphereGeometryCache, CubedSphereMassFluxWorkspace
export allocate_cs_massflux_workspace
export max_cfl_x_cs, max_cfl_y_cs
export compute_air_mass_panel!, compute_cm_panel!, compute_cm_pressure_fixer_panel!
export apply_mass_fixer!
export compute_cm_pressure_fixer_panel!, compute_dm_per_sub_panel!
export compute_cm_mass_weighted_panel!
export apply_dry_delp_panel!, apply_dry_am_panel!, apply_dry_bm_panel!
export apply_dry_cmfmc_panel!, apply_dry_dtrain_panel!
export apply_dry_mref_ll!, apply_dry_am_ll!, apply_dry_bm_ll!
export apply_dry_cmfmc_ll!, apply_dry_dtrain_ll!, recompute_cm_ll!
export apply_divergence_damping_cs!
export advect_x_cs_panel!, advect_y_cs_panel!, advect_z_cs_panel!
export advect_z_cs_panel_column!
export LinRoodWorkspace, fv_tp_2d_cs!, strang_split_linrood_ppm!
export VerticalRemapWorkspace, vertical_remap_cs!, fix_target_bottom_pe!
export compute_target_pressure_from_next_delp!, compute_target_pressure_from_mass!
export compute_target_pressure_from_delp_direct!, compute_target_pressure_from_mass_direct!
export compute_target_pressure_from_dry_delp_direct!
export compute_target_pe_from_hybrid_coords!
export compute_source_pe_from_hybrid!, compute_target_pe_from_ps_hybrid!
export compute_source_pe_from_evolved_mass!, compute_target_pe_from_evolved_ps!
export _lock_surface_pe_kernel!, _copy_dp_tgt_to_dp_work_kernel!
export _scale_dp_tgt_to_source_ps_kernel!
export _column_sum_rm_kernel!, _column_mass_correct_kernel!
export update_air_mass_from_target!
export calc_scaling_factor, apply_scaling_factor!, gchp_calc_scaling_factor, fillz_panels!
export compute_dry_ple!
export fv_tp_2d_cs_q!, rm_to_q_panels!, q_to_rm_panels!, fillz_q!
export compute_dp_from_m_panels!, set_m_from_dp_panels!
export GCHPGridGeometry, GCHPTransportWorkspace
export fv_tp_2d_gchp!, strang_split_gchp_ppm!, compute_area_fluxes!
export fv_tp_2d_gchp_fluxes!, gchp_tracer_2d!
export _correct_mfx_humidity_kernel!, _correct_mfy_humidity_kernel!
export _reverse_mfx_humidity_kernel!, _reverse_mfy_humidity_kernel!
export _multiply_by_1_minus_qv_kernel!, _divide_by_1_minus_qv_kernel!
export _compute_dry_dp_kernel!, _copy_nohalo_to_halo_kernel!,
       _interpolate_dry_dp_kernel!, _interpolate_dp_kernel!,
       _column_dp_correction_kernel!, _column_dp_correction_moist_kernel!

# Kahan compensated addition for Float32 precision in cumulative sums.
# Float32: tracks low-order bits lost in each addition (nearly Float64 precision).
# Float64: plain addition (compensation is always zero — no overhead).
@inline function _kahan_add(s::T, c::T, x::T) where {T <: Union{Float16, Float32}}
    y = x - c
    t = s + y
    c_new = (t - s) - y
    return (t, c_new)
end
@inline _kahan_add(s::T, c::T, x::T) where {T <: Float64} = (s + x, zero(T))

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
include("cubed_sphere_mass_flux.jl")
include("ppm_advection.jl")
include("ppm_subgrid_distributions.jl")
include("cubed_sphere_mass_flux_ppm.jl")
include("cubed_sphere_fvtp2d.jl")
include("cubed_sphere_fvtp2d_gchp.jl")
include("vertical_remap.jl")
include("latlon_mass_flux_ppm.jl")
include("latlon_dry_air.jl")
include("prather_advection.jl")
include("cubed_sphere_prather.jl")

end # module Advection
