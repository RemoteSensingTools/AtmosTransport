"""
    Grids

Grid types for atmospheric transport: latitude-longitude and cubed-sphere,
with hybrid sigma-pressure vertical coordinates.

All physics code accesses grid properties through generic accessor functions
(`xnode`, `ynode`, `znode`, `cell_area`, `cell_volume`, `Δx`, `Δy`, `Δz`)
that dispatch on `AbstractGrid`, ensuring grid-agnostic kernels.

# Concrete grid types

- `LatitudeLongitudeGrid` — regular lat-lon (ERA5/TM5 native)
- `CubedSphereGrid` — equidistant gnomonic cubed-sphere (GEOS/MERRA-2 native)

# Vertical coordinates

- `HybridSigmaPressure` — hybrid σ-p levels (ERA5 137L, MERRA-2 72L, TM5 25-60L)
"""
module Grids

using DocStringExtensions
using KernelAbstractions
using ..Architectures: AbstractArchitecture, CPU, array_type,
                       AbstractPanelMap, SingleGPUMap, PanelGPUMap,
                       allocate_ntuple_panels, sync_all_gpus, is_cross_gpu, is_multi_gpu
import ..Architectures: architecture

export AbstractGrid, AbstractStructuredGrid
export LatitudeLongitudeGrid, CubedSphereGrid
export GRID_COORD_STATUS, set_coord_status!, has_gmao_coords
export AbstractVerticalCoordinate, HybridSigmaPressure
export AbstractTopology, Periodic, Bounded, CubedPanel, Flat
export AbstractLocationType, Center, Face
export xnode, ynode, znode, cell_area, cell_volume
export Δx, Δy, Δz, level_thickness
export topology, halo_size, grid_size, floattype
export ReducedGridSpec, compute_reduced_grid
export allocate_cubed_sphere_field, fill_panel_halos!, fill_panel_halos_nosync!, copy_corners!, fill_cgrid_halos!
export merge_upper_levels, merge_thin_levels
export get_panel_map

include("topology.jl")
include("location_types.jl")
include("abstract_grid.jl")
include("vertical_coordinates.jl")
include("reduced_grid.jl")
include("latitude_longitude_grid.jl")
include("cubed_sphere_grid.jl")
include("panel_connectivity.jl")
include("grid_utils.jl")
include("halo_exchange.jl")

end # module Grids
