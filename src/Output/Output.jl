"""
    Output

Topology-aware diagnostic output for AtmosTransport runtime products.

This module owns the public NetCDF output contract. Runtime code should capture
model state with [`capture_snapshot`](@ref) and write files with
[`write_snapshot_netcdf`](@ref) instead of defining ad-hoc NetCDF layouts in
runner code.

The writer is intentionally topology-dispatched:

- [`LatLonMesh`](@ref) writes regular CF lon/lat coordinates.
- [`ReducedGaussianMesh`](@ref) writes native cell-indexed diagnostics plus a
  legacy lon/lat raster view for current debug plots.
- [`CubedSphereMesh`](@ref) writes panel-native diagnostics with GEOS-style
  `lons`, `lats`, `corner_lons`, `corner_lats`, `cell_area`, and
  `cubed_sphere` metadata so Panoply and downstream tools have enough geometry
  context to render C-grid snapshots.

New topologies should add methods for the small internal schema/diagnostic
functions in this folder; they should not special-case the runner.
"""
module Output

using NCDatasets
using Printf

using ..Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh,
               GnomonicPanelConvention, GEOSNativePanelConvention,
               nx, ny, nrings, ring_longitudes, cell_index, ncells,
               cell_area, panel_cell_center_lonlat, panel_cell_corner_lonlat
using ..State: DryBasis, MoistBasis, mass_basis, tracer_names, get_tracer

export SnapshotFrame, SnapshotWriteOptions
export capture_snapshot, write_snapshot_netcdf
export column_mean_mixing_ratio, layer_mass_per_area, column_mass_per_area

include("snapshots.jl")
include("diagnostics.jl")
include("netcdf_schema.jl")
include("netcdf_writer.jl")

end # module Output
