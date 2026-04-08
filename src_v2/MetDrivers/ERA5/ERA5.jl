"""
    ERA5 met driver (v2)

ERA5-specific dry flux building from spectral or gridded fields.
"""
module ERA5

import ..AbstractMassFluxMetDriver, ..AbstractMassClosure
import ..DiagnoseVerticalFromHorizontal
import ..build_dry_fluxes!, ..build_air_mass!
import ..supports_moisture
using ...State: MetState,
    AbstractStructuredFaceFluxState, StructuredFaceFluxState,
    AbstractMassFluxBasis, MoistMassFluxBasis, DryMassFluxBasis,
    MoistStructuredFluxState, DryStructuredFluxState
using ...Grids: AtmosGrid, LatLonMesh, HybridSigmaPressure,
    n_levels, pressure_at_interface, level_thickness, b_diff, cell_areas_by_latitude

include("BinaryReader.jl")
include("VerticalClosure.jl")
include("DryFluxBuilder.jl")

end # module ERA5
