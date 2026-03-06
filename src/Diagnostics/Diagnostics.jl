"""
    Diagnostics

Computed diagnostic fields for output and monitoring.

Provides GPU-compatible (KernelAbstractions) diagnostic computations
that dispatch on grid type via multiple dispatch:

    column_mean!(c_col, c, m)                                       # lat-lon
    column_mean!(c_col_panels, rm_panels, m_panels, Nc, Nz, Hp)    # cubed-sphere

    surface_slice!(c_sfc, c)                                        # lat-lon
    surface_slice!(c_sfc_panels, c_panels, Nc, Nz, Hp)             # cubed-sphere

    regrid_cs_to_latlon(panels, grid; Nlon, Nlat)                   # CS → lat-lon

# Abstract diagnostic types

Diagnostic types can be stored in output writer field dicts for automatic
computation during output:

    ColumnMeanDiagnostic(:co2)
    SurfaceSliceDiagnostic(:co2)
    RegridDiagnostic(:co2; Nlon=720, Nlat=361)
"""
module Diagnostics

using DocStringExtensions
using KernelAbstractions: @kernel, @index, @Const, @atomic, synchronize, get_backend
using ..Grids: AbstractGrid, LatitudeLongitudeGrid, CubedSphereGrid

export AbstractDiagnostic, ColumnMeanDiagnostic, ColumnMassDiagnostic,
       SurfaceSliceDiagnostic, RegridDiagnostic
export Full3DDiagnostic, MetField2DDiagnostic, SigmaLevelDiagnostic
export ColumnFluxDiagnostic, EmissionFluxDiagnostic
export column_mean!, column_mass!, surface_slice!, sigma_level_slice!, compute_diagnostics!
export column_tracer_flux!
export regrid_cs_to_latlon, regrid_cs_to_latlon!, RegridMapping, build_regrid_mapping

# --- Abstract diagnostic types ---

"""
$(TYPEDEF)

Supertype for diagnostic field computations. Subtypes can be stored in
`NetCDFOutputWriter` field dicts for automatic computation during output.
"""
abstract type AbstractDiagnostic end

"""
$(TYPEDEF)

Mass-weighted column mean of a tracer.

$(FIELDS)
"""
struct ColumnMeanDiagnostic <: AbstractDiagnostic
    "tracer symbol (e.g. :co2)"
    species :: Symbol
end

"""
$(TYPEDEF)

Surface layer slice of a tracer (k = Nz).

$(FIELDS)
"""
struct SurfaceSliceDiagnostic <: AbstractDiagnostic
    "tracer symbol"
    species :: Symbol
end

"""
$(TYPEDEF)

Cubed-sphere to lat-lon regridded diagnostic.

$(FIELDS)
"""
struct RegridDiagnostic <: AbstractDiagnostic
    "tracer symbol"
    species :: Symbol
    "output longitude count"
    Nlon    :: Int
    "output latitude count"
    Nlat    :: Int
end

RegridDiagnostic(species::Symbol; Nlon=720, Nlat=361) =
    RegridDiagnostic(species, Nlon, Nlat)

"""
$(TYPEDEF)

Full 3D tracer field diagnostic. Outputs the complete (lon, lat, lev) tracer
array. Required by CATRINE protocol for all 4 tracers every 3 hours.

$(FIELDS)
"""
struct Full3DDiagnostic <: AbstractDiagnostic
    "tracer symbol (e.g. :co2)"
    species :: Symbol
end

"""
$(TYPEDEF)

2D met field diagnostic (surface pressure, PBL height, tropopause height, etc.).
The field is retrieved from model.met_data or computed on-the-fly.

$(FIELDS)
"""
struct MetField2DDiagnostic <: AbstractDiagnostic
    "met field identifier (e.g. :surface_pressure, :pbl_height, :tropopause_height)"
    field_name :: Symbol
end

"""
$(TYPEDEF)

Tracer mixing ratio at a target sigma level (pressure fraction of surface pressure).
For sigma=0.8, extracts at the level where p ≈ 0.8 × p_surface (~800 hPa).

$(FIELDS)
"""
struct SigmaLevelDiagnostic <: AbstractDiagnostic
    "tracer symbol (e.g. :co2)"
    species :: Symbol
    "target sigma level (0..1, where 1 = surface)"
    sigma   :: Float64
end

"""
$(TYPEDEF)

Column-integrated tracer mass (kg/m²). Computes mc = Σ_k(c_k × m_k) / A
for lat-lon, or mc = Σ_k(rm_k) / A for cubed-sphere, where A is cell area.
Required by CATRINE protocol: mc = (1/g) ∫₀ᵖˢ qc dp.

$(FIELDS)
"""
struct ColumnMassDiagnostic <: AbstractDiagnostic
    "tracer symbol (e.g. :co2)"
    species :: Symbol
end

"""
$(TYPEDEF)

Time-integrated vertically-integrated column tracer flux (kg/m).

Computes FE = ∫₀ᵀ (1/g) ∫₀ᵖˢ u×qc dp dt  (eastward) or
         FN = ∫₀ᵀ (1/g) ∫₀ᵖˢ v×qc dp dt  (northward)

where T is the output interval (e.g. 3 hours). Requires an accumulator
in the output writer that is updated every met window and reset on output.

$(FIELDS)
"""
struct ColumnFluxDiagnostic <: AbstractDiagnostic
    "tracer species symbol (e.g. :co2, :sf6)"
    species   :: Symbol
    "flux direction: :east for zonal (u×qc) or :north for meridional (v×qc)"
    direction :: Symbol
end

"""
$(TYPEDEF)

Surface emission flux for a tracer species [kg/m²/s].
Extracts the current emission flux from the model's source vector.
For time-varying emissions, returns the active snapshot.

$(FIELDS)
"""
struct EmissionFluxDiagnostic <: AbstractDiagnostic
    "tracer species symbol (e.g. :fossil_co2, :sf6)"
    species :: Symbol
end

include("column_diagnostics.jl")
include("regridding_diagnostics.jl")

end # module Diagnostics
