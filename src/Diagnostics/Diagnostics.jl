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

export AbstractDiagnostic, ColumnMeanDiagnostic, SurfaceSliceDiagnostic, RegridDiagnostic
export Full3DDiagnostic, MetField2DDiagnostic
export column_mean!, surface_slice!, compute_diagnostics!
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

include("column_diagnostics.jl")
include("regridding_diagnostics.jl")

end # module Diagnostics
