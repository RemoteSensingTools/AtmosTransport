# ---------------------------------------------------------------------------
# GEOS-FP Native Cubed-Sphere Data Reader
#
# Reads MFXC (eastward mass flux), MFYC (northward mass flux), and DELP
# (pressure thickness) from native GEOS-FP cubed-sphere NetCDF files
# archived by the GEOS-Chem Support Team at Washington University.
#
# Data source: http://geoschemdata.wustl.edu/ExtData/GEOS_C720/
# Reference: Martin et al. (2022), GMD, doi:10.5194/gmd-15-8325-2022
#
# File: GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.YYYYMMDD_HHMM.V01.nc4
#
# Dimension order: (Xdim=720, Ydim=720, nf=6, lev=72, time=1)
#   - `nf` comes BEFORE `lev` (panel index before vertical)
#
# Variables:
#   MFXC: (720,720,6,72,1) pressure-weighted accumulated eastward mass flux [Pa m²]
#   MFYC: (720,720,6,72,1) pressure-weighted accumulated northward mass flux [Pa m²]
#   DELP: (720,720,6,72,1) pressure thickness [Pa]
#   PS:   (720,720,6,1)    surface pressure [Pa]
#   CX:   (720,720,6,72,1) eastward accumulated Courant number
#   CY:   (720,720,6,72,1) northward accumulated Courant number
#
# Mass flux convention (C-grid):
#   MFXC(i,j) = flux through the EAST face of cell (i,j)
#   MFYC(i,j) = flux through the NORTH face of cell (i,j)
#   Both stored at cell-center indices (NOT staggered Nc+1).
#
# Unit conversion to kg/s:
#   mass_flux_kgs = MFXC / (g × Δt_met)
#   where g = 9.80665 m/s², Δt_met = accumulation interval in seconds
# ---------------------------------------------------------------------------

using NCDatasets
using Dates

const GRAV_GEOSFP = 9.80665  # standard gravity [m/s²]

"""
$(TYPEDEF)

Container for one timestep of GEOS-FP cubed-sphere mass-flux data.

Mass fluxes are stored in C-grid convention: MFXC(i,j) is the flux through
the east face of cell (i,j). Unlike a staggered grid, both arrays are
(Nc, Nc, Nz) per panel.

The raw values from the file are "pressure-weighted accumulated" (Pa m²).
Use `convert_massflux_to_kgs!` to convert to kg/s.

$(FIELDS)
"""
struct GeosFPCubedSphereTimestep{FT}
    "X-direction mass flux per panel, NTuple{6, Array{FT,3}}, each (Nc, Nc, Nz)"
    mfxc :: NTuple{6, Array{FT, 3}}
    "Y-direction mass flux per panel, NTuple{6, Array{FT,3}}, each (Nc, Nc, Nz)"
    mfyc :: NTuple{6, Array{FT, 3}}
    "Pressure thickness per panel, NTuple{6, Array{FT,3}}, each (Nc, Nc, Nz)"
    delp :: NTuple{6, Array{FT, 3}}
    "Surface pressure per panel, NTuple{6, Array{FT,2}}, each (Nc, Nc)"
    ps :: NTuple{6, Array{FT, 2}}
    "Timestamp"
    time :: DateTime
    "Cells per panel edge"
    Nc :: Int
    "Number of vertical levels"
    Nz :: Int
    "Accumulation interval in seconds (for unit conversion)"
    dt_met :: FT
end

const WASHU_BASE_URL = "http://geoschemdata.wustl.edu/ExtData/GEOS_C720"

"""
    geosfp_cs_tavg_url(date::Date, hour::Int; collection="GEOS_FP_Native")

Build URL for a GEOS-FP C720 cubed-sphere file from the Washington University
archive (geoschemdata.wustl.edu).

The `tavg_1hr_ctm` collection contains MFXC, MFYC, DELP, PS, CX, CY.
Timestamps are at HH30 (center of hour-long averaging window).
"""
function geosfp_cs_tavg_url(date::Date, hour::Int;
                            collection::String = "GEOS_FP_Native")
    datestr = Dates.format(date, "yyyymmdd")
    timestamp = lpad(hour, 2, '0') * "30"
    y = Dates.year(date)
    m = lpad(Dates.month(date), 2, '0')
    d = lpad(Dates.day(date), 2, '0')
    return "$WASHU_BASE_URL/$collection/Y$y/M$m/D$d/" *
           "GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.$(datestr)_$(timestamp).V01.nc4"
end

geosfp_cs_url(date::Date, hour::Int; kwargs...) = geosfp_cs_tavg_url(date, hour; kwargs...)
geosfp_cs_asm_url(date::Date, hour::Int; kwargs...) = geosfp_cs_tavg_url(date, hour; kwargs...)

"""
    geosfp_cs_local_path(date::Date, hour::Int;
                         base_dir=joinpath(homedir(), "data", "geosfp_cs"))

Build the local file path for a downloaded GEOS-FP cubed-sphere file.
"""
function geosfp_cs_local_path(date::Date, hour::Int;
                              base_dir::String = joinpath(homedir(), "data", "geosfp_cs"))
    datestr = Dates.format(date, "yyyymmdd")
    timestamp = lpad(hour, 2, '0') * "30"
    fname = "GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.$(datestr)_$(timestamp).V01.nc4"
    return joinpath(base_dir, datestr, fname)
end

"""
$(SIGNATURES)

Read one timestep of GEOS-FP cubed-sphere data from a local NetCDF file.

The file should be from the `tavg_1hr_ctm_c0720_v72` collection.
All variables (MFXC, MFYC, DELP, PS) are read from the same file.

Raw mass fluxes are in Pa⋅m² (pressure-weighted accumulated).
Set `convert_to_kgs=true` to automatically convert to kg/s.

# Arguments
- `filepath`: Path to the NetCDF4 file
- `FT`: Float type (default Float32, matching file storage)
- `time_index`: Time index within the file (default 1)
- `dt_met`: Accumulation interval in seconds (default 3600.0 for 1-hour)
- `convert_to_kgs`: If true, convert mass fluxes from Pa⋅m² to kg/s
"""
function read_geosfp_cs_timestep(filepath::String;
                                  FT::Type{<:AbstractFloat} = Float32,
                                  time_index::Int = 1,
                                  dt_met::Real = 3600.0,
                                  convert_to_kgs::Bool = false)
    ds = NCDataset(filepath, "r")
    try
        # Dimension order in file: (Xdim, Ydim, nf, lev, time)
        mfxc_raw = Array{FT}(ds["MFXC"][:, :, :, :, time_index])  # (720,720,6,72)
        mfyc_raw = Array{FT}(ds["MFYC"][:, :, :, :, time_index])
        delp_raw = Array{FT}(ds["DELP"][:, :, :, :, time_index])
        ps_raw   = Array{FT}(ds["PS"][:, :, :, time_index])       # (720,720,6)

        Nc = size(mfxc_raw, 1)
        Nz = size(mfxc_raw, 4)
        @assert size(mfxc_raw) == (Nc, Nc, 6, Nz)
        @assert size(mfyc_raw) == (Nc, Nc, 6, Nz)
        @assert size(delp_raw) == (Nc, Nc, 6, Nz)
        @assert size(ps_raw) == (Nc, Nc, 6)

        conv = convert_to_kgs ? FT(1 / (GRAV_GEOSFP * dt_met)) : FT(1)

        # Split into per-panel arrays: (Nc, Nc, Nz)
        mfxc_panels = ntuple(6) do p
            arr = mfxc_raw[:, :, p, :]
            conv != 1 && (arr .*= conv)
            arr
        end
        mfyc_panels = ntuple(6) do p
            arr = mfyc_raw[:, :, p, :]
            conv != 1 && (arr .*= conv)
            arr
        end
        delp_panels = ntuple(6) do p
            delp_raw[:, :, p, :]
        end
        ps_panels = ntuple(6) do p
            ps_raw[:, :, p]
        end

        t = try
            ds["time"][time_index]
        catch
            DateTime(2000, 1, 1)
        end

        return GeosFPCubedSphereTimestep{FT}(
            mfxc_panels, mfyc_panels, delp_panels, ps_panels,
            t, Nc, Nz, FT(dt_met)
        )
    finally
        close(ds)
    end
end

"""
$(SIGNATURES)

Convert a `GeosFPCubedSphereTimestep` to haloed panel arrays suitable for
the cubed-sphere mass-flux advection kernels.

Returns `(delp_haloed, mfxc, mfyc)` where:
- `delp_haloed`: NTuple{6, Array{FT,3}} with halo padding (Nc+2Hp × Nc+2Hp × Nz)
- `mfxc`, `mfyc`: unchanged interior-only flux arrays
"""
function to_haloed_panels(ts::GeosFPCubedSphereTimestep{FT}; Hp::Int = 3) where FT
    Nc, Nz = ts.Nc, ts.Nz

    delp_haloed = ntuple(6) do p
        arr = zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)
        arr[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :] .= ts.delp[p]
        arr
    end

    return delp_haloed, ts.mfxc, ts.mfyc
end

"""
    cgrid_to_staggered_panels(mfxc_panels, mfyc_panels, Nc, Nz)

Convert C-grid mass fluxes (where MFXC(i,j) = flux through east face of cell (i,j))
to staggered arrays compatible with the advection kernels:
  - `am`: (Nc+1, Nc, Nz) — flux through X-faces (am(i,j) = flux into cell i from left)
  - `bm`: (Nc, Nc+1, Nz) — flux through Y-faces (bm(i,j) = flux into cell j from below)

Panel boundary fluxes (am[1,:,:] and bm[:,1,:]) are extracted from the east/north
face of the adjacent panel using the standard cubed-sphere gnomonic connectivity.

The standard connectivity for GEOS cubed-sphere panels (1-indexed):
  Panel 1: west=5, south=3, east=2, north=6
  Panel 2: west=1, south=3, east=4, north=6
  Panel 3: west=1, south=5, east=4, north=2
  Panel 4: west=3, south=5, east=6, north=2
  Panel 5: west=3, south=1, east=6, north=4
  Panel 6: west=5, south=1, east=2, north=4
"""
function cgrid_to_staggered_panels(mfxc_panels::NTuple{6, Array{FT,3}},
                                    mfyc_panels::NTuple{6, Array{FT,3}}) where FT
    Nc = size(mfxc_panels[1], 1)
    Nz = size(mfxc_panels[1], 3)

    # Connectivity: west_neighbor[p] gives the panel whose east edge abuts p's west edge.
    # For GEOS gnomonic CS (from file contacts = [west, south, east, north]):
    west_neighbor = (5, 1, 1, 3, 3, 5)
    south_neighbor = (3, 3, 5, 5, 1, 1)

    am_panels = ntuple(6) do p
        am = zeros(FT, Nc + 1, Nc, Nz)
        mfxc = mfxc_panels[p]

        @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
            am[i + 1, j, k] = mfxc[i, j, k]
        end

        # West boundary: flux from the east edge of the western neighbor
        wp = west_neighbor[p]
        mfxc_w = mfxc_panels[wp]
        @inbounds for k in 1:Nz, j in 1:Nc
            am[1, j, k] = mfxc_w[Nc, j, k]
        end

        am
    end

    bm_panels = ntuple(6) do p
        bm = zeros(FT, Nc, Nc + 1, Nz)
        mfyc = mfyc_panels[p]

        @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
            bm[i, j + 1, k] = mfyc[i, j, k]
        end

        sp = south_neighbor[p]
        mfyc_s = mfyc_panels[sp]
        @inbounds for k in 1:Nz, i in 1:Nc
            bm[i, 1, k] = mfyc_s[i, Nc, k]
        end

        bm
    end

    return am_panels, bm_panels
end

"""
    inspect_geosfp_cs_file(filepath::String)

Print diagnostic information about a GEOS-FP cubed-sphere NetCDF file.
"""
function inspect_geosfp_cs_file(filepath::String)
    ds = NCDataset(filepath, "r")
    try
        println("File: $(basename(filepath))")
        println("\nDimensions:")
        for (name, dim) in ds.dim
            println("  $name = $(length(dim))")
        end
        println("\nVariables:")
        for (name, var) in ds
            attrs = String[]
            haskey(var.attrib, "long_name") && push!(attrs, var.attrib["long_name"])
            haskey(var.attrib, "units") && push!(attrs, "[$(var.attrib["units"])]")
            println("  $name: $(dimnames(var)) → $(size(var))  $(join(attrs, " "))")
        end
    finally
        close(ds)
    end
end

"""
    read_geosfp_cs_grid_info(filepath::String)

Read the cubed-sphere grid coordinates (center lons/lats and corner lons/lats)
from a GEOS-FP file. Returns `(lons, lats, corner_lons, corner_lats)`.
"""
function read_geosfp_cs_grid_info(filepath::String)
    ds = NCDataset(filepath, "r")
    try
        lons = Array{Float64}(ds["lons"][:, :, :])         # (720,720,6)
        lats = Array{Float64}(ds["lats"][:, :, :])
        clons = Array{Float64}(ds["corner_lons"][:, :, :]) # (721,721,6)
        clats = Array{Float64}(ds["corner_lats"][:, :, :])
        return lons, lats, clons, clats
    finally
        close(ds)
    end
end
