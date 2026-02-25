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
Registry of GEOS cubed-sphere products available from the WashU archive.

Two layout styles:
- `:hourly` — one file per hour, organized in `Y<year>/M<mm>/D<dd>/` subdirs (GEOS-FP)
- `:daily`  — one file per day with 24 timesteps, organized in `YYYY/MM/` subdirs (GEOS-IT)
"""
const GEOS_CS_PRODUCTS = Dict(
    "geosfp_c720" => (base_url   = "http://geoschemdata.wustl.edu/ExtData/GEOS_C720",
                      collection = "GEOS_FP_Native",
                      Nc         = 720,
                      layout     = :hourly,
                      est_gb     = 2.7),    # per hourly file
    "geosit_c180" => (base_url   = "http://geoschemdata.wustl.edu/ExtData/GEOS_C180",
                      collection = "GEOS_IT",
                      Nc         = 180,
                      layout     = :daily,
                      est_gb     = 4.2),    # per daily file
)

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

"""
    _geosfp_filename(date::Date, hour::Int, product::String)

Build the remote filename for a GEOS cubed-sphere file.
For `:hourly` products (GEOS-FP): one file per hour, e.g.
  `GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20240601_0030.V01.nc4`
For `:daily` products (GEOS-IT): one file per day (24 timesteps), e.g.
  `GEOSIT.20230601.CTM_A1.C180.nc`
"""
function _geosfp_filename(date::Date, product::String; hour::Int = -1)
    info = GEOS_CS_PRODUCTS[product]
    datestr = Dates.format(date, "yyyymmdd")
    if info.layout === :hourly
        timestamp = lpad(hour, 2, '0') * "30"
        return "GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.$(datestr)_$(timestamp).V01.nc4"
    else  # :daily
        return "GEOSIT.$(datestr).CTM_A1.C$(info.Nc).nc"
    end
end

"""
    geosfp_cs_tavg_url(date::Date, hour::Int; product="geosfp_c720")

Build URL for a GEOS cubed-sphere file from the Washington University archive.

`product` selects the data source — see `GEOS_CS_PRODUCTS` for available keys
(e.g. `"geosfp_c720"`, `"geosit_c180"`).
For hourly products, `hour` selects which hourly file. For daily products,
`hour` is ignored (the URL points to a whole-day file).
"""
function geosfp_cs_tavg_url(date::Date, hour::Int;
                            product::String = "geosfp_c720")
    info = GEOS_CS_PRODUCTS[product]
    fname = _geosfp_filename(date, product; hour)
    y = Dates.year(date)
    m = lpad(Dates.month(date), 2, '0')
    d = lpad(Dates.day(date), 2, '0')
    if info.layout === :hourly
        return "$(info.base_url)/$(info.collection)/Y$y/M$m/D$d/$fname"
    else  # :daily — directory is YYYY/MM/
        return "$(info.base_url)/$(info.collection)/$y/$m/$fname"
    end
end

geosfp_cs_url(date::Date, hour::Int; kwargs...) = geosfp_cs_tavg_url(date, hour; kwargs...)
geosfp_cs_asm_url(date::Date, hour::Int; kwargs...) = geosfp_cs_tavg_url(date, hour; kwargs...)

"""
    geosfp_cs_local_path(date::Date; hour::Int=-1,
                         base_dir=joinpath(homedir(), "data", "geosfp_cs"),
                         product="geosfp_c720")

Build the local file path for a downloaded GEOS cubed-sphere file.
"""
function geosfp_cs_local_path(date::Date; hour::Int = -1,
                              base_dir::String = joinpath(homedir(), "data", "geosfp_cs"),
                              product::String = "geosfp_c720")
    fname = _geosfp_filename(date, product; hour)
    datestr = Dates.format(date, "yyyymmdd")
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

        # Auto-detect inverted vertical ordering: if DELP at level 1 >> level Nz,
        # the file stores levels bottom-to-top (e.g. GEOS-IT) instead of the
        # standard top-to-bottom (e.g. GEOS-FP). Flip to ensure level 1 = TOA.
        mid = div(Nc, 2)
        delp_top = FT(delp_raw[mid, mid, 1, 1])
        delp_bot = FT(delp_raw[mid, mid, 1, Nz])
        need_flip = delp_top > FT(10) * delp_bot  # surface DELP is ~1000× TOA DELP
        if need_flip
            @info "Detected inverted vertical ordering — flipping to TOA-first" maxlog=1
            reverse!(mfxc_raw, dims=4)
            reverse!(mfyc_raw, dims=4)
            reverse!(delp_raw, dims=4)
        end

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
    cgrid_to_staggered_panels(mfxc_panels, mfyc_panels)

Convert C-grid mass fluxes (where MFXC(i,j) = flux through east face of cell (i,j),
MFYC(i,j) = flux through north face of cell (i,j)) to staggered arrays compatible
with the advection kernels:
  - `am`: (Nc+1, Nc, Nz) — flux through X-faces (am(i,j) = flux into cell i from left)
  - `bm`: (Nc, Nc+1, Nz) — flux through Y-faces (bm(i,j) = flux into cell j from below)

Panel boundary fluxes (am[1,:,:] and bm[:,1,:]) are extracted from the adjacent
panel's shared edge, using the correct flux variable (MFXC for east/west edges,
MFYC for north/south edges) and applying the along-edge orientation flip.

The connectivity matches the GEOS-FP native cubed-sphere file convention
(nf=1..6), verified from corner coordinate data.

West boundary connectivity: am[1, j, k] reads from neighbor's outgoing flux.
  Panels 2, 4, 6: neighbor's east edge (highX) → read MFXC[q](Nc, j, k), aligned
  Panels 1, 3, 5: neighbor's north edge (highY) → read MFYC[q](Nc+1-j, Nc, k), reversed

South boundary connectivity: bm[i, 1, k] reads from neighbor's outgoing flux.
  Panels 1, 3, 5: neighbor's north edge (highY) → read MFYC[q](i, Nc, k), aligned
  Panels 2, 4, 6: neighbor's east edge (highX) → read MFXC[q](Nc, Nc+1-i, k), reversed
"""
function cgrid_to_staggered_panels(mfxc_panels::NTuple{6, Array{FT,3}},
                                    mfyc_panels::NTuple{6, Array{FT,3}}) where FT
    Nc = size(mfxc_panels[1], 1)
    Nz = size(mfxc_panels[1], 3)

    # West neighbor panel and which edge connects (from GEOS-FP file contacts):
    #   Panel p west → Panel q at edge_type
    #   :east  = neighbor's east edge (read MFXC[q](Nc, j, k)), aligned
    #   :north = neighbor's north edge (read MFYC[q](Nc+1-j, Nc, k)), reversed
    west_info = (
        (panel=5, edge=:north),  # P1 west → P5 north, reversed
        (panel=1, edge=:east),   # P2 west → P1 east, aligned
        (panel=1, edge=:north),  # P3 west → P1 north, reversed
        (panel=3, edge=:east),   # P4 west → P3 east, aligned
        (panel=3, edge=:north),  # P5 west → P3 north, reversed
        (panel=5, edge=:east),   # P6 west → P5 east, aligned
    )

    # South neighbor panel and which edge connects:
    #   :north = neighbor's north edge (read MFYC[q](i, Nc, k)), aligned
    #   :east  = neighbor's east edge (read MFXC[q](Nc, Nc+1-i, k)), reversed
    south_info = (
        (panel=6, edge=:north),  # P1 south → P6 north, aligned
        (panel=6, edge=:east),   # P2 south → P6 east, reversed
        (panel=2, edge=:north),  # P3 south → P2 north, aligned
        (panel=2, edge=:east),   # P4 south → P2 east, reversed
        (panel=4, edge=:north),  # P5 south → P4 north, aligned
        (panel=4, edge=:east),   # P6 south → P4 east, reversed
    )

    am_panels = ntuple(6) do p
        am = zeros(FT, Nc + 1, Nc, Nz)
        mfxc = mfxc_panels[p]

        # Interior: shift C-grid to staggered
        @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
            am[i + 1, j, k] = mfxc[i, j, k]
        end

        # West boundary: flux from neighbor's shared edge
        wi = west_info[p]
        if wi.edge === :east
            # Neighbor's east (highX) edge, aligned: MFXC[q](Nc, j, k)
            mfxc_q = mfxc_panels[wi.panel]
            @inbounds for k in 1:Nz, j in 1:Nc
                am[1, j, k] = mfxc_q[Nc, j, k]
            end
        else  # :north
            # Neighbor's north (highY) edge, reversed: MFYC[q](Nc+1-j, Nc, k)
            mfyc_q = mfyc_panels[wi.panel]
            @inbounds for k in 1:Nz, j in 1:Nc
                am[1, j, k] = mfyc_q[Nc + 1 - j, Nc, k]
            end
        end

        am
    end

    bm_panels = ntuple(6) do p
        bm = zeros(FT, Nc, Nc + 1, Nz)
        mfyc = mfyc_panels[p]

        # Interior: shift C-grid to staggered
        @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
            bm[i, j + 1, k] = mfyc[i, j, k]
        end

        # South boundary: flux from neighbor's shared edge
        si = south_info[p]
        if si.edge === :north
            # Neighbor's north (highY) edge, aligned: MFYC[q](i, Nc, k)
            mfyc_q = mfyc_panels[si.panel]
            @inbounds for k in 1:Nz, i in 1:Nc
                bm[i, 1, k] = mfyc_q[i, Nc, k]
            end
        else  # :east
            # Neighbor's east (highX) edge, reversed: MFXC[q](Nc, Nc+1-i, k)
            mfxc_q = mfxc_panels[si.panel]
            @inbounds for k in 1:Nz, i in 1:Nc
                bm[i, 1, k] = mfxc_q[Nc, Nc + 1 - i, k]
            end
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
from a GEOS cubed-sphere file. Returns `(lons, lats, corner_lons, corner_lats)`.

GEOS-FP C720 files have all four; GEOS-IT C180 files only have center
coordinates, so `corner_lons` and `corner_lats` may be `nothing`.
"""
function read_geosfp_cs_grid_info(filepath::String)
    ds = NCDataset(filepath, "r")
    try
        lons = Array{Float64}(ds["lons"][:, :, :])
        lats = Array{Float64}(ds["lats"][:, :, :])
        clons = haskey(ds, "corner_lons") ? Array{Float64}(ds["corner_lons"][:, :, :]) : nothing
        clats = haskey(ds, "corner_lats") ? Array{Float64}(ds["corner_lats"][:, :, :]) : nothing
        return lons, lats, clons, clats
    finally
        close(ds)
    end
end
