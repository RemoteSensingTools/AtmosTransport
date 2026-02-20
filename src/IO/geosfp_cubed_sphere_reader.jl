# ---------------------------------------------------------------------------
# GEOS-FP Native Cubed-Sphere Data Reader
#
# Reads MFXC (eastward mass flux), MFYC (northward mass flux), and DELP
# (pressure thickness) from native GEOS-FP cubed-sphere NetCDF files.
#
# GEOS-FP cubed-sphere files use dimensions:
#   nf=6 (face/panel), Xdim=Nc, Ydim=Nc, lev=72
# Variables are stored as (Xdim, Ydim, lev, nf, time) or similar orderings.
#
# MFXC: eastward mass flux on X cell faces → (Nc+1, Nc, lev, nf)
# MFYC: northward mass flux on Y cell faces → (Nc, Nc+1, lev, nf)
# DELP: pressure thickness at cell centers → (Nc, Nc, lev, nf)
#
# The `tavg3_3d_mst_Cp` collection on NCCS contains MFXC/MFYC.
# The `inst3_3d_asm_Cp` or `tavg3_3d_asm_Cp` collection contains DELP.
# ---------------------------------------------------------------------------

using NCDatasets
using Dates

"""
$(TYPEDEF)

Container for one timestep of GEOS-FP cubed-sphere mass-flux data.

$(FIELDS)
"""
struct GeosFPCubedSphereTimestep{FT}
    "X-direction mass flux per panel, NTuple{6, Array{FT,3}}, each (Nc+1, Nc, Nz)"
    mfxc :: NTuple{6, Array{FT, 3}}
    "Y-direction mass flux per panel, NTuple{6, Array{FT,3}}, each (Nc, Nc+1, Nz)"
    mfyc :: NTuple{6, Array{FT, 3}}
    "Pressure thickness per panel, NTuple{6, Array{FT,3}}, each (Nc, Nc, Nz)"
    delp :: NTuple{6, Array{FT, 3}}
    "Timestamp"
    time :: DateTime
    "Cells per panel edge"
    Nc :: Int
    "Number of vertical levels"
    Nz :: Int
end

"""
    geosfp_cs_url(date::Date, hour::Int; stream="f5295_fp", collection="tavg3_3d_mst_Cp")

Build the NCCS download URL for a GEOS-FP cubed-sphere file.
"""
function geosfp_cs_url(date::Date, hour::Int;
                       stream::String = "f5295_fp",
                       collection::String = "tavg3_3d_mst_Cp")
    datestr = Dates.format(date, "yyyymmdd")
    timestr = lpad(hour, 2, '0') * "30"
    y = Dates.format(date, "yyyy")
    m = Dates.format(date, "mm")
    d = Dates.format(date, "dd")
    return "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/" *
           "Y$(y)/M$(m)/D$(d)/$(stream).$(collection).$(datestr)_$(timestr)z.nc4"
end

"""
    geosfp_cs_asm_url(date::Date, hour::Int)

Build URL for instantaneous assembly data (contains DELP) on cubed-sphere.
"""
function geosfp_cs_asm_url(date::Date, hour::Int;
                           stream::String = "f5295_fp",
                           collection::String = "inst3_3d_asm_Cp")
    datestr = Dates.format(date, "yyyymmdd")
    timestr = lpad(hour, 2, '0') * "00"
    y = Dates.format(date, "yyyy")
    m = Dates.format(date, "mm")
    d = Dates.format(date, "dd")
    return "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/" *
           "Y$(y)/M$(m)/D$(d)/$(stream).$(collection).$(datestr)_$(timestr)z.nc4"
end

"""
$(SIGNATURES)

Read one timestep of GEOS-FP cubed-sphere data from NetCDF file(s).

The `mst_file` should contain MFXC and MFYC (mass flux collection).
The `asm_file` should contain DELP (assembly collection), or `nothing` if
DELP is available in the same file as MFXC/MFYC.

Returns a `GeosFPCubedSphereTimestep`.
"""
function read_geosfp_cs_timestep(mst_file::String;
                                  asm_file::Union{String, Nothing} = nothing,
                                  FT::Type{<:AbstractFloat} = Float64,
                                  time_index::Int = 1)
    mfxc_panels, mfyc_panels, Nc, Nz = _read_mass_fluxes(mst_file, FT, time_index)

    if asm_file !== nothing
        delp_panels = _read_delp(asm_file, FT, time_index, Nc, Nz)
    else
        delp_panels = _read_delp(mst_file, FT, time_index, Nc, Nz)
    end

    ds = NCDataset(mst_file, "r")
    t = ds["time"][time_index]
    close(ds)

    return GeosFPCubedSphereTimestep{FT}(mfxc_panels, mfyc_panels, delp_panels,
                                          t, Nc, Nz)
end

"""
Read MFXC and MFYC from a cubed-sphere NetCDF file.

GEOS-FP cubed-sphere files store data with dimensions that vary by collection.
This function handles common orderings:
- (Xdim, Ydim, lev, nf)      — most common for cell-center variables
- (XCdim, Ydim, lev, nf)     — MFXC (staggered in X)
- (Xdim, YCdim, lev, nf)     — MFYC (staggered in Y)

The `nf` dimension indexes the 6 cubed-sphere panels.
"""
function _read_mass_fluxes(filepath::String, FT::Type, tidx::Int)
    ds = NCDataset(filepath, "r")
    try
        mfxc_raw = _read_cs_var(ds, "MFXC", FT, tidx)
        mfyc_raw = _read_cs_var(ds, "MFYC", FT, tidx)

        nf = 6
        nx_mfxc = size(mfxc_raw, 1)
        ny_mfxc = size(mfxc_raw, 2)
        nz      = size(mfxc_raw, 3)

        nx_mfyc = size(mfyc_raw, 1)
        ny_mfyc = size(mfyc_raw, 2)

        Nc = ny_mfxc
        @assert nx_mfxc == Nc + 1 "MFXC X-dim should be Nc+1=$(Nc+1), got $nx_mfxc"
        @assert nx_mfyc == Nc "MFYC X-dim should be Nc=$Nc, got $nx_mfyc"
        @assert ny_mfyc == Nc + 1 "MFYC Y-dim should be Nc+1=$(Nc+1), got $ny_mfyc"

        mfxc_panels = ntuple(nf) do p
            Array{FT}(mfxc_raw[:, :, :, p])
        end

        mfyc_panels = ntuple(nf) do p
            Array{FT}(mfyc_raw[:, :, :, p])
        end

        return mfxc_panels, mfyc_panels, Nc, nz
    finally
        close(ds)
    end
end

"""
Read DELP (pressure thickness) from a cubed-sphere NetCDF file.
"""
function _read_delp(filepath::String, FT::Type, tidx::Int, Nc::Int, Nz::Int)
    ds = NCDataset(filepath, "r")
    try
        delp_raw = _read_cs_var(ds, "DELP", FT, tidx)
        @assert size(delp_raw, 1) == Nc "DELP X-dim should be $Nc, got $(size(delp_raw, 1))"
        @assert size(delp_raw, 2) == Nc "DELP Y-dim should be $Nc, got $(size(delp_raw, 2))"
        @assert size(delp_raw, 3) == Nz "DELP lev-dim should be $Nz, got $(size(delp_raw, 3))"

        return ntuple(6) do p
            Array{FT}(delp_raw[:, :, :, p])
        end
    finally
        close(ds)
    end
end

"""
Read a 4D variable (X, Y, lev, nf) from a cubed-sphere NetCDF dataset,
handling different possible dimension orderings.
"""
function _read_cs_var(ds::NCDataset, varname::String, FT::Type, tidx::Int)
    var = ds[varname]
    dims = dimnames(var)
    ndim = length(dims)

    if ndim == 5
        raw = Array{FT}(var[:, :, :, :, tidx])
    elseif ndim == 4
        raw = Array{FT}(var[:, :, :, :])
    else
        error("Unexpected number of dimensions ($ndim) for variable $varname")
    end

    nf_idx = findfirst(d -> d in ("nf", "face", "nface", "tile"), dims)
    if nf_idx !== nothing && nf_idx != ndim && nf_idx != ndim - 1
        perm = _move_nf_to_last(nf_idx, ndim == 5 ? 4 : ndim)
        raw = permutedims(raw, perm)
    end

    return raw
end

function _move_nf_to_last(nf_idx::Int, ndims_notime::Int)
    perm = collect(1:ndims_notime)
    splice!(perm, nf_idx)
    push!(perm, nf_idx)
    return Tuple(perm)
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
