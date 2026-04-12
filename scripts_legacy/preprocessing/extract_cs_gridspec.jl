#!/usr/bin/env julia
#
# Extract cubed-sphere grid specification (corners, centers, areas) from a
# GEOS-Chem CATRINE diagnostic file into a compact NetCDF for use in regridding.
#
# Usage:
#   julia scripts/preprocessing/extract_cs_gridspec.jl
#
# Input:  ~/data/AtmosTransport/catrine/geos-chem/GEOSChem.CATRINE_inst.20211201_0300z.nc4
# Output: data/grids/cs_c180_gridspec.nc (~6 MB)

using NCDatasets

gc_path = expanduser("~/data/AtmosTransport/catrine/geos-chem/GEOSChem.CATRINE_inst.20211201_0300z.nc4")
out_path = joinpath(@__DIR__, "..", "..", "data", "grids", "cs_c180_gridspec.nc")

@info "Reading from $gc_path"
ds = NCDataset(gc_path, "r")

# Read arrays — NCDatasets gives Julia-order (reversed dims)
# NetCDF (nf=6, YCdim=181, XCdim=181) → Julia (181, 181, 6)
corner_lons = Array(ds["corner_lons"])  # (181, 181, 6)
corner_lats = Array(ds["corner_lats"])  # (181, 181, 6)
lons = Array(ds["lons"])                # (180, 180, 6)
lats = Array(ds["lats"])                # (180, 180, 6)
areas = Array(ds["Met_AREAM2"])         # (180, 180, 6, 1) — has time dim

close(ds)

# Drop time dimension from areas
areas = dropdims(areas, dims=4)  # (180, 180, 6)

Nc = 180
Nf = 6

@info "Array shapes" size(corner_lons) size(lons) size(areas)
@info "Total area: $(sum(areas)) m² (expected 4πR² ≈ 5.1e14)"

# Write compact NetCDF
@info "Writing to $out_path"
mkpath(dirname(out_path))

NCDataset(out_path, "c") do ds_out
    # Dimensions
    defDim(ds_out, "Xdim", Nc)
    defDim(ds_out, "Ydim", Nc)
    defDim(ds_out, "XCdim", Nc + 1)
    defDim(ds_out, "YCdim", Nc + 1)
    defDim(ds_out, "nf", Nf)

    # Corner coordinates
    v = defVar(ds_out, "corner_lons", Float64, ("XCdim", "YCdim", "nf"))
    v.attrib["long_name"] = "cell corner longitudes"
    v.attrib["units"] = "degrees_east"
    v[:, :, :] = corner_lons

    v = defVar(ds_out, "corner_lats", Float64, ("XCdim", "YCdim", "nf"))
    v.attrib["long_name"] = "cell corner latitudes"
    v.attrib["units"] = "degrees_north"
    v[:, :, :] = corner_lats

    # Cell centers
    v = defVar(ds_out, "lons", Float64, ("Xdim", "Ydim", "nf"))
    v.attrib["long_name"] = "cell center longitudes"
    v.attrib["units"] = "degrees_east"
    v[:, :, :] = lons

    v = defVar(ds_out, "lats", Float64, ("Xdim", "Ydim", "nf"))
    v.attrib["long_name"] = "cell center latitudes"
    v.attrib["units"] = "degrees_north"
    v[:, :, :] = lats

    # Cell areas
    v = defVar(ds_out, "areas", Float64, ("Xdim", "Ydim", "nf"))
    v.attrib["long_name"] = "cell areas from Met_AREAM2"
    v.attrib["units"] = "m2"
    v[:, :, :] = areas

    # Global attributes
    ds_out.attrib["title"] = "Cubed-sphere C180 grid specification"
    ds_out.attrib["source"] = basename(gc_path)
    ds_out.attrib["Nc"] = Nc
    ds_out.attrib["history"] = "Extracted by extract_cs_gridspec.jl"
end

filesize_mb = filesize(out_path) / 1e6
@info "Done. Output: $out_path ($(round(filesize_mb, digits=1)) MB)"
