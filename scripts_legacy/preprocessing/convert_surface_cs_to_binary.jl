#!/usr/bin/env julia
# ===========================================================================
# Convert CS surface NetCDF → flat binary for fast mmap reading
#
# Supports any GEOS cubed-sphere product (GEOS-FP C720, GEOS-IT C180, etc.)
# via TOML configuration. File naming and directory layout are product-aware.
#
# Binary layout:
#   [8192-byte JSON header]
#   [timestep 1 data] [timestep 2 data] ... [timestep Nt data]
#
# A1 per-timestep:  PBLH_p1..p6 USTAR_p1..p6 HFLUX_p1..p6 T2M_p1..p6
# A3mstE per-step:  CMFMC_p1..p6   (each panel Nc×Nc×Nz_edge)
#
# Usage:
#   julia --project=. scripts/convert_surface_cs_to_binary.jl <config.toml>
#
# TOML config:
#   [product]
#   name = "geosit_c180"   # key in GEOS_CS_PRODUCTS
#
#   [input]
#   data_dir   = "~/data/geosit_c180_catrine"
#   start_date = "2021-12-01"
#   end_date   = "2022-01-31"
#
#   [output]
#   directory = "/temp2/catrine-runs/met/geosit_c180/surface_bin"
# ===========================================================================

using NCDatasets
using JSON3
using Dates
using Printf
using TOML

const HEADER_SIZE = 8192
const FT = Float32

# ---------------------------------------------------------------------------
# Product-aware file discovery
# ---------------------------------------------------------------------------

"""Product info needed for file discovery and naming."""
struct SurfaceProduct
    name     :: String
    Nc       :: Int
    layout   :: Symbol   # :hourly (GEOS-FP flat dir) or :daily (GEOS-IT date subdirs)
end

"""Registry of known products — mirrors GEOS_CS_PRODUCTS in the main code."""
const SURFACE_PRODUCTS = Dict(
    "geosfp_c720" => SurfaceProduct("geosfp_c720", 720, :hourly),
    "geosit_c180" => SurfaceProduct("geosit_c180", 180, :daily),
)

"""
    find_surface_nc(product, datadir, date, collection) → String

Find the NetCDF surface file for a given date and collection (A1 or A3mstE).
Returns empty string if not found.

GEOS-FP pre-regridded: `datadir/GEOSFP_CS720.YYYYMMDD.<collection>.nc`
GEOS-IT native:         `datadir/YYYYMMDD/GEOSIT.YYYYMMDD.<collection>.C<Nc>.nc`
"""
function find_surface_nc(prod::SurfaceProduct, datadir::String,
                          date::Date, collection::String)
    datestr = Dates.format(date, "yyyymmdd")

    if prod.layout === :hourly
        # GEOS-FP: flat directory with pre-regridded CS files
        path = joinpath(datadir, "GEOSFP_CS$(prod.Nc).$(datestr).$(collection).nc")
        return isfile(path) ? path : ""
    else
        # GEOS-IT: date subdirectory with native files
        daydir = joinpath(datadir, datestr)
        isdir(daydir) || return ""
        # Try GEOSIT naming first, then GEOSFP naming
        for prefix in ["GEOSIT", "GEOSFP"]
            for suffix in [".C$(prod.Nc).nc", ".nc"]
                path = joinpath(daydir, "$(prefix).$(datestr).$(collection)$(suffix)")
                isfile(path) && return path
            end
        end
        return ""
    end
end

"""
    surface_bin_name(product, date, collection) → String

Output binary filename, matching the naming convention in the run-time reader
(`_surface_bin_path` in geosfp_cs_met_driver.jl).
"""
function surface_bin_name(prod::SurfaceProduct, date::Date, collection::String)
    datestr = Dates.format(date, "yyyymmdd")
    return "GEOSFP_CS$(prod.Nc).$(datestr).$(collection).bin"
end

# ---------------------------------------------------------------------------
# Core converters (product-agnostic — works on any CS file with Xdim/Ydim/nf)
# ---------------------------------------------------------------------------

"""
    convert_a1(nc_path, bin_path; ctm_path="")

Convert A1 surface fields to flat binary. Includes:
  - From A1: PBLH, USTAR, HFLUX, T2M, TROPPT (if available)
  - From CTM_A1: PS (surface pressure, if ctm_path provided)

The `ctm_path` argument is the co-located CTM_A1 file for the same date.
PS is essential for output (avoids expensive column sum of DELP at runtime).
"""
function convert_a1(nc_path::String, bin_path::String; ctm_path::String="")
    ds = NCDataset(nc_path, "r")
    Nc = Int(ds.dim["Xdim"])
    Nt = Int(ds.dim["time"])

    # Build variable list: core vars + optional TROPPT + optional PS
    vars = ["PBLH", "USTAR", "HFLUX", "T2M"]
    if haskey(ds, "TROPPT")
        push!(vars, "TROPPT")
    end
    # PS comes from CTM_A1 (same Nt, same panel layout)
    ctm_ds = nothing
    if !isempty(ctm_path) && isfile(ctm_path)
        ctm_ds = NCDataset(ctm_path, "r")
        if haskey(ctm_ds, "PS")
            push!(vars, "PS")
        else
            close(ctm_ds); ctm_ds = nothing
        end
    end

    n_vars = length(vars)
    header = Dict{String,Any}(
        "magic"        => "SFC1",
        "version"      => 2,
        "collection"   => "A1",
        "Nc"           => Nc,
        "n_panels"     => 6,
        "Nt"           => Nt,
        "n_vars"       => n_vars,
        "var_names"    => vars,
        "float_type"   => "Float32",
        "float_bytes"  => 4,
        "header_bytes" => HEADER_SIZE,
        "panel_elems"  => Nc * Nc,
        "elems_per_timestep" => n_vars * 6 * Nc * Nc,
    )

    open(bin_path, "w") do io
        hdr_json = JSON3.write(header)
        hdr_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(hdr_buf, 1, Vector{UInt8}(hdr_json), 1, length(hdr_json))
        write(io, hdr_buf)

        panel_buf = Array{FT}(undef, Nc, Nc)
        for t in 1:Nt
            for varname in vars
                # Read from CTM_A1 for PS, from A1 for everything else
                src_ds = (varname == "PS" && ctm_ds !== nothing) ? ctm_ds : ds
                raw = FT.(coalesce.(src_ds[varname][:, :, :, t], FT(0)))  # (Xdim, Ydim, nf)
                for p in 1:6
                    copyto!(panel_buf, 1, view(raw, :, :, p), 1, Nc * Nc)
                    write(io, panel_buf)
                end
            end
        end
    end
    close(ds)
    ctm_ds !== nothing && close(ctm_ds)
    return filesize(bin_path)
end

function convert_a3mste(nc_path::String, bin_path::String)
    ds = NCDataset(nc_path, "r")
    Nc = Int(ds.dim["Xdim"])
    Nz_edge = Int(ds.dim["lev"])
    Nt = Int(ds.dim["time"])

    header = Dict{String,Any}(
        "magic"        => "SFC3",
        "version"      => 1,
        "collection"   => "A3mstE",
        "Nc"           => Nc,
        "n_panels"     => 6,
        "Nz_edge"      => Nz_edge,
        "Nt"           => Nt,
        "float_type"   => "Float32",
        "float_bytes"  => 4,
        "header_bytes" => HEADER_SIZE,
        "panel_elems"  => Nc * Nc * Nz_edge,
        "elems_per_timestep" => 6 * Nc * Nc * Nz_edge,
    )

    open(bin_path, "w") do io
        hdr_json = JSON3.write(header)
        hdr_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(hdr_buf, 1, Vector{UInt8}(hdr_json), 1, length(hdr_json))
        write(io, hdr_buf)

        panel_buf = Array{FT}(undef, Nc, Nc, Nz_edge)
        for t in 1:Nt
            raw = FT.(coalesce.(ds["CMFMC"][:, :, :, :, t], FT(0)))  # (Xdim, Ydim, nf, lev)
            for p in 1:6
                copyto!(panel_buf, 1, view(raw, :, :, p, :), 1, Nc * Nc * Nz_edge)
                write(io, panel_buf)
            end
        end
    end
    close(ds)
    return filesize(bin_path)
end

function convert_a3dyn(nc_path::String, bin_path::String)
    ds = NCDataset(nc_path, "r")
    Nc = Int(ds.dim["Xdim"])
    Nz = Int(ds.dim["lev"])
    Nt = Int(ds.dim["time"])

    # Auto-detect vertical ordering from DTRAIN data:
    # GEOS-IT (bottom-to-top): active values at small k (lower troposphere)
    # GEOS-FP (top-to-bottom): active values at large k (lower troposphere)
    dtrain_sample = FT.(ds["DTRAIN"][:, :, 1, :, 1])  # (Nc, Nc, Nz) panel 1, time 1
    mid = div(Nc, 2)
    q1 = div(Nz, 4)
    abs_lo = sum(abs(dtrain_sample[mid, mid, k]) for k in 1:q1)
    abs_hi = sum(abs(dtrain_sample[mid, mid, k]) for k in (Nz - q1 + 1):Nz)
    vertical_order = abs_lo > FT(10) * abs_hi ? "bottom_to_top" : "top_to_bottom"
    @info "  A3dyn vertical ordering: $vertical_order"

    header = Dict{String,Any}(
        "magic"        => "DYN3",
        "version"      => 1,
        "collection"   => "A3dyn",
        "Nc"           => Nc,
        "n_panels"     => 6,
        "Nz"           => Nz,
        "Nt"           => Nt,
        "float_type"   => "Float32",
        "float_bytes"  => 4,
        "header_bytes" => HEADER_SIZE,
        "panel_elems"  => Nc * Nc * Nz,
        "elems_per_timestep" => 6 * Nc * Nc * Nz,
        "vertical_order" => vertical_order,
    )

    open(bin_path, "w") do io
        hdr_json = JSON3.write(header)
        hdr_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(hdr_buf, 1, Vector{UInt8}(hdr_json), 1, length(hdr_json))
        write(io, hdr_buf)

        panel_buf = Array{FT}(undef, Nc, Nc, Nz)
        for t in 1:Nt
            raw = FT.(coalesce.(ds["DTRAIN"][:, :, :, :, t], FT(0)))  # (Xdim, Ydim, nf, lev)
            for p in 1:6
                copyto!(panel_buf, 1, view(raw, :, :, p, :), 1, Nc * Nc * Nz)
                write(io, panel_buf)
            end
        end
    end
    close(ds)
    return filesize(bin_path)
end

function convert_i3_qv(nc_path::String, bin_path::String)
    ds = NCDataset(nc_path, "r")
    Nc = Int(ds.dim["Xdim"])
    Nz = Int(ds.dim["lev"])
    Nt = Int(ds.dim["time"])

    # Auto-detect vertical ordering from QV data:
    # GEOS-IT (bottom-to-top): high QV at small k (surface = moist)
    # GEOS-FP (top-to-bottom): high QV at large k (surface = moist)
    qv_sample = FT.(coalesce.(ds["QV"][:, :, 1, :, 1], FT(0)))  # (Nc, Nc, Nz) panel 1
    mid = div(Nc, 2)
    qv_k1 = qv_sample[mid, mid, 1]
    qv_kN = qv_sample[mid, mid, Nz]
    vertical_order = qv_k1 > FT(10) * qv_kN ? "bottom_to_top" : "top_to_bottom"
    @info "  I3 (QV) vertical ordering: $vertical_order"

    header = Dict{String,Any}(
        "magic"        => "QV3D",
        "version"      => 1,
        "collection"   => "I3",
        "Nc"           => Nc,
        "n_panels"     => 6,
        "Nz"           => Nz,
        "Nt"           => Nt,
        "float_type"   => "Float32",
        "float_bytes"  => 4,
        "header_bytes" => HEADER_SIZE,
        "panel_elems"  => Nc * Nc * Nz,
        "elems_per_timestep" => 6 * Nc * Nc * Nz,
        "vertical_order" => vertical_order,
    )

    open(bin_path, "w") do io
        hdr_json = JSON3.write(header)
        hdr_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(hdr_buf, 1, Vector{UInt8}(hdr_json), 1, length(hdr_json))
        write(io, hdr_buf)

        panel_buf = Array{FT}(undef, Nc, Nc, Nz)
        for t in 1:Nt
            raw = FT.(coalesce.(ds["QV"][:, :, :, :, t], FT(0)))  # (Xdim, Ydim, nf, lev)
            for p in 1:6
                copyto!(panel_buf, 1, view(raw, :, :, p, :), 1, Nc * Nc * Nz)
                write(io, panel_buf)
            end
        end
    end
    close(ds)
    return filesize(bin_path)
end

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

function main()
    if isempty(ARGS)
        println(stderr, """
Usage: julia --project=. scripts/convert_surface_cs_to_binary.jl <config.toml>

TOML config example:
  [product]
  name = "geosit_c180"

  [input]
  data_dir   = "~/data/geosit_c180_catrine"
  start_date = "2021-12-01"
  end_date   = "2022-01-31"

  [output]
  directory = "/temp2/catrine-runs/met/geosit_c180/surface_bin"
""")
        exit(1)
    end

    config = TOML.parsefile(ARGS[1])
    product_name = config["product"]["name"]
    haskey(SURFACE_PRODUCTS, product_name) || error(
        "Unknown product '$product_name'. Known: $(join(keys(SURFACE_PRODUCTS), ", "))")
    prod = SURFACE_PRODUCTS[product_name]

    input_dir  = expanduser(config["input"]["data_dir"])
    start_date = Date(config["input"]["start_date"])
    end_date   = Date(config["input"]["end_date"])
    output_dir = expanduser(config["output"]["directory"])

    mkpath(output_dir)
    @info "Converting CS surface NetCDF → flat binary"
    @info "  Product:  $(prod.name) (C$(prod.Nc), layout=$(prod.layout))"
    @info "  Input:    $input_dir"
    @info "  Output:   $output_dir"
    @info "  Dates:    $start_date to $end_date"

    n_a1 = 0; n_a3 = 0; n_dyn = 0; n_i3 = 0; total_bytes = 0

    for date in start_date:Day(1):end_date
        datestr = Dates.format(date, "yyyymmdd")

        # Convert A1 (with PS from CTM_A1 if available)
        a1_nc = find_surface_nc(prod, input_dir, date, "A1")
        ctm_nc = find_surface_nc(prod, input_dir, date, "CTM_A1")
        if !isempty(a1_nc)
            a1_bin = joinpath(output_dir, surface_bin_name(prod, date, "A1"))
            t0 = time()
            sz = convert_a1(a1_nc, a1_bin; ctm_path=ctm_nc)
            extra = !isempty(ctm_nc) ? " +PS" : ""
            @info @sprintf("  [%s] A1%s: %.1f MB in %.1fs", datestr, extra, sz / 1e6, time() - t0)
            n_a1 += 1; total_bytes += sz
        else
            @warn "  [$datestr] A1 not found in $input_dir"
        end

        # Convert A3mstE
        a3_nc = find_surface_nc(prod, input_dir, date, "A3mstE")
        if !isempty(a3_nc)
            a3_bin = joinpath(output_dir, surface_bin_name(prod, date, "A3mstE"))
            t0 = time()
            sz = convert_a3mste(a3_nc, a3_bin)
            @info @sprintf("  [%s] A3mstE: %.1f MB in %.1fs", datestr, sz / 1e6, time() - t0)
            n_a3 += 1; total_bytes += sz
        else
            @warn "  [$datestr] A3mstE not found in $input_dir"
        end

        # Convert A3dyn (DTRAIN)
        dyn_nc = find_surface_nc(prod, input_dir, date, "A3dyn")
        if !isempty(dyn_nc)
            dyn_bin = joinpath(output_dir, surface_bin_name(prod, date, "A3dyn"))
            t0 = time()
            sz = convert_a3dyn(dyn_nc, dyn_bin)
            @info @sprintf("  [%s] A3dyn: %.1f MB in %.1fs", datestr, sz / 1e6, time() - t0)
            n_dyn += 1; total_bytes += sz
        else
            @warn "  [$datestr] A3dyn not found in $input_dir"
        end

        # Convert I3 (QV for dry mole fractions)
        i3_nc = find_surface_nc(prod, input_dir, date, "I3")
        if !isempty(i3_nc)
            i3_bin = joinpath(output_dir, surface_bin_name(prod, date, "I3"))
            t0 = time()
            sz = convert_i3_qv(i3_nc, i3_bin)
            @info @sprintf("  [%s] I3 (QV): %.1f MB in %.1fs", datestr, sz / 1e6, time() - t0)
            n_i3 += 1; total_bytes += sz
        end
    end

    @info @sprintf("Done. %d A1 + %d A3mstE + %d A3dyn + %d I3 files → %.1f GB total",
                   n_a1, n_a3, n_dyn, n_i3, total_bytes / 1e9)
end

main()
