#!/usr/bin/env julia
# ===========================================================================
# Convert preprocessed CS surface NetCDF → flat binary for fast mmap reading
#
# Input:  GEOSFP_CS720.YYYYMMDD.A1.nc       (PBLH, USTAR, HFLUX, T2M)
#         GEOSFP_CS720.YYYYMMDD.A3mstE.nc   (CMFMC)
# Output: GEOSFP_CS720.YYYYMMDD.A1.bin
#         GEOSFP_CS720.YYYYMMDD.A3mstE.bin
#
# Binary layout:
#   [8192-byte JSON header]
#   [timestep 1 data] [timestep 2 data] ... [timestep Nt data]
#
# A1 per-timestep layout:
#   PBLH_p1..p6 USTAR_p1..p6 HFLUX_p1..p6 T2M_p1..p6
#   Each panel: Nc×Nc Float32
#
# A3mstE per-timestep layout:
#   CMFMC_p1..p6
#   Each panel: Nc×Nc×Nz_edge Float32
#
# Usage:
#   julia --project=. scripts/convert_surface_cs_to_binary.jl <input_dir> <output_dir> [start_date] [end_date]
# ===========================================================================

using NCDatasets
using JSON3
using Dates
using Printf

const HEADER_SIZE = 8192
const FT = Float32

function convert_a1(nc_path::String, bin_path::String)
    ds = NCDataset(nc_path, "r")
    Nc = Int(ds.dim["Xdim"])
    Nt = Int(ds.dim["time"])
    vars = ["PBLH", "USTAR", "HFLUX", "T2M"]
    n_vars = length(vars)

    header = Dict{String,Any}(
        "magic"        => "SFC1",
        "version"      => 1,
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
                raw = FT.(ds[varname][:, :, :, t])  # (Nc, Nc, 6)
                for p in 1:6
                    copyto!(panel_buf, 1, view(raw, :, :, p), 1, Nc * Nc)
                    write(io, panel_buf)
                end
            end
        end
    end
    close(ds)
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
            raw = FT.(ds["CMFMC"][:, :, :, :, t])  # (Nc, Nc, 6, Nz_edge)
            for p in 1:6
                copyto!(panel_buf, 1, view(raw, :, :, p, :), 1, Nc * Nc * Nz_edge)
                write(io, panel_buf)
            end
        end
    end
    close(ds)
    return filesize(bin_path)
end

function main()
    input_dir  = length(ARGS) >= 1 ? ARGS[1] : "/temp1/atmos_transport/geosfp_c720/surface_fields_cs"
    output_dir = length(ARGS) >= 2 ? ARGS[2] : "/temp1/atmos_transport/geosfp_c720/surface_fields_bin"
    start_date = length(ARGS) >= 3 ? Date(ARGS[3]) : Date(2024, 6, 1)
    end_date   = length(ARGS) >= 4 ? Date(ARGS[4]) : Date(2024, 6, 30)

    mkpath(output_dir)
    @info "Converting CS surface NetCDF → flat binary"
    @info "  Input:  $input_dir"
    @info "  Output: $output_dir"
    @info "  Dates:  $start_date to $end_date"

    for date in start_date:Day(1):end_date
        datestr = Dates.format(date, "yyyymmdd")

        # Convert A1
        a1_nc = joinpath(input_dir, "GEOSFP_CS720.$(datestr).A1.nc")
        a1_bin = joinpath(output_dir, "GEOSFP_CS720.$(datestr).A1.bin")
        if isfile(a1_nc)
            t0 = time()
            sz = convert_a1(a1_nc, a1_bin)
            @info @sprintf("  [%s] A1: %.1f MB in %.1fs", datestr, sz / 1e6, time() - t0)
        else
            @warn "  [$datestr] A1 NetCDF not found: $a1_nc"
        end

        # Convert A3mstE
        a3_nc = joinpath(input_dir, "GEOSFP_CS720.$(datestr).A3mstE.nc")
        a3_bin = joinpath(output_dir, "GEOSFP_CS720.$(datestr).A3mstE.bin")
        if isfile(a3_nc)
            t0 = time()
            sz = convert_a3mste(a3_nc, a3_bin)
            @info @sprintf("  [%s] A3mstE: %.1f MB in %.1fs", datestr, sz / 1e6, time() - t0)
        else
            @warn "  [$datestr] A3mstE NetCDF not found: $a3_nc"
        end
    end

    total = sum(filesize(joinpath(output_dir, f)) for f in readdir(output_dir) if endswith(f, ".bin"); init=0)
    @info @sprintf("Done. Total binary size: %.1f GB", total / 1e9)
end

main()
