#!/usr/bin/env julia
# ===========================================================================
# Convert CTM_I1 NetCDF (hourly QV + PS) → flat binary for fast mmap reading
#
# CTM_I1 is the GCHP-standard hourly instantaneous thermodynamic collection.
# Contains QV (3D specific humidity) and PS (2D surface pressure) at 1-hour
# cadence — same as GCHP's SPHU1/SPHU2 and PS1/PS2 imports.
#
# Two output files per day:
#   GEOSFP_CS<Nc>.<date>.CTM_I1.bin — 3D QV (same format as I3 binary, 24 timesteps)
#   (PS is included in the A1 binary via convert_surface_cs_to_binary.jl)
#
# Binary layout for QV:
#   [8192-byte JSON header]
#   [timestep 1: QV_p1(Nc×Nc×Nz) QV_p2 ... QV_p6]
#   [timestep 2: ...]
#   ...
#   [timestep 24: ...]
#
# The runtime reader (_load_qv_from_bin!) already handles this format when
# the collection name is "CTM_I1" in _surface_bin_path().
#
# Usage:
#   julia --project=. scripts/preprocessing/convert_ctm_i1_to_binary.jl \
#     ~/data/geosit_c180_catrine /temp1/catrine/met/geosit_c180/surface_bin \
#     2021-12-01 2021-12-22
# ===========================================================================

using NCDatasets
using JSON3
using Dates
using Printf

const HEADER_SIZE = 8192
const FT = Float32

function convert_ctm_i1_day(nc_path::String, bin_path::String)
    ds = NCDataset(nc_path, "r")
    Nc = Int(ds.dim["Xdim"])
    Nz = Int(ds.dim["lev"])
    Nt = Int(ds.dim["time"])

    header = Dict{String,Any}(
        "magic"           => "QVBIN",
        "version"         => 1,
        "collection"      => "CTM_I1",
        "Nc"              => Nc,
        "Nz"              => Nz,
        "n_panels"        => 6,
        "Nt"              => Nt,
        "float_type"      => "Float32",
        "float_bytes"     => 4,
        "header_bytes"    => HEADER_SIZE,
        "panel_elems"     => Nc * Nc * Nz,
        "elems_per_timestep" => 6 * Nc * Nc * Nz,
        "vertical_order"  => "bottom_to_top",  # GEOS-IT native ordering
    )

    open(bin_path, "w") do io
        hdr_json = JSON3.write(header)
        hdr_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(hdr_buf, 1, Vector{UInt8}(hdr_json), 1, length(hdr_json))
        write(io, hdr_buf)

        panel_buf = Array{FT}(undef, Nc, Nc, Nz)
        for t in 1:Nt
            # Read QV for all panels at this timestep
            # NCDatasets dim order: (Xdim, Ydim, nf, lev, time) or (Xdim, Ydim, lev, nf, time)
            # Check actual ordering
            qv_raw = FT.(coalesce.(ds["QV"][:, :, :, :, t], FT(0)))
            # qv_raw is (Xdim=Nc, Ydim=Nc, nf=6, lev=Nz) — need to split by panel
            for p in 1:6
                for k in 1:Nz, j in 1:Nc, i in 1:Nc
                    panel_buf[i, j, k] = qv_raw[i, j, p, k]
                end
                write(io, panel_buf)
            end
        end
    end
    close(ds)

    sz_mb = round(filesize(bin_path) / 1e6; digits=1)
    return sz_mb
end

function main()
    if length(ARGS) < 4
        println("Usage: julia convert_ctm_i1_to_binary.jl <data_dir> <output_dir> <start_date> <end_date>")
        println("  data_dir:   ~/data/geosit_c180_catrine")
        println("  output_dir: /temp1/catrine/met/geosit_c180/surface_bin")
        println("  start_date: 2021-12-01")
        println("  end_date:   2021-12-22")
        return
    end

    data_dir = expanduser(ARGS[1])
    output_dir = expanduser(ARGS[2])
    start_date = Date(ARGS[3])
    end_date = Date(ARGS[4])

    mkpath(output_dir)
    dates = start_date:Day(1):end_date
    Nc = 180  # auto-detect from first file

    n_threads = Threads.nthreads()
    @info "Converting CTM_I1 NetCDF → binary" data_dir output_dir start=start_date stop=end_date n_days=length(dates) threads=n_threads

    n_done = Threads.Atomic{Int}(0)
    n_skip = Threads.Atomic{Int}(0)
    n_fail = Threads.Atomic{Int}(0)

    Threads.@threads for d in collect(dates)
        datestr = Dates.format(d, "yyyymmdd")
        nc_path = joinpath(data_dir, datestr, "GEOSIT.$(datestr).CTM_I1.C$(Nc).nc")
        bin_path = joinpath(output_dir, "GEOSFP_CS$(Nc).$(datestr).CTM_I1.bin")

        if isfile(bin_path) && filesize(bin_path) > 100_000_000
            Threads.atomic_add!(n_skip, 1)
        elseif !isfile(nc_path)
            @warn "  Missing: $(basename(nc_path))"
            Threads.atomic_add!(n_fail, 1)
        else
            try
                sz = convert_ctm_i1_day(nc_path, bin_path)
                Threads.atomic_add!(n_done, 1)
                @info @sprintf("  %s → %s (%.1f MB)", basename(nc_path), basename(bin_path), sz)
            catch e
                @warn "  Failed: $(basename(nc_path)) — $e"
                Threads.atomic_add!(n_fail, 1)
                isfile(bin_path) && rm(bin_path)
            end
        end
    end

    @info @sprintf("Done: %d converted, %d skipped, %d failed", n_done[], n_skip[], n_fail[])
end

main()
