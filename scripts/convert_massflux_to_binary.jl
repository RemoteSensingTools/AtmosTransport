#!/usr/bin/env julia
# ===========================================================================
# Convert preprocessed mass-flux NetCDF to flat binary for fast mmap I/O
#
# The NetCDF uses deflate compression and HDF5 chunking, which add CPU
# overhead on every read. This script writes a single flat binary file
# with contiguous per-window data that can be memory-mapped at zero cost.
#
# Usage:
#   julia --project=. scripts/convert_massflux_to_binary.jl [input.nc] [output.bin]
#
# Default input:  ~/data/metDrivers/era5/massflux_era5_202406_float32.nc
# Default output: <input_dir>/massflux_era5_202406_float32.bin
#
# Binary layout:
#   [Header]  — 512 bytes (JSON metadata, zero-padded)
#   [Window 1]: m(Nx*Ny*Nz) | am((Nx+1)*Ny*Nz) | bm(Nx*(Ny+1)*Nz) |
#               cm(Nx*Ny*(Nz+1)) | ps(Nx*Ny)     — all FT (Float32/64)
#   [Window 2]: same layout
#   ...
#   [Window Nt]
# ===========================================================================

using NCDatasets
using JSON3
using Printf

const HEADER_SIZE = 4096  # bytes, zero-padded JSON (needs room for lons/lats arrays)

function convert_massflux(nc_path::String, bin_path::String)
    @info "Reading metadata from $nc_path"
    ds = NCDataset(nc_path)

    lons = Float64.(ds["lon"][:])
    lats = Float64.(ds["lat"][:])
    Nx   = length(lons)
    Ny   = length(lats)
    Nz   = ds.dim["lev"]
    Nt   = ds.dim["time"]

    dt_seconds      = Float64(ds.attrib["dt_seconds"])
    half_dt_seconds = Float64(ds.attrib["half_dt_seconds"])
    steps_per_met   = Int(ds.attrib["steps_per_met_window"])
    level_top       = Int(ds.attrib["level_top"])
    level_bot       = Int(ds.attrib["level_bot"])

    ft_str   = get(ds.attrib, "float_type", "Float32")
    FT       = ft_str == "Float64" ? Float64 : Float32
    ft_bytes = sizeof(FT)

    n_m  = Nx * Ny * Nz
    n_am = (Nx + 1) * Ny * Nz
    n_bm = Nx * (Ny + 1) * Nz
    n_cm = Nx * Ny * (Nz + 1)
    n_ps = Nx * Ny
    elems_per_window = n_m + n_am + n_bm + n_cm + n_ps
    bytes_per_window = elems_per_window * ft_bytes

    @info @sprintf("  Grid: Nx=%d, Ny=%d, Nz=%d, Nt=%d, FT=%s", Nx, Ny, Nz, Nt, ft_str)
    @info @sprintf("  Elements/window: %d (%.1f MB)", elems_per_window,
                   bytes_per_window / 1e6)
    @info @sprintf("  Total binary: %.2f GB (+ %d B header)",
                   bytes_per_window * Nt / 1e9, HEADER_SIZE)

    header = Dict{String,Any}(
        "magic"               => "MFLX",
        "version"             => 1,
        "Nx"                  => Nx,
        "Ny"                  => Ny,
        "Nz"                  => Nz,
        "Nt"                  => Nt,
        "float_type"          => ft_str,
        "float_bytes"         => ft_bytes,
        "header_bytes"        => HEADER_SIZE,
        "window_bytes"        => bytes_per_window,
        "n_m"                 => n_m,
        "n_am"                => n_am,
        "n_bm"                => n_bm,
        "n_cm"                => n_cm,
        "n_ps"                => n_ps,
        "dt_seconds"          => dt_seconds,
        "half_dt_seconds"     => half_dt_seconds,
        "steps_per_met_window"=> steps_per_met,
        "level_top"           => level_top,
        "level_bot"           => level_bot,
        "lons"                => lons,
        "lats"                => lats,
    )
    header_json = JSON3.write(header)
    length(header_json) < HEADER_SIZE ||
        error("Header JSON too large ($(length(header_json)) >= $HEADER_SIZE); increase HEADER_SIZE")

    @info "Writing binary file: $bin_path"
    open(bin_path, "w") do io
        # Padded JSON header
        header_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(header_buf, 1, Vector{UInt8}(header_json), 1, length(header_json))
        write(io, header_buf)

        for win in 1:Nt
            t0 = time()
            m_data  = FT.(ds["m"][:, :, :, win])
            am_data = FT.(ds["am"][:, :, :, win])
            bm_data = FT.(ds["bm"][:, :, :, win])
            cm_data = FT.(ds["cm"][:, :, :, win])
            ps_data = FT.(ds["ps"][:, :, win])

            write(io, vec(m_data))
            write(io, vec(am_data))
            write(io, vec(bm_data))
            write(io, vec(cm_data))
            write(io, vec(ps_data))

            t_win = round(time() - t0, digits=2)
            if win <= 3 || win == Nt || win % 20 == 0
                @info @sprintf("  Window %d/%d  (%.2fs)", win, Nt, t_win)
            end
        end
    end

    close(ds)

    actual_size = filesize(bin_path)
    expected    = HEADER_SIZE + bytes_per_window * Nt
    @info @sprintf("Done: %s (%.2f GB)", bin_path, actual_size / 1e9)
    actual_size == expected ||
        @warn @sprintf("Size mismatch: expected %d, got %d", expected, actual_size)
end

# --- Entry point ---
nc_default = expanduser(get(ENV, "MASSFLUX_FILE",
    "~/data/metDrivers/era5/massflux_era5_202406_float32.nc"))
nc_path = length(ARGS) >= 1 ? ARGS[1] : nc_default
bin_path = if length(ARGS) >= 2
    ARGS[2]
else
    replace(nc_path, r"\.nc$" => ".bin")
end

isfile(nc_path) || error("Input NetCDF not found: $nc_path")
convert_massflux(nc_path, bin_path)
