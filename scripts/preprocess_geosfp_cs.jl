#!/usr/bin/env julia
# ===========================================================================
# Offline preprocessing: GEOS cubed-sphere NetCDF → flat binary
#
# Reads native GEOS mass fluxes (MFXC, MFYC, DELP) from NetCDF, converts
# C-grid fluxes to staggered panels (am, bm), pads DELP with halos, and
# writes per-day flat binary files for mmap-based GPU ingestion.
#
# Supports two products:
#   geosfp_c720 — GEOS-FP C720, hourly files (24 .nc4 per day)
#   geosit_c180 — GEOS-IT C180, daily files (1 .nc with 24 timesteps)
#
# Binary layout per file:
#   [Header]  8192 bytes — JSON metadata (zero-padded)
#   [Window 1]:
#     delp_panel_1..6  — each (Nc+2Hp)×(Nc+2Hp)×Nz Float32
#     am_panel_1..6    — each (Nc+1)×Nc×Nz Float32
#     bm_panel_1..6    — each Nc×(Nc+1)×Nz Float32
#   [Window 2]: same layout
#   ...
#
# Usage:
#   julia --project=. scripts/preprocess_geosfp_cs.jl
#
# Environment variables:
#   PRODUCT         — "geosfp_c720" or "geosit_c180" (default: geosfp_c720)
#   GEOSFP_DATA_DIR — directory with date subdirs containing NetCDF files
#   OUTDIR          — output directory for binary files
#   GEOSFP_START    — start date (default: 2024-06-01)
#   GEOSFP_END      — end date (default: 2024-06-05)
#   FT_PRECISION    — "Float32" or "Float64" (default: Float32)
# ===========================================================================

using AtmosTransport
using AtmosTransport.IO: read_geosfp_cs_timestep, to_haloed_panels,
                              cgrid_to_staggered_panels
import AtmosTransport.IO: GEOS_CS_PRODUCTS
using NCDatasets
using Dates
using Printf
using JSON3

const FT_STR  = get(ENV, "FT_PRECISION", "Float32")
const FT      = FT_STR == "Float64" ? Float64 : Float32
const Hp      = 3
const PRODUCT = get(ENV, "PRODUCT", "geosfp_c720")

const GEOSFP_DIR = expanduser(get(ENV, "GEOSFP_DATA_DIR",
                       joinpath(homedir(), "data", "geosfp_cs")))
const OUTDIR = expanduser(get(ENV, "OUTDIR",
                   joinpath(homedir(), "data", "geosfp_cs", "preprocessed")))
const START_DATE = Date(get(ENV, "GEOSFP_START", "2024-06-01"))
const END_DATE   = Date(get(ENV, "GEOSFP_END", "2024-06-05"))

const HEADER_SIZE = 8192

# ---------------------------------------------------------------------------
# File discovery — dispatch on product layout
# ---------------------------------------------------------------------------

"""
Find input files for a given day. Returns (files, n_timesteps_per_file).
For hourly products: 24 separate files, 1 timestep each.
For daily products: 1 file, 24 timesteps.
"""
function find_files_for_day(datadir, date, product)
    info = GEOS_CS_PRODUCTS[product]
    daydir = joinpath(datadir, Dates.format(date, "yyyymmdd"))
    isdir(daydir) || return String[], 0

    if info.layout === :hourly
        files = String[]
        for f in sort(readdir(daydir))
            if contains(f, "tavg_1hr_ctm_c0720_v72") && endswith(f, ".nc4")
                push!(files, joinpath(daydir, f))
            end
        end
        return files, 1  # 1 timestep per file
    else  # :daily
        tag = "CTM_A1.C$(info.Nc)"
        for f in readdir(daydir)
            if contains(f, tag) && endswith(f, ".nc")
                filepath = joinpath(daydir, f)
                # Count timesteps in file
                nt = NCDataset(filepath, "r") do ds
                    length(ds["time"])
                end
                return [filepath], nt
            end
        end
        return String[], 0
    end
end

# ---------------------------------------------------------------------------
# Per-day preprocessing
# ---------------------------------------------------------------------------

function preprocess_day(date::Date, files::Vector{String},
                        n_timesteps_per_file::Int, outpath::String)
    isempty(files) && return

    # Total windows for this day
    Nt = length(files) * n_timesteps_per_file
    Nt == 0 && return

    @info "  Reading first file for dimensions..."
    ts0 = read_geosfp_cs_timestep(files[1]; FT, convert_to_kgs=true)
    Nc, Nz = ts0.Nc, ts0.Nz

    n_delp_panel = (Nc + 2Hp) * (Nc + 2Hp) * Nz
    n_am_panel   = (Nc + 1) * Nc * Nz
    n_bm_panel   = Nc * (Nc + 1) * Nz

    n_delp_total = 6 * n_delp_panel
    n_am_total   = 6 * n_am_panel
    n_bm_total   = 6 * n_bm_panel
    elems_per_window = n_delp_total + n_am_total + n_bm_total
    bytes_per_window = elems_per_window * sizeof(FT)

    @info @sprintf("  Grid: C%d, Nz=%d, Hp=%d, Nt=%d", Nc, Nz, Hp, Nt)
    @info @sprintf("  Per window: %.1f MB (%d elems)", bytes_per_window / 1e6, elems_per_window)
    @info @sprintf("  Total binary: %.2f GB", (HEADER_SIZE + bytes_per_window * Nt) / 1e9)

    header = Dict{String,Any}(
        "magic"            => "CSFLX",
        "version"          => 1,
        "grid_type"        => "cubed_sphere",
        "Nc"               => Nc,
        "Nz"               => Nz,
        "Hp"               => Hp,
        "Nt"               => Nt,
        "float_type"       => FT_STR,
        "float_bytes"      => sizeof(FT),
        "header_bytes"     => HEADER_SIZE,
        "window_bytes"     => bytes_per_window,
        "n_delp_panel"     => n_delp_panel,
        "n_am_panel"       => n_am_panel,
        "n_bm_panel"       => n_bm_panel,
        "n_panels"         => 6,
        "elems_per_window" => elems_per_window,
        "date"             => Dates.format(date, "yyyy-mm-dd"),
        "dt_met_seconds"   => 3600.0,
        "product"          => PRODUCT,
    )
    header_json = JSON3.write(header)
    length(header_json) < HEADER_SIZE ||
        error("Header JSON too large ($(length(header_json)) >= $HEADER_SIZE)")

    @info "  Writing: $outpath"
    open(outpath, "w") do io
        header_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(header_buf, 1, Vector{UInt8}(header_json), 1, length(header_json))
        write(io, header_buf)

        win = 0
        for filepath in files
            for tidx in 1:n_timesteps_per_file
                win += 1
                t0 = time()

                ts = read_geosfp_cs_timestep(filepath; FT,
                        time_index=tidx, convert_to_kgs=true)
                delp_haloed, mfxc, mfyc = to_haloed_panels(ts; Hp)
                am_panels, bm_panels = cgrid_to_staggered_panels(mfxc, mfyc)

                for p in 1:6
                    write(io, vec(delp_haloed[p]))
                end
                for p in 1:6
                    write(io, vec(am_panels[p]))
                end
                for p in 1:6
                    write(io, vec(bm_panels[p]))
                end

                elapsed = round(time() - t0, digits=2)
                if win <= 3 || win == Nt || win % 6 == 0
                    @info @sprintf("    Window %d/%d: %.2fs", win, Nt, elapsed)
                end
            end
        end
    end

    actual = filesize(outpath)
    expected = HEADER_SIZE + bytes_per_window * Nt
    @info @sprintf("  Done: %.2f GB (expected %.2f GB)", actual / 1e9, expected / 1e9)
    actual == expected || @warn "Size mismatch: expected $expected, got $actual"
end

# ===========================================================================
# Main
# ===========================================================================

function main()
    @info "=" ^ 70
    @info "GEOS Cubed-Sphere Preprocessing (product: $PRODUCT)"
    @info "=" ^ 70
    @info "  Data dir: $GEOSFP_DIR"
    @info "  Output:   $OUTDIR"
    @info "  Dates:    $START_DATE to $END_DATE"
    @info "  FT=$FT, Hp=$Hp"

    mkpath(OUTDIR)
    wall_start = time()
    days_done = 0

    for date in START_DATE:Day(1):END_DATE
        datestr = Dates.format(date, "yyyymmdd")
        outpath = joinpath(OUTDIR, "geosfp_cs_$(datestr)_$(lowercase(FT_STR)).bin")

        if isfile(outpath) && filesize(outpath) > HEADER_SIZE
            @info "  [$datestr] Already exists ($(round(filesize(outpath)/1e9, digits=2)) GB) — skipping"
            days_done += 1
            continue
        end

        files, nts = find_files_for_day(GEOSFP_DIR, date, PRODUCT)
        if isempty(files)
            @warn "  [$datestr] No files found — skipping"
            continue
        end

        @info "\n--- [$datestr] Processing ($(length(files)) file(s), $nts timesteps each) ---"
        preprocess_day(date, files, nts, outpath)
        days_done += 1
    end

    wall_total = round(time() - wall_start, digits=1)
    @info "\n" * "=" ^ 70
    @info "Preprocessing complete!"
    @info "  Wall time: $(wall_total)s"
    @info "  Days processed: $days_done"
    @info "  Output directory: $OUTDIR"
    for date in START_DATE:Day(1):END_DATE
        datestr = Dates.format(date, "yyyymmdd")
        fp = joinpath(OUTDIR, "geosfp_cs_$(datestr)_$(lowercase(FT_STR)).bin")
        if isfile(fp)
            @info @sprintf("    %s: %.2f GB", datestr, filesize(fp) / 1e9)
        end
    end
    @info "=" ^ 70
end

main()
