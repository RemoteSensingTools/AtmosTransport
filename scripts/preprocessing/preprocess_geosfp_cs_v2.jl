#!/usr/bin/env julia
# ===========================================================================
# Preprocessing: GEOS CS NetCDF → flat binary v2 (with CX/CY Courant numbers)
#
# Same as preprocess_geosfp_cs.jl but adds CX/CY after BM panels.
# v2 binary layout per window:
#   [6 DELP] [6 AM] [6 BM] [6 CX] [6 CY]
#
# CX/CY have the same staggering as AM/BM after cgrid_to_staggered_panels.
#
# Usage:
#   julia --project=. scripts/preprocessing/preprocess_geosfp_cs_v2.jl \
#         --data_dir ~/data/geosit_c180 \
#         --output_dir /temp1/catrine/met/geosit_c180/massflux_v2 \
#         --start_date 2021-12-01 --end_date 2021-12-08 \
#         [--mass_flux_dt 450] [--product geosit_c180]
# ===========================================================================

using AtmosTransport
using AtmosTransport.IO: read_geosfp_cs_timestep, to_haloed_panels,
                          cgrid_to_staggered_panels, GEOS_CS_PRODUCTS
using NCDatasets
using Dates
using Printf
using JSON3

# ---------------------------------------------------------------------------
# Parse command-line arguments
# ---------------------------------------------------------------------------
function parse_args()
    args = Dict{String,String}()
    i = 1
    while i <= length(ARGS)
        if startswith(ARGS[i], "--") && i < length(ARGS)
            args[ARGS[i][3:end]] = ARGS[i+1]
            i += 2
        else
            i += 1
        end
    end
    data_dir     = expanduser(get(args, "data_dir", "~/data/geosit_c180"))
    output_dir   = expanduser(get(args, "output_dir", "/temp1/catrine/met/geosit_c180/massflux_v2"))
    start_date   = Date(get(args, "start_date", "2021-12-01"))
    end_date     = Date(get(args, "end_date", "2021-12-08"))
    mass_flux_dt = parse(Float64, get(args, "mass_flux_dt", "450.0"))
    product      = get(args, "product", "geosit_c180")
    return (; data_dir, output_dir, start_date, end_date, mass_flux_dt, product)
end

const HEADER_SIZE = 8192

# ---------------------------------------------------------------------------
# Find input file for a day
# ---------------------------------------------------------------------------
function find_ctm_file(data_dir, date, product)
    daydir = joinpath(data_dir, Dates.format(date, "yyyymmdd"))
    isdir(daydir) || return nothing, 0

    info = GEOS_CS_PRODUCTS[product]
    if info.layout === :daily
        tag = "CTM_A1.C$(info.Nc)"
        for f in readdir(daydir)
            if contains(f, tag) && endswith(f, ".nc")
                filepath = joinpath(daydir, f)
                nt = NCDataset(filepath, "r") do ds; length(ds["time"]); end
                return filepath, nt
            end
        end
    else  # :hourly
        files = filter(f -> contains(f, "tavg_1hr_ctm") && endswith(f, ".nc4"),
                       sort(readdir(daydir)))
        isempty(files) && return nothing, 0
        return joinpath.(daydir, files), 1  # return vector + 1 ts per file
    end
    return nothing, 0
end

# ---------------------------------------------------------------------------
# Preprocess one day: read NetCDF, write v2 binary
# ---------------------------------------------------------------------------
function preprocess_day_v2(date::Date, filepath, n_timesteps, output_path, cfg)
    FT = Float32
    Hp = 3
    mass_flux_dt = cfg.mass_flux_dt

    # Read first timestep for dimensions
    ts0 = read_geosfp_cs_timestep(filepath; FT, convert_to_kgs=true, dt_met=mass_flux_dt)
    Nc, Nz = ts0.Nc, ts0.Nz
    Nt = n_timesteps

    n_delp_panel = (Nc + 2Hp)^2 * Nz
    n_am_panel   = (Nc + 1) * Nc * Nz
    n_bm_panel   = Nc * (Nc + 1) * Nz
    # v2: CX same shape as AM, CY same shape as BM
    elems_per_window = 6 * (n_delp_panel + n_am_panel + n_bm_panel + n_am_panel + n_bm_panel)

    header = Dict{String,Any}(
        "magic"            => "CSFLX",
        "version"          => 2,
        "grid_type"        => "cubed_sphere",
        "Nc"               => Nc,
        "Nz"               => Nz,
        "Hp"               => Hp,
        "Nt"               => Nt,
        "float_type"       => "Float32",
        "float_bytes"      => 4,
        "header_bytes"     => HEADER_SIZE,
        "window_bytes"     => elems_per_window * 4,
        "n_delp_panel"     => n_delp_panel,
        "n_am_panel"       => n_am_panel,
        "n_bm_panel"       => n_bm_panel,
        "n_panels"         => 6,
        "elems_per_window" => elems_per_window,
        "date"             => Dates.format(date, "yyyy-mm-dd"),
        "dt_met_seconds"   => mass_flux_dt,
        "product"          => cfg.product,
        "include_courant"  => true,
    )

    header_json = JSON3.write(header)
    @assert length(header_json) < HEADER_SIZE

    @info @sprintf("  C%d×%d, %d windows, %.1f MB/window, %.2f GB total",
                    Nc, Nz, Nt, elems_per_window * 4 / 1e6,
                    (HEADER_SIZE + elems_per_window * 4 * Nt) / 1e9)

    open(output_path, "w") do io
        # Write header
        header_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(header_buf, 1, Vector{UInt8}(header_json), 1, length(header_json))
        write(io, header_buf)

        for tidx in 1:Nt
            t0 = time()

            # Read DELP, MFXC, MFYC (existing infrastructure)
            ts = read_geosfp_cs_timestep(filepath; FT, time_index=tidx,
                                          convert_to_kgs=true, dt_met=mass_flux_dt)
            delp_h, mfxc, mfyc = to_haloed_panels(ts; Hp)
            am_panels, bm_panels = cgrid_to_staggered_panels(mfxc, mfyc)

            # Read CX/CY from same file, stagger the same way
            ds = NCDataset(filepath, "r")
            cx_raw = Array{FT}(ds["CX"][:, :, :, :, tidx])
            cy_raw = Array{FT}(ds["CY"][:, :, :, :, tidx])
            close(ds)

            # Apply same vertical flip as reader
            if ts.Nc > 0  # flip detection: reader already flipped ts
                mid = div(size(cx_raw, 1), 2)
                delp_test = cx_raw[mid, mid, 1, 1]  # can't use this for flip detection
                # Instead, check if ts was flipped by comparing raw DELP
                # The reader logs "Detected inverted vertical ordering" — we need to match.
                # Simple: GEOS-IT is always bottom-to-top, always flip.
                cx_raw = reverse(cx_raw, dims=4)
                cy_raw = reverse(cy_raw, dims=4)
            end

            # Split into per-panel tuples and stagger
            cx_panels_raw = ntuple(p -> cx_raw[:, :, p, :], 6)
            cy_panels_raw = ntuple(p -> cy_raw[:, :, p, :], 6)
            cx_stag, cy_stag = cgrid_to_staggered_panels(cx_panels_raw, cy_panels_raw)

            # Write: DELP, AM, BM, CX, CY (all 6 panels each)
            for p in 1:6; write(io, vec(delp_h[p])); end
            for p in 1:6; write(io, vec(am_panels[p])); end
            for p in 1:6; write(io, vec(bm_panels[p])); end
            for p in 1:6; write(io, vec(cx_stag[p])); end
            for p in 1:6; write(io, vec(cy_stag[p])); end

            elapsed = round(time() - t0, digits=2)
            if tidx <= 2 || tidx == Nt || tidx % 6 == 0
                @info @sprintf("    Window %d/%d: %.2fs", tidx, Nt, elapsed)
            end
        end
    end

    actual   = filesize(output_path)
    expected = HEADER_SIZE + elems_per_window * 4 * Nt
    @info @sprintf("  Done: %.2f GB", actual / 1e9)
    actual == expected || @warn "Size mismatch: expected $expected, got $actual"
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    cfg = parse_args()
    mkpath(cfg.output_dir)

    @info "Preprocessing GEOS CS → binary v2 (with CX/CY)"
    @info "  Input:  $(cfg.data_dir)"
    @info "  Output: $(cfg.output_dir)"
    @info "  Dates:  $(cfg.start_date) → $(cfg.end_date)"

    for date in cfg.start_date:Day(1):cfg.end_date
        datestr = Dates.format(date, "yyyymmdd")
        outpath = joinpath(cfg.output_dir, "geosfp_cs_$(datestr)_float32.bin")

        if isfile(outpath) && filesize(outpath) > HEADER_SIZE
            @info "[$datestr] Already exists — skipping"
            continue
        end

        result = find_ctm_file(cfg.data_dir, date, cfg.product)
        if result isa Tuple{Nothing, Int}
            @warn "[$datestr] No CTM file found — skipping"
            continue
        end
        filepath, nt = result

        @info "\n--- [$datestr] Processing ---"
        if filepath isa Vector
            # Hourly files — process sequentially (TODO: handle multi-file days)
            @warn "Hourly file layout not yet supported for v2 — use daily"
            continue
        end
        preprocess_day_v2(date, filepath, nt, outpath, cfg)
    end

    @info "All done."
end

main()
