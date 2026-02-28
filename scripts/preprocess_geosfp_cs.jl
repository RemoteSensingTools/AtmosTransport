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
#   julia --project=. scripts/preprocess_geosfp_cs.jl <config.toml>
#
# TOML config sections:
#   [product]   name, mass_flux_dt
#   [input]     data_dir, start_date, end_date
#   [output]    directory
#   [grid]      halo_width
#   [numerics]  float_type
#   [diagnostics] verbose
# ===========================================================================

using AtmosTransport
using AtmosTransport.IO: read_geosfp_cs_timestep, to_haloed_panels,
                              cgrid_to_staggered_panels
import AtmosTransport.IO: GEOS_CS_PRODUCTS
using NCDatasets
using Dates
using Printf
using JSON3
using TOML
using Statistics: mean, std

length(ARGS) >= 1 || error(
    "Usage: julia --project=. scripts/preprocess_geosfp_cs.jl <config.toml>")

const cfg = TOML.parsefile(ARGS[1])

# --- Product ---
const PRODUCT      = cfg["product"]["name"]
const MASS_FLUX_DT = Float64(get(cfg["product"], "mass_flux_dt",
                                  PRODUCT == "geosit_c180" ? 450.0 : 3600.0))

# --- Input / output ---
const GEOSFP_DIR = expanduser(cfg["input"]["data_dir"])
const START_DATE = Date(cfg["input"]["start_date"])
const END_DATE   = Date(cfg["input"]["end_date"])
const OUTDIR     = expanduser(cfg["output"]["directory"])

# --- Grid ---
const Hp = Int(get(get(cfg, "grid", Dict()), "halo_width", 3))

# --- Numerics ---
const FT_STR = get(get(cfg, "numerics", Dict()), "float_type", "Float32")
const FT     = FT_STR == "Float64" ? Float64 : Float32

# --- Diagnostics ---
const VERBOSE = get(get(cfg, "diagnostics", Dict()), "verbose", false)

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
                nt = NCDataset(filepath, "r") do ds; length(ds["time"]); end
                return [filepath], nt
            end
        end
        return String[], 0
    end
end

# ---------------------------------------------------------------------------
# Sanity checks
#
# Print statistics that help catch common bugs:
#   - Wrong mass_flux_dt (fluxes 8× too small/large)
#   - Inverted vertical ordering (DELP decreasing from k=1)
#   - NaN / Inf contamination
#   - Unrealistic wind speeds
# ---------------------------------------------------------------------------

const GRAV = 9.80616f0   # m/s²
const Nc_CELL_SIZE_DEG = 360.0 / (PRODUCT == "geosit_c180" ? 180 : 720)

function sanity_check_window(delp_panels, am_panels, bm_panels, win::Int)
    p1_delp = delp_panels[1]   # (Nc+2Hp, Nc+2Hp, Nz) — includes halo
    p1_am   = am_panels[1]     # (Nc+1, Nc, Nz)
    p1_bm   = bm_panels[1]     # (Nc, Nc+1, Nz)

    # Strip halo for DELP stats
    inner_delp = p1_delp[Hp+1:end-Hp, Hp+1:end-Hp, :]

    # --- NaN / Inf check ---
    n_nan  = count(isnan, inner_delp) + count(isnan, p1_am) + count(isnan, p1_bm)
    n_inf  = count(isinf, inner_delp) + count(isinf, p1_am) + count(isinf, p1_bm)
    if n_nan > 0 || n_inf > 0
        msg = @sprintf("  [sanity win=%d] NaN=%d, Inf=%d — data quality issue!", win, n_nan, n_inf)
        @warn msg
    end

    # --- Column pressure sum (should be ~1e5 Pa = surface pressure) ---
    col_ps = sum(inner_delp, dims=3)[:, :, 1]   # sum over levels
    ps_mean = mean(col_ps)
    ps_min  = minimum(col_ps)
    ps_max  = maximum(col_ps)
    if !(8e4 < ps_mean < 1.1e5)
        msg = @sprintf("  [sanity win=%d] DELP column sum = %.0f Pa — expected ~1e5 Pa. Check accumulation units or vertical ordering.", win, ps_mean)
        @warn msg
    end

    # --- Vertical ordering: DELP[k=1] should be thin (stratosphere) ---
    # DELP at level 1 should be << level Nz (surface)
    delp_top    = mean(inner_delp[:, :, 1])
    delp_bot    = mean(inner_delp[:, :, end])
    if delp_top > delp_bot
        msg = @sprintf("  [sanity win=%d] DELP[k=1]=%.2f Pa > DELP[k=Nz]=%.2f Pa — vertical ordering looks inverted (bottom-to-top)! Check auto-detection in read_geosfp_cs_timestep.", win, delp_top, delp_bot)
        @warn msg
    end

    # --- Approximate wind speed from am (interior cells, lowest level) ---
    # am [kg/(m²·s)] × area [m²] / DELP [Pa] × g [m/s²] ≈ u [m/s] × (rho × dz)/(rho × dz) = u
    # Simpler: u ≈ am / (DELP / g) = am * g / DELP (per unit area, kg/m²/s / (kg/m²) = 1/s × L)
    # Better: use mean |am| over interior, estimate |u| ≈ am_mean * g / delp_mean / (pi/180 * R_earth * Nc_cells)
    am_interior = p1_am[2:end-1, :, end]   # avoid boundary, surface level
    delp_interior = inner_delp[:, :, end]
    rms_am  = sqrt(mean(x -> x^2, am_interior))
    mean_dp = mean(delp_interior)
    # u ≈ |am| * g / DELP  (where am is mass flux per unit face area in kg/m²/s)
    # But am is total flux through face; to get velocity, divide by DELP/g (air column mass/area)
    # u [m/s] = am [kg/m²/s] / (DELP/g [kg/m²]) = am * g / DELP
    u_est = rms_am * GRAV / mean_dp
    if u_est > 200.0
        msg = @sprintf("  [sanity win=%d] Estimated surface wind ~%.0f m/s — suspiciously high! Check mass_flux_dt (should be %.0fs for %s).", win, u_est, MASS_FLUX_DT, PRODUCT)
        @warn msg
    elseif u_est < 0.1
        msg = @sprintf("  [sanity win=%d] Estimated surface wind ~%.3f m/s — suspiciously low! Check mass_flux_dt (should be %.0fs for %s).", win, u_est, MASS_FLUX_DT, PRODUCT)
        @warn msg
    end

    msg = @sprintf("  [sanity win=%d] DELP col-sum: mean=%.0f min=%.0f max=%.0f Pa | DELP top=%.3f bot=%.1f Pa | est |u_sfc|=%.2f m/s", win, ps_mean, ps_min, ps_max, delp_top, delp_bot, u_est)
    @info msg
end

# ---------------------------------------------------------------------------
# Per-day preprocessing
# ---------------------------------------------------------------------------

function preprocess_day(date::Date, files::Vector{String},
                        n_timesteps_per_file::Int, outpath::String)
    isempty(files) && return

    Nt = length(files) * n_timesteps_per_file
    Nt == 0 && return

    @info "  Reading first file for dimensions..."
    ts0 = read_geosfp_cs_timestep(files[1]; FT, convert_to_kgs=true, dt_met=MASS_FLUX_DT)
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
    @info @sprintf("  mass_flux_dt=%.0fs  (product=%s)", MASS_FLUX_DT, PRODUCT)
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
        "dt_met_seconds"   => MASS_FLUX_DT,
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
                        time_index=tidx, convert_to_kgs=true, dt_met=MASS_FLUX_DT)
                delp_haloed, mfxc, mfyc = to_haloed_panels(ts; Hp)
                am_panels, bm_panels = cgrid_to_staggered_panels(mfxc, mfyc)

                # Sanity check on first window and every 6th
                if VERBOSE && (win == 1 || win % 6 == 0)
                    sanity_check_window(delp_haloed, am_panels, bm_panels, win)
                end

                for p in 1:6; write(io, vec(delp_haloed[p])); end
                for p in 1:6; write(io, vec(am_panels[p]));   end
                for p in 1:6; write(io, vec(bm_panels[p]));   end

                elapsed = round(time() - t0, digits=2)
                if win <= 3 || win == Nt || win % 6 == 0
                    @info @sprintf("    Window %d/%d: %.2fs", win, Nt, elapsed)
                end
            end
        end
    end

    actual   = filesize(outpath)
    expected = HEADER_SIZE + bytes_per_window * Nt
    @info @sprintf("  Done: %.2f GB (expected %.2f GB)", actual / 1e9, expected / 1e9)
    actual == expected || @warn "Size mismatch: expected $expected, got $actual"
end

# ===========================================================================
# Main
# ===========================================================================

function main()
    @info "=" ^ 70
    @info "GEOS Cubed-Sphere Preprocessing"
    @info "=" ^ 70
    @info "  Product:       $PRODUCT"
    @info "  mass_flux_dt:  $(MASS_FLUX_DT)s"
    @info "  Data dir:      $GEOSFP_DIR"
    @info "  Output:        $OUTDIR"
    @info "  Dates:         $START_DATE to $END_DATE"
    @info "  FT=$FT, Hp=$Hp"
    @info "  Verbose:       $VERBOSE"

    mkpath(OUTDIR)
    wall_start = time()
    days_done  = 0

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
        isfile(fp) && @info @sprintf("    %s: %.2f GB", datestr, filesize(fp) / 1e9)
    end
    @info "=" ^ 70
end

main()
