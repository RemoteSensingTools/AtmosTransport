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
# Usage (sequential):
#   julia --project=. scripts/preprocess_geosfp_cs.jl <config.toml>
#
# Parallel (8 workers — one day per worker):
#   julia --project=. -p 8 scripts/preprocess_geosfp_cs.jl <config.toml>
#
# TOML config sections:
#   [product]      name, mass_flux_dt
#   [input]        data_dir, start_date, end_date
#   [output]       directory
#   [grid]         halo_width, merge_levels_above_Pa (optional)
#   [numerics]     float_type
#   [diagnostics]  verbose
# ===========================================================================

using Distributed

# Load packages on all processes (main + workers)
@everywhere begin
    using AtmosTransport
    using AtmosTransport.IO: read_geosfp_cs_timestep, to_haloed_panels,
                                  cgrid_to_staggered_panels,
                                  default_met_config, build_vertical_coordinate
    import AtmosTransport.IO: GEOS_CS_PRODUCTS
    using AtmosTransport.Grids: merge_upper_levels, n_levels
    using NCDatasets
    using Dates
    using Printf
    using JSON3
    using TOML
    using Statistics: mean, std
end

# ---------------------------------------------------------------------------
# Parse config (main process only)
# ---------------------------------------------------------------------------

length(ARGS) >= 1 || error(
    "Usage: julia --project=. [-p N] scripts/preprocess_geosfp_cs.jl <config.toml>")

const cfg = TOML.parsefile(ARGS[1])

# --- Product ---
const PRODUCT      = cfg["product"]["name"]
const MASS_FLUX_DT = Float64(get(cfg["product"], "mass_flux_dt", 450.0))

# --- Input / output ---
const GEOSFP_DIR = expanduser(cfg["input"]["data_dir"])
const START_DATE = Date(cfg["input"]["start_date"])
const END_DATE   = Date(cfg["input"]["end_date"])
const OUTDIR     = expanduser(cfg["output"]["directory"])

# --- Grid ---
const Hp = Int(get(get(cfg, "grid", Dict()), "halo_width", 3))
const MERGE_PA = let v = get(get(cfg, "grid", Dict()), "merge_levels_above_Pa", nothing)
    v === nothing ? nothing : Float64(v)
end

# --- Numerics ---
const FT_STR = get(get(cfg, "numerics", Dict()), "float_type", "Float32")
const FT     = FT_STR == "Float64" ? Float64 : Float32

# --- Diagnostics ---
const VERBOSE = get(get(cfg, "diagnostics", Dict()), "verbose", false)

# --- Level merging ---
const MERGE_MAP = if MERGE_PA !== nothing
    met_source = PRODUCT == "geosit_c180" ? "geosit" : "geosfp"
    met_cfg = default_met_config(met_source)
    vc = build_vertical_coordinate(met_cfg; FT=Float64)
    _, mm = merge_upper_levels(vc, MERGE_PA)
    mm
else
    nothing
end

# Bundle config into a NamedTuple for worker processes.
# pmap serializes this automatically — no global state needed on workers.
const PCFG = (
    product      = PRODUCT,
    mass_flux_dt = MASS_FLUX_DT,
    geosfp_dir   = GEOSFP_DIR,
    outdir       = OUTDIR,
    Hp           = Hp,
    merge_pa     = MERGE_PA,
    merge_map    = MERGE_MAP,
    FT           = FT,
    FT_str       = FT_STR,
    verbose      = VERBOSE,
)

# ---------------------------------------------------------------------------
# Constants and functions (defined on all processes via @everywhere)
# ---------------------------------------------------------------------------

@everywhere begin

const HEADER_SIZE = 8192
const GRAV        = 9.80616f0     # m/s²
const R_EARTH     = 6.371e6       # m

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
# ---------------------------------------------------------------------------

function sanity_check_window(delp_panels, am_panels, bm_panels, win::Int,
                             _Hp::Int, mass_flux_dt::Float64, product::String)
    p1_delp = delp_panels[1]   # (Nc+2Hp, Nc+2Hp, Nz) — includes halo
    p1_am   = am_panels[1]     # (Nc+1, Nc, Nz)
    p1_bm   = bm_panels[1]     # (Nc, Nc+1, Nz)

    # Strip halo for DELP stats
    inner_delp = p1_delp[_Hp+1:end-_Hp, _Hp+1:end-_Hp, :]

    # --- NaN / Inf check ---
    n_nan  = count(isnan, inner_delp) + count(isnan, p1_am) + count(isnan, p1_bm)
    n_inf  = count(isinf, inner_delp) + count(isinf, p1_am) + count(isinf, p1_bm)
    if n_nan > 0 || n_inf > 0
        @warn @sprintf("  [sanity win=%d] NaN=%d, Inf=%d — data quality issue!", win, n_nan, n_inf)
    end

    # --- Column pressure sum (should be ~1e5 Pa = surface pressure) ---
    col_ps = sum(inner_delp, dims=3)[:, :, 1]
    ps_mean = mean(col_ps)
    ps_min  = minimum(col_ps)
    ps_max  = maximum(col_ps)
    if !(8e4 < ps_mean < 1.1e5)
        @warn @sprintf("  [sanity win=%d] DELP column sum = %.0f Pa — expected ~1e5 Pa.", win, ps_mean)
    end

    # --- Vertical ordering: DELP[k=1] should be thin (stratosphere) ---
    delp_top = mean(inner_delp[:, :, 1])
    delp_bot = mean(inner_delp[:, :, end])
    if delp_top > delp_bot
        @warn @sprintf("  [sanity win=%d] DELP[k=1]=%.2f Pa > DELP[k=Nz]=%.2f Pa — inverted?",
                        win, delp_top, delp_bot)
    end

    # --- Approximate surface wind speed from am ---
    Nc = size(p1_am, 2)
    dy = 2π * R_EARTH / (4 * Nc)
    am_interior = p1_am[2:end-1, :, end]
    delp_interior = inner_delp[:, :, end]
    rms_am  = sqrt(mean(x -> x^2, am_interior))
    mean_dp = mean(delp_interior)
    u_est = rms_am * GRAV / (mean_dp * dy)
    if u_est > 80.0
        @warn @sprintf("  [sanity win=%d] |u_sfc| ~%.1f m/s — suspiciously high! mass_flux_dt=%.0fs?",
                        win, u_est, mass_flux_dt)
    elseif u_est < 0.5
        @warn @sprintf("  [sanity win=%d] |u_sfc| ~%.3f m/s — suspiciously low! mass_flux_dt=%.0fs?",
                        win, u_est, mass_flux_dt)
    end

    @info @sprintf("  [sanity win=%d] DELP col-sum: mean=%.0f min=%.0f max=%.0f Pa | top=%.3f bot=%.1f Pa | |u_sfc|=%.1f m/s (dy=%.0fm)",
                    win, ps_mean, ps_min, ps_max, delp_top, delp_bot, u_est, dy)
end

# ---------------------------------------------------------------------------
# Per-day preprocessing
# ---------------------------------------------------------------------------

function preprocess_day(date::Date, files::Vector{String},
                        n_timesteps_per_file::Int, outpath::String, pcfg)
    isempty(files) && return

    Nt = length(files) * n_timesteps_per_file
    Nt == 0 && return

    _FT = pcfg.FT
    _Hp = pcfg.Hp
    mm  = pcfg.merge_map

    @info "  [worker $(myid())] Reading first file for dimensions..."
    ts0 = read_geosfp_cs_timestep(files[1]; FT=_FT, convert_to_kgs=true,
                                   dt_met=pcfg.mass_flux_dt)
    Nc, Nz_native = ts0.Nc, ts0.Nz

    # Output Nz: merged if merge_map is set, native otherwise
    Nz = mm !== nothing ? maximum(mm) : Nz_native

    n_delp_panel = (Nc + 2_Hp) * (Nc + 2_Hp) * Nz
    n_am_panel   = (Nc + 1) * Nc * Nz
    n_bm_panel   = Nc * (Nc + 1) * Nz
    n_delp_total = 6 * n_delp_panel
    n_am_total   = 6 * n_am_panel
    n_bm_total   = 6 * n_bm_panel
    elems_per_window = n_delp_total + n_am_total + n_bm_total
    bytes_per_window = elems_per_window * sizeof(_FT)

    merge_str = mm !== nothing ? " (merged from $Nz_native)" : ""
    @info @sprintf("  Grid: C%d, Nz=%d%s, Hp=%d, Nt=%d", Nc, Nz, merge_str, _Hp, Nt)
    @info @sprintf("  mass_flux_dt=%.0fs  (product=%s)", pcfg.mass_flux_dt, pcfg.product)
    @info @sprintf("  Per window: %.1f MB (%d elems)", bytes_per_window / 1e6, elems_per_window)
    @info @sprintf("  Total binary: %.2f GB", (HEADER_SIZE + bytes_per_window * Nt) / 1e9)

    header = Dict{String,Any}(
        "magic"            => "CSFLX",
        "version"          => 1,
        "grid_type"        => "cubed_sphere",
        "Nc"               => Nc,
        "Nz"               => Nz,
        "Hp"               => _Hp,
        "Nt"               => Nt,
        "float_type"       => pcfg.FT_str,
        "float_bytes"      => sizeof(_FT),
        "header_bytes"     => HEADER_SIZE,
        "window_bytes"     => bytes_per_window,
        "n_delp_panel"     => n_delp_panel,
        "n_am_panel"       => n_am_panel,
        "n_bm_panel"       => n_bm_panel,
        "n_panels"         => 6,
        "elems_per_window" => elems_per_window,
        "date"             => Dates.format(date, "yyyy-mm-dd"),
        "dt_met_seconds"   => pcfg.mass_flux_dt,
        "product"          => pcfg.product,
    )
    if mm !== nothing
        header["Nz_native"]          = Nz_native
        header["merge_levels_above"] = pcfg.merge_pa
    end
    header_json = JSON3.write(header)
    length(header_json) < HEADER_SIZE ||
        error("Header JSON too large ($(length(header_json)) >= $HEADER_SIZE)")

    # Pre-allocate merged panel buffers (reused across windows)
    merged_delp = mm !== nothing ? ntuple(_ -> Array{_FT}(undef, Nc + 2_Hp, Nc + 2_Hp, Nz), 6) : nothing
    merged_am   = mm !== nothing ? ntuple(_ -> Array{_FT}(undef, Nc + 1, Nc, Nz), 6) : nothing
    merged_bm   = mm !== nothing ? ntuple(_ -> Array{_FT}(undef, Nc, Nc + 1, Nz), 6) : nothing

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

                ts = read_geosfp_cs_timestep(filepath; FT=_FT,
                        time_index=tidx, convert_to_kgs=true, dt_met=pcfg.mass_flux_dt)
                delp_haloed, mfxc, mfyc = to_haloed_panels(ts; Hp=_Hp)
                am_panels, bm_panels = cgrid_to_staggered_panels(mfxc, mfyc)

                # Sanity check on first window and every 6th (on native data)
                if pcfg.verbose && (win == 1 || win % 6 == 0)
                    sanity_check_window(delp_haloed, am_panels, bm_panels, win,
                                        _Hp, pcfg.mass_flux_dt, pcfg.product)
                end

                if mm !== nothing
                    # Merge native levels → output levels by summing within groups
                    for p in 1:6
                        fill!(merged_delp[p], zero(_FT))
                        fill!(merged_am[p],   zero(_FT))
                        fill!(merged_bm[p],   zero(_FT))
                        for k in 1:length(mm)
                            km = mm[k]
                            merged_delp[p][:, :, km] .+= delp_haloed[p][:, :, k]
                            merged_am[p][:, :, km]   .+= am_panels[p][:, :, k]
                            merged_bm[p][:, :, km]   .+= bm_panels[p][:, :, k]
                        end
                    end
                    for p in 1:6; write(io, vec(merged_delp[p])); end
                    for p in 1:6; write(io, vec(merged_am[p]));   end
                    for p in 1:6; write(io, vec(merged_bm[p]));   end
                else
                    for p in 1:6; write(io, vec(delp_haloed[p])); end
                    for p in 1:6; write(io, vec(am_panels[p]));   end
                    for p in 1:6; write(io, vec(bm_panels[p]));   end
                end

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

"""Process a single day: find files, check skip, run preprocess_day."""
function process_one_day(date::Date, pcfg)
    datestr = Dates.format(date, "yyyymmdd")
    outpath = joinpath(pcfg.outdir, "geosfp_cs_$(datestr)_$(lowercase(pcfg.FT_str)).bin")

    if isfile(outpath) && filesize(outpath) > HEADER_SIZE
        @info "  [$datestr] Already exists ($(round(filesize(outpath)/1e9, digits=2)) GB) — skipping"
        return (date, :skipped)
    end

    files, nts = find_files_for_day(pcfg.geosfp_dir, date, pcfg.product)
    if isempty(files)
        @warn "  [$datestr] No files found — skipping"
        return (date, :missing)
    end

    @info "\n--- [$datestr] Processing on worker $(myid()) ($(length(files)) file(s), $nts timesteps each) ---"
    preprocess_day(date, files, nts, outpath, pcfg)
    return (date, :done)
end

end # @everywhere

# ===========================================================================
# Main (runs on process 1 only)
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
    if MERGE_MAP !== nothing
        @info "  Level merging: above $(MERGE_PA) Pa → $(maximum(MERGE_MAP)) levels"
    end
    @info "  Verbose:       $VERBOSE"
    @info "  Workers:       $(nworkers())"

    mkpath(OUTDIR)
    wall_start = time()

    dates = collect(START_DATE:Day(1):END_DATE)

    # pmap distributes days across workers; with 1 worker it runs sequentially
    results = pmap(d -> process_one_day(d, PCFG), dates)

    days_done    = count(r -> r[2] === :done, results)
    days_skipped = count(r -> r[2] === :skipped, results)
    days_missing = count(r -> r[2] === :missing, results)

    wall_total = round(time() - wall_start, digits=1)
    @info "\n" * "=" ^ 70
    @info "Preprocessing complete!"
    @info "  Wall time: $(wall_total)s"
    @info "  Days processed: $days_done, skipped: $days_skipped, missing: $days_missing"
    @info "  Workers used: $(nworkers())"
    @info "  Output directory: $OUTDIR"
    for date in dates
        datestr = Dates.format(date, "yyyymmdd")
        fp = joinpath(OUTDIR, "geosfp_cs_$(datestr)_$(lowercase(FT_STR)).bin")
        isfile(fp) && @info @sprintf("    %s: %.2f GB", datestr, filesize(fp) / 1e9)
    end
    @info "=" ^ 70
end

main()
