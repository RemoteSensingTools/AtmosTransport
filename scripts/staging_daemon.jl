#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# NVMe Staging Daemon — rolling copy of preprocessed binary files
#
# Keeps a small window of met data on fast NVMe storage while the simulation
# runs. Copies files ahead of the current window and cleans up behind it.
#
# Design:
#   - The model writes its current date to a progress file (1 line: YYYY-MM-DD)
#   - The daemon polls this file and ensures the next N days are on NVMe
#   - Files older than keep_behind days are removed from NVMe
#   - Source is the NAS archive (slow but large)
#   - Destination is NVMe staging (fast but small)
#
# Usage:
#   julia scripts/staging_daemon.jl <config.toml>
#
# Config TOML:
#   [staging]
#   source_dir      = "~/data/AtmosTransport/met/geosfp_c720/preprocessed/massflux_v4"
#   staging_dir     = "/temp2/staging/geosfp_c720/massflux_v4"
#   progress_file   = "/temp2/staging/geosfp_c720/progress.txt"
#   lookahead_days  = 3       # copy this many days ahead
#   keep_behind     = 1       # keep this many days behind current
#   poll_interval   = 10      # seconds between checks
#   file_pattern    = "geosfp_cs_{DATE}_float32.bin"  # {DATE} → YYYYMMDD
#   start_date      = "2021-12-01"   # pre-stage from this date
#   end_date        = "2021-12-31"
#
# The model config should point preprocessed_dir to staging_dir.
# ---------------------------------------------------------------------------

using Dates
using TOML
using Printf

function main()
    if isempty(ARGS) || ARGS[1] in ("-h", "--help")
        println("""
        NVMe Staging Daemon
        ===================
        Usage: julia scripts/staging_daemon.jl <config.toml>

        Keeps a rolling window of preprocessed binary files on NVMe.
        The simulation reads from NVMe; this daemon copies from NAS.

        Config sections: [staging] (see source for keys)
        """)
        return
    end

    cfg = TOML.parsefile(ARGS[1])
    stg = cfg["staging"]

    source_dir     = expanduser(stg["source_dir"])
    staging_dir    = expanduser(stg["staging_dir"])
    progress_file  = expanduser(stg["progress_file"])
    lookahead      = get(stg, "lookahead_days", 3)
    keep_behind    = get(stg, "keep_behind", 1)
    poll_interval  = get(stg, "poll_interval", 10)
    file_pattern   = get(stg, "file_pattern", "geosfp_cs_{DATE}_float32.bin")
    start_date     = Date(stg["start_date"])
    end_date       = Date(stg["end_date"])

    # Also stage surface binary files if configured
    surface_source = expanduser(get(stg, "surface_source_dir", ""))
    surface_staging = expanduser(get(stg, "surface_staging_dir", ""))
    surface_patterns = get(stg, "surface_patterns",
        ["GEOSFP_CS720.{DATE}.A1.bin", "GEOSFP_CS720.{DATE}.A3mstE.bin",
         "GEOSFP_CS720.{DATE}.A3dyn.bin", "GEOSFP_CS720.{DATE}.I3.bin"])

    mkpath(staging_dir)
    !isempty(surface_staging) && mkpath(surface_staging)

    @info """
    NVMe Staging Daemon
    ===================
    Source (NAS):     $source_dir
    Staging (NVMe):   $staging_dir
    Progress file:    $progress_file
    Lookahead:        $lookahead days
    Keep behind:      $keep_behind day(s)
    Poll interval:    $(poll_interval)s
    Date range:       $start_date → $end_date
    Surface source:   $(isempty(surface_source) ? "(none)" : surface_source)
    """

    # --- Pre-stage initial window ---
    current_date = start_date
    _write_progress(progress_file, current_date)
    _stage_window!(source_dir, staging_dir, file_pattern,
                   surface_source, surface_staging, surface_patterns,
                   current_date, lookahead, end_date)
    @info "Pre-staging complete. Waiting for simulation to start..."

    # --- Main loop ---
    n_staged = 0
    n_cleaned = 0
    while true
        sleep(poll_interval)

        # Read current date from progress file
        new_date = _read_progress(progress_file)
        if new_date === nothing
            continue
        end

        if new_date > end_date
            @info "Simulation reached end_date ($end_date). Shutting down."
            break
        end

        if new_date != current_date
            @info @sprintf("Progress: %s → %s", current_date, new_date)
            current_date = new_date

            # Stage ahead
            ns = _stage_window!(source_dir, staging_dir, file_pattern,
                                surface_source, surface_staging, surface_patterns,
                                current_date, lookahead, end_date)
            n_staged += ns

            # Clean behind
            nc = _clean_behind!(staging_dir, file_pattern,
                                surface_staging, surface_patterns,
                                current_date, keep_behind, start_date)
            n_cleaned += nc

            if ns > 0 || nc > 0
                # Report NVMe usage
                n_files = length(filter(f -> endswith(f, ".bin"), readdir(staging_dir)))
                usage = _dir_size_gb(staging_dir)
                @info @sprintf("Staged +%d, cleaned %d | NVMe: %d files, %.1f GB",
                               ns, nc, n_files, usage)
            end
        end
    end

    @info @sprintf("Staging daemon finished. Total staged: %d, cleaned: %d", n_staged, n_cleaned)
end

# =====================================================================
# Helper functions
# =====================================================================

function _date_to_filename(pattern::String, date::Date)
    datestr = Dates.format(date, "yyyymmdd")
    return replace(pattern, "{DATE}" => datestr)
end

function _write_progress(path::String, date::Date)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, Dates.format(date, "yyyy-mm-dd"))
    end
end

function _read_progress(path::String)
    isfile(path) || return nothing
    try
        s = strip(read(path, String))
        return Date(s)
    catch
        return nothing
    end
end

function _stage_file!(src_dir, dst_dir, filename)
    src = joinpath(src_dir, filename)
    dst = joinpath(dst_dir, filename)
    isfile(dst) && return false  # already staged
    isfile(src) || return false  # source missing
    t0 = time()
    cp(src, dst)
    sz = filesize(dst)
    dt = time() - t0
    rate = sz / dt / 1e6
    @info @sprintf("  Staged %s (%.1f MB, %.0f MB/s)", filename, sz / 1e6, rate)
    return true
end

function _stage_window!(src_dir, dst_dir, pattern,
                        sfc_src, sfc_dst, sfc_patterns,
                        current_date, lookahead, end_date)
    n = 0
    for d in 0:lookahead
        date = current_date + Day(d)
        date > end_date && break

        # Mass flux binary
        fname = _date_to_filename(pattern, date)
        _stage_file!(src_dir, dst_dir, fname) && (n += 1)

        # Surface binaries
        if !isempty(sfc_src) && !isempty(sfc_dst)
            for sp in sfc_patterns
                sfname = _date_to_filename(sp, date)
                _stage_file!(sfc_src, sfc_dst, sfname) && (n += 1)
            end
        end
    end
    return n
end

function _clean_behind!(dst_dir, pattern, sfc_dst, sfc_patterns,
                        current_date, keep_behind, start_date)
    n = 0
    # Clean dates before (current - keep_behind)
    cutoff = current_date - Day(keep_behind)
    date = start_date
    while date < cutoff
        fname = _date_to_filename(pattern, date)
        fpath = joinpath(dst_dir, fname)
        if isfile(fpath)
            rm(fpath)
            n += 1
        end

        # Also clean surface files
        if !isempty(sfc_dst)
            for sp in sfc_patterns
                sfpath = joinpath(sfc_dst, _date_to_filename(sp, date))
                if isfile(sfpath)
                    rm(sfpath)
                    n += 1
                end
            end
        end

        date += Day(1)
    end
    return n
end

function _dir_size_gb(dir)
    total = 0
    for f in readdir(dir; join=true)
        isfile(f) && (total += filesize(f))
    end
    return total / 1e9
end

# =====================================================================
# Progress writer for the model (call from run loop)
# =====================================================================

"""
    write_staging_progress(path, date)

Write the current simulation date to the staging progress file.
Called from the run loop so the staging daemon knows where we are.
Add to your run config: `[staging] progress_file = "/temp2/staging/.../progress.txt"`
"""
function write_staging_progress(path::String, date::Date)
    isempty(path) && return
    open(path, "w") do io
        println(io, Dates.format(date, "yyyy-mm-dd"))
    end
end

# =====================================================================
# Entry point
# =====================================================================

main()
