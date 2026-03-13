#!/usr/bin/env julia
# ===========================================================================
# Download CATRINE GeosChem reference data (instantaneous 3D fields)
#
# Source: NASA NAS CATRINE archive
# URL:    https://data.nas.nasa.gov/catrine/CATRINE_C180/instant/
# Files:  GEOSChem.CATRINE_inst.YYYYMMDD_HHMMz.nc4
#         (8 per day: 0000z, 0300z, ..., 2100z; ~552 MB each)
#
# Usage:
#   julia scripts/downloads/download_catrine_geoschem.jl [--dry-run]
#
# Skips files that already exist with correct size (Content-Length check).
# Default period: Dec 2021. Override with START_DATE / END_DATE env vars.
# ===========================================================================

include(joinpath(@__DIR__, "download_utils.jl"))
using .DownloadUtils
using Dates

const BASE_URL = "https://data.nas.nasa.gov/catrine/CATRINE_C180/instant"
const DEST_DIR = joinpath(homedir(), "data", "AtmosTransport", "catrine-geoschem-runs")
const HOURS = ["0000", "0300", "0600", "0900", "1200", "1500", "1800", "2100"]

function main()
    dry_run = "--dry-run" in ARGS

    start_date = Date(get(ENV, "START_DATE", "2021-12-01"))
    end_date   = Date(get(ENV, "END_DATE",   "2021-12-31"))

    mkpath(DEST_DIR)

    # Build file list
    files = Tuple{String, String}[]  # (url, dest)
    for d in start_date:Day(1):end_date
        datestr = Dates.format(d, "yyyymmdd")
        for hh in HOURS
            fname = "GEOSChem.CATRINE_inst.$(datestr)_$(hh)z.nc4"
            url   = "$BASE_URL/$fname"
            dest  = joinpath(DEST_DIR, fname)
            push!(files, (url, dest))
        end
    end

    # Count existing
    existing = count(f -> isfile(f[2]), files)
    needed   = length(files) - existing

    println("=" ^ 70)
    println("CATRINE GeosChem Reference Data Download")
    println("  Period:      $(start_date) to $(end_date)")
    println("  Total files: $(length(files))")
    println("  Existing:    $(existing)")
    println("  To download: $(needed)")
    println("  Destination: $(DEST_DIR)")
    println("=" ^ 70)
    println()

    if needed == 0
        println("All files already downloaded.")
        return
    end

    if dry_run
        println("[dry-run] Would download $needed files. Exiting.")
        return
    end

    # Download with progress
    success = 0
    failed  = 0
    for (url, dest) in files
        if isfile(dest)
            continue
        end
        print("[$(success + failed + 1)/$needed] ")
        if verified_download(url, dest; max_retries=3)
            success += 1
        else
            failed += 1
        end
    end

    println()
    println("=" ^ 70)
    println("Done: $success downloaded, $failed failed, $existing skipped")
    println("=" ^ 70)
end

main()
