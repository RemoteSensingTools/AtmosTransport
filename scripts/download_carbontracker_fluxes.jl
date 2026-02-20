#!/usr/bin/env julia
# ===========================================================================
# Download CarbonTracker CT-NRT flux files from NOAA GML FTP server.
#
# CT-NRT provides 3-hourly surface CO2 fluxes at 1x1 deg:
#   - Biosphere NEE (optimized)
#   - Ocean air-sea exchange (optimized)
#   - Fire emissions (imposed, from GFED4.1s)
#   - Fossil fuel emissions (imposed, Miller dataset)
#
# Files: daily NetCDF, ~2 MB each, ~730 MB/year
#
# Usage:
#   julia --project=. scripts/download_carbontracker_fluxes.jl
#
# Environment variables:
#   CT_VERSION — CT-NRT version (default: CT-NRT.v2025-1)
#   YEAR       — year to download (default: 2024)
#   DATA_DIR   — output directory
# ===========================================================================

using Dates

const CT_VERSION = get(ENV, "CT_VERSION", "CT-NRT.v2025-1")
const YEAR       = parse(Int, get(ENV, "YEAR", "2024"))
const DATA_DIR   = expanduser(get(ENV, "DATA_DIR",
    "~/data/emissions/carbontracker/$(CT_VERSION)"))
const BASE_URL   = "https://gml.noaa.gov/aftp/products/carbontracker/co2/$(CT_VERSION)/fluxes/optimized/flux1x1"

function main()
    mkpath(DATA_DIR)

    println("CarbonTracker CT-NRT flux download:")
    println("  Version: $CT_VERSION")
    println("  Year: $YEAR")
    println("  Output: $DATA_DIR")
    println("  Base URL: $BASE_URL")

    start_date = Date(YEAR, 1, 1)
    end_date   = Date(YEAR, 12, 31)
    days = collect(start_date:Day(1):end_date)

    println("  Days to download: $(length(days))")

    n_downloaded = 0
    n_skipped = 0
    n_failed = 0

    for day in days
        datestr = Dates.format(day, "yyyy-mm-dd")
        filename = "$(CT_VERSION).flux1x1.$(datestr).nc"
        outpath = joinpath(DATA_DIR, filename)

        if isfile(outpath) && filesize(outpath) > 1000
            n_skipped += 1
            continue
        end

        url = "$BASE_URL/$filename"
        println("  Downloading: $filename")
        try
            run(`wget -q -O $outpath $url`)
            if isfile(outpath) && filesize(outpath) > 1000
                n_downloaded += 1
            else
                @warn "Download produced empty/small file: $outpath"
                rm(outpath; force=true)
                n_failed += 1
            end
        catch e
            @warn "Failed to download $filename: $e"
            rm(outpath; force=true)
            n_failed += 1
        end
    end

    println("\nDownload summary:")
    println("  Downloaded: $n_downloaded")
    println("  Skipped (existing): $n_skipped")
    println("  Failed: $n_failed")

    existing = filter(f -> endswith(f, ".nc"), readdir(DATA_DIR))
    total_mb = sum(filesize(joinpath(DATA_DIR, f)) for f in existing; init=0) / 1e6
    println("  Total files: $(length(existing)), size: $(round(total_mb, digits=1)) MB")
end

main()
