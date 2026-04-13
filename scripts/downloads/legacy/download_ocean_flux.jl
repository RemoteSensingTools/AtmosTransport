#!/usr/bin/env julia
# ===========================================================================
# Download Jena CarboScope ocean CO2 flux data.
#
# The oc_v2024 dataset provides daily ocean-atmosphere CO2 flux on a
# 1x1 deg grid in NetCDF format. Units: PgC/yr per grid cell.
#
# Source: https://www.bgc-jena.mpg.de/CarboScope/oc/
# Reference: Rödenbeck et al. (2013)
#
# Usage:
#   julia --project=. scripts/download_ocean_flux.jl
#
# Environment variables:
#   OC_VERSION — CarboScope version (default: oc_v2024)
#   DATA_DIR   — output directory
# ===========================================================================

const OC_VERSION = get(ENV, "OC_VERSION", "oc_v2024")
const DATA_DIR   = expanduser(get(ENV, "DATA_DIR",
    "~/data/emissions/jena_carboscope"))

# Known download URLs for CarboScope ocean products
const DOWNLOAD_URLS = Dict(
    "oc_v2024" => "https://www.bgc-jena.mpg.de/CarboScope/oc/oc_v2024/oc_v2024.nc",
    "oc_v2023" => "https://www.bgc-jena.mpg.de/CarboScope/oc/oc_v2023/oc_v2023.nc",
)

function main()
    mkpath(DATA_DIR)

    println("Jena CarboScope ocean CO2 flux download:")
    println("  Version: $OC_VERSION")
    println("  Output: $DATA_DIR")

    url = get(DOWNLOAD_URLS, OC_VERSION, nothing)
    if url === nothing
        # Try constructing URL from pattern
        url = "https://www.bgc-jena.mpg.de/CarboScope/oc/$(OC_VERSION)/$(OC_VERSION).nc"
        @warn "URL not in known list, trying: $url"
    end

    outfile = joinpath(DATA_DIR, "$(OC_VERSION).nc")

    if isfile(outfile) && filesize(outfile) > 10000
        sz = round(filesize(outfile) / 1e6, digits=1)
        println("  Already exists: $outfile ($(sz) MB)")
        return
    end

    println("  Downloading: $url")
    t0 = time()
    try
        run(`wget -q --show-progress -O $outfile $url`)
        elapsed = round((time() - t0) / 60, digits=1)
        if isfile(outfile) && filesize(outfile) > 10000
            sz = round(filesize(outfile) / 1e6, digits=1)
            println("  Done: $outfile ($(sz) MB, $(elapsed) min)")
        else
            @warn "Download may have failed — file is small or empty"
            rm(outfile; force=true)
        end
    catch e
        @warn "Download failed: $e"
        rm(outfile; force=true)

        println("\nManual download instructions:")
        println("  1. Visit https://www.bgc-jena.mpg.de/CarboScope/oc/")
        println("  2. Select version: $OC_VERSION")
        println("  3. Download the NetCDF file")
        println("  4. Save to: $outfile")
    end
end

main()
