#!/usr/bin/env julia
# ===========================================================================
# CATRINE D7.1 — Data Download Script
#
# Downloads all required input data for the CATRINE intercomparison protocol.
# Creates the directory structure under ~/data/catrine/.
#
# Data sources:
#   1. CAMS CO2 fluxes (OCO-2 disaggregated)  — IPSL THREDDS
#   2. GridFEDv2024.1 fossil CO2               — Zenodo
#   3. EDGAR v8.0 SF6                          — EDGAR JRC
#   4. NOAA SF6 growth rates                   — NOAA GML
#   5. Zhang et al. 2021 222Rn                 — Harvard FTP
#   6. Initial conditions (CO2, SF6)           — IPSL THREDDS
#
# Usage:
#   julia scripts/download_catrine_data.jl [--dry-run]
#
# Some datasets require manual download due to authentication or size.
# The script will print instructions for those cases.
# ===========================================================================

using Downloads

const CATRINE_DIR = joinpath(homedir(), "data", "catrine")

# THREDDS catalog base URL
const THREDDS_BASE = "https://thredds-su.ipsl.fr/thredds/fileServer/CATRINE"

# Data source URLs
const URLS = Dict(
    # NOAA SF6 growth rates (small text file, freely available)
    "sf6_growth" => (
        url = "https://gml.noaa.gov/webdata/ccgg/trends/sf6/sf6_gr_gl.txt",
        dest = "sf6_gr_gl.txt",
        desc = "NOAA SF6 global growth rates"
    ),
)

function main()
    dry_run = "--dry-run" in ARGS

    mkpath(CATRINE_DIR)
    mkpath(joinpath(CATRINE_DIR, "ZHANG_Rn222"))

    println("=" ^ 70)
    println("CATRINE D7.1 Data Download")
    println("Target directory: $CATRINE_DIR")
    println("=" ^ 70)
    println()

    # --- 1. NOAA SF6 growth rates (auto-download) ---
    println("1. NOAA SF6 growth rates")
    sf6_url  = "https://gml.noaa.gov/webdata/ccgg/trends/sf6/sf6_gr_gl.txt"
    sf6_dest = joinpath(CATRINE_DIR, "sf6_gr_gl.txt")
    _download(sf6_url, sf6_dest; dry_run)
    println()

    # --- 2. CAMS CO2 fluxes ---
    println("2. CAMS CO2 fluxes (OCO-2 disaggregated)")
    println("   Source: IPSL THREDDS server")
    println("   URL: https://thredds-su.ipsl.fr/thredds/catalog/CATRINE/catalog.html")
    println("   >> Browse the catalog and download the CO2 flux files to:")
    println("   >> $CATRINE_DIR/cams_co2_flux.nc")
    println()

    # --- 3. GridFED fossil CO2 ---
    println("3. GridFEDv2024.1 fossil CO2")
    println("   Source: Zenodo")
    println("   URL: https://zenodo.org/records/8386803")
    println("   >> Download the monthly total CO2 emissions file to:")
    println("   >> $CATRINE_DIR/GridFEDv2024.1_monthly.nc")
    println()

    # --- 4. EDGAR v8.0 SF6 ---
    println("4. EDGAR v8.0 SF6 annual emissions")
    println("   Source: EDGAR JRC")
    println("   URL: https://edgar.jrc.ec.europa.eu/dataset_ghg80")
    println("   >> Download 'SF6 TOTALS' for year 2022 to:")
    println("   >> $CATRINE_DIR/v8.0_FT2022_GHG_SF6_2022_TOTALS_emi.nc")
    println()

    # --- 5. Zhang Rn222 ---
    println("5. Zhang et al. 2021 222Rn emissions")
    println("   Source: Harvard FTP (HEMCO)")
    rn_base = "http://ftp.as.harvard.edu/gcgrid/data/ExtData/HEMCO/ZHANG_Rn222/v2021-11/"
    rn_dest_dir = joinpath(CATRINE_DIR, "ZHANG_Rn222")
    rn_files = ["222Rn_Zhang_Liu_2011_0.5x0.5_$(lpad(m, 2, '0')).nc" for m in 1:12]
    for f in rn_files
        _download(rn_base * f, joinpath(rn_dest_dir, f); dry_run, warn_on_fail=true)
    end
    println("   Note: File names may differ. Check the FTP directory if downloads fail:")
    println("   $rn_base")
    println()

    # --- 6. Initial conditions ---
    println("6. Initial conditions (CO2, SF6 from LSCE inversions)")
    println("   Source: IPSL THREDDS server")
    println("   URL: https://thredds-su.ipsl.fr/thredds/catalog/CATRINE/catalog.html")
    println("   >> Download the initial conditions file (1 Dec 2021) to:")
    println("   >> $CATRINE_DIR/initial_conditions_20211201.nc")
    println()

    println("=" ^ 70)
    println("Download complete. Files requiring manual download are listed above.")
    println("After downloading, verify config paths in:")
    println("  config/runs/catrine_era5.toml")
    println("  config/runs/catrine_geosfp_cs.toml")
    println("=" ^ 70)
end

function _download(url, dest; dry_run=false, warn_on_fail=false)
    if isfile(dest)
        println("   [skip] $(basename(dest)) already exists")
        return
    end
    if dry_run
        println("   [dry-run] would download: $url")
        println("             to: $dest")
        return
    end
    print("   Downloading $(basename(dest))... ")
    try
        Downloads.download(url, dest)
        sz = round(filesize(dest) / 1024, digits=1)
        println("OK ($(sz) KB)")
    catch e
        if warn_on_fail
            println("FAILED ($(e))")
            println("   >> Manual download may be required: $url")
        else
            rethrow(e)
        end
    end
end

main()
