#!/usr/bin/env julia
# ===========================================================================
# Download CAMS GFAS fire CO2 emissions via ADS API.
#
# GFAS v1.4 provides daily wildfire CO2 emissions at 0.1° resolution.
# Variable: "cofire" — wildfire flux of CO2 [kg/m²/s]
#
# Data source: Atmosphere Data Store (ADS)
#   https://ads.atmosphere.copernicus.eu/datasets/cams-global-fire-emissions-gfas
#
# Prerequisites:
#   - Python with cdsapi installed
#   - ~/.cdsapirc configured for ADS (url: https://ads.atmosphere.copernicus.eu/api)
#
# Usage:
#   julia --project=. scripts/download_gfas_fire.jl
#
# Environment variables:
#   YEAR     — year to download (default: 2024)
#   DATA_DIR — output directory
#   PYTHON   — path to python3 with cdsapi
# ===========================================================================

using Dates

const YEAR     = parse(Int, get(ENV, "YEAR", "2024"))
const DATA_DIR = expanduser(get(ENV, "DATA_DIR",
    "~/data/emissions/gfas/$(YEAR)"))
const PYTHON   = get(ENV, "PYTHON", "/opt/miniconda/bin/python3")

function main()
    mkpath(DATA_DIR)

    println("CAMS GFAS fire CO2 download:")
    println("  Year: $YEAR")
    println("  Output: $DATA_DIR")

    # Download month by month (ADS has size limits per request)
    for month in 1:12
        month_str = lpad(month, 2, '0')
        outfile = joinpath(DATA_DIR, "gfas_co2fire_$(YEAR)$(month_str).nc")

        if isfile(outfile) && filesize(outfile) > 1000
            sz = round(filesize(outfile) / 1e6, digits=1)
            println("  $(YEAR)-$(month_str): already exists ($(sz) MB)")
            continue
        end

        n_days = Dates.daysinmonth(YEAR, month)
        date_start = "$(YEAR)-$(month_str)-01"
        date_end = "$(YEAR)-$(month_str)-$(lpad(n_days, 2, '0'))"

        println("  $(YEAR)-$(month_str): requesting GFAS CO2 fire emissions...")

        # Build day list for the API
        day_list = join([lpad(d, 2, '0') for d in 1:n_days], "', '")

        python_script = """
import cdsapi
import os

output = "$outfile"
os.makedirs(os.path.dirname(output), exist_ok=True)

c = cdsapi.Client(url="https://ads.atmosphere.copernicus.eu/api")
c.retrieve(
    'cams-global-fire-emissions-gfas',
    {
        'date': '$date_start/$date_end',
        'variable': 'wildfire_flux_of_carbon_dioxide',
        'data_format': 'netcdf',
    },
    output
)
print(f"Downloaded: {output} ({os.path.getsize(output) / 1e6:.1f} MB)")
"""

        tmp = tempname() * ".py"
        write(tmp, python_script)
        try
            t0 = time()
            run(`$PYTHON $tmp`)
            elapsed = round((time() - t0) / 60, digits=1)
            if isfile(outfile)
                sz = round(filesize(outfile) / 1e6, digits=1)
                println("    Done: $(sz) MB in $(elapsed) min")
            end
        catch e
            @warn "Download failed for $(YEAR)-$(month_str): $e"
        finally
            rm(tmp; force=true)
        end
    end

    println("\nDownload summary:")
    existing = filter(f -> endswith(f, ".nc"), readdir(DATA_DIR))
    total_mb = sum(filesize(joinpath(DATA_DIR, f)) for f in existing; init=0) / 1e6
    println("  Files: $(length(existing)), total: $(round(total_mb, digits=1)) MB")
end

main()
