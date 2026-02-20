#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Download GEOS-FP native cubed-sphere mass flux data for June 2024
#
# Collections:
#   f5295_fp.tavg3_3d_mst_Cp — MFXC, MFYC (3-hourly time-averaged mass fluxes)
#   f5295_fp.inst3_3d_asm_Cp — DELP (3-hourly instantaneous pressure thickness)
#
# Data source: NASA NCCS portal (no authentication required)
# ---------------------------------------------------------------------------

using Dates
using Downloads

const BASE_URL = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das"
const OUTPUT_DIR = get(ENV, "GEOSFP_DATA_DIR",
                       joinpath(homedir(), "data", "geosfp_cs"))

function build_url(date::Date, hour::Int, collection::String;
                   stream::String = "f5295_fp")
    datestr = Dates.format(date, "yyyymmdd")
    suffix = collection == "inst3_3d_asm_Cp" ? "$(lpad(hour, 2, '0'))00" :
                                                "$(lpad(hour, 2, '0'))30"
    y = Dates.format(date, "yyyy")
    m = Dates.format(date, "mm")
    d = Dates.format(date, "dd")
    return "$BASE_URL/Y$y/M$m/D$d/$(stream).$(collection).$(datestr)_$(suffix)z.nc4"
end

function download_file(url::String, dest::String)
    if isfile(dest)
        @info "Already exists: $(basename(dest))"
        return
    end
    @info "Downloading $(basename(dest))..."
    mkpath(dirname(dest))
    Downloads.download(url, dest)
    @info "  → $(filesize(dest) ÷ 1_000_000) MB"
end

function main()
    start_date = Date(2024, 6, 1)
    end_date   = Date(2024, 6, 30)

    collections = [
        "tavg3_3d_mst_Cp",   # MFXC, MFYC
        "inst3_3d_asm_Cp",   # DELP
    ]
    hours_3h = 0:3:21  # 3-hourly: 0, 3, 6, 9, 12, 15, 18, 21

    n_total = Dates.value(end_date - start_date + Day(1)) * length(hours_3h) * length(collections)
    @info "Downloading $n_total files to $OUTPUT_DIR"

    n = 0
    for date in start_date:Day(1):end_date
        datestr = Dates.format(date, "yyyymmdd")
        for hour in hours_3h, coll in collections
            url = build_url(date, hour, coll)
            fname = basename(url)
            dest = joinpath(OUTPUT_DIR, datestr, fname)
            try
                download_file(url, dest)
            catch e
                @warn "Failed to download $fname: $e"
            end
            n += 1
            n % 50 == 0 && @info "Progress: $n / $n_total"
        end
    end

    @info "Done. Downloaded to $OUTPUT_DIR"
end

main()
