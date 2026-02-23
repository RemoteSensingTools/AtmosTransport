#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Download GEOS-FP native cubed-sphere CTM data for mass-flux advection
#
# Data source: Washington University GEOS-Chem Support Team archive
#   http://geoschemdata.wustl.edu/ExtData/GEOS_C720/GEOS_FP_Native/
#
# Collection: tavg_1hr_ctm_c0720_v72
#   Contains: MFXC (east-west mass flux), MFYC (north-south mass flux),
#             DELP (pressure thickness), SPHU (specific humidity),
#             CYC (vertical Courant number)
#
# File naming: GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.YYYYMMDD_HHMM.V01.nc4
#   Hourly time-averaged, timestamps at HH30 (center of hour)
#   24 files per day, ~2.7 GB each → ~65 GB/day
#
# Reference: Martin et al. (2022), GMD, doi:10.5194/gmd-15-8325-2022
# Native cubed-sphere mass fluxes as used by GCHP.
# ---------------------------------------------------------------------------

using Dates
using Downloads

const BASE_URL = "http://geoschemdata.wustl.edu/ExtData/GEOS_C720/GEOS_FP_Native"
const OUTPUT_DIR = get(ENV, "GEOSFP_DATA_DIR",
                       joinpath(homedir(), "data", "geosfp_cs"))

function build_tavg_url(date::Date, hour::Int)
    datestr = Dates.format(date, "yyyymmdd")
    timestamp = lpad(hour, 2, '0') * "30"
    y = Dates.format(date, "yyyy")
    m = lpad(Dates.month(date), 2, '0')
    d = lpad(Dates.day(date), 2, '0')
    return "$BASE_URL/Y$(Dates.year(date))/M$m/D$d/" *
           "GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.$(datestr)_$(timestamp).V01.nc4"
end

function download_file(url::String, dest::String; max_retries::Int=3)
    if isfile(dest) && filesize(dest) > 1_000_000
        @info "Already exists: $(basename(dest)) ($(filesize(dest) ÷ 1_000_000) MB)"
        return true
    end
    mkpath(dirname(dest))
    for attempt in 1:max_retries
        try
            @info "Downloading $(basename(dest)) (attempt $attempt)..."
            Downloads.download(url, dest)
            sz = filesize(dest) ÷ 1_000_000
            @info "  → $sz MB"
            return true
        catch e
            @warn "Attempt $attempt failed: $e"
            isfile(dest) && rm(dest; force=true)
            attempt < max_retries && sleep(5 * attempt)
        end
    end
    @error "Failed after $max_retries attempts: $(basename(dest))"
    return false
end

function main()
    start_date = get(ENV, "GEOSFP_START_DATE", "2024-06-01") |> Date
    end_date   = get(ENV, "GEOSFP_END_DATE",   "2024-06-30") |> Date
    hours      = 0:23

    n_days = Dates.value(end_date - start_date + Day(1))
    n_total = n_days * length(hours)
    est_gb = n_total * 2.7

    @info """
    GEOS-FP C720 Cubed-Sphere Download
    ===================================
    Date range: $start_date → $end_date ($n_days days)
    Files: $n_total (hourly tavg_1hr_ctm)
    Estimated size: $(round(est_gb; digits=0)) GB
    Output: $OUTPUT_DIR
    Source: geoschemdata.wustl.edu (Washington University)

    Note: June 16-18, 2024 may be missing from the archive.
    """

    n_done = 0
    n_fail = 0
    t0 = time()
    for date in start_date:Day(1):end_date
        datestr = Dates.format(date, "yyyymmdd")
        daydir = joinpath(OUTPUT_DIR, datestr)
        for hour in hours
            url = build_tavg_url(date, hour)
            fname = basename(url)
            dest = joinpath(daydir, fname)
            ok = download_file(url, dest)
            ok ? (n_done += 1) : (n_fail += 1)

            if (n_done + n_fail) % 24 == 0
                elapsed = time() - t0
                rate = n_done > 0 ? elapsed / n_done : Inf
                remaining = (n_total - n_done - n_fail) * rate
                @info "Progress: $(n_done + n_fail) / $n_total " *
                      "($(n_fail) failed, ETA: $(round(remaining / 3600; digits=1)) h)"
            end
        end
    end

    elapsed = time() - t0
    @info """
    Download complete.
    ==================
    Succeeded: $n_done / $n_total
    Failed:    $n_fail
    Elapsed:   $(round(elapsed / 3600; digits=1)) hours
    Output:    $OUTPUT_DIR
    """
end

main()
