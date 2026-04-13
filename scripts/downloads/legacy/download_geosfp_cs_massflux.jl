#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Download GEOS native cubed-sphere CTM data for mass-flux advection
#
# Supports multiple products from the Washington University archive:
#   geosfp_c720 — GEOS-FP C720 (~12.5 km), March 2021–present
#                  Hourly files, ~2.7 GB each, 24 files/day
#   geosit_c180 — GEOS-IT C180 (~50 km),   1998–2023
#                  Daily files, ~4.2 GB each, 1 file/day
#
# Environment variables:
#   GEOSFP_PRODUCT    — product key (default: "geosfp_c720")
#   GEOSFP_DATA_DIR   — output directory (default: ~/data/geosfp_cs)
#   GEOSFP_START_DATE — start date YYYY-MM-DD (default: 2024-06-01)
#   GEOSFP_END_DATE   — end date YYYY-MM-DD   (default: 2024-06-30)
#
# Reference: Martin et al. (2022), GMD, doi:10.5194/gmd-15-8325-2022
# ---------------------------------------------------------------------------

using Dates

include(joinpath(@__DIR__, "download_utils.jl"))
using .DownloadUtils

# Product registry (mirrors GEOS_CS_PRODUCTS in geosfp_cubed_sphere_reader.jl)
const PRODUCTS = Dict(
    "geosfp_c720" => (base_url   = "http://geoschemdata.wustl.edu/ExtData/GEOS_C720",
                      collection = "GEOS_FP_Native",
                      Nc         = 720,
                      layout     = :hourly,
                      est_gb     = 2.7),
    "geosit_c180" => (base_url   = "http://geoschemdata.wustl.edu/ExtData/GEOS_C180",
                      collection = "GEOS_IT",
                      Nc         = 180,
                      layout     = :daily,
                      est_gb     = 4.2),
)

const PRODUCT    = get(ENV, "GEOSFP_PRODUCT", "geosfp_c720")
const OUTPUT_DIR = get(ENV, "GEOSFP_DATA_DIR",
                       joinpath(homedir(), "data", "geosfp_cs"))

function build_url(date::Date, hour::Int, info)
    datestr = Dates.format(date, "yyyymmdd")
    y = Dates.year(date)
    m = lpad(Dates.month(date), 2, '0')
    d = lpad(Dates.day(date), 2, '0')
    if info.layout === :hourly
        timestamp = lpad(hour, 2, '0') * "30"
        fname = "GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.$(datestr)_$(timestamp).V01.nc4"
        return "$(info.base_url)/$(info.collection)/Y$y/M$m/D$d/$fname"
    else  # :daily
        fname = "GEOSIT.$(datestr).CTM_A1.C$(info.Nc).nc"
        return "$(info.base_url)/$(info.collection)/$y/$m/$fname"
    end
end

# download_file replaced by verified_download from DownloadUtils

function main()
    haskey(PRODUCTS, PRODUCT) || error(
        "Unknown product '$PRODUCT'. Available: $(join(keys(PRODUCTS), ", "))")
    info = PRODUCTS[PRODUCT]

    start_date = get(ENV, "GEOSFP_START_DATE", "2024-06-01") |> Date
    end_date   = get(ENV, "GEOSFP_END_DATE",   "2024-06-30") |> Date

    n_days = Dates.value(end_date - start_date + Day(1))

    if info.layout === :hourly
        hours = 0:23
        n_total = n_days * 24
        est_gb = n_total * info.est_gb
    else
        hours = [0]  # one file per day
        n_total = n_days
        est_gb = n_total * info.est_gb
    end

    @info """
    GEOS Cubed-Sphere Download
    ==========================
    Product:    $PRODUCT (C$(info.Nc), $(info.layout))
    Date range: $start_date → $end_date ($n_days days)
    Files: $n_total
    Estimated size: $(round(est_gb; digits=1)) GB
    Output: $OUTPUT_DIR
    Source: $(info.base_url)
    """

    n_done = 0
    n_fail = 0
    t0 = time()
    for date in start_date:Day(1):end_date
        datestr = Dates.format(date, "yyyymmdd")
        daydir = joinpath(OUTPUT_DIR, datestr)
        for hour in hours
            url = build_url(date, hour, info)
            fname = basename(url)
            dest = joinpath(daydir, fname)
            ok = verified_download(url, dest)
            ok ? (n_done += 1) : (n_fail += 1)

            if (n_done + n_fail) % max(1, length(hours)) == 0
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
    Product:   $PRODUCT (C$(info.Nc))
    Succeeded: $n_done / $n_total
    Failed:    $n_fail
    Elapsed:   $(round(elapsed / 3600; digits=1)) hours
    Output:    $OUTPUT_DIR
    """
end

if "--verify" in ARGS
    # Verify-only mode: scan existing downloads and report status
    info = PRODUCTS[PRODUCT]
    start_date = get(ENV, "GEOSFP_START_DATE", "2024-06-01") |> Date
    end_date   = get(ENV, "GEOSFP_END_DATE",   "2024-06-30") |> Date
    dates = start_date:Day(1):end_date

    @info "Verifying downloads in $OUTPUT_DIR ($start_date → $end_date)..."
    result = verify_downloads(
        OUTPUT_DIR, dates,
        datestr -> basename(build_url(Date(datestr, dateformat"yyyymmdd"), 0, info));
        url_builder = datestr -> build_url(Date(datestr, dateformat"yyyymmdd"), 0, info))

    @info "  OK:      $(length(result.ok))"
    isempty(result.corrupt) || @warn "  Corrupt: $(length(result.corrupt))"
    for f in result.corrupt
        @warn "    $f ($(filesize(f) ÷ 1_000_000) MB)"
    end
    isempty(result.missing) || @warn "  Missing: $(length(result.missing))"
    for f in result.missing
        @warn "    $f"
    end
else
    main()
end
