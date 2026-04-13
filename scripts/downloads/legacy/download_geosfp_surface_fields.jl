#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Download GEOS-FP 0.25° × 0.3125° physics collections (A1, A3mstE)
#
# These lat-lon fields are regridded to C720 cubed-sphere by
# preprocess_geosfp_surface_to_cs.jl before use in the model.
#
# Collections:
#   A1     — hourly surface fields (PBLH, USTAR, HFLUX, T2M, etc.)
#   A3mstE — 3-hourly moist processes on edges (CMFMC: convective mass flux)
#
# Environment variables:
#   GEOSFP_DATA_DIR   — output directory (default: ~/data/geosfp_025)
#   GEOSFP_START_DATE — start date YYYY-MM-DD (default: 2024-06-01)
#   GEOSFP_END_DATE   — end date YYYY-MM-DD   (default: 2024-06-30)
#   COLLECTIONS       — comma-separated list  (default: "A1,A3mstE")
#
# Data source: https://s3.amazonaws.com/gcgrid/GEOS_0.25x0.3125/GEOS_FP/ (public)
# ---------------------------------------------------------------------------

using Dates
using Printf

include(joinpath(@__DIR__, "download_utils.jl"))
using .DownloadUtils

const S3_BASE_URL = "https://s3.amazonaws.com/gcgrid/GEOS_0.25x0.3125/GEOS_FP"

const OUTPUT_DIR  = get(ENV, "GEOSFP_DATA_DIR",
                        joinpath(homedir(), "data", "geosfp_025"))
const COLLECTIONS = split(get(ENV, "COLLECTIONS", "A1,A3mstE"), ",")

function build_url(date::Date, collection::AbstractString)
    datestr = Dates.format(date, "yyyymmdd")
    y = Dates.year(date)
    m = lpad(Dates.month(date), 2, '0')
    fname = "GEOSFP.$(datestr).$(collection).025x03125.nc"
    return "$(S3_BASE_URL)/$y/$m/$fname"
end

## download_file replaced by verified_download from DownloadUtils

function main()
    start_date = get(ENV, "GEOSFP_START_DATE", "2024-06-01") |> Date
    end_date   = get(ENV, "GEOSFP_END_DATE",   "2024-06-30") |> Date
    n_days  = Dates.value(end_date - start_date + Day(1))
    n_total = n_days * length(COLLECTIONS)

    @info """
    GEOS-FP 0.25° Surface Field Download
    =====================================
    Collections: $(join(COLLECTIONS, ", "))
    Date range:  $start_date → $end_date ($n_days days)
    Files:       $n_total
    Output:      $OUTPUT_DIR
    Source:      $(S3_BASE_URL)/
    """

    n_done = 0
    n_fail = 0
    t0 = time()

    for date in start_date:Day(1):end_date
        datestr = Dates.format(date, "yyyymmdd")
        for coll in COLLECTIONS
            url   = build_url(date, coll)
            fname = basename(url)
            dest  = joinpath(OUTPUT_DIR, fname)
            ok = verified_download(url, dest)
            ok ? (n_done += 1) : (n_fail += 1)
        end

        elapsed = time() - t0
        rate = n_done > 0 ? elapsed / n_done : Inf
        remaining = (n_total - n_done - n_fail) * rate
        @info @sprintf("Day %s: %d/%d done (%d failed, ETA: %.1f h)",
                        datestr, n_done + n_fail, n_total, n_fail,
                        remaining / 3600)
    end

    elapsed = time() - t0
    @info """
    Download complete.
    ==================
    Collections: $(join(COLLECTIONS, ", "))
    Succeeded:   $n_done / $n_total
    Failed:      $n_fail
    Elapsed:     $(round(elapsed / 3600; digits=1)) hours
    Output:      $OUTPUT_DIR
    """
end

if "--verify" in ARGS
    start_date = get(ENV, "GEOSFP_START_DATE", "2024-06-01") |> Date
    end_date   = get(ENV, "GEOSFP_END_DATE",   "2024-06-30") |> Date
    dates = start_date:Day(1):end_date

    @info "Verifying downloads in $OUTPUT_DIR ($start_date → $end_date)..."
    for coll in COLLECTIONS
        result = verify_downloads(
            OUTPUT_DIR, dates,
            datestr -> "GEOSFP.$(datestr).$(coll).025x03125.nc";
            url_builder = datestr -> build_url(Date(datestr, dateformat"yyyymmdd"), coll))
        @info "  $(coll): $(length(result.ok)) OK, $(length(result.corrupt)) corrupt, $(length(result.missing)) missing"
        for f in result.corrupt
            @warn "    Corrupt: $f ($(filesize(f) ÷ 1_000_000) MB)"
        end
    end
else
    main()
end
