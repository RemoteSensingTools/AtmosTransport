#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Download GEOS-IT C180 physics collections (A3mstE, A1) for convection & PBL
#
# These are supplementary to the CTM_A1 (mass flux) files already downloaded
# by download_geosfp_cs_massflux.jl.
#
# Collections:
#   A3mstE — 3-hourly moist processes on edges (CMFMC: convective mass flux)
#   A1     — hourly assimilation (PBLH, USTAR, HFLUX, T2M, etc.)
#   A3dyn  — 3-hourly dynamics (DTRAIN: detrainment, U, V, OMEGA, RH)
#
# Environment variables:
#   GEOSFP_DATA_DIR   — output directory (default: ~/data/geosit_c180)
#   GEOSFP_START_DATE — start date YYYY-MM-DD (default: 2023-06-01)
#   GEOSFP_END_DATE   — end date YYYY-MM-DD   (default: 2023-06-30)
#   COLLECTIONS       — comma-separated list (default: "A3mstE,A1,A3dyn")
#
# Data source: http://geoschemdata.wustl.edu/ExtData/GEOS_C180/GEOS_IT/
# ---------------------------------------------------------------------------

using Dates

include(joinpath(@__DIR__, "download_utils.jl"))
using .DownloadUtils

const BASE_URL   = "http://geoschemdata.wustl.edu/ExtData/GEOS_C180/GEOS_IT"
const NC         = 180
const OUTPUT_DIR = get(ENV, "GEOSFP_DATA_DIR",
                       joinpath(homedir(), "data", "geosit_c180"))

# Parse collections to download
const COLLECTIONS = split(get(ENV, "COLLECTIONS", "A3mstE,A1,A3dyn,I3"), ",")

function build_url(date::Date, collection::AbstractString)
    datestr = Dates.format(date, "yyyymmdd")
    y = Dates.year(date)
    m = lpad(Dates.month(date), 2, '0')
    fname = "GEOSIT.$(datestr).$(collection).C$(NC).nc"
    return "$(BASE_URL)/$y/$m/$fname"
end

# download_file replaced by verified_download from DownloadUtils

function main()
    start_date = get(ENV, "GEOSFP_START_DATE", "2023-06-01") |> Date
    end_date   = get(ENV, "GEOSFP_END_DATE",   "2023-06-30") |> Date
    n_days = Dates.value(end_date - start_date + Day(1))
    n_total = n_days * length(COLLECTIONS)

    @info """
    GEOS-IT C180 Physics Download
    ==============================
    Collections: $(join(COLLECTIONS, ", "))
    Date range:  $start_date → $end_date ($n_days days)
    Files:       $n_total
    Output:      $OUTPUT_DIR
    Source:      $BASE_URL
    """

    n_done = 0
    n_fail = 0
    t0 = time()

    for date in start_date:Day(1):end_date
        datestr = Dates.format(date, "yyyymmdd")
        daydir = joinpath(OUTPUT_DIR, datestr)
        for coll in COLLECTIONS
            url = build_url(date, coll)
            fname = basename(url)
            dest = joinpath(daydir, fname)
            ok = verified_download(url, dest)
            ok ? (n_done += 1) : (n_fail += 1)
        end

        elapsed = time() - t0
        rate = n_done > 0 ? elapsed / n_done : Inf
        remaining = (n_total - n_done - n_fail) * rate
        @info "Day $datestr: $(n_done + n_fail) / $n_total done " *
              "($(n_fail) failed, ETA: $(round(remaining / 3600; digits=1)) h)"
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
    start_date = get(ENV, "GEOSFP_START_DATE", "2023-06-01") |> Date
    end_date   = get(ENV, "GEOSFP_END_DATE",   "2023-06-30") |> Date
    dates = start_date:Day(1):end_date

    @info "Verifying downloads in $OUTPUT_DIR ($start_date → $end_date)..."
    for coll in COLLECTIONS
        result = verify_downloads(
            OUTPUT_DIR, dates,
            datestr -> "GEOSIT.$(datestr).$(coll).C$(NC).nc";
            url_builder = datestr -> build_url(Date(datestr, dateformat"yyyymmdd"), coll))
        @info "  $(coll): $(length(result.ok)) OK, $(length(result.corrupt)) corrupt, $(length(result.missing)) missing"
        for f in result.corrupt
            @warn "    Corrupt: $f ($(filesize(f) ÷ 1_000_000) MB)"
        end
    end
else
    main()
end
