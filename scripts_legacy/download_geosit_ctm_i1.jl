#!/usr/bin/env julia
# ===========================================================================
# Download GEOS-IT CTM_I1 (hourly instantaneous PS + QV) from WashU archive
#
# CTM_I1 contains the same fields GCHP uses for SPHU1/SPHU2 and PS1/PS2
# in mass-flux advection mode. This gives us hourly QV at the same cadence
# as the mass fluxes in CTM_A1, eliminating the 3-hourly I3 temporal mismatch.
#
# Usage:
#   julia scripts/download_geosit_ctm_i1.jl [start_date] [end_date] [dest_dir]
#
# Defaults: Dec 1–22, 2021 (CATRINE period) → ~/data/geosit_c180_catrine/
# ===========================================================================

using Dates
using Downloads: download

const WASHU_BASE = "http://geoschemdata.wustl.edu/ExtData/GEOS_C180/GEOS_IT"

function download_ctm_i1(; start_date::Date = Date(2021, 12, 1),
                           end_date::Date   = Date(2021, 12, 22),
                           dest_root::String = expanduser("~/data/geosit_c180_catrine"))
    dates = start_date:Day(1):end_date
    n_total = length(dates)
    n_done = 0
    n_skip = 0

    @info "Downloading GEOS-IT CTM_I1 (hourly QV+PS)" start_date end_date n_days=n_total dest=dest_root

    for d in dates
        yyyy = Dates.format(d, "yyyy")
        mm   = Dates.format(d, "mm")
        ds   = Dates.format(d, "yyyymmdd")

        dest_dir = joinpath(dest_root, ds)
        mkpath(dest_dir)

        filename = "GEOSIT.$(ds).CTM_I1.C180.nc"
        dest_path = joinpath(dest_dir, filename)

        if isfile(dest_path) && filesize(dest_path) > 100_000_000  # >100 MB = likely complete
            n_skip += 1
            @info "  Skip (exists): $filename"
            continue
        end

        url = "$WASHU_BASE/$yyyy/$mm/$filename"
        @info "  Downloading: $filename (~851 MB)"

        try
            download(url, dest_path)
            sz_mb = round(filesize(dest_path) / 1e6; digits=1)
            n_done += 1
            @info "  Done: $filename ($sz_mb MB) [$n_done+$n_skip/$n_total]"
        catch e
            @warn "  FAILED: $filename — $e"
            isfile(dest_path) && rm(dest_path)
        end
    end

    @info "Download complete: $n_done downloaded, $n_skip skipped, $(n_total - n_done - n_skip) failed"
end

# Parse CLI args
if length(ARGS) >= 2
    sd = Date(ARGS[1])
    ed = Date(ARGS[2])
    dest = length(ARGS) >= 3 ? ARGS[3] : expanduser("~/data/geosit_c180_catrine")
    download_ctm_i1(; start_date=sd, end_date=ed, dest_root=dest)
else
    download_ctm_i1()
end
