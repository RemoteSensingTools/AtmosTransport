#!/usr/bin/env julia
# Quick utility: merge lnsp from .lnsp_tmp files into ERA5 .nc files
# that are missing the lnsp variable.

using NCDatasets

dir = expanduser(get(ENV, "ERA5_DIR",
    "~/data/metDrivers/era5/era5_ml_10deg_20230601_20230630"))

for f in sort(readdir(dir))
    endswith(f, ".nc") || continue
    contains(f, "tmp") && continue
    fp = joinpath(dir, f)
    lnsp_tmp = fp * ".lnsp_tmp"

    ds = NCDataset(fp, "r")
    has_lnsp = haskey(ds, "lnsp")
    close(ds)

    if has_lnsp
        println("  $f: already has lnsp")
        continue
    end

    if !isfile(lnsp_tmp)
        println("  $f: NO lnsp and no .lnsp_tmp — SKIP")
        continue
    end

    println("  $f: merging lnsp from temp...")

    # Read lnsp from temp (shape: lon, lat, model_level=1, time)
    ds_lnsp = NCDataset(lnsp_tmp, "r")
    lnsp_data = Float32.(ds_lnsp["lnsp"][:, :, 1, :])  # squeeze level dim
    close(ds_lnsp)

    # Append to main file
    ds = NCDataset(fp, "a")
    outv = defVar(ds, "lnsp", Float32, ("longitude", "latitude", "valid_time"))
    outv.attrib["long_name"] = "Logarithm of surface pressure"
    outv.attrib["units"] = "~"
    outv[:, :, :] = lnsp_data
    close(ds)

    # Verify
    ds = NCDataset(fp, "r")
    println("    → lnsp added: $(haskey(ds, "lnsp")), shape=$(size(ds["lnsp"]))")
    close(ds)
end
