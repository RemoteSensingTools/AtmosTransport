using NCDatasets
dir = expanduser("~/data/metDrivers/era5/era5_ml_10deg_20230601_20230630")
missing_files = String[]
for f in sort(readdir(dir))
    if endswith(f, ".nc") && !contains(f, "tmp") && !contains(f, "massflux")
        fp = joinpath(dir, f)
        ds = NCDataset(fp, "r")
        has_lnsp = haskey(ds, "lnsp")
        close(ds)
        if !has_lnsp
            push!(missing_files, f)
            lnsp_tmp = fp * ".lnsp_tmp"
            if isfile(lnsp_tmp)
                println("Fixing: $f")
                ds_lnsp = NCDataset(lnsp_tmp, "r")
                # lnsp is (lon, lat, model_level, time) — squeeze out model_level
                raw = Array(ds_lnsp["lnsp"])  # read all dims
                close(ds_lnsp)
                # Squeeze singleton model_level dim (dim 3)
                lnsp_data = dropdims(raw; dims=3)  # now (lon, lat, time)
                ds = NCDataset(fp, "a")
                outv = defVar(ds, "lnsp", Float32, ("longitude", "latitude", "valid_time"))
                outv[:, :, :] = Float32.(lnsp_data)
                close(ds)
                println("  Done: added lnsp $(size(lnsp_data))")
            else
                println("MISSING lnsp and NO .lnsp_tmp: $f")
            end
        end
    end
end
println("\n$(length(missing_files)) files were missing lnsp")
