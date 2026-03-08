#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Compare AtmosTransport output against GEOS-Chem CATRINE snapshots
#
# GEOS-Chem snapshots contain:
#   SpeciesConcVV_{CO2,FossilCO2,SF6,Rn222}  — dry mole fractions [mol/mol dry]
#   Met_AD         — dry air mass [kg]
#   Met_PSC2WET/DRY — wet/dry surface pressure [hPa]
#   ColumnMass_*   — column-integrated mass [~ mol/mol * kg_air/m²]
#
# Usage:
#   julia --project=. scripts/diagnostics/compare_vs_geoschem.jl [output_dir] [gc_dir]
#
# Defaults:
#   output_dir = ~/data/AtmosTransport/catrine/output/geosit_c180/
#   gc_dir     = ~/data/AtmosTransport/catrine/geos-chem/
# ---------------------------------------------------------------------------

using NCDatasets, Dates, Printf, Statistics

const OUTPUT_DIR = get(ARGS, 1,
    expanduser("~/data/AtmosTransport/catrine/output/geosit_c180"))
const GC_DIR = get(ARGS, 2,
    expanduser("~/data/AtmosTransport/catrine/geos-chem"))

# Map our variable names to GC variable names
const SPECIES_MAP = [
    ("co2",        "SpeciesConcVV_CO2",       "CO2"),
    ("fossil_co2", "SpeciesConcVV_FossilCO2", "Fossil CO2"),
    ("sf6",        "SpeciesConcVV_SF6",        "SF6"),
    ("rn222",      "SpeciesConcVV_Rn222",      "Rn222"),
]

# GC snapshot timestamps and corresponding output file + time index
# GC files: GEOSChem.CATRINE_inst.YYYYMMDD_HHMMz.nc4
# Our output: 3-hourly, 8 steps/day, time in "minutes since start"
function find_gc_snapshots(gc_dir::String)
    snapshots = Tuple{String, Date, Int}[]  # (gc_path, date, hour)
    for f in readdir(gc_dir)
        m = match(r"GEOSChem\.CATRINE_inst\.(\d{8})_(\d{2})\d{2}z\.nc4", f)
        m === nothing && continue
        date = Date(m.captures[1], dateformat"yyyymmdd")
        hour = parse(Int, m.captures[2])
        push!(snapshots, (joinpath(gc_dir, f), date, hour))
    end
    sort!(snapshots, by=x -> (x[2], x[3]))
    return snapshots
end

function find_output_match(output_dir::String, target_date::Date, target_hour::Int,
                          start_date::Date)
    # Our output files: catrine_geosit_c180_YYYYMMDD.nc
    datestr = Dates.format(target_date, "yyyymmdd")
    candidates = filter(f -> contains(f, datestr) && endswith(f, ".nc"), readdir(output_dir))
    isempty(candidates) && return nothing, -1

    fpath = joinpath(output_dir, first(candidates))
    ds = NCDataset(fpath, "r")
    times_min = ds["time"][:]  # minutes since start_date
    close(ds)

    target_min = Float64(Dates.value(target_date - start_date) * 24 * 60 + target_hour * 60)
    idx = findfirst(t -> abs(t - target_min) < 1.0, times_min)
    return idx === nothing ? (nothing, -1) : (fpath, idx)
end

struct LevelStats
    level::Int
    our_mean::Float64
    gc_mean::Float64
    bias::Float64
    rel_pct::Float64
    rmse::Float64
    corr::Float64
end

function compare_3d(our_data, gc_data; levels=[1, 6, 11, 21, 31, 41, 51, 61, 66, 71, 72])
    stats = LevelStats[]
    for k in levels
        k > size(our_data, 1) && continue
        our_k = vec(our_data[k, :, :, :])
        gc_k  = vec(gc_data[k, :, :, :])
        mask = isfinite.(our_k) .& isfinite.(gc_k) .& (abs.(gc_k) .> 1e-30)
        sum(mask) < 10 && continue

        o = our_k[mask]
        g = gc_k[mask]
        bias = mean(o .- g)
        rel = 100 * bias / mean(g)
        rmse = sqrt(mean((o .- g).^2))
        c = length(o) > 2 ? cor(o, g) : NaN
        push!(stats, LevelStats(k, mean(o), mean(g), bias, rel, rmse, c))
    end
    return stats
end

function print_level_stats(stats::Vector{LevelStats}, label::String)
    println("\n  $label:")
    @printf("  %5s  %12s  %12s  %12s  %8s  %12s  %6s\n",
            "Level", "Our mean", "GC mean", "Bias", "Rel%", "RMSE", "Corr")
    for s in stats
        @printf("  %5d  %12.6e  %12.6e  %12.3e  %+8.4f  %12.3e  %6.4f\n",
                s.level, s.our_mean, s.gc_mean, s.bias, s.rel_pct, s.rmse, s.corr)
    end
end

function main()
    snapshots = find_gc_snapshots(GC_DIR)
    isempty(snapshots) && error("No GEOS-Chem snapshots found in $GC_DIR")

    # Detect start date from first output file
    first_nc = filter(f -> endswith(f, ".nc"), readdir(OUTPUT_DIR))
    isempty(first_nc) && error("No .nc files in $OUTPUT_DIR")
    ds_tmp = NCDataset(joinpath(OUTPUT_DIR, first(sort(first_nc))), "r")
    time_units = ds_tmp["time"].attrib["units"]  # "minutes since YYYY-MM-DDTHH:MM:SS"
    start_date = Date(match(r"since (\d{4}-\d{2}-\d{2})", time_units).captures[1])
    close(ds_tmp)

    println("=" ^ 80)
    println("AtmosTransport vs GEOS-Chem — CATRINE Comparison")
    println("=" ^ 80)
    println("  Output dir:  $OUTPUT_DIR")
    println("  GC dir:      $GC_DIR")
    println("  Start date:  $start_date")
    println("  Snapshots:   $(length(snapshots))")

    for (gc_path, gc_date, gc_hour) in snapshots
        our_path, tidx = find_output_match(OUTPUT_DIR, gc_date, gc_hour, start_date)
        if our_path === nothing
            @warn "No matching output for $(gc_date) $(gc_hour):00"
            continue
        end

        datestr = Dates.format(gc_date, "yyyy-mm-dd")
        println("\n" * "─" ^ 80)
        @printf("Snapshot: %s %02d:00 UTC  (output time index: %d)\n", datestr, gc_hour, tidx)
        println("─" ^ 80)

        ds_our = NCDataset(our_path, "r")
        ds_gc  = NCDataset(gc_path, "r")

        # --- Species concentrations ---
        for (our_var, gc_var, label) in SPECIES_MAP
            our_key = "$(our_var)_3d"
            haskey(ds_our, our_key) || continue
            haskey(ds_gc, gc_var)   || continue

            our = Float64.(coalesce.(ds_our[our_key][tidx, :, :, :, :], NaN))
            gc  = Float64.(coalesce.(ds_gc[gc_var][1, :, :, :, :], NaN))

            # Mask fill values
            gc[abs.(gc) .> 1e10] .= NaN

            # Reshape: our is (lev, nf, Y, X), gc is (lev, nf, Y, X)
            # Reorder to (lev, nf*Y*X) for statistics

            # Global stats
            mask = isfinite.(our) .& isfinite.(gc) .& (abs.(gc) .> 1e-30)
            n_valid = sum(mask)
            if n_valid > 0
                bias = mean(our[mask] .- gc[mask])
                rel = 100 * bias / mean(gc[mask])
                rmse = sqrt(mean((our[mask] .- gc[mask]).^2))
                @printf("\n%s: global bias=%+.3e (%.4f%%), RMSE=%.3e, N=%d\n",
                        label, bias, rel, rmse, n_valid)
            end

            stats = compare_3d(our, gc)
            print_level_stats(stats, label)
        end

        # --- Surface pressure ---
        if haskey(ds_our, "surface_pressure") && haskey(ds_gc, "Met_PSC2WET")
            our_ps = Float64.(coalesce.(ds_our["surface_pressure"][tidx, :, :, :], NaN))
            gc_wet = Float64.(coalesce.(ds_gc["Met_PSC2WET"][1, :, :, :], NaN))
            gc_dry = Float64.(coalesce.(ds_gc["Met_PSC2DRY"][1, :, :, :], NaN))
            gc_wet[abs.(gc_wet) .> 1e10] .= NaN
            gc_dry[abs.(gc_dry) .> 1e10] .= NaN

            mask = isfinite.(our_ps) .& isfinite.(gc_wet)
            if sum(mask) > 0
                bias_wet = mean(our_ps[mask] .- gc_wet[mask])
                bias_dry = mean(our_ps[mask] .- gc_dry[mask])
                @printf("\nSurface Pressure: our=%.1f  gc_wet=%.1f  gc_dry=%.1f hPa\n",
                        mean(our_ps[mask]), mean(gc_wet[mask]), mean(gc_dry[mask]))
                @printf("  vs wet: bias=%+.1f hPa  vs dry: bias=%+.1f hPa\n",
                        bias_wet, bias_dry)
            end
        end

        close(ds_our)
        close(ds_gc)
    end

    println("\n" * "=" ^ 80)
    println("Done.")
end

main()
