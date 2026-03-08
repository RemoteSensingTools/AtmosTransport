#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Compare AtmosTransport output against GEOS-Chem CATRINE snapshots
#
# Computes per-tracer, per-timestep:
#   - 3D flattened correlation (slope + R²)
#   - Column-mean VMR correlation (slope + R²)
#   - Per-level bias, RMSE, correlation
#   - Surface pressure comparison (Pa→hPa)
#
# GEOS-Chem snapshots contain:
#   SpeciesConcVV_{CO2,FossilCO2,SF6,Rn222}  — dry mole fractions [mol/mol]
#   Met_AD         — dry air mass per cell [kg]
#   Met_PSC2WET/DRY — wet/dry surface pressure [hPa]
#   ColumnMass_*   — column-integrated mass
#
# Usage:
#   julia --project=. scripts/diagnostics/compare_vs_geoschem.jl [output_dir] [gc_dir]
# ---------------------------------------------------------------------------

using NCDatasets, Dates, Printf, Statistics

const OUTPUT_DIR = get(ARGS, 1,
    expanduser("~/data/AtmosTransport/catrine/output/geosit_c180"))
const GC_DIR = get(ARGS, 2,
    expanduser("~/data/AtmosTransport/catrine-geoschem-runs"))

# (AT 3d var, GC var, display label)
const SPECIES_MAP = [
    ("co2_3d",        "SpeciesConcVV_CO2",       "CO2"),
    ("fossil_co2_3d", "SpeciesConcVV_FossilCO2", "Fossil CO2"),
    ("sf6_3d",        "SpeciesConcVV_SF6",       "SF6"),
    ("rn222_3d",      "SpeciesConcVV_Rn222",     "Rn222"),
]

# (AT column_mass var, GC column_mass var)
const COLMASS_MAP = Dict(
    "CO2"        => ("co2_column_mass",    "ColumnMass_CO2"),
    "Fossil CO2" => ("fco2_column_mass",   "ColumnMass_FossilCO2"),
    "SF6"        => ("sf6_column_mass",    "ColumnMass_SF6"),
    "Rn222"      => ("rn222_column_mass",  "ColumnMass_Rn222"),
)

# ── Helpers ──────────────────────────────────────────────────────────────

"""Linear regression slope and R² between two vectors. Ignores NaN/Inf and
values below `floor`. Returns (slope, R², N_valid)."""
function slope_r2(x::AbstractVector, y::AbstractVector; floor=1e-30)
    mask = isfinite.(x) .& isfinite.(y) .& (abs.(x) .> floor)
    n = sum(mask)
    n < 10 && return (NaN, NaN, 0)
    xm, ym = x[mask], y[mask]
    r = cor(xm, ym)
    s = cov(xm, ym) / var(xm)
    return (s, r^2, n)
end

"""Anomaly correlation: subtract per-level mean, then flatten and correlate.
This removes the background and correlates spatial perturbation patterns."""
function anomaly_slope_r2(at_4d::AbstractArray{T,4}, gc_4d::AbstractArray{T,4};
                          floor=1e-30) where T
    Nz = size(at_4d, 4)
    at_anom = similar(at_4d)
    gc_anom = similar(gc_4d)
    for k in 1:Nz
        at_k = @view at_4d[:,:,:,k]
        gc_k = @view gc_4d[:,:,:,k]
        mask_k = isfinite.(gc_k) .& isfinite.(at_k)
        at_mean_k = sum(mask_k) > 0 ? mean(at_k[mask_k]) : zero(T)
        gc_mean_k = sum(mask_k) > 0 ? mean(gc_k[mask_k]) : zero(T)
        at_anom[:,:,:,k] .= at_k .- at_mean_k
        gc_anom[:,:,:,k] .= gc_k .- gc_mean_k
    end
    return slope_r2(vec(gc_anom), vec(at_anom); floor)
end

"""Trimmed slope/R²: exclude cells where either AT or GC is in the top/bottom
`pct` percentile, to reduce influence of extreme emission hotspots."""
function trimmed_slope_r2(x::AbstractVector, y::AbstractVector;
                          pct::Float64=0.01, floor=1e-30)
    mask = isfinite.(x) .& isfinite.(y) .& (abs.(x) .> floor)
    n = sum(mask)
    n < 100 && return (NaN, NaN, 0)
    xm, ym = x[mask], y[mask]
    xlo, xhi = quantile(xm, [pct, 1-pct])
    ylo, yhi = quantile(ym, [pct, 1-pct])
    keep = (xm .>= xlo) .& (xm .<= xhi) .& (ym .>= ylo) .& (ym .<= yhi)
    sum(keep) < 10 && return (NaN, NaN, 0)
    xt, yt = xm[keep], ym[keep]
    r = cor(xt, yt)
    s = cov(xt, yt) / var(xt)
    return (s, r^2, sum(keep))
end

"""Replace GC fill values (|v| > 1e14) with NaN. GC _FillValue = 1e15."""
function mask_fill!(a::AbstractArray)
    a[abs.(a) .> 1e14] .= NaN
    return a
end

# ── Snapshot discovery ───────────────────────────────────────────────────

function find_gc_snapshots(gc_dir::String)
    snapshots = Tuple{String, DateTime}[]
    for f in readdir(gc_dir)
        m = match(r"GEOSChem\.CATRINE_inst\.(\d{8})_(\d{2})(\d{2})z\.nc4", f)
        m === nothing && continue
        dt = DateTime(m.captures[1], dateformat"yyyymmdd") +
             Hour(parse(Int, m.captures[2])) + Minute(parse(Int, m.captures[3]))
        push!(snapshots, (joinpath(gc_dir, f), dt))
    end
    sort!(snapshots, by=last)
    return snapshots
end

function find_output_match(output_dir::String, target_dt::DateTime,
                           start_dt::DateTime)
    datestr = Dates.format(target_dt, "yyyymmdd")
    candidates = filter(f -> contains(f, datestr) && endswith(f, ".nc"),
                        readdir(output_dir))
    isempty(candidates) && return nothing, -1

    fpath = joinpath(output_dir, first(candidates))
    ds = NCDataset(fpath, "r")
    times_raw = ds["time"][:]          # NCDatasets auto-converts to DateTime
    close(ds)

    # Handle both DateTime and numeric time values
    if eltype(times_raw) <: DateTime
        idx = findfirst(t -> abs(Dates.value(t - target_dt)) < 60_000, times_raw)  # <1 min
    else
        target_min = Float64(Dates.value(target_dt - start_dt) / 60_000)
        idx = findfirst(t -> abs(t - target_min) < 1.0, times_raw)
    end
    return idx === nothing ? (nothing, -1) : (fpath, idx)
end

# ── Per-level statistics ─────────────────────────────────────────────────

struct LevelStats
    level::Int
    at_mean::Float64
    gc_mean::Float64
    bias::Float64
    rel_pct::Float64
    rmse::Float64
    corr::Float64
end

function compare_levels(at_data, gc_data;
                        levels=[1, 10, 20, 30, 40, 50, 60, 66, 70, 72])
    # at_data, gc_data: (X, Y, nf, lev)
    stats = LevelStats[]
    Nz = size(at_data, 4)
    for k in levels
        k > Nz && continue
        at_k = vec(at_data[:, :, :, k])
        gc_k = vec(gc_data[:, :, :, k])
        mask = isfinite.(at_k) .& isfinite.(gc_k) .& (abs.(gc_k) .> 1e-30)
        sum(mask) < 10 && continue

        a, g = at_k[mask], gc_k[mask]
        bias = mean(a .- g)
        rel  = 100 * bias / mean(g)
        rmse = sqrt(mean((a .- g) .^ 2))
        c    = length(a) > 2 ? cor(a, g) : NaN
        push!(stats, LevelStats(k, mean(a), mean(g), bias, rel, rmse, c))
    end
    return stats
end

function print_level_table(stats::Vector{LevelStats}, label::String)
    isempty(stats) && return
    println("\n  $label — per-level:")
    @printf("  %5s  %12s  %12s  %12s  %8s  %12s  %8s\n",
            "Level", "AT mean", "GC mean", "Bias", "Rel%", "RMSE", "Corr")
    for s in stats
        @printf("  %5d  %12.6e  %12.6e  %12.3e  %+8.4f  %12.3e  %8.6f\n",
                s.level, s.at_mean, s.gc_mean, s.bias, s.rel_pct, s.rmse, s.corr)
    end
end

# ── Column-mean VMR ──────────────────────────────────────────────────────

"""Mass-weighted column-mean VMR:  Σ(q_k * w_k) / Σ(w_k) over levels."""
function column_mean_vmr(q3d::AbstractArray{T,4},
                         weights::AbstractArray{T,4}) where T
    # q3d, weights: (X, Y, nf, lev)
    num = sum(q3d .* weights; dims=4)
    den = sum(weights; dims=4)
    col = dropdims(num ./ den; dims=4)  # (X, Y, nf)
    return col
end

# ── Main ─────────────────────────────────────────────────────────────────

function main()
    snapshots = find_gc_snapshots(GC_DIR)
    isempty(snapshots) && error("No GEOS-Chem snapshots found in $GC_DIR")

    # Detect start date from first AT output file
    nc_files = sort(filter(f -> endswith(f, ".nc"), readdir(OUTPUT_DIR)))
    isempty(nc_files) && error("No .nc files in $OUTPUT_DIR")
    ds_tmp = NCDataset(joinpath(OUTPUT_DIR, first(nc_files)), "r")
    time_units = ds_tmp["time"].attrib["units"]
    start_dt = DateTime(match(r"since (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})",
                              time_units).captures[1])
    close(ds_tmp)

    println("=" ^ 90)
    println("AtmosTransport vs GEOS-Chem — CATRINE Comparison")
    println("=" ^ 90)
    println("  AT output:   $OUTPUT_DIR")
    println("  GC dir:      $GC_DIR")
    println("  Start:       $start_dt")
    println("  GC snaps:    $(length(snapshots))")

    # Accumulate summary results
    SummaryRow = NamedTuple{
        (:time, :s3d, :r2_3d, :s_col, :r2_col, :anom_s, :anom_r2, :trim_s, :trim_r2),
        Tuple{DateTime, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64}}
    summary = Dict{String, Vector{SummaryRow}}()
    for (_, _, label) in SPECIES_MAP
        summary[label] = SummaryRow[]
    end

    n_matched = 0
    for (gc_path, gc_dt) in snapshots
        at_path, tidx = find_output_match(OUTPUT_DIR, gc_dt, start_dt)
        if at_path === nothing
            continue    # silently skip — no matching AT output
        end
        n_matched += 1

        ts = Dates.format(gc_dt, "yyyy-mm-dd HH:MM")
        println("\n" * "─" ^ 90)
        @printf("Snapshot: %s UTC  (AT file time index: %d)\n", ts, tidx)
        println("─" ^ 90)

        ds_at = NCDataset(at_path, "r")
        ds_gc = NCDataset(gc_path, "r")

        # Load GC dry air mass for column-mean weighting
        gc_ad = nothing
        if haskey(ds_gc, "Met_AD")
            gc_ad = Float64.(ds_gc["Met_AD"][:, :, :, :, 1])  # (X,Y,nf,lev)
            mask_fill!(gc_ad)
        end

        # ── Species concentrations ──
        for (at_var, gc_var, label) in SPECIES_MAP
            haskey(ds_at, at_var) || continue
            haskey(ds_gc, gc_var) || continue

            # Time is the LAST dimension: [:, :, :, :, tidx]
            at_q = Float64.(ds_at[at_var][:, :, :, :, tidx])  # (X,Y,nf,lev)
            gc_q = Float64.(ds_gc[gc_var][:, :, :, :, 1])
            mask_fill!(gc_q)

            # 3D flattened correlation
            s3d, r2_3d, n3d = slope_r2(vec(gc_q), vec(at_q))
            # Anomaly correlation (subtract per-level mean)
            anom_s, anom_r2, _ = anomaly_slope_r2(at_q, gc_q)
            # Trimmed correlation (exclude top/bottom 1%)
            trim_s, trim_r2, _ = trimmed_slope_r2(vec(gc_q), vec(at_q))
            @printf("\n  %-12s  3D:   slope=%8.5f  R²=%8.6f  (N=%d)\n",
                    label, s3d, r2_3d, n3d)
            @printf("  %-12s  Anom: slope=%8.5f  R²=%8.6f\n",
                    label, anom_s, anom_r2)
            @printf("  %-12s  Trim: slope=%8.5f  R²=%8.6f\n",
                    label, trim_s, trim_r2)

            # Column-mean VMR correlation (using GC air mass as weights)
            s_col, r2_col = NaN, NaN
            if gc_ad !== nothing
                at_col = column_mean_vmr(at_q, gc_ad)
                gc_col = column_mean_vmr(gc_q, gc_ad)
                s_col, r2_col, n_col = slope_r2(vec(gc_col), vec(at_col))
                @printf("  %-12s  Col:  slope=%8.5f  R²=%8.6f  (N=%d)\n",
                        label, s_col, r2_col, n_col)
            end

            push!(summary[label],
                  (time=gc_dt, s3d=s3d, r2_3d=r2_3d, s_col=s_col, r2_col=r2_col,
                   anom_s=anom_s, anom_r2=anom_r2, trim_s=trim_s, trim_r2=trim_r2))

            # Per-level diagnostics (only print for first and last snapshot)
            if n_matched == 1 || gc_dt == last(snapshots)[2]
                lstats = compare_levels(at_q, gc_q)
                print_level_table(lstats, label)
            end
        end

        # ── Surface pressure ──
        if haskey(ds_at, "surface_pressure") && haskey(ds_gc, "Met_PSC2WET")
            at_ps = Float64.(ds_at["surface_pressure"][:, :, :, tidx])  # already hPa
            gc_wet = Float64.(ds_gc["Met_PSC2WET"][:, :, :, 1])
            mask_fill!(gc_wet)

            s_ps, r2_ps, _ = slope_r2(vec(gc_wet), vec(at_ps))
            mask = isfinite.(at_ps) .& isfinite.(gc_wet)
            bias = sum(mask) > 0 ? mean(at_ps[mask] .- gc_wet[mask]) : NaN
            @printf("\n  Surface P:   slope=%8.5f  R²=%8.6f  bias=%+.2f hPa\n",
                    s_ps, r2_ps, bias)
        end

        close(ds_at)
        close(ds_gc)
    end

    # ── Summary table ──
    println("\n\n" * "=" ^ 90)
    println("SUMMARY — Tracer-by-tracer correlation evolution")
    println("=" ^ 90)

    for (_, _, label) in SPECIES_MAP
        res = summary[label]
        isempty(res) && continue
        println("\n  $label:")
        @printf("  %-18s  %10s  %10s  %10s  %10s  %10s  %10s\n",
                "Time (UTC)", "3D Slope", "3D R²", "Anom R²", "Trim R²",
                "Col Slope", "Col R²")
        @printf("  %-18s  %10s  %10s  %10s  %10s  %10s  %10s\n",
                "─" ^ 18, "─"^10, "─"^10, "─"^10, "─"^10, "─"^10, "─"^10)
        for r in res
            @printf("  %-18s  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n",
                    Dates.format(r.time, "yyyy-mm-dd HH:MM"),
                    r.s3d, r.r2_3d, r.anom_r2, r.trim_r2, r.s_col, r.r2_col)
        end
        # Mean across all timesteps
        ms3d  = mean(r.s3d     for r in res if isfinite(r.s3d))
        mr3d  = mean(r.r2_3d   for r in res if isfinite(r.r2_3d))
        mar2  = mean(r.anom_r2 for r in res if isfinite(r.anom_r2))
        mtr2  = mean(r.trim_r2 for r in res if isfinite(r.trim_r2))
        mscol = mean(r.s_col   for r in res if isfinite(r.s_col))
        mrcol = mean(r.r2_col  for r in res if isfinite(r.r2_col))
        @printf("  %-18s  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n",
                "MEAN", ms3d, mr3d, mar2, mtr2, mscol, mrcol)
    end

    println("\n" * "=" ^ 90)
    @printf("Matched %d / %d GC snapshots.\n", n_matched, length(snapshots))
    println("Done.")
end

main()
