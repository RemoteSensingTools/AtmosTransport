#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Compare ERA5 (lat-lon) AtmosTransport output against GEOS-Chem (cubed-sphere)
#
# Regrids GEOS-Chem C180 column-mean VMR to the ERA5 lat-lon grid via
# nearest-neighbor, then computes per-tracer slope, R², and anomaly R².
#
# Uses air mass from the preprocessed met file for mass-weighted column means.
#
# Usage:
#   julia --project=. scripts/diagnostics/compare_era5_vs_geoschem.jl \
#       ~/data/AtmosTransport/catrine/output/era5_spectral/catrine_era5_spectral.nc \
#       ~/data/AtmosTransport/catrine-geoschem-runs/ \
#       /temp1/atmos_transport/era5_spectral_catrine/massflux_era5_spectral_202112_float32.nc
# ---------------------------------------------------------------------------

using NCDatasets, Dates, Printf, Statistics

const ERA5_PATH = get(ARGS, 1,
    expanduser("~/data/AtmosTransport/catrine/output/era5_spectral/catrine_era5_spectral.nc"))
const GC_DIR = get(ARGS, 2,
    expanduser("~/data/AtmosTransport/catrine-geoschem-runs"))
const MET_PATH = get(ARGS, 3,
    "/temp1/atmos_transport/era5_spectral_catrine/massflux_era5_spectral_202112_float32.nc")

const SPECIES_MAP = [
    ("co2_3d",        "SpeciesConcVV_CO2",       "CO2"),
    ("fossil_co2_3d", "SpeciesConcVV_FossilCO2", "Fossil CO2"),
    ("sf6_3d",        "SpeciesConcVV_SF6",       "SF6"),
    ("rn222_3d",      "SpeciesConcVV_Rn222",     "Rn222"),
]

# ── Helpers ────────────────────────────────────────────────────────────────

function slope_r2(x::AbstractVector, y::AbstractVector; floor=1e-30)
    mask = isfinite.(x) .& isfinite.(y) .& (abs.(x) .> floor)
    n = sum(mask)
    n < 10 && return (NaN, NaN, 0)
    xm, ym = x[mask], y[mask]
    r = cor(xm, ym)
    s = cov(xm, ym) / var(xm)
    return (s, r^2, n)
end

"""Anomaly correlation: subtract global mean from each 2D field, then correlate.
Focuses on spatial patterns rather than absolute values."""
function anomaly_slope_r2(x::AbstractVector, y::AbstractVector; floor=1e-30)
    mask = isfinite.(x) .& isfinite.(y) .& (abs.(x) .> floor)
    n = sum(mask)
    n < 10 && return (NaN, NaN, 0)
    xm, ym = x[mask], y[mask]
    xa = xm .- mean(xm)
    ya = ym .- mean(ym)
    r = cor(xa, ya)
    s = cov(xa, ya) / var(xa)
    return (s, r^2, n)
end

function mask_fill!(a::AbstractArray)
    a[abs.(a) .> 1e14] .= NaN
    return a
end

# ── Build CS→LL regrid map ─────────────────────────────────────────────────

"""Build nearest-neighbor mapping from C180 CS to Nlon×Nlat lat-lon grid."""
function build_cs_to_ll_map(gc_path::String, Nlon::Int, Nlat::Int)
    ds = NCDataset(gc_path, "r")
    c_lons = Float64.(ds["corner_lons"][:, :, :])  # (XCdim, YCdim, nf) in Julia
    c_lats = Float64.(ds["corner_lats"][:, :, :])
    Nc = size(ds["SpeciesConcVV_CO2"], 1)
    close(ds)

    # Compute cell centers from corners
    clon = zeros(Nc, Nc, 6)
    clat = zeros(Nc, Nc, 6)
    for p in 1:6, j in 1:Nc, i in 1:Nc
        lons4 = [c_lons[i,j,p], c_lons[i+1,j,p], c_lons[i,j+1,p], c_lons[i+1,j+1,p]]
        lats4 = [c_lats[i,j,p], c_lats[i+1,j,p], c_lats[i,j+1,p], c_lats[i+1,j+1,p]]
        if maximum(lons4) - minimum(lons4) > 180
            lons4 = [l < 0 ? l + 360 : l for l in lons4]
        end
        clon[i,j,p] = mod(mean(lons4) + 180, 360) - 180
        clat[i,j,p] = mean(lats4)
    end

    dlon = 360.0 / Nlon
    dlat = 180.0 / (Nlat - 1)

    cs_to_ll_i = zeros(Int, Nc, Nc, 6)
    cs_to_ll_j = zeros(Int, Nc, Nc, 6)
    for p in 1:6, j in 1:Nc, i in 1:Nc
        lon = clon[i,j,p]
        lat = clat[i,j,p]
        ii = clamp(round(Int, (lon + 180 - dlon/2) / dlon) + 1, 1, Nlon)
        jj = clamp(round(Int, (lat + 90) / dlat) + 1, 1, Nlat)
        cs_to_ll_i[i,j,p] = ii
        cs_to_ll_j[i,j,p] = jj
    end

    return cs_to_ll_i, cs_to_ll_j, Nc
end

"""Regrid a 2D CS field (Nc×Nc×6) to lat-lon using precomputed mapping."""
function regrid_cs_2d(field_cs::Array{Float64,3}, cs_to_ll_i, cs_to_ll_j,
                       Nlon::Int, Nlat::Int)
    Nc = size(field_cs, 1)
    out   = zeros(Nlon, Nlat)
    count = zeros(Int, Nlon, Nlat)
    for p in 1:6, j in 1:Nc, i in 1:Nc
        val = field_cs[i,j,p]
        isfinite(val) || continue
        ii = cs_to_ll_i[i,j,p]
        jj = cs_to_ll_j[i,j,p]
        out[ii,jj] += val
        count[ii,jj] += 1
    end
    for idx in eachindex(out)
        count[idx] > 0 && (out[idx] /= count[idx])
    end
    return out
end

# ── Column-mean VMR ───────────────────────────────────────────────────────

"""Mass-weighted column-mean VMR for lat-lon: Σ(q*m)/Σ(m) over levels (dim 3)."""
function column_mean_ll(q3d::Array{Float64,3}, m3d::Array{Float64,3})
    num = dropdims(sum(q3d .* m3d; dims=3); dims=3)
    den = dropdims(sum(m3d; dims=3); dims=3)
    return num ./ den
end

"""Mass-weighted column-mean VMR for CS (4D: X×Y×nf×lev) using air mass weights."""
function column_mean_cs(q4d::Array{Float64,4}, ad4d::Array{Float64,4})
    num = dropdims(sum(q4d .* ad4d; dims=4); dims=4)
    den = dropdims(sum(ad4d; dims=4); dims=4)
    return num ./ den
end

# ── Snapshot discovery ────────────────────────────────────────────────────

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

# ── Main ──────────────────────────────────────────────────────────────────

function main()
    snapshots = find_gc_snapshots(GC_DIR)
    isempty(snapshots) && error("No GEOS-Chem snapshots found in $GC_DIR")

    # Read ERA5 output time axis and grid
    ds_era = NCDataset(ERA5_PATH, "r")
    era_times = ds_era["time"][:]
    era_lons = Float64.(ds_era["lon"][:])
    era_lats = Float64.(ds_era["lat"][:])
    Nlon = length(era_lons)
    Nlat = length(era_lats)
    Nz_era = size(ds_era["co2_3d"], 3)
    close(ds_era)

    # Read met file time axis for air mass matching
    ds_met = NCDataset(MET_PATH, "r")
    met_times = ds_met["time"][:]
    close(ds_met)

    # Build CS→LL regrid map from first GC file
    cs_to_ll_i, cs_to_ll_j, Nc = build_cs_to_ll_map(first(snapshots)[1], Nlon, Nlat)

    println("=" ^ 90)
    println("ERA5 AtmosTransport vs GEOS-Chem — CATRINE Cross-Grid Comparison")
    println("=" ^ 90)
    println("  ERA5 output: $ERA5_PATH")
    println("  Met file:    $MET_PATH")
    println("  GC dir:      $GC_DIR")
    println("  ERA5 grid:   $(Nlon)×$(Nlat)×$(Nz_era)")
    println("  CS grid:     C$(Nc)")
    println("  GC snaps:    $(length(snapshots))")
    println("  Column mean: mass-weighted (using air mass from met file)")

    # Summary accumulator
    SummaryRow = NamedTuple{
        (:time, :s_col, :r2_col, :anom_s, :anom_r2, :era_mean, :gc_mean),
        Tuple{DateTime, Float64, Float64, Float64, Float64, Float64, Float64}}
    summary = Dict{String, Vector{SummaryRow}}()
    for (_, _, label) in SPECIES_MAP
        summary[label] = SummaryRow[]
    end

    n_matched = 0
    for (gc_path, gc_dt) in snapshots
        # Find matching ERA5 time index
        era_tidx = findfirst(t -> abs(Dates.value(t - gc_dt)) < 120_000, era_times)
        era_tidx === nothing && continue

        # Find matching met file time index
        met_tidx = findfirst(t -> abs(Dates.value(t - gc_dt)) < 120_000, met_times)

        n_matched += 1

        ts = Dates.format(gc_dt, "yyyy-mm-dd HH:MM")
        if n_matched <= 5
            println("\n" * "─" ^ 90)
            @printf("Snapshot: %s UTC  (ERA5 tidx: %d, met tidx: %s)\n",
                    ts, era_tidx, met_tidx === nothing ? "none" : string(met_tidx))
            println("─" ^ 90)
        end

        ds_era = NCDataset(ERA5_PATH, "r")
        ds_gc = NCDataset(gc_path, "r")

        # Load air mass from met file for ERA5 column weighting
        era_m = nothing
        if met_tidx !== nothing
            ds_met = NCDataset(MET_PATH, "r")
            era_m = Float64.(ds_met["m"][:, :, :, met_tidx])  # (lon, lat, lev) in Julia
            close(ds_met)
        end

        # GC air mass for column weighting
        gc_ad = haskey(ds_gc, "Met_AD") ? Float64.(ds_gc["Met_AD"][:,:,:,:,1]) : nothing
        gc_ad !== nothing && mask_fill!(gc_ad)

        for (era_var, gc_var, label) in SPECIES_MAP
            haskey(ds_era, era_var) || continue
            haskey(ds_gc, gc_var) || continue

            # ERA5: (lon, lat, lev, time) in Julia
            era_q = Float64.(ds_era[era_var][:, :, :, era_tidx])

            # GC: (Xdim, Ydim, nf, lev, time) in Julia
            gc_q = Float64.(ds_gc[gc_var][:, :, :, :, 1])
            mask_fill!(gc_q)

            # Mass-weighted column-mean VMR
            if era_m !== nothing
                era_col = column_mean_ll(era_q, era_m)
            else
                # Fallback: simple average (wrong but better than nothing)
                era_col = dropdims(mean(era_q; dims=3); dims=3)
            end

            if gc_ad !== nothing
                gc_col_cs = column_mean_cs(gc_q, gc_ad)
            else
                gc_col_cs = dropdims(mean(gc_q; dims=4); dims=4)
            end

            # Regrid GC column-mean to lat-lon
            gc_col_ll = regrid_cs_2d(gc_col_cs, cs_to_ll_i, cs_to_ll_j, Nlon, Nlat)

            # Raw column correlation
            s_col, r2_col, _ = slope_r2(vec(gc_col_ll), vec(era_col))

            # Anomaly column correlation (subtract spatial mean)
            anom_s, anom_r2, _ = anomaly_slope_r2(vec(gc_col_ll), vec(era_col))

            era_mean = mean(filter(isfinite, vec(era_col)))
            gc_mean  = mean(filter(isfinite, vec(gc_col_ll)))

            if n_matched <= 5
                @printf("  %-12s  Col:  slope=%8.5f  R²=%8.6f  ERA=%.4e  GC=%.4e\n",
                        label, s_col, r2_col, era_mean, gc_mean)
                @printf("  %-12s  Anom: slope=%8.5f  R²=%8.6f\n",
                        label, anom_s, anom_r2)
            end

            push!(summary[label],
                  (time=gc_dt, s_col=s_col, r2_col=r2_col,
                   anom_s=anom_s, anom_r2=anom_r2,
                   era_mean=era_mean, gc_mean=gc_mean))
        end

        close(ds_era)
        close(ds_gc)
    end

    # ── Summary ──
    println("\n\n" * "=" ^ 90)
    println("SUMMARY — Column-Mean VMR Correlation (ERA5 vs GeosChem)")
    println("=" ^ 90)

    for (_, _, label) in SPECIES_MAP
        res = summary[label]
        isempty(res) && continue
        println("\n  $label:")
        @printf("  %-18s  %10s  %10s  %10s  %10s  %12s  %12s\n",
                "Time (UTC)", "Col Slope", "Col R²", "Anom Slope", "Anom R²",
                "ERA5 mean", "GC mean")
        @printf("  %-18s  %10s  %10s  %10s  %10s  %12s  %12s\n",
                "─"^18, "─"^10, "─"^10, "─"^10, "─"^10, "─"^12, "─"^12)
        for r in res
            @printf("  %-18s  %10.6f  %10.6f  %10.6f  %10.6f  %12.6e  %12.6e\n",
                    Dates.format(r.time, "yyyy-mm-dd HH:MM"),
                    r.s_col, r.r2_col, r.anom_s, r.anom_r2,
                    r.era_mean, r.gc_mean)
        end
        vals_s  = [r.s_col   for r in res if isfinite(r.s_col)]
        vals_r  = [r.r2_col  for r in res if isfinite(r.r2_col)]
        vals_as = [r.anom_s  for r in res if isfinite(r.anom_s)]
        vals_ar = [r.anom_r2 for r in res if isfinite(r.anom_r2)]
        if !isempty(vals_s)
            @printf("  %-18s  %10.6f  %10.6f  %10.6f  %10.6f\n",
                    "MEAN", mean(vals_s), mean(vals_r),
                    isempty(vals_as) ? NaN : mean(vals_as),
                    isempty(vals_ar) ? NaN : mean(vals_ar))
        end
    end

    println("\n" * "=" ^ 90)
    @printf("Matched %d / %d GC snapshots.\n", n_matched, length(snapshots))
    println("Done.")
end

main()
