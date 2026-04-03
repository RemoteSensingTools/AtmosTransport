#!/usr/bin/env julia
# ===========================================================================
# Snapshot comparison: ERA5 merged-level run vs GeosChem at day 5
# Surface CO2 + column-mean CO2: AT | GC | Difference
# Also: Fossil CO2 surface, SF6 surface
# ===========================================================================

using CairoMakie
using NCDatasets
using Dates
using Statistics
using Printf

include(joinpath(@__DIR__, "cs_regrid_utils.jl"))

const GC_DIR = expanduser("~/data/AtmosTransport/catrine-geoschem-runs")
const AT_FILE = expanduser("~/data/AtmosTransport/catrine/output/era5_merged_dec2021_pbl/catrine_era5_merged_pbl.bin")
const OUT_PNG = expanduser("~/data/AtmosTransport/catrine/output/era5_merged_dec2021_pbl/snapshot_day5_vs_geoschem.png")

# Target: Dec 5, 2021 12:00 UTC (day 5, midday)
const TARGET_DT = DateTime(2021, 12, 5, 12)
const SIM_START = DateTime(2021, 12, 1, 0)  # actual simulation start

# GeosChem levels
const LEV_SFC_GC = 1   # surface
const LEV_750_GC = 15  # ~750 hPa

# ---------------------------------------------------------------------------
# Load GeosChem snapshot (C180 → 1° lat-lon)
# ---------------------------------------------------------------------------
function load_gc_snapshot(gc_dir, rmap, target_dt)
    files = sort(filter(f -> endswith(f, ".nc4") && contains(f, "CATRINE_inst"), readdir(gc_dir)))

    # Find closest file to target
    best_f, best_dt = nothing, Inf
    for f in files
        m = match(r"(\d{8})_(\d{4})z", f)
        m === nothing && continue
        fdt = DateTime(m[1] * m[2], dateformat"yyyymmddHHMM")
        d = abs(Dates.value(fdt - target_dt)) / 60_000  # minutes
        if d < best_dt
            best_dt = d; best_f = f
        end
    end
    best_f === nothing && error("No GC file found near $target_dt")
    @info "GC file: $best_f (Δ=$(best_dt) min)"

    buf = zeros(Float32, rmap.nlon, rmap.nlat)
    gc = Dict{String, Matrix{Float32}}()

    NCDataset(joinpath(gc_dir, best_f)) do ds
        # CO2 surface + 750 hPa + column mean
        co2 = Float32.(ds["SpeciesConcVV_CO2"][:,:,:,:,1])  # (Xdim, Ydim, nf, lev)
        Nc, _, Np, Nz = size(co2)
        regrid_cs!(buf, co2[:,:,:,LEV_SFC_GC] .* 1f6, rmap)
        gc["co2_sfc"] = copy(buf)
        regrid_cs!(buf, co2[:,:,:,LEV_750_GC] .* 1f6, rmap)
        gc["co2_750"] = copy(buf)
        # Column mean
        xcs = zeros(Float32, Nc, Nc, Np)
        for p in 1:Np, j in 1:Nc, i in 1:Nc
            s = 0f0
            for k in 1:Nz; s += co2[i,j,p,k]; end
            xcs[i,j,p] = s / Nz * 1f6
        end
        regrid_cs!(buf, xcs, rmap)
        gc["xco2"] = copy(buf)

        # Fossil CO2 surface
        fco2 = Float32.(ds["SpeciesConcVV_FossilCO2"][:,:,:,:,1])
        regrid_cs!(buf, fco2[:,:,:,LEV_SFC_GC] .* 1f6, rmap)
        gc["fossil_sfc"] = copy(buf)

        # SF6 surface
        sf6 = Float32.(ds["SpeciesConcVV_SF6"][:,:,:,:,1])
        regrid_cs!(buf, sf6[:,:,:,LEV_SFC_GC] .* 1f12, rmap)  # ppt
        gc["sf6_sfc"] = copy(buf)

        # Rn222 surface
        rn = Float32.(ds["SpeciesConcVV_Rn222"][:,:,:,:,1])
        regrid_cs!(buf, rn[:,:,:,LEV_SFC_GC] .* 1f21, rmap)  # zmol/mol
        gc["rn222_sfc"] = copy(buf)
    end
    return gc
end

# ---------------------------------------------------------------------------
# Load AT snapshot (already lat-lon)
# ---------------------------------------------------------------------------
function load_at_snapshot(at_file, target_dt, sim_start)
    ds = NCDataset(at_file, "r")
    at_times = ds["time"][:]
    Nz = ds.dim["lev"]

    # Map target_dt to sim offset
    target_hours = Dates.value(target_dt - sim_start) / 3_600_000.0
    # AT times are relative to "2000-01-01" — find by offset from first time
    at_offsets_h = [(Dates.value(t - at_times[1]) / 3_600_000.0) for t in at_times]
    # The first AT time is at sim_hour = interval (e.g., 3h for 3-hourly output)
    # We want the one closest to target_hours
    # AT sim_hours: output interval * (1:Nt)
    # From the data: at_times[1] = 2000-01-01T06:00 (first output at 6h = window 1)
    # target_hours from sim_start: Dec5 12:00 - Dec1 00:00 = 108h
    # So we want the AT step where offset from at_times[1] = 108 - 6 = 102h
    # But more robust: compute hours-since-sim-start for each AT step
    first_out_h = 6.0  # first output at hour 6 (see 2000-01-01T06:00)
    at_sim_hours = [first_out_h + Dates.value(t - at_times[1]) / 3_600_000.0 for t in at_times]
    diffs = [abs(h - target_hours) for h in at_sim_hours]
    bi = argmin(diffs)
    @info "AT step $bi: sim_hour=$(at_sim_hours[bi]), target=$target_hours, Δ=$(diffs[bi])h"

    at = Dict{String, Matrix{Float32}}()

    # CO2 surface + column mean
    co2 = Float32.(ds["co2_3d"][:,:,:,bi]) .* 1f6  # mol/mol → ppm
    at["co2_sfc"] = co2[:, :, Nz]  # surface = last level (k=Nz)
    # 750 hPa level: ~75% of the way down
    lev_750 = max(1, round(Int, Nz * 0.75))
    at["co2_750"] = co2[:, :, lev_750]
    # Column mean (simple average)
    xco2 = zeros(Float32, size(co2, 1), size(co2, 2))
    for j in axes(co2, 2), i in axes(co2, 1)
        s = 0f0
        for k in 1:Nz; s += co2[i,j,k]; end
        xco2[i,j] = s / Nz
    end
    at["xco2"] = xco2

    # Fossil CO2 surface
    fco2 = Float32.(ds["fossil_co2_3d"][:,:,:,bi]) .* 1f6
    at["fossil_sfc"] = fco2[:, :, Nz]

    # SF6 surface
    sf6 = Float32.(ds["sf6_3d"][:,:,:,bi]) .* 1f12  # ppt
    at["sf6_sfc"] = sf6[:, :, Nz]

    # Rn222 surface
    rn = Float32.(ds["rn222_3d"][:,:,:,bi]) .* 1f21  # zmol/mol
    at["rn222_sfc"] = rn[:, :, Nz]

    close(ds)
    return at
end

# ---------------------------------------------------------------------------
# Downsample 720×361 → 360×180
# ---------------------------------------------------------------------------
function downsample(data::AbstractMatrix)
    Nx, Ny = size(data)
    Nx2, Ny2 = Nx ÷ 2, min(Ny, 360) ÷ 2
    out = zeros(Float32, Nx2, Ny2)
    for j in 1:Ny2, i in 1:Nx2
        i2, j2 = 2i-1, 2j-1
        out[i,j] = (data[i2,j2] + data[i2+1,j2] + data[i2,j2+1] + data[i2+1,j2+1]) / 4f0
    end
    return out
end

# ---------------------------------------------------------------------------
# Main: create 4-row snapshot
# ---------------------------------------------------------------------------
function main()
    # Build CS→LL regrid map from GC file
    gc_files = sort(filter(f -> endswith(f, ".nc4"), readdir(GC_DIR)))
    gc_ref = joinpath(GC_DIR, gc_files[1])
    cs_lons, cs_lats = NCDataset(gc_ref) do ds
        Float64.(ds["lons"][:,:,:]), Float64.(ds["lats"][:,:,:])
    end
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)
    @info "Regrid map: $(rmap.nlon)×$(rmap.nlat)"

    gc = load_gc_snapshot(GC_DIR, rmap, TARGET_DT)
    at = load_at_snapshot(AT_FILE, TARGET_DT, SIM_START)

    # Downsample AT 720×361 → 360×180 and shift lons from 0..360 to -180..180
    for (k, v) in at
        ds_v = downsample(v)
        # Circshift by half to go from 0..360 → -180..180
        at[k] = circshift(ds_v, (size(ds_v, 1) ÷ 2, 0))
    end

    # Print diagnostics
    for key in ("co2_sfc", "xco2", "fossil_sfc", "sf6_sfc")
        a, g = at[key], gc[key]
        nlon = min(size(a,1), size(g,1)); nlat = min(size(a,2), size(g,2))
        av, gv = vec(a[1:nlon,1:nlat]), vec(g[1:nlon,1:nlat])
        # Clip AT to reasonable range for correlation
        valid = (av .> quantile(av, 0.01)) .& (av .< quantile(av, 0.99))
        r_raw = cor(av, gv)
        r_clip = cor(av[valid], gv[valid])
        @info @sprintf("%-12s AT: %.4f..%.4f  GC: %.4f..%.4f  R=%.3f (clipped R=%.3f)",
                       key, minimum(av), maximum(av), minimum(gv), maximum(gv), r_raw, r_clip)
    end

    # Build figure: 4 rows × 3 cols (AT | GC | Diff)
    # Use GC-derived color ranges for AT+GC, symmetric for diff
    species = [
        ("co2_sfc",    "CO₂ Surface (ppm)",     nothing,     (-8, 8)),
        ("xco2",       "XCO₂ Column Mean (ppm)", nothing,    (-5, 5)),
        ("fossil_sfc", "Fossil CO₂ Sfc (ppm)",   nothing,    (-5, 5)),
        ("sf6_sfc",    "SF₆ Surface (ppt)",       nothing,   (-0.5, 0.5)),
    ]

    fig = Figure(size=(1400, 1200), fontsize=12)

    for (row, (key, label, _, dlim)) in enumerate(species)
        at_data = at[key]
        gc_data = gc[key]

        # Match dimensions (GC is 360×180, AT downsample is 360×180)
        nlon = min(size(at_data, 1), size(gc_data, 1))
        nlat = min(size(at_data, 2), size(gc_data, 2))
        a = at_data[1:nlon, 1:nlat]
        g = gc_data[1:nlon, 1:nlat]

        # Derive color range from GC (percentile-based)
        gv = filter(isfinite, vec(g))
        clim = (Float64(quantile(gv, 0.02)), Float64(quantile(gv, 0.98)))
        # Clip AT to same range for fair comparison
        a_clip = clamp.(a, Float32(clim[1]), Float32(clim[2]))
        d = a_clip .- g

        ax1 = Axis(fig[row, 1]; title=(row==1 ? "AtmosTransport" : ""), ylabel=label,
                   aspect=DataAspect())
        ax2 = Axis(fig[row, 2]; title=(row==1 ? "GeosChem" : ""),
                   aspect=DataAspect())
        ax3 = Axis(fig[row, 3]; title=(row==1 ? "AT - GC" : ""),
                   aspect=DataAspect())

        # Plot (transpose for lat on y-axis)
        hm1 = heatmap!(ax1, a_clip'; colorrange=clim, colormap=:viridis)
        hm2 = heatmap!(ax2, g'; colorrange=clim, colormap=:viridis)
        hm3 = heatmap!(ax3, d'; colorrange=dlim, colormap=:RdBu)

        Colorbar(fig[row, 4], hm1; width=12)

        hidedecorations!.([ax1, ax2, ax3])

        # Stats (on clipped data)
        r = cor(vec(a_clip), vec(g))
        bias = mean(a_clip .- g)
        Label(fig[row, 3], @sprintf("R=%.3f\nbias=%.2f", r, bias),
              fontsize=10, halign=:right, valign=:top,
              tellwidth=false, tellheight=false,
              padding=(0, 5, 5, 0))
    end

    Label(fig[0, 1:3], "ERA5 Merged (68 levels) vs GeosChem C180 — Day 5 ($(Dates.format(TARGET_DT, "yyyy-mm-dd HH:MM")))",
          fontsize=16, font=:bold)

    mkpath(dirname(OUT_PNG))
    save(OUT_PNG, fig; px_per_unit=2)
    @info "Saved: $OUT_PNG"
end

main()
