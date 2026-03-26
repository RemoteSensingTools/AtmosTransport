#!/usr/bin/env julia
# ===========================================================================
# 6-panel comparison: GEOS-Chem vs AtmosTransport GCHP (full physics, v4)
#
# Layout (3 rows × 2 cols, no difference panels):
#   Row 1: Surface CO2          (GEOS-Chem | AtmosTransport)
#   Row 2: CO2 ~750 hPa         (GEOS-Chem | AtmosTransport)
#   Row 3: Column-avg XCO2      (GEOS-Chem | AtmosTransport)
#
# XCO2: dp-weighted column average using hybrid coordinate pressure weights.
# dp_k = (ak[k+1]-ak[k]) + (bk[k+1]-bk[k]) × PS_ref  (fixed weights, PS_ref=1013.25 hPa)
# ===========================================================================

using CairoMakie
using GeoMakie
using NCDatasets
using Dates
using Statistics

include(joinpath(@__DIR__, "cs_regrid_utils.jl"))

const GC_DIR = get(ENV, "GC_DIR",
    joinpath(homedir(), "data", "AtmosTransport", "catrine-geoschem-runs"))
const AT_DIR   = get(ENV, "AT_DIR", "/temp1/catrine/output")
const AT_PATTERN = get(ENV, "AT_PATTERN", "catrine_gchp_v4_21d_fullphys")
const OUT_GIF  = get(ENV, "OUT_GIF", "/temp1/catrine/output/gchp_v4_21d_fullphys_vs_geoschem.gif")
const FPS      = parse(Int, get(ENV, "FPS", "8"))

const LEV_SURFACE = 1
const LEV_750HPA  = 15
const DATE_START = DateTime(2021, 12, 1, 3)
const DATE_END   = DateTime(2021, 12, 31, 21)

# MW ratio for VMR→mass conversion
const MW_CO2_OVER_AIR = 44.009 / 28.97

# Reference surface pressure for hybrid dp weights [Pa]
const PS_REF = 101325.0

"""Compute hybrid dp weights (fixed, independent of actual surface pressure).
Returns a vector of length Nz with dp_k = (ak[k+1]-ak[k]) + (bk[k+1]-bk[k])×PS_REF."""
function hybrid_dp_weights(ak, bk)
    Nz = length(ak) - 1
    dp = zeros(Float64, Nz)
    for k in 1:Nz
        dp[k] = (ak[k+1] - ak[k]) + (bk[k+1] - bk[k]) * PS_REF
    end
    return dp
end

"""Compute dp-weighted column average XCO2 from 3D VMR field (Nc×Nc×6×Nz).
Uses fixed hybrid dp weights."""
function column_avg_xco2(vmr_3d, dp_weights)
    Nx, Ny, Np, Nz = size(vmr_3d)
    xco2 = zeros(Float32, Nx, Ny, Np)
    dp_total = sum(dp_weights)
    for p in 1:Np, j in 1:Ny, i in 1:Nx
        s = 0.0
        for k in 1:Nz
            s += Float64(vmr_3d[i,j,p,k]) * dp_weights[k]
        end
        xco2[i,j,p] = Float32(s / dp_total * 1e6)  # ppm
    end
    return xco2
end

# ---------------------------------------------------------------------------
# Load AtmosTransport
# ---------------------------------------------------------------------------
function load_atmostr(at_dir, rmap, target_times, dp_wts)
    daily_files = sort(filter(f -> endswith(f, ".nc") && contains(f, AT_PATTERN),
                               readdir(at_dir)))
    nt  = length(target_times)
    buf = zeros(Float32, rmap.nlon, rmap.nlat)
    sfc_co2  = zeros(Float32, rmap.nlon, rmap.nlat, nt)
    hpa_co2  = zeros(Float32, rmap.nlon, rmap.nlat, nt)
    xco2_col = zeros(Float32, rmap.nlon, rmap.nlat, nt)

    for fname in daily_files
        NCDataset(joinpath(at_dir, fname), "r") do ds
            at_times = ds["time"][:]
            for (ti, tgt) in enumerate(target_times)
                diffs = [abs(Dates.value(at_t - tgt)) for at_t in at_times]
                best_idx = argmin(diffs)
                diffs[best_idx] / 60_000 > 90 && continue

                if haskey(ds, "co2_3d")
                    # k=1 = surface, k=15 ≈ 750 hPa (same as GC convention)
                    data_cs = Float32.(ds["co2_3d"][:, :, :, 1, best_idx]) .* 1f6
                    regrid_cs!(buf, data_cs, rmap)
                    sfc_co2[:, :, ti] .= buf

                    data_cs = Float32.(ds["co2_3d"][:, :, :, LEV_750HPA, best_idx]) .* 1f6
                    regrid_cs!(buf, data_cs, rmap)
                    hpa_co2[:, :, ti] .= buf

                    # Column-average XCO2 (dp-weighted)
                    vmr_3d = ds["co2_3d"][:, :, :, :, best_idx]
                    xco2_cs = column_avg_xco2(vmr_3d, dp_wts)
                    regrid_cs!(buf, xco2_cs, rmap)
                    xco2_col[:, :, ti] .= buf
                end
            end
        end
    end
    @info "AtmosTransport: loaded $nt timesteps"
    return (; sfc_co2, hpa_co2, xco2_col)
end

# ---------------------------------------------------------------------------
# Load GEOS-Chem
# ---------------------------------------------------------------------------
function load_geoschem(gc_dir, rmap, dp_wts; date_start, date_end)
    all_files = sort(filter(f -> endswith(f, ".nc4") && contains(f, "CATRINE_inst"),
                             readdir(gc_dir)))
    files = String[]
    times = DateTime[]
    for f in all_files
        m = match(r"(\d{8})_(\d{4})z", f)
        m === nothing && continue
        dt = DateTime(m[1] * m[2], dateformat"yyyymmddHHMM")
        if date_start <= dt <= date_end
            push!(files, f); push!(times, dt)
        end
    end
    @info "GEOS-Chem: $(length(files)) snapshots"

    nt  = length(files)
    buf = zeros(Float32, rmap.nlon, rmap.nlat)
    sfc_co2  = zeros(Float32, rmap.nlon, rmap.nlat, nt)
    hpa_co2  = zeros(Float32, rmap.nlon, rmap.nlat, nt)
    xco2_col = zeros(Float32, rmap.nlon, rmap.nlat, nt)

    bad_indices = Int[]
    for (ti, fname) in enumerate(files)
        try
            NCDataset(joinpath(gc_dir, fname), "r") do ds
                # Surface CO2 (GC level 1 = surface)
                data_cs = Float64.(ds["SpeciesConcVV_CO2"][:, :, :, 1, 1]) .* 1e6
                regrid_cs!(buf, data_cs, rmap); sfc_co2[:, :, ti] .= buf

                # ~750 hPa CO2 (GC level 15)
                data_cs = Float64.(ds["SpeciesConcVV_CO2"][:, :, :, LEV_750HPA, 1]) .* 1e6
                regrid_cs!(buf, data_cs, rmap); hpa_co2[:, :, ti] .= buf

                # Column-average XCO2 (dp-weighted)
                vmr_3d = ds["SpeciesConcVV_CO2"][:,:,:,:,1]
                xco2_cs = column_avg_xco2(vmr_3d, dp_wts)
                regrid_cs!(buf, xco2_cs, rmap)
                xco2_col[:, :, ti] .= buf
            end
        catch e
            @warn "Skipping corrupted GC file: $fname ($e)"
            push!(bad_indices, ti)
        end
    end
    # Remove bad timesteps
    if !isempty(bad_indices)
        @warn "Removed $(length(bad_indices)) corrupted timestep(s)"
        good = setdiff(1:nt, bad_indices)
        times = times[good]
        sfc_co2 = sfc_co2[:, :, good]
        hpa_co2 = hpa_co2[:, :, good]
        xco2_col = xco2_col[:, :, good]
    end
    return (; times, sfc_co2, hpa_co2, xco2_col)
end

# ---------------------------------------------------------------------------
# Animation: 3 rows × 2 cols
# ---------------------------------------------------------------------------
function make_animation(gc, at, times, rmap; fps=FPS)
    nframes = length(times)
    @info "Animating $nframes frames at $fps fps"

    sfc_lo, sfc_hi = 350f0, 500f0
    hpa_lo, hpa_hi = 380f0, 430f0
    xco2_lo, xco2_hi = 405f0, 425f0

    lon2d, lat2d = lon_lat_meshes(rmap)

    fig = Figure(size=(1400, 1200), fontsize=13)

    ax11 = GeoAxis(fig[1, 1]; dest="+proj=robin", title="GEOS-Chem — Surface CO2")
    ax12 = GeoAxis(fig[1, 2]; dest="+proj=robin", title="AtmosTransport — Surface CO2")
    ax21 = GeoAxis(fig[2, 1]; dest="+proj=robin", title="GEOS-Chem — CO2 ~750 hPa")
    ax22 = GeoAxis(fig[2, 2]; dest="+proj=robin", title="AtmosTransport — CO2 ~750 hPa")
    ax31 = GeoAxis(fig[3, 1]; dest="+proj=robin", title="GEOS-Chem — XCO2 (column avg)")
    ax32 = GeoAxis(fig[3, 2]; dest="+proj=robin", title="AtmosTransport — XCO2 (column avg)")

    z_sfc_gc  = Observable(gc.sfc_co2[:,:,1]')
    z_sfc_at  = Observable(at.sfc_co2[:,:,1]')
    z_hpa_gc  = Observable(gc.hpa_co2[:,:,1]')
    z_hpa_at  = Observable(at.hpa_co2[:,:,1]')
    z_xco2_gc = Observable(gc.xco2_col[:,:,1]')
    z_xco2_at = Observable(at.xco2_col[:,:,1]')

    sf1 = surface!(ax11, lon2d, lat2d, z_sfc_gc; shading=NoShading,
        colormap=Reverse(:RdYlBu), colorrange=(sfc_lo, sfc_hi))
    surface!(ax12, lon2d, lat2d, z_sfc_at; shading=NoShading,
        colormap=Reverse(:RdYlBu), colorrange=(sfc_lo, sfc_hi))

    sf2 = surface!(ax21, lon2d, lat2d, z_hpa_gc; shading=NoShading,
        colormap=Reverse(:RdYlBu), colorrange=(hpa_lo, hpa_hi))
    surface!(ax22, lon2d, lat2d, z_hpa_at; shading=NoShading,
        colormap=Reverse(:RdYlBu), colorrange=(hpa_lo, hpa_hi))

    sf3 = surface!(ax31, lon2d, lat2d, z_xco2_gc; shading=NoShading,
        colormap=Reverse(:RdYlBu), colorrange=(xco2_lo, xco2_hi))
    surface!(ax32, lon2d, lat2d, z_xco2_at; shading=NoShading,
        colormap=Reverse(:RdYlBu), colorrange=(xco2_lo, xco2_hi))

    for ax in [ax11, ax12, ax21, ax22, ax31, ax32]
        lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    end

    Colorbar(fig[1, 3], sf1; label="CO2 [ppm]", width=14)
    Colorbar(fig[2, 3], sf2; label="CO2 [ppm]", width=14)
    Colorbar(fig[3, 3], sf3; label="XCO2 [ppm]", width=14)

    title_obs = Observable(Dates.format(times[1], "yyyy-mm-dd HH:MM") *
        " UTC — CATRINE: AtmosTransport (GCHP v4 + RAS + PBL) vs GEOS-Chem")
    Label(fig[0, 1:3], title_obs; fontsize=16, font=:bold)

    @info "Writing $nframes frames to $OUT_GIF"
    Makie.record(fig, OUT_GIF, 1:nframes; framerate=fps) do fn
        z_sfc_gc[]  = gc.sfc_co2[:,:,fn]'
        z_sfc_at[]  = at.sfc_co2[:,:,fn]'
        z_hpa_gc[]  = gc.hpa_co2[:,:,fn]'
        z_hpa_at[]  = at.hpa_co2[:,:,fn]'
        z_xco2_gc[] = gc.xco2_col[:,:,fn]'
        z_xco2_at[] = at.xco2_col[:,:,fn]'
        title_obs[] = Dates.format(times[fn], "yyyy-mm-dd HH:MM") *
            " UTC — CATRINE: AtmosTransport (GCHP v4 + RAS + PBL) vs GEOS-Chem"
    end
    @info "Saved: $OUT_GIF ($nframes frames)"
end

function main()
    gc_files = sort(filter(f -> endswith(f, ".nc4"), readdir(GC_DIR)))
    @info "Loading CS coordinates..."
    cs_lons, cs_lats = load_cs_coordinates(joinpath(GC_DIR, gc_files[1]))
    @info "Building CS -> lat-lon map (1 deg)..."
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

    # Load hybrid ak/bk from our output for dp weights
    @info "Loading hybrid coordinates for XCO2 weighting..."
    at_files = sort(filter(f -> endswith(f, ".nc") && contains(f, AT_PATTERN),
                            readdir(AT_DIR)))
    ak, bk = NCDataset(joinpath(AT_DIR, at_files[1]), "r") do ds
        Float64.(ds["hyai"][:]), Float64.(ds["hybi"][:])
    end
    dp_wts = hybrid_dp_weights(ak, bk)
    @info "dp weights: $(length(dp_wts)) levels, total=$(round(sum(dp_wts)/100, digits=1)) hPa"

    @info "Loading GEOS-Chem..."
    gc = load_geoschem(GC_DIR, rmap, dp_wts; date_start=DATE_START, date_end=DATE_END)
    @info "Loading AtmosTransport..."
    at = load_atmostr(AT_DIR, rmap, gc.times, dp_wts)
    make_animation(gc, at, gc.times, rmap)
end

main()
