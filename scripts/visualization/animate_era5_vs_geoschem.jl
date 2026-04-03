#!/usr/bin/env julia
# ===========================================================================
# Fast diagnostic animation: GeosChem vs AT-F64 vs AT-F32
# Uses image! on plain Axis (no GeoMakie), 110m coastlines, small figures.
# ===========================================================================

using CairoMakie
using NCDatasets
using Dates
using Statistics
using GeoMakie  # only for coastlines()

include(joinpath(@__DIR__, "cs_regrid_utils.jl"))

const GC_DIR  = expanduser("~/data/AtmosTransport/catrine-geoschem-runs")
const AT_F32  = expanduser("~/data/AtmosTransport/catrine/output/era5_spectral_dec2021_pbl/catrine_era5_spectral_pbl.bin")
const AT_F64  = expanduser("~/data/AtmosTransport/catrine/output/era5_spectral_dec2021_pbl_f64/catrine_era5_spectral_pbl_f64.bin")
const OUT_MP4 = expanduser("~/data/AtmosTransport/catrine/output/era5_f32_f64_vs_geoschem.mp4")
const FPS = 6

const DATE_START = DateTime(2021, 12, 1, 6)
const DATE_END   = DateTime(2021, 12, 7, 21)
const LEV_SFC_GC = 1
const LEV_750_GC = 15
const LEV_750_FRAC = 0.75

# ---------------------------------------------------------------------------
# Colormap helper: map data to RGB image
# ---------------------------------------------------------------------------
function data_to_image(data::Matrix{Float32}, lo, hi, cmap)
    nr, nc = size(data)
    normed = clamp.((data .- lo) ./ (hi - lo), 0f0, 1f0)
    img = Array{RGBf}(undef, nc, nr)  # transposed for image display
    for j in 1:nc, i in 1:nr
        img[nc+1-j, i] = Makie.interpolated_getindex(cmap, normed[i, j])
    end
    return img
end

# ---------------------------------------------------------------------------
# Downsample 720×361 → 360×180
# ---------------------------------------------------------------------------
function downsample_ll(data::AbstractMatrix)
    Nx, Ny = size(data)
    out = zeros(Float32, Nx ÷ 2, min(Ny, 360) ÷ 2)
    for j in 1:size(out,2), i in 1:size(out,1)
        i2, j2 = 2i-1, 2j-1
        out[i,j] = (data[i2,j2] + data[i2+1,j2] + data[i2,j2+1] + data[i2+1,j2+1]) / 4f0
    end
    return out
end

# ---------------------------------------------------------------------------
# Load GeosChem C180 → 1° LL
# ---------------------------------------------------------------------------
function load_geoschem(gc_dir, rmap, times)
    files = sort(filter(f -> endswith(f, ".nc4") && contains(f, "CATRINE_inst"), readdir(gc_dir)))
    nt = length(times)
    buf = zeros(Float32, rmap.nlon, rmap.nlat)
    sfc = zeros(Float32, rmap.nlon, rmap.nlat, nt)
    hpa = zeros(Float32, rmap.nlon, rmap.nlat, nt)
    xco2 = zeros(Float32, rmap.nlon, rmap.nlat, nt)

    for (ti, tgt) in enumerate(times)
        best_f, best_dt = nothing, Inf
        for f in files
            m = match(r"(\d{8})_(\d{4})z", f)
            m === nothing && continue
            d = abs(Dates.value(DateTime(m[1]*m[2], dateformat"yyyymmddHHMM") - tgt)) / 60_000
            d < best_dt && (best_dt = d; best_f = f)
        end
        (best_f === nothing || best_dt > 120) && continue
        try
            NCDataset(joinpath(gc_dir, best_f)) do ds
                co2 = ds["SpeciesConcVV_CO2"]
                regrid_cs!(buf, Float32.(co2[:,:,:,LEV_SFC_GC,1]) .* 1f6, rmap); sfc[:,:,ti] .= buf
                regrid_cs!(buf, Float32.(co2[:,:,:,LEV_750_GC,1]) .* 1f6, rmap); hpa[:,:,ti] .= buf
                vmr = co2[:,:,:,:,1]; Nc,_,Np,Nz = size(vmr)
                xcs = zeros(Float32, Nc, Nc, Np)
                for p in 1:Np, j in 1:Nc, i in 1:Nc
                    s = 0.0; for k in 1:Nz; s += vmr[i,j,p,k]; end
                    xcs[i,j,p] = Float32(s/Nz*1e6)
                end
                regrid_cs!(buf, xcs, rmap); xco2[:,:,ti] .= buf
            end
        catch e
            @warn "Skip GC: $e" maxlog=3
        end
    end
    return (; sfc, hpa, xco2)
end

# ---------------------------------------------------------------------------
# Load AT (NC on disk, 720×361 → 360×180)
# ---------------------------------------------------------------------------
function load_at(path, times)
    ds = NCDataset(path)
    at_t = ds["time"][:]; Nz = ds.dim["lev"]
    lev_sfc, lev_750 = Nz, max(1, round(Int, Nz * LEV_750_FRAC))
    nt = length(times); nlon, nlat = 360, 180
    sfc = zeros(Float32, nlon, nlat, nt)
    hpa = zeros(Float32, nlon, nlat, nt)
    xco2 = zeros(Float32, nlon, nlat, nt)
    for (ti, tgt) in enumerate(times)
        diffs = [abs(Dates.value(a - tgt))/60_000 for a in at_t]
        bi = argmin(diffs); diffs[bi] > 120 && continue
        co2 = Float32.(ds["co2_3d"][:,:,:,bi]) .* 1f6
        sfc[:,:,ti] .= downsample_ll(co2[:,:,lev_sfc])
        hpa[:,:,ti] .= downsample_ll(co2[:,:,lev_750])
        Nx,Ny,Nzz = size(co2)
        xf = zeros(Float32, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            s=0f0; for k in 1:Nzz; s+=co2[i,j,k]; end; xf[i,j]=s/Nzz
        end
        xco2[:,:,ti] .= downsample_ll(xf)
    end
    close(ds)
    @info "AT $(basename(path)): $nt steps, Nz=$Nz"
    return (; sfc, hpa, xco2)
end

# ---------------------------------------------------------------------------
# Pre-extract coastline coordinates
# ---------------------------------------------------------------------------
function get_coastlines()
    # Extract coordinates from GeoMakie's bundled coastlines
    coast = GeoMakie.coastlines()
    xs, ys = Float32[], Float32[]
    for geom in coast.geometry
        coords = GeoMakie.geo2basic(geom)
        for pt in coords
            push!(xs, Float32(pt[1])); push!(ys, Float32(pt[2]))
        end
        push!(xs, NaN32); push!(ys, NaN32)
    end
    return xs, ys
end

# ---------------------------------------------------------------------------
# Animation: 4 rows × 3 cols, image! on plain Axis
# ---------------------------------------------------------------------------
function animate(gc, at64, at32, times)
    nf = length(times)
    @info "Animating $nf frames"

    coast_x, coast_y = get_coastlines()

    cmap_co2 = Makie.to_colormap(Reverse(:RdYlBu))
    cmap_diff = Makie.to_colormap(:RdBu)

    sfc_rng  = (380f0, 460f0)
    hpa_rng  = (400f0, 418f0)
    xco2_rng = (408f0, 416f0)
    diff_rng = (-0.3f0, 0.3f0)

    fig = Figure(size=(1200, 1000), fontsize=10)
    title_obs = Observable("")
    Label(fig[1, 1:4], title_obs; fontsize=14, font=:bold)

    col_labels = ["GEOS-Chem", "AT F64", "AT F32"]
    row_labels = ["Surface CO₂", "~750 hPa", "XCO₂", "F64−F32"]

    axes = Matrix{Axis}(undef, 4, 3)
    for r in 1:4, c in 1:3
        ttl = r == 1 ? col_labels[c] : ""
        axes[r,c] = Axis(fig[r+1, c]; title=ttl, aspect=DataAspect(),
            limits=(-180, 180, -90, 90),
            xticklabelsvisible=false, yticklabelsvisible=false,
            xticksvisible=false, yticksvisible=false)
    end
    for (r, lab) in enumerate(row_labels)
        Label(fig[r+1, 0], lab; fontsize=11, font=:bold, rotation=π/2, tellheight=false)
    end

    lon_rng = (-180f0, 180f0)
    lat_rng = (-90f0, 90f0)

    # Build initial images
    all_data = [
        (gc.sfc, sfc_rng, cmap_co2), (at64.sfc, sfc_rng, cmap_co2), (at32.sfc, sfc_rng, cmap_co2),
        (gc.hpa, hpa_rng, cmap_co2), (at64.hpa, hpa_rng, cmap_co2), (at32.hpa, hpa_rng, cmap_co2),
        (gc.xco2, xco2_rng, cmap_co2), (at64.xco2, xco2_rng, cmap_co2), (at32.xco2, xco2_rng, cmap_co2),
        (at64.sfc .- at32.sfc, diff_rng, cmap_diff),
        (at64.hpa .- at32.hpa, diff_rng, cmap_diff),
        (at64.xco2 .- at32.xco2, diff_rng, cmap_diff),
    ]

    img_obs = Observable{Matrix{RGBf}}[]
    idx = 0
    for r in 1:3, c in 1:3
        idx += 1
        arr, (lo, hi), cm = all_data[idx]
        obs = Observable(data_to_image(arr[:,:,1], lo, hi, cm))
        push!(img_obs, obs)
        image!(axes[r,c], lon_rng..., lat_rng..., obs; interpolate=false)
    end
    # Row 4: diff
    for c in 1:3
        idx += 1
        arr, (lo, hi), cm = all_data[idx]
        obs = Observable(data_to_image(arr[:,:,1], lo, hi, cm))
        push!(img_obs, obs)
        image!(axes[4,c], lon_rng..., lat_rng..., obs; interpolate=false)
    end

    # Coastlines (static, drawn once)
    for r in 1:4, c in 1:3
        lines!(axes[r,c], coast_x, coast_y; color=(:black, 0.4), linewidth=0.4)
    end

    # Colorbars
    Colorbar(fig[2, 4], colormap=Reverse(:RdYlBu), limits=sfc_rng, label="ppm", width=10)
    Colorbar(fig[3, 4], colormap=Reverse(:RdYlBu), limits=hpa_rng, label="ppm", width=10)
    Colorbar(fig[4, 4], colormap=Reverse(:RdYlBu), limits=xco2_rng, label="ppm", width=10)
    Colorbar(fig[5, 4], colormap=:RdBu, limits=diff_rng, label="Δppm", width=10)

    @info "Recording $nf frames → $OUT_MP4"
    record(fig, OUT_MP4, 1:nf; framerate=FPS) do fn
        idx = 0
        for r in 1:3, c in 1:3
            idx += 1
            arr, (lo, hi), cm = all_data[idx]
            img_obs[idx][] = data_to_image(arr[:,:,fn], lo, hi, cm)
        end
        for c in 1:3
            idx += 1
            arr, (lo, hi), cm = all_data[idx]
            img_obs[idx][] = data_to_image(arr[:,:,fn], lo, hi, cm)
        end
        title_obs[] = Dates.format(times[fn], "yyyy-mm-dd HH:MM") *
            " UTC — CATRINE: GeosChem vs F64 vs F32"
    end
    @info "Saved: $OUT_MP4"
end

function main()
    times = collect(DATE_START:Hour(3):DATE_END)
    @info "$(length(times)) snapshots"

    gc_files = sort(filter(f -> endswith(f, ".nc4") && contains(f, "CATRINE_inst"), readdir(GC_DIR)))
    cs_lons, cs_lats = load_cs_coordinates(joinpath(GC_DIR, gc_files[1]))
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

    @info "Loading GeosChem..."; gc = load_geoschem(GC_DIR, rmap, times)
    @info "Loading AT F64...";   at64 = load_at(AT_F64, times)
    @info "Loading AT F32...";   at32 = load_at(AT_F32, times)

    animate(gc, at64, at32, times)

    # Drift analysis
    nt = length(times)
    println("\n=== F64 − F32 Systematic Drift (7 days) ===")
    for (name, f64, f32) in [("Surface", at64.sfc, at32.sfc),
                              ("750hPa", at64.hpa, at32.hpa),
                              ("XCO2", at64.xco2, at32.xco2)]
        diffs = [mean(f64[:,:,t] .- f32[:,:,t]) for t in 1:nt]
        stds  = [std(f64[:,:,t] .- f32[:,:,t]) for t in 1:nt]
        trend = (diffs[end] - diffs[1]) / nt
        @info "$name: mean_Δ=$(round.(diffs[[1,nt÷2,nt]], sigdigits=3)) std=$(round.(stds[[1,nt÷2,nt]], sigdigits=3)) trend=$(round(trend,sigdigits=3))/step $(abs(trend)>0.001 ? "SYSTEMATIC" : "ok")"
    end
end

main()
