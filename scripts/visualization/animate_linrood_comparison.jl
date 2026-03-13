#!/usr/bin/env julia
# ===========================================================================
# 3-way comparison animation: Strang vs Lin-Rood (old) vs Lin-Rood (fixed)
#
# Layout (3 rows × 2 columns):
#   Row 1: Surface fossil CO2    (Strang | Lin-Rood old)
#   Row 2: Surface fossil CO2    (Lin-Rood fixed | Difference: fixed - old)
#   Row 3: fossil CO2 at ~750hPa (Strang | Lin-Rood fixed)
#
# Usage:
#   julia --project=. scripts/visualization/animate_linrood_comparison.jl
# ===========================================================================

using CairoMakie
using GeoMakie
using Dates

include(joinpath(@__DIR__, "cs_regrid_utils.jl"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const AT_DIR     = get(ENV, "AT_DIR", "/temp2/catrine-runs/output")
const OUT_GIF    = get(ENV, "OUT_GIF", "linrood_comparison.gif")
const FPS        = parse(Int, get(ENV, "FPS", "4"))

const LEV_SURFACE = 1
const LEV_750HPA  = 15

const DATE_START = DateTime(2021, 12, 1)
const DATE_END   = DateTime(2021, 12, 31, 21, 0, 0)

# File patterns for each version
const STRANG_PATTERN  = "catrine_geosit_c180_2021"
const LINROOD_OLD     = "catrine_geosit_c180_linrood_diffuse_2021"
const LINROOD_FIXED   = "catrine_geosit_c180_linrood_fixed_2021"

# ---------------------------------------------------------------------------
# Generic loader for AtmosTransport CS binary → NC output
# ---------------------------------------------------------------------------
function load_at_data(at_dir, pattern, rmap, var_name, levs;
                      date_start, date_end, label="",
                      exclude_pattern::String="")
    daily_files = sort(filter(f -> endswith(f, ".nc") && contains(f, pattern), readdir(at_dir)))
    if !isempty(exclude_pattern)
        daily_files = filter(f -> !contains(f, exclude_pattern), daily_files)
    end

    nl   = length(levs)
    nlon = rmap.nlon
    nlat = rmap.nlat
    buf  = zeros(Float32, nlon, nlat)

    all_times  = DateTime[]
    all_fields = [Vector{Matrix{Float32}}() for _ in 1:nl]

    for fname in daily_files
        NCDataset(joinpath(at_dir, fname), "r") do ds
            file_times = ds["time"][:]
            data_var = ds[var_name]
            for (ti, dt) in enumerate(file_times)
                date_start <= dt <= date_end || continue
                push!(all_times, dt)
                for (li, lev) in enumerate(levs)
                    data_cs = Float32.(data_var[:, :, :, lev, ti]) .* 1f6  # mol/mol → ppm
                    regrid_cs!(buf, data_cs, rmap)
                    push!(all_fields[li], copy(buf))
                end
            end
        end
    end

    perm = sortperm(all_times)
    all_times = all_times[perm]
    for li in 1:nl
        all_fields[li] = all_fields[li][perm]
    end

    nt = length(all_times)
    lbl = isempty(label) ? pattern : label
    @info "$lbl: $nt snapshots ($(all_times[1]) → $(all_times[end]))"

    fields = [zeros(Float32, nlon, nlat, nt) for _ in 1:nl]
    for li in 1:nl, ti in 1:nt
        fields[li][:, :, ti] .= all_fields[li][ti]
    end

    return (; times=all_times, fields)
end

# ---------------------------------------------------------------------------
# Find common timesteps across datasets
# ---------------------------------------------------------------------------
function intersect_times(datasets...)
    common = Set(datasets[1].times)
    for ds in datasets[2:end]
        intersect!(common, Set(ds.times))
    end
    sorted = sort(collect(common))

    results = map(datasets) do ds
        idx = [findfirst(==(t), ds.times) for t in sorted]
        fields = [ds.fields[l][:, :, idx] for l in eachindex(ds.fields)]
        (; times=sorted, fields)
    end

    return sorted, results...
end

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
function make_animation(strang, lr_old, lr_fixed, times, rmap; fps=FPS)
    nframes = length(times)
    @info "Animating $nframes frames at $fps fps"

    sfc_max = 20f0
    hpa_max = 5f0
    diff_max = 2f0

    lon2d, lat2d = lon_lat_meshes(rmap)

    fig = Figure(size=(1600, 1300), fontsize=12)

    # Row 1: Surface — Strang vs Lin-Rood old
    ax11 = GeoAxis(fig[1, 1]; dest="+proj=robin", title="Strang — Surface fossil CO2")
    ax12 = GeoAxis(fig[1, 2]; dest="+proj=robin", title="Lin-Rood (old) — Surface fossil CO2")

    # Row 2: Surface — Lin-Rood fixed vs Difference
    ax21 = GeoAxis(fig[2, 1]; dest="+proj=robin", title="Lin-Rood (fixed) — Surface fossil CO2")
    ax22 = GeoAxis(fig[2, 2]; dest="+proj=robin", title="Difference (fixed − old) — Surface")

    # Row 3: 750 hPa — Strang vs Lin-Rood fixed
    ax31 = GeoAxis(fig[3, 1]; dest="+proj=robin", title="Strang — fossil CO2 ~750 hPa")
    ax32 = GeoAxis(fig[3, 2]; dest="+proj=robin", title="Lin-Rood (fixed) — fossil CO2 ~750 hPa")

    # Observables
    z_s_strang = Observable(strang.fields[1][:, :, 1]')
    z_s_old    = Observable(lr_old.fields[1][:, :, 1]')
    z_s_fixed  = Observable(lr_fixed.fields[1][:, :, 1]')
    z_s_diff   = Observable((lr_fixed.fields[1][:, :, 1] .- lr_old.fields[1][:, :, 1])')
    z_h_strang = Observable(strang.fields[2][:, :, 1]')
    z_h_fixed  = Observable(lr_fixed.fields[2][:, :, 1]')

    # Surface plots (rows 1-2)
    sf1 = surface!(ax11, lon2d, lat2d, z_s_strang;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, sfc_max), colorscale=safe_sqrt)
    surface!(ax12, lon2d, lat2d, z_s_old;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, sfc_max), colorscale=safe_sqrt)
    surface!(ax21, lon2d, lat2d, z_s_fixed;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, sfc_max), colorscale=safe_sqrt)

    # Difference plot
    sf_diff = surface!(ax22, lon2d, lat2d, z_s_diff;
        shading=NoShading, colormap=:RdBu, colorrange=(-diff_max, diff_max))

    # 750 hPa plots (row 3)
    sf3 = surface!(ax31, lon2d, lat2d, z_h_strang;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, hpa_max), colorscale=safe_sqrt)
    surface!(ax32, lon2d, lat2d, z_h_fixed;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, hpa_max), colorscale=safe_sqrt)

    # Coastlines
    for ax in [ax11, ax12, ax21, ax22, ax31, ax32]
        lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    end

    # Colorbars
    Colorbar(fig[1, 3], sf1;
        label="Surface fossil CO2 [ppm]", width=16,
        ticks=range(0, sfc_max, length=6) .|> (x -> round(x, digits=1)))
    Colorbar(fig[2, 3], sf_diff;
        label="Difference [ppm]", width=16,
        ticks=range(-diff_max, diff_max, length=5) .|> (x -> round(x, digits=1)))
    Colorbar(fig[3, 3], sf3;
        label="fossil CO2 at ~750 hPa [ppm]", width=16,
        ticks=range(0, hpa_max, length=6) .|> (x -> round(x, digits=2)))

    title_obs = Observable(Dates.format(times[1], "yyyy-mm-dd HH:MM") *
                           " UTC — Strang vs Lin-Rood Advection")
    Label(fig[0, 1:3], title_obs; fontsize=18, font=:bold)

    @info "Writing $nframes frames to $OUT_GIF"

    record(fig, OUT_GIF, 1:nframes; framerate=fps) do frame_num
        z_s_strang[] = strang.fields[1][:, :, frame_num]'
        z_s_old[]    = lr_old.fields[1][:, :, frame_num]'
        z_s_fixed[]  = lr_fixed.fields[1][:, :, frame_num]'
        z_s_diff[]   = (lr_fixed.fields[1][:, :, frame_num] .- lr_old.fields[1][:, :, frame_num])'
        z_h_strang[] = strang.fields[2][:, :, frame_num]'
        z_h_fixed[]  = lr_fixed.fields[2][:, :, frame_num]'

        title_obs[] = Dates.format(times[frame_num], "yyyy-mm-dd HH:MM") *
                      " UTC — Strang vs Lin-Rood Advection"
    end

    @info "Saved animation: $OUT_GIF ($nframes frames)"
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    # Load CS coordinates from first Strang output file
    first_nc = joinpath(AT_DIR, sort(filter(f -> endswith(f, ".nc") && contains(f, STRANG_PATTERN) &&
                        !contains(f, "linrood"), readdir(AT_DIR)))[1])
    @info "Loading CS coordinates from $first_nc"
    cs_lons, cs_lats = NCDataset(first_nc, "r") do ds
        Float64.(ds["lons"][:, :, :]), Float64.(ds["lats"][:, :, :])
    end

    @info "Building CS → lat-lon regridding map (1°)..."
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

    levs = [LEV_SURFACE, LEV_750HPA]
    var = "fossil_co2_3d"

    # Check which datasets are available
    strang_files = filter(f -> endswith(f, ".nc") && contains(f, STRANG_PATTERN) &&
                           !contains(f, "linrood") && !contains(f, "fixed"), readdir(AT_DIR))
    lr_old_files = filter(f -> endswith(f, ".nc") && contains(f, LINROOD_OLD), readdir(AT_DIR))
    lr_fix_files = filter(f -> endswith(f, ".nc") && contains(f, LINROOD_FIXED), readdir(AT_DIR))

    @info "Files found: Strang=$(length(strang_files)), LR_old=$(length(lr_old_files)), LR_fixed=$(length(lr_fix_files))"

    if isempty(lr_fix_files)
        @warn "Lin-Rood fixed output not yet available — run may still be in progress"
        @warn "Falling back to 2-way comparison: Strang vs Lin-Rood old"
    end

    @info "Loading Strang data..."
    strang = load_at_data(AT_DIR, STRANG_PATTERN, rmap, var, levs;
                          date_start=DATE_START, date_end=DATE_END, label="Strang",
                          exclude_pattern="linrood")

    @info "Loading Lin-Rood (old) data..."
    lr_old = load_at_data(AT_DIR, LINROOD_OLD, rmap, var, levs;
                          date_start=DATE_START, date_end=DATE_END, label="Lin-Rood old")

    if !isempty(lr_fix_files)
        @info "Loading Lin-Rood (fixed) data..."
        lr_fixed = load_at_data(AT_DIR, LINROOD_FIXED, rmap, var, levs;
                                date_start=DATE_START, date_end=DATE_END, label="Lin-Rood fixed")
    else
        lr_fixed = lr_old  # placeholder
    end

    # Align to common timesteps
    times, strang, lr_old, lr_fixed = intersect_times(strang, lr_old, lr_fixed)
    @info "Common timesteps: $(length(times))"

    make_animation(strang, lr_old, lr_fixed, times, rmap)
end

main()
