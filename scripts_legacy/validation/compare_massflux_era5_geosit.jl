#!/usr/bin/env julia
# ===========================================================================
# Single-timestep comparison: ERA5 vs GEOS-IT C180 mass fluxes
#
# Loads one timestep from each and compares:
#   1. Air mass (Δp-based), total atmospheric mass
#   2. Horizontal mass fluxes: magnitudes, global mean, extremes
#   3. Column mass conservation (∑divergence ≈ 0)
#
# Usage:
#   julia --project=. scripts/compare_massflux_era5_geosit.jl
# ===========================================================================

using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection
using AtmosTransport.Parameters
using AtmosTransport.IO: default_met_config, build_vertical_coordinate,
                              load_vertical_coefficients,
                              read_geosfp_cs_timestep
using NCDatasets
using Printf
using Statistics

const FT = Float32
const LEVEL_TOP = 50
const LEVEL_BOT = 137
const LEVEL_RANGE = LEVEL_TOP:LEVEL_BOT

const ERA5_FILE = expanduser(get(ENV, "ERA5_FILE",
    "~/data/metDrivers/era5/era5_ml_10deg_20230601_20230630/era5_ml_20230601.nc"))
const GEOSIT_FILE = expanduser(get(ENV, "GEOSIT_FILE",
    "~/data/geosit_c180/20230601/GEOSIT.20230601.CTM_A1.C180.nc"))

# --- ERA5 reader (same as preprocess_mass_fluxes.jl) ---
function load_era5_timestep(filepath::String, tidx::Int, ::Type{FT}) where {FT}
    ds = NCDataset(filepath)
    try
        u = FT.(ds["u"][:, :, :, tidx])[:, end:-1:1, :]
        v = FT.(ds["v"][:, :, :, tidx])[:, end:-1:1, :]

        # lnsp may be in the main file or in a separate .lnsp_tmp file
        if haskey(ds, "lnsp")
            lnsp = FT.(ds["lnsp"][:, :, tidx])[:, end:-1:1]
        else
            lnsp_file = filepath * ".lnsp_tmp"
            isfile(lnsp_file) || error("No lnsp in $filepath and no $lnsp_file found")
            @info "Reading lnsp from separate file: $(basename(lnsp_file))"
            ds_lnsp = NCDataset(lnsp_file)
            # lnsp has shape (lon, lat, model_level=1, time) — squeeze out level dim
            lnsp_raw = FT.(ds_lnsp["lnsp"][:, :, 1, tidx])[:, end:-1:1]
            close(ds_lnsp)
            lnsp = lnsp_raw
        end
        return u, v, exp.(lnsp)
    finally
        close(ds)
    end
end

function get_era5_info(filepath::String, ::Type{FT}) where {FT}
    ds = NCDataset(filepath)
    try
        lons = FT.(ds["longitude"][:])
        lats = FT.(ds["latitude"][:])
        levs = ds["model_level"][:]
        Nt   = length(ds["valid_time"][:])
        return lons, reverse(lats), levs, length(lons), length(lats), length(levs), Nt
    finally
        close(ds)
    end
end

function stagger_winds!(u_stag, v_stag, u_cc, v_cc, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = i == Nx ? 1 : i + 1
        u_stag[i, j, k] = (u_cc[i, j, k] + u_cc[ip, j, k]) / 2
    end
    u_stag[Nx + 1, :, :] .= u_stag[1, :, :]
    @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
        v_stag[i, j, k] = (v_cc[i, j - 1, k] + v_cc[i, j, k]) / 2
    end
    v_stag[:, 1, :] .= 0
    v_stag[:, Ny + 1, :] .= 0
    return nothing
end

# --- Pretty-print helpers ---
function stats_str(arr; name="")
    mn = minimum(arr); mx = maximum(arr)
    μ = mean(arr); σ = std(arr)
    absmax = max(abs(mn), abs(mx))
    @sprintf("  %-12s min=%+12.4e  max=%+12.4e  mean=%+12.4e  std=%10.4e  |max|=%10.4e",
             name, mn, mx, μ, σ, absmax)
end

function section(title)
    println("\n", "=" ^ 72)
    println(" ", title)
    println("=" ^ 72)
end

# ===========================================================================
function main()
    isfile(ERA5_FILE) || error("ERA5 file not found: $ERA5_FILE")
    isfile(GEOSIT_FILE) || error("GEOS-IT file not found: $GEOSIT_FILE")

    # ---------------------------------------------------------------
    section("ERA5 — Computing mass fluxes from winds (timestep 1)")
    # ---------------------------------------------------------------
    config = default_met_config("era5")
    vc = build_vertical_coordinate(config; FT, level_range=LEVEL_RANGE)
    A_full, B_full = load_vertical_coefficients(config; FT)
    A_coeff = A_full[LEVEL_TOP:LEVEL_BOT+1]
    B_coeff = B_full[LEVEL_TOP:LEVEL_BOT+1]

    lons, lats, levs, Nx, Ny, Nz, Nt = get_era5_info(ERA5_FILE, FT)
    @info "ERA5 grid: Nx=$Nx, Ny=$Ny, Nz=$Nz (levels $LEVEL_TOP-$LEVEL_BOT)"
    @info "ERA5 lon range: $(lons[1]) to $(lons[end]), lat range: $(lats[1]) to $(lats[end])"

    params = load_parameters(FT)
    pp = params.planet
    arch = CPU()
    grid = LatitudeLongitudeGrid(arch;
        FT, size = (Nx, Ny, Nz),
        longitude = (FT(lons[1]), FT(lons[end]) + FT(lons[2] - lons[1])),
        latitude = (FT(-90), FT(90)),
        vertical = vc,
        radius = pp.radius, gravity = pp.gravity,
        reference_pressure = pp.reference_surface_pressure)

    DT = FT(900)
    half_dt = DT / 2
    u_stag = Array{FT}(undef, Nx + 1, Ny, Nz)
    v_stag = Array{FT}(undef, Nx, Ny + 1, Nz)
    Δp = Array{FT}(undef, Nx, Ny, Nz)
    m_era5 = Array{FT}(undef, Nx, Ny, Nz)
    am_era5 = Array{FT}(undef, Nx + 1, Ny, Nz)
    bm_era5 = Array{FT}(undef, Nx, Ny + 1, Nz)
    cm_era5 = Array{FT}(undef, Nx, Ny, Nz + 1)

    m_dummy = Array{FT}(undef, Nx, Ny, Nz)
    gc = build_geometry_cache(grid, m_dummy)

    u_cc, v_cc, ps_data = load_era5_timestep(ERA5_FILE, 1, FT)
    stagger_winds!(u_stag, v_stag, u_cc, v_cc, Nx, Ny, Nz)
    Advection._build_Δz_3d!(Δp, grid, ps_data)
    compute_air_mass!(m_era5, Δp, gc)
    compute_mass_fluxes!(am_era5, bm_era5, cm_era5, u_stag, v_stag, gc, Δp, half_dt)

    println("\nERA5 mass flux statistics (kg per half-timestep, dt/2=$(half_dt)s):")
    println(stats_str(m_era5;  name="m (air mass)"))
    println(stats_str(am_era5; name="am (east)"))
    println(stats_str(bm_era5; name="bm (north)"))
    println(stats_str(cm_era5; name="cm (down)"))
    println(stats_str(ps_data; name="ps (Pa)"))
    println(stats_str(Δp;      name="Δp (Pa)"))

    total_mass_era5 = sum(m_era5)
    @info @sprintf("ERA5 total atmospheric mass: %.4e kg", total_mass_era5)
    @info @sprintf("ERA5 expected (5.15e18 kg): ratio = %.4f", total_mass_era5 / 5.15e18)

    # Check column mass conservation: cm at surface should be ~0
    cm_surface = cm_era5[:, :, end]
    @info @sprintf("ERA5 cm at surface: max|cm_surface| = %.4e kg", maximum(abs.(cm_surface)))

    # ---------------------------------------------------------------
    section("GEOS-IT C180 — Reading mass fluxes directly (timestep 1)")
    # ---------------------------------------------------------------
    ts = read_geosfp_cs_timestep(GEOSIT_FILE; FT, time_index=1, convert_to_kgs=false)
    @info "GEOS-IT grid: Nc=$(ts.Nc), Nz=$(ts.Nz), 6 panels"

    # Raw units: Pa⋅m² accumulated over 3600s
    println("\nGEOS-IT raw mass flux statistics (Pa⋅m², accumulated 3600s):")
    for p in 1:6
        mn_mfx = minimum(ts.mfxc[p]); mx_mfx = maximum(ts.mfxc[p])
        mn_mfy = minimum(ts.mfyc[p]); mx_mfy = maximum(ts.mfyc[p])
        mn_delp = minimum(ts.delp[p]); mx_delp = maximum(ts.delp[p])
        mn_ps = minimum(ts.ps[p]); mx_ps = maximum(ts.ps[p])
        @printf("  Panel %d: MFXC=[%+.3e, %+.3e]  MFYC=[%+.3e, %+.3e]  DELP=[%.1f, %.1f]  PS=[%.0f, %.0f]\n",
                p, mn_mfx, mx_mfx, mn_mfy, mx_mfy, mn_delp, mx_delp, mn_ps, mx_ps)
    end

    # Convert to kg/s for better comparison
    g = FT(9.80665)
    dt_met = FT(3600)
    conv = FT(1) / (g * dt_met)

    println("\nGEOS-IT converted mass flux statistics (kg/s, after ÷ g⋅Δt):")
    for p in 1:6
        mfx_kgs = ts.mfxc[p] .* conv
        mfy_kgs = ts.mfyc[p] .* conv
        @printf("  Panel %d: MFXC=[%+.3e, %+.3e]  MFYC=[%+.3e, %+.3e] kg/s\n",
                p, minimum(mfx_kgs), maximum(mfx_kgs), minimum(mfy_kgs), maximum(mfy_kgs))
    end

    # GEOS-IT air mass: DELP in Pa → m = DELP × area / g
    # For C180 cubed sphere, each cell area ≈ (R_earth/Nc)² ≈ (6371e3/180)² ≈ 1.25e9 m²
    R_earth = FT(6.371e6)
    cell_area_approx = (4π * R_earth^2) / (6 * ts.Nc^2)  # total sphere area / total cells
    @info @sprintf("GEOS-IT approx cell area: %.4e m²", cell_area_approx)

    total_mass_geosit = FT(0)
    for p in 1:6
        total_mass_geosit += sum(ts.delp[p]) * cell_area_approx / g
    end
    @info @sprintf("GEOS-IT total atmospheric mass (approx): %.4e kg", total_mass_geosit)
    @info @sprintf("GEOS-IT expected (5.15e18 kg): ratio = %.4f", total_mass_geosit / 5.15e18)

    # DELP statistics
    println("\nGEOS-IT DELP statistics (Pa):")
    all_delp = vcat([vec(ts.delp[p]) for p in 1:6]...)
    println(stats_str(all_delp; name="DELP"))
    all_ps = vcat([vec(ts.ps[p]) for p in 1:6]...)
    println(stats_str(all_ps; name="PS"))

    # ---------------------------------------------------------------
    section("CROSS-COMPARISON")
    # ---------------------------------------------------------------

    # 1. Surface pressure
    era5_ps_mean = mean(ps_data)
    geosit_ps_mean = mean(all_ps)
    println("\nSurface pressure (Pa):")
    @printf("  ERA5:    mean=%.1f  min=%.1f  max=%.1f\n",
            era5_ps_mean, minimum(ps_data), maximum(ps_data))
    @printf("  GEOS-IT: mean=%.1f  min=%.1f  max=%.1f\n",
            geosit_ps_mean, minimum(all_ps), maximum(all_ps))
    @printf("  Note: GEOS-IT PS may be in hPa (×100 if < 2000)\n")

    # 2. Pressure thickness
    era5_dp_mean = mean(Δp)
    geosit_dp_mean = mean(all_delp)
    println("\nPressure thickness (Pa):")
    @printf("  ERA5 Δp:     mean=%.2f  min=%.2f  max=%.2f\n",
            era5_dp_mean, minimum(Δp), maximum(Δp))
    @printf("  GEOS-IT DELP: mean=%.2f  min=%.2f  max=%.2f\n",
            geosit_dp_mean, minimum(all_delp), maximum(all_delp))

    # 3. Total atmospheric mass
    @printf("\nTotal atmospheric mass:\n")
    @printf("  ERA5:    %.4e kg\n", total_mass_era5)
    @printf("  GEOS-IT: %.4e kg (approx, uniform cell area)\n", total_mass_geosit)
    @printf("  Earth:   5.15e+18 kg (expected)\n")
    @printf("  Ratio GEOS-IT/ERA5: %.4f\n", total_mass_geosit / total_mass_era5)

    # 4. Mass flux magnitudes
    # ERA5 am is in kg/half-timestep. Convert to kg/s for comparison:
    era5_am_kgs = am_era5 ./ half_dt  # kg/s
    era5_bm_kgs = bm_era5 ./ half_dt  # kg/s

    # GEOS-IT: already computed mfx_kgs above, let's get global stats
    all_mfxc_kgs = vcat([vec(ts.mfxc[p] .* conv) for p in 1:6]...)
    all_mfyc_kgs = vcat([vec(ts.mfyc[p] .* conv) for p in 1:6]...)

    println("\nHorizontal mass fluxes (kg/s):")
    @printf("  ERA5 am:      |max|=%.4e  mean=%.4e  std=%.4e\n",
            maximum(abs.(era5_am_kgs)), mean(era5_am_kgs), std(era5_am_kgs))
    @printf("  GEOS-IT MFXC: |max|=%.4e  mean=%.4e  std=%.4e\n",
            maximum(abs.(all_mfxc_kgs)), mean(all_mfxc_kgs), std(all_mfxc_kgs))
    @printf("  ERA5 bm:      |max|=%.4e  mean=%.4e  std=%.4e\n",
            maximum(abs.(era5_bm_kgs)), mean(era5_bm_kgs), std(era5_bm_kgs))
    @printf("  GEOS-IT MFYC: |max|=%.4e  mean=%.4e  std=%.4e\n",
            maximum(abs.(all_mfyc_kgs)), mean(all_mfyc_kgs), std(all_mfyc_kgs))

    # 5. ERA5 vertical flux
    era5_cm_kgs = cm_era5 ./ half_dt
    println("\nVertical mass flux (kg/s) — ERA5 only (GEOS-IT cm is runtime-computed):")
    @printf("  ERA5 cm: |max|=%.4e  mean=%.4e  std=%.4e\n",
            maximum(abs.(era5_cm_kgs)), mean(era5_cm_kgs), std(era5_cm_kgs))

    # 6. Quick magnitude comparison
    section("SUMMARY — Are magnitudes consistent?")

    # Scale factor: ERA5 is ~1° lat-lon, GEOS-IT C180 is ~2° CS
    # Cell area ratio: ERA5 cell ≈ (111km × cos(lat) × 111km), GEOS-IT C180 cell ≈ 35000m²
    # ERA5 mass flux is per lat-lon face, GEOS-IT is per CS face
    # They shouldn't match 1:1 due to different grid geometries, but should be same order of magnitude

    era5_mf_scale = std(era5_am_kgs)
    geosit_mf_scale = std(all_mfxc_kgs)
    ratio = geosit_mf_scale / era5_mf_scale
    @printf("\n  Mass flux scale (std of am vs MFXC in kg/s):\n")
    @printf("    ERA5:    %.4e\n", era5_mf_scale)
    @printf("    GEOS-IT: %.4e\n", geosit_mf_scale)
    @printf("    Ratio (GEOS-IT/ERA5): %.2f\n", ratio)
    @printf("    Expected ratio ~ (Nc_cell_width/ERA5_cell_width)² ~ (2°/1°)² ~ 4\n")
    @printf("    (C180 cells are ~4× larger in area than 1° cells → ~4× larger mass flux per face)\n")

    println("\n  Check: both should be O(1e6–1e8 kg/s) for realistic atmospheric mass fluxes.")
    println("  If one is orders of magnitude different, there's a unit problem.")
    println()
end

main()
