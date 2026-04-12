# ---------------------------------------------------------------------------
# Check vertical profiles of wind speed and pressure at specific locations
#
# Derives wind speed from CX/CY and cell geometry.
# If A3dyn file is available, also reads actual U, V, T.
#
# Usage:
#   julia --project=. scripts/check_profiles.jl
# ---------------------------------------------------------------------------

using NCDatasets, Statistics, Printf, LinearAlgebra

const R_EARTH = 6.371e6
const GRAV = 9.80665
const DT_MET = 3600.0
const DT_DYN = 450.0   # GEOS-IT C180 dynamics timestep (CX accumulation time)

# ── Gnomonic helpers (same as cubed_sphere_grid.jl) ───────────────────────
function gnomonic_xyz(ξ, η, p::Int)
    d = sqrt(1 + ξ^2 + η^2)
    if p == 1;     return ( 1/d, ξ/d, η/d)
    elseif p == 2;  return (-ξ/d, 1/d, η/d)
    elseif p == 3;  return (-ξ/d, -η/d, 1/d)
    elseif p == 4;  return (-1/d, -ξ/d, η/d)
    elseif p == 5;  return (ξ/d, -1/d, η/d)
    else            return (ξ/d, η/d, -1/d)
    end
end

function xyz_to_lonlat(x, y, z)
    lon = atand(y, x)
    lat = atand(z, sqrt(x^2 + y^2))
    return (lon, lat)
end

function compute_grid_info(Nc::Int)
    α_faces = range(-π/4, π/4, length=Nc+1)
    α_centers = [(α_faces[i] + α_faces[i+1]) / 2 for i in 1:Nc]

    lons = zeros(Nc, Nc, 6)
    lats = zeros(Nc, Nc, 6)
    areas = zeros(Nc, Nc)  # same for all panels
    dx = zeros(Nc, Nc)
    dy = zeros(Nc, Nc)

    for p in 1:6, j in 1:Nc, i in 1:Nc
        ξc = tan(α_centers[i])
        ηc = tan(α_centers[j])
        xyz = gnomonic_xyz(ξc, ηc, p)
        lon, lat = xyz_to_lonlat(xyz...)
        lons[i, j, p] = lon
        lats[i, j, p] = lat
    end

    # Compute areas and metric terms for panel 1 (same for all)
    for j in 1:Nc, i in 1:Nc
        ξ1, ξ2 = tan(α_faces[i]), tan(α_faces[i+1])
        η1, η2 = tan(α_faces[j]), tan(α_faces[j+1])

        # Δx: distance between midpoints of west and east edges
        mid_w = gnomonic_xyz(ξ1, tan(α_centers[j]), 1)
        mid_e = gnomonic_xyz(ξ2, tan(α_centers[j]), 1)
        dw = collect(mid_w); de = collect(mid_e)
        cos_dist = clamp(dot(dw, de) / (norm(dw) * norm(de)), -1.0, 1.0)
        dx[i, j] = R_EARTH * acos(cos_dist)

        # Δy: distance between midpoints of south and north edges
        mid_s = gnomonic_xyz(tan(α_centers[i]), η1, 1)
        mid_n = gnomonic_xyz(tan(α_centers[i]), η2, 1)
        ds = collect(mid_s); dn = collect(mid_n)
        cos_dist2 = clamp(dot(ds, dn) / (norm(ds) * norm(dn)), -1.0, 1.0)
        dy[i, j] = R_EARTH * acos(cos_dist2)

        areas[i, j] = dx[i, j] * dy[i, j]  # approximate
    end

    return (; lons, lats, areas, dx, dy)
end

# ── Find nearest grid cell to a target lat/lon ────────────────────────────
function find_nearest(lons, lats, target_lon, target_lat)
    Nc = size(lons, 1)
    best_dist = Inf
    best_i, best_j, best_p = 1, 1, 1
    for p in 1:6, j in 1:Nc, i in 1:Nc
        d = (lons[i,j,p] - target_lon)^2 + (lats[i,j,p] - target_lat)^2
        if d < best_dist
            best_dist = d
            best_i, best_j, best_p = i, j, p
        end
    end
    return best_i, best_j, best_p
end

# ── Read profiles from CTM_A1 ────────────────────────────────────────────
function read_ctm_profiles(filepath, gi, i, j, p; time_index=12)
    ds = NCDataset(filepath, "r")
    try
        ntime = size(ds["MFXC"], 5)
        ti = min(time_index, ntime)

        # Read single column
        mfxc = Float64.(ds["MFXC"][i, j, p, :, ti])
        mfyc = Float64.(ds["MFYC"][i, j, p, :, ti])
        delp = Float64.(ds["DELP"][i, j, p, :, ti])
        cx   = Float64.(ds["CX"][i, j, p, :, ti])
        cy   = Float64.(ds["CY"][i, j, p, :, ti])
        ps   = Float64(ds["PS"][i, j, p, ti])

        Nz = length(mfxc)

        # Check if needs vertical flip
        if delp[1] > 10 * delp[end]
            reverse!(mfxc); reverse!(mfyc); reverse!(delp)
            reverse!(cx); reverse!(cy)
        end

        # Compute pressure at level midpoints (top-to-bottom after flip)
        # p_mid[k] = p_top + sum(delp[1:k]) - delp[k]/2
        p_top = 0.01  # ~0.01 Pa at model top
        p_mid = zeros(Nz)
        p_cumsum = 0.0
        for k in 1:Nz
            p_cumsum += delp[k]
            p_mid[k] = p_top + p_cumsum - delp[k] / 2
        end

        # Derive wind speed from CX: u = CX × dx / dt_met
        area = gi.areas[i, j]
        dx_cell = gi.dx[i, j]
        dy_cell = gi.dy[i, j]

        # CX is Courant number per dynamics step (dt_dyn ≈ 450s for C180)
        # u = CX * dx / dt_dyn,  v = CY * dy / dt_dyn
        u_from_cx = cx .* dx_cell ./ DT_DYN
        v_from_cy = cy .* dy_cell ./ DT_DYN
        wspd_from_c = sqrt.(u_from_cx.^2 .+ v_from_cy.^2)

        # Also derive from MFXC: u = MFXC / (DELP × dy × dt_met)
        u_from_mfxc = mfxc ./ (delp .* dy_cell .* DT_DYN)
        # MFXC / (DELP × area) = CX, and CX = u*dt_dyn/dx
        # So u = CX * dx / dt_dyn = MFXC / (DELP * dy * dt_dyn)
        v_from_mfyc = mfyc ./ (delp .* dx_cell .* DT_DYN)
        wspd_from_mf = sqrt.(u_from_mfxc.^2 .+ v_from_mfyc.^2)

        return (; p_mid, delp, cx, cy, mfxc, mfyc, ps,
                  u_from_cx, v_from_cy, wspd_from_c,
                  u_from_mfxc, v_from_mfyc, wspd_from_mf,
                  dx_cell, dy_cell, area, Nz)
    finally
        close(ds)
    end
end

# ── Read profiles from A3dyn (U, V, T) if available ──────────────────────
function read_a3dyn_profiles(filepath, i, j, p; time_index=5, flip=true)
    # time_index=5 for 3-hourly data → ~12:00 UTC (indices: 0,3,6,9,12,15,18,21)
    ds = NCDataset(filepath, "r")
    try
        ntime = size(ds["U"], 5)
        ti = min(time_index, ntime)

        u = Float64.(ds["U"][i, j, p, :, ti])
        v = Float64.(ds["V"][i, j, p, :, ti])
        Nz = length(u)

        has_omega = haskey(ds, "OMEGA")
        omega = has_omega ? Float64.(ds["OMEGA"][i, j, p, :, ti]) : zeros(Nz)

        # GEOS-IT stores levels bottom-to-top (k=1=surface, k=72=TOA).
        # Flip to top-to-bottom (k=1=TOA, k=Nz=surface) to match CTM_A1 convention.
        if flip
            reverse!(u); reverse!(v); reverse!(omega)
        end

        wspd = sqrt.(u.^2 .+ v.^2)
        return (; u, v, wspd, omega, Nz)
    finally
        close(ds)
    end
end

# ── Main ──────────────────────────────────────────────────────────────────
function main()
    ctm_file = expanduser("~/data/geosit_c180/20230601/GEOSIT.20230601.CTM_A1.C180.nc")
    a3dyn_file = expanduser("~/data/geosit_c180/20230601/GEOSIT.20230601.A3dyn.C180.nc")

    Nc = 180
    println("Computing grid info for C$Nc...")
    gi = compute_grid_info(Nc)

    # Locations to check
    locations = [
        ("Kansas, US",    39.0, -97.0),
        ("Beijing, CN",   40.0, 116.0),
        ("Sahara",        25.0,   5.0),
        ("S. Ocean 60S",  -60.0, 0.0),
    ]

    for (name, target_lat, target_lon) in locations
        i, j, p = find_nearest(gi.lons, gi.lats, target_lon, target_lat)
        actual_lon = gi.lons[i, j, p]
        actual_lat = gi.lats[i, j, p]

        println("\n" * "=" ^ 60)
        @printf("  %s (target: %.1f°N, %.1f°E)\n", name, target_lat, target_lon)
        @printf("  Grid cell: panel=%d, i=%d, j=%d (%.2f°N, %.2f°E)\n",
                p, i, j, actual_lat, actual_lon)
        @printf("  dx=%.1f km, dy=%.1f km, area=%.3e m²\n",
                gi.dx[i,j]/1e3, gi.dy[i,j]/1e3, gi.areas[i,j])
        println("=" ^ 60)

        prof = read_ctm_profiles(ctm_file, gi, i, j, p; time_index=13)  # ~12:30 UTC
        @printf("  Surface pressure: %.1f hPa\n", prof.ps)

        # Print profile table
        println("\n  k  | p_mid [hPa] | DELP [Pa] | CX      | wind_CX [m/s] | wind_MF [m/s]")
        println("  " * "-" ^ 75)

        # Print selected levels (surface, PBL, mid-trop, jet, strat, top)
        levels_to_show = [prof.Nz, prof.Nz-1, prof.Nz-3, prof.Nz-7,
                          prof.Nz-15, div(prof.Nz,2), div(prof.Nz,4),
                          div(prof.Nz,8), 5, 1]
        for k in levels_to_show
            k < 1 && continue
            @printf("  %2d | %10.2f | %9.1f | %7.4f | %13.1f | %13.1f\n",
                    k, prof.p_mid[k]/100, prof.delp[k],
                    sqrt(prof.cx[k]^2 + prof.cy[k]^2),
                    prof.wspd_from_c[k], prof.wspd_from_mf[k])
        end

        # Summary stats
        @printf("\n  Max wind (from CX): %.1f m/s at k=%d (%.0f hPa)\n",
                maximum(prof.wspd_from_c), argmax(prof.wspd_from_c),
                prof.p_mid[argmax(prof.wspd_from_c)]/100)
        @printf("  Surface wind: %.1f m/s (k=%d, %.0f hPa)\n",
                prof.wspd_from_c[prof.Nz], prof.Nz, prof.p_mid[prof.Nz]/100)

        # Check A3dyn if available
        if isfile(a3dyn_file) && filesize(a3dyn_file) > 500_000_000  # > 500 MB = complete
            println("\n  --- A3dyn (actual U, V at ~12 UTC) ---")
            try
                # time_index=5 → ~12:00 UTC for 3-hourly data
                # flip=true → same top-to-bottom convention as CTM_A1 after flip
                a3 = read_a3dyn_profiles(a3dyn_file, i, j, p; time_index=5, flip=true)

                println("  k  | p_mid [hPa] | U_a3 [m/s] | V_a3 [m/s] | |V|_a3 | u_CX [m/s] | v_CY [m/s] | |V|_CX | ratio")
                println("  " * "-" ^ 100)
                for k in levels_to_show
                    (k < 1 || k > a3.Nz) && continue
                    r = a3.wspd[k] > 0.5 ? prof.wspd_from_c[k] / a3.wspd[k] : NaN
                    @printf("  %2d | %10.2f | %10.2f | %10.2f | %6.1f | %10.2f | %10.2f | %6.1f | %5.3f\n",
                            k, prof.p_mid[k]/100, a3.u[k], a3.v[k], a3.wspd[k],
                            prof.u_from_cx[k], prof.v_from_cy[k], prof.wspd_from_c[k], r)
                end

                @printf("\n  Max A3dyn wind: %.1f m/s at k=%d (%.0f hPa)\n",
                        maximum(a3.wspd), argmax(a3.wspd), prof.p_mid[argmax(a3.wspd)]/100)
                @printf("  Max CX wind:   %.1f m/s at k=%d (%.0f hPa)\n",
                        maximum(prof.wspd_from_c), argmax(prof.wspd_from_c),
                        prof.p_mid[argmax(prof.wspd_from_c)]/100)
                @printf("  Surface A3dyn: %.1f m/s   Surface CX: %.1f m/s\n",
                        a3.wspd[end], prof.wspd_from_c[end])

                # Mean ratio at tropospheric levels (where wind > 1 m/s)
                ratios = Float64[]
                for k in 1:min(prof.Nz, a3.Nz)
                    if a3.wspd[k] > 1.0
                        push!(ratios, prof.wspd_from_c[k] / a3.wspd[k])
                    end
                end
                if !isempty(ratios)
                    @printf("  Mean ratio CX/A3dyn (|V|>1m/s): %.3f ± %.3f (n=%d levels)\n",
                            mean(ratios), std(ratios), length(ratios))
                end
            catch e
                println("  Failed to read A3dyn: $e")
                println("  ", sprint(showerror, e, catch_backtrace()))
            end
        else
            sz = isfile(a3dyn_file) ? filesize(a3dyn_file) : 0
            @printf("\n  A3dyn file not available (size: %.0f MB, need >500 MB)\n", sz/1e6)
        end
    end
end

main()
