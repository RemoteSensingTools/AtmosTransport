# ---------------------------------------------------------------------------
# Diagnostic: verify mass flux magnitudes and Courant numbers
#
# Uses CX/CY (accumulated Courant numbers from GEOS model) as ground truth
# to validate our MFXC → kg/s → CFL conversion chain.
#
# The GEOS model CX is:  CX(i,j,k) = MFXC(i,j,k) / (DELP(i,j,k) × Area(i,j))
# where Area is the cell area on the sphere.
#
# Our per-half-step CFL should be:  CFL = CX / (2 × n_sub)
#
# Usage:
#   julia --project=. scripts/diagnose_transport.jl
# ---------------------------------------------------------------------------

using NCDatasets, Statistics, Printf, LinearAlgebra

# ── Configuration ─────────────────────────────────────────────────────────
const GEOS_IT_FILE = expanduser("~/data/geosit_c180/20230601/GEOSIT.20230601.CTM_A1.C180.nc")
const GEOS_FP_FILE = let
    d = expanduser("~/data/geosfp_cs/20211210")
    isdir(d) ? joinpath(d, first(sort(filter(f -> endswith(f, ".nc4"), readdir(d))))) : ""
end

const GRAV = 9.80665
const DT_MET = 3600.0   # 1-hour accumulation interval
const R_EARTH = 6.371e6  # Earth radius [m]

# ── Gnomonic cubed-sphere area computation (matches cubed_sphere_grid.jl) ──

"""Compute 3D unit vector on cube face `p` from gnomonic coords (ξ, η)."""
function gnomonic_xyz(ξ, η, p::Int)
    d = sqrt(1 + ξ^2 + η^2)
    if p == 1;     return ( 1/d, ξ/d, η/d)
    elseif p == 2;  return (-ξ/d, 1/d, η/d)
    elseif p == 3;  return (-ξ/d, -η/d, 1/d)  # North pole
    elseif p == 4;  return (-1/d, -ξ/d, η/d)
    elseif p == 5;  return (ξ/d, -1/d, η/d)
    else            return (ξ/d, η/d, -1/d)    # South pole (p == 6)
    end
end

"""Spherical area of quadrilateral from 4 unit vectors (Girard's theorem)."""
function spherical_area_quad(v1, v2, v3, v4)
    # Compute interior angles of the spherical polygon
    function angle_at(va, vb, vc)
        # Angle at vb between great circle arcs vb→va and vb→vc
        ab = collect(va) - dot(collect(va), collect(vb)) * collect(vb)
        cb = collect(vc) - dot(collect(vc), collect(vb)) * collect(vb)
        ab_n = norm(ab)
        cb_n = norm(cb)
        (ab_n < 1e-15 || cb_n < 1e-15) && return 0.0
        cos_angle = clamp(dot(ab, cb) / (ab_n * cb_n), -1.0, 1.0)
        return acos(cos_angle)
    end
    a1 = angle_at(v4, v1, v2)
    a2 = angle_at(v1, v2, v3)
    a3 = angle_at(v2, v3, v4)
    a4 = angle_at(v3, v4, v1)
    return (a1 + a2 + a3 + a4) - 2π
end

"""Compute cell areas for a cubed-sphere panel of resolution Nc."""
function compute_gnomonic_areas(Nc::Int)
    α_faces = range(-π/4, π/4, length=Nc+1)
    areas = zeros(Float64, Nc, Nc)
    for j in 1:Nc, i in 1:Nc
        ξ1, ξ2 = tan(α_faces[i]), tan(α_faces[i+1])
        η1, η2 = tan(α_faces[j]), tan(α_faces[j+1])
        # Panel 1 (areas identical for all panels due to symmetry)
        v1 = gnomonic_xyz(ξ1, η1, 1)
        v2 = gnomonic_xyz(ξ2, η1, 1)
        v3 = gnomonic_xyz(ξ2, η2, 1)
        v4 = gnomonic_xyz(ξ1, η2, 1)
        areas[i, j] = R_EARTH^2 * spherical_area_quad(v1, v2, v3, v4)
    end
    return areas
end

# ── Read one timestep of raw data ─────────────────────────────────────────
function read_raw_timestep(filepath::String; time_index::Int=1)
    ds = NCDataset(filepath, "r")
    try
        ntime = size(ds["MFXC"], 5)
        ti = min(time_index, ntime)
        if ti != time_index
            @info "  Requested time_index=$time_index but file has $ntime steps; using $ti"
        end

        mfxc = Array{Float64}(ds["MFXC"][:, :, :, :, ti])
        mfyc = Array{Float64}(ds["MFYC"][:, :, :, :, ti])
        delp = Array{Float64}(ds["DELP"][:, :, :, :, ti])

        cx = haskey(ds, "CX") ? Array{Float64}(ds["CX"][:, :, :, :, ti]) : nothing
        cy = haskey(ds, "CY") ? Array{Float64}(ds["CY"][:, :, :, :, ti]) : nothing

        mfxc_units = haskey(ds["MFXC"].attrib, "units") ? ds["MFXC"].attrib["units"] : "unknown"
        mfxc_longname = haskey(ds["MFXC"].attrib, "long_name") ? ds["MFXC"].attrib["long_name"] : "unknown"

        Nc = size(mfxc, 1)
        Nz = size(mfxc, 4)

        # Auto-detect vertical flip
        mid = div(Nc, 2)
        if delp[mid, mid, 1, 1] > 10 * delp[mid, mid, 1, Nz]
            @info "  Flipping vertical levels (bottom-to-top → top-to-bottom)"
            reverse!(mfxc, dims=4)
            reverse!(mfyc, dims=4)
            reverse!(delp, dims=4)
            cx !== nothing && reverse!(cx, dims=4)
            cy !== nothing && reverse!(cy, dims=4)
        end

        return (; mfxc, mfyc, delp, cx, cy, Nc, Nz, mfxc_units, mfxc_longname)
    finally
        close(ds)
    end
end

# ── Main diagnostic ───────────────────────────────────────────────────────
function run_diagnostic()
    println("=" ^ 72)
    println("MASS FLUX & COURANT NUMBER DIAGNOSTIC")
    println("=" ^ 72)

    for (label, filepath, product_Nc, dt_sub_cfg, n_sub_cfg) in [
        ("GEOS-IT C180", GEOS_IT_FILE, 180, 300.0, 12),
        ("GEOS-FP C720", GEOS_FP_FILE, 720, 900.0, 4),
    ]
        if !isfile(filepath)
            println("\n  Skipping $label — file not found: $filepath")
            continue
        end

        println("\n" * "─" ^ 40)
        println("  $label")
        println("─" ^ 40)
        println("  File: $(basename(filepath))")

        ts = read_raw_timestep(filepath; time_index=12)  # noon (or last available)
        @printf("  Grid: C%d  (Nc=%d, Nz=%d)\n", ts.Nc, ts.Nc, ts.Nz)
        @printf("  MFXC units: '%s'\n", ts.mfxc_units)
        @printf("  MFXC long_name: '%s'\n", ts.mfxc_longname)

        # Compute proper gnomonic areas
        areas = compute_gnomonic_areas(ts.Nc)
        mean_area = mean(areas)
        min_area = minimum(areas)
        max_area = maximum(areas)
        dx_avg = sqrt(mean_area)
        @printf("  Cell area: mean=%.3e m² (dx≈%.1f km), min=%.3e, max=%.3e\n",
                mean_area, dx_avg/1e3, min_area, max_area)
        @printf("  Area ratio max/min: %.3f (gnomonic stretching)\n", max_area/min_area)

        # ── Raw MFXC statistics ─────────────────────────────────────────
        println("\n  Raw field statistics:")
        @printf("    mean |MFXC| = %.3e Pa·m²\n", mean(abs.(ts.mfxc)))
        @printf("    max  |MFXC| = %.3e Pa·m²\n", maximum(abs.(ts.mfxc)))
        @printf("    mean |MFYC| = %.3e Pa·m²\n", mean(abs.(ts.mfyc)))

        # ── Courant number from file (CX) ──────────────────────────────
        if ts.cx !== nothing
            println("\n  File CX (accumulated Courant number over 1 hour):")
            @printf("    mean |CX| = %.4f\n", mean(abs.(ts.cx)))
            @printf("    max  |CX| = %.4f\n", maximum(abs.(ts.cx)))
            @printf("    mean |CY| = %.4f\n", mean(abs.(ts.cy)))
            @printf("    max  |CY| = %.4f\n", maximum(abs.(ts.cy)))
            u_from_cx = maximum(abs.(ts.cx)) * dx_avg / DT_MET
            @printf("    → max wind from CX: %.1f m/s (approx, using mean dx)\n", u_from_cx)
        end

        # ── Cell-by-cell: derive CX from MFXC/(DELP×Area) ─────────────
        # CX should equal MFXC / (DELP × Area) if MFXC is accumulated Pa·m²
        if ts.cx !== nothing
            println("\n  Cell-by-cell CX verification: CX_derived = MFXC / (DELP × Area)")
            # Compute CFL = MFXC / (DELP × area) for each cell
            cx_derived = similar(ts.mfxc)
            for p in 1:6, k in 1:5:ts.Nz, j in 1:5:ts.Nc, i in 1:5:ts.Nc
                cx_derived[i, j, p, k] = ts.mfxc[i, j, p, k] / (ts.delp[i, j, p, k] * areas[i, j])
            end

            # Compare where CX is significant
            mask = abs.(ts.cx) .> 0.01
            n_sig = count(mask)
            if n_sig > 0
                ratios = cx_derived[mask] ./ ts.cx[mask]
                @printf("    Cells with |CX| > 0.01: %d (%.1f%%)\n", n_sig, 100*n_sig/length(ts.cx))
                @printf("    CX_derived / CX_file:\n")
                @printf("      median = %.4f\n", median(ratios))
                @printf("      mean   = %.4f\n", mean(ratios))
                @printf("      p10    = %.4f\n", quantile(ratios, 0.10))
                @printf("      p90    = %.4f\n", quantile(ratios, 0.90))

                med = median(ratios)
                if 0.95 < med < 1.05
                    println("    ✓ CX = MFXC / (DELP × Area) confirmed — conversion is correct!")
                    println("    ✓ MFXC units are ACCUMULATED Pa·m² (not a rate)")
                else
                    @printf("    ⚠ Median ratio = %.4f — off by factor %.4f\n", med, 1.0/med)
                end
            end
        end

        # ── Our CFL conversion chain ──────────────────────────────────
        half_dt = dt_sub_cfg / 2
        println("\n  Our CFL conversion (dt_sub=$(Int(dt_sub_cfg))s, n_sub=$n_sub_cfg, half_dt=$(Int(half_dt))s):")

        # CFL_ours = MFXC × half_dt / (g × dt_met × m) = MFXC × half_dt / (dt_met × DELP × area)
        # CX = MFXC / (DELP × area)   (verified above)
        # Therefore: CFL_ours = CX × half_dt / dt_met = CX / (2 × n_sub)
        # This is EXACT (no approximation), so our CFL per half-step MUST equal CX / (2 × n_sub)

        cfl_ours = similar(ts.mfxc)
        for p in 1:6, k in 1:5:ts.Nz, j in 1:5:ts.Nc, i in 1:5:ts.Nc
            am_kgs = ts.mfxc[i, j, p, k] / (GRAV * DT_MET)
            am_half = am_kgs * half_dt
            m_air = ts.delp[i, j, p, k] * areas[i, j] / GRAV
            cfl_ours[i, j, p, k] = am_half / max(m_air, 1e-30)
        end

        @printf("    max |CFL_ours| per half-step = %.4f\n", maximum(abs.(cfl_ours)))

        if ts.cx !== nothing
            cfl_expected = ts.cx ./ (2 * n_sub_cfg)
            @printf("    max |CFL_expected| per half-step = %.4f (= max|CX|/%d)\n",
                    maximum(abs.(cfl_expected)), 2*n_sub_cfg)

            # Cell-by-cell comparison
            mask = abs.(ts.cx) .> 0.01
            if any(mask)
                ratios = cfl_ours[mask] ./ cfl_expected[mask]
                @printf("    CFL_ours / CFL_expected (cell-by-cell, |CX|>0.01):\n")
                @printf("      median = %.6f\n", median(ratios))
                @printf("      mean   = %.6f\n", mean(ratios))
                @printf("      std    = %.6f\n", std(ratios))

                med = median(ratios)
                if 0.999 < med < 1.001
                    println("    ✓ PERFECT MATCH — our CFL = CX/(2×n_sub) exactly")
                    println("    ✓ Mass flux conversion chain is verified correct!")
                elseif 0.95 < med < 1.05
                    println("    ✓ Close match (within 5%) — conversion is essentially correct")
                else
                    @printf("    ⚠ SYSTEMATIC ERROR: CFL off by factor %.4f\n", med)
                    @printf("    ⚠ Transport speed is %.1f%% of what it should be!\n", med*100)
                end
            end
        end

        # Implied wind
        max_cfl = maximum(abs.(cfl_ours))
        u_implied = max_cfl * dx_avg / half_dt
        @printf("    → max implied wind: %.1f m/s\n", u_implied)

        # ── Mass conservation (within single panel, ignoring boundaries) ──
        println("\n  Mass conservation check:")
        total_mfxc = sum(abs.(ts.mfxc))
        @printf("    sum |MFXC| = %.3e Pa·m²\n", total_mfxc)

        # ── Level-by-level CFL check ──────────────────────────────────
        println("\n  Level-by-level (select):")
        for k in [ts.Nz, ts.Nz-5, div(ts.Nz,2), div(ts.Nz,4), 1]
            k < 1 && continue
            max_cfl_k = maximum(abs.(cfl_ours[:, :, :, k]))
            mean_delp_k = mean(ts.delp[:, :, :, k])
            if ts.cx !== nothing
                max_cx_k = maximum(abs.(ts.cx[:, :, :, k]))
                cfl_exp_k = max_cx_k / (2 * n_sub_cfg)
                ratio_k = max_cfl_k > 0 ? max_cfl_k / max(cfl_exp_k, 1e-30) : 0.0
                @printf("    k=%2d: CFL=%.4f  CX/24=%.4f  ratio=%.3f  DELP=%.1f Pa\n",
                        k, max_cfl_k, cfl_exp_k, ratio_k, mean_delp_k)
            else
                @printf("    k=%2d: CFL=%.4f  DELP=%.1f Pa\n", k, max_cfl_k, mean_delp_k)
            end
        end

        # ── Key diagnostic: where is max CX and what CFL do we get there? ──
        if ts.cx !== nothing
            println("\n  Point check at max |CX| location:")
            idx_max = argmax(abs.(ts.cx))
            i, j, p, k = Tuple(idx_max)
            cx_here = ts.cx[i, j, p, k]
            mfxc_here = ts.mfxc[i, j, p, k]
            delp_here = ts.delp[i, j, p, k]
            area_here = areas[i, j]
            cfl_here = cfl_ours[i, j, p, k]
            cx_derived_here = mfxc_here / (delp_here * area_here)

            @printf("    Location: panel=%d, i=%d, j=%d, k=%d\n", p, i, j, k)
            @printf("    CX_file  = %.6f\n", cx_here)
            @printf("    CX_derived (MFXC/(DELP×Area)) = %.6f\n", cx_derived_here)
            @printf("    Ratio (derived/file) = %.6f\n", cx_derived_here / cx_here)
            @printf("    MFXC = %.4e Pa·m²\n", mfxc_here)
            @printf("    DELP = %.2f Pa\n", delp_here)
            @printf("    Area = %.4e m²\n", area_here)
            @printf("    CFL_ours = %.6f (should be %.6f = CX/24)\n",
                    cfl_here, cx_here / (2 * n_sub_cfg))
        end
    end

    println("\n" * "=" ^ 72)
    println("DONE")
end

run_diagnostic()
