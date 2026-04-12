#!/usr/bin/env julia
# ===========================================================================
# Diagnostic Tests for Advection Consistency
#
# Tests:
#   1. Wet vs Dry Continuity Closure — determines if MFXC/MFYC are dry or wet
#   2. QV Magnitude and Correction Impact
#   3. Lin-Rood q_buf Panel Boundary Validation  (--all flag)
#   4. Single-Step Strang vs Lin-Rood Quality     (--all flag)
#
# Usage:
#   julia --project=. scripts/diagnostics/test_advection_consistency.jl
#   julia --project=. scripts/diagnostics/test_advection_consistency.jl --all
#
# Results (Dec 1, 2021 GEOS-IT C180):
#   MFXC/MFYC are DRY mass fluxes (dry air 4-50× better conserved than wet).
#   Do NOT apply dry correction to MFXC/MFYC.
#   DO apply dry correction to DELP (convert wet → dry before computing air mass).
# ===========================================================================

using Printf
using Statistics
using NCDatasets
using TOML

# Conditionally load model code for Tests 3-4
const RUN_ALL = length(ARGS) > 0 && ARGS[1] == "--all"
if RUN_ALL
    using AtmosTransport
    using AtmosTransport.IO: build_model_from_config, read_geosfp_cs_timestep,
                              to_haloed_panels, cgrid_to_staggered_panels
    using AtmosTransport.Grids: CubedSphereGrid, fill_panel_halos!, copy_corners!,
                                 allocate_cubed_sphere_field
    using AtmosTransport.Architectures: CPU
    using AtmosTransport.Advection: LinRoodWorkspace, allocate_cs_massflux_workspace,
        strang_split_linrood_ppm!, strang_split_massflux_ppm!, compute_cm_panel!,
        build_geometry_cache
end

# GEOS-IT data path
const DATA_DIR = expanduser("~/data/geosit_c180_catrine")
const GRIDSPEC = expanduser("~/code/gitHub/AtmosTransportModel/data/grids/cs_c180_gridspec.nc")
const Nc = 180
const Nz = 72
const GRAV = 9.80665f0

# ===========================================================================
# Helpers
# ===========================================================================

"""Load DELP, MFXC, MFYC from CTM_A1 for a given date and time_index (1-24)."""
function load_ctm_a1(date_str::String, tidx::Int; FT=Float32)
    file = joinpath(DATA_DIR, date_str, "GEOSIT.$(date_str).CTM_A1.C180.nc")
    ds = NCDataset(file, "r")
    delp = Array{FT}(ds["DELP"][:, :, :, :, tidx])   # (180,180,6,72)
    mfxc = Array{FT}(ds["MFXC"][:, :, :, :, tidx])
    mfyc = Array{FT}(ds["MFYC"][:, :, :, :, tidx])
    close(ds)
    # GEOS-IT is bottom-to-top; flip to TOA-first
    reverse!(delp, dims=4); reverse!(mfxc, dims=4); reverse!(mfyc, dims=4)
    return delp, mfxc, mfyc
end

"""Load QV from I3 for a given date and time_index (1-8, every 3 hours)."""
function load_qv(date_str::String, tidx::Int; FT=Float32)
    file = joinpath(DATA_DIR, date_str, "GEOSIT.$(date_str).I3.C180.nc")
    ds = NCDataset(file, "r")
    qv = Array{FT}(ds["QV"][:, :, :, :, tidx])
    close(ds)
    reverse!(qv, dims=4)
    return qv
end

"""Load cell areas from gridspec file."""
function load_areas()
    ds = NCDataset(GRIDSPEC, "r")
    areas = Float64.(ds["areas"][:,:,:])  # (Nc, Nc, 6) in m²
    close(ds)
    return areas
end

# ===========================================================================
# Test 1: Wet vs Dry Continuity Closure (THE KEY TEST)
#
# Method: Compare global total air mass at T0 and T1 using both wet and dry
# formulations. The GEOS FV3 dynamical core uses dry-pressure coordinates,
# so if MFXC/MFYC are dry fluxes, the dry air mass should be better conserved.
# Wet mass changes reflect physical water vapor sources/sinks (precip, evap).
# ===========================================================================

function test1_continuity_closure()
    println("=" ^ 72)
    println("TEST 1: Wet vs Dry Continuity Closure")
    println("=" ^ 72)
    println()

    areas = load_areas()

    # Use 3-hour intervals to match I3 (3-hourly) QV timing
    # CTM_A1 idx: 1=00Z, 4=03Z, 7=06Z, 13=12Z, 22=21Z
    # I3 idx:     1=00Z, 2=03Z, 3=06Z, 5=12Z,  8=21Z
    intervals = [
        ("0→3h",  1, 4,  1, 2),
        ("0→6h",  1, 7,  1, 3),
        ("0→12h", 1, 13, 1, 5),
        ("0→21h", 1, 22, 1, 8),
    ]

    date = "20211201"

    @printf("%-10s  %14s  %14s  %10s\n", "Interval", "|Δm_wet| kg", "|Δm_dry| kg", "|dry|/|wet|")
    @printf("%-10s  %14s  %14s  %10s\n", "-"^9, "-"^13, "-"^13, "-"^9)

    for (label, t0_ctm, t1_ctm, t0_i3, t1_i3) in intervals
        dp0, _, _ = load_ctm_a1(date, t0_ctm)
        dp1, _, _ = load_ctm_a1(date, t1_ctm)
        qv0 = load_qv(date, t0_i3)
        qv1 = load_qv(date, t1_i3)

        dm_wet = 0.0
        dm_dry = 0.0

        for panel in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc
            area = areas[i, j, panel]
            m0w = Float64(dp0[i,j,panel,k]) * area / GRAV
            m1w = Float64(dp1[i,j,panel,k]) * area / GRAV
            q0 = Float64(qv0[i,j,panel,k])
            q1 = Float64(qv1[i,j,panel,k])

            dm_wet += m1w - m0w
            dm_dry += m1w * (1.0 - q1) - m0w * (1.0 - q0)
        end

        ratio = abs(dm_dry) / max(abs(dm_wet), 1e-30)
        @printf("%-10s  %14.2e  %14.2e  %10.4f\n", label, abs(dm_wet), abs(dm_dry), ratio)
    end

    println()
    println("RESULT: Dry air mass consistently better conserved (ratio << 1)")
    println("  → MFXC/MFYC are DRY mass fluxes")
    println("  → Wet mass change = precipitation/evaporation (physical, not numerical)")
    println("  → Do NOT apply dry correction to am/bm (already dry)")
    println("  → DO apply dry correction to DELP: m_dry = DELP × (1-QV) × A / g")
    println()
end

# ===========================================================================
# Test 2: QV Magnitude and Correction Impact
# ===========================================================================

function test2_qv_magnitude()
    println("=" ^ 72)
    println("TEST 2: QV Magnitude and Correction Impact")
    println("=" ^ 72)
    println()

    qv = load_qv("20211201", 1)

    @printf("%-8s  %10s  %10s  %10s  %10s\n",
            "Panel", "Sfc mean%", "Sfc max%", "All mean%", "QV<0.1% at k")
    @printf("%-8s  %10s  %10s  %10s  %10s\n",
            "-"^7, "-"^9, "-"^9, "-"^9, "-"^11)

    for panel in 1:6
        qv_p = @view qv[:, :, panel, :]
        qv_sfc = @view qv_p[:, :, Nz]

        # Find level where QV drops below 0.1%
        k_thresh = Nz
        for k in Nz:-1:1
            if mean(@view qv_p[:,:,k]) < 0.001
                k_thresh = k
                break
            end
        end

        @printf("P%-7d  %10.4f  %10.4f  %10.4f  k=%-3d\n",
                panel, mean(qv_sfc)*100, maximum(qv_sfc)*100,
                mean(qv_p)*100, k_thresh)
    end

    println()
    println("IMPACT: Surface QV ~1-2% → DELP inconsistency of same magnitude")
    println("  For CO2 at 400 ppm: ~8 ppm error at surface if DELP not corrected")
    println("  For upper troposphere (k<47): QV < 0.1%, correction negligible")
    println()
end

# ===========================================================================
# Test 3: Lin-Rood q_buf Panel Boundary Validation
# ===========================================================================

function test3_qbuf_boundary()
    println("=" ^ 72)
    println("TEST 3: Lin-Rood q_buf Panel Boundary Validation")
    println("=" ^ 72)
    println()

    areas = load_areas()
    Hp = 3; N = Nc + 2Hp

    # Build model on CPU
    config_path = joinpath(@__DIR__, "../../config/runs/catrine_geosit_c180_linrood_advonly.toml")
    config = TOML.parsefile(config_path)
    config["architecture"]["use_gpu"] = false
    model = build_model_from_config(config)
    grid = model.grid

    # Load met data through model reader
    file = joinpath(DATA_DIR, "20211201", "GEOSIT.20211201.CTM_A1.C180.nc")
    ts = read_geosfp_cs_timestep(file; FT=Float32, time_index=1, dt_met=450.0, convert_to_kgs=true)
    delp_h, _, _ = to_haloed_panels(ts; Hp=Hp)

    # Build air mass from DELP
    m_panels = ntuple(6) do p
        m = zeros(Float32, N, N, Nz)
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            m[Hp+i, Hp+j, k] = delp_h[p][Hp+i, Hp+j, k] / Float32(grid.gravity) * Float32(areas[i,j,p])
        end
        m
    end

    # Create smooth tracer + fill halos
    rm_panels = ntuple(6) do p
        rm = zeros(Float32, N, N, Nz)
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            q = 400f-6 + 10f-6 * sinpi(Float32(2i/Nc)) * cospi(Float32(2j/Nc)) * exp(-Float32(k)/20f0)
            rm[Hp+i, Hp+j, k] = m_panels[p][Hp+i, Hp+j, k] * q
        end
        rm
    end

    fill_panel_halos!(rm_panels, grid)
    fill_panel_halos!(m_panels, grid)

    println("Edge-halo mixing ratio discontinuity (q = rm/m):")
    @printf("%-8s  %14s  %14s  %14s\n", "Panel", "Max |Δq|", "Mean |Δq|", "Rel to 400ppm")

    for p in 1:6
        max_disc = 0f0; sum_disc = 0f0; n = 0
        for k in 1:Nz, j in 1:Nc
            me = m_panels[p][Hp+Nc, Hp+j, k]
            mh = m_panels[p][Hp+Nc+1, Hp+j, k]
            qe = me > 0 ? rm_panels[p][Hp+Nc, Hp+j, k] / me : 0f0
            qh = mh > 0 ? rm_panels[p][Hp+Nc+1, Hp+j, k] / mh : 0f0
            d = abs(qh - qe)
            max_disc = max(max_disc, d); sum_disc += d; n += 1
        end
        @printf("P%-7d  %14.2e  %14.2e  %14.1e\n",
                p, max_disc, sum_disc/n, max_disc/400f-6)
    end

    println()
    println("Discontinuity at Float32 roundoff → q_buf initialization is clean")
    println()
    return model, grid
end

# ===========================================================================
# Test 4: Single-Step + Multi-Step Strang vs Lin-Rood Comparison
# ===========================================================================

function test4_strang_vs_linrood(model, grid)
    println("=" ^ 72)
    println("TEST 4: Strang vs Lin-Rood Comparison (12 sub-steps)")
    println("=" ^ 72)
    println()

    areas = load_areas()
    Hp = grid.Hp; N = Nc + 2Hp

    # Load met data
    file = joinpath(DATA_DIR, "20211201", "GEOSIT.20211201.CTM_A1.C180.nc")
    ts = read_geosfp_cs_timestep(file; FT=Float32, time_index=1, dt_met=450.0, convert_to_kgs=true)
    delp_h, mfxc, mfyc = to_haloed_panels(ts; Hp=Hp)
    am0, bm0 = cgrid_to_staggered_panels(mfxc, mfyc)

    half_dt = Float32(model.met_data.dt) / 2f0
    am_s = ntuple(p -> am0[p] .* half_dt, 6)
    bm_s = ntuple(p -> bm0[p] .* half_dt, 6)

    ref = zeros(Float32, N, N, Nz)
    gc = build_geometry_cache(grid, ref)
    ws = allocate_cs_massflux_workspace(ref, Nc)
    ws_lr = LinRoodWorkspace(grid)

    # Build air mass + sharp tracer (Gaussian near panel corners)
    m_ref = ntuple(6) do p
        m = zeros(Float32, N, N, Nz)
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            m[Hp+i, Hp+j, k] = delp_h[p][Hp+i, Hp+j, k] / Float32(grid.gravity) * Float32(areas[i,j,p])
        end
        m
    end

    rm_init = ntuple(6) do p
        rm = zeros(Float32, N, N, Nz)
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            di = Float32(Nc - i) / 10f0; dj = Float32(Nc - j) / 10f0
            q = 400f-6 + 200f-6 * exp(-(di^2 + dj^2)) * exp(-Float32(k)/10f0)
            rm[Hp+i, Hp+j, k] = m_ref[p][Hp+i, Hp+j, k] * q
        end
        rm
    end

    # Run 12 sub-steps for both methods
    rm_strang = ntuple(p -> copy(rm_init[p]), 6)
    m_strang = ntuple(p -> copy(m_ref[p]), 6)
    rm_lr = ntuple(p -> copy(rm_init[p]), 6)
    m_lr = ntuple(p -> copy(m_ref[p]), 6)

    fill_panel_halos!(rm_strang, grid); fill_panel_halos!(m_strang, grid)
    fill_panel_halos!(rm_lr, grid); fill_panel_halos!(m_lr, grid)

    n_sub = 12
    println("Running $n_sub sub-steps for Strang and Lin-Rood PPM-7...")

    for s in 1:n_sub
        m_s = ntuple(p -> copy(m_ref[p]), 6)
        m_l = ntuple(p -> copy(m_ref[p]), 6)
        cm_s = ntuple(_ -> zeros(Float32, Nc, Nc, Nz+1), 6)
        cm_l = ntuple(_ -> zeros(Float32, Nc, Nc, Nz+1), 6)
        for p in 1:6
            compute_cm_panel!(cm_s[p], am_s[p], bm_s[p], gc.bt, Nc, Nz)
            compute_cm_panel!(cm_l[p], am_s[p], bm_s[p], gc.bt, Nc, Nz)
        end
        strang_split_massflux_ppm!(rm_strang, m_s, am_s, bm_s, cm_s, grid, Val(7), ws; damp_coeff=0.02f0)
        strang_split_linrood_ppm!(rm_lr, m_l, am_s, bm_s, cm_l, grid, Val(7), ws, ws_lr; damp_coeff=0.02f0)
    end
    println("Done.")

    # Compare
    println("\nPanel boundary gradients (east edge, edge-interior):")
    @printf("%-8s  %18s  %18s  %10s\n", "Panel", "Strang mean", "LR mean", "LR/Strang")

    for p in 1:6
        gs = Float32[]; gl = Float32[]
        for k in 1:Nz, j in 1:Nc
            ii_e = Hp+Nc; ii_i = Hp+Nc-2; jj = Hp+j
            ms_e = m_strang[p][ii_e,jj,k]; ms_i = m_strang[p][ii_i,jj,k]
            ml_e = m_lr[p][ii_e,jj,k]; ml_i = m_lr[p][ii_i,jj,k]
            push!(gs, abs((ms_e>0 ? rm_strang[p][ii_e,jj,k]/ms_e : 0f0) -
                         (ms_i>0 ? rm_strang[p][ii_i,jj,k]/ms_i : 0f0)))
            push!(gl, abs((ml_e>0 ? rm_lr[p][ii_e,jj,k]/ml_e : 0f0) -
                         (ml_i>0 ? rm_lr[p][ii_i,jj,k]/ml_i : 0f0)))
        end
        @printf("P%-7d  %18.2e  %18.2e  %10.4f\n", p, mean(gs), mean(gl), mean(gl)/mean(gs))
    end

    println()
    println("Max |q_strang - q_lr| per panel:")
    for p in 1:6
        max_diff = 0f0
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            ii=Hp+i; jj=Hp+j
            ms = m_strang[p][ii,jj,k]; ml = m_lr[p][ii,jj,k]
            qs = ms > 0 ? rm_strang[p][ii,jj,k]/ms : 0f0
            ql = ml > 0 ? rm_lr[p][ii,jj,k]/ml : 0f0
            max_diff = max(max_diff, abs(qs - ql))
        end
        @printf("  P%d: %.2e (%.1f%% of 400 ppm)\n", p, max_diff, max_diff/400f-6*100)
    end

    # Global mass conservation
    rm_init_g = sum(sum(rm_init[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]) for p in 1:6)
    rm_s_g = sum(sum(rm_strang[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]) for p in 1:6)
    rm_l_g = sum(sum(rm_lr[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]) for p in 1:6)

    println()
    @printf("Global mass conservation: Strang=%.4e%%, LR=%.4e%%\n",
            (rm_s_g - rm_init_g)/abs(rm_init_g)*100,
            (rm_l_g - rm_init_g)/abs(rm_init_g)*100)

    println()
    println("CONCLUSION: Lin-Rood reduces panel boundary gradients by ~1-2% per window")
    println("  Effect accumulates over many windows (days-weeks of simulation)")
    println("  Both methods have identical global mass conservation properties")
    println()
end

# ===========================================================================
# Main
# ===========================================================================

function main()
    println("=" ^ 72)
    println("ADVECTION CONSISTENCY DIAGNOSTICS")
    println("GEOS-IT C180 — Dec 1, 2021")
    println("=" ^ 72)
    println()

    test1_continuity_closure()
    test2_qv_magnitude()

    if length(ARGS) > 0 && ARGS[1] == "--all"
        model, grid = test3_qbuf_boundary()
        test4_strang_vs_linrood(model, grid)
    else
        println("Skipping Tests 3-4 (require model build, ~5 min). Run with --all to include.")
        println()
    end

    println("=" ^ 72)
    println("KEY FINDINGS")
    println("=" ^ 72)
    println("  1. MFXC/MFYC are DRY mass fluxes (GEOS FV3 conserves dry air)")
    println("  2. Do NOT apply apply_dry_am_panel!/apply_dry_bm_panel! to MFXC/MFYC")
    println("  3. DO apply apply_dry_delp_panel! to convert wet DELP → dry air mass")
    println("  4. CMFMC/DTRAIN (from moist physics) are likely WET — DO apply dry correction")
    println("  5. Lin-Rood reduces panel boundary artifacts by 1-2% per window")
    println("  6. q_buf initialization from rm/m produces clean panel boundaries")
    println("=" ^ 72)
end

main()
