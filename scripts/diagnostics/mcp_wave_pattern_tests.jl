# ===========================================================================
# MCP Wave Pattern Diagnostic Tests
#
# Production-faithful transport tests to track down midlatitude wave artifacts.
# Exactly replicates run_loop.jl + physics_phases.jl:advection_phase! pipeline,
# including double buffer IO, real QV, GCHP hybrid PE, and two fv_tp_2d calls
# per substep.
#
# Vertical convention: k=1=TOA, k=Nz(72)=surface (DELP thinnest at k=1)
#
# Run blocks sequentially in an MCP Julia session.
# Prerequisites: debug_remap_2win.toml config, GEOS-IT C180 met data
# ===========================================================================

# =====================================================================
# BLOCK 1: Setup — load model, extract internals, verify QV + vertical
# =====================================================================
#=
using CUDA; CUDA.allowscalar(true)
using AtmosTransport
using AtmosTransport.IO: build_model_from_config
using AtmosTransport.Models: begin_load!, wait_load!, swap!,
    upload_met!, load_and_upload_physics!, compute_ps_phase!,
    process_met_after_upload!, compute_air_mass_phase!,
    save_reference_mass!, compute_cm_phase!, wait_and_upload_next_delp!,
    IOScheduler, current_gpu, next_gpu, current_cpu
using AtmosTransport.Advection: fv_tp_2d_cs!, vertical_remap_cs!,
    strang_split_linrood_ppm!, strang_split_massflux_ppm!,
    compute_target_pressure_from_dry_delp_direct!,
    compute_target_pressure_from_delp_direct!,
    compute_target_pressure_from_mass_direct!,
    compute_source_pe_from_hybrid!,
    compute_target_pe_from_ps_hybrid!,
    update_air_mass_from_target!,
    fix_target_bottom_pe!
using AtmosTransport.Grids: fill_panel_halos!
import TOML, Statistics, Printf
using Statistics: mean, std

config = TOML.parsefile("config/runs/debug_remap_2win.toml")
# Override to 2 days = 48 windows for wave growth detection
config["met_data"]["end_date"] = "2021-12-03"
model = build_model_from_config(config)

grid = model.grid
Hp = grid.Hp; Nc = grid.Nc; Nz = grid.Nz
FT = Float32
driver = model.met_data

using AtmosTransport.Models: build_io_scheduler, allocate_physics_buffers,
    allocate_tracers, allocate_air_mass, allocate_geometry_and_workspace,
    _needs_linrood, _needs_vertical_remap, physics_load_kwargs,
    initial_load!, total_windows, steps_per_window

sched = build_io_scheduler(grid, model.architecture, model.buffering)
phys  = allocate_physics_buffers(grid, model.architecture, model)
tracers = allocate_tracers(model, grid)
air   = allocate_air_mass(grid, model.architecture)
_use_lr = _needs_linrood(model.advection_scheme)
_use_vr = _needs_vertical_remap(model.advection_scheme)
gc_ws = allocate_geometry_and_workspace(grid, model.architecture;
            use_linrood=_use_lr, use_vertical_remap=_use_vr)
gc    = gc_ws.gc
ws    = gc_ws.ws
ws_lr = gc_ws.ws_lr
ws_vr = gc_ws.ws_vr

n_win_total = total_windows(driver)
n_sub = steps_per_window(driver)
dt_sub = FT(driver.dt)
half_dt = FT(dt_sub / 2)
damp = model.advection_scheme.damp_coeff

# Load first window
kw = physics_load_kwargs(phys, grid)
initial_load!(sched, driver, grid, 1; kw...)
upload_met!(sched)
load_and_upload_physics!(phys, sched, driver, grid, 1; arch=model.architecture)
compute_ps_phase!(phys, sched, grid)
process_met_after_upload!(sched, phys, grid, driver, half_dt)
compute_air_mass_phase!(sched, air, phys, grid, gc;
    dry_correction=get(model.metadata, "dry_correction", true))
save_reference_mass!(sched, air, grid)

# ── Verify setup ──
println("Setup: C$(Nc), $(n_sub) sub/win, $(n_win_total) windows")
println("  linrood=$(_use_lr), vremap=$(_use_vr)")
println("  QV loaded: $(phys.qv_loaded[])")

# Verify vertical ordering: k=1 should be TOA (thin DELP), k=Nz=surface (thick DELP)
gpu = current_gpu(sched)
delp_p1 = Array(gpu.delp[1])
delp_k1 = mean(delp_p1[Hp+1:Hp+Nc, Hp+1:Hp+Nc, 1])
delp_kN = mean(delp_p1[Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz])
Printf.@printf("  Vertical check: DELP[k=1]=%.2f Pa (should be thin/TOA), DELP[k=%d]=%.2f Pa (should be thick/sfc)\n",
    delp_k1, Nz, delp_kN)
@assert delp_k1 < delp_kN "VERTICAL ORDERING WRONG: k=1 should be TOA (thin DELP)"

# QV stats at surface
if phys.qv_loaded[]
    qv_p1 = Array(phys.qv_gpu[1])
    qv_sfc = qv_p1[Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz]
    Printf.@printf("  QV surface: mean=%.4f%%, max=%.4f%%\n",
        mean(qv_sfc)*100, maximum(qv_sfc)*100)
end

# Grid coordinates for IC patterns
φᶜ = grid.φᶜ  # NTuple{6, Matrix{FT}}, Nc×Nc per panel, degrees

println("Setup complete.")
=#

# =====================================================================
# BLOCK 2: Diagnostic functions
# =====================================================================
#=

"""Per-panel statistics for a given level k. Returns (panel, min, max, mean, std)."""
function panel_stats(rm_panels, m_panels, k::Int)
    results = []
    for p in 1:6
        rm = Array(rm_panels[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
        m  = Array(m_panels[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
        q = Float64.(rm) ./ Float64.(m) .* 1e6  # ppm
        push!(results, (panel=p, min=minimum(q), max=maximum(q),
                        mean=mean(q), std=std(q)))
    end
    return results
end

"""Edge vs interior statistics for a given level k."""
function edge_vs_interior_stats(rm_panels, m_panels, k::Int; edge_width::Int=10)
    edge_q = Float64[]
    interior_q = Float64[]
    for p in 1:6
        rm = Array(rm_panels[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
        m  = Array(m_panels[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
        q = Float64.(rm) ./ Float64.(m) .* 1e6
        ew = edge_width
        for j in 1:Nc, i in 1:Nc
            is_edge = (i <= ew || i > Nc-ew || j <= ew || j > Nc-ew)
            if is_edge
                push!(edge_q, q[i, j])
            else
                push!(interior_q, q[i, j])
            end
        end
    end
    return (edge=(min=minimum(edge_q), max=maximum(edge_q),
                  mean=mean(edge_q), std=std(edge_q), n=length(edge_q)),
            interior=(min=minimum(interior_q), max=maximum(interior_q),
                      mean=mean(interior_q), std=std(interior_q), n=length(interior_q)))
end

"""Per-edge-pair statistics: for each of the 24 panel edges, compute std."""
function per_edge_stats(rm_panels, m_panels, k::Int; strip_width::Int=5)
    edges = []
    for p in 1:6
        rm = Array(rm_panels[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
        m  = Array(m_panels[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
        q = Float64.(rm) ./ Float64.(m) .* 1e6
        sw = strip_width
        q_w = vec(q[1:sw, :])
        push!(edges, (panel=p, side="W", mean=mean(q_w), std=std(q_w)))
        q_e = vec(q[Nc-sw+1:Nc, :])
        push!(edges, (panel=p, side="E", mean=mean(q_e), std=std(q_e)))
        q_s = vec(q[:, 1:sw])
        push!(edges, (panel=p, side="S", mean=mean(q_s), std=std(q_s)))
        q_n = vec(q[:, Nc-sw+1:Nc])
        push!(edges, (panel=p, side="N", mean=mean(q_n), std=std(q_n)))
    end
    return edges
end

"""Global level stats (compact)."""
function level_stats_compact(rm_panels, m_panels;
                              levels=[1, 10, 20, 36, 50, 60, 65, 70, 72])
    for k in levels
        q_all = Float64[]
        for p in 1:6
            rm = Array(rm_panels[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
            m  = Array(m_panels[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
            append!(q_all, vec(Float64.(rm) ./ Float64.(m) .* 1e6))
        end
        Printf.@printf("  k=%2d: mean=%9.3f std=%9.5f range=[%9.3f, %9.3f]\n",
            k, mean(q_all), std(q_all), minimum(q_all), maximum(q_all))
    end
end

"""Zonal mean statistics by latitude band. Focus on 30-60°S for wave detection."""
function zonal_mean_by_lat_band(rm_panels, m_panels, k::Int;
        bands=[(-90,-60), (-60,-30), (-30,0), (0,30), (30,60), (60,90)])
    for (lat_lo, lat_hi) in bands
        q_band = Float64[]
        for p in 1:6
            rm = Array(rm_panels[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
            m  = Array(m_panels[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
            lat = φᶜ[p]
            for j in 1:Nc, i in 1:Nc
                if lat_lo <= lat[i, j] < lat_hi
                    push!(q_band, Float64(rm[i, j]) / Float64(m[i, j]) * 1e6)
                end
            end
        end
        if isempty(q_band)
            Printf.@printf("  [%4d,%4d°]: no cells\n", lat_lo, lat_hi)
        else
            Printf.@printf("  [%4d,%4d°]: n=%6d mean=%9.4f std=%9.5f range=[%9.4f,%9.4f]\n",
                lat_lo, lat_hi, length(q_band), mean(q_band), std(q_band),
                minimum(q_band), maximum(q_band))
        end
    end
end

"""Verify vertical ordering: k=1 should be TOA (thin DELP)."""
function verify_vertical_ordering(delp_panels, Hp, Nc, Nz)
    for p in 1:6
        d = Array(delp_panels[p])
        dk1 = mean(d[Hp+1:Hp+Nc, Hp+1:Hp+Nc, 1])
        dkN = mean(d[Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz])
        ok = dk1 < dkN ? "✓" : "✗ WRONG"
        Printf.@printf("  P%d: DELP[k=1]=%.1f  DELP[k=%d]=%.1f  %s\n", p, dk1, Nz, dkN, ok)
    end
end

"""Compare PE from hybrid vs direct methods. Reports per-level max difference."""
function pe_method_comparison(ws_vr, delp_panels, qv_panels, gc, grid)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp

    # Compute hybrid PE → pe_src
    compute_source_pe_from_hybrid!(ws_vr, delp_panels, qv_panels, gc, grid)
    pe_hybrid = [Array(ws_vr.pe_src[p]) for p in 1:6]

    # Compute direct PE → pe_tgt (overwrites pe_tgt, but we save it)
    compute_target_pressure_from_dry_delp_direct!(ws_vr, delp_panels, qv_panels, gc, grid)
    pe_direct = [Array(ws_vr.pe_tgt[p]) for p in 1:6]

    println("  PE comparison (hybrid vs direct cumsum):")
    println("  k   max_abs_diff(Pa)   max_rel_diff(%)   level_type")
    for k in 1:Nz+1
        abs_diffs = Float64[]
        rel_diffs = Float64[]
        for p in 1:6
            for j in 1:Nc, i in 1:Nc
                ph = Float64(pe_hybrid[p][i, j, k])
                pd = Float64(pe_direct[p][i, j, k])
                push!(abs_diffs, abs(ph - pd))
                if abs(pd) > 1.0
                    push!(rel_diffs, abs(ph - pd) / abs(pd) * 100)
                end
            end
        end
        max_abs = maximum(abs_diffs)
        max_rel = isempty(rel_diffs) ? 0.0 : maximum(rel_diffs)
        if k <= 5 || k >= Nz-2 || k % 10 == 0 || max_rel > 0.01
            Printf.@printf("  %3d  %12.4f       %12.6f\n", k, max_abs, max_rel)
        end
    end
end

"""Print reports."""
function print_panel_report(stats)
    println("  Per-panel:")
    for s in stats
        Printf.@printf("    P%d: mean=%9.4f std=%9.5f range=[%9.4f, %9.4f]\n",
            s.panel, s.mean, s.std, s.min, s.max)
    end
    stds = [s.std for s in stats]
    Printf.@printf("  Panel std spread: min=%9.5f max=%9.5f ratio=%.2f\n",
        minimum(stds), maximum(stds), maximum(stds)/max(minimum(stds), 1e-12))
end

function print_edge_interior_report(ei)
    Printf.@printf("  Edge:     mean=%9.4f std=%9.5f (n=%d)\n",
        ei.edge.mean, ei.edge.std, ei.edge.n)
    Printf.@printf("  Interior: mean=%9.4f std=%9.5f (n=%d)\n",
        ei.interior.mean, ei.interior.std, ei.interior.n)
    ratio = ei.edge.std / max(ei.interior.std, 1e-12)
    Printf.@printf("  Edge/Interior std ratio: %.2f%s\n",
        ratio, ratio > 2.0 ? " ⚠️  PANEL BOUNDARY ARTIFACT" : " ✓")
end

function print_edge_report(edges; top_n=10)
    sorted = sort(edges, by=e->e.std, rev=true)
    println("  Worst edges (by std):")
    for (i, e) in enumerate(sorted[1:min(top_n, length(sorted))])
        Printf.@printf("    P%d-%s: mean=%9.4f std=%9.5f\n",
            e.panel, e.side, e.mean, e.std)
    end
end

println("Diagnostic functions loaded.")
=#

# =====================================================================
# BLOCK 3: Production-faithful window loop
#
# Exactly mirrors run_loop.jl:71-171 + physics_phases.jl:422-532
# including IO scheduling, double buffer, real QV, hybrid PE,
# two fv_tp_2d calls per substep, and correct m restore/rescale.
# =====================================================================
#=

"""
    run_production_windows!(tracers, air, phys, sched, model, grid, gc, ws, ws_lr, ws_vr;
                            n_windows=24, pe_method=:hybrid, diag_interval=4,
                            diagnostics_fn=nothing)

Run n_windows of transport exactly matching the production pipeline.

`pe_method`:
  - `:hybrid` — GCHP hybrid PE (production path, PE = ak + bk × PS_dry)
  - `:direct` — Direct cumsum of dry DELP (old MCP test path)

`diagnostics_fn(w, tracers, m_panels)` is called at intervals for progress.
"""
function run_production_windows!(tracers, air, phys, sched, model, grid, gc,
                                  ws, ws_lr, ws_vr;
                                  n_windows::Int=24, pe_method::Symbol=:hybrid,
                                  diag_interval::Int=4, diagnostics_fn=nothing)
    FT = Float32
    driver = model.met_data
    n_sub_loc = steps_per_window(driver)
    dt_sub_loc = FT(driver.dt)
    half_dt_loc = FT(dt_sub_loc / 2)
    damp_loc = model.advection_scheme.damp_coeff
    kw_loc = physics_load_kwargs(phys, grid)
    n_total = total_windows(driver)
    _dry_corr = get(model.metadata, "dry_correction", true)
    step_loc = Ref(0)

    # Track evolution
    history = []

    for w in 1:min(n_windows, n_total)
        has_next = w < n_total

        # ── IO Phase (run_loop.jl:75-86) ──────────────────────────────
        upload_met!(sched)
        load_and_upload_physics!(phys, sched, driver, grid, w; arch=model.architecture)

        if has_next
            begin_load!(sched, driver, grid, w + 1; kw_loc...)
        end

        compute_ps_phase!(phys, sched, grid)

        # ── GPU Compute Phase (run_loop.jl:91-99) ─────────────────────
        process_met_after_upload!(sched, phys, grid, driver, half_dt_loc)
        compute_air_mass_phase!(sched, air, phys, grid, gc; dry_correction=_dry_corr)
        save_reference_mass!(sched, air, grid)

        # ── Deferred DELP fetch (run_loop.jl:127) ─────────────────────
        wait_and_upload_next_delp!(sched, grid)

        # ── ADVECTION: remap path (physics_phases.jl:430-532) ─────────
        gpu = current_gpu(sched)

        # Save prescribed m (physics_phases.jl:447)
        for p in 1:6; copyto!(ws_vr.m_save[p], air.m[p]); end

        for _ in 1:n_sub_loc
            step_loc[] += 1
            for (_, rm_t) in pairs(tracers)
                # Restore prescribed m BEFORE each tracer (physics_phases.jl:454)
                for p in 1:6; copyto!(air.m[p], ws_vr.m_save[p]); end

                # Two half-substep Lin-Rood calls (physics_phases.jl:458-463)
                fv_tp_2d_cs!(rm_t, air.m, gpu.am, gpu.bm,
                              grid, Val(7), ws, ws_lr;
                              damp_coeff=damp_loc)
                fv_tp_2d_cs!(rm_t, air.m, gpu.am, gpu.bm,
                              grid, Val(7), ws, ws_lr;
                              damp_coeff=FT(0))

                # Rescale rm to prescribed-m basis (physics_phases.jl:467-469)
                for p in 1:6
                    rm_t[p] .*= ws_vr.m_save[p] ./ air.m[p]
                end
            end
        end

        # Restore m for PE computation (physics_phases.jl:479)
        for p in 1:6; copyto!(air.m[p], ws_vr.m_save[p]); end

        # ── PE + Remap (physics_phases.jl:495-520) ────────────────────
        use_hybrid = (pe_method == :hybrid) && phys.qv_loaded[]

        if use_hybrid
            compute_source_pe_from_hybrid!(ws_vr, gpu.delp, phys.qv_gpu, gc, grid)
        end

        if has_next
            ng = next_gpu(sched)
            if use_hybrid
                compute_target_pe_from_ps_hybrid!(ws_vr, ng.delp, phys.qv_gpu, gc, grid)
            else
                compute_target_pressure_from_dry_delp_direct!(ws_vr, ng.delp, phys.qv_gpu, gc, grid)
            end
        else
            compute_target_pressure_from_mass_direct!(ws_vr, air.m, gc, grid)
        end

        # Vertical remap (physics_phases.jl:512-514)
        for (_, rm_t) in pairs(tracers)
            vertical_remap_cs!(rm_t, ws_vr.m_save, ws_vr, ws, gc, grid;
                               hybrid_pe=use_hybrid)
        end

        # Update air mass to target (physics_phases.jl:517-520)
        update_air_mass_from_target!(air.m, ws_vr, gc, grid)
        for p in 1:6; copyto!(air.m_ref[p], air.m[p]); end

        # m_wet update (physics_phases.jl:526-532)
        if phys.qv_loaded[]
            for p in 1:6
                air.m_wet[p] .= air.m[p] ./ max.(FT(1) .- phys.qv_gpu[p], eps(FT))
            end
        else
            for p in 1:6; copyto!(air.m_wet[p], air.m[p]); end
        end

        # ── Diagnostics ──────────────────────────────────────────────
        rm_co2 = tracers[:co2]
        if w % diag_interval == 0 || w == 1 || w == min(n_windows, n_total)
            ei_sfc = edge_vs_interior_stats(rm_co2, air.m, Nz)
            ei_mid = edge_vs_interior_stats(rm_co2, air.m, div(Nz, 2))
            ei_toa = edge_vs_interior_stats(rm_co2, air.m, 1)
            push!(history, (w=w,
                sfc_edge_std=ei_sfc.edge.std, sfc_int_std=ei_sfc.interior.std,
                mid_edge_std=ei_mid.edge.std, mid_int_std=ei_mid.interior.std,
                toa_edge_std=ei_toa.edge.std, toa_int_std=ei_toa.interior.std))
            Printf.@printf("  w=%2d: sfc e/i=%.5f/%.5f (%.1fx)  mid=%.5f/%.5f (%.1fx)  toa=%.5f/%.5f (%.1fx)\n",
                w,
                ei_sfc.edge.std, ei_sfc.interior.std,
                ei_sfc.edge.std / max(ei_sfc.interior.std, 1e-12),
                ei_mid.edge.std, ei_mid.interior.std,
                ei_mid.edge.std / max(ei_mid.interior.std, 1e-12),
                ei_toa.edge.std, ei_toa.interior.std,
                ei_toa.edge.std / max(ei_toa.interior.std, 1e-12))

            if diagnostics_fn !== nothing
                diagnostics_fn(w, tracers, air.m)
            end
        end

        # ── Buffer management (run_loop.jl:166-167) ──────────────────
        wait_load!(sched)
        swap!(sched)
    end

    return history
end

println("run_production_windows! defined (production-faithful pipeline).")
=#

# =====================================================================
# BLOCK 4: Test A — Uniform 400 ppm (Baseline)
# Any deviation from 400 ppm is purely numerical error.
# Uses production pipeline with hybrid PE + real QV.
# =====================================================================
#=

# Reset to uniform 400 ppm
rm_co2 = tracers[:co2]
for p in 1:6
    m_cpu = Array(air.m[p])
    copyto!(rm_co2[p], CuArray(400f-6 .* m_cpu))
end

# Verify vertical ordering before test
println("=== TEST A: Uniform 400 ppm, production pipeline (hybrid PE) ===")
println("Vertical ordering check:")
verify_vertical_ordering(current_gpu(sched).delp, Hp, Nc, Nz)

println("\n--- Initial ---")
level_stats_compact(rm_co2, air.m)

# Reload first window for clean state
initial_load!(sched, driver, grid, 1; kw...)

n_test = min(48, n_win_total)
history_A = run_production_windows!(tracers, air, phys, sched, model, grid, gc,
    ws, ws_lr, ws_vr; n_windows=n_test, pe_method=:hybrid, diag_interval=4)

println("\n--- Final state after $(n_test) windows (hybrid PE) ---")
println("Global levels:")
level_stats_compact(rm_co2, air.m)

# Surface (k=Nz)
println("\nSurface (k=$Nz):")
print_panel_report(panel_stats(rm_co2, air.m, Nz))
print_edge_interior_report(edge_vs_interior_stats(rm_co2, air.m, Nz))

println("\nZonal mean by latitude band (surface):")
zonal_mean_by_lat_band(rm_co2, air.m, Nz)

# Mid-level
k_mid = div(Nz, 2)
println("\nMid-level (k=$k_mid):")
print_panel_report(panel_stats(rm_co2, air.m, k_mid))
print_edge_interior_report(edge_vs_interior_stats(rm_co2, air.m, k_mid))

# TOA
println("\nTOA (k=1):")
print_panel_report(panel_stats(rm_co2, air.m, 1))
print_edge_interior_report(edge_vs_interior_stats(rm_co2, air.m, 1))

# Mass conservation
total_rm = sum(p -> sum(Array(rm_co2[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]), 1:6)
total_m  = sum(p -> sum(Array(air.m[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]), 1:6)
Printf.@printf("\nGlobal VMR: %.6f ppm (should be ~400)\n", total_rm / total_m * 1e6)
=#

# =====================================================================
# BLOCK 5: Test B — Latitudinal Gradient (390 SP → 410 NP)
# Tests smooth gradient transport across panel boundaries.
# =====================================================================
#=

println("=== TEST B: Latitudinal gradient 390→410 ppm, production pipeline ===")

# Set IC: linear gradient in latitude, uniform in vertical
rm_co2 = tracers[:co2]
for p in 1:6
    m_cpu = Array(air.m[p])
    rm_cpu = similar(m_cpu)
    lat = φᶜ[p]  # Nc×Nc matrix of latitudes
    for j in 1:Nc, i in 1:Nc
        # Linear: 390 ppm at -90°, 410 ppm at +90°
        vmr = (390.0 + 20.0 * (lat[i, j] + 90.0) / 180.0) * 1e-6
        for k in 1:Nz
            rm_cpu[Hp+i, Hp+j, k] = Float32(vmr) * m_cpu[Hp+i, Hp+j, k]
        end
    end
    # Zero out halos
    fill!(view(rm_cpu, 1:Hp, :, :), 0f0)
    fill!(view(rm_cpu, Hp+Nc+1:Nc+2Hp, :, :), 0f0)
    fill!(view(rm_cpu, :, 1:Hp, :), 0f0)
    fill!(view(rm_cpu, :, Hp+Nc+1:Nc+2Hp, :), 0f0)
    copyto!(rm_co2[p], CuArray(rm_cpu))
end

println("--- Initial ---")
println("Zonal mean by latitude band (surface, k=$Nz):")
zonal_mean_by_lat_band(rm_co2, air.m, Nz)

# Reload and run
initial_load!(sched, driver, grid, 1; kw...)

n_test = min(48, n_win_total)
history_B = run_production_windows!(tracers, air, phys, sched, model, grid, gc,
    ws, ws_lr, ws_vr; n_windows=n_test, pe_method=:hybrid, diag_interval=8)

println("\n--- Final after $(n_test) windows ---")
println("Zonal mean by latitude band (surface, k=$Nz):")
zonal_mean_by_lat_band(rm_co2, air.m, Nz)

println("\nSurface (k=$Nz):")
print_panel_report(panel_stats(rm_co2, air.m, Nz))
print_edge_interior_report(edge_vs_interior_stats(rm_co2, air.m, Nz))

println("\nMid-level (k=$(div(Nz,2))):")
print_panel_report(panel_stats(rm_co2, air.m, div(Nz, 2)))
print_edge_interior_report(edge_vs_interior_stats(rm_co2, air.m, div(Nz, 2)))
=#

# =====================================================================
# BLOCK 6: Test C — Hemisphere Contrast (410 NH, 390 SH)
# Sharp equatorial step — wave artifacts show as wavenumber-4/6 oscillations.
# =====================================================================
#=

println("=== TEST C: Hemisphere contrast 410 NH / 390 SH, production pipeline ===")

rm_co2 = tracers[:co2]
for p in 1:6
    m_cpu = Array(air.m[p])
    rm_cpu = similar(m_cpu)
    lat = φᶜ[p]
    for j in 1:Nc, i in 1:Nc
        vmr = lat[i, j] >= 0.0 ? 410f-6 : 390f-6
        for k in 1:Nz
            rm_cpu[Hp+i, Hp+j, k] = vmr * m_cpu[Hp+i, Hp+j, k]
        end
    end
    fill!(view(rm_cpu, 1:Hp, :, :), 0f0)
    fill!(view(rm_cpu, Hp+Nc+1:Nc+2Hp, :, :), 0f0)
    fill!(view(rm_cpu, :, 1:Hp, :), 0f0)
    fill!(view(rm_cpu, :, Hp+Nc+1:Nc+2Hp, :), 0f0)
    copyto!(rm_co2[p], CuArray(rm_cpu))
end

println("--- Initial ---")
println("Zonal mean by latitude band (surface):")
zonal_mean_by_lat_band(rm_co2, air.m, Nz)

initial_load!(sched, driver, grid, 1; kw...)

n_test = min(48, n_win_total)
history_C = run_production_windows!(tracers, air, phys, sched, model, grid, gc,
    ws, ws_lr, ws_vr; n_windows=n_test, pe_method=:hybrid, diag_interval=8)

println("\n--- Final after $(n_test) windows ---")
println("Zonal mean by latitude band (surface):")
zonal_mean_by_lat_band(rm_co2, air.m, Nz)

println("\nSurface:")
print_panel_report(panel_stats(rm_co2, air.m, Nz))
print_edge_interior_report(edge_vs_interior_stats(rm_co2, air.m, Nz))

# Focus on 30-60°S where the wave pattern is reported
println("\nDetailed 30-60°S latitude band:")
zonal_mean_by_lat_band(rm_co2, air.m, Nz;
    bands=[(-60,-55), (-55,-50), (-50,-45), (-45,-40), (-40,-35), (-35,-30)])
=#

# =====================================================================
# BLOCK 7: Test D — A/B PE Method Comparison
# Same uniform IC, compare hybrid PE (production) vs direct cumsum (old).
# =====================================================================
#=

println("=== TEST D: A/B PE method comparison (uniform 400 ppm) ===")

# First: PE method structural comparison with current met data
println("\n--- PE method structural comparison ---")
gpu = current_gpu(sched)
pe_method_comparison(ws_vr, gpu.delp, phys.qv_gpu, gc, grid)

# Run A: hybrid PE (production)
println("\n--- Run A: hybrid PE (production) ---")
rm_co2 = tracers[:co2]
for p in 1:6
    m_cpu = Array(air.m[p])
    copyto!(rm_co2[p], CuArray(400f-6 .* m_cpu))
end
initial_load!(sched, driver, grid, 1; kw...)

n_test = min(48, n_win_total)
history_D_hybrid = run_production_windows!(tracers, air, phys, sched, model, grid, gc,
    ws, ws_lr, ws_vr; n_windows=n_test, pe_method=:hybrid, diag_interval=12)

# Save final state
vmr_hybrid_sfc = Float64[]
for p in 1:6
    rm = Array(rm_co2[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz]
    m  = Array(air.m[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz]
    append!(vmr_hybrid_sfc, vec(Float64.(rm) ./ Float64.(m) .* 1e6))
end
mass_hybrid = sum(p -> sum(Array(rm_co2[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]), 1:6)

# Run B: direct cumsum PE (old test approach)
println("\n--- Run B: direct cumsum PE (old approach) ---")
for p in 1:6
    m_cpu = Array(air.m[p])
    copyto!(rm_co2[p], CuArray(400f-6 .* m_cpu))
end
initial_load!(sched, driver, grid, 1; kw...)

history_D_direct = run_production_windows!(tracers, air, phys, sched, model, grid, gc,
    ws, ws_lr, ws_vr; n_windows=n_test, pe_method=:direct, diag_interval=12)

vmr_direct_sfc = Float64[]
for p in 1:6
    rm = Array(rm_co2[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz]
    m  = Array(air.m[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz]
    append!(vmr_direct_sfc, vec(Float64.(rm) ./ Float64.(m) .* 1e6))
end
mass_direct = sum(p -> sum(Array(rm_co2[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]), 1:6)

# Compare
println("\n--- A/B Comparison ---")
Printf.@printf("  Hybrid:  sfc std=%.5f ppm, total mass=%.6e\n",
    std(vmr_hybrid_sfc), mass_hybrid)
Printf.@printf("  Direct:  sfc std=%.5f ppm, total mass=%.6e\n",
    std(vmr_direct_sfc), mass_direct)
Printf.@printf("  Mass diff: %.4e%%\n",
    (mass_hybrid - mass_direct) / max(abs(mass_hybrid), 1e-30) * 100)

if !isempty(history_D_hybrid) && !isempty(history_D_direct)
    lh = last(history_D_hybrid)
    ld = last(history_D_direct)
    Printf.@printf("  Final edge/int ratio — Hybrid: sfc=%.2fx mid=%.2fx  Direct: sfc=%.2fx mid=%.2fx\n",
        lh.sfc_edge_std / max(lh.sfc_int_std, 1e-12),
        lh.mid_edge_std / max(lh.mid_int_std, 1e-12),
        ld.sfc_edge_std / max(ld.sfc_int_std, 1e-12),
        ld.mid_edge_std / max(ld.mid_int_std, 1e-12))
end
=#

# =====================================================================
# BLOCK 8: Test E — QV Sensitivity
# Compare transport with real QV vs forced QV=0 (moist air mass path).
# =====================================================================
#=

println("=== TEST E: QV sensitivity (real QV vs QV=0) ===")

# Run A: with real QV (production)
println("\n--- Run A: real QV (production) ---")
rm_co2 = tracers[:co2]
for p in 1:6
    m_cpu = Array(air.m[p])
    copyto!(rm_co2[p], CuArray(400f-6 .* m_cpu))
end
initial_load!(sched, driver, grid, 1; kw...)

# Ensure QV is used
@assert phys.qv_loaded[] "QV should be loaded for this test"

n_test = min(48, n_win_total)
history_E_qv = run_production_windows!(tracers, air, phys, sched, model, grid, gc,
    ws, ws_lr, ws_vr; n_windows=n_test, pe_method=:hybrid, diag_interval=12)

vmr_qv_sfc = Float64[]
for p in 1:6
    rm = Array(rm_co2[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz]
    m  = Array(air.m[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz]
    append!(vmr_qv_sfc, vec(Float64.(rm) ./ Float64.(m) .* 1e6))
end

# Run B: force QV=0 — override qv_loaded to false before each window
# We do this by temporarily wrapping the run loop
println("\n--- Run B: forced QV=0 ---")
for p in 1:6
    m_cpu = Array(air.m[p])
    copyto!(rm_co2[p], CuArray(400f-6 .* m_cpu))
end
initial_load!(sched, driver, grid, 1; kw...)

# Override: disable QV for this run
phys.qv_loaded[] = false

history_E_noqv = run_production_windows!(tracers, air, phys, sched, model, grid, gc,
    ws, ws_lr, ws_vr; n_windows=n_test, pe_method=:hybrid, diag_interval=12)

vmr_noqv_sfc = Float64[]
for p in 1:6
    rm = Array(rm_co2[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz]
    m  = Array(air.m[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz]
    append!(vmr_noqv_sfc, vec(Float64.(rm) ./ Float64.(m) .* 1e6))
end

# Restore QV
phys.qv_loaded[] = true

println("\n--- QV Sensitivity Comparison ---")
Printf.@printf("  With QV:    sfc std=%.5f ppm\n", std(vmr_qv_sfc))
Printf.@printf("  Without QV: sfc std=%.5f ppm\n", std(vmr_noqv_sfc))

if !isempty(history_E_qv) && !isempty(history_E_noqv)
    lq = last(history_E_qv)
    ln = last(history_E_noqv)
    Printf.@printf("  Final edge/int — QV: sfc=%.2fx mid=%.2fx  NoQV: sfc=%.2fx mid=%.2fx\n",
        lq.sfc_edge_std / max(lq.sfc_int_std, 1e-12),
        lq.mid_edge_std / max(lq.mid_int_std, 1e-12),
        ln.sfc_edge_std / max(ln.sfc_int_std, 1e-12),
        ln.mid_edge_std / max(ln.mid_int_std, 1e-12))
end
=#

# =====================================================================
# BLOCK 9: Test F — Vertical Structure Test
# Surface-only perturbation — watch remap spread it vertically.
# Verifies k=1=TOA, k=Nz=surface convention is correct in remap.
# =====================================================================
#=

println("=== TEST F: Surface perturbation (400 ppm + 50 ppm at k=$Nz only) ===")

# Verify vertical ordering one more time
println("Vertical ordering verification:")
verify_vertical_ordering(current_gpu(sched).delp, Hp, Nc, Nz)

# Set IC: 400 ppm everywhere, +50 ppm at surface only (k=Nz)
rm_co2 = tracers[:co2]
for p in 1:6
    m_cpu = Array(air.m[p])
    rm_cpu = 400f-6 .* m_cpu
    # Add 50 ppm at surface level (k=Nz)
    for j in 1:Nc, i in 1:Nc
        rm_cpu[Hp+i, Hp+j, Nz] += 50f-6 * m_cpu[Hp+i, Hp+j, Nz]
    end
    copyto!(rm_co2[p], CuArray(rm_cpu))
end

println("\n--- Initial ---")
println("Vertical profile (should show 450 at k=$Nz, 400 elsewhere):")
level_stats_compact(rm_co2, air.m; levels=[1, 5, 10, 20, 36, 50, 60, 65, 70, Nz-1, Nz])

initial_load!(sched, driver, grid, 1; kw...)

n_test = min(48, n_win_total)
history_F = run_production_windows!(tracers, air, phys, sched, model, grid, gc,
    ws, ws_lr, ws_vr; n_windows=n_test, pe_method=:hybrid, diag_interval=12)

println("\n--- Final after $(n_test) windows ---")
println("Vertical profile (perturbation should spread upward from k=$Nz):")
level_stats_compact(rm_co2, air.m; levels=[1, 5, 10, 20, 36, 50, 60, 65, 70, Nz-1, Nz])

# Check for spurious oscillations in vertical profile
println("\nPer-level std (artifact = non-monotonic std increase toward surface):")
for k in Nz:-1:max(1, Nz-15)
    q_all = Float64[]
    for p in 1:6
        rm = Array(rm_co2[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
        m  = Array(air.m[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
        append!(q_all, vec(Float64.(rm) ./ Float64.(m) .* 1e6))
    end
    Printf.@printf("  k=%2d: mean=%9.3f std=%9.5f\n", k, mean(q_all), std(q_all))
end

println("\nSurface edge/interior:")
print_edge_interior_report(edge_vs_interior_stats(rm_co2, air.m, Nz))
println("\nZonal mean by latitude band (surface):")
zonal_mean_by_lat_band(rm_co2, air.m, Nz)
=#

# =====================================================================
# BLOCK 10: Summary + Interpretation
# =====================================================================
#=

println("=" ^ 70)
println("SUMMARY OF ALL TESTS")
println("=" ^ 70)

function report_test(name, history)
    if isempty(history)
        println("  $name: NO DATA")
        return
    end
    h = last(history)
    sfc_ratio = h.sfc_edge_std / max(h.sfc_int_std, 1e-12)
    mid_ratio = h.mid_edge_std / max(h.mid_int_std, 1e-12)
    toa_ratio = h.toa_edge_std / max(h.toa_int_std, 1e-12)

    # Check if ratio grows over time (wave amplification)
    growing = false
    if length(history) >= 3
        first_ratio = history[1].sfc_edge_std / max(history[1].sfc_int_std, 1e-12)
        growing = sfc_ratio > first_ratio * 1.5
    end

    pass_sfc = sfc_ratio < 1.5
    pass_mid = mid_ratio < 1.5
    status = (pass_sfc && pass_mid && !growing) ? "PASS" : "FAIL"

    Printf.@printf("  %-20s  sfc=%.2fx  mid=%.2fx  toa=%.2fx  growing=%s  → %s\n",
        name, sfc_ratio, mid_ratio, toa_ratio, growing ? "YES" : "no", status)
end

if @isdefined(history_A)
    report_test("A: Uniform", history_A)
end
if @isdefined(history_B)
    report_test("B: Lat gradient", history_B)
end
if @isdefined(history_C)
    report_test("C: Hemisphere", history_C)
end
if @isdefined(history_D_hybrid)
    report_test("D: Hybrid PE", history_D_hybrid)
end
if @isdefined(history_D_direct)
    report_test("D: Direct PE", history_D_direct)
end
if @isdefined(history_E_qv)
    report_test("E: With QV", history_E_qv)
end
if @isdefined(history_E_noqv)
    report_test("E: No QV", history_E_noqv)
end
if @isdefined(history_F)
    report_test("F: Vert struct", history_F)
end

println()
println("Pass criteria: edge/interior ratio < 1.5x, not growing over time")
println("Key diagnostic: if 30-60°S band shows higher std than 0-30°S,")
println("  the wave pattern is latitude-dependent (likely Rossby-like).")
println()
println("If all tests PASS but production run shows waves:")
println("  → Bug is in emissions, diffusion, convection, or output pipeline")
println("If tests FAIL:")
println("  → Compare A/B (Block 7/8) to identify PE method or QV as culprit")
=#

# =====================================================================
# SUCCESS CRITERIA
# =====================================================================
#
# | Test | Metric                        | Pass         | Fail             |
# |------|-------------------------------|--------------|------------------|
# | A    | Uniform: panel std spread     | < 2x         | > 5x             |
# | A    | Uniform: edge/interior ratio  | < 1.5x       | > 3x             |
# | B    | Gradient: zonal monotonicity  | preserved    | reversed bands   |
# | C    | Hemisphere: equatorial smooth | no wavenum-6 | visible pattern  |
# | D    | A/B PE: edge ratio difference | < 20% diff   | > 50% diff       |
# | E    | QV sensitivity: std ratio     | similar      | > 2x different   |
# | F    | Vertical: upward spread       | monotonic    | oscillatory      |
# | ALL  | Wave growth over windows      | stable       | growing per win  |
