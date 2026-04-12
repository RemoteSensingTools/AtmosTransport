# ===========================================================================
# MCP Rossby Wave Detection Tests
#
# Run these blocks sequentially in an MCP Julia session.
# Detects panel-boundary artifacts in CS transport by checking:
#   1. Per-panel statistics (should be identical for uniform IC)
#   2. Panel-edge vs interior noise (edges should not be noisier)
#   3. Multi-window evolution (wave growth over time)
#   4. Linrood vs Strang comparison
#
# Prerequisites: debug_remap_2win.toml config, GEOS-IT C180 met data
# ===========================================================================

# =====================================================================
# BLOCK 1: Setup — load model, extract internals
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
    compute_target_pressure_from_mass_direct!,
    fix_target_bottom_pe!
using AtmosTransport.Grids: halo_exchange!
import TOML, Statistics, Printf
using Statistics: mean, std

config = TOML.parsefile("config/runs/debug_remap_2win.toml")
# Override to get more windows for wave growth detection
config["met_data"]["end_date"] = "2021-12-02"  # 2 days = ~48 windows
model = build_model_from_config(config)

grid = model.grid
Hp = grid.Hp; Nc = grid.Nc; Nz = grid.Nz
FT = Float32

# Extract internals (replicate _run_loop! allocation)
using AtmosTransport.Models: build_io_scheduler, allocate_physics_buffers,
    allocate_tracers, allocate_air_mass, allocate_geometry_and_workspace,
    _needs_linrood, _needs_vertical_remap, physics_load_kwargs,
    initial_load!

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

n_sub = AtmosTransport.Models.steps_per_window(model.met_data)
dt_sub = FT(model.met_data.dt)
half_dt = FT(dt_sub / 2)

# Load first window
kw = physics_load_kwargs(phys, grid)
initial_load!(sched, model.met_data, grid, 1; kw...)
upload_met!(sched)
load_and_upload_physics!(phys, sched, model.met_data, grid, 1; arch=model.architecture)
compute_ps_phase!(phys, sched, grid)
process_met_after_upload!(sched, phys, grid, model.met_data, half_dt)
compute_air_mass_phase!(sched, air, phys, grid, gc;
    dry_correction=get(model.metadata, "dry_correction", true))
save_reference_mass!(sched, air, grid)

println("Setup complete: C$(Nc), $(n_sub) sub/win, linrood=$(_use_lr), vremap=$(_use_vr)")
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

"""Edge vs interior statistics for a given level k.
   edge_width: number of cells from panel edge to include in 'edge' region.
   Returns (region, min, max, mean, std) for 'edge' and 'interior'."""
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

"""Per-edge-pair statistics: for each of the 12 panel edges, compute std in the
   strip adjacent to that edge. High std on specific edges → rotation/halo bug."""
function per_edge_stats(rm_panels, m_panels, k::Int; strip_width::Int=5)
    # 12 edges: each panel has 4 edges (W, E, S, N), but each edge is shared
    # We label by panel and side
    edges = []
    for p in 1:6
        rm = Array(rm_panels[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
        m  = Array(m_panels[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
        q = Float64.(rm) ./ Float64.(m) .* 1e6
        sw = strip_width
        # West edge: i=1:sw, all j
        q_w = vec(q[1:sw, :])
        push!(edges, (panel=p, side="W", mean=mean(q_w), std=std(q_w)))
        # East edge: i=Nc-sw+1:Nc, all j
        q_e = vec(q[Nc-sw+1:Nc, :])
        push!(edges, (panel=p, side="E", mean=mean(q_e), std=std(q_e)))
        # South edge: all i, j=1:sw
        q_s = vec(q[:, 1:sw])
        push!(edges, (panel=p, side="S", mean=mean(q_s), std=std(q_s)))
        # North edge: all i, j=Nc-sw+1:Nc
        q_n = vec(q[:, Nc-sw+1:Nc])
        push!(edges, (panel=p, side="N", mean=mean(q_n), std=std(q_n)))
    end
    return edges
end

"""Global level stats (same as before but more compact)."""
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

"""Print edge diagnostics sorted by std (worst edges first)."""
function print_edge_report(edges; top_n=10)
    sorted = sort(edges, by=e->e.std, rev=true)
    println("  Worst edges (by std):")
    for (i, e) in enumerate(sorted[1:min(top_n, length(sorted))])
        Printf.@printf("    P%d-%s: mean=%9.4f std=%9.5f\n",
            e.panel, e.side, e.mean, e.std)
    end
end

"""Print panel report."""
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

"""Print edge vs interior report."""
function print_edge_interior_report(ei)
    Printf.@printf("  Edge:     mean=%9.4f std=%9.5f (n=%d)\n",
        ei.edge.mean, ei.edge.std, ei.edge.n)
    Printf.@printf("  Interior: mean=%9.4f std=%9.5f (n=%d)\n",
        ei.interior.mean, ei.interior.std, ei.interior.n)
    ratio = ei.edge.std / max(ei.interior.std, 1e-12)
    Printf.@printf("  Edge/Interior std ratio: %.2f%s\n",
        ratio, ratio > 2.0 ? " ⚠️  PANEL BOUNDARY ARTIFACT" : " ✓")
end

println("Diagnostic functions loaded.")
=#

# =====================================================================
# BLOCK 3: Test A — Uniform IC, single window (12 substeps)
# Baseline: any deviation from 400 ppm is purely numerical
# =====================================================================
#=

# Reset to uniform 400 ppm
rm_co2 = tracers[:co2]
for p in 1:6
    m_cpu = Array(air.m[p])
    copyto!(rm_co2[p], CuArray(400f-6 .* m_cpu))
end

# Save initial state
rm_init = [copy(Array(rm_co2[p])) for p in 1:6]

# Prepare remap workspace
if ws_vr !== nothing
    for p in 1:6; copyto!(ws_vr.m_save[p], air.m[p]); end
    compute_target_pressure_from_mass_direct!(ws_vr, air.m, gc, grid)
end

println("=== TEST A: Uniform 400 ppm, 1 window ($(n_sub) substeps) ===")
println("--- Initial ---")
level_stats_compact(rm_co2, air.m)

# Run substeps with rescale (current approach within a window)
step = Ref(0)
for sub in 1:n_sub
    if ws_lr !== nothing
        fv_tp_2d_cs!(rm_co2, air.m, current_gpu(sched).am, current_gpu(sched).bm,
                      grid, Val(7), ws, ws_lr;
                      damp_coeff=model.advection_scheme.damp_coeff)
    else
        strang_split_massflux_ppm!(rm_co2, air.m,
            current_gpu(sched).am, current_gpu(sched).bm, current_gpu(sched).cm,
            grid, Val(7), ws; damp_coeff=model.advection_scheme.damp_coeff)
    end
    # Rescale (within-window approach)
    if ws_vr !== nothing
        for p in 1:6
            rm_co2[p] .*= ws_vr.m_save[p] ./ air.m[p]
            copyto!(air.m[p], ws_vr.m_save[p])
        end
    end
    step[] += 1
end

println("\n--- After 1 window ($(n_sub) substeps) ---")
println("Global levels:")
level_stats_compact(rm_co2, air.m)

for k in [1, 36, 72]
    println("\nLevel k=$k:")
    print_panel_report(panel_stats(rm_co2, air.m, k))
    print_edge_interior_report(edge_vs_interior_stats(rm_co2, air.m, k))
    print_edge_report(per_edge_stats(rm_co2, air.m, k))
end
=#

# =====================================================================
# BLOCK 4: Test B — Uniform IC, multi-window (with remap at boundaries)
# This is where Rossby waves develop — remap at window boundaries
# =====================================================================
#=

# Reset to uniform 400 ppm
for p in 1:6
    m_cpu = Array(air.m[p])
    copyto!(rm_co2[p], CuArray(400f-6 .* m_cpu))
end

n_win_total = AtmosTransport.Models.total_windows(model.met_data)
n_test_windows = min(n_win_total, 24)  # 1 day = 24 windows

println("=== TEST B: Uniform 400 ppm, $(n_test_windows) windows (with remap) ===")
println("--- Initial ---")
level_stats_compact(rm_co2, air.m)

# Track edge/interior ratio over time for k=1 (surface) and k=72 (TOA)
history = []

for w in 1:n_test_windows
    has_next = w < n_win_total

    # Upload met for this window
    upload_met!(sched)
    load_and_upload_physics!(phys, sched, model.met_data, grid, w; arch=model.architecture)
    compute_ps_phase!(phys, sched, grid)

    if has_next
        AtmosTransport.Models.begin_load!(sched, model.met_data, grid, w + 1; kw...)
    end

    process_met_after_upload!(sched, phys, grid, model.met_data, half_dt)
    compute_air_mass_phase!(sched, air, phys, grid, gc;
        dry_correction=get(model.metadata, "dry_correction", true))
    save_reference_mass!(sched, air, grid)

    if ws_vr !== nothing
        for p in 1:6; copyto!(ws_vr.m_save[p], air.m[p]); end
    end

    # Deferred DELP fetch
    wait_and_upload_next_delp!(sched, grid)

    # Substep loop
    for sub in 1:n_sub
        if ws_lr !== nothing
            fv_tp_2d_cs!(rm_co2, air.m, current_gpu(sched).am, current_gpu(sched).bm,
                          grid, Val(7), ws, ws_lr;
                          damp_coeff=model.advection_scheme.damp_coeff)
        else
            strang_split_massflux_ppm!(rm_co2, air.m,
                current_gpu(sched).am, current_gpu(sched).bm, current_gpu(sched).cm,
                grid, Val(7), ws; damp_coeff=model.advection_scheme.damp_coeff)
        end
        if ws_vr !== nothing
            for p in 1:6
                rm_co2[p] .*= ws_vr.m_save[p] ./ air.m[p]
                copyto!(air.m[p], ws_vr.m_save[p])
            end
        end
    end

    # Window-boundary vertical remap
    if ws_vr !== nothing && has_next
        ng = next_gpu(sched)
        _dry_corr = get(model.metadata, "dry_correction", true)
        if _dry_corr && phys.qv_loaded[]
            compute_target_pressure_from_dry_delp_direct!(ws_vr, ng.delp, phys.qv_gpu, gc, grid)
        else
            AtmosTransport.Advection.compute_target_pressure_from_delp_direct!(ws_vr, ng.delp, gc, grid)
        end
        _remap_fix = get(model.metadata, "remap_pressure_fix", true)
        if _remap_fix
            fix_target_bottom_pe!(ws_vr, ws_vr.m_save, gc, grid)
        end
        for (_, rm_t) in pairs(tracers)
            vertical_remap_cs!(rm_t, ws_vr.m_save, ws_vr, ws, gc, grid)
        end
        # Update air mass to target
        AtmosTransport.Advection.update_air_mass_from_target!(air.m, ws_vr, gc, grid)
    end

    # Record diagnostics every 4 windows
    if w % 4 == 0 || w == 1 || w == n_test_windows
        ei_sfc = edge_vs_interior_stats(rm_co2, air.m, 72)  # surface = k=72
        ei_toa = edge_vs_interior_stats(rm_co2, air.m, 1)   # TOA = k=1
        ei_mid = edge_vs_interior_stats(rm_co2, air.m, 36)  # mid
        push!(history, (w=w,
            sfc_edge_std=ei_sfc.edge.std, sfc_int_std=ei_sfc.interior.std,
            toa_edge_std=ei_toa.edge.std, toa_int_std=ei_toa.interior.std,
            mid_edge_std=ei_mid.edge.std, mid_int_std=ei_mid.interior.std))
        Printf.@printf("  w=%2d: sfc edge/int=%.5f/%.5f (%.1fx)  mid=%.5f/%.5f (%.1fx)  toa=%.5f/%.5f (%.1fx)\n",
            w,
            ei_sfc.edge.std, ei_sfc.interior.std,
            ei_sfc.edge.std / max(ei_sfc.interior.std, 1e-12),
            ei_mid.edge.std, ei_mid.interior.std,
            ei_mid.edge.std / max(ei_mid.interior.std, 1e-12),
            ei_toa.edge.std, ei_toa.interior.std,
            ei_toa.edge.std / max(ei_toa.interior.std, 1e-12))
    end

    # Buffer management
    wait_load!(sched)
    swap!(sched)
end

println("\n--- Final state after $(n_test_windows) windows ---")
println("Global levels:")
level_stats_compact(rm_co2, air.m)

println("\nSurface (k=72):")
print_panel_report(panel_stats(rm_co2, air.m, 72))
print_edge_interior_report(edge_vs_interior_stats(rm_co2, air.m, 72))
print_edge_report(per_edge_stats(rm_co2, air.m, 72))

println("\nMid-level (k=36):")
print_panel_report(panel_stats(rm_co2, air.m, 36))
print_edge_interior_report(edge_vs_interior_stats(rm_co2, air.m, 36))

println("\nTOA (k=1):")
print_panel_report(panel_stats(rm_co2, air.m, 1))
print_edge_interior_report(edge_vs_interior_stats(rm_co2, air.m, 1))
print_edge_report(per_edge_stats(rm_co2, air.m, 1))

# Total mass check
total_rm = sum(p -> sum(Array(rm_co2[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]), 1:6)
total_m  = sum(p -> sum(Array(air.m[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]), 1:6)
Printf.@printf("\nGlobal VMR: %.6f ppm\n", total_rm / total_m * 1e6)
Printf.@printf("Total rm: %.6e (should be ~%.6e)\n", total_rm,
    400e-6 * total_m)
=#

# =====================================================================
# BLOCK 5: Test C — Surface perturbation, multi-window
# Hot spot at panel 1 center — watch it spread and check for
# spurious reflections at panel edges
# =====================================================================
#=

# Reset to uniform + surface perturbation
for p in 1:6
    m_cpu = Array(air.m[p])
    rm_cpu = 400f-6 .* m_cpu
    # Add 50 ppm hot spot at center of panel 1, surface level
    if p == 1
        ci = div(Nc, 2)
        for dj in -10:10, di in -10:10
            ii = Hp + ci + di
            jj = Hp + ci + dj
            rm_cpu[ii, jj, 72] += 50f-6 * m_cpu[ii, jj, 72]
        end
    end
    copyto!(rm_co2[p], CuArray(rm_cpu))
end

println("=== TEST C: Surface hot spot (P1 center, +50 ppm), $(n_test_windows) windows ===")
println("--- Initial ---")
for k in [72, 60, 36, 1]
    ps = panel_stats(rm_co2, air.m, k)
    p1_std = ps[1].std
    other_std = mean([ps[p].std for p in 2:6])
    Printf.@printf("  k=%2d: P1 std=%9.5f  P2-6 mean std=%9.5f\n", k, p1_std, other_std)
end

# Re-run the same multi-window loop as Test B
# (Copy the window loop from Block 4 here, or call a shared function)
# For brevity, showing the key diagnostic at each checkpoint:

# ... [run same window loop as Block 4] ...

# After running, check:
# - Does the hot spot spread smoothly across panel boundaries?
# - Is there a "reflection" artifact at the P1 edges?
# - Do panels 2-6 show symmetric spreading?
println("\n--- After transport ---")
for k in [72, 60, 36, 1]
    println("Level k=$k:")
    print_panel_report(panel_stats(rm_co2, air.m, k))
    print_edge_interior_report(edge_vs_interior_stats(rm_co2, air.m, k))
end
=#

# =====================================================================
# BLOCK 6: Comparison — Linrood vs Strang
# Run both with identical IC and compare panel-edge noise
# Requires re-building model with linrood=false for Strang
# =====================================================================
#=

# To run Strang comparison:
# 1. Modify config: config["advection"]["linrood"] = false
# 2. Rebuild model
# 3. Re-run Block 4 (uniform, multi-window)
# 4. Compare history arrays

# Quick toggle (if model supports runtime switching):
# Otherwise, save the linrood history and re-run from scratch:
#   linrood_history = copy(history)
#   # ... rebuild with linrood=false, re-run Block 4 ...
#   strang_history = copy(history)

# Compare:
#   for (lh, sh) in zip(linrood_history, strang_history)
#       Printf.@printf("w=%2d: LR sfc edge/int=%.1fx  Strang=%.1fx\n",
#           lh.w,
#           lh.sfc_edge_std / max(lh.sfc_int_std, 1e-12),
#           sh.sfc_edge_std / max(sh.sfc_int_std, 1e-12))
#   end

println("Block 6: Run manually — rebuild model with linrood=false and re-run Block 4")
=#

# =====================================================================
# SUCCESS CRITERIA
# =====================================================================
#
# | Metric                          | Pass           | Fail (Rossby wave) |
# |---------------------------------|----------------|--------------------|
# | Uniform IC: panel std spread    | < 2x           | > 5x               |
# | Uniform IC: edge/interior ratio | < 1.5x         | > 3x               |
# | Uniform IC: specific edge std   | all similar    | 1-2 edges >> rest   |
# | Multi-window: ratio growth      | stable or slow | growing per window  |
# | Hot spot: symmetric spreading   | P2-6 similar   | asymmetric          |
# | Linrood vs Strang edge ratio    | LR <= Strang   | LR >> Strang        |
#
# Key diagnostic: if edge/interior ratio GROWS over windows, the remap
# is amplifying panel-boundary errors. If it's constant, the error is
# purely from horizontal splitting and doesn't compound.
