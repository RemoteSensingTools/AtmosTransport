#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Side-by-side: existing q-space Lin-Rood vs GCHP-faithful fv_tp_2d
#
# Uses the existing reader infrastructure for proper C-grid staggering.
# Initializes with a latitudinal gradient (not uniform) to expose
# differences in area-based vs mass-based pre-advection.
# Runs horizontal-only (no vertical) for N windows.
# ---------------------------------------------------------------------------

using AtmosTransport
using AtmosTransport.Grids
using AtmosTransport.Advection
using AtmosTransport.Architectures
using AtmosTransport.IO
using NCDatasets
using Statistics
using Printf

const CTM_FILE = expanduser("~/data/geosit_c180/20230601/GEOSIT.20230601.CTM_A1.C180.nc")
const MASS_FLUX_DT = Float32(450.0)
const N_WINDOWS = 24

# =====================================================================
# Read and stagger using existing infrastructure
# =====================================================================
function load_window_data(filepath, win; FT=Float32)
    # Use existing reader for DELP, MFXC, MFYC
    ts = read_geosfp_cs_timestep(filepath; FT, time_index=win,
                                  dt_met=MASS_FLUX_DT, convert_to_kgs=true)
    Hp = 3
    delp_h, mfxc, mfyc = to_haloed_panels(ts; Hp)
    am_stag, bm_stag = cgrid_to_staggered_panels(mfxc, mfyc)

    # Read CX/CY separately, stagger the same way as mass fluxes
    ds = NCDataset(filepath, "r")
    cx_raw = Array{FT}(ds["CX"][:, :, :, :, win])  # (Nc, Nc, 6, Nz)
    cy_raw = Array{FT}(ds["CY"][:, :, :, :, win])
    close(ds)

    # GEOS-IT is bottom-to-top — flip CX/CY to match reader's TOA-first convention
    if true  # GEOS-IT always needs flip (reader already flipped DELP/MFXC/MFYC)
        cx_raw = reverse(cx_raw, dims=4)
        cy_raw = reverse(cy_raw, dims=4)
    end

    # Split into per-panel 3D arrays (Nc × Nc × Nz) for cgrid_to_staggered_panels
    cx_panels = ntuple(p -> cx_raw[:, :, p, :], 6)
    cy_panels = ntuple(p -> cy_raw[:, :, p, :], 6)
    cx_stag, cy_stag = cgrid_to_staggered_panels(cx_panels, cy_panels)

    # Scale CX/CY: accumulated over mass_flux_dt, we apply per sub-step
    steps = ceil(Int, 3600.0 / 900.0)  # 4 sub-steps per window
    for p in 1:6
        cx_stag[p] ./= steps
        cy_stag[p] ./= steps
    end

    return (; delp=delp_h, am=am_stag, bm=bm_stag, cx=cx_stag, cy=cy_stag)
end

# =====================================================================
# Main
# =====================================================================
function main()
    @info "Loading window 1 from $CTM_FILE"
    w = load_window_data(CTM_FILE, 1)
    Nc = 180; Nz = 72; Hp = 3; FT = Float32
    N = Nc + 2Hp

    # Build grid with simple sigma coords (actual coords don't matter for horizontal test)
    arch = CPU()
    A_c = collect(FT, range(0, 101325, Nz + 1))
    B_c = FT.(1.0 .- A_c ./ 101325)
    vert = HybridSigmaPressure(A_c, B_c)
    grid = CubedSphereGrid(arch; FT, Nc, vertical=vert, halo=(Hp, 1))

    @info "Allocating workspaces"
    geom = GCHPGridGeometry(grid)
    ws_gchp = GCHPTransportWorkspace(grid)
    ws_lr = LinRoodWorkspace(grid)
    ref_panel = zeros(FT, N, N, Nz)
    ws = allocate_cs_massflux_workspace(ref_panel, Nc)

    # Air mass from DELP
    m_init = ntuple(6) do p
        m = zeros(FT, N, N, Nz)
        g = grid.gravity
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            m[Hp+i, Hp+j, k] = w.delp[p][Hp+i, Hp+j, k] * grid.Aᶜ[p][i, j] / g
        end
        m
    end

    # Initialize with LATITUDINAL GRADIENT: q = 400 + 10*sin(lat) ppm
    # This ensures non-uniform field so pre-advection differences manifest.
    q_init = ntuple(6) do p
        q = zeros(FT, N, N, Nz)
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            lat = grid.φᶜ[p][i, j]  # degrees
            q[Hp+i, Hp+j, k] = FT(400e-6 + 10e-6 * sind(lat))
        end
        q
    end

    # Copy initial state for both methods
    q_lr = ntuple(p -> copy(q_init[p]), 6)
    m_lr = ntuple(p -> copy(m_init[p]), 6)
    q_gc = ntuple(p -> copy(q_init[p]), 6)
    m_gc = ntuple(p -> copy(m_init[p]), 6)

    # Monitor levels: surface, ~750 hPa (k≈55), ~500 hPa (k≈40), ~200 hPa (k≈20)
    levels = [Nz, 55, 40, 20]
    names = ["Sfc(k=$Nz)", "750hPa(k=55)", "500hPa(k=40)", "200hPa(k=20)"]

    println()
    println("="^100)
    println("  Side-by-side: q-space Lin-Rood (LR) vs GCHP-faithful (GC)")
    println("  Grid: C180×72, $N_WINDOWS windows, Initial: 400 ± 10·sin(lat) ppm")
    println("="^100)
    @printf("%-6s  %-14s  %10s %10s  %10s %10s  %8s\n",
            "Win", "Level", "LR std", "GC std", "LR range", "GC range", "Δstd%")
    println("-"^100)

    for win in 1:N_WINDOWS
        # Load fresh met data each window
        wdata = load_window_data(CTM_FILE, min(win, 24))

        # Compute area fluxes for GCHP
        compute_area_fluxes!(ws_gchp, wdata.cx, wdata.cy, geom, grid)

        # --- Existing q-space Lin-Rood ---
        fv_tp_2d_cs_q!(q_lr, m_lr, wdata.am, wdata.bm,
                         grid, Val(7), ws, ws_lr)

        # --- GCHP-faithful ---
        fv_tp_2d_gchp!(q_gc, m_gc, wdata.am, wdata.bm,
                         wdata.cx, wdata.cy, geom, ws_gchp,
                         grid, Val(7), ws, ws_lr)

        # Report every 4 windows (every 4 hours)
        if win % 4 == 0 || win == 1
            for (k, nm) in zip(levels, names)
                lr_vals = Float64[]
                gc_vals = Float64[]
                for p in 1:6, j in 1:Nc, i in 1:Nc
                    push!(lr_vals, q_lr[p][Hp+i, Hp+j, k])
                    push!(gc_vals, q_gc[p][Hp+i, Hp+j, k])
                end
                lr_s = std(lr_vals) * 1e6
                gc_s = std(gc_vals) * 1e6
                lr_r = (maximum(lr_vals) - minimum(lr_vals)) * 1e6
                gc_r = (maximum(gc_vals) - minimum(gc_vals)) * 1e6
                pct = lr_s > 0 ? (gc_s - lr_s) / lr_s * 100 : 0.0
                @printf("%-6d  %-14s  %10.4f %10.4f  %10.4f %10.4f  %+7.2f%%\n",
                        win, nm, lr_s, gc_s, lr_r, gc_r, pct)
            end
            println("-"^100)
        end
    end

    # Final mass conservation check
    mass_lr = sum(sum(q_lr[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :] .*
                      m_lr[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]) for p in 1:6)
    mass_gc = sum(sum(q_gc[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :] .*
                      m_gc[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]) for p in 1:6)
    mass_init = sum(sum(q_init[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :] .*
                        m_init[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]) for p in 1:6)

    println()
    @printf("Mass conservation after %d windows:\n", N_WINDOWS)
    @printf("  Lin-Rood: Δ = %+.6f%%\n", (mass_lr - mass_init) / mass_init * 100)
    @printf("  GCHP:     Δ = %+.6f%%\n", (mass_gc - mass_init) / mass_init * 100)
end

main()
