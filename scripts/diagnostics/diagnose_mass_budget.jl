#!/usr/bin/env julia
# ============================================================================
# Diagnose mass conservation in ERA5 lat-lon PPM advection
#
# Three modes:
#   A) Flux telescoping — verify am/bm/cm close per-window (data-only, no advection)
#   B) Flat-IC advection — uniform tracer, check if Σrm changes
#   C) CFL alpha check — compare fine-grid vs reduced-grid CFL
#
# Usage:
#   julia --project=. scripts/diagnostics/diagnose_mass_budget.jl [path_to_nc]
# ============================================================================

using Printf, Statistics, NCDatasets

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const NC_PATH = length(ARGS) >= 1 ? expanduser(ARGS[1]) :
    expanduser("~/data/AtmosTransport/met/era5/preprocessed_spectral_catrine/massflux_era5_spectral_202112_float32.nc")
const N_WINDOWS = 20   # how many windows to check (set 0 for all)
const FLAT_C    = 400.0 # flat IC mixing ratio [ppm]

# ---------------------------------------------------------------------------
# Load module for advection
# ---------------------------------------------------------------------------
println("Loading AtmosTransport...")
t0_load = time()
using AtmosTransport
using AtmosTransport.Architectures: CPU
using AtmosTransport.Grids: LatitudeLongitudeGrid, compute_reduced_grid
using AtmosTransport.Advection: allocate_massflux_workspace, strang_split_massflux_ppm!,
    max_cfl_massflux_x, max_cfl_massflux_x_fine
println(@sprintf("  loaded in %.1f s", time() - t0_load))

# ---------------------------------------------------------------------------
# Helper: build grid + workspace on CPU
# ---------------------------------------------------------------------------
function build_grid_and_workspace(Nx, Ny, Nz, FT)
    met_cfg = AtmosTransport.IO.default_met_config("era5")
    vc = AtmosTransport.IO.build_vertical_coordinate(met_cfg; FT, level_range=1:Nz)
    grid = LatitudeLongitudeGrid(CPU(); FT, size=(Nx, Ny, Nz), vertical=vc)

    # Dummy arrays to size the workspace
    m_dummy  = zeros(FT, Nx, Ny, Nz)
    am_dummy = zeros(FT, Nx+1, Ny, Nz)
    bm_dummy = zeros(FT, Nx, Ny+1, Nz)
    cm_dummy = zeros(FT, Nx, Ny, Nz+1)

    # Cluster sizes from reduced grid
    cs_cpu = if grid.reduced_grid !== nothing
        Int32.(grid.reduced_grid.cluster_sizes)
    else
        ones(Int32, Ny)
    end

    ws = allocate_massflux_workspace(m_dummy, am_dummy, bm_dummy, cm_dummy;
                                      cluster_sizes_cpu=cs_cpu)
    return grid, ws, cs_cpu
end

# ---------------------------------------------------------------------------
# Helper: per-latitude-band mass summary
# ---------------------------------------------------------------------------
const LAT_BANDS = [(0, 30, "0-30"), (30, 60, "30-60"), (60, 80, "60-80"), (80, 91, "80-90")]

function lat_band_mass(arr, lats)
    Ny = length(lats)
    results = Dict{String, Float64}()
    for (lo, hi, name) in LAT_BANDS
        mask_S = findall(j -> -hi < lats[j] <= -lo, 1:Ny)
        mask_N = findall(j ->  lo <= lats[j] <  hi, 1:Ny)
        sS = isempty(mask_S) ? 0.0 : sum(Float64, @view arr[:, mask_S, :])
        sN = isempty(mask_N) ? 0.0 : sum(Float64, @view arr[:, mask_N, :])
        results["S$name"] = sS
        results["N$name"] = sN
    end
    return results
end

# ===================================================================
# MODE A: Flux Telescoping Check (no advection, data only)
# ===================================================================
function mode_a_telescoping(ds, n_win)
    println("\n" * "="^72)
    println("MODE A: Flux telescoping check (data only)")
    println("="^72)
    println("Checks if Σ(flux divergence) = 0 per window")
    println()

    Nx, Ny, Nz = ds.dim["lon"], ds.dim["lat"], ds.dim["lev"]
    Nt = min(ds.dim["time"], n_win > 0 ? n_win : ds.dim["time"])

    # Accumulate to see if residuals grow
    cum_div_x = 0.0
    cum_div_y = 0.0
    cum_div_z = 0.0
    cum_m_change = 0.0
    m_prev = nothing

    @printf("%-4s  %13s  %13s  %13s  %11s  %13s  %13s\n",
            "Win", "Σdiv_x", "Σdiv_y", "Σdiv_z", "max|div|", "Δm(t→t+1)", "cum_Δm")

    for t in 1:Nt
        am_t = Float64.(ds["am"][:, :, :, t])
        bm_t = Float64.(ds["bm"][:, :, :, t])
        cm_t = Float64.(ds["cm"][:, :, :, t])
        m_t  = Float64.(ds["m"][:, :, :, t])

        # Global telescoping per direction
        # X (periodic): Σ_i (am[i] - am[i+1]) = am[1] - am[Nx+1] = 0 (periodic)
        sum_div_x = 0.0
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            sum_div_x += am_t[i, j, k] - am_t[i+1, j, k]
        end

        # Y (closed poles): Σ_j (bm[j] - bm[j+1])
        sum_div_y = 0.0
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            sum_div_y += bm_t[i, j, k] - bm_t[i, j+1, k]
        end

        # Z (closed top/bottom): Σ_k (cm[k] - cm[k+1])
        sum_div_z = 0.0
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            sum_div_z += cm_t[i, j, k] - cm_t[i, j, k+1]
        end

        # Max per-cell residual
        max_res = 0.0
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            div_total = (am_t[i,j,k] - am_t[i+1,j,k]) +
                        (bm_t[i,j,k] - bm_t[i,j+1,k]) +
                        (cm_t[i,j,k] - cm_t[i,j,k+1])
            max_res = max(max_res, abs(div_total))
        end

        # Mass change between windows
        dm = 0.0
        if m_prev !== nothing
            dm = sum(m_t) - sum(m_prev)
            cum_m_change += dm
        end
        m_prev = sum(m_t)

        cum_div_x += sum_div_x
        cum_div_y += sum_div_y
        cum_div_z += sum_div_z

        @printf("%3d  %+13.5e  %+13.5e  %+13.5e  %11.3e  %+13.5e  %+13.5e\n",
                t, sum_div_x, sum_div_y, sum_div_z, max_res, dm, cum_m_change)
    end

    println()
    @printf("Cumulative: Σdiv_x = %+.5e, Σdiv_y = %+.5e, Σdiv_z = %+.5e\n",
            cum_div_x, cum_div_y, cum_div_z)
    println("If cumulative values grow linearly, the flux data has a systematic imbalance.")
end

# ===================================================================
# MODE B: Flat-IC Advection Mass Tracking
# ===================================================================
function mode_b_flat_ic(ds, grid, ws, n_win, FT)
    println("\n" * "="^72)
    println("MODE B: Flat-IC advection (c = $FLAT_C everywhere)")
    println("  Float type: $FT")
    println("="^72)

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Nt = min(ds.dim["time"], n_win > 0 ? n_win : ds.dim["time"])
    lats = Float64.(ds["lat"][:])

    # Load first window's air mass
    m  = FT.(ds["m"][:, :, :, 1])
    c  = fill(FT(FLAT_C), Nx, Ny, Nz)
    tracers = (co2 = c,)

    initial_rm  = Float64(sum(Float64, m .* c))
    initial_m   = Float64(sum(Float64, m))
    initial_bands = lat_band_mass(m .* c, lats)

    @printf("Initial total rm = %.10e\n", initial_rm)
    @printf("Initial total m  = %.10e\n\n", initial_m)

    @printf("%-4s  %15s  %13s  %15s  %13s  %11s\n",
            "Win", "Σrm", "Δrm%", "Σm", "Δm%", "max|c-400|")

    for t in 1:Nt
        # Load met data for this window
        am = FT.(ds["am"][:, :, :, t])
        bm = FT.(ds["bm"][:, :, :, t])
        cm_data = FT.(ds["cm"][:, :, :, t])
        m  = FT.(ds["m"][:, :, :, t])

        # Compute rm from current c and new m (matching run loop)
        # First window: rm = m * FLAT_C, subsequent: rm = m * c (carried forward)

        # Run Strang split PPM advection (X→Y→Z→Z→Y→X)
        strang_split_massflux_ppm!(tracers, m, am, bm, cm_data, grid, Val(7), ws;
                                    cfl_limit=FT(0.95))

        # After advection, c has been updated in-place (tracers is a NamedTuple of views)
        total_rm = sum(Float64, m .* c)
        total_m  = sum(Float64, m)
        drm_pct = (total_rm - initial_rm) / initial_rm * 100
        dm_pct  = (total_m  - initial_m)  / initial_m  * 100
        max_dev = maximum(abs.(Float64.(c) .- FLAT_C))

        @printf("%3d  %+15.8e  %+13.7f  %+15.8e  %+13.7f  %11.3e\n",
                t, total_rm, drm_pct, total_m, dm_pct, max_dev)
    end

    # Per-latitude band summary after final window
    println("\nPer-latitude-band rm change (final vs initial):")
    final_rm_arr = m .* c
    final_bands = lat_band_mass(final_rm_arr, lats)
    for (_, _, name) in LAT_BANDS
        for prefix in ["S", "N"]
            key = "$prefix$name"
            if haskey(initial_bands, key) && haskey(final_bands, key)
                init_val = initial_bands[key]
                final_val = final_bands[key]
                dpct = init_val > 0 ? (final_val - init_val) / init_val * 100 : 0.0
                @printf("  %-6s: init=%.5e  final=%.5e  Δ=%+.5f%%\n",
                        key, init_val, final_val, dpct)
            end
        end
    end
end

# ===================================================================
# MODE C: CFL Alpha — Fine vs Reduced Grid
# ===================================================================
function mode_c_cfl_check(ds, grid, ws, cs_cpu, n_win, FT)
    println("\n" * "="^72)
    println("MODE C: CFL alpha — fine-grid vs reduced-grid comparison")
    println("="^72)

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Nt = min(ds.dim["time"], n_win > 0 ? n_win : ds.dim["time"])
    lats = Float64.(ds["lat"][:])

    # Print reduced grid info
    n_reduced = count(>(1), cs_cpu)
    if n_reduced > 0
        println("Reduced grid: $n_reduced of $Ny latitudes have cluster_size > 1")
        println("Max cluster size: $(maximum(cs_cpu))")
        println()
        println("Reduced latitudes:")
        for j in 1:Ny
            r = cs_cpu[j]
            if r > 1
                @printf("  j=%3d  lat=%+7.2f°  cluster_size=%d  Nx_red=%d\n",
                        j, lats[j], r, Nx ÷ r)
            end
        end
    else
        println("No reduced grid active (all cluster_sizes = 1)")
    end
    println()

    @printf("%-4s  %10s  %10s  %10s  %10s  %8s\n",
            "Win", "CFL_reduc", "CFL_fine", "ratio", "max_alpha", "n(α>1)")

    for t in 1:min(Nt, 5)  # Only check a few windows (expensive)
        am = FT.(ds["am"][:, :, :, t])
        m  = FT.(ds["m"][:, :, :, t])

        # 1. CFL from existing function (uses cluster_sizes)
        cfl_reduced = max_cfl_massflux_x(am, m, ws.cfl_x, ws.cluster_sizes)

        # 2. Fine-grid CFL: compute alpha at every face on fine grid
        max_alpha_fine = FT(0)
        n_alpha_gt1 = 0

        for k in 1:Nz, j in 1:Ny, i in 1:Nx+1
            am_val = am[i, j, k]
            if am_val >= zero(FT)
                il = i == 1 ? Nx : i - 1
                md = m[il, j, k]
            else
                ir = i > Nx ? 1 : i
                md = m[ir, j, k]
            end
            alpha = md > zero(FT) ? abs(am_val) / md : zero(FT)
            if alpha > max_alpha_fine
                max_alpha_fine = alpha
            end
            if alpha > one(FT)
                n_alpha_gt1 += 1
            end
        end

        ratio = cfl_reduced > 0 ? Float64(max_alpha_fine) / Float64(cfl_reduced) : 0.0

        @printf("%3d  %10.4f  %10.4f  %10.2fx  %10.4f  %8d\n",
                t, cfl_reduced, max_alpha_fine, ratio, max_alpha_fine, n_alpha_gt1)
    end

    # Detailed per-latitude analysis for first window
    if Nt >= 1
        println("\nPer-latitude CFL detail (window 1):")
        am = FT.(ds["am"][:, :, :, 1])
        m  = FT.(ds["m"][:, :, :, 1])

        @printf("%-5s  %8s  %4s  %10s  %10s  %8s  %10s\n",
                "j", "lat", "r", "max_α_fine", "max_α_red", "n(α>1)", "Σ|am|/Σm")

        for j in 1:Ny
            r = Int(cs_cpu[j])
            max_af = FT(0)
            max_ar = FT(0)
            n_gt1 = 0
            sum_am = FT(0)
            sum_m  = FT(0)

            for k in 1:Nz
                for i in 1:Nx+1
                    am_val = am[i, j, k]
                    # Fine-grid alpha
                    if am_val >= zero(FT)
                        il = i == 1 ? Nx : i - 1
                        md = m[il, j, k]
                    else
                        ir = i > Nx ? 1 : i
                        md = m[ir, j, k]
                    end
                    af = md > zero(FT) ? abs(am_val) / md : zero(FT)
                    max_af = max(max_af, af)
                    if af > one(FT)
                        n_gt1 += 1
                    end
                end

                # Reduced-grid alpha (at cluster boundaries only)
                if r > 1
                    Nx_red = Nx ÷ r
                    for ic in 1:Nx_red
                        am_face = am[(ic - 1) * r + 1, j, k]
                        # Cluster sum for donor
                        if am_face >= zero(FT)
                            donor_ic = ic == 1 ? Nx_red : ic - 1
                        else
                            donor_ic = ic
                        end
                        m_donor = FT(0)
                        i_start = (donor_ic - 1) * r + 1
                        for off in 0:r-1
                            m_donor += m[i_start + off, j, k]
                        end
                        ar = m_donor > zero(FT) ? abs(am_face) / m_donor : zero(FT)
                        max_ar = max(max_ar, ar)
                    end
                else
                    max_ar = max_af
                end

                for i in 1:Nx
                    sum_am += abs(am[i, j, k])
                    sum_m  += m[i, j, k]
                end
            end

            # Only print rows with interesting data (reduced or large alpha)
            if r > 1 || max_af > FT(0.5)
                flux_ratio = sum_m > 0 ? Float64(sum_am) / Float64(sum_m) : 0.0
                @printf("j=%3d  %+7.2f  %4d  %10.4f  %10.4f  %8d  %10.4f\n",
                        j, lats[j], r, max_af, max_ar, n_gt1, flux_ratio)
            end
        end
    end
end

# ===================================================================
# MAIN
# ===================================================================
function main()
    println("Mass Budget Diagnostic for ERA5 LL PPM Advection")
    println("=" ^ 72)
    println("Data: $NC_PATH")

    ds = NCDataset(NC_PATH, "r")
    Nx, Ny, Nz = ds.dim["lon"], ds.dim["lat"], ds.dim["lev"]
    Nt = ds.dim["time"]
    println("Grid: $Nx × $Ny × $Nz, $Nt windows")

    n_win = N_WINDOWS > 0 ? min(N_WINDOWS, Nt) : Nt

    # --- Mode A: data-only telescoping check ---
    mode_a_telescoping(ds, n_win)

    # --- Build grid and workspace (Float32) ---
    println("\nBuilding grid and workspace...")
    FT = Float32
    grid, ws, cs_cpu = build_grid_and_workspace(Nx, Ny, Nz, FT)

    # --- Mode C: CFL check (before advection, for diagnosis) ---
    mode_c_cfl_check(ds, grid, ws, cs_cpu, n_win, FT)

    # --- Mode B: Flat-IC advection ---
    mode_b_flat_ic(ds, grid, ws, n_win, FT)

    # --- Mode B again in Float64 ---
    println("\n\n--- Repeating Mode B in Float64 ---")
    FT64 = Float64
    grid64, ws64, _ = build_grid_and_workspace(Nx, Ny, Nz, FT64)
    mode_b_flat_ic(ds, grid64, ws64, min(n_win, 5), FT64)

    close(ds)
    println("\nDone.")
end

main()
