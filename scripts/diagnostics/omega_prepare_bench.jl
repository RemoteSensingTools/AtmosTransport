# ===========================================================================
# Micro-benchmark + bit-identity harness for the `:omega_consistent` window
# prepare (the per-level Poisson reconstruct that dominates GEOS-CS build time).
#
# Builds the real reader + workspace from a preprocessing config (single day),
# ingests window 1 once to warm the JIT + regridder cache, then times
# `_geos_prepare_window_for_steps!` for a fixed `steps` on a chosen window and
# dumps the reconstructed `cm` to a .jld2 so two runs (e.g. -t1 vs -t14) can be
# diffed for bit-identity.
#
# Usage:
#   ATMOS_OMEGA_TIMING=1 julia --project=. -t14 \
#     scripts/diagnostics/omega_prepare_bench.jl <config.toml> <date> <win> <steps> <out.jld2>
# ===========================================================================
using AtmosTransport
using AtmosTransport.Preprocessing
const P = AtmosTransport.Preprocessing
using Dates, Printf, JLD2, TOML

function build_day_context(cfg_path::String, date::Date)
    cfg = TOML.parsefile(cfg_path)
    src_cfg = cfg["source"]
    toml_relpath = String(src_cfg["toml"])
    toml_path = isabspath(toml_relpath) ? toml_relpath :
                joinpath(dirname(cfg_path), "..", "..", toml_relpath)
    FT = P._resolve_float_type(cfg)
    grid = P.build_target_geometry(cfg["grid"], FT)
    skw = (root_dir = P.expand_data_path(String(src_cfg["root_dir"])),)
    haskey(src_cfg, "include_surface") &&
        (skw = (skw..., include_surface = Bool(src_cfg["include_surface"])))
    haskey(src_cfg, "include_convection") &&
        (skw = (skw..., include_convection = Bool(src_cfg["include_convection"])))
    haskey(src_cfg, "include_vdiff_fields") &&
        (skw = (skw..., include_vdiff_fields = Bool(src_cfg["include_vdiff_fields"])))
    cfg_vertical = get(cfg, "vertical", Dict())
    haskey(cfg_vertical, "coefficients") &&
        (skw = (skw..., coefficients_file = P.expand_data_path(String(cfg_vertical["coefficients"]))))
    settings = P.load_met_settings(toml_path; skw...)
    vc = P.load_hybrid_coefficients(P.expand_data_path(settings.coefficients_file))
    vertical = P._build_native_vertical_setup(cfg_vertical, vc, FT)
    numerics = get(cfg, "numerics", Dict{String, Any}())
    mass_fix = get(cfg, "mass_fix", Dict{String, Any}())
    target_kg = P._native_mass_fix_target_kg(cfg, grid)
    return (cfg, FT, grid, settings, vertical, numerics, mass_fix, target_kg)
end

function main()
    cfg_path = abspath(ARGS[1]); date = Date(ARGS[2])
    win = parse(Int, ARGS[3]); steps = parse(Int, ARGS[4])
    out = ARGS[5]
    closure = length(ARGS) >= 6 ? Symbol(ARGS[6]) : :omega_consistent
    (cfg, FT, grid, settings, vertical, numerics, mass_fix, target_kg) =
        build_day_context(cfg_path, date)
    dt_met = Float64(get(numerics, "dt_met_seconds", 3600.0))
    reader = P.open_reader(settings, date, FT; seed = nothing,
                           chain_mass = false, next_day_handle = true)
    nw = P.windows_per_day(reader)
    ws = P.allocate_window_workspace(grid, settings, vertical, FT;
            dt_met_seconds = dt_met, chain_mass = false,
            adaptive_substeps = false, substep_cfl_target = 0.95,
            min_steps_per_window = 1, max_steps_per_window = 512,
            windows_per_day = nw,
            global_mass_pin = Bool(get(mass_fix, "enable", false)),
            global_mass_target_kg = target_kg,
            balance_mode = :column, cm_closure = closure, smooth_iters = 8)
    P._OMEGA_TIMING[] = true

    @printf("nthreads=%d  win=%d steps=%d\n", Threads.nthreads(), win, steps)
    # Ingest the target window: reads raw + endpoints, builds vdiv_om, and runs
    # its own adaptive prepare (which JITs everything + warms the regridder cache;
    # we discard its result). The timed pass below re-prepares at a FIXED `steps`
    # so -t1 vs -tN compare on identical inputs. To keep the JIT warm-up cheap we
    # cap the adaptive ceiling at `steps` via min/max bounds set at allocation.
    P.ingest_window!(ws, reader, win, grid, settings, vertical)
    P._reset_omega_timing!()

    # Timed fixed-steps prepare (the apples-to-apples reconstruct cost).
    t0 = time()
    P._geos_prepare_window_for_steps!(ws, grid, steps)
    elapsed = time() - t0
    s = P._OMEGA_TIMING_STATE
    @printf("PREPARE win=%d steps=%d  wall=%.3fs  recon=%.3fs  solves=%d cg_iters=%d\n",
            win, steps, elapsed, s.recon_time, s.solves, s.cg_iters)

    # Per-column continuity residual + cm[Nz+1] closure, exactly as the binary
    # mass-balance audit (omega_consistent_mass_balance.jl 5a) does it. dm here is
    # already the per-half-substep tendency (m_next - m_cur)/(2·steps) stored in
    # ws.dm_v4 by the prepare. Residual r = dm - div_h - (cm[k]-cm[k+1]).
    Nc = grid.Nc; Nz = size(ws.m_cur[1], 3)
    colmax = 0.0
    for p in 1:6, j in 1:Nc, i in 1:Nc
        c = 0.0
        @inbounds for k in 1:Nz; c += Float64(ws.m_cur[p][i, j, k]); end
        colmax = max(colmax, c)
    end
    ss = 0.0; n = 0; mx = 0.0; cmtop = 0.0
    for p in 1:6
        am = ws.am_v4[p]; bm = ws.bm_v4[p]; cm = ws.cm_v4[p]; dm = ws.dm_v4[p]
        for j in 2:Nc-1, i in 2:Nc-1
            @inbounds for k in 1:Nz
                div_h = (Float64(am[i, j, k]) - Float64(am[i+1, j, k])) +
                        (Float64(bm[i, j, k]) - Float64(bm[i, j+1, k]))
                vdiv = Float64(cm[i, j, k]) - Float64(cm[i, j, k+1])
                r = (Float64(dm[i, j, k]) - div_h - vdiv) / colmax
                ss += r*r; n += 1; mx = max(mx, abs(r))
            end
            cmtop = max(cmtop, abs(Float64(cm[i, j, Nz+1])) / colmax)
        end
    end
    @printf("CONTINUITY win=%d  RMS|res|/colmass=%.3e  max|res|/colmass=%.3e  |cm[Nz+1]|/colmass=%.3e\n",
            win, sqrt(ss/n), mx, cmtop)

    # Dump cm (and am/bm) for bit-identity diff.
    cm = [Array(ws.cm_v4[p]) for p in 1:6]
    am = [Array(ws.am_v4[p]) for p in 1:6]
    bm = [Array(ws.bm_v4[p]) for p in 1:6]
    jldsave(out; cm = cm, am = am, bm = bm,
            nthreads = Threads.nthreads(), elapsed = elapsed,
            recon = s.recon_time, solves = s.solves, cg_iters = s.cg_iters)
    @printf("wrote %s\n", out)
    P.close_reader!(reader)
end
main()
