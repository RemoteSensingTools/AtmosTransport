# 2-GPU Sync Optimization Change Ledger

Timestamp root: `20260326T185423Z`
Config gate: `config/runs/catrine_geosit_c180_gchp_v4_7d_fullphys_2gpu.toml`
Host commit at start: `7ab86ce8d5257b96612b66dc1cfe08cfd7496688`

| Step | Timestamp (UTC) | Hypothesis | Files touched | Sync/barrier before -> after | Correctness result | Perf result | Decision | Tag |
|---|---|---|---|---|---|---|---|---|
| 1 baseline lock | 2026-03-26T18:54Z-20:46Z | Establish stable 3-run baseline before sync changes | `artifacts/perf/2gpu_sync/20260326T185423Z/*` | N/A | CO2/SF6 drift stable across all 3 runs | Median total=946.4s, GPU=4.71 s/win | Keep as baseline | worked |
| 2 dispatch API | 2026-03-26T20:47Z | Public no-sync batch API removes forced batch barriers in planned callers | `src/Architectures/multi_gpu.jl`, `ext/AtmosTransportCUDAExt.jl`, `src/Advection/Advection.jl` | per-batch sync API only -> explicit sync/no-sync API split | Pending post-change run | Pending post-change run | Keep and validate | pending |
| 3 halo redesign | 2026-03-26T20:48Z | Split halo fill into no-sync and sync wrapper, remove mid-pass barrier | `src/Grids/halo_exchange.jl`, `src/Grids/Grids.jl`, `src/Advection/Advection.jl` | 2 global syncs/call (multi-GPU) -> 1 sync in wrapper; optional no-sync batching | Pending post-change run | Pending post-change run | Keep and validate | pending |
| 4 PPM sweep no-sync | 2026-03-26T20:49Z | Use no-sync dispatch in PPM subcycle path and one sweep-end sync | `src/Advection/cubed_sphere_mass_flux_ppm.jl` | many per-subcycle batch syncs -> one final sweep sync | Pending post-change run | Pending post-change run | Keep and validate | pending |
| 5 copy helper | 2026-03-26T20:49Z | No-sync interior copy removes redundant per-panel sync waits in PPM path | `src/Advection/cubed_sphere_mass_flux.jl`, `src/Advection/cubed_sphere_mass_flux_ppm.jl` | copy kernel sync per call -> optional no-sync + explicit barrier | Pending post-change run | Pending post-change run | Keep and validate | pending |
| 6 adjacent hotspot | 2026-03-26T20:49Z | Apply no-sync halo batching in non-PPM CS X/Y sweeps | `src/Advection/cubed_sphere_mass_flux.jl` | two halo sync calls/sweep -> one explicit sync after paired fills | Pending post-change run | Pending post-change run | Keep and validate | pending |

## Baseline Metrics (Step 1)

- Run 1: total=943.5s, IO=0.13, GPU=4.69, Out=0.11, CO2 Δ=7.0591e-02%, SF6 Δ=3.8005e-02%
- Run 2: total=971.6s, IO=0.13, GPU=4.84, Out=0.11, CO2 Δ=7.0591e-02%, SF6 Δ=3.8005e-02%
- Run 3: total=946.4s, IO=0.13, GPU=4.71, Out=0.10, CO2 Δ=7.0591e-02%, SF6 Δ=3.8005e-02%
- Median: total=946.4s, IO=0.13, GPU=4.71, Out=0.11, CO2 Δ=7.0591e-02%, SF6 Δ=3.8005e-02%
