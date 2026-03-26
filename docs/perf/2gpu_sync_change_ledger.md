# 2-GPU Sync Optimization Change Ledger

Timestamp root: `20260326T185423Z`
Config gate: `config/runs/catrine_geosit_c180_gchp_v4_7d_fullphys_2gpu.toml`
Host commit at start: `7ab86ce8d5257b96612b66dc1cfe08cfd7496688`

| Step | Commit | Timestamp (UTC) | Hypothesis | Files touched | Sync/barrier before -> after | Correctness result | Perf result | Decision | Tag |
|---|---|---|---|---|---|---|---|---|---|
| 1 baseline lock | `32e650b` | 2026-03-26T18:54Z-20:46Z | Establish stable 3-run baseline before sync changes | `artifacts/perf/2gpu_sync/20260326T185423Z/*`, `docs/perf/2gpu_sync_change_ledger.md` | N/A | CO2/SF6 drift stable across all 3 runs | Median total=946.4s, GPU=4.71 s/win | Keep as baseline | worked |
| 2 dispatch API | `9b5845e` | 2026-03-26T20:47Z | Public no-sync batch API removes forced batch barriers in planned callers | `src/Architectures.jl`, `src/Architectures/multi_gpu.jl`, `ext/AtmosTransportCUDAExt.jl` | synced-only batch entrypoint -> explicit no-sync entrypoint | bit-identical (no drift change) | contributes to net median ΔGPU=-0.04 s/win | Keep | safe but negligible |
| 3 halo redesign | `d0d5e0d` | 2026-03-26T20:48Z | Split halo fill into no-sync and sync wrapper, remove mid-pass barrier | `src/Grids/halo_exchange.jl`, `src/Grids/Grids.jl`, `src/Advection/Advection.jl` | multi-GPU fill call had 2 syncs -> 1 sync wrapper + explicit no-sync batching | bit-identical (no drift change) | contributes to net median ΔGPU=-0.04 s/win | Keep | safe but negligible |
| 4 PPM sweep no-sync | `ed7471a` | 2026-03-26T20:49Z | Use no-sync dispatch in PPM subcycle path and one sweep-end sync | `src/Advection/cubed_sphere_mass_flux_ppm.jl` | many subcycle/batch barriers -> one sweep-end barrier | bit-identical (no drift change) | contributes to net median ΔGPU=-0.04 s/win | Keep | safe but negligible |
| 5 copy helper + 6 adjacent hotspot | `550280d` | 2026-03-26T20:49Z | No-sync interior copy + non-PPM halo batching reduce redundant waits in adjacent path | `src/Advection/cubed_sphere_mass_flux.jl` | copy helper sync optional; x/y sweep halo sync pair -> single explicit sync | bit-identical (no drift change) | contributes to net median ΔGPU=-0.04 s/win | Keep | safe but negligible |

## Baseline Metrics (Step 1)

- Run 1: total=943.5s, IO=0.13, GPU=4.69, Out=0.11, CO2 Δ=7.0591e-02%, SF6 Δ=3.8005e-02%
- Run 2: total=971.6s, IO=0.13, GPU=4.84, Out=0.11, CO2 Δ=7.0591e-02%, SF6 Δ=3.8005e-02%
- Run 3: total=946.4s, IO=0.13, GPU=4.71, Out=0.10, CO2 Δ=7.0591e-02%, SF6 Δ=3.8005e-02%
- Median: total=946.4s, IO=0.13, GPU=4.71, Out=0.11, CO2 Δ=7.0591e-02%, SF6 Δ=3.8005e-02%

## Post-change Metrics

- Run 1: total=932.9s, IO=0.12, GPU=4.65, Out=0.11, CO2 Δ=7.0591e-02%, SF6 Δ=3.8005e-02%
- Run 2: total=937.3s, IO=0.12, GPU=4.67, Out=0.11, CO2 Δ=7.0591e-02%, SF6 Δ=3.8005e-02%
- Run 3: total=939.9s, IO=0.12, GPU=4.68, Out=0.11, CO2 Δ=7.0591e-02%, SF6 Δ=3.8005e-02%
- Median: total=937.3s, IO=0.12, GPU=4.67, Out=0.11, CO2 Δ=7.0591e-02%, SF6 Δ=3.8005e-02%

## Median Delta (Post - Baseline)

- Total: -9.1s (-0.96%)
- GPU s/win: -0.04 (-0.85%)
- IO s/win: -0.01
- Out s/win: 0.00
- CO2 drift delta: 0.0
- SF6 drift delta: 0.0

## Acceptance Outcome

- Correctness gate: pass (bit-identical mass drift for CO2/SF6 across all baseline/post runs).
- Performance gate (>=10% GPU s/win reduction): not met.
- Status: changes are safe and retained, but measured gain is small; deeper redesign remains needed.
