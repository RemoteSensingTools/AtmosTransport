# 2-GPU Sync Optimization Memo for Claude

Date: 2026-03-26
Owner: Codex (GPT-5)
Repo state at start: `7ab86ce8d5257b96612b66dc1cfe08cfd7496688` (dirty)
Artifact root: `artifacts/perf/2gpu_sync/20260326T185423Z`
Primary benchmark gate: `config/runs/catrine_geosit_c180_gchp_v4_7d_fullphys_2gpu.toml`

## Executive Summary

- Implemented synchronization-focused transport changes with explicit no-sync APIs and wrappers.
- Verified correctness across 3 baseline + 3 post-change full 7-day 2-GPU runs.
- CO2/SF6 drift is bit-identical in all runs.
- Performance improvement is real but small: median `GPU s/win` improved from `4.71` to `4.67` (-0.85%), median total runtime improved from `946.4s` to `937.3s` (-0.96%).
- 10% target was not met; deeper redesign is still needed.

## What Was Implemented

### Commits

1. `32e650b` — `perf(2gpu): step-1 baseline lock + ledger`
2. `9b5845e` — `perf(2gpu): step-2 dispatch no-sync batch API`
3. `d0d5e0d` — `perf(2gpu): step-3 halo no-sync wrapper + barrier cleanup`
4. `550280d` — `perf(2gpu): step-5/6 copy nosync + non-PPM halo batching`
5. `ed7471a` — `perf(2gpu): step-4 ppm sweep nosync batching`

### Code Changes

- Added public `foreach_gpu_batch_nosync(f, panel_map)` and exported it.
- Added `fill_panel_halos_nosync!(data, grid)` and kept `fill_panel_halos!` as synchronized wrapper.
- Removed mid-pass multi-GPU barrier inside halo fill; now single explicit sync in wrapper.
- PPM X/Y sweeps switched to no-sync batch dispatch and no-sync interior copy path, with explicit sweep-end `sync_all_gpus(pm)`.
- Added `_copy_interior_nosync!` and kept `_copy_interior!` as compatibility wrapper.
- Applied no-sync halo batching to non-PPM cubed-sphere X/Y sweeps as adjacent hotspot pass.

## Baseline vs Post-change Results

## Baseline (3 runs)
- Run1: total `943.5s`, GPU `4.69 s/win`
- Run2: total `971.6s`, GPU `4.84 s/win`
- Run3: total `946.4s`, GPU `4.71 s/win`
- Median: total `946.4s`, GPU `4.71 s/win`

## Post-change (3 runs)
- Run1: total `932.9s`, GPU `4.65 s/win`
- Run2: total `937.3s`, GPU `4.67 s/win`
- Run3: total `939.9s`, GPU `4.68 s/win`
- Median: total `937.3s`, GPU `4.67 s/win`

## Median delta (post - baseline)
- Total: `-9.1s` (`-0.96%`)
- GPU s/win: `-0.04` (`-0.85%`)
- IO s/win: `-0.01`
- Out s/win: `0.00`

## Correctness

- CO2 drift: identical (`7.0591e-02%`)
- SF6 drift: identical (`3.8005e-02%`)
- No observed regression in final mass diagnostics.

## Worked vs Didn’t Work

## Worked
- No-sync API split is safe and usable.
- Halo no-sync wrapper + explicit sync model is safe.
- PPM no-sync subcycle dispatch + no-sync copy path is safe.
- End-to-end transport remains bit-identical under tested gate.

## Safe but negligible
- The full implemented set produced <1% median runtime improvement.
- Confirms synchronization overhead is only part of the bottleneck for this workload.

## Didn’t work / not implemented in this pass
- No stream/event graph redesign for cross-GPU dependencies.
- No kernel fusion in PPM advection/copy path.
- No deeper overlap of halo exchange with CFL/advection phases.

## Remaining Bottlenecks (most likely)

- PPM sweep structure still has high launch density and repeated global phase boundaries.
- Kernel launch/dispatch overhead remains significant in day-6/day-7 windows.
- Halo exchange still performs edge-by-edge kernels with substantial control overhead.

## Recommended Next Experiments (priority order)

1. Introduce stream/event choreography per GPU and remove final sweep-wide barriers where data dependencies allow.
2. Fuse advection + interior copy into a single kernel (or staged kernel chain without host barrier points).
3. Batch multiple halo fields in one traversal (`rm`, `m`, possibly others) to reduce edge-loop overhead.
4. Profile late-window slowdown specifically (day 6-8) with kernel-level timeline to isolate phase spikes.
5. Evaluate replacing repeated small kernel launches with wider fused kernels in PPM sweep loops.

## Reproduction Commands

- Baseline/post run command:
  - `julia --project=. scripts/run.jl config/runs/catrine_geosit_c180_gchp_v4_7d_fullphys_2gpu.toml`
- All logs and JSON summaries are in:
  - `artifacts/perf/2gpu_sync/20260326T185423Z`

## Tracking Files

- Change ledger: `docs/perf/2gpu_sync_change_ledger.md`
- Benchmark summaries:
  - `artifacts/perf/2gpu_sync/20260326T185423Z/baseline_summary.json`
  - `artifacts/perf/2gpu_sync/20260326T185423Z/post_summary.json`
  - `artifacts/perf/2gpu_sync/20260326T185423Z/comparison_summary.json`
