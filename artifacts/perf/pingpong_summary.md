# Ping-Pong Refactor — Performance Summary

Refactor: `restructure/dry-flux-interface` branch, commits
`8b8d283` (baseline) → `1a7e2ba` (post-refactor, after commit 4/7).

## Measurement methodology

- Benchmark: [scripts/benchmarks/bench_strang_sweep.jl](../scripts/benchmarks/bench_strang_sweep.jl)
- Config: `medium` (Nx=288, Ny=144, Nz=32, Nt=5, 20 timed steps, 3 warmup)
- Synthetic fluxes at CFL ≈ 0.3 → `n_sub = 1` for every direction, so every
  palindrome sweep is a single ping-pong kernel launch with zero copyto
- CPU: single-threaded (`JULIA_NUM_THREADS=1`) on the dev host
- Reports: median / mean / std of per-step wall time (20 steps)

## CPU results (this host, single-threaded Float64 & Float32)

| FT      | Scheme | Before (ms) | After (ms) | Δ (ms) | Speedup |
|---------|--------|-------------|------------|--------|---------|
| Float64 | Upwind | 458.686     | 423.002    |  -35.7 |  -7.8%  |
| Float64 | Slopes | 2564.061    | 2510.517   |  -53.5 |  -2.1%  |
| Float64 | PPM    | 2327.489    | 2272.198   |  -55.3 |  -2.4%  |
| Float32 | Upwind | 395.660     | 379.025    |  -16.6 |  -4.2%  |
| Float32 | Slopes | 2421.618    | 2400.191   |  -21.4 |  -0.9%  |
| Float32 | PPM    | 2175.612    | 2159.508   |  -16.1 |  -0.7%  |

Per-step savings: ~20–55 ms on CPU. This is consistent with eliminating
12 `copyto!` of ~10.6 MB per tracer per step (5 tracers × 2 copyto ×
6 sweeps × 10.6 MB / bandwidth ≈ 50 ms at ~13 GB/s single-thread
memcpy).

## Interpretation

The ping-pong win on CPU is bounded by
`(per-step copyto bandwidth time) / (per-step total time)`. For
kernel-bound schemes (Slopes, PPM) the copyto fraction of total time is
small, so the per-step speedup is small. For the bandwidth-bound Upwind
scheme the fraction is larger and the speedup is too.

The GPU ratio is different: HBM bandwidth is much higher relative to
device compute, so the copyto fraction of total time is much larger —
the plan's 20%+ GPU prediction still stands. The dev host has no CUDA
GPU available; GPU numbers need to be captured separately on curry
(A100) for Float64 and wurst (L40S) for Float32 by rerunning this
benchmark in the CUDA environment.

## Correctness

Verified at every commit:

- All 84 advection-kernel cross-backend/precision tests pass, unchanged
  ULP tolerances (F64 1-step ≤ 4 ULP, 4-step ≤ 16 ULP; F32 same)
- Mass conservation unchanged at F64 (`|Δsum|/sum₀ < 1e-12` for 4 steps)
- **Bit-identical MT ≡ per-tracer** test passes with `max_diff = 0.0`
  for both F64 and F32 — the strictest invariant of the refactor
- Pre-existing failures (`test_basis_explicit_core`: 2,
  `test_structured_mesh_metadata`: 3) unchanged — same baseline numbers
  before and after

## Workspace memory

At C180 × 30 tracers Float64 (production target):

| Component                           | Before    | After     | Delta      |
|-------------------------------------|-----------|-----------|------------|
| 3D rm / m ping-pong (A only → A+B)  | 224 MB    | 448 MB    | +224 MB    |
| 4D tracer buffer (A only → A+B)     | 3.4 GB    | 6.8 GB    | +3.4 GB    |
| CFL scratch (new, 2 × m-sized)      | 0         | 224 MB    | +224 MB    |
| m_save, cluster_sizes, faces        | ~112 MB   | ~112 MB   |  0         |
| **Total workspace**                 | ~3.7 GB   | ~7.6 GB   | **+3.9 GB**|

~2× of baseline, matching the plan's memory budget expectations.
