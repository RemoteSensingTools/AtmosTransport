# Float64 collaborative matrix convection on V100

CUDA can now run the existing workgroup matrix solve in Float64 for unmerged
convection depths 1–73. Shared matrices, tracer batches, plume recurrences, and
timesteps retain Float64 precision. Each column is factored once and reused for
all tracers in batches of six. The default operator remains the legacy solve.

## Matched full-day timings

C90 L66, one complete ERA5 day, PPM, conservative Dkg, no emissions, and
`preserve_tracer_mass` window resets. Each pressure-layer tracer targets 1e35
molecules at pressure fractions 0.2–0.9. Both methods use the full 66 layers,
`lmax_conv=0`, and `n_merge=1`. Only `use_collab_lu` differs in these paired runs.
Sample 0 warms each method; samples 1 and 2 alternate their execution order.

| Float64 tracers | Legacy median | Collaborative median | Speedup |
|---|---:|---:|---:|
| 6 | 92.853 s | 21.759 s | 4.27× |
| 32 | 147.673 s | 63.943 s | 2.31× |

Cumulative host allocation is essentially unchanged: 10.782 → 10.781 GB for
six tracers and 11.719 → 11.719 GB for 32. These are allocation totals, not
peak RAM. The shared solver avoids the legacy global matrix scratch; its peak
device effect is measured separately. Two measured days per method on one
V100 do not establish a universal timing distribution.

In the first measured six-tracer day, the convection section falls from
74.891 to 4.709 s. Advection becomes the largest transport section. Prefetch
runs concurrently: its 13–17 s section total must not be added to the wall
time. Exposed fetch waits are about 6 ms in those samples. The CSV allocation
columns are unmeasured zeroes, and nested timers can include synchronization
waits. No cold-storage throughput claim follows from this cached workload.

## Scientific checks

- 748 full-day Float64 output/completeness comparisons pass at 6 and 32 tracers.
  Maximum relative L2 difference in saved column means is 6.64e-17.
- Snapshot fields use Float32 on disk, so sample 1 additionally writes each
  final packed panel at its native Float64 precision outside the timed region.
  All 242 full-state checks pass; maximum per-panel, per-tracer interior
  relative L2 difference is 1.73e-16 across all 66 layers.
- Maximum final relative drift in compensated Float64 snapshot totals is
  1.9828344355813393e-16. TOML `final_totals` instead use ordinary state reductions.
- The Float32 full-day regression passes 454 checks: all 150 output arrays
  across its two measured samples match the preceding initialization-reuse
  checkpoint exactly. Its arithmetic and launch selection are unchanged.
- The maintained V100 test passes 1,920 Float64 field, conservation, ownership,
  and batching checks against dense CPU LAPACK solves. It covers all three
  topologies, depths 1/8/66/73, positive and signed tracers, cloud-free columns,
  ordinary and sub-ulp downdrafts, and tracer counts through 65. The 73-level
  case uses a 91-level model to check unchanged upper layers. Existing
  Float32 batching checks (720), workspace checks (7), and CUDA gates (22) pass.
- The full CPU suite on the numerical port passes 120 core files and 628
  regridding assertions. Subsequent capability-API integration and prose cleanup
  pass focused source/reader, health, and executable-syntax checks.
  Aqua passes 10 checks; JET remains at 152 reports against the unchanged
  baseline. Julia 1.10.12 passes all 982 host-selection/shared-RHS checks.
  The strict documentation build and 160 local link/map checks pass.

No conservation rescaling or positivity clamp is added. These are numerical
and performance checks of the experimental three-hour-convection/hourly-
transport archive, not atmospheric forcing validation. The dense CPU solve
independently checks factorization/solution arithmetic; matrix construction
also retains the previously validated TM5 formulation.

## Seven-day extension

Seven consecutive December 2018 inputs contain 168 windows and 1,778
substeps. The collaborative Float64 run retains the same maximum daily drift
as the legacy reference: 1.9828344355813393e-16. All 83 cross-run checks pass;
maximum relative L2 difference across the saved weekly column means is
2.61e-16. The weekly field files are Float32; the full-precision day comparison
above supplements them. Daily samples do not bound within-day drift.

The complete process peaks at 6.366 GiB RSS, including compilation and mmap
pages. Device samples every 500 ms peak at 2,654 MiB, including CUDA context
and allocator pool, versus 2,976 MiB in the earlier legacy week. Brief device
peaks can be missed. Total wall time is 266.1 s including first compilation;
the earlier legacy week took 752.1 s, but those runs are not the matched warmed
performance comparison. The earlier week also predates private initializer
reuse. Use the day table for the isolated solver timing result.

`run_weekly.sh f64collab Float64 6 /tmp/atmos-f64-collab` wraps `weekly_run.jl`
with process and device monitoring. `check_weekly.jl` compares with the retained
legacy Float64 weekly files on tofu. Small results are archived under `week/`.

## Memory envelope and public behavior

For matrix depth L and a six-tracer buffer, declared Float64 shared storage is
`8*(L^2 + 9L + 2) + 4*(L + 2)` bytes. L=73 uses 48,204 bytes; L=74 exceeds
48 KiB. The CUDA extension opts into this bounded precision path. Float32
retains its portable 1–85-level envelope. Total tracer count never changes
these allocations.

Unsupported Float64 depths, other backends, and `n_merge>1` retain the prior
warn-and-fallback behavior. The Float32 invalid-GPU-envelope error remains.
An eligible existing Float64 request now engages instead of falling back,
so a positive `lmax_conv` applies its explicitly requested vertical truncation.
No automatic depth or precision change is made. CS adjoint footprints retain
the full-column, unmerged legacy requirement; this change does not expand that
adjoint API.

## Reproduction and review

Run `profile.jl Float64`, then `profile.jl Float32`, from the configured candidate
export `/tmp/atmos-f64-collab`, with `ATMOSTR_TIMERS=1`, four Julia threads,
one OpenBLAS thread, and the V100 selected explicitly:
`CUDA_VISIBLE_DEVICES=GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb`.
The environment is Julia 1.12.6, CUDA.jl 5.11.3/runtime 12.6, on tofu GPU 0.
GPU scalar indexing is disabled. `check_outputs.jl` runs with GPUs hidden;
its Float32 reference comes from `a2a5e818`'s initialization-reuse experiment.
The candidate derives from `982f8773`; both Float64 methods use the same source
export and differ only in the requested solver.

The actual input is dry format 4 with explicit endpoint mass changes. The
first panel is `(96,96,66)` Float64 after load, including three halo cells;
first-interior level-33 air mass is 2.829608026112e12 kg. Forcing panels are
`(90,90,66)` in kg/m²/s and first-cell area is 6.640061366844813e9 m².
The original archive remains Float32.

Critical Codex self-review checked precision at every shared allocation and
launch, unchanged Float32 operations, shared-memory bounds, preserved fallback
contracts, six-tracer tail indexing, the Float64 aggregation exclusion, and the
existing adjoint guard. Field comments now state current defaults and
approximation limits rather than old debugging chronology. The documentation
also distinguishes a backward-Euler matrix's unit column sums from a
nonnegative stochastic matrix. Two executable-syntax comparisons confirm that
these prose cleanups leave both operator implementations unchanged. The TM5 Fortran `TM5_Conv_Matrix` recurrence and
`dGeTrf`/`dGeTrs` solve were reread before the precision port. No numerical
formula or workspace ownership was changed, and no extra agent was used.
Large NetCDF and native-state files remain on tofu; configurations, timing
samples, comparison metrics, and check summaries are retained here.
