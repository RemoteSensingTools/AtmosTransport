# September 5 autonomous revamp status

This continues the recovered [Astra review](2026-09-04_astra_codebase_review.md)
and its [initial implementation](2026-09-04_review_followup.md). The user's later
request authorized tofu's V100 GPU 0 for this follow-up. New work did not use
local GPUs or tofu GPU 1.

## Original review coverage

| Review item | Delivered |
| --- | --- |
| Trustworthy configuration | Typed physics preflight, unknown-key rejection, topology/capability checks, and explicit rejection of unsupported partial-window runs. |
| Scientific readability | Current operator families/topologies; tracer-storage basis, equations, units, layouts, vertical order, and execution cadence; executable custom-loss example. |
| Runtime organization | Configuration, construction, output, progress, and resource ownership separated from topology loops; initial-condition and surface-flux loaders separated; shared geometry helpers moved out of preprocessing. |
| Scratch allocation | Typed diffusion ownership; reused Lin–Rood/RG mass backups and Poisson scratch; deferred legacy matrix scratch; CS numerical workspaces retained across input files. |
| Diagnostic/output traffic | Contiguous CPU column traversal, selected backend reductions/transfers, and streaming single-file NetCDF. Daily writes now drain on all run exits. |
| Input lifetime and I/O | Staging checks source identity, owns its directory, handles repeated paths, and preserves unrelated files. Runner cleanup drains window prefetch before closing readers. |
| Matrix convection scaling | Fixed six-slot RHS batches remove the previous six-tracer eligibility limit; matrix factorization is shared across all batches; updraft-only Hessenberg LU, a specialized RHS solve, and parallel final assembly reduce work. |
| Benchmarks | Actual binary-reader → runtime → NetCDF cases, all tracers/output times checked, plus reproducible CPU/A100/V100 kernel measurements. |
| Package development | Cached package loading in scripts/core tests, corrected module/test links, explicit reader/regridding imports, and quieter optional provenance probes. |

The chemistry extension, scientific inventory configurations, and other unrelated
edits already present in the workspace remain separate from these commits.

## New findings worth retaining

- A reused model could retain the previous driver's convection cache. The
  [handoff regression](2026-09-05_convection_driver_handoff.md) reproduced stale
  matrix rates and CMFMC using one subcycle instead of twelve. New first-window
  forcing now invalidates those caches.
- Async output and GPU prefetch are resources owned by the run, not merely local
  tasks. [Output cleanup](2026-09-05_output_task_lifetime.md) and
  [input cleanup](2026-09-05_input_resource_lifetime.md) now cover setup, stepping,
  and write failures, preserving simultaneous errors.
- Equal file sizes do not establish forcing identity. The
  [staging fix](2026-09-05_input_staging_ownership.md) also prevents concurrent
  eviction. Its metadata checks require immutable source files; they are not
  content checksums.
- The V100's six shared-memory RHS slots are an internal batch capacity, not a
  total-tracer limit. [Parallel matrix assembly](2026-09-05_matrix_convection_parallel_assembly.md)
  gives a further 2.33x improvement for the 85-level/six-tracer updraft benchmark
  and 1.44x for 65 tracers over the already optimized preceding kernel.
- A wider RHS batch helps large tracer counts but hurts small ones. The
  [tuning report](2026-09-05_matrix_convection_tuning.md) preserves both source
  patches and raw measurements; neither exploratory variant is enabled globally.
- [CS workspace reuse](2026-09-05_cs_workspace_reuse.md) reduces repeated setup
  while preserving numerical agreement across file handoffs. The larger V100
  pipeline completes 65 tracers; column-only output takes 2.99 s and writes
  52.2 MB versus 11.59 s and 1.09 GB for full layers in that synthetic case.
- [Window-range validation](2026-09-05_runner_window_bounds.md) rejects a CS
  start option that was previously ignored and multi-file ranges that would
  skip forcing between handoffs.

## Continued runtime follow-up

- [Run instrumentation lifetime](2026-09-05_run_instrumentation_lifetime.md)
  fixes timing/allocation/NVTX flags leaking after failed input inspection or
  setup. The 58-check input-resource suite passes on Julia 1.10 and 1.12.
- [Prefetch startup](2026-09-05_prefetch_startup.md) removes a duplicate
  first-window load while preserving independent GPU forcing buffers.
  The V100 startup suite passes 140 checks. In the three-file C48/L40 pipeline,
  cumulative host allocations fall by about 56.4 MB; column-only median time
  falls from 0.219 s to 0.183 s across five warm-cache samples.
- [Failed prefetch consumption](2026-09-05_prefetch_failure_consumption.md)
  prevents input cleanup from reporting an already observed task failure a
  second time. An interrupted window-boundary fetch retains its unfinished
  task for cleanup; eight failure checks and 140 startup checks pass on V100.
- [Runtime flow](../20_RUNTIME_FLOW.md) now follows current function signatures,
  prefetch ownership, forcing fields, and transport/window physics cadence.

## Validation and limits

A fresh isolated export completes all 112 core test files and the regridding
suite (113 files total, 81,817 passed checks summed from emitted test summaries).
Subsequent import/window-bound changes pass focused tests and final Aqua/JET
checks; Aqua passes all ten checks and JET reports 180 against the unchanged
181-report threshold. The continued runtime changes also pass a fresh clean export with
226 focused checks, including Aqua and JET (180/181). The subsequent failed-
prefetch cleanup change repeats the 58 resource checks and JET successfully;
every committed Julia source file matches that final clean export. Julia 1.10.12 and 1.12.6 cover the new resource, staging, cache-handoff,
and CS multifile behavior. Documentation builds execute the tutorial, doctests,
cross-references, and VitePress rendering with deployment explicitly disabled.

GPU evidence includes the 487-assertion matrix suite, independent bitwise matrix
assembly comparison, Compute Sanitizer racecheck with zero errors/warnings,
and the 75-assertion CS multifile integration suite. Whole-runtime Float32
comparisons use an explicit rounding tolerance; the matrix assembly's bitwise
claim applies only to its separately measured cases.

V100 measurements are synthetic and use warm file caches. They do not establish
cold-NAS throughput, campaign-scale throughput, or peak GPU memory. The original
review's A100 checks remain recorded in its report; the new matrix changes were
validated on V100, not rerun on A100. Metal hardware was not tested. Full tracer
fusion, portable adaptive RHS-batch selection, and deeper shared-memory matrix
representations remain performance experiments requiring further evidence.

The completed follow-up is committed on `fix/era-tm5-physics-equivalence`.
The linked reports and raw artifacts are the durable handoff; no deployment or
production campaign was started. All benchmark GPU processes have finished.
