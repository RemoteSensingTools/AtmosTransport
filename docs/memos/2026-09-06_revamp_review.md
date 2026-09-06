# Current revamp review — September 6, 2026

This review continues on `revamp/current-main`, based on main `a0698dde`.
The original `fix/era-tm5-physics-equivalence` worktree and its unrelated edits
remain untouched. The [integration record](2026-09-05_main_integration.md)
links the individual changes and measured workloads. The May review is
historical: its cited paths, defect claims, and performance rankings must not
be treated as current evidence.

## Scientific behavior

Paired physical seam transfers remove an advection mass imbalance that
persists in Float64. Independent solid-body rotation checks validate transported
fields alongside totals; the tracer adjoints include the same coupling.
The precomputed Dkg solve now factors the implicit equation in mass space,
reducing Float32 drift and independent column errors while preserving weak
recipient values. A lower-drift prototype was rejected because it erased
sub-ulp physical exchange. No global normalization or positivity clamp was added.

The one-day C90 L66 PPM result is 3.14e-7 maximum final relative mass drift in
Float32, compared with 2.24e-5 before these corrections. That remains above the
exploratory 1e-7 target. Exact final Float64 compensated totals on that workload
do not imply exact conservation for all stored states, precision choices, or
longer runs. A [seven-day comparison](../../scripts/benchmarks/results/main_weekly_mass_v100_20260906/README.md)
now reduces the worst final Float32 drift from 3.62e-6 to 5.64e-7, improves all
six final column-mean fields relative to Float64, and gives the same maximum
with 32 tracers. Float64 remains within 1.98e-16. Process RSS peaks at
5.82–6.30 GiB for these conservative runs, including compilation and mmap
pages; 500 ms device samples peak at 1,502–2,976 MiB.

Mass conservation, field accuracy, positivity, and adjoint fidelity are separate
criteria. Positive pressure layers can still develop small negative column
means under PPM/Lin–Rood. The split seam fixture does not establish second-order
time accuracy. Tracer adjoints at prescribed meteorology do not establish a
complete meteorological adjoint or full TM5/GCHP forcing parity.

## Performance and ownership

- CUDA Dkg tracer solves now read shared column factors in parallel. The
  32-tracer day falls from 38.635 to 29.543 s median with identical outputs.
  No new persistent workspace is added. CPU and Metal retain the fused loop.
- Float64 CUDA now supports the collaborative matrix solve on unmerged depths
  through 73. The C90 L66 day falls from 92.853 to 21.759 s with six tracers
  and from 147.673 to 63.943 s with 32. Full-precision state comparisons stay
  within 1.73e-16 relative L2; the week remains within 1.99e-16 total drift.
  Unsupported Float64 requests retain the legacy fallback and the CS adjoint
  API still requires the full-column, unmerged legacy variant.
- A [32-thread CUDA Float64 PPM launch](../../scripts/benchmarks/results/main_f64_ppm_tiles_v100_20260906/README.md)
  reduces the same six-/32-tracer days from fresh baselines of 21.590/63.061 s
  to 17.455/47.482 s. Complete native Float64 tracer storage remains byte-identical;
  the change only reduces launch padding and adds no workspace.
- The convection shared-memory buffer holds batches of six tracers, not a
  six-species model limit. The collaborative solver is tested through 65
  positive and signed tracers. Changing vertical truncation or aggregation
  would change the science and is separate from these optimizations.
- Cubed-sphere state is initialized into its final packed slots; device
  workspaces are constructed on the state backend. Snapshot output streams
  selected fields, and input resources own/drain prefetch before driver closure.
- Mmap avoids redundant file opens but does not eliminate host copies or GPU
  transfers. Whole-run timing, cumulative allocation, process RSS, and device
  pool usage measure different things. Current speedups are measured on tofu's
  V100, not universal CUDA/Metal predictions.

## Rechecking the old defect list

| May observation | Verified current status |
|---|---|
| Missing cubed-sphere `NoConvection.apply!` | Both state layouts have explicit no-op methods in `Convection/operators.jl`. |
| CMFMC fallback creates CPU arrays on GPU | `_cmfmc_dtrain_array` now uses `similar(cmfmc, ...)`, including panels. |
| Kz area caches use `Ref{Any}` | `WindowPBLKzField` and `LocalHoltslagBovilleKzField` use typed cache parameters and `_typed_area_cache_ref`. |
| CS regridding cannot share a run cache or promote through common machinery | `regrid_ll_binary_to_cs` accepts `run_cache` and calls `promote_streaming_binary!`. This check alone is not a claim that every preprocessing path is identical. |
| Pinned tape policy stores unconstrained `Any` fields | Its cache and synchronization hook now have explicit unions. |
| Tape operation lists use `Any[]` | Still present in footprint/Lin–Rood recording. Runtime impact needs profiling before choosing a replacement. |
| Reusing all recorded face buffers is an automatic speedup | Tape records retain distinct forward states. Any reuse/recomputation proposal must preserve those lifetimes and pass adjoint checks. |
| Source-before-diffusion is equivalent only for small diffusivity | Incorrect for fixed linear backward Euler: both forms add the source to the same right-hand side. The manual now states that algebra and the separate full-model parity limitation. |

## Readability and maintenance

The maintained manual under `docs/src/` describes current behavior; dated
experiments retain their original results and rejected alternatives. Module
READMEs identify ownership and file maps. New numerical helpers keep units,
indices, boundaries, and fixed-meteorology assumptions beside their equations.
The public initial-condition builder owns fresh arrays; private runner reuse
must never expose shared tracer storage. Test checks cover this ownership and
signed-field contract in addition to conserved totals.

The field-layout and storage-unit docstrings for `CellState` and
`CubedSphereState` now attach to the public types. Previously those sections
were attached to private validation helpers, so public Julia help omitted them.
Before/after documentation-binding probes confirm the repair. Critical Codex
self-review and two executable-syntax comparisons verify unchanged constructors,
validation, and storage layout; 191 documentation checks and the strict manual
build pass.

Before adopting further changes, use actual profiling to choose the next cost,
keep independent scientific probes, and validate the changed path. Do not
replace evidence with a blanket `Any` purge or repository-wide formatting.
Outstanding work includes long-duration drift/memory measurements, transported
field positivity and seam time accuracy, and adjoint tape profiling. Hardware
outside the measured V100 needs its own validation.

The public `has_surface` and `has_vdiff_fields` queries now share one generic
between binary readers and preprocessing settings. Previously the top-level
exports could not dispatch on settings and produced competing-export warnings
on Julia 1.10. Source flags, reader payload checks, Aqua, JET, and the updated
API documentation validate the unified queries.
