# Follow-up to the September 4 Astra review

This implements the recovered review in
[2026-09-04_astra_codebase_review.md](2026-09-04_astra_codebase_review.md).
The work includes the existing uncommitted runner split. The checkout also
contains unrelated scientific configuration, chemistry, and preprocessing work;
the initial validation did not commit, merge, deploy, or run a production simulation.

## Configuration and scientist-facing documentation

Preflight now runs the typed advection, diffusion, convection and chemistry
parsers, validates output settings and tracer table shape, and rejects unknown
root tables and options in the validated runtime/physics sections. The editor
schema applies the same unknown-key restriction to those sections. Misspellings such as `scheem` no longer
silently choose upwind. Lin–Rood order, diffusion coefficients, and convection
workspace/depth controls are checked before model allocation. Reduced-Gaussian
slopes and PPM requests fail at recipe construction, matching the existing
kernel limitations. README and operator tables reflect that support boundary.

An audit also found existing run configurations with unsupported `tm5_dkg`
diffusion, Lin–Rood ORD=3, or both legacy and modern advection selection.
These configurations require a valid scientific scheme choice or removal of
the ambiguous selection before execution; their existing settings were
preserved. Preflight now exposes these errors before allocation or binary reads.

The contributor guide now describes the current interfaces and links an
executable custom-loss operator in `examples/custom_loss.jl`. It explains the
scientific equation, units, layouts, cadence and the distinction between physical
species mass and dry-VMR × dry-air-mass storage. Documentation also distinguishes
CPU hosted CI from separately run CUDA regressions.

## Structure and allocation ownership

- The saved runner split separates configuration, progress, model setup and
  output from the topology-specific execution loops. Those loops retain their
  distinct scheduling behavior.
- Initial-condition loading, cubed-sphere mapping, surface-flux loading and
  surface-flux regridding live in separate files under `Models/initial_conditions/`.
  Shared panel packing helpers moved to `Regridding`, removing the Models import
  of `Preprocessing`.
- Diffusion has a typed column workspace, passed explicitly through the transport
  palindrome. Diffusion-only CS models allocate no advection workspace. Existing
  column scratch can be shared without allocating duplicate buffers, including
  during GPU adaptation.
- Lin–Rood and RG multi-tracer stepping reuse pre-step air-mass backups. The packed
  LL path does not allocate an unused backup. Poisson scratch pools persist across
  preprocessing windows, including the GEOS regularization path.
- The canonical runner, inversion scripts used by tests, and 91 core test files
  load the cached package instead of rebuilding a private copy of the source.
  Test globals remain isolated by the existing test runner.

## Output and measured performance

NetCDF capture accepts its output field specification. Only named tracers and
the union of requested layers are retained. Column sums are computed on the
backend before transfer; original level metadata and NetCDF values are preserved.
The low-level full snapshot API and full binary snapshot behavior remain
available. Selected levels use GPU-safe copies with scalar indexing disabled.
CPU and CUDA column sums accumulate in Float64; Metal dispatch selects Float32.

| Measurement | Result |
| --- | --- |
| CPU column diagnostic, C90/C180 × 72 levels, F32/F64 | 1.30–3.02× faster; sampled results exactly equal |
| Retained C180/L72/four-tracer F32 snapshot | 267 MiB full → 7.42 MiB column-only (about 36× smaller) |
| A100 Lin–Rood horizontal update, C12/C90 × 32 levels, ORD=5/7, F32/F64 | 1.03–1.066× faster; bitwise equal to the original |
| Small A100 CS pipeline, four tracers, two windows, three snapshots | 80.39 ms full → 70.46 ms column-only |

The Lin–Rood change uses existing per-panel output arrays to remove five host
barriers in the horizontal update. It retains per-tracer stepping; full tracer
fusion was a profiling candidate in the review and is not implemented here.

The old serialization I/O surrogate has been replaced with actual NetCDF
capture/write/read for all tracers. `benchmarking/run_pipeline_benchmarks.jl`
adds binary-reader → driven-runtime → writer measurements on LL, RG and CS.
[Raw results and comparison scripts](../../scripts/benchmarks/results/codebase_review_20260904/)
record dimensions, precision, tracer count, device and allocation measurements.

These timings use warm OS caches and synthetic forcing. They do not establish
cold-NAS throughput, campaign-scale throughput, peak GPU memory, or an end-to-end
speedup from the isolated kernel changes. Single-file output still retains its
captured frames until completion; daily output bounds retention to a day plus
one in-flight daily write. Metal hardware was not available for testing.

## Validation

- Core coverage: 104 files exercised in an isolated-module sweep (excluding
  the two separately run health gates). The sweep exposed two inversion-script
  package-loading failures, which were fixed, and an output test that observed
  stale code while this session was still editing. All three passed in fresh
  targeted reruns. This was a sweep plus targeted reruns, not one uninterrupted
  clean run of the final tree.
- Regridding: 628 assertions passed. Targeted Poisson, initial-condition,
  driven-builder, diffusion-only, runtime-contract and inversion checks passed.
- Selected output: 336 exact NetCDF value/schema checks and 21 invalid-config
  preflight checks passed on Julia 1.12.6 and 1.10.12. The last schema/CLI check
  also passed 51 assertions on Julia 1.12.6. Julia 1.10 used a separately resolved
  environment because the local ignored test manifest was generated by 1.12.
- Package health: Aqua passed all 10 checks; JET retained its existing baseline
  of 181 reports, without raising the allowed threshold.
- A100: 360 assertions passed with scalar indexing disabled, covering Float32
  and Float64, four-tracer CS advection (upwind, PPM, Lin–Rood ORD=5/7), diffusion
  workspace ownership and CPU/GPU parity, and selected capture on LL/RG/CS.
  CUDA runs used GPU 0 on `curry`; local CPU checks excluded CUDA devices.
- Benchmarks: all 12 binary-reader/runtime/NetCDF pipeline cases completed on
  CPU and A100. Kernel comparisons asserted bitwise equality; real NetCDF I/O
  cases exercised all tracers. The custom-loss example executed successfully.

- Documentation: the build completed successfully, including the executed
  Literate tutorial, doctests, cross-references and VitePress rendering.
  `ATMOSTR_DOCS_BUILD_ONLY=1` skipped deployment. The only build warning was
  Documenter's inability to detect a deployment environment for this local
  build. A fresh temporary docs environment avoided stale dependency pins in
  the local ignored docs manifest. The repository's existing permissive
  `warnonly=true` setting was retained.
- Final `git diff --check` passed. The review changes were subsequently prepared for commit separately from
  the pre-existing chemistry, inventory and coarsening changes.

Reproduce the canonical checks from the repository root (with instantiated
environments):

```bash
CUDA_VISIBLE_DEVICES='' julia --project=test test/runtests.jl
CUDA_VISIBLE_DEVICES='' julia --project=test test/core/test_selected_snapshots.jl
# On curry, after confirming device 0 is the A100:
CUDA_VISIBLE_DEVICES=0 julia --project=benchmarking test/a100/test_review_a100.jl
ATMOSTR_DOCS_BUILD_ONLY=1 CUDA_VISIBLE_DEVICES='' julia --project=docs docs/make.jl
```

## Publication check

The review-only Git index was exported to `/tmp/atmos-review-commit` and
validated independently of the uncommitted chemistry/inventory/coarsening work.
Selected-output/preflight, diffusion-only, CS builder, initial-condition and
CS chemistry tests passed. Aqua passed 10 checks; JET reported 179 against the
unchanged 181 baseline; schema/CLI validation passed 51 checks. The working-tree
results above include the pre-existing chemistry implementation, explaining the
slightly different JET report count.
