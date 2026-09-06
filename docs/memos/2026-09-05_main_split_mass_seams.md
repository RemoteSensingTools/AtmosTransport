# Paired cubed-sphere exchange for split advection

Split PPM, Slopes, and Upwind previously reconstructed a physical panel contact
twice, sometimes at different stages: one panel's X edge can meet another's Y
edge. Mirrored air fluxes and halo exchange therefore did not make tracer
transfers cancel. The error persisted in Float64. The preceding
[investigation](2026-09-05_main_mass_seams.md) contains the operator ablations,
reference-code comparison, and standalone grouped-sweep prototype.

## Conservative directional groups

Each physical contact has one owner, the lower-numbered panel. Its local edge
axis assigns that contact to the X or Y group. A group contains the corresponding
panel-interior faces and its owned contacts. Before updating any panel, the
runtime reconstructs each owned seam transfer once from the current state.
The existing local kernels receive a read-only flux view that masks boundary
faces. After their interior updates, the cached transfer is subtracted from
one physical cell and added to its neighbor, including rotated X/Y contacts.
Air mass uses the same paired update. The full X/Y/Z/Z/Y/X ordering remains.

This is local conservation by construction, with no global normalization,
positivity clipping, or tracer-total correction. Input air fluxes must still be
mirrored to match the binary's continuity budget. Panel-interior reconstruction
and vertical kernels retain their existing arithmetic. Scalar, packed copy-back,
and packed ping-pong entry points use the same grouping.

The forward cache contains `12 * Nc * Nz * (Nt + 1)` values, with the final slot
holding air transfer. C90 L66 with 32 Float32 tracers adds 9,408,960 bytes of
device workspace, allocated directly from the state prototype. The seam cache
does not impose the matrix convection kernel's shared-memory tracer batch size.
Lin–Rood disables this buffer because it has its own shared-face construction.
Contacts meeting at corners launch in stream order to avoid competing writes.

## Tracer adjoint and tape

For oriented seam transfer `T`, the two updates are `r_a -= T` and `r_b += T`.
The reconstruction's reverse seed is therefore `lambda_b - lambda_a`, with the
owner-edge normal sign. All seam seeds are cached before local panel reverses
overwrite output gradients. The reverse first handles masked interior faces,
then differentiates the shared owner reconstruction into its input stencil;
the usual halo transpose accumulates neighboring dependencies.

The tape's forward recorders and size estimator use the coupled production
sweep. The record format remains unchanged. These gradients treat meteorology
and air-mass evolution as prescribed. They do not establish derivatives with
respect to optimized meteorology. The adjoint adds `12 * Nc * Nz` seed values.

## Conservation and field accuracy

Six pressure-layer tracers, C90 L66, 24 hourly windows, TM5 convection, exact
Dkg diffusion, no emissions, and air resets preserving tracer mass:

| Precision | Maximum absolute final relative drift before | After |
|---|---:|---:|
| Float32 | 2.238896e-5 | 8.168655e-7 |
| Float64 | 2.280581e-5 | 9.914172e-16 |

The Float32 improvement is about 27-fold. All initial compensated totals match
the baseline exactly, and all 25 hourly totals pass the precision-specific
checks. Float32 uses collaborative TM5 LU; Float64 uses the legacy solve, with
Float32 archive arrays in both cases. The same experimental convection archive
as the earlier studies is used; this is a numerical test, not forcing validation.

The analytic field check rotates a Gaussian plus positive background around a
tilted axis on a unit sphere. Mass fluxes come from corner streamfunction
differences, with mirrored contacts. Initialization and comparison use the
analytic values at cell centers, not exact cell averages. Both implementations
use those same inputs. The baseline replays the committed `bf9ae4cb` scalar
palindrome with unchanged panel kernels; the candidate uses production grouping.

| Grid, convention | Full-rotation relative field L2 before | After |
|---|---:|---:|
| C8, gnomonic | 0.370361 | 0.368018 |
| C8, GEOS native | 0.378926 | 0.377745 |
| C16, gnomonic | 0.114197 | 0.113137 |
| C16, GEOS native | 0.113295 | 0.112213 |
| C32, gnomonic | 0.040720 | 0.040631 |
| C32, GEOS native | 0.040624 | 0.040308 |

Quarter-rotation errors also decrease in every case. Relative L2 uses cell-area
weights and the Gaussian component's norm. Candidate mass drift stays below
6.1e-15, versus up to 1.68e-3 before. This supports field accuracy on these
cases, not a universal convergence or positivity guarantee. The earlier fixed-grid
seam fixture has first-order timestep refinement in both implementations; a
palindrome alone does not make inaccurate subflows second order.

The corrected full-day PPM output still has negative column-mean undershoots,
down to about -4.27e-11 mol/mol versus positive maxima around 3.71e-9–5.12e-9.
Conserving totals does not solve that separate issue. Residual Float32 drift also
needs its own operator-level precision budget; diffusion arithmetic is unchanged.

## Runtime and storage cost

In the original 32-tracer, 255-substep V100 full-day workload, median whole-run
time is 34.574 s versus 32.598 s before (about 6.1% higher). Maximum final
relative drift falls from 2.565585e-5 to 8.203781e-7, about 31-fold. Cumulative
host allocation is 6.419 GB versus 6.397 GB; peak memory was not measured.
Both measured repetitions pass all 432 output checks. The earlier baseline and
candidate each use a warmup plus two samples, run sequentially; the result is
not a universal cost estimate. Raw measurements are in the experiment directory.

## Validation and review

- Complete 119-file CPU core collection plus regridding passes; optional
  diagnostic, real-data, and orphan tiers were excluded. Existing broken/skipped
  fixtures remain marked as such.
- Final focused CS tests pass 692 assertions, including stronger seam tests at
  appreciable Courant number in both precisions and panel conventions.
- Independent scalar-loop comparisons, fixed-meteorology finite-difference
  adjoints, and workspace guards pass 260 assertions. The reference uses the
  preprocessor contact map and established reconstruction helpers, independently
  implementing seam indexing and paired accumulation.
- CPU footprint tests pass 72 assertions; the V100-enabled footprint file passes
  all 91, including device, pinned-host, and mmap tape paths. V100 forward tests
  pass 2,552 assertions through 65 tracers, and CUDA seam adjoints match CPU in
  all 24 cases.
- Julia 1.10.12 passes the 256 reference and adjoint assertions run before the
  four cache guards were added. Julia 1.12.6 passes all 260.
- The strict manual build passes with deployment disabled. Documentation links
  and module README maps pass all 159 final checks.
- Aqua passes. JET 0.11.5 on Julia 1.12 reports 148 known patterns versus 144
  before: four reports arise from two new generic KA kernel launch sites.
  An isolated Julia 1.10 JET run reports zero, which is not comparable evidence
  for tightening its historical allowance of 130.

Codex diff review traced physical-edge ownership, reversal and normal signs,
corner write ordering, cache-before-mutation ordering, CPU/CUDA adaptation,
copy-back/ping-pong parity, and the tape/adjoint transpose. It identified and
corrected stale temporal-order and CFL-pilot documentation. The old local-panel
test reference now compares coupled copy-back against coupled ping-pong;
independent scalar loops and analytic rotation provide additional references.
No test tolerance was loosened. No unresolved implementation defect was found
within the tested valid-CFL, fixed-meteorology contract. The accuracy limitations
above remain open scientific work.

Reproduction scripts, hourly totals, field results, V100 runtime measurements,
and validation logs are in the
[experiment directory](../../scripts/benchmarks/results/main_split_mass_seams_v100_20260905/README.md).
