# Cubed-sphere mass drift: ablation and Lin–Rood seam fix

The full-day tracer drift persists in Float64. It is dominated by inconsistent
horizontal tracer exchange at panel seams, not Float32 roundoff. Lin–Rood now
shares one final face estimate between both panels before applying divergence.
The split PPM seam defect remains open.

## Conservation result

Six tracers, C90 L66, 24 hourly windows, Lin–Rood ORD=7, TM5 convection,
exact Dkg diffusion, no emissions, `preserve_tracer_mass` air-mass resets:

| Precision | Maximum absolute final relative drift, before | After |
|---|---:|---:|
| Float32 | 3.772281236544833e-5 | 6.980537426159092e-7 |
| Float64 | 3.800514634197679e-5 | 7.931337742325357e-16 |

Float32 drift is about 54 times smaller. The Float64 result reaches accumulation
roundoff. No total normalization is applied. All initial totals match exactly;
`before/` and `after/` contain 25 compensated totals for every tracer.
`check_totals.jl` checks all hourly totals and writes `drift_summary.csv`.
All 153 assertions pass. `check_outputs.jl` checks finite fields and all 25
snapshots in the remote NetCDF files; all 124 assertions pass.

Conservation is not a complete scientific validation. Positive pressure-layer
initial conditions develop negative column-mean undershoots in this workload
both before and after the fix (minimum roughly -2.09e-10 mol/mol versus maxima
around 3.3e-9–4.6e-9). The extrema are recorded in `output_checks.txt`; this
change does not solve the pre-existing undershoot problem.

## Scope and provenance

- Host: tofu; Tesla V100, authorized GPU 0 only,
  UUID `GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb`.
- Julia 1.12.6, CUDA.jl 5.11.3, CUDA runtime 12.6; four Julia threads,
  one OpenBLAS thread, `CUDA.allowscalar(false)`.
- Before: exported `b7e850a3` transport sources, also unchanged in `8b71935b`.
  After: the Lin–Rood shared-seam change on `revamp/current-main`.
- Input: `era5_n320_transport_20181201_float32.bin` under
  `era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour`;
  3,114,263,552 bytes, dry format 4, explicit dm, 24 hourly windows.
- Physical probe: Float32 air panels `(96,96,66)`, level-33 first interior
  mass 2.829608e12 kg; surface pressure 100869.164 Pa. Dkg panels
  `(90,90,66)`, maximum 2.0023729e11 kg/s.
- Six initial layers span surface-pressure fractions 0.2–0.9, each targeting
  1e35 molecules. Float32 uses collaborative TM5 LU; Float64 uses the legacy
  solve. Archive arrays are Float32 in both runs.
- This archive has experimental 1-degree, three-hour convection held against
  hourly transport. These runs establish numerical conservation, not forcing
  fidelity or agreement with a reference atmospheric simulation.

`linrood_day.jl before` and `linrood_day.jl after` reproduce the day runs from
the corresponding exported project, writing `/tmp/atmos-lr-drift-<stage>/`.
Set `ATMOSTR_MASS_INPUT` to override the archive path. Set the authorized
`CUDA_VISIBLE_DEVICES` UUID explicitly before running any GPU script here.
The NetCDF files stay on tofu; lightweight totals and checks are archived here.

## Isolation experiments

`ppm_ablation.jl` disables diffusion, convection, or both in turn, then repeats
all operators in Float64. `ablation/` contains the hourly totals; see the CSV
for each tracer. The maximum final advection-only drift is about 1.51e-5 and
the all-operator Float64 drift about 2.28e-5. Thus upgrading precision alone
does not fix the PPM imbalance.

`baseline_seam_probe.jl` is the exact historical CPU probe: run it from an
`8b71935b` source checkout because its helper extraction uses that version's
test-file line range. With exactly mirrored mass fluxes and Float64 storage,
raising the seam Courant number to about 0.1 gives tracer drift 2.912e-6
(Upwind), -8.671e-6 (PPM), and 6.739e-6 (Lin–Rood). Air drift stays at roundoff.
The strengthened maintained tests avoid this probe's line-range dependency.

`diffusion_columns.jl` separately compares the current scalar Thomas recurrence
against an independent Float64 direct-mass `Tridiagonal` solve on sampled
archive columns. Maximum single-step drift is 1.232e-6 for the existing Float32
recurrence, 7.601e-7 for an exploratory diagonal rearrangement, and 2.060e-15
in Float64. `diffusion_columns.txt` records the summary; its full per-column
TOML is written to `/tmp`. No diffusion arithmetic is changed in this patch.

## Runtime cost

`profile.jl before|after` measures the same forcing for two model hours with
6 and 32 tracers, excluding sample 0 as warmup. It uses `ATMOSTR_TIMERS=1`.
These are warm-cache whole-run timings, including startup and output, with
unchanged tracer initialization and output selection. Two measured samples
per case are retained in `profile_before/` and `profile_after/`.

| Tracers | Before median | After median | Cumulative host allocation before → after |
|---|---:|---:|---:|
| 6 | 2.046 s | 2.076 s | 1.001 → 1.004 GB |
| 32 | 6.695 s | 7.044 s | 2.093 → 2.109 GB |

The 32-tracer cost increases about 5.2%; six-tracer timings are noisier.
This is a conservation fix, not a speedup claim. No persistent forward buffer
is added. The reverse pass holds all panels' face gradients together and
therefore uses more temporary memory than the old independent-panel reverse.

## Correctness gates

The core tests check both panel conventions and both precisions against the
independent preprocessing contact map, unchanged interior faces, idempotence,
and the projection's transpose identity. Stronger mass and q-space tests use
Courant numbers around 0.1 and both ORD=5/7. The full horizontal tracer adjoint
is checked against production-forward finite differences at fixed meteorology;
tape and production outputs agree exactly. The existing outer-face adjoint
freezes donor air mass, so this does not validate gradients for optimizing
meteorology. A joint air/tracer perturbation exposes that existing limitation.

The full 118-file core collection and regridding runner are covered by
`cpu_initial.txt` and `cpu_resume.txt`: the initial run stopped at README
freshness because the new source file was missing from the file map. That
entry was fixed and the failed file plus all remaining tests pass on resume
(`resume_cpu_suite.jl`). The final focused cubed-sphere file passes 660 checks,
including the 12 stronger tracer-adjoint assertions added during review.
`check_profile.jl` also passes 692 checks on the 6/32-tracer V100 outputs.

The opt-in V100 diagnostic passes 16 GPU contact-map checks plus 12 CPU checks.
The footprint diagnostic passes 74 assertions. JET reports 144 findings against
the unchanged 144-report allowance; `jet_seam_reports.txt` shows the two new
known GPU-kernel dispatch reports. The allowance has not been increased.

A fused-launch experiment (`fused_trial.jl`) passes 12 additional GPU checks
against the independent contact map, but takes 651 microseconds versus
91 microseconds for the separate-edge implementation on C90 L66 Float32.
The trial remains an archived experiment; it is not part of the runtime.

## Split-PPM follow-up experiment

`split_grouped_trial.jl` assigns each physical seam to its owning panel's axis
and applies its tracer transfer once, with equal and opposite updates to the
two neighboring cells. This CPU prototype conserves mass to roundoff and
preserves uniform mixing ratio. However, refinement toward a 512-substep
reference shows only first-order convergence on the seam fixture: relative
field errors decrease from 6.15e-5 to 3.11e-5 to 1.57e-5 with 1, 2, and 4
substeps. Grouping contacts alone is therefore insufficient to establish the
intended temporal accuracy. The prototype is not integrated into production;
the next implementation must account for transverse evolution at the coupled
interfaces and be checked against a suitable independent transport reference.

The follow-up `split_reference_trial.jl` checks the original PPM against its
own 512-substep reference as well. It also shows first-order refinement on
this fixture (errors 7.27e-5, 3.60e-5, 1.80e-5 at 1, 2, 4 substeps). The grouped
prototype has smaller errors against that same reference, and the two refined
solutions differ by 9.43e-8. Thus this experiment does not demonstrate an
accuracy regression from grouping, but it does expose an existing temporal
accuracy limitation. Broader forward, adjoint, and real-input validation is
still required before adopting the grouped exchange.

The strict documentation build passes in an isolated docs environment pointing
to this worktree, with `ATMOSTR_DOCS_BUILD_ONLY=true`; no site is deployed.
