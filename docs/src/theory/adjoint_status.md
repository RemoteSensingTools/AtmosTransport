# Adjoint status

This page is a candid statement of what is shipped on the adjoint
side. As of 2026-05-17 the picture is significantly different from
what earlier drafts of this page claimed: Plan 25 (LinRood adjoint)
and Plan 26 (TM5-style inversion scaffold) are largely shipped, and
the inversion stack is on CI. The roadmap section below tracks what
is and is not done by stage.

## What is shipped

### Forward operators (unchanged)

The forward transport model — advection (four schemes: `UpwindScheme`,
`SlopesScheme`, `PPMScheme`, `LinRoodPPMScheme`), convection (CMFMC +
TM5), implicit vertical diffusion (Beljaars–Viterbo / Holtslag–Boville
Kz fields), surface flux source — is fully shipped, GPU-portable,
mass-conserving, and covered by the test suite documented in
[Conservation budgets](@ref).

### Tape, checkpoint, and reverse pass

`src/Adjoints/` ships a full discrete-adjoint pipeline for the
cubed-sphere split-sweep path, with a separate kernel-level pipeline
for `LinRoodPPMScheme` integrated through `cs_surface_emission_footprint`.

The shipped pieces, all exported through `AtmosTransport.Adjoints`:

| Surface | What it does |
| --- | --- |
| `cs_surface_emission_footprint` | Forward → tape → reverse pass driver for CS, returning `dJ/dE_t` at every surface step |
| `cs_surface_emission_footprint_from_seed` | Same, but accepts an explicit final `dJ/drm` seed for arbitrary observation operators |
| `cs_surface_flux_jacobian` | Batches layer/column objectives, aggregates per-step footprints into named user windows |
| `cs_surface_flux_4dvar` | 4D-Var cost / gradient with named controls, step-indexed scalar observations, diagonal background |
| `cs_surface_flux_4dvar_optimize` | Dependency-free gradient-descent driver for the above |
| `cs_surface_flux_4dvar_solve` | Production driver with optimizer choice (`CSGradientDescent` / `CSLBFGS`) |

The adjoint supports the following advection schemes (full union in
`CSAdjointSupportedScheme`):

- `UpwindScheme()`
- `SlopesScheme(NoLimiter())`
- `PPMScheme(NoLimiter())`
- `PPMScheme(MonotoneLimiter())` — via a stored tracer-branch tape around the base trajectory
- `LinRoodPPMScheme(; ppm_order = 5)` — Plan 25 Commits 1–6 shipped
- `LinRoodPPMScheme(; ppm_order = 7)` — Plan 25 ORD=7 also shipped

The supporting kernel adjoints in
`src/Operators/Advection/linrood_adjoint_kernels.jl` are
transposition-tested (`test/test_linrood_kernel_adjoints.jl`) and
finite-difference VJP-tested via single-panel and cross-panel halo
compositions.

### Checkpoint schedules

`src/Tape/CheckpointSchedule.jl` ships three checkpoint policies,
selectable via the `checkpoint` kwarg of `cs_surface_emission_footprint`:

| Policy | Memory | Recompute |
| --- | --- | --- |
| `FullCheckpoint` | O(N) state snapshots | none — fastest reverse pass |
| `StrideCheckpoint(stride)` | O(N/stride) | replays each stride forward |
| `RevolveCheckpoint(budget)` | O(log N) | bisection-based; logarithmic-memory variant of Griewank–Walther |

`RevolveCheckpoint` is a bisection variant rather than the optimal
binomial Revolve, but it ships the logarithmic-memory contract and is
on the test suite (`test_cs_stride_checkpoint.jl`,
`test_cs_revolve_checkpoint.jl`).

### Tape storage backends

Tape records can live on the device, in pinned-host memory (default
for GPU runs), or on disk via mmap — selectable via the `tape_storage`
kwarg (`:device` / `:pinned_host` / `:mmap`). Storage backends are
defined in `src/Tape/TapeStorage.jl` and `src/Tape/MmapTapeStorage.jl`;
the mmap path is covered by `test_cs_tape_mmap_roundtrip.jl`.

### Inversion scaffold (Plan 26)

`src/Inversion/` ships:

| Module | Surface |
| --- | --- |
| `Covariance.jl` | `DiagonalCSCovariance`, `IsotropicGaussianCSCovariance`, `apply_B_half!`, `apply_B_half_adjoint!`, `apply_B_half_inverse!` |
| `Preconditioning.jl` | `apply_preconditioner!`, `LinearOptimType`, `LogNormalOptimType` |
| `Optimizer.jl` | `AbstractCSOptimizer`, `CSGradientDescent`, `CSLBFGS` (via multiple-dispatch surfaces) |
| `Jacobian.jl` | Footprint-Jacobian assembly for batched observation operators |
| `CostGradient.jl` | 4D-Var cost / gradient evaluator with preconditioning |
| `Observations.jl`, `ObservationBinding.jl`, `ObservationsIO.jl` | Typed observation surface + Plan-26-D3 schema |
| `DeparturesIO.jl` | Departures-file IO |

These are all on CI. Full inversion tests:
`test_cs_4dvar_preconditioned.jl`, `test_cs_lbfgs.jl`,
`test_cs_inversion_driver.jl`, `test_cs_inversion_truth_recovery.jl`,
`test_cs_iteration_log.jl`, `test_cs_observations_io.jl`,
`test_cs_observation_binding.jl`, `test_cs_departures_io.jl`,
`test_cs_covariance.jl`, `test_cs_preconditioning.jl`,
`test_cs_optimizer_dispatch.jl`.

## What is not yet shipped

| Item | Notes |
| --- | --- |
| **Optimal binomial Revolve** | `RevolveCheckpoint` ships as the bisection variant — logarithmic memory but not the Griewank–Walther optimal recompute count. Optimal binomial Revolve is the next refinement. |
| **CMFMC convection adjoint** | The four-field TM5 convection has a transposed column solve; CMFMC does not have an adjoint kernel yet, so `cs_surface_emission_footprint` with `CMFMCConvection` is forward-only. |
| **`copy_corners` reverse** | Cubed-sphere halo-corner exchange in the reverse pass is the remaining CS-side gap. |
| **TM5-4DVAR cross-validation** | Synthetic truth-recovery via `test_cs_inversion_truth_recovery.jl` is on CI; a side-by-side parity run against TM5-4DVAR on real data has not been published. |
| **Tangent-linear model** | Forward TL is not exposed as a separate driver. If you need it, use the reverse pass plus identity seeding. |
| **LinRood ORD=7 panel-boundary adjoint records** | ORD=7 kernel adjoints ship, but the panel-boundary correction is folded into the ORD=5 record type; a dedicated ORD=7 record family is a roadmap item. |

## Adjoint-readiness in the forward design

Three concrete forward-design choices that pay off in the adjoint:

- **Vertical diffusion** — the Thomas-tridiagonal coefficients `(a, b,
  c)` are kept as **named locals at every level `k`** rather than fused
  into a pre-factored `(b, factor)` form. The Diffusion module
  docstring records this as a deliberate adjoint-readiness choice. The
  CS reverse pass uses that layout to transpose the Backward-Euler
  column solve, including the tracer-mass / VMR scaling.
- **Convection (CMFMC + TM5)** — `apply!` takes a `ConvectionForcing`
  carrier explicitly so the operator does not call `current_time`
  internally; this keeps the operator pure-functional in the time
  variable. The TM5 reverse pass rebuilds the same per-column matrix
  and solves with the transposed LU factors.
- **Advection** — the Strang palindrome's time symmetry means the
  forward integrator is its own time-reverse; the adjoint of the
  composition is the composition of the adjoints in reverse order,
  which is structurally the same code path with each operator's
  adjoint substituted in.

## How to use the adjoint today

```julia
using AtmosTransport
using AtmosTransport.Adjoints

# Run the forward + tape + reverse pipeline for a single objective.
result = cs_surface_emission_footprint(
    driver, model;
    scheme       = LinRoodPPMScheme(; ppm_order = 5),
    objective    = SiteConcentrationObjective(:MLO; lev = 1),
    checkpoint   = RevolveCheckpoint(budget = 8),
    tape_storage = :pinned_host,
)
# result.footprints[t] :: NTuple{6, Matrix{FT}} — dJ/dE_t at step t.
```

For an end-to-end inversion (synthetic truth recovery), see
`test/test_cs_inversion_truth_recovery.jl` as the runnable template.

## Where to read next

- [Validation status](@ref) — what the forward model has been
  validated against.
- [Conservation budgets](@ref) — the explicit verification tests
  the forward operators pass.
- [Adjoints on top of the binary](../for_tm5_gchp_users/adjoints.md) —
  TM5-4DVAR / GIGC-adjoint user perspective on how the pipeline maps
  to those workflows.
- The Plan 25 LinRood adjoint progress ledger:
  `docs/plans/25_LINROOD_ADJOINT/`.
- The Plan 26 inversion scaffold progress ledger:
  `docs/plans/26_TM5_STYLE_INVERSION/`.
