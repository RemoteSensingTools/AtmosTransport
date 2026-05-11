# Adjoint status

This page is a candid statement of what's actually shipped on the
adjoint side **vs** what the top-level README claims. The short
version: **a production discrete adjoint is not yet implemented**.
There is a tested CS surface-flux prototype, but the README's
"Hand-coded discrete adjoint: TM5-4DVar-style adjoint with Revolve
checkpointing" remains a roadmap goal, not shipped production code.

This page exists so that anyone reading the docs gets the truth from
the docs and not from outdated marketing copy.

## What is shipped

The forward transport model — advection (4 schemes), convection
(CMFMC + TM5), vertical diffusion (Implicit / Backward Euler), surface
flux source — is fully shipped, GPU-portable, mass-conserving, and
covered by the test suite documented in [Conservation budgets](@ref).

There is now one deliberately limited adjoint-adjacent prototype:
`AtmosTransport.Adjoints.cs_surface_emission_footprint` runs a
kernelized reverse pass for CS split-sweep advection, with optional
midpoint implicit vertical diffusion and post-transport
`CMFMCConvection` or `TM5Convection`,
accumulating source-receptor footprints for surface-emission controls.
The supported linearized advection schemes are `UpwindScheme()`,
`SlopesScheme(NoLimiter())`, and `PPMScheme(NoLimiter())`. The
monotone-limited split-sweep `PPMScheme()` is also supported via a
stored tracer branch tape around the base trajectory.
`LinRoodPPMScheme` is supported at the **kernel** level (Plan 25
Commits 1–4) for ORD=5 only. The shipped reverse-mode counterparts in
`src/Operators/Advection/linrood_adjoint_kernels.jl` are:

1. `apply_linrood_update_adjoint!` (averaged-flux update)
2. `apply_pre_advect_x_adjoint!`
3. `apply_pre_advect_y_adjoint!`
4. `apply_ppm_x_face_from_q_adjoint!`
5. `apply_ppm_y_face_from_q_adjoint!`
6. `apply_ppm_x_face_adjoint!` (rm-input variant)
7. `apply_ppm_y_face_adjoint!` (rm-input variant)

…with transposition tests in `test/test_linrood_kernel_adjoints.jl`. A
single-panel zero-halo composition
`apply_linrood_horizontal_adjoint_single_panel!` is verified by a
finite-difference JVP/VJP check. Cross-panel halo adjoint integration
into `cs_surface_emission_footprint` (Plan 25 Commits 5–6) is the
remaining gap; until that lands,
`cs_surface_emission_footprint(scheme=LinRoodPPMScheme(),…)` errors
because `LinRoodPPMScheme` is not in `CSAdjointSupportedScheme`. The
plan and progress are tracked in
`docs/plans/25_LINROOD_ADJOINT/NOTES.md`. The companion
`cs_surface_emission_footprint_from_seed` accepts an explicit final
`dJ/drm` seed so arbitrary observation operators can reuse the same
CPU/GPU kernels. `cs_surface_flux_jacobian` batches layer/column
objectives and aggregates per-step footprints into named user-defined
time windows. `cs_surface_flux_4dvar` adds the prototype cost/gradient
layer: step-indexed scalar observations, named surface-flux controls,
and optional diagonal background terms. `cs_surface_flux_4dvar_optimize`
wraps that evaluator in a small dependency-free gradient-descent driver
with kernelized control updates, so the path can run a real prototype
inversion without an external optimizer package. The tests check the
resulting vector-Jacobian products and 4D-Var gradients against
directional finite-difference probes, including the Backward-Euler
diffusion transpose and the CMFMC/TM5 convection transposes. This is
useful for inverse-system design and plotting, but it is **not** the
full production adjoint suite.

The forward operators are written so a future adjoint pass can
transpose them mechanically. Three concrete examples:

- **Vertical diffusion** — the Thomas-tridiagonal coefficients
  `(a, b, c)` are kept as **named locals at every level `k`** rather
  than fused into a pre-factored `(b, factor)` form. The Diffusion
  module docstring (`src/Operators/Diffusion/Diffusion.jl:20-22`)
  records this as a deliberate adjoint-readiness choice. The CS
  footprint prototype now uses that layout to transpose the
  Backward-Euler column solve, including the tracer-mass/VMR scaling.
- **Convection (CMFMC + TM5)** — the apply!() contract takes a
  `ConvectionForcing` carrier explicitly so the operator does not
  call `current_time` internally; this keeps the operator pure-
  functional in the time variable, which simplifies the eventual
  adjoint integration. The CS footprint prototype includes
  `CMFMCConvection` by transposing the well-mixed sub-cloud,
  updraft, and tendency passes, and includes `TM5Convection` by
  rebuilding the same per-column matrix and solving with the
  transposed LU factors.
- **Advection** — the Strang palindrome's time symmetry means the
  forward integrator is its own time-reverse; the adjoint of the
  composition is the composition of the adjoints in reverse order,
  which is structurally the same code path with each operator's
  adjoint substituted in.

## What is NOT shipped

There is no production adjoint suite in `src/`. Specifically:

| Claimed in README | Actual status |
|---|---|
| "Hand-coded discrete adjoint: TM5-4DVar-style adjoint" | Not shipped as a full suite. `src/Adjoints` has a limited CS split-sweep surface-emission footprint reverse pass for the schemes listed above plus midpoint implicit vertical diffusion and post-transport CMFMC/TM5 convection. Plan 25 (in progress) ships the kernel-level LinRood adjoints and a single-panel composition; no CS-footprint-integrated / cross-panel LinRood adjoint driver yet (Plan 25 Commits 5-6 remaining). No general production `adjoint_advect!`, `adjoint_diffuse!`, `adjoint_convect!`, or checkpointed driver exists. |
| "with Revolve checkpointing for bounded memory" | No Revolve integration. No checkpoint scheduler. |
| Adjoint test suite | No production forward+backward suite and no Revolve scheduler tests. However: `test/test_diffusion_kernels.jl:181-232` and `test/test_cmfmc_convection.jl:213-263` contain **adjoint-identity / transposition checks** for the specific kernels they cover, asserting `⟨A x, y⟩ = ⟨x, Aᵀ y⟩` to floating-point. `test/test_cs_ppm_adjoint_footprint.jl` adds kernelized CS split-sweep + limited-PPM branch-tape + diffusion + CMFMC/TM5-convection surface-emission footprint/VJP checks against directional finite differences, time-window Jacobian aggregation tests, and prototype 4D-Var cost-gradient checks. These verify adjoint contracts and prototype source-receptor generation, not a full production discrete adjoint. |
| 4DVar driver | Prototype `cs_surface_flux_4dvar` and `cs_surface_flux_4dvar_optimize` exist for CS surface-flux controls and scalar layer/column observations. No checkpointed production driver yet. |

Legacy adjoint **templates** live under
`docs/resources/developer_notes/legacy_adjoint_templates/` —
specifically `Adjoint.jl`, `boundary_layer_diffusion_adjoint.jl`,
`checkpointing.jl`, `cost_functions.jl`, `gradient_test.jl`. **These
files are not compiled into the package.** They are reference
material from earlier prototypes; rolling them forward into the
current architecture is the work that is not yet done.

## The roadmap

Adjoint development is staged. The current status is:

| Stage | Done? |
|---|---|
| Forward operator design that does not preclude an adjoint | yes (advection / diffusion / convection apply!() contract; Thomas solver coefficient layout; ConvectionForcing time-pure dispatch) |
| Per-operator hand-coded adjoint kernels | partial — CS split-sweep advection, implicit vertical diffusion, and CMFMC/TM5 convection in the footprint path |
| Adjoint test suite (gradient checks via finite-difference probe) | partial — CS PPM emission-footprint prototype, time-window Jacobian aggregation, plus existing kernel transposition checks |
| Revolve-style checkpoint scheduler | not yet |
| 4DVar driver | partial — prototype CS surface-flux cost/gradient evaluator plus dependency-free descent wrapper, no checkpointed production driver |
| Cross-validation against TM5-4DVar | not yet |

In-source comments tag the adjoint as "future" / "plan 19" in
several locations:

- `src/Operators/Diffusion/Diffusion.jl:20-22` — coefficient layout
  rationale.
- `src/Operators/Diffusion/diffusion_kernels.jl:27-32` — references
  the legacy template under `docs/resources/`.
- `src/Operators/Diffusion/thomas_solve.jl:28-31` — explicit "future
  adjoint kernel calls this same `solve_tridiagonal!` after
  coefficient transposition."
- `src/Operators/Convection/CMFMCConvection.jl` and
  `TM5Convection.jl` — both reference plan-19 adjoint as the future
  consumer of the current forward design.

## What this means for users

If your work needs gradients of model output with respect to model
input — surface fluxes, initial conditions, parameter values — you
have three options today:

1. **Use the prototype CS surface-flux path.**
   `cs_surface_flux_4dvar` provides tested gradients for named
   surface-flux windows, step-indexed layer/column observations,
   optional implicit vertical diffusion, and CMFMC/TM5 convection.
   `cs_surface_flux_4dvar_optimize` adds a simple line-searched
   descent loop around those gradients.
2. **Use external automatic differentiation.** Some users have
   reported success wrapping the forward `step!` call with
   `Enzyme.jl` or `ReverseDiff.jl` in source-to-source AD mode for
   small problems. The runtime is not designed for AD efficiency;
   memory will be the limiting factor at production resolutions.
3. **Use TM5-4DVar for production inversions until the checkpointed
   driver lands.** The TM5 four-field convection
   (`entu/detu/entd/detd`) parity work means the forward physics in
   AtmosTransport closely matches TM5; running TM5-4DVar on the same
   data, then forward-only AtmosTransport for analysis, is a
   workable workaround.

## Where to read next

- [Validation status](@ref) — what the forward model HAS been
  validated against.
- [Conservation budgets](@ref) — the explicit verification tests
  the forward operators pass.
- `docs/resources/developer_notes/TM5_ADJOINT_CONTROLS.md` — what
  TM5-4DVAR optimizes with its adjoint and how that maps to the first
  AtmosTransport controls.
- *Phase 7: Configuration & Runtime* — the run-side TOML schema.

!!! note "Why this page exists"
    The README's adjoint claim was caught during the codex review of
    this documentation overhaul. Rather than soften the README and
    leave a future reader to wonder, this page states the truth
    directly. The README will be updated in Phase 9 of the docs
    overhaul to point at this page.
