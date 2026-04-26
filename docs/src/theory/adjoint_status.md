# Adjoint status

This page is a candid statement of what's actually shipped on the
adjoint side **vs** what the top-level README claims. The short
version: **the discrete adjoint is not yet implemented** — the
README's "Hand-coded discrete adjoint: TM5-4DVar-style adjoint with
Revolve checkpointing" is a roadmap goal, not shipped code.

This page exists so that anyone reading the docs gets the truth from
the docs and not from outdated marketing copy.

## What is shipped

The forward transport model — advection (4 schemes), convection
(CMFMC + TM5), vertical diffusion (Implicit / Backward Euler), surface
flux source — is fully shipped, GPU-portable, mass-conserving, and
covered by the test suite documented in [Conservation budgets](@ref).

The forward operators are written so a future adjoint pass can
transpose them mechanically. Three concrete examples:

- **Vertical diffusion** — the Thomas-tridiagonal coefficients
  `(a, b, c)` are kept as **named locals at every level `k`** rather
  than fused into a pre-factored `(b, factor)` form. The Diffusion
  module docstring (`src/Operators/Diffusion/Diffusion.jl:20-22`)
  records this as a deliberate adjoint-readiness choice: a future
  adjoint kernel can transpose the `(a, b, c)` local triple into the
  adjoint-coefficient triple `(c̄, b̄, ā)` without rewriting the
  forward kernel.
- **Convection (CMFMC + TM5)** — the apply!() contract takes a
  `ConvectionForcing` carrier explicitly so the operator does not
  call `current_time` internally; this keeps the operator pure-
  functional in the time variable, which simplifies the eventual
  adjoint integration.
- **Advection** — the Strang palindrome's time symmetry means the
  forward integrator is its own time-reverse; the adjoint of the
  composition is the composition of the adjoints in reverse order,
  which is structurally the same code path with each operator's
  adjoint substituted in. The infrastructure for that swap is the
  `apply!` contract; the adjoint kernels themselves are not written.

## What is NOT shipped

There is no production adjoint code in `src/`. Specifically:

| Claimed in README | Actual status |
|---|---|
| "Hand-coded discrete adjoint: TM5-4DVar-style adjoint" | Forward-only `src/`. No `adjoint_advect!`, `adjoint_diffuse!`, `adjoint_convect!`, `adjoint_surface_flux!` kernels exist. |
| "with Revolve checkpointing for bounded memory" | No Revolve integration. No checkpoint scheduler. |
| Adjoint test suite | No `test/test_*adjoint*.jl` driver files (no full forward+backward gradient checks, no Revolve scheduler tests). However: `test/test_diffusion_kernels.jl:181-232` and `test/test_cmfmc_convection.jl:213-263` DO contain **adjoint-identity / transposition checks** for the specific kernels they cover, asserting `⟨A x, y⟩ = ⟨x, Aᵀ y⟩` to floating-point. These verify that the adjoint contract the forward kernels were designed against actually holds — they are a partial step toward the full adjoint roadmap, not a full discrete adjoint themselves. |
| 4DVar driver | Not present. |

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
| Per-operator hand-coded adjoint kernels | not yet — single biggest piece of work |
| Adjoint test suite (gradient checks via finite-difference probe) | not yet |
| Revolve-style checkpoint scheduler | not yet |
| 4DVar driver | not yet |
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

1. **Wait for the adjoint roadmap.** The forward model is
   adjoint-ready in the design sense; the kernels will land in plan
   19 (no committed timeline as of 2026-04-25).
2. **Use external automatic differentiation.** Some users have
   reported success wrapping the forward `step!` call with
   `Enzyme.jl` or `ReverseDiff.jl` in source-to-source AD mode for
   small problems. The runtime is not designed for AD efficiency;
   memory will be the limiting factor at production resolutions.
3. **Use TM5-4DVar.** The TM5 four-field convection
   (`entu/detu/entd/detd`) parity work means the forward physics in
   AtmosTransport closely matches TM5; running TM5-4DVar on the same
   data, then forward-only AtmosTransport for analysis, is a
   workable workaround.

## Where to read next

- [Validation status](@ref) — what the forward model HAS been
  validated against.
- [Conservation budgets](@ref) — the explicit verification tests
  the forward operators pass.
- *Phase 7: Configuration & Runtime* — the run-side TOML schema.

!!! note "Why this page exists"
    The README's adjoint claim was caught during the codex review of
    this documentation overhaul. Rather than soften the README and
    leave a future reader to wonder, this page states the truth
    directly. The README will be updated in Phase 9 of the docs
    overhaul to point at this page.
