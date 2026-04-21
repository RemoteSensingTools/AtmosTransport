# Legacy adjoint templates (archival)

These five files are **archival references** for the future adjoint
operator suite (plan 19). They are NOT part of the build, NOT loaded by
`AtmosTransport.jl`, and have NOT been tested against the current
operator architecture.

## Why they exist here

Plan 21 (stabilization) deleted `src_legacy/` from the working tree.
Two live hot-path modules carry comments pointing at these files as
"adjoint templates":

- [`src/Operators/Diffusion/thomas_solve.jl:30`](../../../../src/Operators/Diffusion/thomas_solve.jl#L30)
- [`src/Operators/Diffusion/diffusion_kernels.jl:32`](../../../../src/Operators/Diffusion/diffusion_kernels.jl#L32)

Both cite `boundary_layer_diffusion_adjoint.jl:74-84` as the
transposition pattern the forward Thomas solve must dualize. Plan 19
needs that reference; git archaeology is a worse UX than a first-class
archival copy.

## What each file is

| File | Size | Purpose |
|---|---|---|
| `Adjoint.jl` | 609 B | Legacy adjoint module preamble (abstract interface skeleton) |
| `boundary_layer_diffusion_adjoint.jl` | 3.4 KB | Transposed tridiagonal solve for implicit vertical diffusion — the direct reference for plan 19's diffusion adjoint |
| `checkpointing.jl` | 5.9 KB | Recompute-from-checkpoint scaffolding for the reverse sweep |
| `cost_functions.jl` | 1.9 KB | Cost-function skeletons (legacy J definitions) |
| `gradient_test.jl` | 4.5 KB | Finite-difference gradient-test harness, useful as a template |

Line numbers are preserved from the pre-cleanup `src_legacy/` tree
(commit `ec2d2c0` on branch `convection`).

## What these files are NOT

- They are NOT compiled. `AtmosTransport.jl` has no `include(...)` of
  anything in this directory.
- They are NOT guaranteed to compile today. API drift since they were
  written (type renames, state-storage refactors, basis-explicit
  advection) means they reference symbols that no longer exist.
- They are NOT a spec. Plan 19's adjoint module will rebuild the suite
  against current symbols and the `apply!(state, ..., dt; workspace)`
  contract.

## Using these as plan 19 starts

1. Read `boundary_layer_diffusion_adjoint.jl:74-84` first — it's the
   canonical transposition example referenced from the forward kernels.
2. Cross-reference against the current forward implementations:
   - `src/Operators/Diffusion/diffusion_kernels.jl`
   - `src/Operators/Diffusion/thomas_solve.jl`
3. Port symbol-by-symbol into the new adjoint module; do not
   `include(...)` these files directly.

## Provenance

- Source branch: `convection`
- Last commit containing `src_legacy/`: `ec2d2c0`
- Originally authored pre–plan-14 (pre-4D `tracers_raw` storage)
