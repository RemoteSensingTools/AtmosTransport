# Plan 16a Execution Notes — `TimeVaryingField` Abstraction

**Plan:** first half of [16_VERTICAL_DIFFUSION_PLAN.md](../16_VERTICAL_DIFFUSION_PLAN.md)
split at the user's direction on 2026-04-18. Plan 16 as originally
written bundled three efforts:

1. Introduce the `TimeVaryingField` abstraction (plan 15 had
   referenced it as a future dependency; the abstraction itself
   did not exist).
2. Ship `ImplicitVerticalDiffusion` using three `TimeVaryingField{FT, 3}`
   concrete types for Kz.
3. Retrofit chemistry decay rates to use the abstraction (plan 15
   D2 explicit deferral).

Precondition survey on 2026-04-18 confirmed that (1) the abstraction
was absent, and (2) [TIME_VARYING_FIELD_MODEL.md](../TIME_VARYING_FIELD_MODEL.md)
— which plan 16 cites as authoritative — did not exist in the repo.
Tackling (1) + (2) + (3) simultaneously would conflate a ~300-line
physics port (Beljaars-Viterbo) with a foundational abstraction,
validating neither in isolation.

**Split:**

- **Plan 16a** (this plan): introduce `AbstractTimeVaryingField{FT, N}`,
  ship `ConstantField{FT, N}` as the minimum concrete type, validate
  the abstraction by retrofitting `ExponentialDecay` decay rates.
  Scalar case only (`N = 0`).
- **Plan 16b** (follow-up): ship `ImplicitVerticalDiffusion`, add
  `ProfileKzField` / `PreComputedKzField` / `DerivedKzField`
  concrete types at `N = 3`, palindrome integration, TransportModel
  wiring.

This separation gives the abstraction a working first-contact
(chemistry rates) before the larger physics port lands on top of it.

## Baseline

- **Parent commit:** `78bcce9` (plan 15 Commit 5, tip of
  `slow-chemistry`).
- **Branch:** `time-varying-fields` created from `slow-chemistry`.
- **Pre-existing test failures:** 77 (plan-14 baseline, unchanged
  through plans 14-15). Plan 16a must preserve this count.

## Scope decisions — A + A per user clarification (2026-04-18)

**Q1 (kernel calling convention for scalar field read):** `field_value(f, idx)`
always takes an index tuple, even for `N = 0` where `idx = ()`.
Uniform with `N > 0` cases. Rejected no-arg variant
`field_value(f)` for lexical consistency across ranks.

**Q2 (storage for `ConstantField`):** store one scalar, no array.
Rejected a "filled backend-typed array" variant because (a) it
allocates, (b) it duplicates a value that's identical at every
index, (c) a scalar is backend-agnostic by construction. Z-
dependent fields (for Kz) are a rank-3 concrete type with per-
level storage, not a `ConstantField` variant — that lives in
plan 16b.

## Commit sequence

- **Commit 0** (this commit) — NOTES.md + TIME_VARYING_FIELD_MODEL.md
  spec + baseline capture. No source changes.
- **Commit 1** — `src/State/Fields/` directory,
  `AbstractTimeVaryingField{FT, N}`, `ConstantField{FT, N}`, unit
  tests in `test/test_fields.jl`.
- **Commit 2** — Retrofit `ExponentialDecay.decay_rates` from
  `NTuple{N, FT}` to `NTuple{N, ConstantField{FT, 0}}`. Kernel
  signature unchanged (still takes `NTuple{N, FT}` of scalars);
  `apply!` materializes via `field_value`. Public kwarg
  constructor unchanged.
- **Commit 3** — Retrospective.

## Commit-by-commit notes

### Commit 0 (this commit)

Wrote [TIME_VARYING_FIELD_MODEL.md](../TIME_VARYING_FIELD_MODEL.md)
as the authoritative interface spec. Reserved `integral_between`,
`time_bounds`, `requires_subfluxing` for future concrete types
(16b+) but did not require them of `ConstantField`.

Captured baseline commit to [artifacts/perf/plan16a/baseline_commit.txt](../../../artifacts/perf/plan16a/baseline_commit.txt).

## Decisions beyond the plan

(fill in as commits land)

## Surprises

(fill in)

## Interface validation findings

(fill in)
