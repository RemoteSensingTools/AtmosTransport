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

### Commit 0 — NOTES + spec + baseline (`3353d68`)

Wrote [TIME_VARYING_FIELD_MODEL.md](../TIME_VARYING_FIELD_MODEL.md)
as the authoritative interface spec. Reserved `integral_between`,
`time_bounds`, `requires_subfluxing` for future concrete types
(16b+) but did not require them of `ConstantField`.

Captured baseline commit to [artifacts/perf/plan16a/baseline_commit.txt](../../../artifacts/perf/plan16a/baseline_commit.txt).

### Commit 1 — `AbstractTimeVaryingField` + `ConstantField` (`fc86fcf`)

- Created [src/State/Fields/Fields.jl](../../../src/State/Fields/Fields.jl)
  as a submodule of State. `Fields` does not reference Grids, Mesh,
  or Basis — it stands alone so upstream operators can depend on it
  without pulling in unrelated machinery.
- Exported `AbstractTimeVaryingField`, `ConstantField`, `field_value`,
  `update_field!` from `State` and re-exported from `AtmosTransport`.
- One subtlety: Julia's default inner constructor for parametric
  structs is `ConstantField{FT, N}(value)` with implicit
  `convert(FT, value)`. My initial attempt added a `::Real` method
  which was ambiguous with that default and produced `MethodError`
  on every test. Removed; the default inner already does the right
  coercion (`ConstantField{Float32, 0}(2.5) === ConstantField{Float32, 0}(2.5f0)`).
- Tests: 21 passing in [test/test_fields.jl](../../../test/test_fields.jl).

### Commit 2 — retrofit `ExponentialDecay` rates (`e749b2a`)

Struct change:

```julia
struct ExponentialDecay{FT, N, R} <: AbstractChemistryOperator
    decay_rates  :: R   # R <: NTuple{N, AbstractTimeVaryingField{FT, 0}}
    tracer_names :: NTuple{N, Symbol}
end
```

Added an inner constructor that validates `R <: NTuple{N, AbstractTimeVaryingField{FT, 0}}`
and throws `ArgumentError` otherwise — shifts a later
`MethodError` in `apply!` to a clear failure at construction time.

Public keyword constructor (`ExponentialDecay(FT; Rn222 = …)`) is
unchanged externally — the internals wrap each rate in
`ConstantField{FT, 0}(FT(log(2) / T))`. All existing call sites
(bench script, test fixtures, DrivenSimulation wiring) kept working
without edits.

`apply!` materializes scalars from fields before launching the
kernel:

```julia
t = zero(FT)                               # placeholder until plan 16b
rates = ntuple(N) do n
    r = op.decay_rates[n]
    update_field!(r, t)                     # no-op for ConstantField
    field_value(r, ())
end
kernel!(raw, indices, rates, FT(dt), Int32(N); ndrange = …)
```

The kernel itself (`_exp_decay_kernel!`) was untouched — it still
receives `NTuple{N, FT}` of scalars. This cleanly confirms the
abstraction does not leak into the device-side code path.

One test assertion leaked the old scalar-typed `op.decay_rates[1]`;
updated it to read via `field_value`. One-line change.

## Decisions beyond the plan

1. **Three type parameters on `ExponentialDecay` (`{FT, N, R}`), not
   two (`{FT, N}`).** Considered locking the field type to
   `NTuple{N, ConstantField{FT, 0}}` directly, but that forecloses
   substituting other `AbstractTimeVaryingField{FT, 0}` concretions
   later (e.g. temperature-dependent rates, which plan 16b+ might
   introduce as a `DerivedField{FT, 0}` backed by meteorology).
   `R` as a type parameter keeps the struct flexible at zero runtime
   cost. The inner constructor enforces the `<: NTuple{N, AbstractTimeVaryingField{FT, 0}}`
   contract so misuse fails at construction.

2. **`t = zero(FT)` placeholder in chemistry `apply!`.** Plan 16a
   scope is scalar/constant fields only. Chemistry doesn't consume
   meteorology, and `ConstantField.update_field!` ignores its `t`.
   When plan 16b adds time-varying concrete types + the
   `current_time(meteo)` accessor, this call site becomes
   `t = current_time(meteo)`. Marked in code with a specific
   "plan 16b+" comment so the follow-up is discoverable by grep.

3. **Did NOT extend `Fields` to depend on Grids or Mesh.** The
   `Fields` submodule only uses `Base` — no Grids, no Basis, no
   CellState. This keeps the abstraction legitimately foundational.
   Rank-3 Kz concrete types in 16b will need grid geometry for
   Δz calculations, but they can pull that in at their own
   submodule level without poisoning the abstraction.

## Surprises

1. **Default inner constructor handles `Real`-to-`FT` coercion for
   free.** My original draft added an explicit
   `ConstantField{FT, N}(value::Real) = ConstantField{FT, N}(FT(value))`
   outer constructor. That was ambiguous with the default inner
   `(ctor_self::Type{ConstantField{FT, N}} where {FT, N})(value)`,
   which Julia synthesizes to call `new{FT, N}(convert(FT, value))`.
   The default already accepts any `value` that converts to `FT`.
   Lesson: `struct Foo{FT}; x::FT end; Foo{Float64}(1)` just works;
   no extra method needed.

2. **No dependency chain changes were needed.** Adding a new
   submodule to `State` is isolated; `AtmosTransport.jl` only
   needed five new exported symbols. Zero churn in downstream
   modules (Operators, Models, MetDrivers, IO).

## Interface validation findings

1. **`apply!` signature unchanged.** The chemistry `apply!` kept
   its plan-15 `(state, meteo, grid, op, dt; workspace)` shape.
   The retrofit is entirely inside `op.decay_rates` handling —
   every bench, every test, every DrivenSimulation call continued
   to work without a single caller-side edit. Confirms plan 15's
   signature decision survives the abstraction change.

2. **`TimeVaryingField` works cleanly for the `N = 0` scalar case.**
   `field_value(f, ())` is ergonomically acceptable and
   `@inferred`-clean. No sign that the "empty tuple for scalars"
   convention is a footgun.

3. **Kernel-safety via materialization, not through the field.**
   The decision to materialize scalars on the host before kernel
   launch (rather than calling `field_value` inside the kernel)
   worked well for chemistry, where the rates are few (N ≤ ~5 in
   practice). For 3D Kz (many values per launch) plan 16b will
   call `field_value` directly inside the diffusion kernel, and
   the kernel-safety contract in [TIME_VARYING_FIELD_MODEL.md §6](../TIME_VARYING_FIELD_MODEL.md)
   will earn its keep. For 16a's scope, both patterns are valid;
   materialization was the simpler change.

## Scope for plan 16b (follow-up)

- `ProfileKzField{FT, 3}` — per-level `Vector{FT}` of Kz values,
  `field_value(f, (i, j, k))` returns `f.profile[k]`. Minimum type
  that exercises the `N = 3` case.
- `PreComputedKzField{FT, 3}` — full 3D array backed by a
  `StepwiseField` (new concrete type in 16b).
- `DerivedKzField{FT, 3}` — the Beljaars-Viterbo port from
  [src_legacy/Diffusion/pbl_diffusion.jl](../../../src_legacy/Diffusion/pbl_diffusion.jl).
- `current_time(meteo)` accessor on the meteorology object.
- `ImplicitVerticalDiffusion` operator + Thomas solve + palindrome
  integration + TransportModel diffusion field + DrivenSimulation
  wiring.

The abstraction this plan introduced is in place and validated.
Plan 16b is pure additions — no need to modify
`AbstractTimeVaryingField`, `ConstantField`, or the `Fields`
submodule shape.
