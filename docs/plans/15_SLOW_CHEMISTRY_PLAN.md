# Slow Chemistry Operator — Implementation Plan for Plan Agent (v1)

**Status:** Ready for execution after plan 14 has shipped.
**Target branch:** new branch from wherever plan 14's shipped work
  lives. Verify in §4.1.
**Estimated effort:** 3-5 days, single engineer / agent.
**Primary goal:** Implement a pointwise slow-chemistry operator
  (exponential decay) conforming to the operator interface from
  `OPERATOR_COMPOSITION.md` §6. Ship the chemistry block structure
  from `OPERATOR_COMPOSITION.md` §3.1 with N_chem_substeps=1
  default, future-ready for 2:1 ratio.

This is the simplest possible non-trivial operator. Its purpose is
as much **validating the operator interface** as it is implementing
physics. If plan 14's advection-oriented design choices don't
generalize cleanly to chemistry, this plan uncovers that mismatch
now, not during plan 17 (vertical diffusion) or plan 18 (convection)
where the cost is much higher.

Self-contained document. ~15 minutes to read.

**Dependencies verified:**
- Plan 11 (ping-pong) shipped
- Plan 12 (scheme consolidation) shipped
- Plan 13 (sync + CFL + rename) shipped
- Plan 14 (advection unification) shipped — `CellState.tracers_raw`
  4D layout in place, storage-agnostic accessor API
  (`ntracers`, `get_tracer`, `eachtracer`) available

**Companion documents:**
- `OPERATOR_COMPOSITION.md` §3.1 (step-level composition),
  §3.3 (chemistry block), §6 (operator interface) — authoritative
- `TIME_VARYING_FIELD_MODEL.md` — for representing decay rates
  (`ConstantField` is sufficient for current stage)
- `CLAUDE.md` § "Performance tips" — measurement discipline

---

# Revisions note

v1 (this version). Written after plan 14's shipping. Incorporates
three lessons from plan 14 retrospective:

1. **Measurement-first Commit 0 is non-negotiable.** Even for a
   trivial operator, baseline capture and end-to-end mass-
   conservation test before any code change.
2. **Test contract: observe through accessor API, not input
   arrays.** Any test helper that caches `caller_provided_array`
   and expects it to reflect post-operator state will be wrong.
   Tests must use `state.tracers.CO2` or `get_tracer(state, :CO2)`.
3. **CPU/GPU dispatch via `parent(arr) isa Array`, not
   `arr isa Array`.** The 4D views returned by `get_tracer` are
   `SubArray{FT, 3, Array{FT, 4}}`, not `Array`. Any new dispatch
   code must account for this.

---

# Part 1 — Orientation

## 1.1 The problem in one paragraph

AtmosTransport currently has a slow chemistry (exponential decay)
feature for tracers like Rn-222 (3.82-day half-life), implemented
ad hoc. Plan 15 formalizes this as a first-class operator
conforming to the `apply!` interface, wired into a chemistry
block that runs after the transport block. The operator is
pointwise: each cell decays independently per its tracer's decay
rate, integrated exactly via `Q *= exp(-k·dt)`. Zero kernel
launches per tracer (one multi-tracer kernel handles all tracers),
zero sub-stepping, unconditionally stable.

## 1.2 Why this is validation-motivated, not physics-motivated

The physics is trivial. The architecture is what matters here.

**What plan 15 tests about the architecture:**
- Does `apply!(state, meteo, grid, operator, dt; workspace)` work
  for a NON-ADVECTION operator, or did advection's needs leak into
  the signature?
- Does the chemistry block (`OPERATOR_COMPOSITION.md` §3.1)
  compose cleanly with the transport block?
- Does the `TimeVaryingField` abstraction (even as a trivial
  `ConstantField` for decay rates) integrate without friction?
- Does the step-level orchestration (`step(dt)`) handle the
  transport/chemistry split correctly?

**What plan 15 does NOT test:**
- Sub-stepping within operators (slow chemistry needs none)
- Column operators (chemistry is cell-local, not column-local)
- Implicit solvers (that's future stiff chemistry)
- Per-substep integration of `TimeVaryingField`s (that's plan 16)

Future plans (17, 18) add real operator complexity. Plan 15's
job is to ensure the scaffolding works before then.

## 1.3 What's new vs. existing ad-hoc decay

If the repo has existing slow-decay code (state: user said "I'm not
sure what we did in legacy, we had rapid developments"), plan 15's
Commit 1 surveys what's there and documents the current state.
Then the refactor:

- Replaces the ad-hoc implementation with a proper operator
- Defines `AbstractChemistryOperator` type hierarchy
- Implements `ExponentialDecay <: AbstractChemistryOperator`
- Wires into a `chemistry_block` called after `transport_block`
- Establishes `step(dt)` as the outer timestep function

Once shipped, the operator is: add a new subtype of
`AbstractChemistryOperator`, implement `apply!`, register in the
chemistry block. No other architectural work needed.

## 1.4 Scope keywords

- **Operator type:** `AbstractChemistryOperator`
- **Concrete type:** `ExponentialDecay`
- **Interface:** `apply!(state, meteo, grid, op, dt; workspace)`
- **Composition:** `step(dt)` → `transport_block(dt)` →
  `chemistry_block(dt)`
- **Per-tracer config:** decay rate `k` per tracer, as
  `TimeVaryingField` (usually `ConstantField`)

## 1.5 Test suite discipline

Same as plans 11-14:
- Baseline failures captured at Commit 0
- Per-commit test runs compare pass/fail to baseline
- Pre-existing 77 failures remain (unchanged through plans 11-14)

---

# Part 2 — Why This Specific Change

## 2.1 What gets cleaner

Before:
- Slow chemistry (if present) is ad hoc — no consistent interface
- No formal chemistry block; decay is probably inlined in driver
  code somewhere
- No path to adding new chemistry operators (e.g., a second
  decay process for a different tracer)

After:
- `AbstractChemistryOperator` hierarchy with concrete leaves
- `chemistry_block(state, meteo, grid, operators, dt)` composes
  multiple chemistry operators
- `step(state, meteo, grid, transport_op, chemistry_ops, dt)`
  is the top-level orchestration
- Future operators (stiff chemistry, photolysis) subtype
  `AbstractChemistryOperator` and implement `apply!` — no
  architectural work

## 2.2 What this validates for plans 16-18

- The operator interface generalizes beyond advection ✓/✗
- The step-level composition handles transport → chemistry ✓/✗
- `TimeVaryingField` integrates without friction ✓/✗
- The chemistry block can handle multiple operators in sequence
  ✓/✗

Each ✓/✗ will be answered during plan 15 execution. If any comes
back ✗, the composition doc needs revision before plan 16 starts.

## 2.3 What this does NOT enable

- Stiff chemistry (needs implicit solvers, 2:1 timestep ratio)
- Photolysis (needs radiation fields)
- Aerosol microphysics
- Heterogeneous chemistry

All deferred to future plans when the architecture is more
stressed.

---

# Part 3 — Out of Scope

## 3.1 Do NOT touch

- Advection machinery — plan 14 is done
- Chemistry KERNEL arithmetic beyond `exp(-k·dt)` — keep it simple
- `TimeVaryingField` abstractions — consume, don't extend
- LinRood, CubedSphereStrang — not affected by chemistry
- Meteorological input handling — chemistry is state-dependent
  only through existing tracer fields

## 3.2 Do NOT add

- Stiff chemistry scaffolding — wait until actually needed
- Photolysis rate calculation
- Temperature-dependent decay rates — Rn-222 decay is temperature-
  INDEPENDENT; don't add hooks that aren't needed
- Multiple outer-timestep support (dt_chem ≠ dt_transport) — the
  code should STRUCTURE for this but the default and only tested
  config is dt_chem = dt_transport

## 3.3 Potential confusion — clarified

**"Slow" chemistry is cell-local, not column-local.** Some chemistry
parameterizations (photolysis) are column-local (need radiation
profile). Slow decay is purely cell-local: `dQ/dt = -k(t) · Q`
independent at every (i, j, k). No stencil, no column access.

**Decay rate `k` is PER TRACER, not global.** CO2 has k=0. SF6 has
k=0 (effectively infinite lifetime). Rn-222 has k ≈ 2.1e-6 s⁻¹
(3.82-day half-life). Fossil CO2 has k=0. Each tracer's rate is
independent.

**Integration is exact, not approximated.** For constant k,
`Q(t+dt) = Q(t) · exp(-k·dt)` is exact, regardless of dt. No
sub-stepping. No stability concern. This is the CHEAPEST possible
operator to implement.

**Decay of zero-k tracers is literally a no-op.** The chemistry
operator should SKIP tracers with k=0 for perf. Or apply a no-op
`exp(-0·dt) = 1.0` multiplication — but the compiler can't
always see this. Explicit `if k == 0 continue` saves kernel work.

---

# Part 4 — Implementation Plan

## 4.1 Precondition verification

```bash
# 1. Determine parent branch (plan 14's shipped work)
git branch -a | head -20
git log --oneline --all | grep -i "plan 14\|pipeline\|advection-unification" | head -10
git checkout <parent-branch>
git pull
git log --oneline | head -20
# Expected: plans 11-14 commits visible

git checkout -b slow-chemistry

# 2. Clean working tree
git status

# 3. Verify dependency state
grep -c "tracers_raw" src/State/CellState.jl
# Expected: non-zero (plan 14 established this)

grep -c "get_tracer\|ntracers\|eachtracer" src/State/Tracers.jl
# Expected: multiple (plan 14 established accessor API)

grep -c "MassCFLPilot\|AbstractAdvection\b" src/ --include="*.jl" -r
# Expected: zero (plans 12, 13 cleaned these up)

# 4. Survey existing chemistry-like code (if any)
grep -rn "decay\|half_life\|exponential\|Rn[-_]?222\|chemistry" \
    src/ --include="*.jl" | tee artifacts/existing_chemistry_survey.txt
# Review the list. Output SHOULD be a mix of:
# - Test code with prescribed decay
# - Comments referring to chemistry
# - Maybe a few active code sites that apply decay

# 5. Capture baseline failure set
for testfile in test/test_basis_explicit_core.jl \
                test/test_advection_kernels.jl \
                test/test_structured_mesh_metadata.jl \
                test/test_reduced_gaussian_mesh.jl \
                test/test_driven_simulation.jl \
                test/test_cubed_sphere_advection.jl \
                test/test_poisson_balance.jl; do
    echo "=== $testfile ==="
    julia --project=. $testfile 2>&1 | tail -20
done | tee artifacts/baseline_test_summary.log

# 6. Record baseline
git rev-parse HEAD > artifacts/baseline_commit.txt
mkdir -p artifacts/perf/plan15
```

If preconditions fail, STOP.

**If the existing-chemistry survey (step 4) reveals substantial
existing infrastructure:** reality differs from the plan's
assumption. Before writing any chemistry operator code, document
what exists in NOTES.md and flag for human review. The plan
assumes the existing code is minimal and ad-hoc; if there's a
substantial architecture already, plan 15 becomes a refactor,
not a green-field write.

## 4.2 Change scope — the expected file list

Lines approximate; re-grep at execution time.

**Files to ADD (new):**
- `src/Operators/Chemistry/` — new directory
- `src/Operators/Chemistry/Chemistry.jl` — module file
- `src/Operators/Chemistry/AbstractChemistryOperator.jl` — type
  hierarchy
- `src/Operators/Chemistry/ExponentialDecay.jl` — concrete operator
- `src/Operators/Chemistry/chemistry_block.jl` — block-level
  composition
- `src/Operators/Chemistry/chemistry_kernels.jl` — the multi-tracer
  decay kernel
- `test/test_chemistry_kernels.jl` — unit tests
- `test/test_chemistry_operator.jl` — end-to-end operator tests
- `docs/plans/15_SLOW_CHEMISTRY_PLAN/NOTES.md`

**Files to MODIFY:**
- `src/AtmosTransport.jl` — include new `Operators/Chemistry/` module
- `src/Models/TransportModel.jl` — extend to support chemistry
  operators alongside transport
- `src/Models/DrivenSimulation.jl` — invoke `step(dt)` instead of
  `strang_split!` directly; route chemistry operators
- Any existing ad-hoc decay code — migrate to the new operator

## 4.3 Design decisions (pre-answered)

Every decision final. If ambiguous, STOP and ask.

**Decision 1: Type hierarchy.**

```julia
abstract type AbstractChemistryOperator end

struct ExponentialDecay{FT, K} <: AbstractChemistryOperator
    decay_rates::K  # typically NTuple{Nt, AbstractTimeVaryingField{FT, 0}}
                    # one scalar decay rate per tracer
    tracer_names::NTuple  # which tracers this operator applies to
end
```

`decay_rates` is per-tracer. For CATRINE (CO2, fossil CO2, SF6,
Rn-222), only Rn-222 has non-zero k. Configuration:

```julia
decay_rates = (
    ConstantField(0.0),          # CO2: no decay
    ConstantField(0.0),          # fossil CO2: no decay
    ConstantField(0.0),          # SF6: effectively no decay
    ConstantField(2.1e-6),       # Rn-222: 3.82-day half-life
)
```

Rationale: uniform API (every tracer has a field, even if it's
`ConstantField(0.0)`). Compiler eliminates zero-decay work at the
kernel level. Alternative (skip zero tracers) considered —
rejected because the kernel branch is cheap and the uniform
interface is simpler.

**Decision 2: Decay rates as `TimeVaryingField{FT, 0}`, not
scalar `FT`.**

Even though Rn-222's decay rate is a true physical constant, use
`ConstantField(k)` rather than `k`. Rationale:
- Uniform interface with future time-varying decay rates
  (temperature-dependent, etc.)
- Consistency with `OPERATOR_COMPOSITION.md` §6 guidance
- Zero runtime cost (ConstantField is essentially free)

**Decision 3: Chemistry is cell-local, multi-tracer kernel.**

Single KernelAbstractions.jl kernel processes all tracers at all
cells:

```julia
@kernel function _chem_decay_kernel!(tracers_raw, decay_rates, dt)
    i, j, k, t = @index(Global, NTuple)
    @inbounds begin
        rate = field_value(decay_rates[t], (i, j, k))  # usually scalar
        tracers_raw[i, j, k, t] *= exp(-rate * dt)
    end
end
```

This is one launch, covers all tracers and all cells. Launch
overhead is amortized; arithmetic is minimal.

**Decision 4: Zero-rate optimization via branch.**

Inside the kernel:

```julia
rate = field_value(decay_rates[t], (i, j, k))
if rate > 0
    tracers_raw[i, j, k, t] *= exp(-rate * dt)
end
# else: no-op (skip the exp call)
```

Compiler on GPU MIGHT branch-predict this well (most tracers have
rate=0). If branch overhead is measurable, alternative: skip the
loop iteration entirely for zero-rate tracers by dispatching on
the operator's tracer set.

**Decision 5: `apply!` signature follows the composition doc.**

```julia
function apply!(state::CellState,
                meteo::AbstractMeteorology,
                grid::AbstractGrid,
                op::ExponentialDecay,
                dt::Real;
                workspace=nothing)
    _chem_decay_kernel!(backend, ...)(state.tracers_raw, op.decay_rates, dt;
                                       ndrange=(Nx, Ny, Nz, Nt))
    synchronize(backend)
    return state
end
```

No workspace needed for pure decay (no scratch arrays). The kwarg
is kept for interface consistency, ignored for this operator.

**Decision 6: Chemistry block structure.**

```julia
function chemistry_block!(state, meteo, grid, operators, dt;
                          workspace=nothing)
    for op in operators
        apply!(state, meteo, grid, op, dt; workspace)
    end
    return state
end
```

Operators applied in sequence. For plan 15 with only
`ExponentialDecay`, there's a single operator in the tuple. Future
plans (stiff chemistry, photolysis) add more; the block structure
stays.

**Decision 7: `step!` is the top-level function.**

```julia
function step!(state, meteo, grid,
               transport_op, chemistry_ops, dt;
               workspace=nothing,
               N_chem_substeps::Int = 1)

    dt_transport = dt / N_chem_substeps

    for _ in 1:N_chem_substeps
        transport_block!(state, meteo, grid, transport_op, dt_transport;
                         workspace)
    end

    chemistry_block!(state, meteo, grid, chemistry_ops, dt; workspace)

    return state
end
```

`transport_block!` is a thin wrapper over `strang_split!`. Not
renaming `strang_split!` yet — that's a later plan when the
transport block composes multiple transport operators (advection
+ diffusion + convection). For plan 15, `transport_block!` just
calls `strang_split!`.

N_chem_substeps defaults to 1 (current stage: no need for 2:1
ratio). Parameter kept for future stiff chemistry.

**Decision 8: No CFL sub-stepping in chemistry.**

Exact integration means stability for any dt. No CFL check, no
sub-stepping, no iteration. The chemistry kernel runs once per
chemistry timestep.

Future stiff chemistry may need internal sub-stepping (Rosenbrock,
etc.); that's the concern of the implicit solver, not the block-
level orchestration.

**Decision 9: Existing ad-hoc decay code is replaced, not wrapped.**

If Commit 1's survey (§4.1) finds existing decay code, migrate it
to the new operator. Don't maintain both.

If the existing code is in test files only, delete tests that are
redundant with the new `test_chemistry_operator.jl` tests.

If the existing code is in src, delete it after the new operator
is wired up and passing tests.

**Decision 10: Tests use accessor API exclusively.**

Per lesson from plan 14 retrospective: observe tracer state via
`state.tracers.CO2` or `get_tracer(state, :CO2)`, never via the
original input array. Test helpers that cache caller-provided
arrays will see stale data after the chemistry operator runs.

Explicit test pattern:

```julia
# GOOD:
state = CellState(air_mass; CO2=zeros(FT, Nx, Ny, Nz))
apply!(state, meteo, grid, decay_op, dt; workspace)
@test get_tracer(state, :CO2) ≈ expected   # accessor API

# BAD:
rm_CO2 = zeros(FT, Nx, Ny, Nz)
state = CellState(air_mass; CO2=rm_CO2)
apply!(state, meteo, grid, decay_op, dt; workspace)
@test rm_CO2 ≈ expected  # WRONG — rm_CO2 is storage from pre-
                          # plan-14 design; after plan 14, CellState
                          # copies this into tracers_raw.
                          # rm_CO2 is stale.
```

Document this in the new `test_chemistry_operator.jl` file.

## 4.4 Atomic commit sequence

### Commit 0: NOTES.md + baseline

Standard pattern.

```bash
mkdir -p docs/plans/15_SLOW_CHEMISTRY_PLAN
cat > docs/plans/15_SLOW_CHEMISTRY_PLAN/NOTES.md << 'EOF'
# Plan 15 Execution Notes — Slow Chemistry Operator

Plan: `docs/plans/15_SLOW_CHEMISTRY_PLAN.md` (v1)

## Baseline
Commit: (fill in)
Pre-existing test failures: (fill in)

## Existing chemistry code survey
(fill in from §4.1 step 4)

## Commit-by-commit notes
(fill in as execution proceeds)

## Decisions beyond the plan
(fill in)

## Surprises
(fill in)

## Test anomalies
(fill in)
EOF

# Capture baseline: see §4.1

git add docs/ artifacts/
git commit -m "Commit 0: NOTES.md + baseline for plan 15"
```

### Commit 1: Chemistry module scaffolding

Create new files (empty module structure, no implementation yet):

- `src/Operators/Chemistry/Chemistry.jl` (empty module definition)
- `src/Operators/Chemistry/AbstractChemistryOperator.jl`:
  ```julia
  abstract type AbstractChemistryOperator end
  ```
- `src/Operators/Chemistry/ExponentialDecay.jl`:
  ```julia
  struct ExponentialDecay{FT, K, N} <: AbstractChemistryOperator
      decay_rates::K
      tracer_names::NTuple{N, Symbol}
  end
  ```
  Just the struct, no methods yet.
- `src/Operators/Chemistry/chemistry_block.jl`:
  ```julia
  function chemistry_block! end  # stub
  ```
- Wire into `src/AtmosTransport.jl` with `include(...)` statements

Test: `using AtmosTransport` compiles. Full test suite still passes.

```bash
git commit -m "Commit 1: Chemistry module scaffolding (no implementation)"
```

### Commit 2: `ExponentialDecay` kernel + `apply!`

Implement the kernel and the `apply!` method.

`src/Operators/Chemistry/chemistry_kernels.jl`:
```julia
using KernelAbstractions: @kernel, @index

@kernel function _chem_decay_kernel!(
    tracers_raw,
    decay_rate_values,  # per-tracer scalar rates, NTuple of FT
    dt
)
    i, j, k, t = @index(Global, NTuple)
    @inbounds begin
        rate = decay_rate_values[t]
        if rate > zero(rate)
            tracers_raw[i, j, k, t] *= exp(-rate * dt)
        end
    end
end
```

Note: `decay_rate_values` is a resolved NTuple of scalars (after
calling `field_value` on each `TimeVaryingField`). This is
computed in `apply!`, not in the kernel, because
`TimeVaryingField` may not be kernel-compatible.

`src/Operators/Chemistry/ExponentialDecay.jl` (add `apply!`):
```julia
function apply!(state::CellState{FT},
                meteo,  # unused for pure decay
                grid,   # unused
                op::ExponentialDecay,
                dt::Real;
                workspace=nothing) where FT
    # Resolve time-varying fields to scalar rates at t = current_time
    # (TODO: plumb current_time through. For plan 15, use t=0 since
    #  all rates are ConstantField.)
    rates = map(field -> field_value(field, ()), op.decay_rates)

    backend = get_backend(state.tracers_raw)
    Nx, Ny, Nz, Nt = size(state.tracers_raw)

    kernel = _chem_decay_kernel!(backend, (16, 16, 4, 1))
    kernel(state.tracers_raw, rates, FT(dt); ndrange=(Nx, Ny, Nz, Nt))
    synchronize(backend)

    return state
end
```

Test with unit tests in `test/test_chemistry_kernels.jl`:
1. Zero-rate tracer → unchanged
2. Nonzero-rate tracer → decays by `exp(-k·dt)` exactly
3. Multi-tracer mix (some zero, some nonzero) → per-tracer
   correctness
4. CPU vs GPU bit-identical (ULP tolerance)
5. Mass conservation: `sum(Q_new) == sum(Q_old) * exp(-k·dt)`
   for uniform-rate tracer

```bash
git commit -m "Commit 2: ExponentialDecay kernel + apply! + unit tests"
```

### Commit 3: `chemistry_block!` + tests

Implement the chemistry block:

```julia
function chemistry_block!(state, meteo, grid, operators, dt;
                          workspace=nothing)
    for op in operators
        apply!(state, meteo, grid, op, dt; workspace)
    end
    return state
end
```

Test with `test/test_chemistry_operator.jl`:
1. Single operator in block → same as direct `apply!`
2. Two operators → composes correctly (e.g., decay + hypothetical
   no-op)
3. Empty operator list → identity (state unchanged)

```bash
git commit -m "Commit 3: chemistry_block! composition + tests"
```

### Commit 4: `step!` top-level + TransportModel integration

Implement `step!`:

```julia
function step!(state, meteo, grid,
               transport_op, chemistry_ops, dt;
               workspace=nothing,
               N_chem_substeps::Int = 1)

    dt_transport = dt / N_chem_substeps

    for _ in 1:N_chem_substeps
        transport_block!(state, meteo, grid, transport_op, dt_transport;
                         workspace)
    end

    chemistry_block!(state, meteo, grid, chemistry_ops, dt; workspace)

    return state
end
```

`transport_block!` is a thin wrapper:
```julia
function transport_block!(state, meteo, grid, scheme, dt; workspace)
    return strang_split!(state, fluxes_from_meteo(meteo), grid, scheme,
                         dt; workspace)
end
```

(The fluxes/meteo mapping is an implementation detail; the agent
chooses based on current code state.)

Update `src/Models/TransportModel.jl` to hold chemistry operators
alongside the transport scheme:

```julia
struct TransportModel{T, C}
    transport_op::T       # e.g., SlopesScheme(MonotoneLimiter())
    chemistry_ops::C      # Tuple of AbstractChemistryOperator
end

function step!(model::TransportModel, state, meteo, grid, dt; workspace=nothing)
    return step!(state, meteo, grid,
                 model.transport_op, model.chemistry_ops, dt;
                 workspace)
end
```

Update `src/Models/DrivenSimulation.jl` to invoke
`step!(model, state, meteo, grid, dt)` instead of a direct
`strang_split!` call.

Test:
1. `step!` with no chemistry operators → same result as direct
   `strang_split!`
2. `step!` with decay operator → produces decayed tracers after
   the transport
3. Mass conservation for non-decaying tracers preserved through
   `step!`
4. End-to-end test: advect Rn-222 for 1 day, verify decay factor
   matches analytic solution

```bash
git commit -m "Commit 4: step! top-level + TransportModel integration"
```

### Commit 5: Migrate existing ad-hoc decay code

If §4.1 survey found existing decay code, migrate it now.

- Replace inline decay with `ExponentialDecay` operator configured
  at TransportModel construction
- Delete old decay code
- Update any tests that used the old API

If no existing code, this commit is empty; skip.

```bash
git commit -m "Commit 5: Migrate existing ad-hoc decay to operator"
```

### Commit 6: End-to-end test with advection + decay

New test file or extension of existing: full pipeline test:

- Initialize state with Rn-222 emission from surface (uniform or
  from a test pattern)
- Run `step!` for N steps
- Verify:
  - Total Rn-222 mass decays per `M(t) = M(0) * exp(-k·t)` (net
    of whatever sources/sinks exist)
  - Advection + decay composition gives second-order Strang
    accuracy (compare step-by-step result to running advection
    alone then decay alone then composing)

This test validates the operator composition, not just the
individual operator.

```bash
git commit -m "Commit 6: End-to-end advection + decay composition test"
```

### Commit 7: Benchmarks + documentation

Run benchmarks:
- Measure per-step time with and without chemistry operator
- Expected: chemistry adds < 5% per-step time on GPU (one extra
  kernel launch + bandwidth for tracers_raw)
- CPU impact: similar order of magnitude

Update documentation:
- `ARCHITECTURAL_SKETCH.md` — add chemistry block to the advection
  subsystem overview
- `CLAUDE.md` Performance tips — chemistry cost note
- NOTES.md retrospective

```bash
git commit -m "Commit 7: Benchmarks and documentation"
```

## 4.5 Test plan per commit

After EACH commit:

```bash
julia --project=. -e 'using AtmosTransport'
julia --project=. test/runtests.jl
```

Compare pass/fail to baseline. New failures → STOP, revert.

## 4.6 Acceptance criteria

**Correctness (hard):**
- All tests that passed in baseline pass post-refactor
- No NEW test failures beyond the pre-existing 77
- Chemistry decay matches `exp(-k·dt)` exactly (ULP tolerance)
- Mass conservation: non-decaying tracers preserved exactly,
  decaying tracers decay per analytic solution

**Code cleanliness (hard):**
- `src/Operators/Chemistry/` directory exists with the files
  listed in §4.2
- `AbstractChemistryOperator` is defined; `ExponentialDecay` is
  a concrete subtype
- `chemistry_block!` composes multiple operators in sequence
- `step!` is the top-level timestep function
- `TransportModel` carries chemistry operators alongside transport
- Existing ad-hoc decay code (if any) migrated and deleted

**Performance (soft):**
- Chemistry adds < 5% per-step wall time with one decay operator
- Zero-rate tracer optimization means CO2 / SF6 decay is
  effectively free (branch predicted well)
- No regression in advection-only per-step time

**Interface validation (hard):**
- `apply!(state, meteo, grid, op, dt; workspace)` signature works
  for `ExponentialDecay` without modification
- If the signature needed modification, NOTES.md explains why and
  OPERATOR_COMPOSITION.md §6 is updated to reflect reality

**Documentation:**
- NOTES.md complete with decisions, surprises, interface findings
- ARCHITECTURAL_SKETCH.md updated with chemistry block
- CLAUDE.md updated if new insight emerged

## 4.7 Rollback plan

Standard:
- Do not "fix forward"
- Revert to last-known-good commit
- Write failure in NOTES.md
- Stop and ask if stuck >30 minutes

Specific rollback points:
- **Commit 2 kernel fails GPU compilation.** Likely a
  KernelAbstractions idiom issue. Revert; debug on CPU first.
- **Commit 4 step! integration breaks TransportModel construction.**
  Likely a type mismatch in the chemistry_ops field. Revert; test
  TransportModel construction in isolation first.
- **Commit 6 mass conservation test fails.** This is the critical
  check. If mass is lost (or gained) beyond floating-point error,
  the kernel has a bug. Revert; check the bounds of the kernel
  loop (is every cell hit exactly once?).

## 4.8 Known pitfalls

1. **"I'll make decay rate a scalar Float instead of TimeVaryingField."**
   NO per Decision 2. Uniform interface matters for future
   extensibility.

2. **"I'll skip the chemistry_block wrapper — ExponentialDecay.apply!
   is enough."** NO. The block exists so plans 16+ can add
   multiple operators without restructuring.

3. **"I'll use Vector{AbstractChemistryOperator} for chemistry_ops
   in TransportModel."** Probably a bad idea. `Vector{AbstractT}`
   is type-unstable. Use `Tuple` or `NamedTuple` for type-stable
   iteration.

4. **"Tests should verify `caller_array === state.tracers.CO2`
   post-step."** NO per Decision 10. Accessor API only.

5. **"The decay kernel needs a workspace for scratch arrays."**
   NO. No scratch needed for pure `Q *= exp(-k·dt)`. Workspace
   kwarg is interface consistency only.

6. **"I can call `field_value(decay_rate, ())` inside the kernel."**
   MAYBE — depends on whether `TimeVaryingField` types are kernel-
   compatible. Safer: resolve rates to scalars on CPU before
   kernel launch (as `apply!` above does), pass NTuple of scalars
   to kernel.

7. **"I notice existing ad-hoc decay in src/. Let me refactor it
   and write the operator at the same time."** NO. Do Commit 1
   scaffolding FIRST, then Commit 2 kernel, THEN migrate. Keeps
   the diff mechanical.

8. **"The exp call is expensive on GPU; let me approximate."**
   NO. Exact integration is free (one exp per cell per step, and
   zero-rate tracers skip it). Precision matters — Rn-222 decays
   by 1% over ~4 hours, so approximation errors compound.

9. **"Chemistry should be applied INSIDE the transport loop for
   better accuracy."** NO per OPERATOR_COMPOSITION.md §3.1.
   Chemistry is a separate block after transport. This is what
   the palindrome structure expects (emissions at palindrome
   center, chemistry outside). Plan 15 doesn't override this.

---

# Part 5 — How to Work

## 5.1 Session cadence

- Session 1: Commit 0 + Commit 1 (scaffolding)
- Session 2: Commit 2 (ExponentialDecay kernel + apply!) + tests
- Session 3: Commits 3-4 (block + step!)
- Session 4: Commits 5-6 (migration + end-to-end)
- Session 5: Commit 7 (bench, docs, retrospective)

Short plan. If it's taking longer, something is hitting architectural
friction. Stop and ask.

## 5.2 When to stop and ask

- Commit 2 kernel fails and the fix isn't obvious
- `apply!` signature needs modification — this is interface
  validation failing, important signal
- Commit 4 reveals TransportModel refactor is bigger than expected
- End-to-end composition test (Commit 6) fails — splitting error
  could indicate a real problem
- Scope creep toward "let me also add photolysis hooks"

## 5.3 NOTES.md discipline

Update after each commit. The "interface validation" question
(does `apply!` work as-is for chemistry?) is the most important
data point for plans 17-18. Capture it explicitly.

---

# End of Plan

After this refactor ships:
- `AbstractChemistryOperator` hierarchy with `ExponentialDecay`
- Chemistry block composes multiple operators
- `step!` orchestrates transport + chemistry
- Rn-222 decays correctly in production runs
- Operator interface validated for non-advection operator

The next plans:
- Plan 16: surface flux BCs in advection + TimeVaryingField-driven
  emissions
- Plan 17: vertical diffusion
- Plan 18: convection

Each references `OPERATOR_COMPOSITION.md`, `TIME_VARYING_FIELD_MODEL.md`,
and plan 15's NOTES for interface validation lessons.
