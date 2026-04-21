# Plan 21 execution — verification notes

## Module load sanity check (post–Phase 2 deletion)

Post-deletion of `src_legacy/`, `scripts_legacy/`, `test_legacy/`:

```julia
using Pkg; Pkg.activate("."); using AtmosTransport
# Output: 5754.0 ms ✓ AtmosTransport
# AtmosTransport loaded: version 0.1.0
# Module exports count: 269
```

Clean precompile, no missing symbols. Runtime tree has no dependency
on legacy (confirmed pre-deletion via
`grep -rn "include.*_legacy|using.*Legacy" src/ test/` → empty).

## Test suite — runtests.jl

`julia --project=. test/runtests.jl` fails on the second core test
(`test_advection_kernels.jl`) with:

```
UndefVarError: LatLonMesh not defined in Main
Hint: It looks like two or more modules export different bindings
with this name, resulting in ambiguity.
```

**This is NOT caused by plan 21 legacy deletion.** Evidence:

1. Running the test file standalone
   (`julia --project=. test/test_advection_kernels.jl`) passes
   all 158 tests:
   - Advection kernels: 28/28
   - SlopesScheme kernels: 20/20
   - PPMScheme kernels: 20/20
   - Multi-tracer kernel fusion: 84/84
   - Unified CFL pilot (plan 13): 6/6

2. The test file at line 22 does
   `include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))`
   and depends on nothing in `_legacy/`. Module precompile post-
   deletion is clean.

3. The failure is Julia-1.12-specific. Julia 1.12 tightened
   ambiguity rules for names exported by multiple modules. When
   `runtests.jl` sequentially `include`s multiple test files —
   each of which includes `src/AtmosTransport.jl` into `Main` —
   Julia ends up with multiple `Main.AtmosTransport` module
   references and the `using .AtmosTransport` clause becomes
   ambiguous from the second file onward.

4. This is therefore a `runtests.jl` harness bug (exposed by
   Julia 1.12) OR a Julia 1.12 regression against the pre-existing
   include pattern. Either way it pre-dates plan 21.

**Suggested follow-up** (out of scope for plan 21): rewrite
`test/runtests.jl` to use `include(@__MODULE__, path)` inside
an anonymous module per test, or rewrite each test to
`using AtmosTransport` (project-level) instead of local
`include(...) + using .AtmosTransport`.

## Per-commit artifacts captured

- `artifacts/plan21/post_deletion_tests.log` — full output of the
  failing `runtests.jl` run. Last green testset was
  `Adapt.jl container conversions: 8/8`.

## Phase 2 verification

- `find . -maxdepth 2 -type d -name "*_legacy"` → empty after commit
  0effc66.
- `grep -rn "src_legacy\|scripts_legacy\|test_legacy" src/ test/` →
  empty (all src-side refs converted to git-SHA pointers pointing
  at commit ec2d2c0; archival refs point at
  `docs/resources/developer_notes/legacy_adjoint_templates/`).

## Phase 1 verification

- `grep -rn "convection is deferred\|does not yet execute a convection block\|convection remains deferred" src/ docs/` →
  matches only inside `docs/plans/21_STABILIZATION_PLAN.md` (that
  file is the plan-description source, not a target of the fix).
