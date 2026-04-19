# Plan 16b Execution Notes — Vertical Diffusion Operator

**Plan:** [16_VERTICAL_DIFFUSION_PLAN.md](16_VERTICAL_DIFFUSION_PLAN.md) (v1)
— the second half of the original plan 16, after 16a ([docs/plans/16A_TIME_VARYING_FIELDS_PLAN/NOTES.md](../16A_TIME_VARYING_FIELDS_PLAN/NOTES.md)) shipped the
`AbstractTimeVaryingField{FT, N}` + `ConstantField{FT, N}`
abstraction and retrofitted `ExponentialDecay.decay_rates`.

## Baseline

- **Parent commit:** `3cd9010` (plan 16a Commit 3, tip of
  `time-varying-fields`).
- **Branch:** `vertical-diffusion` created from `time-varying-fields`
  HEAD. The resume memo stated `time-varying-fields` had been
  merged into `advection-unification`; verification on 2026-04-19
  showed the merge had NOT happened (`git log advection-unification..time-varying-fields`
  shows the four 16a commits). Branching directly from
  `time-varying-fields` keeps history linear and defers the
  `advection-unification` merge as a separate concern.
- **Pre-existing test failures:** 77, inherited from plan 16a
  (`test_basis_explicit_core: 2`, `test_structured_mesh_metadata: 3`,
  `test_poisson_balance: 72`). Plan 16b must preserve this count.
- **Fast-path baseline** (captured in
  [artifacts/plan16/baseline_test_summary.log](../../../artifacts/plan16/baseline_test_summary.log)):
  - `test_fields.jl`: 21 pass (N=0 ConstantField only)
  - `test_chemistry.jl`: 37 pass

## Deviations from plan doc §4.4

Two scope decisions recorded up front:

1. **Commit 2a (chemistry rates retrofit) is already done.** Plan 16a
   landed it as its Commit 2 (`e749b2a`); `ExponentialDecay.decay_rates`
   already stores `NTuple{N, ConstantField{FT, 0}}`. Skip.

2. **`ConstantKzField` is redundant.** Plan doc §4.3 Decision 3 lists
   `ConstantKzField{FT} <: AbstractTimeVaryingField{FT, 3}` as one of
   three Kz field types; that predates `ConstantField{FT, N}`, which
   already provides the scalar spatially-uniform case at any rank. The
   existing `test_fields.jl` includes a `@testset "volume (N=3) — anticipating
   Kz in plan 16b"` (lines 55–60) that already demonstrates
   `ConstantField{Float64, 3}` returning a scalar from a 3-tuple index.
   The first new concrete type will be **`ProfileKzField`** (per-level
   profile, rank-3, vertically-varying) — the simplest field type that
   actually exercises the rank-3 path beyond the spatially-uniform
   case.

## Commit sequence (initial draft; will evolve)

- **Commit 0** — NOTES.md + baseline, plan doc moved into subfolder
  (convention-matching with 14/15/16a).
- **Commit 1a** — `ProfileKzField` rank-3 profile field + tests.
- **Commit 1b** — `PreComputedKzField` (deferred; needs `StepwiseField`
  design).
- **Commit 1c** — `DerivedKzField` Beljaars-Viterbo port
  (~300 lines, highest-risk).
- **Commit 2** — Thomas solve + diffusion kernel.
- **Commit 3** — `ImplicitVerticalDiffusion` operator + `apply!`.
- **Commit 4** — Palindrome integration (`X Y Z V Z Y X`) in
  `strang_split_mt!`.
- **Commit 5** — `TransportModel` + `DrivenSimulation` wiring.
- **Commit 6** — Benchmarks.
- **Commit 7** — Documentation + retrospective.

## Commit-by-commit notes

### Commit 0 — NOTES + baseline

Moved `docs/plans/16_VERTICAL_DIFFUSION_PLAN.md` into the subfolder
to match the plan-folder layout used by 14/15/16a. Wrote this file
as the execution log.

### Commit 1a — `ProfileKzField` rank-3 profile

- Created [src/State/Fields/ProfileKzField.jl](../../../src/State/Fields/ProfileKzField.jl):
  `struct ProfileKzField{FT} <: AbstractTimeVaryingField{FT, 3}`
  with a single `profile::Vector{FT}` field. `field_value(f, (i,j,k))`
  returns `@inbounds f.profile[k]`; `update_field!` is a no-op.
- Re-exported `ProfileKzField` through the module chain
  (`Fields` → `State` → `AtmosTransport`).
- Extended [test/test_fields.jl](../../../test/test_fields.jl) with
  a `ProfileKzField` testset (7 blocks, 26 tests):
  - Construction + type bounds (FT=Float64, Float32)
  - `field_value` selects the k coordinate
  - `field_value` ignores i, j (horizontal invariance)
  - `update_field!` is a no-op (field unchanged, returns `f`)
  - Type stability (`@inferred`)
  - Rank-mismatched index → MethodError
  - Kernel-safety on CPU backend (KA kernel writes k-varying profile
    into every column)

**Results:** 26 new tests pass; 21 pre-existing `ConstantField`
tests unchanged; chemistry regression unchanged (37/37). Rank-3
path validated beyond the spatially-uniform `ConstantField{FT, 3}`
special case.

**Storage note:** `profile` is a host `Vector{FT}`. On CPU backends
the kernel-safety test passes directly. GPU dispatch is deferred
until Commit 3 (diffusion operator) — the first call site that
launches a kernel consuming `ProfileKzField`. If GPU dispatch on
`Vector{FT}.getindex` fails inside a kernel, the fallback noted
in the plan (`NTuple{Nz, FT}` storage) is mechanical.

### Commit 1b — `PreComputedKzField` rank-3 array wrapper

- Created [src/State/Fields/PreComputedKzField.jl](../../../src/State/Fields/PreComputedKzField.jl):
  `struct PreComputedKzField{FT, A} <: AbstractTimeVaryingField{FT, 3}`
  wrapping a 3D `AbstractArray{FT, 3}` (Array, CuArray, MtlArray —
  parametric on `A`). `field_value(f, (i,j,k))` is a direct indexed
  read; `update_field!` is a no-op.
- Inner constructor enforces rank-3 via
  `A <: AbstractArray{FT, 3}` — a 2D or 4D input raises
  `MethodError` at construction, not at the first `field_value` call.
- Caller-owned storage: the field holds a reference, not a copy.
  Operational path (met window advances) is "caller mutates
  `f.data` in place"; `update_field!` is a no-op because the array
  is already current. A future stepwise-in-time variant (4D buffer
  + window index) can be added as a separate concrete type.
- Wired through `Fields → State → AtmosTransport`.
- Extended `test/test_fields.jl` with 19 new tests:
  construction + type bounds, (i,j,k)-independent access (distinct
  fingerprinted data ≠ fingerprint-elsewhere), no-op `update_field!`,
  caller-owned mutation visibility, type stability, rank-mismatched
  construction rejection, CPU kernel-safety via element-wise copy.

**Results:** 19 new tests pass; 21 `ConstantField` + 26 `ProfileKzField`
tests unchanged (total 66 in `test_fields.jl`). GPU kernel-safety
deferred to Commit 3.

### Commit 1c — `DerivedKzField` Beljaars-Viterbo port

- Created [src/State/Fields/DerivedKzField.jl](../../../src/State/Fields/DerivedKzField.jl):
  - `PBLPhysicsParameters{FT}` struct (β_h, Kz_bg, Kz_min, Kz_max,
    kappa_vk, gravity, cp_dry, rho_ref — legacy defaults).
  - `_beljaars_viterbo_kz` — line-for-line port of legacy `_pbl_kz`
    (src_legacy/Diffusion/pbl_diffusion.jl:66). Pure, GPU-safe,
    `@inline`. Returns Kz in m²/s geometric.
  - `_obukhov_length` — pure, returns `(L_ob, H_kin)`. Includes
    the signed 1e-10 safety offset from the legacy implementation
    so hflux=0 doesn't blow up.
  - `_prandtl_inverse` — pure, matches legacy formula
    (diffusion.F90:1213-1230 via pbl_diffusion.jl:198-210). Returns
    `one(FT)` unless L_ob<0, H_kin>0, h_pbl>10.
  - `DerivedKzField{FT, SF, DELP, A, P} <: AbstractTimeVaryingField{FT, 3}`
    struct holding 4 rank-2 surface fields, rank-3 delp,
    3D cache, params. Inner constructor validates rank of each
    surface entry.
  - `update_field!(f, t)` — refreshes every input field via
    `update_field!`, then fills cache column-by-column on the host
    (two-pass hydrostatic integration: first for z_col, second
    for cell-center heights + Kz).
  - `field_value(f, (i,j,k))` — direct cache read.
- Wired through `Fields → State → AtmosTransport`.
- Extended [test/test_fields.jl](../../../test/test_fields.jl) with
  79 new tests across 4 testsets:
  - `PBLPhysicsParameters defaults` (13) — value checks, FT propagation,
    kwarg overrides.
  - `_beljaars_viterbo_kz` (15) — regime corners: above taper returns
    Kz_bg exactly; stable, unstable surface-layer, unstable mixed-layer
    spot values checked against hand-expanded formulas; Kz_min / Kz_max
    clamping; taper-zone blend formula; type stability.
  - `_obukhov_length` (9) — sign convention (hflux>0 → unstable L<0);
    zero-hflux safety (finite L via offset); hand-verified spot value.
  - `_prandtl_inverse` (7) — returns 1 in stable, non-positive H_kin,
    tiny h_pbl; > 1 in convective unstable.
  - `DerivedKzField` (35) — construction + type bounds; rank-mismatch
    rejection; `update_field!` populates cache; high-altitude cells
    return Kz_bg; `field_value` reads from cache; changing surface
    fields changes cache; type stability.

**Results:** 145 tests in `test_fields.jl` (was 66); 37 chemistry
tests unchanged. The physics port line-matches the legacy reference;
deviations are limited to scope (no pressure-basis D coefficients,
no Thomas solve — those belong to Commits 2/3).

**Subtlety found during tests:** The taper-zone blend formula is
`Kz = Kz_bg + frac × (Kz_in_pbl_clamped - Kz_bg)`. When the in-PBL
Kz clamps to `Kz_min` (0.01) because `(1-z/h)² → 0` near the PBL
top, the blend *increases* Kz from 0.01 at z=h through 0.055 at
z=1.1h to 0.1 (=Kz_bg) at z=1.2h. My initial test assumed a
decreasing blend (which only holds when in-PBL > Kz_bg, i.e. strong
mixing). Fixed to allow either direction; test now verifies the
blend formula directly, not a particular sign.

**Julia gotcha hit:** default kwarg `params = PBLPhysicsParameters{FT}()`
in a `where FT` method evaluates at module scope, not method-body
scope, producing `UndefVarError: FT not defined`. Worked around by
using `params = nothing` and resolving inside the body — a pattern
worth remembering when writing generic constructors with type-
parameterized defaults.

### Commit 2 — Thomas solve + vertical diffusion kernel

Shipped the solver infrastructure that will sit under Commit 3's
`ImplicitVerticalDiffusion.apply!`. No operator type yet; no
palindrome integration yet.

- Created [src/Operators/Diffusion/](../../../src/Operators/Diffusion/):
  - `Diffusion.jl` — module file, `using ...State: AbstractTimeVaryingField, field_value`,
    exports `solve_tridiagonal!`, `build_diffusion_coefficients`,
    `_vertical_diffusion_kernel!`.
  - `thomas_solve.jl` — `solve_tridiagonal!(x, a, b, c, d, w)`
    with caller-supplied workspace (plan's Commit-2 constraint #2);
    `a, b, c, d` read-only. `build_diffusion_coefficients(Kz_col, dz_col, dt)`
    as the **reference** Backward-Euler coefficient builder, returning
    three `Vector{FT}`s. Adjoint-transposition rule documented in the
    header.
  - `diffusion_kernels.jl` — `_vertical_diffusion_kernel!` with
    ndrange `(Nx, Ny, Nt)`. Inlines the same coefficient formulas as
    the reference; `(a_k, b_k, c_k, d_k)` are named locals at each k
    rather than pre-factored (plan's Commit-2 constraint #3).
    `w_scratch[i, j, k]` holds the Thomas forward-elimination factor
    between the two passes.
- Wired into `Operators.jl` (`include("Diffusion/Diffusion.jl")`;
  `using .Diffusion`; exports). Top-level re-export from
  `AtmosTransport.jl`.
- Created [test/test_diffusion_kernels.jl](../../../test/test_diffusion_kernels.jl)
  with 33 tests across 7 testsets:
  1. `build_diffusion_coefficients` (19) — hand-computed uniform case;
     Neumann BCs (`a[1] = c[Nz] = 0`); `dt = 0` gives identity;
     varying Kz produces asymmetric `a` vs `c` at the same interface;
     dimension-mismatch throws; type stability.
  2. `solve_tridiagonal!` (8) — identity matrix returns `d`; matches
     Julia's dense `Tridiagonal \ d` on a random SPD system; does
     not mutate `a, b, c, d`; workspace-length check fires via
     `@boundscheck`; type stability.
  3. Adjoint-structure test (2) — built forward L on Nz=8, constructed
     L^T via documented rule, confirmed `Matrix(L_T) ≈ Matrix(L)'`,
     and verified adjoint identity `⟨y, Lx⟩ = ⟨e, x⟩` (with
     `L^T y = e`).
  4. KA kernel vs. pure-Julia reference (1) — random (Nx=3, Ny=2,
     Nz=6, Nt=2) with random Kz and dz, tight agreement (atol=1e-12).
  5. Gaussian broadening (1) — Nz=201, uniform K=1 m²/s, dt=0.5,
     fitted variance growth matches `σ₀² + 2Kt` within 5% (tolerance
     scales with K·dt/dz²=0.5).
  6. Mass conservation under Neumann BCs (1) — `Σq` preserved to
     relative 1e-12 after one step.
  7. `ConstantField{FT, 3}` dispatch (1) — kernel accepts it via
     `field_value`, matches reference.
- Test totals: `test_diffusion_kernels.jl` 33/33; `test_fields.jl`
  145/145 (unchanged); `test_chemistry.jl` 37/37 (unchanged).

**Key design choices locked in** (all three per user's revised scope):

1. **`dz` is `AbstractArray{FT, 3}`, not a `TimeVaryingField`.**
   The codebase already represents layer thicknesses as plain arrays
   (`level_thickness`, `delp`); wrapping dz just for diffusion would
   introduce cross-operator inconsistency. Commit 3's `apply!` will
   supply a `dz_scratch::AbstractArray{FT, 3}` via the workspace.
2. **Workspace supplied by caller.** `solve_tridiagonal!` takes
   `w::AbstractVector{FT}` and does not allocate. The kernel takes
   `w_scratch::AbstractArray{FT, 3}`. Commit 3 extends
   `AdvectionWorkspace` with `w_scratch` and `dz_scratch` fields.
3. **Kernel inlines coefficient arithmetic; reference is a separate
   function.** `_vertical_diffusion_kernel!` computes `(a_k, b_k, c_k, d_k)`
   as named locals; `build_diffusion_coefficients` exists for tests
   and for the eventual adjoint kernel's reference. Docstrings at
   both sites cross-reference and note that "any change to the
   formulas here requires updating the reference (or vice versa)".

**Adjoint-structure preservation — verified by test.** The
transposition rule (`a_T[k] = c[k-1]`, `b_T[k] = b[k]`, `c_T[k] = a[k+1]`)
is documented at the top of `thomas_solve.jl` and at the kernel's
adjoint-note comment. The adjoint-identity test ships in
`test_diffusion_kernels.jl`; a future adjoint solver only needs to
apply the transposition rule and call `solve_tridiagonal!`.

## Decisions beyond the plan

(To be filled in as they arise.)

## Surprises

(To be filled in as they arise.)

## Interface validation findings

(To be filled in — per plan doc §5.3, three specific items:
`apply!` signature works unchanged for diffusion; tridiagonal
structure is transposable without rewrite; `TimeVaryingField`
works cleanly for 3D Kz.)
