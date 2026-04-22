# Plan 23 Commit 7 — Production-Readiness Audit

Grade TM5Convection against the ten production principles agreed
in plan 23's drafting. Each principle has a verifiable grep /
inspection step and a pass / fail result.

**Branch tip at audit time:** `ae94383` (plan 23 Commit 6).

## Principle 1 — No runtime orientation gymnastics

> TM5 Fortran convention is k=1=surface; AtmosTransport runtime is
> k=Nz=surface. The preprocessor writes all TM5 fields in
> AtmosTransport-native orientation. The runtime column solver
> and kernels have zero orientation gymnastics.

**Check:** `grep -rnE "reverse\(|reverse!|Nz:-1:1|flip" src/Operators/Convection/ src/Preprocessing/tm5_convection_conversion.jl`

**Hits:**

- `src/Operators/Convection/cmfmc_kernels.jl:428, 549, 662` —
  CMFMC-only (not plan 23 scope).
- `src/Operators/Convection/tm5_column_solve.jl:335` —
  back-substitution direction `for k in Nz:-1:1` in `_tm5_solve!`.
  This is the ALGORITHMIC direction of upper-triangular solve
  (U x = y from bottom up); not a data orientation flip.
  Distinguishable: the data indices `k_atm` keep AtmosTransport
  orientation; only the solve sweeps through rows in the natural
  back-sub order.
- `src/Preprocessing/tm5_convection_conversion.jl:22, 33, 88, 149` —
  docstring text on ec2tm's sign flip of `mfld_ec`, happening in
  PREPROCESSING per principle 1.
- `src/Operators/Convection/TM5Convection.jl:50` — docstring text
  on "Orientation conversion + sign flip on entd happen in the
  preprocessor".

**Verdict: PASS.** No runtime orientation flips of data; all
hits are either algorithmic solver direction or preprocessor-side
sign-conversion references.

## Principle 2 — Exploit matrix structure

> TM5 `conv1 = I - dt·D` has known sparsity. Solver must use it —
> `O(Nz)` or `O(Nz²)` shippable; `O(Nz³)` general GE only if Commit 0
> survey proves no usable structure.

Commit 0 survey
([`matrix_structure.md`](matrix_structure.md)) showed the active
matrix block is dense lower + dense upper triangular within the
cloud window — NOT banded. Full partial-pivot GE on the `[1, Nz]`
range is the correct complexity class. Identity rows above the
cloud factorize trivially.

**`lmc`-limited factorization optimization (reduce work on columns
with shallow or no convection) is documented as a latent Commit 7
optimization target in NOTES.md.** Deferred — current cost is
acceptable at Nz ≤ 72 production grids.

**Verdict: PASS.** Solver class matches Commit 0 survey
recommendation; no placeholder that needs replacement later.

## Principle 3 — Adjoint-preserved factorization

> Partial-pivot `pivots::AbstractArray{Int}` stored in
> `TM5Workspace`. Forward solve applies permutation; plan 19
> adjoint replays the same factorization with transposed back-sub.
> Do NOT fuse permutation into solved coefficients.

**Check:**

- `TM5Workspace` has `pivots :: P` field
  ([`convection_workspace.jl:133`](../../src/Operators/Convection/convection_workspace.jl)).
- `_tm5_lu!` writes to `pivots[k]` at each step
  ([`tm5_column_solve.jl:249–281`](../../src/Operators/Convection/tm5_column_solve.jl)).
- `_tm5_solve!` reads `pivots[k]` for permutation replay
  ([`tm5_column_solve.jl:297–319`](../../src/Operators/Convection/tm5_column_solve.jl)).
- Factorization stored in-place in `conv1`; not fused into
  solved coefficients.

**Verdict: PASS.** Adjoint port in plan 19 is a mechanical
transpose of the forward solve + stored pivots.

## Principle 4 — Backend dispatch via `get_backend` / `parent(arr)`

> Never `arr isa Array` — breaks for SubArrays from `selectdim`.

**Check:** `grep -n "isa Array" src/Operators/Convection/{TM5Convection,tm5_kernels,tm5_column_solve}.jl`

**Hits:** zero.

**Check:** `grep -n "get_backend" src/Operators/Convection/TM5Convection.jl`

**Hits:** three — one at each `apply_convection!` method
(LatLon, RG, CS). CS uses `get_backend(q_raw[1])` on the panel
NTuple per CLAUDE.md Julia gotchas.

**Verdict: PASS.**

## Principle 5 — Parametric workspace types; NTuple{6} for CS

> `TM5Workspace{FT, M, P, C, F, A}` parametric on matrix / pivot /
> cloud-dim / scratch array types. CS variant holds NTuple{6}.

**Check:**

- Struct signature is `TM5Workspace{FT, M, P, C, F, A}`
  ([`convection_workspace.jl:132`](../../src/Operators/Convection/convection_workspace.jl)).
- LL / RG single-array constructor exists alongside NTuple{6} CS
  constructor
  ([`convection_workspace.jl:210–239`](../../src/Operators/Convection/convection_workspace.jl)).
- `Adapt.adapt_structure(::TM5Workspace)` threads all seven
  slabs through backend swap
  ([`convection_workspace.jl:242–257`](../../src/Operators/Convection/convection_workspace.jl)).

**Verdict: PASS.** GPU smoke test in bench verified
CuArray-backed workspace works end-to-end.

## Principle 6 — Basis explicit

> TM5 runs on moist basis in upstream Fortran. Decide moist-only
> vs dual-basis; reject wrong basis with clear error or document
> conversion. No silent coercion.

Commit 0 decision
([`basis_decision.md`](basis_decision.md)): `TM5Convection` is
**basis-polymorphic, identical to CMFMCConvection**. Forcing
fields must be on the same basis as `state.air_mass`. Preprocessor
defaults to moist; dry-basis variant out of plan 23 scope.

Runtime validator `_validate_convection_window!(::TM5Convection,
window, driver)` rejects `nothing` `tm5_fields` with a message
pointing at the preprocessor invocation (principle 10) but does
NOT enforce basis — that's the caller's contract.

**Verdict: PASS.** Basis policy is explicit and documented. No
silent coercion path exists.

## Principle 7 — Zero TODO / HACK / FIXME in shipped code

**Check:** `grep -rnE "# TODO|# HACK|# FIXME|not yet implemented|not supported yet" src/Operators/Convection/ src/MetDrivers/ConvectionForcing.jl src/Preprocessing/tm5_convection_conversion.jl`

**Hits:** zero.

**Verdict: PASS.** Commit 1's transitional stub messages are
fully retired (replaced by real kernel launches in Commit 4).

## Principle 8 — Doc-cited file:line pairs must resolve

> Every CLAUDE.md invariant / gotcha / NOTES.md citation is
> verified at commit time.

**Check:** `julia --project=. scripts/checks/check_markdown_citations.jl <file>`
run at every commit in this plan (see per-commit NOTES.md entries).

All plan-23 markdown files pass the citation check:

- `docs/plans/23_TM5_CONVECTION/NOTES.md`
- `artifacts/plan23/matrix_structure.md`
- `artifacts/plan23/basis_decision.md`
- `docs/plans/PLAN_HISTORY.md`
- `src/Models/README.md`
- `src/Operators/Convection/README.md`
- `docs/00_SCOPE_AND_STATUS.md`

The checker (`scripts/checks/check_markdown_citations.jl`)
caught two broken outbound links in Commit 0 drafts before they
were committed. Reusable for future plans.

**Verdict: PASS.**

## Principle 9 — Column-major loop order

> Matrix assembly loops leftmost-innermost. Per Invariant 8.
> 5× slowdown from wrong order caught before it ships.

**Check:** `_tm5_build_conv1!`, `_tm5_lu!`, `_tm5_solve!` all
iterate over column-major `Matrix{FT}` scratches with
leftmost-innermost access patterns (e.g. combine loop at
[`tm5_column_solve.jl:210–217`](../../src/Operators/Convection/tm5_column_solve.jl)
iterates `for k_atm in Nz:-1:2, for kk_atm in 1:Nz` — kk_atm is
the column index (Nz), accessed innermost).

**Commit 2 smoke benchmark** in `test_tm5_convection.jl` gates
against catastrophic layout bugs (1024-column batch timed in
under 1s on any modern CPU).

**Verdict: PASS.**

## Principle 10 — Error messages name the fix

**Check:** TM5 error messages inspected:

- `_assert_tm5_forcing`
  ([`TM5Convection.jl:173–180`](../../src/Operators/Convection/TM5Convection.jl)):
  "TM5Convection requires `forcing.tm5_fields` (NamedTuple with
  :entu, :detu, :entd, :detd) to be populated. Use
  `with_convection_forcing(model, ConvectionForcing(nothing,
  nothing, tm5_fields))` or ensure the driver populates
  `window.convection.tm5_fields`."
- `_validate_convection_window!(::TM5Convection, window, driver)`
  ([`DrivenSimulation.jl:160–170`](../../src/Models/DrivenSimulation.jl)):
  "TM5Convection requires `window.convection.tm5_fields` ...
  Preprocess the binary with
  `scripts/preprocessing/preprocess_spectral_v4_binary.jl` and
  `tm5_convection = true` in the run config, or fall back to
  `CMFMCConvection()` if you have GEOS-FP CMFMC data instead."
- `_validate_convection_window!(::AbstractConvectionOperator, ...)`
  fallback:
  "DrivenSimulation does not support convection operator
  $(typeof(op)) yet. Add a
  `_validate_convection_window!(::$(typeof(op)), window, driver)`
  method in `src/Models/DrivenSimulation.jl` that checks its
  forcing requirements."

**Verdict: PASS.** Each error message names both the fix and
the file to edit.

---

## Summary

| Principle | Verdict |
|-----------|---------|
| 1. No runtime orientation flips | **PASS** |
| 2. Exploit matrix structure | **PASS** (full GE; `lmc`-limited optimization latent) |
| 3. Adjoint-preserved factorization | **PASS** |
| 4. `get_backend` / `parent(arr)` | **PASS** |
| 5. Parametric workspace types | **PASS** |
| 6. Basis explicit (polymorphic) | **PASS** |
| 7. Zero TODO / HACK / FIXME | **PASS** |
| 8. Doc citations resolve | **PASS** |
| 9. Column-major loop order | **PASS** |
| 10. Error messages name the fix | **PASS** |

**Ten out of ten.** Ready to ship as Commit 7 retrospective.
