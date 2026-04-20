# Plan 18 Addendum — Adjoint Considerations

**Purpose:** small additions to plan 18 scope to preserve adjoint
support for convection operators. The backward-mode infrastructure
itself (a dedicated plan 19) is out of scope here; this addendum
only adds cheap forward-time guards that keep that future plan
mechanical rather than a rewrite.

**Source:** inspection of TM5-4DVAR adjoint implementation
(`tm5_source/toupload/base/src/adj_*.F90`, `tm5_conv.F90`,
`convection.F90`), which reveals that the TM5 convection adjoint
is literally the forward LU factorization with `trans='T'` on the
LAPACK solve call — structurally trivial when the forward
operator preserves the right shape.

---

## Context: how the TM5 adjoint actually works

TM5's adjoint runs the operator-splitting sequence in reverse
order and applies transposed operators to an adjoint tracer field.
For the linear transport operators (convection, diffusion,
emissions), the transposition is essentially free:

- **Convection (`TM5_Conv_Apply`):** the forward solve is
  `conv1 · rm_new = rm_old` via LU factorization of
  `conv1 = I - dt·D`. The adjoint solve is
  `conv1^T · adj_rm_old = adj_rm_new`, using the SAME LU
  factorization but with LAPACK's `trans='T'` flag. Zero
  recomputation. See `convection.F90:356-360`:
  ```fortran
  if ( revert == -1 ) then
      trans = 'T'    ! adjoint run (transpose)
  else
      trans = 'N'    ! forward run
  end if
  call TM5_Conv_Apply( trans, ... )
  ```

- **Emissions:** for the linear operator `c_new = c_old + em·ew·dt`,
  the adjoint accumulates `adj_em += adj_c · ew · dt`. See
  `emission_adj.F90:447`:
  ```fortran
  adj_em(i,j,1,i_period) = adj_em(i,j,1,i_period) + adj_rm(i,j,1,itr) * x
  ```

- **Step ordering:** `adj_modelIntegration.F90:569-731` walks the
  split sequence backward. Where the forward order is
  `X Y Z V S C`, the adjoint order is `C S V Z Y X`.

**This means:** to preserve adjoint support in the forward code,
we need only ensure the forward operator structure is
transpose-able. Specifically:

1. The forward operator is a linear map on tracer state
2. Any matrix factorization is reusable in transposed form
3. Coefficients are stored in a form that allows transposition
4. Kernel math is structured so "swap inputs for outputs" works

Plan 16b established this pattern for diffusion (`(a, b, c)`
coefficients as named locals, transposition rule documented).
Plan 18 should continue the pattern for convection.

---

## Changes to plan 18 scope

### Addition A: Tier A adjoint-identity test per operator

Add ONE test to each of Commit 3's `test_cmfmc_convection.jl` and
Commit 4's `test_tm5_convection.jl` (Tier A section):

**Test: Adjoint identity holds for the linear operator.**

For an operator `L: q_in → q_out = L · q_in` (zero-forcing
case, where the operator is purely a linear transport):

```
⟨y, L · x⟩ = ⟨L^T · y, x⟩  for arbitrary x, y
```

Test recipe:

```julia
@testset "Adjoint identity (linear operator property)" begin
    # Small column — enough to exercise the matrix
    Nx, Ny, Nz = 3, 2, 8

    # Build representative CMFMC/DTRAIN (or entu/detu/entd/detd)
    # profiles
    # ... (test-specific)

    op = CMFMCConvection(...)  # or TM5Convection(...)

    # Random input and "observation" vectors (tracer-state shape)
    x = rand(FT, Nx, Ny, Nz)
    y = rand(FT, Nx, Ny, Nz)

    # Forward: Lx
    state_x = construct_state_with_tracer(x)
    apply!(state_x, nothing, grid, op, dt; workspace)
    Lx = get_tracer(state_x, :tr1)

    # Adjoint: L^T y
    # Plan 18 does NOT ship the adjoint kernel; the test builds
    # L^T by explicit construction on this small problem. For
    # TM5Convection: L = I - dt·D after the matrix builder;
    # extract L, compute L^T, solve L^T y. For CMFMCConvection:
    # build the dense linear map via finite-difference columns
    # of L, then transpose.
    L_dense = build_linear_map(op, state_shape, grid, dt)
    LTy = reshape(L_dense' * vec(y), size(y))

    # Identity: ⟨y, Lx⟩ ≈ ⟨L^T y, x⟩
    @test dot(y, Lx) ≈ dot(LTy, x)  rtol=1e-10
end
```

**For TM5Convection specifically:** the matrix `conv1` is
ALREADY built explicitly by the kernel. The adjoint identity test
can verify `⟨y, conv1⁻¹·x⟩ ≈ ⟨(conv1^T)⁻¹·y, x⟩` directly by
extracting `conv1` from the workspace and solving both systems.
No separate adjoint kernel needed — just compute LU once, solve
twice (`trans='N'` then `trans='T'`).

**For CMFMCConvection:** the operator is implicit in the two-pass
kernel. For an Nz=8 column, build the dense `L` by applying the
kernel to each unit vector `e_k` (one kernel call per level) to
fill columns of `L`. Then check `⟨y, L·x⟩ ≈ ⟨L^T·y, x⟩` for
random `x, y`. 8 kernel calls = negligible cost.

**Why this test matters:** it verifies the forward operator has
the linear structure needed for a future adjoint port. If the
test fails, the kernel has introduced some nonlinearity
(positivity clamp, conditional branch depending on `q`) that
would break adjoint transposition. Better to catch this in plan
18 than in plan 19.

### Addition B: Docstring note on adjoint path

Add to both `CMFMCConvection` and `TM5Convection` struct
docstrings a paragraph on the adjoint path:

```julia
"""
    TM5Convection(entu, detu, entd, detd; lmax_conv=0)

[existing docstring content]

# Adjoint path (not shipped in plan 18)

The forward operator `L = I - dt·D` is explicit in `conv1_ws`
after the matrix-build kernel runs. The adjoint operator is
`L^T`, and the adjoint solve reuses the same LU factorization
with `trans='T'`. This mirrors TM5's `TM5_Conv_Apply`
(tm5_conv.F90:197-341) which dispatches on a `trans` character
argument.

For a future adjoint kernel port:

1. Forward: LU factorize `conv1_ws` once per column, solve for
   each tracer (`trans='N'`).
2. Adjoint: reuse the SAME LU factorization, solve for each
   tracer with `trans='T'`. The factorization is NOT recomputed.
3. Step ordering is reversed in the adjoint time integration
   (see TM5 adj_modelIntegration.F90:569-731).

The Tier A adjoint-identity test in test_tm5_convection.jl
verifies the forward operator is transposable — this property
is preserved throughout plan 18 but the backward-mode kernel
is deferred to a dedicated plan.
"""
```

Parallel docstring for `CMFMCConvection`, noting:

```
# Adjoint path (not shipped in plan 18)

The forward operator is a two-pass column kernel (updraft
accumulation bottom-up, tendency top-down). For a future adjoint:

1. The kernel is linear in tracer mixing ratio (verified by
   Tier A adjoint-identity test) — the two-term tendency
   `tsum = cmfmc·(q_above - q_env) + dtrain·(q_cloud - q_env)`
   is linear in q throughout, as is the updraft formula.
2. The adjoint kernel applies the passes in reversed order
   with swapped roles: top-down accumulation of adj_q_cloud,
   bottom-up tendency accumulation. See TM5's adj_advectz
   for the pattern of "reverse passes, swap input/output".
3. Step ordering is reversed in adjoint time integration.

The "four-term form" needed for wet scavenging (see §5.3 of
GCHP_NOTES) introduces `q_post_mix = (CMFMC_BELOW·qc + ENTRN·q)/CMOUT`
whose division makes the operator appear nonlinear in
intermediate quantities. For INERT tracers (plan 18 scope), the
algebraic simplification to the two-term form keeps the adjoint
trivial. Future wet-deposition plan will need to handle this
carefully — see GCHP Adjoint references for a discrete-adjoint
approach that retains the four-term form.
```

### Addition C: Acceptance criterion — adjoint structure preserved

Add to §4.6 Acceptance Criteria (in "Interface validation (hard)"
section):

```
- **Adjoint structure preserved:** forward operator is linear
  in tracer mixing ratio (for inert tracers). Verified by
  adjoint-identity test ⟨y, L·x⟩ ≈ ⟨L^T·y, x⟩ in each operator's
  Tier A test suite. No kernel introduces a nonlinearity
  (positivity clamp, conditional branch on q) that would break
  transposition.
- **Operator docstrings document adjoint path:** each concrete
  convection operator includes a "Adjoint path (not shipped in
  plan 18)" section describing how a future adjoint kernel would
  reuse the forward structure.
```

### Addition D: NO positivity clamp INSIDE the linear operator

**Warning for execution agent:** GCHP's Fortran includes a
positivity clamp (`convection_mod.F90:1002-1004`):

```fortran
IF ( Q(K) + DELQ < 0 ) THEN
    DELQ = -Q(K)    ! don't let Q go negative
ENDIF
Q(K) = Q(K) + DELQ
```

**Do NOT port this clamp directly into the CMFMCConvection
kernel.** The clamp makes the operator nonlinear in q — the
adjoint-identity test would fail, and a future adjoint port
would require storing forward states to know where the clamp
fired.

**Preferred approach:** omit the clamp in the core kernel. If
negativity arises in practice, it indicates inconsistent met
data (cmfmc / dtrain / delp combination that would remove more
mass than present). Options:

1. **Add a pre-step validation** (`_check_mass_flux_consistency`)
   that detects the problematic configuration and either warns
   or caps the mass flux before the kernel sees it. This moves
   the nonlinearity to a pre-processing step whose input is
   the met data, not the tracer state — so it doesn't break
   the tracer-state adjoint.

2. **Document that the kernel assumes consistent met data** and
   that callers are responsible for validation upstream.

3. **Accept tiny negativities in practice and let a global mass
   fixer handle them** (as legacy Julia does per `ras_convection.jl:208-214`
   comment: "The global mass fixer in the run loop handles the
   small column drift from entrainment clipping").

Recommend option 1 + option 3 in combination: pre-validate the
met data upstream (once per met window), let tiny drifts be
absorbed by the global mass fixer. Kernel stays pure-linear.

If the execution agent decides option 1 is too invasive, option
3 alone is acceptable — but document clearly that the kernel
is linear-only-when-inputs-are-consistent and note the
adjoint-test case must use consistent test inputs.

### Addition E: Follow-up plan candidate — "Plan 19: Adjoint operator suite"

Register in §5.4 Follow-up plan candidates:

```
- Plan 19: Adjoint operator suite. Add reverse_mode kwarg to
  step!(model, dt) that reverses the operator-splitting sequence
  and applies transposed operators. Adjoint emission accumulator
  (adj_em += adj_rm * dt pattern from TM5 emission_adj.F90:447).
  Adjoint kernels that reuse forward factorization/coefficients.
  Observation-injection mechanism (observations as adjoint
  boundary conditions). Memory strategy for storing forward
  trajectories needed by any nonlinear pieces (advection
  limiter). Reference: TM5-4DVAR adj_*.F90 files as proven
  pattern; GEOS-Chem Adjoint (GCAdj) as alternative reference.
  Estimated 3-4 weeks, parallel in complexity to plan 18.
```

---

## What to do with this addendum

The execution agent should treat this addendum as additions to
plan 18, not replacements. All existing decisions stand. The
additions are:

- **Addition A** — adds ONE test to each of Commit 3 and Commit 4
- **Addition B** — adds a docstring paragraph to each operator
  struct in Commits 3 and 4
- **Addition C** — adds two bullets to §4.6
- **Addition D** — guidance during Commit 3 kernel implementation
- **Addition E** — adds one item to the follow-up plan candidates
  list at Commit 0

Total additional work: roughly a half-day of implementation
spread across Commits 0, 3, 4, and 10. No new commits needed.

If during execution the adjoint-identity test fails for a
specific test case, STOP and investigate — either the kernel has
an unintended nonlinearity, or the test inputs are inconsistent.
Do not lower the test tolerance or remove the test. The
linearity property is the deliverable.

---

**End of addendum.**
