# Plan 25 — LinRood CS adjoint (advection + diffusion + convection)

Branch: `convection`
Owner: claude (with codex review at every commit)
Started: 2026-05-10
Status: planning

## Goal

Extend the existing `src/Adjoints/Adjoints.jl` reverse pass so the cubed-sphere
LinRood–PPM finite-volume scheme is a fully supported advection scheme inside
the surface-emission footprint / 4D-Var driver, on the same footing as
`UpwindScheme`, `SlopesScheme(NoLimiter)`, `PPMScheme(NoLimiter)`, and the
limited `PPMScheme(MonotoneLimiter)` (branch-tape) that already work.

The diffusion (implicit Backward-Euler / Thomas) and convection (CMFMC + TM5
column LU) adjoints already exist and are decoupled from the advection scheme
— this plan reuses them as-is and adds end-to-end FD coverage with LinRood as
the advection step.

## Out of scope

- Revolve checkpointing (separate roadmap item).
- Adjoint of the LinRood vertical remap (`VerticalRemap.jl`); the current
  vertical Z step uses the same Strang Z sweep as the other PPM schemes and
  reuses the existing kernels.
- Production 4D-Var driver beyond the existing `cs_surface_flux_4dvar` /
  `_optimize` prototypes.
- LL/RG LinRood (this branch is CS-only — there is no LL LinRood operator).

## Forward LinRood structure being transposed

`fv_tp_2d_cs!()` (LinRood.jl:695) and `fv_tp_2d_cs_q!()` (LinRood.jl:805)
implement the GMAO Lin–Rood three-phase unsplit horizontal update per
substep:

- **Phase 1 (pre-advect):** apply a half-step Y advection to `q` to produce an
  intermediate `q*`; apply a half-step X advection to `q` to produce `q'`.
  (`_pre_advect_y_kernel!` LinRood.jl:364 and `_pre_advect_x_kernel!`
  LinRood.jl:377.)
- **Phase 2 (cross fluxes):** PPM-X face flux from `q*` → `fx_in`; PPM-Y face
  flux from `q'` → `fy_in`. (`_ppm_x_face_from_q_kernel!` LinRood.jl:299,
  `_ppm_y_face_from_q_kernel!` LinRood.jl:325.)
- **Phase 3 (direct fluxes):** PPM-X face flux from `q` → `fx_out`; PPM-Y face
  flux from `q` → `fy_out`. (`_ppm_x_face_kernel!` LinRood.jl:270,
  `_ppm_y_face_kernel!` LinRood.jl:241.)
- **Update:** in **mass space** (LinRood.jl:394) the kernel writes
  ```
  rm_new = rm + am_w · ½(fx_in[i,j] + fx_out[i,j]) − am_e · ½(fx_in[i+1,j] + fx_out[i+1,j])
              + bm_s · ½(fy_in[i,j] + fy_out[i,j]) − bm_n · ½(fy_in[i,j+1] + fy_out[i,j+1])
  m_new  = m + (am_w − am_e) + (bm_s − bm_n)
  ```
  No explicit `dt/dp` factor — `am`, `bm` are the **signed mass-flux times
  dt** already (units of mass). The face arrays `fx_*`, `fy_*` carry mixing
  ratios. The q-space variant `_linrood_update_q_kernel!` (LinRood.jl:506)
  divides by `m_new` to return mixing ratios; the rm-space variant returns
  `rm_new`, `m_new` directly. The CS Strang orchestrator uses the rm-space
  path (`fv_tp_2d_cs!`).

The vertical Strang Z sweep is wrapped by `strang_split_linrood_ppm!`
(LinRood.jl:909) and `_strang_split_linrood_ppm_cs!` (LinRood.jl:921),
which call `_sweep_z!` (the same per-direction sweep used by the existing
PPM scheme path). The Z reverse path therefore **can** reuse the existing
`_adjoint_scheme_sweep!(direction = :z, scheme = PPMScheme(MonotoneLimiter), …)`
machinery — Commit 5 verifies this by running a Z-only FD test against the
LinRood forward + reused Z-adjoint composition before recording the full
LinRood horizontal record.

## Architectural choice: stencil-recompute, no per-face branch tape

`LinRoodPPMScheme{ORD}` is **inherently nonlinear** in `q`: the four PPM
face kernels (`_ppm_x/y_face_kernel!` and the `_from_q` variants) all call
`_apply_monotonicity` (LinRood.jl:200) unconditionally and `_ppm_face_value`
(LinRood.jl:215) picks the donor cell on the sign of the mass flux `am`/`bm`.
There is no `LinRoodPPMScheme{NoLimiter}` variant — the limiter is built in.
The two pre-advect kernels (`_pre_advect_x/y_kernel!`) and
`_linrood_update_kernel!` are linear given fixed velocities.

LinRood adopts the same **recompute strategy** as the existing
`PPMScheme{MonotoneLimiter}` adjoint
(`_ppm_monotone_face_coeffs` Adjoints.jl:610-668): the per-substep tape
stores **only the field state** — the substep's input `rm`/`m` plus the
two intermediates `q*` (from phase-1 pre-advect-Y) and `q'` (from
phase-2 pre-advect-X) — and the reverse pass recomputes face-value
sensitivities on-the-fly. No per-face decision tape is recorded; this
matches the existing PPM(MonotoneLimiter) memory profile (modulo the
larger field-state footprint discussed under "Risks").

The mathematical body of the recompute, however, is **not** shared
with the existing PPM(MonotoneLimiter) helper. The existing helper's
downstream chain
(`_ppm_edge_value_ad → _ppm_limit_profile_monotone_ad →
_limited_moment_monotone_ad`, Adjoints.jl:561-607) implements the
limited-moment scaled-mass formulation used for tracer-mass fluxes in
the split-sweep PPM code. The LinRood face kernels implement a different
operator (`_apply_monotonicity` on edge values + parabolic
`_ppm_face_value` on mixing ratios with α-form donor-mass denominator,
LinRood.jl:200-231). Commit 3 therefore ships fresh d6-AD helpers
(`_linrood_ppm_edge_values_d6`, `_linrood_apply_monotonicity_d6`,
`_linrood_ppm_face_value_d6`) that match LinRood's forward chain
exactly. The only shared building blocks are the very low-level
`_d6_basis`, `_d6_add`, `_d6_sub`, `_d6_scale`, `_d6_zero` tangent-vector
primitives (Adjoints.jl:538-557).

The `_from_q` and `_from_rm` variants differ from each other only in
the input pathway: cell values come from `q[ii, jj, k]` directly with
`_d6_basis(FT, n, one(FT))` tangents (q-input variant) versus
`rm[ii, jj, k] / m[ii, jj, k]` with two d6 contributions per cell
(`drm/m` and `−rm/m²·dm`, rm-input variant). The donor-cell mass
additionally feeds the α-denominator in `_ppm_face_value` and
contributes a separate term — see Commit 3.

## Staged commit plan

Each commit gets `codex review --uncommitted` first; only land if cleared.

### Commit 0 — plan only (docs)

- This NOTES.md as a standalone commit so the staged plan can be reviewed
  before any code change. No source touched.

### Commit 1 — Adjoints.jl scaffolding + adjoint of `_linrood_update_kernel!`

Scaffolding first (required even for the kernel-only test):
- Extend the `using ..Operators.Advection: …` import in
  [Adjoints.jl:20](src/Adjoints/Adjoints.jl#L20) to include
  `LinRoodPPMScheme`.
- Add a new `CSAdjointLinRoodScheme = LinRoodPPMScheme{ORD} where ORD`
  constant, and extend `CSAdjointSupportedScheme`
  (Adjoints.jl:33-35) to include it. No code yet dispatches on the new
  union member — `_record_cs_adjoint_tape` will need a third method in
  Commit 5.

Kernel adjoint:
- Implement `_linrood_update_kernel_adjoint!` that, given
  `lambda_rm_new[ii, jj, k]`, accumulates with `@atomic` into
  `lambda_rm[ii, jj, k]`, `lambda_fx_in/fx_out[i, j, k]`,
  `lambda_fx_in/fx_out[i+1, j, k]`, `lambda_fy_in/fy_out[i, j, k]`,
  `lambda_fy_in/fy_out[i, j+1, k]`. The coefficients are signed `am`/`bm`
  values from the meteo tape, halved (the ½ in front of the
  averaged-flux expression). `m_new` is independent of tracer rm so
  `lambda_m` receives nothing from this kernel.
- Place the kernel in a new
  `src/Operators/Advection/linrood_adjoint_kernels.jl` (included from
  `LinRood.jl`) so the forward + adjoint live in the same module; the
  existing `Adjoints.jl` imports them via the `Operators.Advection`
  re-export, matching the pattern used for the existing scheme face
  kernels.
- Test (in a new `test/test_linrood_kernel_adjoints.jl`): pick random
  panel-shaped arrays `(rm, m, am, bm, fx_in, fx_out, fy_in, fy_out)`
  and random adjoint seed `lambda_rm_new`; compute the forward and
  reverse passes and assert
  `⟨lambda_rm_new, rm_new(rm, fx_*, fy_*)⟩
    == ⟨lambda_rm, rm⟩ + ⟨lambda_fx_in, fx_in⟩ + … + ⟨lambda_fy_out, fy_out⟩`
  to `rtol ≤ 100·eps(FT)` for `FT ∈ {Float32, Float64}`.

### Commit 2 — adjoints of `_pre_advect_x_kernel!` / `_pre_advect_y_kernel!`

The forward `_pre_advect_y_kernel!` (LinRood.jl:364) is **not** an
upwind-face accumulation. It consumes:
- `rm[ii, jj, k]`, `m[ii, jj, k]` (cell-centered mass and air mass);
- `bm[i, j, k]`, `bm[i, j+1, k]` (signed mass-flux × dt at S and N
  cell faces, already on the meteo tape);
- `fy_face[i, j, k]`, `fy_face[i, j+1, k]` (face mixing ratios produced
  by `_ppm_y_face_kernel!` upstream — these are taped intermediates).

It writes
```
rm_new = rm[ii, jj, k] + bm[i, j, k] · fy_face[i, j, k]
                       − bm[i, j+1, k] · fy_face[i, j+1, k]
m_new  = m[ii, jj, k]  + bm[i, j, k] − bm[i, j+1, k]
q_i[ii, jj, k] = _safe_mixing_ratio(rm_new, m_new)
```
The output `q_i` is a smooth function of `(rm, m, fy_face)` only — `bm`
is fixed by the velocity tape. So the operator is **piecewise smooth and
linear in `rm`/`fy_face`, hyperbolic in `m_new`**. Its adjoint
distributes a single `lambda_q_i[ii, jj, k]` into four lambda fields:
```
inv_m_new = m_new > thresh ? 1/m_new : 0       # mirror _safe_mixing_ratio
lambda_rm[ii, jj, k]         += lambda_q_i · inv_m_new
lambda_fy_face[i,   j, k]    += lambda_q_i · bm[i, j, k]    · inv_m_new
lambda_fy_face[i, j+1, k]    += lambda_q_i · (−bm[i, j+1, k]) · inv_m_new
lambda_m[ii, jj, k]          += lambda_q_i · (−q_i[ii, jj, k]) · inv_m_new
```
The same shape holds for `_pre_advect_x_kernel!` with the X-direction
face indexing.

Implementation:
- New `_pre_advect_y_kernel_adjoint!` and `_pre_advect_x_kernel_adjoint!`
  in `linrood_adjoint_kernels.jl`. Each takes a tape-stored
  `q_i_out_tape` (or recomputed `m_new`, since `m_new` is determined by
  `(m, bm)` which are tape inputs) and the signed mass fluxes. The
  small-`m_new` zeroing must exactly mirror `_safe_mixing_ratio`
  (LinRood.jl) — i.e., the same `100·eps(FT)` threshold.
- These kernels do **not** reuse `_add_y_face_adjoint!(::UpwindScheme, …)`
  or any of the existing face helpers — the divergence-then-divide
  structure is different.
- Transposition tests in `test/test_linrood_kernel_adjoints.jl` with
  `rtol ≤ 100·eps(FT)`, plus one fixture with deliberately small `m_new`
  to confirm the zero-gradient guard.

### Commit 3 — adjoints of all four LinRood PPM face kernels (new derivation)

Forward kernels (LinRood.jl:241-348):
- `_ppm_y_face_kernel!`, `_ppm_x_face_kernel!` — read `(rm, m)`,
  compute `c_n = _safe_mixing_ratio(rm_n, m_n)` then run PPM
- `_ppm_y_face_from_q_kernel!`, `_ppm_x_face_from_q_kernel!` — read `q`
  directly, skip the safe-division step

All four share the downstream chain
`_ppm_edge_values → _apply_ord7_boundary → _apply_monotonicity →
_ppm_face_value` (LinRood.jl:200-231, 257-266). The LinRood
`_ppm_face_value` uses the **parabolic-integral form** `c + (1−α)(br − α·b0)`
on mixing ratios with `α = flux / m_donor` clamped via `max(m_donor,
100·eps)`. This differs from the limited-moment scaled-mass formulation
in `_ppm_monotone_face_coeffs` (Adjoints.jl:610-668), so neither that
function nor the helper chain it composes
(`_ppm_limit_profile_monotone_ad`, `_limited_moment_monotone_ad`) is
mathematically applicable here. Commit 3 derives the LinRood face
adjoint from scratch and does **not** reuse those helpers — only the
low-level `_d6_*` 6-cell tangent primitives (Adjoints.jl:538-557) and
the d6-AD version of `_ppm_edge_values` (re-derived as
`_linrood_ppm_edge_values_d6`) are shared building blocks.

Implementation:
- New helpers in `linrood_adjoint_kernels.jl`:
  - `_linrood_ppm_edge_values_d6` — d6-AD version of LinRood's
    `_ppm_edge_values`.
  - `_linrood_apply_monotonicity_d6(q_L, dq_L, c, dc, q_R, dq_R)` —
    d6-AD transpose of `_apply_monotonicity` (LinRood.jl:200): when
    `(q_R−c)(c−q_L) ≤ 0` it flattens both to `c`, propagating
    `dq_L_out = dq_R_out = dc`; otherwise pass-through.
  - `_linrood_ppm_face_value_d6(F, m_lo, m_hi, c_lo, dc_lo, c_hi, dc_hi,
    q_L_lo, dq_L_lo, q_R_lo, dq_R_lo, q_L_hi, dq_L_hi, q_R_hi, dq_R_hi)` —
    d6-AD transpose of `_ppm_face_value` (LinRood.jl:215-231). Returns
    `(face, dface)` where `dface` is a length-6 tangent w.r.t. the six
    q-stencil cells **only**. The α-denominator `m_donor` is held
    fixed; see the donor-mass note below.
  - `_linrood_ppm_face_from_q_d6(F, q_m3, q_m2, q_m1, q_0, q_p1, q_p2,
    m_l, m_r, face_idx, Nc, Val(ORD))` — composes the three helpers
    above on `(q, dq)` pairs with `dq_n = _d6_basis(FT, n, one(FT))`,
    returns the 6-tuple `∂f/∂q_n`.
  - `_linrood_ppm_face_from_rm_d6(F, rm_m3, …, m_m3, …, face_idx, Nc,
    Val(ORD))` — same downstream chain but with the safe-division step
    folded in. The chain-rule contribution from each cell is
    `dc_n = (1/m_n) · drm_n − (rm_n / m_n²) · dm_n`. Returns two
    6-tuples: `∂f/∂rm_n` and `∂f/∂m_n` (via the `c_n = rm_n / m_n`
    coupling). The donor-cell `m_l` (or `m_r`) additionally feeds
    `_ppm_face_value` directly through `α = F / m_donor`; this
    **additional** `∂f/∂m_donor` is added to the donor cell's `∂f/∂m`
    contribution. **The rm-input adjoint therefore accumulates into
    both `lambda_rm` and `lambda_m`**, with the donor cell's `lambda_m`
    receiving an extra term from the α-denominator. Commit 3 unit tests
    perturb both `rm` and `m` independently to exercise this.
- `_add_x/y_face_adjoint!(::LinRoodPPMScheme, …)` overloads in
  `Adjoints.jl` that wrap the helpers above and atomically accumulate
  into `lambda_q[ii, jj, k]` (for the `_from_q` variants) or
  `lambda_rm[ii, jj, k]` and `lambda_m[ii, jj, k]` (for the rm-input
  variants, including the donor-mass term).
- Per-kernel transposition tests in
  `test/test_linrood_kernel_adjoints.jl` covering both ORD=5 and ORD=7
  (the latter exercising `_apply_ord7_boundary`), with both
  `positive` and `negative` flux fixtures to exercise both `_ppm_face_value`
  branches and the `_apply_monotonicity` `(q_R-c)(c-q_L) ≤ 0` flatten
  branch. Tolerance `rtol ≤ 100·eps(FT)`. The rm-input tests perturb
  **both** `rm` and `m` to exercise the donor-mass derivative path.

### Commit 4 — composed 3-phase horizontal LinRood adjoint

- Implement `_adjoint_linrood_3phase_horizontal!(panel, …)` that applies the
  reverse of phases 3 → 2 → 1 in order, accumulating into `q_adj` using the
  per-kernel adjoints from Commits 1–3.
- Hook it into `_adjoint_scheme_sweep!(::LinRoodPPMScheme, …)`.
- Standalone advection-only FD test: a single substep of LinRood horizontal
  vs. forward-difference probe, no diffusion/convection. `rtol ≤ 3e-5` on a
  smooth tracer field, matching the existing scheme tests.

### Commit 5 — tape integration + Strang Z + multi-substep

LinRood's horizontal step is a 3-phase **unsplit** update with internal
halo/corner refreshes — it does not map onto the existing per-direction
`_CSSweepRecord`. Tape design:

- New record type `_CSLinRoodHorizRecord` per substep, holding:
  - input `rm`, `m` (panel arrays at the start of the substep, with
    halos);
  - intermediate `q_buf` array(s) at each phase boundary — specifically
    the state after phase-1 pre-advect-Y (the `q*` used by the phase-2
    X-PPM-from-q kernel) and the state after phase-2 pre-advect-X (the
    `q'` used by the phase-3 Y-PPM-from-q kernel). The forward path
    repeatedly refreshes halos and `copy_corners!(…, dir=…)` between
    phases (LinRood.jl:718, 740, 752, 767) — these halo/corner copies
    are part of the reconstruction the reverse pass must replay. The
    tape stores the **full haloed `q_buf` and haloed `rm`/`m` arrays**
    at each phase boundary, so the reverse pass does NOT need to
    re-run halo fills (which would require a separate adjoint halo
    record); it just reads what was forward-stored.
  - reference to the meteo-tape `am[p]`, `bm[p]` per panel for the
    substep;
  - `flux_scale`.
- No per-face limiter or `_ppm_face_value` decisions are stored — those
  are recomputed from the stored field state by the reverse-pass helpers
  from Commit 3.
- Z reverse path: `_strang_split_linrood_ppm_cs!` (LinRood.jl:921) calls
  `_sweep_z!` (`Operators.Advection._sweep_z_panel!`) with the same per-
  panel kernel the existing PPM(MonotoneLimiter) Strang path uses, so the
  Z record reuses `_CSSweepRecord` with `direction = :z, scheme =
  PPMScheme(MonotoneLimiter)` — verified in Commit 5 by a Z-only
  forward/reverse FD test before the horizontal record is added.
- Extend the reverse-loop dispatch (Adjoints.jl:2929-2954) to recognise
  `_CSLinRoodHorizRecord` and call `_adjoint_linrood_3phase_horizontal!`
  from Commit 4. Add a `_record_cs_adjoint_tape(…, scheme::LinRoodPPMScheme)`
  method dispatched off the new `CSAdjointLinRoodScheme` union member.
- Multi-substep FD test on the standalone advection pipeline.

### Commit 6 — end-to-end FD test with diffusion + convection

- Add `LinRoodPPMScheme()` to the scheme loop in
  `test/test_cs_ppm_adjoint_footprint.jl` (line ~432) for the advection-only,
  diffusion-on, TM5-on, and CMFMC-on testsets.
- Verify gradient FD agreement on the full
  `cs_surface_flux_4dvar` cost/gradient evaluator with named surface-flux
  windows.

### Commit 7 — documentation

- Update `docs/src/theory/adjoint_status.md` to list LinRood as supported.
- Update README adjoint claim if/where it implicitly excluded LinRood.
- Append a short "what's new" line to MEMORY.md and (if a per-commit ledger
  exists for this branch) the appropriate notes file.

## Reusable pieces (touched but not re-derived)

- `_vertical_diffusion_cs_single_adjoint_kernel!` (Adjoints.jl:1560) —
  Backward-Euler tridiagonal transpose. Decoupled from advection; reused
  unchanged via the existing `cs_surface_emission_footprint` composition.
- `_tm5_solve_column_vector_adjoint!` (Adjoints.jl:1824) and
  `_cmfmc_cs_panel_column_single_adjoint_kernel!` (Adjoints.jl:2034) —
  reused unchanged.
- `_d6_basis`, `_d6_add`, `_d6_sub`, `_d6_scale`, `_d6_zero`
  (Adjoints.jl:538-557) — the 6-cell forward-AD tangent type. Reused
  unchanged as the building block for the new LinRood face-coeff
  derivations in Commit 3. The higher-level helpers
  `_ppm_edge_value_ad`, `_ppm_limit_profile_monotone_ad`,
  `_limited_moment_monotone_ad` (Adjoints.jl:561-607) are NOT reused —
  see "Not reused" below.
- `_adjoint_scheme_sweep!(…, direction = :z, scheme =
  PPMScheme(MonotoneLimiter), …)` — reused for the LinRood vertical
  Strang sweep (Commit 5).

**Not reused** (intentionally re-derived rather than refactored):
- `_ppm_monotone_face_coeffs` (Adjoints.jl:610) and the three helpers
  it composes — `_ppm_edge_value_ad`, `_ppm_limit_profile_monotone_ad`,
  `_limited_moment_monotone_ad` (Adjoints.jl:561-607). These implement
  the limited-moment scaled-mass formulation used by the existing
  PPM(MonotoneLimiter) split-sweep code, which produces tracer-mass
  fluxes (`m₁ · (q_R − c₁)` etc.). The LinRood path uses
  `_apply_monotonicity` (LinRood.jl:200) + parabolic `_ppm_face_value`
  (LinRood.jl:215) on mixing ratios with the α-form donor-mass
  denominator, which is a different operator. Commit 3 ships fresh
  `_linrood_ppm_edge_values_d6`, `_linrood_apply_monotonicity_d6`,
  `_linrood_ppm_face_value_d6` helpers matching the LinRood forward
  chain exactly.
- `_add_x/y_face_adjoint!(::UpwindScheme, …)` — the LinRood pre-advect
  kernels are not donor-cell upwind face accumulations (Commit 2).

## Risks / things to verify

- LinRood's flux averaging means the same input cell `q` contributes to
  fluxes via three different paths (direct PPM in phase 3, plus phase-2
  cross fluxes through both `q*` and `q'`). The adjoint contributions from
  all three paths must accumulate into `q_adj`, **not** overwrite. A
  misplaced `=` instead of `+=` is the most likely bug — the transposition
  unit tests in Commits 1–3 catch it before composition lands.
- The pre-advect kernels couple velocity into `q*` and `q'`, which in turn
  feed the cross fluxes. For fixed velocities (control variables are
  surface fluxes, not winds) the adjoint w.r.t. winds is zero — confirm
  this is consistent with what `cs_surface_emission_footprint` actually
  controls.
- LinRood `_ppm_face_value` (LinRood.jl:215-231) is **already known** to
  differ from the existing `_ppm_monotone_face_coeffs` formulation
  (Adjoints.jl:610). Commit 3 ships a separate
  `_linrood_ppm_face_from_*_d6` helper. The Commit 3 transposition test
  compares the new helper's forward chain against the live
  `_ppm_face_value` call to guarantee the d6-AD matches the actual
  forward kernel.
- ORD=7 boundary modification (`_apply_ord7_boundary` LinRood.jl:186) is
  data-independent (depends on `face_idx`, `Nc`) so it does not introduce
  additional branch tape, but the reverse pass must apply the same
  branch when reconstructing the d6 tangents. Easy to forget — covered by
  the ORD=7 unit test in Commit 3.
- `_safe_mixing_ratio` guards against division by tiny `m`. The reverse
  pass must reproduce the same guard so that small-mass cells contribute
  zero gradient (matching the forward zeroing behavior). Test cases
  include a column with deliberately small `m` to exercise this.
- GPU portability: every adjoint kernel must compile under
  `KernelAbstractions` for the same backends as the forward path
  (CPU + CUDA). Tests should run the same problem through both.
- Tape memory: per LinRood horizontal substep we store **haloed**
  `rm`, `m`, `q*`, `q'` arrays. With `N = Nc + 2·Hp` per side, that is
  4 arrays × N × N × Nz × 6 panels. At C180 × Nz=72 × Float32 × Hp=3
  this is 4 × 186 × 186 × 72 × 6 × 4 B ≈ 240 MB per substep — enough
  that the test problem must stay at C24/C48 until Revolve checkpointing
  lands. Document the per-substep budget in
  `cs_surface_emission_footprint`'s docstring after Commit 5.

## Definition of done

- All six commits (1–6) land with green tests, including
  `test_cs_ppm_adjoint_footprint.jl` covering LinRood with diffusion + TM5
  + CMFMC.
- `cs_surface_flux_4dvar` returns gradients that match centered FD probes
  for at least one LinRood + diffusion + TM5-convection configuration at CS
  resolution C24, `rtol ≤ 1e-4`.
- Adjoint status doc reflects LinRood support.
- No regression in existing scheme adjoint tests.
