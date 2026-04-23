# Dry-Basis cm Closure Bug (2026-04-22)

## Summary

Transport-binary preprocessor (LL + RG paths) diagnosed `cm` from a
**hybrid Δb × pit closure** that is only exact under moist hybrid
coordinates. Applied to **dry-basis** fluxes it produced a `cm` that
did not integrate with the stored `(am, bm)` to the stored `m_next` —
a stored-endpoint inconsistency of ~27 % per level in
`dm_dry[k]` vs `dB[k] × Σ_k dm_dry[k]`, translating to a ~0.75 %
per-cell runtime `m` mismatch at day boundaries and ±4 ppm VMR jumps
on uniform-IC tracers.

Mass itself was conserved (the error is just per-level redistribution),
so global mass diagnostics were clean but day-boundary tracer fields
jumped whenever a fresh driver window replaced `state.air_mass`.

Found by the plan-39 F64 day-boundary probe
(`scripts/probe_f64_day_boundary.jl`). Fixed by switching to the
explicit-dm closure that CS already used.

## The broken closure

Both LL and RG preprocessor paths called the same shape of closure after
Poisson-balancing the horizontal fluxes. Pseudocode (LL sign convention):

    pit = Σ_k ((am[i+1] - am[i]) + (bm[j+1] - bm[j]))
    cm[k+1] = cm[k] − div_h[k] + dB[k] × pit                    (1)

Equation (1) implicitly enforces `dm[k] = −dB[k] × pit` per
application. Under moist hybrid pressure coordinates, `dp[k] = dA[k] +
dB[k] × dps`, and the total-column `dm = area × dps`, so indeed
`dm_moist[k] = dB[k] × total_column_dm`. Exact by construction.

Under dry basis, `m_dry[k] = m_moist[k] × (1 − qv[k])`, and therefore

    dm_dry[k] = dm_moist[k] × (1 − qv[k])  −  m_moist[k] × d(qv[k])

The second term, `−m_moist · dqv`, varies independently per level. So
`dm_dry[k] ≠ dB[k] × Σ_k dm_dry[k]`. Equation (1) is wrong.

### Quantitative measurement

On the actual Dec 2 2021 F64 binary:

    max|dm_dry − dB × Σ_k dm_dry| = 9.01 × 10⁹ kg
    max|…| / max|dm_dry|          = 0.272   (27 %)

Per-window worst-case endpoint miss: 3.53 × 10⁻³ at window 22. Across
24 windows + the day-boundary the per-cell mismatch reached
7.491 × 10⁻³. Probe replay with a pure `dB × pit` cm integration
reproduced this number exactly.

## The fix

Explicit per-level dm target closure, matching the CS path at
[`cs_poisson_balance.jl:783-807`](../src/Preprocessing/cs_poisson_balance.jl):

    cm[k+1] = cm[k] − div_h[k] − dm_target[k]                   (2)

where `dm_target[k] = (m_next[k] − m_cur[k]) × 1/(2·steps_per_window)`
is the same per-application tendency used by the Poisson horizontal-flux
balance upstream. Any residual at `cm[Nz+1]` (from Poisson-balance
tolerance or null-space components) is redistributed proportional to
column `m` so `cm[Nz+1] = 0` exactly.

Basis-agnostic — valid for both moist and dry binaries.

New entry points:
- [`recompute_cm_from_dm_target!`](../src/Preprocessing/mass_support.jl) (LL)
- [`recompute_faceindexed_cm_from_dm_target!`](../src/Preprocessing/reduced_transport_helpers.jl) (RG)

Wired into three post-Poisson call sites:
- LL: `apply_poisson_balance!` in
  [`binary_pipeline.jl`](../src/Preprocessing/binary_pipeline.jl)
- RG batched: `apply_reduced_poisson_balance!` in
  [`reduced_transport_helpers.jl`](../src/Preprocessing/reduced_transport_helpers.jl)
- RG streaming: `balance_window!` in
  [`reduced_transport_helpers.jl`](../src/Preprocessing/reduced_transport_helpers.jl)

The legacy `recompute_cm_from_divergence!` /
`recompute_faceindexed_cm_from_divergence!` functions remain in place
for pre-Poisson staging / on-the-fly synthesis paths whose `cm`
output is subsequently discarded under the `:window_constant` contract.
Their docstrings now mark them deprecated for correctness-critical use.

## Validation

F64 day-boundary probe on regenerated Dec 2 + Dec 3 2021 binaries,
before vs after fix:

| Metric | Before fix | After fix | Target |
|---|---|---|---|
| `max\|m_runtime − m_stored\| / max\|m\|` at day boundary | 7.491 × 10⁻³ | **4.888 × 10⁻¹³** | F64 ULP |
| Day 2 t=0 uniform-IC VMR range | 395.80 … 403.92 ppm | **400.000 … 400.000 ppm** | flat |
| Variable-IC below-min deviation at t=0 | −3.4 ppm | −3.9 × 10⁻⁹ ppm | sub-ULP |
| Global mass drift, 2 days | 0 % | 0 % | — |

10-order collapse. Residual 4.9 × 10⁻¹³ is cumulative F64 ULP over
720 × 361 × 34 × 96 substeps of Poisson-balance tail residuals after
redistribution.

## Gates that would have caught this

Three gates are now in place:

1. **Write-time replay gate (plan 39 Commit E)** — inside
   `apply_poisson_balance!` and RG equivalents. Asserts stored
   `(am, bm, cm)` integrate to stored `m_next` to within `tol_rel =
   1e-10` (F64) / `1e-4` (F32). Errors loudly with the worst cell
   index. Runs unconditionally; bypass with
   `ENV["ATMOSTR_NO_WRITE_REPLAY_CHECK"]="1"`.

2. **Load-time replay gate (plan 39 Commit F)** — inside
   `TransportBinaryDriver` constructor, LL-only, opt-in via
   `validate_replay=true` kwarg or `ENV["ATMOSTR_REPLAY_CHECK"]="1"`.
   For suspect binaries (manual imports, older preprocessor versions,
   file corruption). Bypass an enabled gate with
   `ENV["ATMOSTR_NO_REPLAY_CHECK"]="1"`. RG load-time gate deferred —
   RG face connectivity currently lives in the advection workspace,
   not the driver/grid; see plan 39 NOTES.

3. **Regression tests (plan 39 Commit H)** —
   [`test/test_replay_consistency.jl`](../test/test_replay_consistency.jl),
   18 tests covering continuity-consistent tuples, deliberately broken
   cm, deliberately broken horizontal flux, and the zero-flux /
   nonzero-dm pattern used by legacy test helpers.

## Binary regeneration required

All LL and RG binaries produced before commit `c3a48dc` encode the
broken `cm`. Regenerate via
`scripts/preprocessing/preprocess_transport_binary.jl` with the
relevant TOML. The Dec 2 + Dec 3 2021 F64 `ll720x361_v4` binaries were
regenerated immediately for the probe. All other dates / grids /
precisions need a sweep. Pre-fix copies of Dec 2 + Dec 3 saved as
`*.bin.plan39preclosurefix` under
`~/data/AtmosTransport/met/era5/ll720x361_v4/transport_binary_v2_tropo34_dec2021_f64/`
for reference; delete when not needed.

The opt-in load-time gate is the fastest way to audit existing binary
sets — enable and watch it reject each unfixed binary with a specific
window + cell diagnostic.

## Attribution

Root cause identified by Codex (GPT) via binary-level replay: measured
the per-cell `Δb × pit` vs stored-`dm` mismatch and reconstructed the
exact 7.491 × 10⁻³ boundary miss from a pure-`dB × pit` forward
integration of the binary.

Diagnosis chain on the other side: plan-39 F64 probe established the
symptom was persistent under F64 (ruling out F32 precision) and
quantified it; Codex's replay identified the math.

## See also

- Plan 39 NOTES: [`docs/plans/39_TRANSPORT_CONTRACT/NOTES.md`](plans/39_TRANSPORT_CONTRACT/NOTES.md)
- CLAUDE.md invariant 10 (rewritten 2026-04-22)
- Plan 24 Commit 4 NOTES — cross-references this bug (the "contract breach at
  day boundaries" deferred investigation closes here)
- Probe script: [`scripts/probe_f64_day_boundary.jl`](../scripts/probe_f64_day_boundary.jl)
