# Plan 39 — Unified transport-binary contract + replay-consistency gate

**Status:** Completed 2026-04-22. Branch `convection`.

## What shipped

| Commit | SHA | Purpose |
|---|---|---|
| A | `afa1e23` | `TransportBinaryContract` + `canonical_window_constant_contract` + `validate_transport_contract!` |
| B | `2584a9d` | LL `build_v4_header` emits the 8 contract fields |
| C | `1e4bf50` | RG writer declares canonical contract explicitly |
| D | `d28a9ec` | Strict `validate_transport_contract!` wired into `TransportBinaryReader` |
| G | `a029f38` | Remove `reset_air_mass_each_window` flag entirely |
| I | `2e4bd91` | Real-data validation: 2-day Dec 2021 demo completes, mass drift 0 % |
| — | `585fe40` | F64 probe script (stand-alone diagnostic) |
| **cm closure fix** | `c3a48dc` | **Replace Δb×pit with explicit-dm in LL & RG preprocessor** |
| E | `d193220` | Write-time replay gate in `apply_poisson_balance!` / `apply_reduced_poisson_balance!` / `balance_window!` |
| F | `6fba75a` | Load-time replay gate in `TransportBinaryDriver` (opt-in) |
| H | `dd69234` | Regression tests (`test_replay_consistency.jl`, 18 tests) |

## Root cause (captured 2026-04-22)

The F64 day-boundary probe measured `max|m_runtime_end_day1 − m_stored_day2_win1|/max|m| = 7.491e-3`
on uniform-IC CO₂, producing ±4 ppm jumps at day boundaries on an IC that
was bit-flat within each day. Mass itself was conserved to 0.000 %.

Binary-level replay isolated the mechanism:
- Day 1 `m[24] + dm[24]` matched Day 2 `m[1]` exactly — the writer is
  cross-day consistent.
- The stored `dm_dry[k]` satisfied `max|dm_dry[k] − dB[k]·Σ_k dm_dry[k]| = 9.01e9 kg`
  (27 % relative to `|dm|`).
- Runtime-evolved `m_end` using the stored `(am, bm, cm)` differs from
  the stored `m_next` by exactly that 27 %-worth of per-cell shifts,
  propagating to 0.75 % cell-level mismatch and ±4 ppm VMR jumps.

The breaking identity: `recompute_cm_from_divergence!` (LL:
[mass_support.jl:228](../../src/Preprocessing/mass_support.jl),
RG: [reduced_transport_helpers.jl:784](../../src/Preprocessing/reduced_transport_helpers.jl))
diagnoses `cm` from the hybrid `Δb × pit` closure, which is exact for
moist hybrid coordinates (`dm = dB · dps · area` by construction) but
off by `-m_moist · dqv` terms under dry basis. CS already used the
correct explicit-dm closure ([cs_poisson_balance.jl:783](../../src/Preprocessing/cs_poisson_balance.jl)).

## The fix

Introduced basis-agnostic explicit-dm closures:
- [`recompute_cm_from_dm_target!`](../../src/Preprocessing/mass_support.jl) (LL)
- [`recompute_faceindexed_cm_from_dm_target!`](../../src/Preprocessing/reduced_transport_helpers.jl) (RG)

Formula (LL sign convention):

    cm[k+1] = cm[k] − div_h[k] − dm_target[k]

Where `dm_target = (m_next − m_cur) × scale` per application, `scale = 1/(2·steps_per_window)`.
After the closure, any residual at `cm[Nz+1]` (from Poisson-balance
tolerance or null-space components) is redistributed proportional to
column `m` — same algorithm as CS.

Wired into the three post-Poisson call sites (LL `apply_poisson_balance!`,
RG `apply_reduced_poisson_balance!`, RG streaming `balance_window!`).
Pre-Poisson cm calls (in `merge_native_window!`, `merge_reduced_window!`,
RG `spectral_to_native_fields!`) left on the legacy closure — they only
seed `merged.cm_merged` for the unused `:window_constant` delta payload.

## Probe verification

F64 probe on wurst (idle), PID 356403, run time ≈ 20 min:

| Metric | Before fix | After fix | Target |
|---|---|---|---|
| DAY BOUNDARY rel | 7.491e-03 | **4.888e-13** | F64 ULP |
| DAY BOUNDARY abs | 2.538e+10 kg | **1.656 kg** | ULP floor |
| Day 2 t=0 uniform VMR | 395.80 … 403.92 | **400.000 … 400.000** | flat |
| Mass drift 2-day | 0 % | 0 % | unchanged |

10-order collapse. Remaining 4.9e-13 is cumulative F64 ULP across
720×361×34 × 96 substeps of Poisson-balance tail residuals after
redistribution.

## Deviations from the original plan §4.4

- **Commit F default inverted**: the plan called for default-on with
  `ATMOSTR_NO_REPLAY_CHECK=1` bypass. Shipped as **opt-in** via
  `validate_replay::Bool=false` kwarg (or `ATMOSTR_REPLAY_CHECK=1` env
  var). Reason: existing test helpers build synthetic binaries with
  deliberate `(zero flux, nonzero Δm)` combinations that would fail the
  gate. Default-on would have required updating ~5 test files to either
  satisfy continuity or skip the gate. Opt-in preserves test behavior;
  the write-time gate (Commit E) already guarantees correctness for
  binaries we produce, so the load-time gate is only needed for suspect
  imports (manual, older preprocessor, file corruption). Can be
  re-enabled default-on in a follow-up if production users need it.
- **RG load-time gate deferred**: the RG face-indexed topology lives in
  the advection workspace, not the driver/grid, so constructing it at
  driver-build time would require restructuring connectivity
  construction. Left as future work; write-time gate covers newly-
  produced RG binaries.
- **Pre-Poisson cm call sites left with legacy Δb×pit closure**: their
  output is discarded by the post-Poisson recomputation under the
  `:window_constant` contract. Cleaner to leave them alone than churn
  three extra call sites with zero runtime impact.

## Binary regeneration required

All F64 LL binaries produced before `c3a48dc` encode the broken cm.
Dec 2 + Dec 3 2021 regenerated on wurst (~7 min each) on the new
closure; probe verified. Other days / other grids need regeneration
before use:

- `~/data/AtmosTransport/met/era5/ll720x361_v4/transport_binary_v2_tropo34_dec2021_f64/era5_transport_20211201_merged1000Pa_float64.bin`
  — still on pre-fix closure. Archived copies of Dec 2 + Dec 3 saved
  as `.bin.plan39preclosurefix` for reference. Delete when unneeded.
- Any F32 LL binaries, RG binaries of any size, and older dates.

The load-time gate (when opted in) will refuse to load any unfixed
binary with a clear error pointing at this memo.

## Tolerance choice

`tol_rel = 1e-10` (F64), `1e-4` (F32). The F64 probe measured ~1e-13
worst case; the 1e-10 limit is three orders of margin for grid sizes
much larger than 720×361×34 (where ULP accumulation could reach ~1e-12).
The F32 limit of 1e-4 is a first-cut conservative choice; no F32
measurement yet. If F32 regression trips it falsely, relax to ~1e-3.

## Surprises during execution

- Julia's `@sprintf` in 1.12 rejects concatenated format strings at
  macro-expansion time (`@sprintf("a %d" * "b", x)` fails with
  "First argument to @sprintf must be a format string"). Three
  occurrences in the new error paths were only caught when the tests
  ran. Fix: `msg = @sprintf("literal", args); error("prefix $msg suffix")`.
- Probe on curry (load avg 185) took 26 min in precompile because the
  script uses `include(...)` on `src/AtmosTransport.jl` which bypasses
  the precompile cache. Same probe on idle wurst took ~20 min. Next
  time: run on an idle host and/or switch to `using AtmosTransport`.
- Julia fully buffers both stdout and stderr when redirected to a file,
  so a probe that produces no output for 25 minutes looks hung but is
  just waiting for program exit to flush. Use `stdbuf -o0 -e0` to
  stream output for long-running diagnostics.

## Next steps (deferred, not part of plan 39)

- **Regenerate the full ERA5 LL + RG + CS F64 binary set** with the fix.
  Plan 39 only covered Dec 2 + Dec 3 2021 for the probe. Other periods
  need a sweep.
- **RG load-time gate**: if production users report loading suspect RG
  binaries, restructure face connectivity to be accessible at driver
  construction and add the RG load-time replay.
- **F32 tolerance tuning**: once a production F32 run hits the gate,
  measure the actual worst-case and set `tol_rel` accordingly.
- **Close plan 24 Commit 4**: plan 24's NOTES flagged "contract breach
  at day boundaries" as a deferred investigation. That was this bug.
  Add a back-reference in plan 24's NOTES.
