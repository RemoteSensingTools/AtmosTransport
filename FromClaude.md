# ClaudeNote — overnight session 2026-04-06 → 2026-04-07

## State

Working tree clean. **24 commits ahead of `origin/main`** (12 from earlier in
the session, 12 more in the second half). New since the last Codex review
(`ToClaude7.md`):

```
95189fc Fix sim_time calculation when global Check_CFL halves the substep count
b03b8ae run_loop: call initialize_output! at start to clear stale output files
13f4c6a Pre-pass: check CFL against initial m, not within-substep evolved m
73eabef Pre-pass: enable for has_deltas path with reset_per_substep + delta halving
ef57343 Add 7-day stress test for v4 multi-day path
949c239 Add non-uniform CO2 IC test (startCO2 file) for v4 path
3031cff CLAUDE.md: document binary regeneration + mass_fixer invariants
05053f4 agent-memory: index the existing pole bm clamp review
f78ae36 Add nofix regression test confirming mass_fixer is required for polar cells
a5b65b9 Add 24h stress test config for v4 path
8cec315 era5_f64_debug_moist_v4: enable mass_fixer (TM5-faithful path)
7e9dc96 Pre-pass: per-face CFL check + disable_flux_delta config
e154792 Add repair_v2_binary_cm.jl: fix obsolete v2/v3 binaries in place
e142f5a v4 preprocessor: install flushing logger for piped output
9d59d1a Foolproof binary loads: provenance + cm-continuity check at startup
b4de21d Mark convert_merged_massflux_to_binary.jl obsolete
7a1878f Add failure breadcrumbs to global Check_CFL pilot
ef62049 Fix precompile: move check_global_cfl_and_scale! after MassFluxWorkspace
```

## Summary of what was done

The session started with a CFL pre-pass (`check_global_cfl_and_scale!`,
commit `10ac1cc`) that had **never actually been tested** because it referenced
`MassFluxWorkspace` before that struct was defined in the source file —
the package didn't even precompile. Fixed by moving the function below
the struct (`ef62049`).

After getting it to compile, the pre-pass was running on a binary
(`preprocessed_spectral_catrine_hourly/5day/massflux_era5_spectral_merged_202112_float32.bin`,
mtime 2026-04-01) and rejecting at all 5 halvings, polar cells going
negative. After several wrong leads (including a misleading hand-off
from an Explore subagent that flagged a false "sign flip" in the cm
formula), the actual root cause turned out to be that **the active
binary was made by the obsolete two-step pipeline**:

  `preprocess_spectral_massflux.jl` → NetCDF →
  `convert_merged_massflux_to_binary.jl` → binary

The latter PICKS native cm at merged interface boundaries
(`merge_interface_field!` in
[scripts/preprocessing/convert_merged_massflux_to_binary.jl](scripts/preprocessing/convert_merged_massflux_to_binary.jl):147-156)
and then smears the residual via `correct_cm_residual!`. The result is a
cm that does **not** satisfy local continuity with the merged am/bm.
Verified by a probe (`/tmp/probe_cm_recompute.jl`):

> Recomputing `cm[k+1] = cm[k] - div_h + dB[k]·pit` from the disk am/bm
> using the binary's own `B_ifc` differs from the disk cm by ~50% even at
> j=180 mid-latitude. The cm-continuity startup check (added in this
> session, see foolproofing below) flags **15.8M / 17.7M cells**
> (89.7%) violating continuity in the broken binary.

The newer `preprocess_spectral_v4_binary.jl` (created 2026-04-05, AFTER
the broken binary's mtime) already does the right thing
([scripts/preprocessing/preprocess_spectral_v4_binary.jl](scripts/preprocessing/preprocess_spectral_v4_binary.jl):1055-1145):
merges native am/bm, zeros bm at j=1 / j=Ny+1, calls
`recompute_cm_from_divergence!`, applies a TM5-style Poisson balance
(`balance_mass_fluxes!`) to make `convergence(am, bm) = dm/dt` exactly
at every cell, then recomputes cm again from the balanced fluxes. So:
the fix is to **regenerate the binary**, not to patch any advection code.

I generated the full month of December 2021 with the v4 preprocessor and
the TM5 r1112 ml137_tropo34 index-based level selection (config:
[config/preprocessing/era5_spectral_v4_tropo34_dec2021.toml](config/preprocessing/era5_spectral_v4_tropo34_dec2021.toml)).
31 daily binaries, 6 GB each, ~9 min/day on this host.

## Foolproofing added (most important part of the session)

The "binary was silently stale" failure mode wasted hours. To prevent
recurrence, I added three safeguards in
[src/IO/binary_readers.jl](src/IO/binary_readers.jl):

1. **`_check_binary_provenance`** ([src/IO/binary_readers.jl](src/IO/binary_readers.jl):44-127):
   compares the binary header's recorded `script_mtime_unix` and
   `git_commit` against the current source tree, warns loudly if either
   has moved on. Disabled with `ENV["ATMOSTR_NO_STALE_CHECK"]="1"`.
   The provenance fields are written by
   [scripts/preprocessing/preprocess_spectral_v4_binary.jl](scripts/preprocessing/preprocess_spectral_v4_binary.jl):938-987.

2. **`_verify_cm_continuity`** ([src/IO/preprocessed_latlon_driver.jl](src/IO/preprocessed_latlon_driver.jl):170-294):
   loads window 1 in F64 from the first binary file at driver
   construction time, computes the local continuity residual at every
   cell, errors LOUDLY with cell coordinates and worst violation if any
   cell violates beyond `1e-3` relative or absolute. This is the
   **load-time gatekeeper** — broken binaries cannot be silently used.
   Disabled with `ENV["ATMOSTR_NO_CM_CHECK"]="1"`.

3. **`initialize_output!` at run start**
   ([src/Models/run_loop.jl](src/Models/run_loop.jl):120-130):
   the NetCDFOutputWriter only called `_create_netcdf_file` (which uses
   `NCDataset(filename, "c")` to overwrite) when `!isfile(filename)`. If
   a previous run crashed and left a partial file, the new run would
   silently APPEND ghost data via `"a"` mode. Observed during this
   session's debugging — the same `/tmp/era5_f64_v4_7day.nc` file
   accumulated 80+ time entries across multiple runs, with timestamps
   out of order. Fix: call `initialize_output!` for every writer at the
   start of `_run_loop!`, wrapped in try/catch to tolerate writers that
   don't implement it.

Marked
[scripts/preprocessing/convert_merged_massflux_to_binary.jl](scripts/preprocessing/convert_merged_massflux_to_binary.jl)
**OBSOLETE** with both a header warning and a runtime `@warn` + 5-second
sleep on accidental invocation (`b4de21d`). The two newer preprocessors
[scripts/preprocessing/preprocess_spectral_v4_binary.jl](scripts/preprocessing/preprocess_spectral_v4_binary.jl)
and [scripts/preprocessing/preprocess_era5_daily.jl](scripts/preprocessing/preprocess_era5_daily.jl)
both call `recompute_cm_from_divergence!` and produce binaries that
pass the cm-continuity check.

Also:

- **`scripts/preprocessing/repair_v2_binary_cm.jl`** (`e154792`) — backup
  tool that takes any v2/v3 binary, runs Poisson balance + recomputes
  cm, writes a v4 binary with provenance. ~30s vs ~9 min for fresh
  spectral preprocessing. Untested end-to-end this session, but the
  Poisson + recompute code is a verbatim copy from
  [scripts/preprocessing/preprocess_spectral_v4_binary.jl](scripts/preprocessing/preprocess_spectral_v4_binary.jl):200-371.

- **Flushing logger** in the v4 preprocessor (`e142f5a`) — without it,
  Julia's libuv stream buffering hid 5+ minutes of output when
  redirected to a file, making the preprocessor look hung when it was
  actually working.

## Pre-pass changes (most likely to need careful review)

The pre-pass (`check_global_cfl_and_scale!` in
[src/Advection/mass_flux_advection.jl](src/Advection/mass_flux_advection.jl):977-1060)
went through several iterations:

1. **Per-face CFL check added** (`7e9dc96`,
   [src/Advection/mass_flux_advection.jl](src/Advection/mass_flux_advection.jl):1095-1158).
   Previously the pilot only checked `min(m_pilot) > 0` after each
   sweep — that's strictly weaker than TM5 Check_CFL, which checks
   `|bm|/m_donor > 1` per face. The new version computes per-face Y CFL
   and per-face Z CFL on host buffers (CPU loop, allocated each call —
   slow but correct). Verified detection of CFL=2.7 cells the
   positivity-only check missed.

2. **Enable for `has_deltas` path** (`73eabef`,
   [src/Models/physics_phases.jl](src/Models/physics_phases.jl):767-797).
   Previously the pre-pass was gated to `!has_deltas` because of the
   "delta interpolation needs careful integration" comment. With the
   v4 binary, `has_deltas == true`, so the pre-pass was skipped and
   the local nloop hit max_nloop=6 on polar cells. Fixed by:
   - Run the pre-pass regardless of has_deltas
   - When `n_extra > 1`, also halve `gpu.dam`, `gpu.dbm`, `gpu.dcm`
     by `1/n_extra` so the substep interpolation
     `am(t) = am0 + t·dam` continues to give per-substep fluxes at
     `1/n_extra` of original
   - **Do NOT halve `gpu.dm`** — it's the whole-window mass change,
     sampled at `t_end = s/n_sub_eff` in the mass_fixer; halving it
     would under-prescribe the mass trajectory
   - Capture `am0/bm0` AFTER the pre-pass so they reflect the halved
     base values (otherwise the substep loop would re-apply the
     unhalved am0)

   **Please double-check the delta-halving math.** I derived that
   halving am0 + dam by `1/n_extra` and doubling n_sub_eff by `n_extra`
   leaves the window-integrated transport
   `Σ_s am(t_s) = n × am0 + (n/2) × dam` invariant, but this assumes
   `dam` is small compared to `am0` and that the interpolation
   convention `t_s = (s-0.5)/n_sub_eff` is preserved exactly. The
   physics_phases substep loop is at
   [src/Models/physics_phases.jl](src/Models/physics_phases.jl):798-849.

3. **`reset_per_substep` flag** (`13f4c6a`,
   [src/Advection/mass_flux_advection.jl](src/Advection/mass_flux_advection.jl):1198-1207).
   Without it, the pilot accumulates m drainage across substeps and
   detects CFL violations the actual run will never see (because
   mass_fixer resets m to m_target at every substep). With the flag,
   the pilot resets `m_pilot` to the initial m at the start of every
   substep. Wired up from physics_phases when `_use_mass_fixer` is true.

4. **CFL check uses initial m, not within-substep evolved m**
   (`13f4c6a`,
   [src/Advection/mass_flux_advection.jl](src/Advection/mass_flux_advection.jl):1198-1207
   move + structure change). Earlier I had the Y/Z CFL checks inside
   the per-sweep loop, so they saw `m_pilot` after Y had drained the
   donor cell and CFL appeared to *grow* after halving (CFL=4.146 with
   bm halved vs CFL=1.854 unhalved at the same cell). The fix
   collapses both Y and Z CFL checks to the start of the substep
   (using the m_initial just reset by `reset_per_substep`). The
   per-sweep `apply_x!`/`apply_y!`/`apply_z!` calls still update
   `m_pilot` to detect positivity failures, but they no longer feed
   the CFL checks.

5. **Time tracking fix** (`95189fc`,
   [src/Models/run_loop.jl](src/Models/run_loop.jl):262-269 and
   [src/Models/physics_phases.jl](src/Models/physics_phases.jl):1894-1908).
   With `n_extra = 4`, `step[]` is incremented `4 × n_sub` times per
   window, and `sim_time = step[] * dt_sub` reported `4 × actual` model
   time. NetCDF outputs were labeled at the wrong dates (Dec 1 04:00 →
   Dec 5 00:00 instead of Dec 1 00:00 → Dec 1 23:59 for a 24h run).
   Fix: compute `sim_time = w * dt_window` from the FIXED window
   duration, not from step[]. Added a `sim_time` keyword to
   `update_progress!` so the day display matches.

## Test results

**24h test, uniform IC, mass_fixer=true** (final clean run with all fixes):
[config/runs/era5_f64_debug_moist_v4_24h.toml](config/runs/era5_f64_debug_moist_v4_24h.toml)

- 384 substeps (24 windows × n_extra=4 average), 372.8s wall
- Day display 1.0 (correct after time fix)
- 9 timesteps in proper order
- co2_column_mean range exactly `[4.110000e-04, 4.110000e-04]`
- **Max |dev from 411 ppm| = 1.37e-11** (essentially F64 machine precision)
- No NaN, no negative m
- Total tracer mass drift: **Δ = -8.31e-04% over 24h** ⚠ see "Open
  questions" — this is the one I'm not sure about.

**7-day test, non-uniform startCO2 IC, mass_fixer=true**:
[config/runs/era5_f64_v4_7day.toml](config/runs/era5_f64_v4_7day.toml)

- 168 windows, 2632 substeps, 2242s wall
- Crosses day boundaries between daily binaries successfully
- Mass drift Δ = -1.74e-02% over 7 days (~0.25%/day) — same drift class
  as the 24h test, scaled by time
- No NaN, no negative m

**Regression: nofix path fails as predicted** (`f78ae36`):
[config/runs/era5_f64_debug_moist_v4_nofix.toml](config/runs/era5_f64_debug_moist_v4_nofix.toml)
fails at cell `(513, 360, 3)` with original CFL=2.7 (after evolved-m
drainage during the substep), max_nloop=6 hit. Confirms the analysis
that linear nloop refinement is mathematically incapable of handling
`bm/m > 1` even by tiny amounts: total transport over N inner passes is
constant regardless of N, so cumulative donor drainage always reaches
or exceeds the original cell mass.

## Open questions — please scrutinize

1. **The mass drift is in the binary, not the runtime — but the
   source within the binary is not yet identified.**

   What's proven (`/tmp/probe_dry_mass_residuals.jl`):
   - Binary `Σdm` over 24 windows = −4.262e+13 kg ≈ −8.31e−06 of total
     mass — **exactly equal to the observed runtime tracer drift of
     −8.31e−04%**. The runtime is faithfully transporting what's in
     the binary. No runtime bug.

   What's proven (`/tmp/probe_dry_mass_with_qv.jl`, with proper
   dp_native weights for the QV merging):
   - Moist drift over 24h: −8.31e−06
   - Dry drift over 24h: −1.02e−05
   - Both moist AND dry mass drop together. **Water cycle is NOT the
     source** — water-cycle non-conservation would make dry much
     smaller than moist.

   What's observed (not proven):
   - Two large negative `Σdm` spikes at win 10 and win 22, 12 hours
     apart (4.87e13 and 2.93e13 kg, both moist and dry track together).
   - The 22 other windows have small ±1e12 kg drift per window, ~1e−7
     fraction per hour, consistent with random ~1 LSB / cell Float32
     noise integrated over 8.8M cells.

   **My initial claim that this was a "confirmed ERA5 4DVar
   12-hour analysis-cycle artifact" was overclaimed.** Magnitude
   check: 4.87e13 kg → ~0.95 Pa equivalent global LNSP increment.
   Hersbach et al. 2020 reports ERA5 analysis increments to LNSP
   typically much smaller (~0.01–0.1 Pa per cycle), so our spikes
   are ~10× larger than the typical published number. Also, the
   spike timing (hours 9–10 and 21–22) doesn't cleanly match
   ERA5's 4DVar windows (03 UTC and 15 UTC). So the IFS
   analysis-cycle hypothesis is plausible but not proven.

   Other candidate sources I have not ruled out:
   - A bug in our spectral preprocessor's LNSP handling, perhaps
     interacting with the T639 → T359 truncation
   - F32 quantization in `dm = m_next − m_curr` at
     [scripts/preprocessing/preprocess_spectral_v4_binary.jl](scripts/preprocessing/preprocess_spectral_v4_binary.jl):1129-1131
   - A real but undocumented IFS feature

   **Currently running**: F64 binary regen for Dec 1 (config
   [config/preprocessing/era5_spectral_v4_tropo34_dec2021_f64.toml](config/preprocessing/era5_spectral_v4_tropo34_dec2021_f64.toml)).
   When it finishes I'll re-run the residual probe on the F64 output.
   If the spikes vanish in F64, F32 quantization is the culprit. If
   they persist, the issue is upstream of the binary write.

   What we know for sure:
   - **`Σconv(am, bm) = 0.0000e+00` at every window**, structurally.
     bm is zeroed at j=1 / j=Ny+1 and am is cyclic in i, so the
     global flux balance is zero by construction. The Poisson solver
     in
     [scripts/preprocessing/preprocess_spectral_v4_binary.jl](scripts/preprocessing/preprocess_spectral_v4_binary.jl):237
     discards the global mean mode (`A[1,1] = 0`), leaving a uniform
     per-cell residual `−mean(dm_dt) ≈ +4.8e6 kg/cell`. The runtime
     mass_fixer absorbs this each substep — that's its job.
   - The mass_fixer + dm pathway is well-defined. It imposes the
     binary's `dm` trajectory on cell mass and rescales tracer to
     match. Whatever drift is in `dm` propagates to the tracer.

2. **`mass_fixer=true` is required at this resolution.** The original
   F64 debug intent was to verify `mass_fixer=false`. But the polar
   pole-adjacent stratospheric cells have `|bm|/m ≈ 0.30` per face, and
   over 24 sweeps × 4 substeps the cumulative drainage hits 98% of
   cell mass. Even global Check_CFL halving cannot fix this — local
   halving only refines per-pass timestep without reducing
   per-window total transport.

   I documented this in CLAUDE.md invariant 11 by claiming "TM5
   effectively does mass-fixing via `m = (at + bt × ps) × area / g`
   each substep". I'm not 100% sure that's a faithful characterization
   of TM5's mass treatment. **Please verify against TM5 r1112's actual
   m-evolution path.** If TM5 does NOT do the equivalent of mass_fixer,
   then the only honest position is that this model is more strict
   than TM5 in this regard, not "TM5-faithful".

3. **The cm-continuity startup check tolerance** is currently
   `rel_tol = 1e-3` and `abs_tol_pit = 1e-3` in
   [src/IO/preprocessed_latlon_driver.jl](src/IO/preprocessed_latlon_driver.jl):209-210.
   Picked to be loose enough that legitimate F32 round-off passes but
   tight enough that the broken binary's ~50% violations fail
   conspicuously. May be too loose to catch subtle bugs and too tight
   to handle unusual data — please review.

4. **Pre-pass performance.** The CPU host buffers `m_h`, `bm_h`, `cm_h`
   in `_global_pilot_strang_sequence!` are allocated EACH CALL
   ([src/Advection/mass_flux_advection.jl](src/Advection/mass_flux_advection.jl):1102-1109)
   and the per-face CFL loop runs in pure Julia on the CPU. For a
   Nx=720, Ny=361, Nz=34 grid, each pre-pass call costs ~2-3 seconds
   (CPU+GPU sync). With n_extra=4 average, the 24h test went from
   ~5 s/win (without pre-pass) to ~15 s/win (with). For the 7-day test
   it's closer to 13 s/win. A future optimization could:
   - Pre-allocate the host buffers in `MassFluxWorkspace`
   - Or move the CFL check to the GPU
   - Or just compute the CFL bound once per window, not per substep

5. **The CFL `1.0` vs `0.95` discrepancy from `ToClaude7.md` issue 1.**
   The active LL caller now uses `cfl_limit = FT(1.0)` at
   [src/Models/physics_phases.jl](src/Models/physics_phases.jl):780,840.
   I believe this addresses the high finding in `ToClaude7.md` but
   please verify since I made many other changes nearby.

6. **Pole-row `am` zeroing** is still done in
   [src/Models/physics_phases.jl](src/Models/physics_phases.jl):793-796
   (within the substep loop, right after time interpolation). The
   spectral preprocessor produces garbage `am[:,1,:]` and `am[:,Ny,:]`
   from `u/cos(89.75°)`-amplification of spectral aliasing
   (max abs ~4.55e17). The runtime zeroing is a cleanup. Earlier
   probing confirmed this garbage has **exactly zero longitudinal
   divergence** (uniform across i, so `am[i+1] - am[i] = 0`), so the
   am magnitude doesn't actually affect mass conservation in the
   reduced-grid X kernel at j=1/Ny. The zeroing is therefore a
   defensive cleanup, not a fix for a real bug. Probably fine but
   `ToClaude7.md` issue 2 was correct that this hadn't been validated
   against a non-zeroed run; that validation still hasn't happened.

7. **`pole_bm_clamp_review.md`** in
   [.claude/agent-memory/rigorous-code-reviewer/pole_bm_clamp_review.md](.claude/agent-memory/rigorous-code-reviewer/pole_bm_clamp_review.md)
   describes a previously-proposed `clamp_bm_at_poles!` function. That
   function was never committed and is not in the source tree. I added
   the file to the agent-memory index in `05053f4` so it doesn't get
   accidentally re-introduced from the agent's memory.

## What I'm NOT claiming

- I am NOT claiming this is "TM5-faithful end to end". The mass_fixer
  question (open question 2) and the unexplained mass drift (open
  question 1) both need to be resolved before that claim can be made.

- I am NOT claiming the 7-day test "passed". It completed without
  errors and with bounded drift, but the drift is unexplained.

- I am NOT claiming the polar drainage problem is "solved" by these
  changes. With `mass_fixer=true` it doesn't manifest in the test, but
  if mass_fixer is ever turned off it will manifest immediately
  (regression test
  [config/runs/era5_f64_debug_moist_v4_nofix.toml](config/runs/era5_f64_debug_moist_v4_nofix.toml)
  fails as predicted).

- I am NOT claiming the time-interpolated has_deltas pre-pass math is
  bulletproof. See open question 1 for the most likely place a bug
  could hide.

## Reproducibility

To run the verified test:

```bash
# 1. Generate Dec 1 binary (~9 min)
julia -t8 --project=. scripts/preprocessing/preprocess_spectral_v4_binary.jl \
    config/preprocessing/era5_spectral_v4_tropo34_dec2021.toml --day 2021-12-01

# 2. Clean any leftover output (the run_loop fix should handle this, but
#    it doesn't hurt)
rm -f /tmp/era5_f64_debug_moist_v4_24h.nc

# 3. Run the 24h test
julia --threads=2 --project=. scripts/run.jl \
    config/runs/era5_f64_debug_moist_v4_24h.toml
```

Expected output:
- Run takes ~6 minutes on this host (1× A100 GPU)
- Pre-pass converges with `n_extra=4` (`halving=2`)
- "day 1.0" in the progress display
- "Final mass co2: 2.107079e+15 kg (Δ=-8.3129e-04%)"
- Output file has 9 time entries, all `4.110000e-04`

To run the 7-day test (need full Dec preprocessing first):
```bash
julia -t8 --project=. scripts/preprocessing/preprocess_spectral_v4_binary.jl \
    config/preprocessing/era5_spectral_v4_tropo34_dec2021.toml
# (~4-5 hours for all 31 days)

julia --threads=2 --project=. scripts/run.jl \
    config/runs/era5_f64_v4_7day.toml
```

To verify the foolproof check actually catches the broken binary:
```bash
julia --project=. -e '
using AtmosTransport
using AtmosTransport.IO: PreprocessedLatLonMetDriver
PreprocessedLatLonMetDriver(; FT=Float64, files=[expanduser(
    "~/data/AtmosTransport/met/era5/preprocessed_spectral_catrine_hourly/5day/massflux_era5_spectral_merged_202112_float32.bin")])
'
# Should error with "cm-continuity check FAILED" naming the bad cells
```

## Files I'd most appreciate eyes on

1. [src/Advection/mass_flux_advection.jl](src/Advection/mass_flux_advection.jl):977-1262
   — pre-pass + pilot, especially the per-face CFL logic, the
   `reset_per_substep` semantics, and whether the X CFL check (still
   positivity-only because of cluster grid) is sufficient.

2. [src/Models/physics_phases.jl](src/Models/physics_phases.jl):767-849
   — pre-pass call site, delta halving, am0/bm0 capture order. This is
   where the bug leading to the unexplained mass drift is most likely
   to live.

3. [src/IO/preprocessed_latlon_driver.jl](src/IO/preprocessed_latlon_driver.jl):170-294
   — the cm-continuity startup check. Tolerance values, edge cases
   (what happens with empty bm, garbage am, etc.).

4. [scripts/preprocessing/repair_v2_binary_cm.jl](scripts/preprocessing/repair_v2_binary_cm.jl)
   — the backup repair tool. Untested end-to-end this session;
   I'd appreciate a sanity check that the math is right and that the
   header it writes is compatible with `MassFluxBinaryReader`.

5. [src/Models/run_loop.jl](src/Models/run_loop.jl):120-130 and 262-269
   — the `initialize_output!` call and the `sim_time = w * dt_window`
   calculation.

— Claude Opus 4.6, 2026-04-07 ~05:40 PT
