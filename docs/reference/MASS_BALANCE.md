# Mass Balance: the Float32 Conservation Story

AtmosTransport runs production tracer transport in **Float32 on GPU** and
closes tracer mass budgets to the 0.01–0.1 %-of-emission level — to our
knowledge the first offline CTM to demonstrate working F32 conservation at
this standard (global CTMs conventionally run Float64 partly *because* of
conservation). This page records how mass balance is verified, every
conservation issue found and fixed (December-2021 campaign, C180, 4 tracers),
the measured floors per configuration, and when Float64 is still the right
tool.

## How to verify mass balance (the rules)

Hard-won methodology — violating any of these produced false conclusions at
least once during the 2026-06 campaign:

1. **Use the model's own applied emission, never a hardcoded inventory
   rate.** The runner logs
   `Surface source <tracer> total model-storage rate: <R> kg_air_equiv/s`.
   Working in storage units, `deficit = 1 − Δ(Σ vmr·air_mass) / (R·Δt)` —
   the molar-mass factor cancels, so no unit assumption can corrupt the
   check. (A double-applied EDGAR scale factor once fabricated a +1.17-point
   SF6 "deficit".)
2. **Use the exact F64 total, not an integral of the saved field.** Every
   snapshot file carries `<tracer>_total_mass` (Float64, dims `(time,)`),
   computed from the model state at capture. The spatial field is written at
   the output `float_type` (F32) and — for reference-state tracers — is an
   F32 *reconstruction* whose integral is polluted at the background-rounding
   scale. The F64 variable is the authoritative budget quantity.
3. **A no-flux run is the cleanest conservation test.** Real IC, zero
   emission: any drift is pure transport non-conservation with no
   rate confound. Used to bisect operators (advection vs diffusion vs
   convection).
4. **Single-tracer runs attribute cleanly.** Operator-level diagnostics
   (e.g. `ATMOSTR_FILLZ_MASS_DIAG=1`) sum over all tracers.
5. **Percent-of-emission is a treacherous unit** for mature tracers: F32
   noise scales with the standing *burden*, so a tracer whose monthly
   emission is only ~0.3 % of its burden (SF6) shows %-of-emission numbers
   ~300× larger than an IC=0 tracer with identical relative noise. Compare
   absolute drift vs burden when judging the *numerics*; use %-of-emission
   when judging the *science impact* on that tracer's budget.

## The issue catalog (found → fixed, December-2021 C180 campaign)

| # | Symptom | Root cause | Fix | Commit | Residual |
|---|---------|-----------|-----|--------|----------|
| 1 | SF6 deficit ~0.4 % even with emissions off at IC=0 | F32 rounding when adding small per-step emission increments to large accumulated cell masses | Kahan compensated addition in the surface-flux kernels | `4fe0b61e` | IC=0 emission test → 0.015 % |
| 2 | SF6 deficit 1.86 % (GEOS-IT) / 0.96 % (ERA5) per month | F32 non-conservation in the implicit vertical-diffusion Thomas solve on background-dominated columns | Per-column anomaly subtraction (column-min reference) inside both CS diffusion kernels — mathematically a no-op, only F32 rounding changes | `9ddf3d68` | 0.36 % / 0.42 % (the remaining is issue 5 + advection floor) |
| 3 | co2_natural **+27 % of emission** (+1 Pg/month) surplus; clean day 1, sharp onset at t=24 h | **Day-1 flux replay**: multi-binary runners restarted `sim.time` per daily binary, so time-varying (lmdz/CAMS) sources re-selected day-1 slices all month. dm_end = 31×day-1 emission exactly. Constant-rate tracers immune. | `start_time` clock origin threaded through both runner loops | `1c1c7005` | −0.09 % (raw) — *the +27 % was never a precision problem* |
| 4 | LinRood co2_fossil transient surplus (+5 % of emission at t=3 h, saturating ~0.3 % at month scale); absent in PPM | The GCHP `fillz` positivity fixer: its mass→VMR→fill→VMR→mass round-trip is not F32-conservative; injects net mass each time it repairs an advection undershoot of a sharp plume. Attribution exact: fillz/surplus = **1.000** | `[advection] fillz = false` knob (LinRood-only); flux-form advection is then exactly conservative | `79132f78` | +0.0005 % (1-day A/B); see ADVECTION_SCHEMES.md §fillz for the negatives trade-off |
| 5 | Residual background-proportional F32 transport drift on large-background tracers | Every flux-form update rounds relative to the background-dominated cell mass | **Reference-state (anomaly) transport**: carry `q = q_ref + q_anom`, transport only the anomaly in F32, the uniform `q_ref` rides the air mass analytically (exact eigenstate). Opt-in `[tracers.X.transport] reference = "global_mean"`; LinRood-only | plan 45, `35204257`…`9e584089` | co2_natural −0.09 % → −0.03 % (ERA5); neutral for SF6 (see below) |
| 6 | Referenced-tracer budgets from output files read ~0.1 % wrong | The saved spatial field is an F32 full-field reconstruction (`anom + q_ref·m`) | Exact F64 `<tracer>_total_mass` variable written per frame | `aa403559` | budgets exact |
| 7 | (latent) `air_mass_reset_mode = "preserve_tracer_mass"` would break referenced burdens | The `q_ref·m` part of a referenced burden rides the air mass through a reset | Reset absorbs `anom += q_ref·(m_old − m_new)`; `preserve_vmr` rejected for referenced tracers | `55de815b` | validated −0.09 % across 744 window resets |

## Measured conservation floors (December 2021, C180, F32, full physics)

True mass balance, % of the month's emission:

| tracer | PPM (GEOS-IT / ERA5) | **LinRood⁷ + fillz=false + reference** (GEOS-IT / ERA5) |
|---|---|---|
| co2_natural (412 ppm, time-varying flux) | +0.09 / −0.62 | **−0.10 / −0.03** |
| co2_fossil (IC=0, sharp sources) | +0.034 / +0.024 | **+0.011 / −0.004** |
| sf6 (10 ppt mature background) | +0.36 / +0.42 | **+0.14 / +0.36** |
| rn222 | decay-dominated (dm ≪ ∫F is the sink, not a leak) | — |

Reference outputs: `campaign_winter2021/{geosit,era5}_linrood_nofillz_dec2021.nc`.
The "maximally conservative" configuration is
`scheme = "linrood"`, `ppm_order = 7`, `fillz = false`, plus
`reference = "global_mean"` for large-background tracers.

## Why SF6 sits at ~0.1–0.4 % and what (doesn't) help

SF6 is the structurally hardest case, **not because ppt-scale numbers are
small** — floating point is scale-invariant in relative terms, so rescaling
units cannot help — but because of its **emission-to-burden ratio**: one
month of emission is only ~0.3 % of the 2.8×10⁸ kg standing burden that F32
churns through every substep. The same ~10⁻⁶/month relative transport noise
that is invisible for an IC=0 tracer (burden = emission) reads as 0.1–0.4 %
"of emission" for SF6.

Measured non-fixes: reference-state transport is **neutral** for SF6 (its
field spans 9–16 ppt around a 10.2 ppt mean — the anomaly is only ~2–5×
smaller, and mixed-sign anomaly columns cancel slightly worse in the
diffusion solve, eating the gain). Reset mode is irrelevant (2×2 tested).

## When Float64 is needed

- **F64 floor (measured, curry A100, SF6-only 5-day): +0.060 %** vs F32's
  +0.213 % — ~3.6× tighter. Use F64 when a mature small-flux tracer's budget
  must close below ~0.1 % of emission.
- F64 requires real hardware double support: **curry (A100)**, not wurst
  (L40S has no FP64 units — software emulation at ~1/64 speed). A full
  4-tracer F64 C180 run needs ~34 GB GPU memory; run per-tracer on a shared
  card.
- Everything else demonstrated in this page runs production F32: CO₂-class
  tracers close to ≤0.1 % of emission with the maximally-conservative
  configuration, which is at or below met-driven uncertainty.

## Open items / potential work

- **Conservative fillz**: if GCHP-faithful positivity AND exact conservation
  are both needed, fillz's borrow could be made F32-conservative (Kahan the
  round-trip). Verify with `ATMOSTR_FILLZ_MASS_DIAG=1` (must read 0 net).
- **Per-level reference `q_ref[k]`** (plan 45 deferred extension): would also
  remove the mean vertical profile (~100× background reduction for CO₂);
  requires a reference-aware vertical sweep + adjoint term.
- **SF6-class tracers below 0.1 %**: per-tracer F64, or accept the floor.
- The LinRood `fillz=false` negatives trade-off is documented in
  ADVECTION_SCHEMES.md §fillz — revisit if a positivity-requiring operator
  is ever added (the compatibility gate must then reject `fillz=false`?
  currently positivity consumers do not exist).
