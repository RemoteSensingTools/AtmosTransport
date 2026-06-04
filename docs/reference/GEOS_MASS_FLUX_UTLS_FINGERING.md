# GEOS native-mass-flux SH-UTLS "fingering": diagnosis and resolution

**Status (2026-06-03):** root cause established, every cm-side fix ruled out *with
evidence*, corroborated by the GCHP team's own open work
([geoschem/GCHP#342](https://github.com/geoschem/GCHP/issues/342)). **Production
guidance: use the wind-derived / spectral path (ERA5, or 3-hourly GEOS/MERRA-2
winds) for UTLS-sensitive science; the native GEOS-IT/FP hourly mass-flux path
carries an intrinsic grid-scale artifact at the tropopause.**

---

## 1. Symptom

GEOS-IT (and GEOS-FP) native cubed-sphere transport binaries, built from the
archived hourly C-grid mass fluxes (`MFXC`/`MFYC`) and the analyzed dry pressure
endpoints, produce **grid-scale "fingering" in tracer fields at the
stratosphere–troposphere transition (UTLS, ~100–280 hPa), most visible in the
quiescent clean SH** (the real variability is small there, so the spurious
structure stands out). It appears in **advection-only** runs and is **not masked
by convection or diffusion** (both are essentially absent at the UTLS).

## 2. Quantified baseline — GEOS fingers 3–5× more than ERA5

`scripts/diagnostics/finger_era5_vs_geos.py`: absolute grid-scale Laplacian RMS
of `co2_natural` over the SH (lat < −30) at the UTLS, last output time.

| run | grid-noise `|Lap|RMS/std` @ ~130 hPa | @ ~165 hPa |
|---|---|---|
| GEOS adv-only | 0.096 | 0.108 |
| GEOS full-physics | 0.117 | 0.125 |
| **ERA5 full-physics** | **0.046** | **0.054** |

GEOS carries **2.7–5× more** grid-scale noise than ERA5 at the SH-UTLS, and
GEOS-fullphys ≈ GEOS-adv-only (physics does not mask it). **ERA5 is the
existence proof that the artifact is avoidable**, and it sets the target.

## 3. Root cause — an intrinsic MFXC↔DELP residual `M`

Define the per-column residual
```
M(i,j) = Σ_k dm_dry(i,j,k) − pit_native(i,j)
```
where `dm_dry` is the dry-air mass tendency from the analyzed DELP_dry endpoints
and `pit_native = Σ_k (am[i]−am[i+1] + bm[j]−bm[j+1])` is the column convergence
of the native dry mass fluxes. By dry-air conservation on a closed sphere `M`
**should be exactly 0** (the column dry mass changes only via horizontal dry-air
convergence — no vertical flux through TOA/surface). Measured: `M ≈ 5.6e-4` of
column mass (RMS), up to 1.4% (max), **grid-noisy and tropopause-concentrated.**

So `M ≠ 0` is a **data inconsistency between the accumulated horizontal mass flux
and the instantaneous analyzed pressure endpoints** — i.e. the offline-archived
`MFXC` and `DELP_dry` do not discretely close FV3's dry continuity. The cm
diagnosis (`diagnose_cs_cm!`) is then forced to inject `M` into the vertical mass
flux per cell to satisfy the analyzed endpoint → grid-noisy `cm` → fingering when
that `cm` advects a tracer across the sharp UTLS gradient.

### It is intrinsic to native FV3, not a GEOS-IT or preprocessing artifact
`scripts/diagnostics/moist_budget_IT_vs_FP.jl` (same `geos_native_to_face_flux!`
machinery, `mass_flux_dt = 450`):

| product | normalized `M` RMS/colmass |
|---|---|
| GEOS-IT C180 (replay) | 5.61e-4 |
| **GEOS-FP C720 (online native)** | **5.78e-4** |

**FP/IT ratio = 1.03.** GEOS-FP's online native fluxes carry the *same* residual,
at the actual model grid (C720) — so it is not a GEOS-IT replay artifact, not a
resolution/regrid artifact, and (ruled out separately) not the dry/moist
conversion (`M` is uncorrelated with the column water-mass tendency, corr 0.01),
not endpoint time-sampling (`M_I1 = M_A1` to 4 digits using CTM_A1's own DELP),
and not condensate (too small). It is fundamental to reconstructing dry
continuity offline from FV3's accumulated `MFXC` + instantaneous `DELP`.

### `M` is temporal, not a per-snapshot flux↔dp geometric mismatch
GMAO builds `MFXC` from the dynamics Courant numbers, `mfx = dp·cx·area` (FV3
`fv_computeMassFluxes`). The archived `CX` (accumulated Courant) lets us invert
that: `dp_flux = MFXC/CX/area` is the flux-weighted dp the mass flux actually
carried. `scripts/diagnostics/cx_implied_dp_vs_delp.jl` (GEOS-IT C180, Dec-11):
the SH-median `dp_flux/DELP` is **1.0007, dead flat from 26 → 431 hPa** — so the
archived `MFXC`, `CX`, and `DELP` satisfy `MFXC = CX·DELP·area` *at a snapshot*
to <0.1% at every level. The grid-scale roughness of that ratio is tiny
everywhere (≤0.66%) but does spike ~8–16× at the upper troposphere (266 hPa,
0.0066, vs 0.0003 in the stratosphere) — a faint UTLS-localized flux↔dp
disagreement, far too small to be the dominant `M`. **So `M` is overwhelmingly
the temporal/accumulation residual** (hour-mean flux *convergence* vs
instantaneous-endpoint *tendency* — the sub-hourly dynamics lost to hourly
archiving), not a geometric inconsistency at a single time. This also **closes
the "use the Courant numbers directly" route**: `MFXC/CX` just returns `DELP`, so
`CX` carries no information beyond `MFXC`+`DELP`, and no offline reconstruction
from `CX` escapes `M`.

## 4. Why ERA5 is clean (the consistency that GEOS lacks)

ERA5's preprocessing synthesizes winds *and* pressure from one self-consistent
spectral state, and reconstructs per-layer mass from the surface pressure via the
hybrid coordinate (`ΔA + ΔB·ps`). So its `M ≈ 0`, and it uses the **same**
`diagnose_cs_cm!` — confirming the cure is **input consistency, not the cm
method.** ERA5 is hourly *and* clean. (Caveat learned the hard way: a *normalized
cm-roughness* metric is misleading across products with different cm magnitudes —
trust the tracer-level metric.)

## 5. Map of dead-ends (do not re-tread — all ruled out with evidence)

The palindrome replay gate (`ReplayContinuity.jl`) ties `cm` exactly to the
written endpoint: `m_evolved = m_cur − 2·steps·(div_h + Δcm)`, so the gate closes
iff `m_next == m_cur + 2·steps·dm`. That contract makes the following mutually
exclusive — you can satisfy **three of {native MFXC, analyzed dry-DELP endpoint,
smooth UTLS cm, replay-gate}, not all four**:

| attempt | result | why |
|---|---|---|
| **cm-output smoothing** | fails replay gate | `cm` is tied exactly to unsmoothed am/bm + endpoint |
| **balanced pressure-fixer** | = Path-A (fingers, 0.34) | the column balance forces `pit = ΔPS_analyzed`, which is grid-noisy (= `M`) |
| **flux-adv + vertical REMAP** (FV3-style) | = cm-advection (ratio 1.00) | both inject `cumsum(M)`; the vertical operator is irrelevant — `prototype_remap_vs_cm.jl` |
| **`moisture_filtered`** (smooth the residual) | NO-OP | balancing first moves the noise into `div_h`; smoothing a clean residual does nothing |
| **pure pressure-fixer** | smooth cm but ps drift / m<0 | endpoint `m_cur+ΔB·pit` drifts from analyzed and goes negative at thin UTLS layers |
| **`pfix_corrected`** (native pf + zero-sum spatial low-pass toward analyzed PS) | **mixed → not a fix** | reduces grid-noise 3× at ~100–130 hPa but the pf cm's vertical flux makes ~165–280 hPa *worse* (45–61×); chain_mass=true accumulates negative mass at the tropopause (238→2854 cells over a day) |

**The fundamental tension:** a *smooth* cm necessarily implies an air-mass
evolution that differs from the analyzed endpoint by `M`. Put `M` in the cm → it
advects the tracer (grid-noise, or — for the pressure-fixer — a worse large-scale
vertical redistribution). Put `M` in the endpoint → the mass drifts from analysis
and goes negative. `M` is intrinsic; no cm-side closure escapes it.

## 6. The cure — reduce `M` (wind-derived fluxes)

The only route that removes `M` rather than relocating it is to derive the
horizontal fluxes from a state **consistent** with the analyzed pressure:
- **ERA5 spectral → CS** (our existing path): `M ≈ 0`, hourly, clean. Recommended
  for UTLS-sensitive production.
- **3-hourly GEOS/MERRA-2 winds + a Cameron-Smith-style pressure fix** (the
  GEOS-Chem Classic / GCHP-benchmark path): trades the hourly cadence of native
  `MFXC` for consistency. GEOS-IT A3dyn `U/V` (3-hourly C180) and MERRA-2
  `inst3_3d_asm_Nv` are on disk; MERRA-2 has no native mass flux (winds only).

There is **no GMAO product that is simultaneously native-flux, hourly, and
`M`-free.** Native mass fluxes are hourly but carry `M`; winds are consistent but
3-hourly; ERA5 is hourly + consistent but a different model.

## 7. Reference lineage and upstream corroboration

- **FV3/GCHP** (`FVdycoreCubed_GridComp/.../fv_tracer2d.F90`): `offline_tracer_advection`
  does horizontal flux-adv → Lagrangian `dpA` → conservative PPM vertical *remap*
  to analyzed pressure + a global/trop tracer rescale (`calcScalingFactor`). No
  diagnosed cm. GEOS-Chem Classic (`pjc_pfix_mod.F90` + `tpcore_fvdas_mod.F90`)
  derives fluxes from winds + the Cameron-Smith pressure fixer; its
  `Calc_Vert_Mass_Flux` `wz = cumsum(dpi − dbk·dps)` is identical to our
  `compute_cs_cm_pressure_fixer!`, and its `delp2 = dap + dbk·ps2` is the
  PS-hybrid endpoint. **Our prototype showed the remap does not escape `M`
  offline** (GCHP only escapes it *online*, where dpA and ple are one consistent
  state).
- **[geoschem/GCHP#342](https://github.com/geoschem/GCHP/issues/342)** (open,
  3-year discussion): official benchmarks use **3-hourly winds**, mass fluxes are
  a newer option still under validation; the team saw "wonky stratospheric
  tracers" with mass fluxes; **their open to-do #4 is exactly our PS-hybrid vs
  per-level dry-pressure-edge question.** Closed mass-flux bugs (#445 moisture-
  correction direction, #503 timestep scaling, #377 regridding) are uniform/
  bookkeeping fixes — **none addresses the grid-scale UTLS noise.** The Harvard
  transport-evaluation project is parked.

## 8. Diagnostic tooling

- `scripts/diagnostics/finger_era5_vs_geos.py` — tracer-level SH-UTLS grid-noise, GEOS vs ERA5.
- `scripts/diagnostics/finger_pfix_vs_endpoint.py` — tracer-level pfix vs endpoint.
- `scripts/diagnostics/moist_budget_IT_vs_FP.jl` — `M` residual, GEOS-IT C180 vs GEOS-FP C720.
- `scripts/diagnostics/moisture_residual_I1_vs_A1.jl` — `M` time-sampling + moist-flux decomposition.
- `scripts/diagnostics/cx_implied_dp_vs_delp.jl` — Courant-implied `dp_flux = MFXC/CX/area` vs `DELP` (snapshot consistency → `M` is temporal, not geometric).
- `scripts/diagnostics/cm_sh_roughness_profile.jl` — cm SH-UTLS roughness of any CS binary.
- `scripts/diagnostics/prototype_remap_vs_cm.jl`, `prototype_pfix_balanced.jl` — the cm-vs-remap and pressure-fixer prototypes.
- Closures `:pressure_fixer`, `:moisture_filtered`, `:pfix_corrected` in
  `cubed_sphere_geos.jl` are **diagnostic-only** (warned at parse time);
  `:endpoint_balanced` is the production default.

## 9. Recommendation

1. **Production / UTLS-sensitive science:** ERA5 (hourly, clean) or 3-hourly
   wind-derived GEOS/MERRA-2. This is what GEOS-Chem uses for CO₂.
2. **Native GEOS-IT/FP mass-flux binaries** (`:endpoint_balanced`): fine for
   tropospheric / column work; carries a known, quantified, intrinsic SH-UTLS
   tropopause artifact — document the caveat, do not use for clean-SH UTLS
   gradients.
3. Do **not** pursue further cm-side closures — the dead-end map above is
   exhaustive and evidence-backed.

## 10. Confirmed: the MERRA-2 wind-derived path removes the fingering (2026-06-04)

Route 1 is now **built and tracer-validated**, not just recommended. A production
MERRA-2 preprocessor (`src/Preprocessing/sources/merra2.jl` +
`transport_binary/merra2_latlon_regrid.jl`, a near-clone of the ERA5 N320 path)
derives the horizontal mass fluxes from MERRA-2 **time-averaged winds**
(`tavg3_3d_asm_Nv`) + **instantaneous PS** (`inst3_3d_asm_Nv`) and applies the
Poisson **column** balance (= the Cameron-Smith pressure-fix, `pjc_pfix`) against
the dry-PS endpoints — exactly the GEOS-Chem CO₂ recipe (winds, no archived
flux). The Dec-11 C180 binary passes every gate (replay rel 4e-6, positivity
0.948, adaptive substeps 46–56/window).

Three **matched** Dec-11 advection-only `co2_natural` runs (same `catrine_co2` IC
+ `lmdz_co2` flux + PPM; only the met binary differs;
`scripts/diagnostics/finger_route1_dec11.py`) give the verdict — SH-UTLS absolute
grid-noise (`|Lap|RMS`) relative to native GEOS:

| level (~p) | MERRA-2 / GEOS | ERA5 / GEOS |
|---|---|---|
| ~52 hPa | 0.38 | 0.24 |
| ~85 hPa | **0.28** | 0.19 |
| ~139 hPa | **0.29** | 0.23 |

Relative roughness (`|Lap|RMS/std`) at the UTLS core: GEOS 0.096–0.119,
**MERRA-2 0.028–0.042 ≈ ERA5 0.027–0.061**. **MERRA-2 tracks ERA5 at every
level; native GEOS is the lone outlier** — closing the diagnosis end-to-end: the
fingering *is* the native-MFXC↔DELP artifact, and a wind-derived flux removes it
(~3× cleaner at the UTLS core, at ERA5's level).

**Negative result:** GEOS-IT A3dyn *native* winds do **not** work (proxy only
~0.8× — they are the same FV3 cubed-sphere winds that produced `MFXC`, since
`MFXC = CX·DELP·area`). A **lat-lon** source (MERRA-2 / ERA5) is required so the
conservatively-regridded flux carries no cubed-sphere grid imprint. Faithful
winds use the `tavg3` collection (`winds_collection = "tavg3"`); `inst3`
instantaneous winds are a slightly-less-faithful fallback.
