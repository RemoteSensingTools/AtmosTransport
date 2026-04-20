# GCHP Convection — Upstream Fortran Reference Notes

**Source:** GEOS-Chem source code, extracted from
`gchp_tar.gz`:`GCHP/src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/GeosCore/`
(dated 2026-03-10, corresponds to GEOS-Chem 14.x).

**Companion files:**
- `convection_mod.F90` (2282 lines) — both RAS and Grell-Freitas schemes
- `calc_met_mod.F90` (1677 lines) — met field conventions

**Purpose:** Comparison reference for plan 18's `CMFMCConvection`
port. Per CLAUDE_additions_v2 § "Validation discipline for physics
ports", legacy Julia is starting point, not ground truth. This
document captures the Fortran reference for independent validation.

---

## 1. Algorithm summary

**Scheme:** S-J Lin's cumulus transport for GSFC-CTM (Lin 1996),
adapted to GEOS-Chem. A pseudo-tendency approach: updraft
concentration computed bottom-up from surface, then environment
tendency applied top-down using two CMFMC values per layer
(inflow at bottom, outflow at top).

**Inputs (per column, from State_Met):**
- `CMFMC(1:NZ+1)` — cloud updraft mass flux at **interfaces**
  [kg/m²/s]
- `DTRAIN(1:NZ)` — detrainment mass flux at **layer centers**
  [kg/m²/s]
- `DELP_DRY(1:NZ)` — dry pressure thickness [hPa]
- `DQRCU(1:NZ)`, `PFICU(1:NZ+1)`, `PFLCU(1:NZ+1)`, `REEVAPCN(1:NZ)`
  — precip fields for wet deposition (NOT needed for inert
  tracers; plan 18 scope skips these)

**Driver dispatch (lines 267-299):** `Input_Opt%Grell_Freitas_Convection`
is a boolean flag picking between RAS and GF paths. Both consume
the same CMFMC + DTRAIN interface but have different internal
physics.

**Level convention:** k=1 = surface (bottom), k=NZ = TOA (top).
OPPOSITE of AtmosTransport (k=1=TOA). Port must reverse columns
at the interface.

## 2. CMFMC indexing convention

Line 574:
```fortran
CMFMC => State_Met%CMFMC(I, J, 2:State_Grid%NZ+1)
```

**CMFMC is pointer-sliced starting at index 2.** This means
inside DO_RAS_CLOUD_CONVECTION, when the code says `CMFMC(K)`, it
actually means `State_Met%CMFMC(I, J, K+1)` — the flux at the TOP
of layer K.

Paired with `CMFMC_BELOW = CMFMC(K-1)` (line 808), which means
`State_Met%CMFMC(I, J, K)` = flux at the BOTTOM of layer K.

**Summary:**
- `CMFMC_BELOW` = flux INTO layer K from below (K-1 interface)
- `CMFMC(K)` = flux OUT of layer K upward (K interface)

These TWO different values appear in the tendency equation. The
legacy Julia uses a SIMPLIFIED tendency with only ONE CMFMC value
— see §5 below.

## 3. RAS algorithm core

Main driver: `DO_RAS_CLOUD_CONVECTION`, lines 422-1419. Called
per (I,J) column from OpenMP outer loop.

### 3.1 Cloud base (lines 626-634)

```fortran
CLDBASE = 1
DO K = 1, NLAY
   IF ( DQRCU(K) > 0.0 ) THEN
      CLDBASE = K
      EXIT
   ENDIF
ENDDO
```

Cloud base = lowest level where convective precipitation is being
formed (`DQRCU > 0`).

**Note:** plan 18 scope excludes wet deposition — DQRCU not needed
for inert tracer transport. But CLDBASE determination still
needs some diagnostic. Alternative: use lowest level where CMFMC
is non-zero (that's the SURFACE of cloud).

### 3.2 Sub-stepping (lines 1601-1612)

```fortran
IF (State_Grid%NZ > 72 .or. Input_Opt%MetField == "MODELE2.1") THEN
   NS = NDT / 60      ! sub-timestep 60 sec for high-res
ELSE
   NS = NDT / 300     ! sub-timestep 300 sec standard
ENDIF
NS  = MAX(NS, 1)
SDT = DBLE(NDT) / DBLE(NS)    ! seconds per internal step
```

GEOS-Chem sub-steps for CFL stability. Plan 18 should replicate
this — an internal timestep shorter than dynamic timestep.
Typical value 300s; high-res runs use 60s.

### 3.3 Below cloud base — well-mixed layer (lines 725-782)

```fortran
! Initial updraft concentration is from CLDBASE
QC = Q(CLDBASE)

IF (CLDBASE > 1) THEN
   IF (CMFMC(CLDBASE-1) > TINYNUM) THEN
      ! QB = weighted avg mixing ratio below cloud base
      QB = 0.0
      DELP_DRY_NUM = 0.0
      DO K = 1, CLDBASE-1
         QB = QB + Q(K) * DELP_DRY(K)
         DELP_DRY_NUM = DELP_DRY_NUM + DELP_DRY(K)
      ENDDO
      QB = QB / DELP_DRY_NUM
      
      ! Mix in updraft flux from cloud base
      QC = (MB*QB + CMFMC(CLDBASE-1) * Q(CLDBASE) * SDT) / &
           (MB    + CMFMC(CLDBASE-1) * SDT)
      
      ! Uniform concentration across sub-cloud layers
      Q(1:CLDBASE-1) = QC
   ENDIF
ENDIF
```

This well-mixed below-cloud layer treatment is NOT in the legacy
Julia port. The Julia just uses `q_cloud_below = 0` at the
surface and starts accumulating. This is a **potential bug**
in the legacy: it fails to represent the well-mixed sub-cloud
layer.

For CATRINE tracers with strong surface emissions (CO2 tropics,
Rn-222), this matters — the sub-cloud mixing redistributes
surface mass before convection kicks in.

### 3.4 Updraft concentration (lines 785-920)

Bottom-to-top pass through layers CLDBASE to KTOP. At each level:

```fortran
CMFMC_BELOW = 0.0
IF (K > 1) CMFMC_BELOW = CMFMC(K-1)

IF (CMFMC_BELOW > TINYNUM) THEN
   CMOUT  = CMFMC(K) + DTRAIN(K)                   ! air leaving cloud
   ENTRN  = CMOUT - CMFMC_BELOW                    ! air entrained from env
   
   QC_PRES = QC * (1 - F(K,NA))                    ! preserved against scavenging
   QC_SCAV = QC * F(K,NA)                          ! lost to scavenging
   
   ! [scavenging re-evaporation block - skipped for inert]
   
   IF (ENTRN >= 0 .and. CMOUT > 0) THEN
      QC = (CMFMC_BELOW * QC_PRES + ENTRN * Q(K)) / CMOUT
   ENDIF
   
   ! ... four-term tendency (§3.5)
   
ELSE
   ! No inflow — simplified treatment
   QC = Q(K)
   IF (CMFMC(K) > TINYNUM) THEN
      T2 = -CMFMC(K) * QC
      T3 =  CMFMC(K) * Q(K+1)
      DELQ = (SDT / BMASS(K)) * (T2 + T3)
      Q(K) = Q(K) + DELQ
   ENDIF
ENDIF
```

### 3.5 Four-term tendency (lines 988-1007) — THE CRITICAL PART

**This is the exact Fortran for the environment mass balance:**

```fortran
T0 =  CMFMC_BELOW * QC_SCAV    ! scavenging loss (inert: 0)
T1 =  CMFMC_BELOW * QC_PRES    ! updraft inflow (preserved fraction)
T2 = -CMFMC(K)    * QC         ! updraft outflow (post-mix QC)
T3 =  CMFMC(K)    * Q(K+1)     ! subsidence inflow (from above)
T4 = -CMFMC_BELOW * Q(K)       ! subsidence outflow (to below)

TSUM = T1 + T2 + T3 + T4
DELQ = (SDT / BMASS(K)) * TSUM

IF (Q(K) + DELQ < 0) DELQ = -Q(K)    ! positivity clamp
Q(K) = Q(K) + DELQ
```

**Key observations:**

1. **TWO different CMFMC values appear.** `CMFMC_BELOW` (= flux at
   bottom of layer K) appears in T1 and T4. `CMFMC(K)` (= flux at
   top of layer K) appears in T2 and T3.

2. **QC in T1 vs T2 is DIFFERENT.** T1 uses QC_PRES (pre-mix,
   from level below). T2 uses QC (post-mix, just updated for
   level K). For inert tracers with F=0, QC_PRES = old_QC, so
   T1 uses pre-mix and T2 uses post-mix.

3. **For inert tracers**, simplifying:
   ```
   TSUM_inert = CMFMC_BELOW * old_QC - CMFMC(K) * new_QC
              + CMFMC(K) * Q(K+1) - CMFMC_BELOW * Q(K)
   ```
   where `new_QC = (CMFMC_BELOW * old_QC + ENTRN * Q(K)) / CMOUT`.

4. **DTRAIN appears implicitly** through `CMOUT = CMFMC(K) + DTRAIN(K)`
   and `ENTRN = CMOUT - CMFMC_BELOW`. There's no explicit
   `DTRAIN(K) * (QC - Q(K))` term in GEOS-Chem's tendency —
   detrainment's effect is embedded in the `new_QC` calculation.

## 4. Sign and unit conventions

**Mass flux sign:** CMFMC is always positive (upward-only mass
flux). No downdraft (contrast with TM5's four-field scheme).

**Units:** CMFMC in kg/m²/s; DTRAIN in kg/m²/s; DELP in hPa
(converted to kg/m² for BMASS via DELP * 100 / g).

**Tracer storage:** GEOS-Chem stores in mixing ratio [kg/kg dry
air] for convection — see line 699 `Q => Spc(IC)%Conc(I,J,:)`.
Before/after conversion handled by UnitConv_Mod. Plan 18's
AtmosTransport stores in mass; kernel must convert internally.

**Dry air basis throughout GEOS-Chem transport.** BMASS =
DELP_DRY * 100/g uses DRY pressure, not wet (line 652).

Comment block in `calc_met_mod.F90:185-200` clarifies: "Despite
similar names, DELP_DRY is fundamentally different from
PEDGE_DRY. ... The latter [DELP_DRY] is needed for transport
calculations ... to conserve mass."

Tracer in kg/kg dry, layer mass dry, tendencies in dry basis.
Consistent throughout.

**CMFMC/DTRAIN basis.** The Fortran is less explicit here.
Comment at line 651: "This is done to keep BMASS in the same
units as CMFMC * SDT" — establishes UNIT compatibility (both
in kg/m²) but doesn't explicitly state basis.

Consistency argument: if BMASS is dry and the tendency
`DELQ = (SDT/BMASS) * TSUM` must be dimensionally consistent
with Q (kg/kg dry), then CMFMC must also be dry-basis
(kg dry-air / m² / s). Any moist CMFMC input would need
conversion via `cmfmc_dry = cmfmc_moist × (1 - qv_interface)`
upstream.

**TM5 comparison (CORRECTED):**

TM5 uses **moist basis end-to-end**:
- Grid-box mass `m = phlb_top - phlb_bot / g × area` uses TOTAL
  pressure (moist)
- `entu/detu/entd/detd` are moist-basis fluxes
- No dry-air correction in TM5's convection pathway

**AtmosTransport legacy** (important subtlety): has dry-air
correction kernels in `src_legacy/Advection/latlon_dry_air.jl`
and `src_legacy/Advection/cubed_sphere_mass_flux.jl` that
convert moist → dry. When applied upstream of convection, legacy
matches GCHP dry-basis convention.

**However:** the comment at `src_legacy/Convection/ras_convection.jl:41-46`
says "No dry conversion is applied before convection" — this is
**inconsistent with the legacy's own dry-correction code** and
appears stale. Plan 18 should correct this comment during port.

**Implications for cross-scheme test (Commit 5):**

Comparing `CMFMCConvection` on dry-basis fields with
`TM5Convection` on moist-basis fields mixes two error sources:
1. Discretization (explicit vs implicit)
2. Basis mismatch (dry vs moist)

Option A (recommended): run BOTH operators on dry-basis for the
cross-scheme test. Apply dry correction to entu/detu/etc before
passing to `TM5Convection`. Isolates discretization error.
Tolerance ~5%.

Option B: run both on moist-basis. No correction on either side.
Also isolates discretization. Tolerance ~5%.

Option C (NOT recommended): native conventions on each side.
Basis difference adds ~1-3% noise to the discretization
comparison. Tolerance would need to be ~7-10%, but this hides
bugs that would show up at 5%.

Plan 18 Decision 10 uses Option A.

## 5. Legacy Julia port comparison (ras_convection.jl)

Spot-check of `/home/claude/AtmosTransport/src_legacy/Convection/ras_convection.jl`
against GEOS-Chem `convection_mod.F90:422-1419`.

### 5.1 Updraft pass — FAITHFUL

Legacy Julia lines 118-159 match GEOS-Chem lines 805-920 for
inert tracers. Updraft concentration formula:
`qc = (cmfmc_below_eff * q_cloud_below + entrn * q_k) / cmout`
matches the Fortran:
`QC = (CMFMC_BELOW * QC_PRES + ENTRN * Q(K)) / CMOUT`
when QC_PRES = old_QC (inert case).

**One addition in legacy Julia:** `cmfmc_below_eff = min(cmfmc_below, cmout)`
to cap inflow at outflow. This is a safety check for inconsistent
met data. GEOS-Chem doesn't have this explicitly but handles the
same issue via `IF (ENTRN >= 0)` check. Both approaches
produce non-negative ENTRN.

### 5.2 Tendency pass — SIMPLIFIED / POTENTIAL BUG

Legacy Julia lines 188-193:
```julia
if k > 1
    tsum += cmfmc[ii, jj, k] * (q_env_prev - q_k)   # ONE CMFMC
end
tsum += dtrain[ii, jj, k] * (q_cloud_ws[ii, jj, k] - q_k)
```

This is NOT what GEOS-Chem does. The Fortran uses FOUR terms
with TWO different CMFMC values:

**Fortran (GEOS-Chem, for inert tracer):**
```
TSUM = CMFMC_BELOW·old_QC - CMFMC(K)·new_QC
     + CMFMC(K)·Q(K+1)    - CMFMC_BELOW·Q(K)
```

**Legacy Julia (simplified):**
```
tsum = cmfmc[k] · (q_env_above - q_k)          [subsidence, ONE cmfmc]
     + dtrain[k] · (q_cloud[k] - q_k)          [detrainment, explicit]
```

These agree only when:
- `CMFMC(K) = CMFMC_BELOW` (mass flux constant through layer), OR
- `DTRAIN · (QC - Q) = CMFMC_BELOW · old_QC - CMFMC(K) · new_QC`
  (updraft balance exactly reconstructed)

Whether the second identity holds depends on the updraft
equation. Let me check:

`CMOUT · new_QC = CMFMC_BELOW · old_QC + ENTRN · Q(K)`
`CMOUT = CMFMC(K) + DTRAIN(K)`
`ENTRN = CMOUT - CMFMC_BELOW`

So: `CMFMC_BELOW · old_QC = CMOUT · new_QC - ENTRN · Q(K)`
                         `= (CMFMC(K) + DTRAIN(K)) · new_QC - (CMFMC(K) + DTRAIN(K) - CMFMC_BELOW) · Q(K)`

Substituting into Fortran TSUM:
```
TSUM_inert = [(CMFMC(K)+DTRAIN(K))·new_QC - (CMFMC(K)+DTRAIN(K)-CMFMC_BELOW)·Q(K)]
           - CMFMC(K)·new_QC
           + CMFMC(K)·Q(K+1)
           - CMFMC_BELOW·Q(K)

         = DTRAIN(K)·new_QC - (CMFMC(K)+DTRAIN(K)-CMFMC_BELOW)·Q(K)
           + CMFMC(K)·Q(K+1) - CMFMC_BELOW·Q(K)
           - CMFMC_BELOW·Q(K) (wait, this is getting messy)
```

Let me redo more carefully:
```
TSUM_inert = CMFMC_BELOW·old_QC                      (T1)
           - CMFMC(K)·new_QC                         (T2)
           + CMFMC(K)·Q(K+1)                         (T3)
           - CMFMC_BELOW·Q(K)                        (T4)
```

Substitute `CMFMC_BELOW·old_QC = CMOUT·new_QC - ENTRN·Q(K)`:
```
TSUM_inert = CMOUT·new_QC - ENTRN·Q(K) - CMFMC(K)·new_QC + CMFMC(K)·Q(K+1) - CMFMC_BELOW·Q(K)
           = (CMOUT - CMFMC(K))·new_QC - ENTRN·Q(K) + CMFMC(K)·Q(K+1) - CMFMC_BELOW·Q(K)
           = DTRAIN(K)·new_QC - ENTRN·Q(K) + CMFMC(K)·Q(K+1) - CMFMC_BELOW·Q(K)
```

And `ENTRN = CMOUT - CMFMC_BELOW = CMFMC(K) + DTRAIN(K) - CMFMC_BELOW`:
```
TSUM_inert = DTRAIN(K)·new_QC - (CMFMC(K) + DTRAIN(K) - CMFMC_BELOW)·Q(K)
             + CMFMC(K)·Q(K+1) - CMFMC_BELOW·Q(K)
           = DTRAIN(K)·(new_QC - Q(K)) + CMFMC(K)·(Q(K+1) - Q(K)) + CMFMC_BELOW·Q(K) - CMFMC_BELOW·Q(K)
           = DTRAIN(K)·(new_QC - Q(K)) + CMFMC(K)·(Q(K+1) - Q(K))
```

**So after algebraic simplification:**
```
TSUM_inert = DTRAIN(K)·(new_QC - Q(K)) + CMFMC(K)·(Q(K+1) - Q(K))
```

**Legacy Julia equivalent (for comparison):**
```
tsum = cmfmc[k]·(q_env_above - q_k) + dtrain[k]·(q_cloud[k] - q_k)
```

Mapping Julia `q_env_above` to GEOS-Chem `Q(K+1)` (both = "Q in
the layer above"):
```
tsum = CMFMC(K)·(Q(K+1) - Q(K)) + DTRAIN(K)·(q_cloud - Q(K))
```

**These are identical** if `q_cloud = new_QC`. Which is what the
legacy Julia stores in `q_cloud_ws[k]` during Pass 1!

### 5.3 Correction to §5.2 — the legacy Julia tendency IS correct

After careful algebraic expansion, **the simplified two-term
Julia tendency is mathematically equivalent to GEOS-Chem's
four-term tendency for inert tracers.** The simplification is
rigorous, not a bug.

Proof: `TSUM_inert = DTRAIN(K)·(new_QC - Q(K)) + CMFMC(K)·(Q(K+1) - Q(K))`
after substituting the updraft balance relation
`CMFMC_BELOW·old_QC = CMOUT·new_QC - ENTRN·Q(K)`.

The Julia form `cmfmc[k]·(q_env_above - q_k) + dtrain[k]·(q_cloud - q_k)`
matches this exactly when `q_cloud` stores the post-mix updraft
concentration (which `q_cloud_ws[k]` does per Pass 1 line 157).

**Caveat:** this equivalence is for INERT tracers only. When
F(K, NA) > 0 (soluble tracers), QC_PRES ≠ old_QC, and the
algebra differs. The four-term form explicitly tracks scavenging
through T0 and T1 = CMFMC_BELOW · QC_PRES. The simplified
Julia form would need a correction term for scavenging.

### 5.4 Well-mixed sub-cloud layer — MISSING from legacy

GEOS-Chem lines 742-782 implement a well-mixed treatment below
the cloud base: `QB` is a weighted average of environment
concentrations below CLDBASE, mixed with updraft flux from
CLDBASE, then uniformly applied to `Q(1:CLDBASE-1)`.

**Legacy Julia does NOT implement this.** The updraft pass just
starts with `q_cloud_below = 0` at the surface.

**Impact:** For CATRINE tracers with strong boundary-layer
sources (surface emissions), the well-mixed sub-cloud layer
redistributes surface tracer mass before convection transports
it aloft. Missing this feature means:
- Surface concentration may be higher than it should be
  (since source-region emissions don't mix horizontally within
  the sub-cloud layer before convection)
- Upper-level concentrations may be LOWER than they should be
  (less well-mixed source concentration entering the updraft)

For Rn-222 (surface source, tropical convection lifts to the
free troposphere), this is a potentially significant bias.

**Recommendation for plan 18:** port the well-mixed sub-cloud
layer treatment. Commit 3 should include this as a deliberate
addition vs. legacy Julia, with Tier C test verifying the
improvement against GEOS-Chem output.

## 6. Comparison with RAS vs Grell-Freitas

Both schemes in GEOS-Chem consume the same CMFMC + DTRAIN +
DELP interface. The DIFFERENCE is purely in the underlying
atmospheric model's scheme — different physics generates
different CMFMC/DTRAIN profiles, but the OFFLINE TRACER
TRANSPORT is identical.

Plan 18's `CMFMCConvection` works for BOTH cases. No separate
operator type needed. The June 2020 GEOS-FP discontinuity
(RAS → GF switch) is a data problem, not an algorithm problem —
the fields have the same meaning regardless of scheme.

**Confirmed by inspection of DO_GF_CLOUD_CONVECTION (lines 1424-2282):**
same variable names (CMFMC, DTRAIN, BMASS, QC, etc.), same
four-term tendency structure. Only differences are in auxiliary
fields for wet deposition (which we skip anyway).

## 7. Wet deposition (plan 18 SKIPS these)

Fortran uses FSOL, F(K,NA), QC_SCAV, QC_PRES, WASHOUT, etc. for
soluble species. Plan 18's scope is inert only. All of this is
stripped in the plan 18 port.

Legacy Julia also skips wet deposition — `ras_convection.jl`
has no scavenging code. Good starting point.

**Future wet deposition plan** will need to re-introduce the
QC_PRES / QC_SCAV distinction and the T0 scavenging term in the
tendency. At that point, the simplified two-term form won't
suffice and the four-term form will need to be restored.

This is why plan 18's `# TODO: scavenging hook here` comments
should be placed BOTH in the updraft pass (where QC would split
into QC_PRES / QC_SCAV) AND in the tendency pass (where T0 and
T1 vs T2 would diverge).

## 8. Column reversal

GEOS-Chem: k=1 surface, k=NZ TOA.
AtmosTransport: k=1 TOA, k=Nz surface.

Column must be reversed at the operator boundary. The legacy
Julia's `_ras_column_kernel!` operates in the AtmosTransport
convention directly (loops `k in Nz:-1:1` for the updraft pass,
`k in 1:Nz` for the tendency pass — opposite order from
GEOS-Chem but same physical sense after accounting for the flip).

Plan 18 can adopt either:
1. Reverse at boundary, internal kernel in GEOS-Chem convention
2. No reversal, internal kernel in AtmosTransport convention
   (like legacy)

Option 2 is simpler (one less data movement) but requires careful
comment about the flipped loop directions. Legacy uses option 2.

## 9. Plan 18 port decisions based on this reference

1. **`CMFMCConvection` tendency:** use the simplified two-term
   form `CMFMC(K)·(Q(K+1)-Q(K)) + DTRAIN(K)·(QC-Q(K))` which is
   mathematically equivalent to Fortran's four-term form for
   inert tracers. Legacy Julia approach is correct.

2. **Well-mixed sub-cloud layer:** ADD this (not in legacy).
   Tier C test should confirm improvement.

3. **Sub-stepping:** port the CFL-based sub-stepping
   (NS = NDT/300 for standard, NDT/60 for high-res).

4. **Scavenging hooks at two sites:** in updraft pass (QC_PRES/
   QC_SCAV split) and in tendency (T0, T1 vs T2 split).

5. **No cloud base from DQRCU:** use lowest level where
   CMFMC > TINYNUM instead (DQRCU is wet-dep-only).

6. **Cleanup vs legacy:** medium. Fix the sub-cloud layer
   omission. Keep the two-term tendency (it's right). Keep the
   two-pass structure (it's clean).

## 10. Tier B test targets — hand-expand specific Fortran lines

For plan 18 Commit 3 Tier B tests:

- Hand-expand Fortran 838-842: CMOUT = CMFMC(K) + DTRAIN(K);
  ENTRN = CMOUT - CMFMC_BELOW
- Hand-expand Fortran 917-920: QC = (CMFMC_BELOW·QC_PRES + ENTRN·Q(K))/CMOUT
- Hand-expand Fortran 991-999: four-term tendency for inert case
- Verify Julia port's two-term tendency matches the four-term after
  algebraic substitution (§5.3 of this document)

Small-column reference case:
- NZ=5, CLDBASE=2, KTOP=4
- CMFMC=[0, 1, 2, 3, 2, 0] (interfaces, surface to TOA)
- DTRAIN=[0, 0.5, 0.5, 0.5, 1.0]
- DELP=[100, 100, 100, 100, 100] hPa each
- Initial Q=[1, 1, 1, 1, 1] kg/kg
- dt=300s
- Expected updraft profile and tendency hand-computed

## 11. Cross-scheme consistency

For plan 18 Commit 5 (cross-scheme test) between
`CMFMCConvection` and `TM5Convection`:

**Dry vs moist basis difference is real.** GEOS-Chem's BMASS
uses DELP_DRY; TM5's m uses moist. Expect ~1-3% disagreement
from this alone.

**CMFMC/DTRAIN ↔ entu/detu/entd/detd mapping:**

Going from GEOS-Chem to TM5 fields:
```
detu(K) = DTRAIN(K)                           ! updraft detrainment
entu(K) = max(0, CMFMC(K) + DTRAIN(K) - CMFMC(K-1))    ! diagnosed entrainment
entd(K) = 0                                   ! no downdraft in GCHP
detd(K) = 0                                   ! no downdraft in GCHP
```

With no downdraft in GCHP path, TM5Convection on matched forcing
effectively becomes a single-updraft-only TM5 scheme. The
matrix build degenerates (downdraft rows → 0).

**For cross-scheme test:**
- Set up deep convective column with CMFMC + DTRAIN
- Derive entu/detu from above; set entd = detd = 0
- Run both operators
- Expect column-integrated agreement to within ~5% (dry/moist basis differs by ~1-3%)
- Expect similar qualitative vertical profile (both redistribute
  surface mass to cloud top)
- Matrix scheme is unconditionally stable; explicit scheme
  sub-steps. Small difference at layer interfaces expected.

## 12. What GCHP does NOT do (vs TM5)

- **No downdrafts.** GCHP convection is updraft-only (plus
  compensating subsidence). TM5 has explicit downdraft
  entrainment/detrainment (entd, detd).
- **No implicit solve.** GCHP is explicit with sub-timestepping
  for stability. TM5 is implicit (unconditionally stable).
- **No matrix.** GCHP is column-serial point-wise. TM5 builds
  full Nz×Nz matrix.
- **Wet scavenging embedded.** GCHP tightly couples convective
  transport with wet deposition (QC_PRES/QC_SCAV split). TM5
  separates the wet-removal matrix `lbdcv` from the transport
  matrix `conv1`.

For plan 18 inert-only scope, these differences don't matter
for CORRECTNESS but do matter for PERFORMANCE CHARACTERISTICS:
- `CMFMCConvection` has explicit sub-step overhead proportional
  to (dt_dyn / 300s) typically ~5 sub-steps per dynamic step
- `TM5Convection` has matrix-build + LU-solve overhead
  proportional to Nz³ ops per column (Nz² storage)
- At Nz=72, TM5Convection's matrix cost likely dominates
- At Nz=32, CMFMCConvection's sub-step cost may dominate

---

**End of notes.**

Plan 18 Commit 0 saves this (or an updated version) to
`artifacts/plan18/upstream_fortran_notes_gchp.md` for future
reference. TM5 notes are a sibling document.
