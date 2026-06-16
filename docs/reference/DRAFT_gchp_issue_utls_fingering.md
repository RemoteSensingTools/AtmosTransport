<!-- DRAFT GitHub issue for geoschem/GCHP — review before posting. -->
<!-- Suggested title: [DISCUSSION] Grid-scale UTLS noise from native mass fluxes:
     an intrinsic MFXC↔DELP residual (independent reproduction + quantification) -->
<!-- Suggested labels: category: Discussion, topic: Mass Fluxes, topic: Input Data -->

### Context

Following up on #342 (item 4, the dry-pressure-edge algorithm). We maintain an
independent offline cubed-sphere CTM and ingest the **same** archived GMAO native
mass fluxes (GEOS-IT C180 `CTM_A1` `MFXC`/`MFYC`, and GEOS-FP C720
`tavg_1hr_ctm_c0720`), so this is an outside-GCHP corroboration of the mass-flux
transport behavior, with a quantified diagnosis that may bear on item 4. Sharing
in case it's useful to the parked transport-evaluation work.

### What we see

A persistent **grid-scale "fingering" in tracers at the UTLS (~100–280 hPa),
clearest in the quiescent SH**, in advection-only runs (not masked by
convection/diffusion). Quantified against an ERA5 (spectral, wind-derived) run on
the **same C180 grid, same advection, same tracer** — absolute grid-scale
Laplacian RMS of CO₂ over the SH:

- native GEOS mass-flux run: ~2.7–5× the grid-noise of the ERA5 run at the SH-UTLS.

This looks distinct from the *vertical-mixing-strength* difference discussed later
in #342 — it's a horizontal grid-scale (checkerboard-ish) signal localized to the
tropopause.

### Diagnosis: an intrinsic MFXC↔DELP residual

Per column define `M = Σ_k dm_dry − Σ_k div_h(MFXC_dry)`, the gap between the
dry-air mass tendency (from the analyzed DELP_dry endpoints) and the vertically
integrated native dry-flux convergence. By dry-air conservation `M` should be 0;
we measure `M ≈ 5.6e-4` of column mass (RMS, up to ~1.4%), **grid-noisy and
tropopause-concentrated.** The vertical-mass-flux closure is then forced to absorb
`M` per cell → grid-noisy implied `cm` → fingering across the sharp UTLS gradient.

Key point for item 4: **`M` is in the column dry-continuity closure, and it is the
same whether the per-layer dry pressure comes from `ΔA+ΔB·ps_dry` (surface
reconstruction) or from the per-level DELP** — in our tests the two endpoint
constructions give an identical residual to 4 digits. So the PS-hybrid-vs-per-level
choice does not remove `M`; it only changes *where* the residual lands.

It is **intrinsic to the native FV3 fields, not a GEOS-IT/replay or regrid
artifact**: GEOS-FP C720 online native gives the same normalized residual as
GEOS-IT C180 (ratio ≈ 1.03), and it is uncorrelated with the column water-mass
tendency (so not a simple moisture-correction sign issue à la #445).

**`M` is a temporal (accumulation-vs-endpoint) residual, not a per-snapshot
geometric one.** Using the archived accumulated Courant numbers `CXC`/`CYC`, we
inverted GMAO's own `mfx = dp·cx·area`: the implied `dp_flux = MFXC/CX/area`
matches the archived `DELP` to a **flat SH-median ratio of 1.0007 from 26 to 431
hPa** (grid-scale roughness ≤0.66%, with a faint ~10× bump localized to the
upper troposphere). So at any single archived hour `MFXC`, `CX`, and `DELP` are
mutually consistent — the residual is the gap between the **hour-mean flux
convergence** and the **instantaneous-endpoint dry-mass tendency**, i.e. the
sub-hourly dynamics lost to hourly archiving. (A corollary: the Courant fields
add no information that lets the offline closure escape `M` — `MFXC/CX` just
returns `DELP`.)

### What does not fix it (offline)

We ruled out, with prototypes: post-hoc cm smoothing; balancing the fluxes to the
analyzed endpoint (re-injects the noise); the FV3-style **flux-adv + conservative
vertical remap** (it injects the same `cumsum(M)` as a diagnosed cm — the remap
only escapes `M` *online*, where the Lagrangian thickness and the analyzed
pressure are one consistent state); and a pressure-fixer (smooth cm but the mass
endpoint drifts from analysis and goes negative at the tropopause). The residual
is intrinsic; any closure relocates it rather than removing it.

### GCHP's own design documents this exact sensitivity

The cubed-sphere advection stream archives only the horizontal mass fluxes (+
Courant + PS + SPHU); the **vertical mass flux is reconstructed offline from the
convergence of the horizontal fluxes**. The GCHP v13 description (Eastham et al.,
GMD 15, 8731, 2022) states this and flags the failure mode verbatim — vertical
mass fluxes are "**expected to be particularly sensitive to errors because they
are computed from the convergence of horizontal mass fluxes**." The fingering is
the realized form of that documented sensitivity: the reconstruction is forced to
absorb `M` (the accumulation-vs-endpoint residual) into the implied `cm`.

Notably, GMAO **already computes and archives** the resolved vertical mass flux —
`MFZ` in `tavg3_3d_lsf_Ne` — but **only on the lat-lon analysis grid**, never on
the cubed-sphere advection stream. So the one field that would let offline
transport skip the noisy reconstruction exists; it's just not produced where a
cubed-sphere CTM can use it.

### The clean route (matches your benchmark practice)

Reducing `M` at the source — i.e. **wind-derived fluxes consistent with the
analyzed pressure** (your 3-hourly winds, or a spectral source like ERA5) — gives
`M ≈ 0` and no fingering. We built and tracer-validated this: deriving the fluxes
from MERRA-2 winds (the GEOS-Chem CO₂ met) + a Cameron-Smith pressure-fix moves
our SH-UTLS field **~3× closer to the GEOS-Chem reference** (state-aligned Dec 1–5
run) with ~3–4× less grid-noise than the native-MFXC run. Consistent with #342:
benchmarks use winds, and our ERA5 run is clean. The one limitation is resolution:
the wind path is regridded from a lat-lon source, so it caps at ≲C180 — which is
why an on-cube `MFZ` matters for native-resolution work.

### Questions

1. Is the **grid-scale UTLS noise** (as opposed to the vertical-mixing-strength
   difference) on your radar for the native mass-flux path, and did the item-4
   dry-pressure-edge overhaul change it either way?
2. Would the `MFXC` source ever be made discretely consistent with the archived
   `DELP` offline (e.g. an archived dry-air `Δp` tendency, or sub-step-consistent
   fluxes), so that the offline closure could be exact?
3. **Could the resolved vertical mass flux (`MFZ`, already archived on lat-lon in
   `tavg3_3d_lsf_Ne`) be exported on the cubed-sphere advection stream?** Ingesting
   the model's own `MFZ` would let offline transport reproduce the *online* vertical
   flux directly — bypassing the convergence-reconstruction that the GMD paper
   itself flags as error-sensitive, and which we trace to the fingering — at full
   native resolution (where the lat-lon wind route can't reach).

Happy to share the quantification scripts, the residual maps, and the
ERA5-vs-GEOS comparison if useful.
