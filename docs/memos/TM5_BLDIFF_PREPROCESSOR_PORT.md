# Plan: Port TM5's `bldiff` boundary-layer diffusion into the preprocessor

**Status:** scoped, not started (2026-05-30)
**Goal:** Enable TM5's *actual* vertical-diffusion scheme (the Holtslag–Boville
non-local boundary-layer parameterization) for ERA5-driven runs, computed in
the preprocessor and baked into the transport binary as a precomputed
mass-flux diffusion field `dkg`.

---

## 1. Why — what TM5 really does

TM5's vertical diffusion is **not** a runtime Kz formula. It is a preprocessor
that runs the Holtslag–Boville (1993) non-local PBL scheme and stores the
resulting per-interface air-mass exchange `dkg` [kg air/s] in the meteo files;
the model just reads `dkg` and runs an implicit Thomas solve.

Reference: `deps/tm5-cy3-4dvar/base/src/diffusion.F90`, subroutine `bldiff`
(L770–1286). The algorithm, in order:

1. **Free-troposphere Kv** (`wkvf`, L976–1019) — Louis local scheme from the
   gradient Richardson number `zrinub = static_stability / shear²`, corrected
   by stability functions.
2. **PBL height** (`pblh`, L1021–1138) — bulk-Richardson parcel method
   (Vogelezang–Holtslag 1996), interpolated to `Ri_bulk = ricr = 0.3`. Floored
   at 100 m.
3. **Surface virtual heat flux** (L918–1037) —
   `wheat = f(sshf, gph, T)`, `wqflx = f(slhf, gph, T)`,
   `wheatv = wheat + zcrdq·θ·wqflx`; Monin–Obukhov length
   `wobkl = -θv·ustar³ / (g·k·wheatv)`.
4. **Eddy diffusivity `kvh`** (L1155–1235) — the non-local core:
   - stable/neutral: `cml2·√shear²·zfstabh` (local Ri-shear),
   - unstable surface layer: `ustr·k·z·(1-z/h)²·(1-βh·z/L)^⅓`,
   - unstable mixed layer: convective velocity scale `wsc = ustr·fmt`,
   - **non-local Prandtl enhancement**: `term3 = ccon·fakn·w*/(ustr·…)` with
     `fakn = 7.2` — the counter-gradient / non-local transport term,
   - **entrainment at PBL top**: `kvhentr = 0.2·wheatv/thvgrad` (prescribed).
5. **Convert to `dkg`** (L1244–1260) —
   `dkg(l) = max(0, kvh(l))·2·(m_l + m_{l+1}) / (gph_{l+2} − gph_l)²` [kg/s];
   zero-flux at TOA and at `lmax_conv`.

**Inputs:** 3D `T`, `q`, geopotential `gph`, winds `u`,`v` (for shear) + surface
`sshf`, `slhf`, `ustar`.

### Why our two current schemes are *not* this

| `[diffusion] kind` | What it is | Gap vs TM5 |
| --- | --- | --- |
| `tm5_beljaars_viterbo_local_kz` (→ `WindowPBLKzField`) | surface-layer Beljaars–Viterbo Kz, runtime | no non-local term, no entrainment, no free-trop shear |
| `geoschem_holtslag_boville_vdiff` (→ `LocalHoltslagBovilleKzField`) | GCHP-style H–B, runtime | the **local** variant; D2 audit flags missing non-local counter-gradient |

Both compute Kz at runtime from the surface PBL payload. Neither reproduces
TM5's full `bldiff`. See `memory/diffusion_full_pipeline_audit_2026_05_25.md`
(D2).

---

## 2. Architecture decision — preprocessor `dkg`, not runtime Kz

Port `bldiff` into the **preprocessor**, mirroring the existing TM5 *convection*
port (`src/Preprocessing/tm5_convection_conversion.jl`, `ec2tm_from_rates!`),
and write `dkg` as the binary `:Kz` payload section. The runtime side is already
done:

- The implicit Thomas solver in `src/Operators/Diffusion/diffusion_kernels.jl`
  is already TM5's mass-flux form (`dkg/m` coefficients; D1 fix). It consumes
  `dkg`-equivalent fields directly and conserves `Σ m·q` to roundoff.
- `PreComputedKzField` (`src/State/Fields/PreComputedKzField.jl`) is the
  ready runtime carrier: a rank-2/3 `AbstractTimeVaryingField` whose `data` the
  caller mutates per met window. GPU-portable (Adapt hook present).

This is the most faithful option because the preprocessor can bit-match TM5's
own `dkg` if a reference meteo file is available, and it keeps the heavy PBL
physics off the runtime hot path.

**Note on `dkg` vs `kvh`:** TM5 stores `dkg` (already mass-weighted, kg/s). Our
runtime Thomas form rebuilds `dkg` from `(Kz, dz, m)`. Two options:
- **(A) store `kvh` (m²/s)** as the `:Kz` payload → runtime rebuilds `dkg` with
  the *runtime* `m`/`dz`. Cleaner contract, but the runtime `dz` must match the
  preprocessor `gph` differencing or the scheme is only approximately faithful.
- **(B) store `dkg` directly** (kg/s) → bit-faithful to TM5, but the payload is
  air-mass-basis and must be regenerated if the mass basis changes. Requires a
  new `PreComputedDkgField` (trivial variant of `PreComputedKzField`) that the
  Thomas kernel consumes pre-weighted.

**Recommendation: (A) store `kvh`.** It reuses `PreComputedKzField` and the
existing `dkg = kvh·2(m_l+m_{l+1})/dz²` kernel path unchanged; the only fidelity
caveat is the `dz`-vs-`gph` consistency, which we already control because the
N320 preprocessor computes both from the same hybrid-σ `(T, q, ps)`.

---

## 3. Data dependency — `slhf` (IN FLIGHT)

`bldiff` needs **both** `sshf` and `slhf` to form `wheatv`. The first surface
pull dropped `slhf` (CDS Modern API split). Re-download launched 2026-05-30
(CDS request `b601c452`, `scripts/downloads/download_era5_surface_netcdf.py`,
output `…/sfc_an_native/era5_surface_202112.nc`, full 11-var set incl. `slhf`).
The previous slhf-less NetCDF is backed up as `…_202112.noslhf.nc.bak`.

The surface reader (`era5_surface_reader.jl`) already needs a new field mapping:
add `:slhf => ("slhf", "slhf", "surface_latent_heat_flux")` and the accumulated
J/m² → W/m² conversion (same as `:hflux`/`sshf`).

---

## 4. Work breakdown

**P0 — surface reader `slhf` support** (small)
- Add `:slhf` to the field map + accumulated-flux unit handling in
  `era5_surface_reader.jl`.
- Probe: print `slhf` min/mean/max for Dec 1 2021 vs an independent ERA5
  reference; confirm sign convention (TM5 flips sign at L930–931).

**P1 — `bldiff` column port** (the core, ~1–2 days)
- New `src/Preprocessing/tm5_bldiff.jl`, mirroring
  `tm5_convection_conversion.jl`'s per-column harness:
  - `bldiff_column!(kvh, T, q, gph, u, v, sshf, slhf, ustar, m, ak, bk)` →
    free-trop Louis Kv, bulk-Ri PBL height, H–B `kvh` with non-local +
    entrainment.
  - Grid-level `tm5_kvh_for_hour!` over `(i, j)` columns, scratch-reused.
- Probe at every sub-step per CLAUDE.md: print `pblh`, `wobkl`, `kvh` profile
  for a known unstable land column and a stable ocean column; compare against
  the Fortran for the same inputs (hand-run a single column if no full
  reference meteo is available).
- Unit + invariance tests: `kvh ≥ 0`; zero surface flux ⇒ collapses to
  free-trop Kv; uniform tracer unchanged through the downstream Thomas solve.

**P2 — wire `:Kz` payload through the N320 preprocessor** (medium)
- In `era5_n320_regrid.jl`, add a `kvh_payload` builder beside
  `surface_vdiff_payload`, gated by a new `include_tm5_diffusion` knob in
  `config/met_sources/era5_n320.toml`.
- Conservatively regrid `kvh` (m²/s, an intensive field → area-weighted /
  bilinear, NOT the conservative mass-flux regridder) from N320 to C180.
- Attach as `window.kz_field` / `:Kz` section; extend the v4 writer +
  `CubedSphereBinaryReader` section table to carry `:Kz` (the schema reserves
  it; confirm the reader round-trips it).

**P3 — runtime recipe** (small)
- `build_runtime_diffusion(::CubedSphereRuntimeRecipeStyle, ::Val{:precomputed_kz}, …)`
  in `CSPhysicsRecipe.jl` → `ImplicitVerticalDiffusion(kz_field =
  PreComputedKzField(window_cache), …)`; require the `:Kz` section, else throw
  the regenerate-binary error (mirror the `geoschem_holtslag_boville_vdiff`
  guard).
- Add `"precomputed_kz"` to the supported-kinds error string.

**P4 — validation**
- Regenerate Dec 2021 N320 → C180 with `include_tm5_diffusion = true`.
- Compare column-mean tracer evolution: `precomputed_kz` (TM5 bldiff) vs
  `geoschem_holtslag_boville_vdiff` vs `tm5_beljaars_viterbo_local_kz`.
- Mass closure: `Σ m·q` to roundoff for an inert tracer (the Thomas form
  guarantees it; this is a regression guard).

---

## 5. Open questions

1. **`lmax_conv` ceiling.** TM5 zeroes `dkg` above `lmax_conv` (couples
   diffusion to the convection top). Decide whether to replicate or diffuse the
   full column. (TM5 default ties them; replicate for faithfulness.)
2. **`ustar` source.** TM5 `dd_calc_ustar` recomputes ustar (Charnock over sea,
   land formula). We have ERA5 `zust` directly — use it (simpler, ERA5-native)
   and note the small divergence, or port `dd_calc_ustar` for bit-fidelity.
   *Lean: use ERA5 `zust`.*
3. **Refresh cadence.** TM5 refreshes `dkg` every 3 h (surface fields are
   3-hourly). Our window is 1 h; recompute `kvh` per window from hourly fields
   (strictly better than TM5).
4. **Store `kvh` vs `dkg`** — see §2; recommend `kvh` (option A).

---

## 6. References

- `deps/tm5-cy3-4dvar/base/src/diffusion.F90` — `bldiff` (L770), `calc_kzz`
  (L268), `dd_calc_ustar` (L582), `write_diffusion`/`read_diffusion` (the
  precompute-and-store architecture).
- `src/Preprocessing/tm5_convection_conversion.jl` — the per-column porting
  template (`ec2tm_from_rates!`).
- `src/Operators/Diffusion/diffusion_kernels.jl` — the runtime mass-flux Thomas
  form (already TM5-faithful; D1).
- `src/State/Fields/PreComputedKzField.jl` — runtime carrier for the `:Kz`
  payload.
- `memory/diffusion_full_pipeline_audit_2026_05_25.md` — D1 (mass-flux form,
  shipped) and D2 (non-local gap, this plan closes it).
