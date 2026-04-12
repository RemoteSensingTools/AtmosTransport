# Global Mean Surface Pressure Treatment in the v4 Spectral Preprocessor

> **Status**: Implemented 2026-04-07. Verified on Dec 1 2021 ERA5 test data.
> Active by default in `era5_spectral_v4_tropo34_dec2021.toml` and the F64
> variant. Disable via `[mass_fix] enable = false` in the preprocessor TOML.

## TL;DR

ERA5's 4DVar reanalysis is **not mass-conserving**: the area-weighted global
mean of its surface pressure `⟨sp⟩_area` drifts by ~10⁻⁴ per day, which
translates directly into a drift in the total atmospheric mass `Σm` carried by
our v4 binaries (~5 × 10¹³ kg of mass spuriously appearing/disappearing every
hour). To eliminate this we apply a **uniform additive offset** to `sp`
immediately after `sp = exp(LNSP_grid)` in the v4 preprocessor, so that
`⟨sp⟩_area` corresponds to a fixed dry-air mass target each window. This
mirrors TM5's `Match('area-aver', sp_region0=p_global)` from
`meteo.F90:1361-1374` and reduces our 24h Σm drift from -8.31×10⁻⁴ % to
+5.88×10⁻⁹ % (≈140,000× improvement, F32 quantization noise floor).

This document explains:
1. The drift in ERA5 and why it exists.
2. The physical argument for pinning **dry** atmospheric mass instead of
   total mass.
3. The math relating `⟨sp⟩_area`, total `Σm`, and the hybrid-sigma
   coordinate.
4. The TM5 reference implementation that we copied.
5. Our implementation in `preprocess_spectral_v4_binary.jl`.
6. The verified numbers on Dec 1 2021 (before vs after).
7. What this does **not** fix.
8. Caveats, configurability, and open questions.

---

## 1. The problem: ERA5's global mean ps drifts

ERA5 is an offline reanalysis: it ingests observations, runs a 12h forecast
window, and produces an analysis state by minimizing the mismatch between the
forecast and the observations (4DVar). The analysis state is then valid for
the next 12h window. This procedure is **not mass-conserving by design**: the
analysis increments shift mass around the globe in a way that matches
observations but does not enforce a closed mass budget.

For our December 2021 ERA5 data (0.5° × 0.5° regular lat-lon grid, hourly
spectral GRIB), we measured this drift directly. Our pipeline reads spectral
LNSP from GRIB, performs SH synthesis, applies the H2 half-cell shift, takes
`exp` to get gridded total surface pressure `sp`, computes hybrid-sigma cell
masses `m = (a + b·sp)·area/g`, and writes them to a binary. The
preprocessor's per-window `Σm` (with no global mean fix applied) showed:

| Window | `Σm` (kg) | `Σm − Σm_0` (kg) |
|---|---|---|
| 0 (h00) | 5.131354×10¹⁸ | 0 |
| 1 | 5.131354×10¹⁸ + 1.34e8 | +1.34e8 |
| 5 | ... | -2.99e7 |
| 10 | ... | +2.25e8 |
| 23 | ... | varies ±~2e8 |

Cumulative `Σ(dm)` over the day = ~4.3×10¹³ kg ≈ 8.3×10⁻⁴ % of `Σm`. Identical
behavior at F64 (rules out F32 quantization), at T639 vs T359 truncation (rules
out our spectral chain), and our `⟨ps⟩` matches the independently downloaded
CDS surface_pressure dataset to within 4% (rules out our SH transforms).
Conclusion: **the drift is in ERA5's source LNSP itself, not in our
preprocessing**.

This is consistent with the published characterization of ERA5: the analysis
cycle introduces small but non-zero increments to ps each window, and over
many windows these increments accumulate.

## 2. Why this matters for transport

Tracer continuity in our model is solved on dry-air mass via mass-flux
divergence:

```
∂(ρ_tracer · m) / ∂t + ∇·(am, bm, cm) = sources
```

The model assumes `m` is internally consistent: i.e., that
`m_next - m_curr = -∇·(am, bm) + boundary` cell-by-cell (the Poisson balance
in the preprocessor enforces this LOCAL constraint), AND that globally
`Σ(m_next - m_curr) = 0` (closed planetary atmosphere). The first holds by
construction after our Poisson-balance step. The second does NOT hold without
the global mean fix, because the Poisson solver discards the global mean mode
of the residual (the matrix is singular for that mode), leaving a uniform
per-cell residual that sums to the global drift.

The runtime `mass_fixer = true` option masks this by globally rescaling tracer
mass each step to preserve `Σ(ρ_tracer · m)`. That's a band-aid: it makes the
column-mean diagnostic look right, but it silently deposits the rescaling
correction into every tracer cell, biasing diagnostics that depend on absolute
mass (e.g., total emissions, mass-balance closure, OH lifetime computations).
Long simulations accumulate this distortion.

The clean fix is to enforce global mass conservation **at the source**, in the
preprocessor's `m` computation, rather than papering over it at the runtime.

## 3. Physics: pin **dry** mass, not total mass

Earth's atmosphere is a closed system on the timescales we care about (gas
exchange with ocean and crust is < 10⁻⁹/yr). Therefore the **total atmospheric
mass should be constant**. But **dry air mass** is more strictly constant than
total mass, because total mass varies seasonally with the global water cycle:
evaporation lifts water vapor into the atmosphere, precipitation removes it,
and the equilibrium total water content varies by ~5% seasonally. Trenberth &
Smith (2005, _J. Climate_ vol. 18, "The mass of the atmosphere: a constraint
on global analyses") quantify:

| Quantity | Value | Variability |
|---|---|---|
| Total atmospheric mass `M_total` | 5.1480 × 10¹⁸ kg | ~0.1% seasonal |
| **Dry atmospheric mass `M_dry`** | **5.1352 × 10¹⁸ kg** | **constant** |
| Mean water vapor mass `M_water` | 1.27 × 10¹⁶ kg | ~5% seasonal |
| Mean column TPW `⟨W⟩` | ~25 kg m⁻² | varies |
| Climatological column-mean specific humidity `⟨q_v⟩_global` | 0.00247 | weak seasonal |

Using `A_Earth = 4π R²` with `R = 6371 km` gives `A_Earth = 5.1006 × 10¹⁴ m²`,
and with `g = 9.80665` m s⁻² we recover the area-weighted mean surface
pressures:

```
⟨ps_dry⟩    = M_dry  · g / A_Earth  ≈ 98726 Pa     ← physical target
⟨ps_total⟩  = M_total · g / A_Earth  ≈ 98972 Pa
⟨ps_water⟩  ≈ 246 Pa  (≈ ⟨W⟩ · g)
```

**We pin `⟨ps_dry⟩` to 98726 Pa.** Any choice of constant target eliminates
the drift, but `⟨ps_dry⟩` is the more defensible choice physically because it
corresponds to the genuinely conserved quantity. (TM5's value of 98500 Pa from
`binas.F90:132` is a 1990s-era nominal round number that differs from
Trenberth's measurement by ~0.25%, which is within the seasonal variability of
total mass — so neither value is "wrong", but Trenberth's is more recent and
authoritative.)

> **Comment on TM5's choice**: TM5 was originally targeted at climate
> applications and chose 98500 Pa as a stable round number. Recent reanalyses
> (Trenberth & Smith 2005, Trenberth et al. 2007) refined this to ~98726 Pa
> (dry) or ~98972 Pa (total). The difference (~0.25%) is below ERA5's typical
> error envelope, but it does mean our binaries will diagnostically differ from
> TM5's by ~250 Pa per cell on average. Both values pin the drift identically;
> the choice of constant only affects the absolute scale.

## 4. The math: how the uniform shift eliminates `Σm` drift

The hybrid-sigma vertical coordinate stores per-cell pressure thickness as:

```
dp_{i,j,k} = dA_k + dB_k · sp_{i,j}
```

where `dA_k = a_{k+1} - a_k` and `dB_k = b_{k+1} - b_k` are vertical-coordinate
constants that satisfy `Σ_k dA_k = TOA_offset` (a small constant) and
**`Σ_k dB_k = 1`** (because `b` runs from 0 at the model top to 1 at the
surface). Per-cell air mass is:

```
m_{i,j,k} = dp_{i,j,k} · area_{i,j} / g
          = (dA_k + dB_k · sp_{i,j}) · area_{i,j} / g
```

Summing over the column at cell `(i,j)`:

```
Σ_k m_{i,j,k} = (TOA_offset + sp_{i,j}) · area_{i,j} / g
```

Summing over all cells:

```
Σ_cells Σ_k m  =  TOA_offset · A_Earth / g  +  (1/g) · Σ_cells sp_{i,j} · area_{i,j}
              =  const  +  (A_Earth / g) · ⟨sp⟩_area
```

where `⟨sp⟩_area` is the area-weighted global mean surface pressure. The
first term is a fixed constant determined entirely by the hybrid coordinate
(it does not depend on the meteorology). The second term is the **only**
window-varying contribution to total mass. **Therefore, pinning `⟨sp⟩_area`
to a constant pins total mass `Σm` to a constant exactly.**

Per-window mass change becomes:

```
Σ_cells (m_next - m_curr) = (A_Earth / g) · (⟨sp_next⟩_area - ⟨sp_curr⟩_area)
                          = 0    (since both are pinned to the same constant)
```

This is the mathematical guarantee: with the uniform shift in place, the
binary's per-window `Σdm` is identically zero (modulo F32 quantization in the
binary write).

A uniform additive shift on `sp` does NOT change:
- Local pressure gradients (which drive winds and transport)
- Local pressure variances or extrema
- Per-cell mass fluxes `am`, `bm` (the dependence on `sp` enters through `dp`,
  but the relative pattern is preserved; the Poisson balance afterwards
  enforces local consistency cell-by-cell)
- The vertical distribution of mass (each interface shifts by `b_k · Δsp`,
  with the surface fully shifted and the model top unchanged — physically
  reasonable)

So the cost of the fix is bounded: a small uniform offset (~+400 Pa for our
Dec 2021 data, see §6) applied identically to every cell. No spatial pattern
is altered.

## 5. The TM5 reference

TM5 cy3-4dvar (the assimilation variant of TM5) calls the same operation at
runtime, on every meteo read, in `meteo.F90`. The relevant lines are 1361-1374
(for `sp1`, the start of the meteo interval) and the symmetric 1458-1470 (for
`sp2`, the end of the interval):

```fortran
! global field (first region) ?
! then match with average global surface pressure to ensure
! global mass balance;
! otherwise, match with parent grid:
if ( n == 1 ) then
  call Match( 'area-aver', 'n', lli(0), sp_region0, &
                                lli(n), sp1_dat(n)%data(1:im(n),1:jm(n),1), status )
  IF_NOTOK_RETURN(status=1)
```

Here `sp_region0` is a 1×1 array initialized to `p_global` from `binas.F90:132`:

```fortran
real,parameter :: p_global = 98500.0   ! Pa
```

The `Match` routine is in `grid_type_ll.F90:1147-1155`, in the `'area-aver'`
case of `Match_cell`:

```fortran
case ( 'area-aver' )
  fsum = 0.0
  do fj = fg_j1, fg_j2
    fsum = fsum + sum(tg(fg_i1:fg_i2,fj)) * tgi%area(fj)
  end do
  fsum = fsum / pgi%area(cj)
  ! add difference in averages to all cells in fine grid:
  tg(fg_i1:fg_i2,fg_j1:fg_j2) = tg(fg_i1:fg_i2,fg_j1:fg_j2) + (pg(ci,cj) - fsum)
```

`fsum` is the area-weighted mean of `tg` (the surface pressure being
adjusted), and the operation `tg += (pg - fsum)` is a uniform additive shift.
For the global region, `pg = sp_region0 = p_global = 98500 Pa`, so this
reduces exactly to:

```
sp .+= (p_global - ⟨sp⟩_area)
```

TM5 applies this BEFORE computing `m` from `sp` via `Pressure_to_Mass`. Our
preprocessor applies it BEFORE calling `compute_dp!` and
`compute_air_mass!`, which is the equivalent insertion point.

> **Why TM5 has this in `meteo.F90` and we have it in the preprocessor**: TM5
> reads gridded ERA5 surface pressure from a file and computes `m` at runtime.
> Its `meteo.F90` is the runtime tool that converts atmospheric input to
> internal model fields. We compute `sp` ourselves from raw spectral GRIB in
> the preprocessor, so the equivalent step in our pipeline is the
> preprocessor's window loop. Both do exactly the same operation; only the
> timing differs (online for TM5, offline for us).

## 6. Our implementation

### Where the fix lives

`scripts/preprocessing/preprocess_spectral_v4_binary.jl`:

- **The helper** `pin_global_mean_ps!(sp, area; target_ps_dry_pa, qv_global)`
  is defined just before the `TargetGrid` struct (line 425). It computes the
  area-weighted mean of `sp`, derives the offset from a configurable dry-mass
  target via the `(1 - q_v)` conversion, applies the offset uniformly, and
  returns the offset value. ~12 lines.
- **The call** is inside `spectral_to_native_fields!` (line 990), immediately
  after `sp = exp(field_2d)` and before `compute_dp!`. The call is gated by
  `mass_fix_enable` (defaults to `true`). When enabled, the offset is recorded
  in a per-window vector for binary header diagnostics.
- **The wiring** flows through `process_day` (kwargs added), `main()` (which
  reads the `[mass_fix]` TOML section), and the binary header (which records
  per-window offsets, the target value, and the qv climatology used).

### The conversion from dry target to total target

We work with TOTAL `sp` (i.e., `exp(LNSP)` from ERA5 includes both dry air and
water vapor partial pressures). To pin dry mass via a uniform shift on total
sp, we use the climatological column-mean specific humidity `⟨q_v⟩_global` to
convert:

```
⟨ps_dry⟩ = (1 - ⟨q_v⟩) · ⟨ps_total⟩    (column-mean approximation)
target_ps_total = target_ps_dry / (1 - ⟨q_v⟩_global)
```

For Trenberth's `target_ps_dry = 98726.0 Pa` and `⟨q_v⟩_global = 0.00247`:

```
target_ps_total = 98726.0 / (1 - 0.00247) = 98970.46 Pa
```

This is the value we actually pin. The error from using a fixed climatology
instead of the per-window true `⟨q_v⟩_global` is bounded by the seasonal
variation in TPW (~5%) times the climatology constant (0.00247), giving
~1.2 × 10⁻⁴ relative error in `target_ps_total`, which translates to
~1.2 × 10⁻⁴ × `target_ps_total` ≈ 12 Pa. Compared to our typical offsets
(~400 Pa) this is small but not negligible. For most uses it's irrelevant; if
needed, see §9 for the upgrade to per-window QV.

### What's in the binary header

The v4 binary header now contains:

```json
{
  "mass_fix_enabled": true,
  "mass_fix_target_ps_dry_pa": 98726.0,
  "mass_fix_qv_global_climatology": 0.00247,
  "ps_offsets_pa_per_window": [+401.92, +401.87, +401.89, ..., +402.85],
  "ps_offsets_next_day_hour0_pa": +402.91
}
```

This lets every downstream tool (driver, probes, regression tests) see
exactly what offset was applied per window. If a binary was generated without
the fix, `mass_fix_enabled` is `false` and the offsets are zero (or absent in
older binaries).

### Configuration

`config/preprocessing/era5_spectral_v4_tropo34_dec2021.toml`:

```toml
[mass_fix]
# TM5-style global mean ps fix — pin DRY atmospheric mass each window.
# Mirrors TM5 cy3-4dvar meteo.F90 Match('area-aver', sp_region0=p_global).
# Eliminates ERA5 4DVar's analysis-cycle mass drift (~10⁻⁴/day) by uniform
# additive shift on sp so that ⟨sp⟩_area corresponds to a fixed dry-mass
# target (Trenberth & Smith 2005: M_dry = 5.1352e18 kg → ⟨ps_dry⟩ ≈ 98726 Pa
# via M_dry·g/A_Earth).
# Set enable=false to keep the raw ERA5 drift (regression diagnostic).
enable = true
target_ps_dry_pa = 98726.0
qv_global_climatology = 0.00247
```

## 7. Verification (Dec 1 2021)

### Per-window offsets

Logged from the preprocessor:

```
Window  1/24 (hour 00)  ps_offset = +401.923 Pa
Window  2/24 (hour 01)  ps_offset = +401.867 Pa
Window  3/24 (hour 02)  ps_offset = +401.887 Pa
...
Window 16/24 (hour 15)  ps_offset = +402.462 Pa
...
Window 24/24 (hour 23)  ps_offset = +402.854 Pa

Mass-fix offsets (Pa) min/max/mean: +401.652 / +402.877 / +402.217
```

Interpretation:
- The +402 Pa mean offset reflects that ERA5's raw `⟨ps⟩` ≈ 98568 Pa is
  ~400 Pa below Trenberth's M_dry-derived target of ~98970 Pa total. This is a
  systematic disagreement between ERA5 climatology and Trenberth's older
  measurement (~0.4% of ps). Either is internally consistent; we chose
  Trenberth's value as the more recent/physical reference.
- The window-to-window variation of the offset (max - min = 1.225 Pa) is the
  ERA5 drift signal we wanted to absorb. Per window: ~0.05 Pa = ~5×10⁻⁵
  fraction of `ps`. Per day: ~10⁻³ fraction. This matches the published ERA5
  drift envelope.

### Σm drift (probe `/tmp/probe_dry_mass_residuals.jl`)

|  | Σm at win 1 | Σm at end of win Nt | Δ% |
|---|---|---|---|
| **Before fix** | 5.131354 × 10¹⁸ kg | 5.131354 × 10¹⁸ + ~4.3 × 10¹³ kg | -8.31 × 10⁻⁴ % |
| **After fix** | 5.147661 × 10¹⁸ kg | 5.147661 × 10¹⁸ kg | **+5.88 × 10⁻⁹ %** |

The remaining +5.88×10⁻⁹ % (~3 × 10⁸ kg over 24 windows) is F32 quantization
noise from writing the binary as Float32. With Float64 binary output it would
drop further. Per-window `Σdm` is now in the range ±10⁸ kg (vs ±5×10¹³ kg
before): a ~500,000× reduction.

### CDS comparison (probe `/tmp/probe_compare_sp.jl`)

After the fix, our `⟨ps⟩_ours` is constant at `9.897045703 × 10⁴ Pa`
(= `target_ps_total = 98726.0 / (1 - 0.00247) = 98970.46 Pa`) every window,
while CDS surface_pressure (the independent ECMWF download) still shows
ERA5's natural drift:

| metric | ours | CDS |
|---|---|---|
| ⟨ps⟩ at h00 | 9.897045703 × 10⁴ | 9.856969 × 10⁴ |
| ⟨ps⟩ at h23 | 9.897045702 × 10⁴ | 9.856880 × 10⁴ |
| Δ⟨ps⟩ over 24h | -7.68 × 10⁻⁶ Pa (F32 noise) | -0.894 Pa (real ERA5 drift) |

The per-cell difference `(ps_ours - ps_cds)` is ~+401 Pa with ~1 Pa
window-to-window variation. Spatial variances and gradients are unchanged
(uniform shift). The probe script's "MISMATCH" warning is now misleading: it
was written before the fix existed and was checking that our chain
*preserved* the ERA5 drift. With the fix, eliminating that drift is exactly
the desired behavior.

### 24h F64 model test (`era5_f64_debug_moist_v4_24h.toml`)

```
Wall time:  138.7 s (96 substeps)
Final mass: co2: 2.115689e+15 kg (Δ = 5.88e-09 %)
co2_column_mean min:    4.110000e-04
co2_column_mean max:    4.110000e-04
co2_column_mean mean:   4.110000e-04
max |dev from 411 ppm|: 1.37e-11    (F64 machine precision)
```

Uniform 411 ppm IC stays at 411 ppm to F64 precision over 24 hours. The
runtime tracer drift exactly matches the binary's intrinsic Σm drift
(both 5.88×10⁻⁹ %), confirming that the source has been removed at the
preprocessing stage and the runtime `mass_fixer` is now idle.

## 8. What this does NOT fix

The mass fix is a **global** correction. It does not change local fluxes,
local pressure gradients, or local mass distributions in any way that depends
on cell location. In particular, it does NOT address:

- **Local polar drainage**: pole-adjacent stratospheric cells
  (j ∈ {1, 2, Ny-1, Ny}) have `|bm|/m ≈ 0.30` per face from the spectral
  preprocessor, and cumulative drainage over a window of substeps still
  exceeds cell mass without the runtime `mass_fixer`. The mass fix shifts
  local `m` by only `b_k · Δps` (~+400 Pa × b ≈ +400 Pa at the surface,
  ~+1 Pa near the model top) — far too small to change the local CFL ratio
  significantly. **`mass_fixer = true` at runtime is still required for
  ERA5 LL stability**, independently of this fix. See CLAUDE.md invariant 11.
- **The Poisson singular mode**: the Poisson balance solver
  (`balance_mass_fluxes!`) discards the global mean residual mode by
  construction (the matrix is singular for the constant function). Before
  this fix, that residual was ~5×10⁶ kg/cell uniformly, which the runtime
  `mass_fixer` absorbed. With the fix, the global residual is now zero by
  construction, so the Poisson singular mode is no longer a problem.
- **CDS / ERA5 absolute reference matching**: our per-cell ps now disagrees
  with the raw ERA5 ps by ~+400 Pa (the offset). This is a deliberate
  trade-off: we are explicitly enforcing global mass conservation in
  preference to matching ERA5's published ps values. If you need to compare
  per-cell ps against ERA5 directly (e.g., for diagnostic plots), use the
  binary written with `[mass_fix] enable = false`, or subtract the per-window
  offset from the binary header from each cell.

## 9. Caveats and open questions

### Why a fixed climatology for `⟨q_v⟩_global` instead of per-window QV?

The climatological `qv_global = 0.00247` is from Trenberth & Smith 2005. Real
`⟨q_v⟩_global` varies seasonally by ~5% (i.e., ranges 0.0023 to 0.0026 over
the year, with hemisphere asymmetry). Using the constant introduces an error
of ~12 Pa in the target_total computation (see §6). Compared to:
- The current 24h drift after fix: 5.88×10⁻⁹ %
- The climatology error: ~1.2 × 10⁻⁴ % at worst (if `⟨q_v⟩` differs by 5%
  from the climatology)

the climatology approach is ~10⁵ × tighter than needed for typical
applications. If a multi-year run shows residual seasonal drift attributable
to TPW variations, the fix is to read the QV NetCDF in the preprocessor's
window loop BEFORE calling `spectral_to_native_fields!`, compute the actual
column-mass-weighted `⟨q_v⟩_global` per window, and pass it as a kwarg
override. This requires reordering `process_day` slightly and adds I/O. We
defer this until measurements show it's necessary.

### Should the target be configurable per dataset?

Yes — `target_ps_dry_pa` is a TOML knob. Defaults are sensible for any
modern reanalysis (ERA5, MERRA2, GEOS-IT, GEOS-FP) since they all target the
same Earth atmosphere. If a specific dataset has known biases that warrant a
different target, override it in the dataset's preprocessor config.

### Why apply the fix in the preprocessor instead of the runtime?

Three reasons:
1. **Self-contained binaries**: a v4 binary with `mass_fix_enabled = true` in
   its header is internally consistent — `m, am, bm, cm, dm` are all derived
   from the same corrected `sp`. A downstream tool reading the binary doesn't
   need to know about the fix; it just sees a clean dataset.
2. **Speed**: the fix is O(Nx × Ny) per window, computed once when the binary
   is built and never again. No runtime cost.
3. **Architectural mirroring of TM5**: TM5 puts the fix in `meteo.F90` because
   that's where it converts atmospheric input to model fields. Our equivalent
   step is the preprocessor (which produces the binary that the model reads).
   Same logical position, just offline.

### Is the absolute target value scientifically sensitive?

Not for transport. The choice of constant target (98500 vs 98726 vs 98972 Pa)
shifts every cell's `sp` by a constant; gradients, variances, and local
features are preserved. The only thing it changes is the absolute scale of
total mass (which is what we explicitly want to control). For diagnostic
applications that depend on absolute ps (e.g., comparison to a specific
observational dataset), consult the binary header to know what offset was
applied and undo it locally if needed.

### Why not instead remove only the time-mean drift, preserving absolute ps?

An alternative would be: compute the time-mean of `⟨sp⟩_area` over a batch
(e.g., a day or month), then pin each window's `⟨sp⟩_area` to that batch
mean. This eliminates window-to-window drift without changing the overall
scale. We considered this and rejected it because:
- It still has small day-to-day jumps at batch boundaries.
- It requires processing a full batch before writing any binary (annoying for
  daily processing).
- The benefit (preserving the absolute ERA5 climatology) is marginal because
  ERA5's climatology disagrees with Trenberth by ~0.4% anyway.
- For long simulations (months to years), pinning to a fixed constant gives
  zero drift across the entire simulation, while batch-mean pinning gives
  small jumps at batch boundaries.

If a specific use case wants the batch-mean approach, it can be added as
another `mass_fix.mode` option.

## 10. References

### TM5 source

- `deps/tm5-cy3-4dvar/base/src/meteo.F90` — runtime call sites
  - Lines 1361-1374: sp1 area-aver Match (start of meteo interval)
  - Lines 1458-1470: sp2 area-aver Match (end of meteo interval)
- `deps/tm5-cy3-4dvar/base/src/grid_type_ll.F90:1045-1083` — `llgrid_Match`
  dispatch
- `deps/tm5-cy3-4dvar/base/src/grid_type_ll.F90:1147-1155` — `Match_cell`,
  the `'area-aver'` case (the actual algorithm)
- `deps/tm5-cy3-4dvar/base/src/binas.F90:132` — `p_global = 98500.0`

### Our source

- `scripts/preprocessing/preprocess_spectral_v4_binary.jl:425-441` —
  `pin_global_mean_ps!` helper
- `scripts/preprocessing/preprocess_spectral_v4_binary.jl:983-998` — the call
  inside `spectral_to_native_fields!`
- `scripts/preprocessing/preprocess_spectral_v4_binary.jl:1230-1280` —
  `process_day` window loop with offset recording
- `config/preprocessing/era5_spectral_v4_tropo34_dec2021.toml` —
  `[mass_fix]` config block

### Verification probes

- `/tmp/probe_dry_mass_residuals.jl` — direct check of per-window `Σdm`
  against runtime tracer drift
- `/tmp/probe_compare_sp.jl` — compare our `⟨ps⟩` against the independent CDS
  surface_pressure dataset

### External

- Trenberth, K. E., and L. Smith, 2005: "The Mass of the Atmosphere: A
  Constraint on Global Analyses". _Journal of Climate_, **18**, 864-875.
  doi:10.1175/JCLI-3299.1
- Trenberth, K. E., J. T. Fasullo, and J. Kiehl, 2009: "Earth's Global Energy
  Budget". _Bull. Amer. Meteor. Soc._, **90**, 311-323.
  doi:10.1175/2008BAMS2634.1 (mean TPW reference)
- ERA5 documentation, ECMWF: see "ERA5: Fact sheet — atmospheric
  reanalyses". The 4DVar assimilation is documented as not strictly
  mass-conserving by design.

## 11. Quick reference for future debugging

If you see any of the following, suspect this fix:

| Symptom | Diagnosis | Action |
|---|---|---|
| Σm drift > 10⁻⁵ %/day on uniform IC | Mass fix disabled or absent in binary | Check `mass_fix_enabled` in binary header; enable if false |
| Per-cell ps disagrees with raw ERA5 by ~+400 Pa | Mass fix is enabled (expected) | Subtract `ps_offsets_pa_per_window[t]` from each cell to recover raw ps |
| Tracer mass drifts but column mean is constant | Runtime `mass_fixer` is hiding the binary drift | Check binary `Σm` directly via `probe_dry_mass_residuals.jl` |
| ⟨ps⟩ in binary disagrees with `target_ps_dry / (1 - qv_global)` | Probe is wrong, OR fix didn't run | Verify `pin_global_mean_ps!` was called; check window loop kwargs |

If you want to **disable** the fix for a regression test or diagnostic
comparison:

```toml
[mass_fix]
enable = false
```

The binary header will record `"mass_fix_enabled": false` and the offsets
will be zero. All other behavior is unchanged.

---

*Document created 2026-04-07 alongside the implementation. Update this file
in the same commit as any changes to `pin_global_mean_ps!` or its call sites.*
