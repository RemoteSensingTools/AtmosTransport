# ERA5 spectral LNSP: global mass drift vs spatial structure

This note documents **diagnostics** on ERA5 **spectral log surface pressure (LNSP)** converted to gridpoint surface pressure `ps = exp(LNSP)` on the same target grid and transform as `scripts/preprocessing/preprocess_spectral_v4_binary.jl` (720×361, `T = N_lon/2 - 1`, half-cell longitude shift). It addresses one question:

> When total atmospheric mass implied by LNSP changes from one analysis to the next, is that change **mostly a uniform shift in `ps` everywhere**, or **mostly spatial redistribution** (local gains and losses that cancel in the global integral)?

The metrics below are **purely diagnostic** (they describe the GRIB fields). They do not, by themselves, identify the cause (assimilation increments, analysis cycle, spectral truncation, etc.).

---

## Global context: integrated mass still “jumps”

Integrated column mass uses the hydrostatic surface-pressure integral (same sign convention as the spectral preprocessor):

\[
M = \frac{1}{g} \sum_{ij} ps_{ij} \, A_{ij}
\]

Hourly spectral LNSP for a month (December 2021, hourly files under `met/era5/spectral_hourly/`) produces a **global** `M(t)` that is smooth in places but shows **sharp steps** when plotted against time; the implied **global** mass tendency `dM/dt` has large spikes aligned with those steps (see figure below). That behaviour is **systematic in time** (repeatable structure in the timeseries), not i.i.d. noise.

![Global mass from LNSP and implied step tendency (December 2021 hourly, v4-consistent grid/transform)](../assets/era5_lnsp_global_mass_timeseries.png)

*Figure: Top — `M(t) − M(t₀)` from spectral LNSP. Bottom — `ΔM/Δt` between consecutive GRIB times (proxy for implied mass tendency). Script: `scripts/diagnostics/spectral_lnsp_total_mass_timeseries.jl`.*

---

## Spatial decomposition of hourly `Δps`

For each consecutive pair of analysis times `(t_k, t_{k+1})` define

\[
\Delta ps_{ij} = ps_{ij}(t_{k+1}) - ps_{ij}(t_k).
\]

Let `A_{ij}` be cell area and `W = \sum_{ij} A_{ij}`. The **area-weighted mean increment**

\[
\overline{\Delta ps} = \frac{1}{W} \sum_{ij} A_{ij}\,\Delta ps_{ij}
\]

is the **spatially uniform** part of `Δps` (the part that changes the global integral `M` in proportion to `∑ A`). Define the **residual** (redistribution with **zero** area-weighted mean)

\[
r_{ij} = \Delta ps_{ij} - \overline{\Delta ps}.
\]

**Area-weighted RMS** for a field `x` on the grid:

\[
\mathrm{RMS}_w(x) = \sqrt{ \frac{\sum_{ij} A_{ij}\, x_{ij}^2}{W} }.
\]

### Metric 1 — `ρ_res` (global vs local in the RMS sense)

\[
\rho_{\mathrm{res},k} =
\frac{\mathrm{RMS}_w(r)}{\mathrm{RMS}_w(\Delta ps)}
\quad\text{at step } k.
\]

- If **`ρ_res ≪ 1`**, most of the **RMS** of `Δps` is captured by a **single offset** `Δps ≈ \overline{\Delta ps}` everywhere: the field “breathes” uniformly.
- If **`ρ_res ≈ 1`**, subtracting `\overline{\Delta ps}` **barely reduces** RMS: the **map-scale variability** of `Δps` is as large as typical `|Δps|`, while the **uniform component that moves global mass** is **small in RMS** compared to weather-scale increments.

### Metric 2 — `ρ_lon` (zonal vs eddy structure)

Let `\langle \Delta ps \rangle_j` be the zonal mean of `Δps` at latitude index `j`. Define the anomaly

\[
\Delta ps'_{ij} = \Delta ps_{ij} - \langle \Delta ps \rangle_j.
\]

\[
\rho_{\mathrm{lon},k} =
\frac{\mathrm{RMS}_w(\Delta ps')}{\mathrm{RMS}_w(\Delta ps)}
\quad\text{at step } k.
\]

- **`ρ_lon ≪ 1`**: increments are **nearly zonal** (little variation along longitude at fixed latitude).
- **`ρ_lon ≈ 1`**: **zonal asymmetry** (longitude structure) is as large as the total RMS — **synoptic / 2D** structure dominates over pure “rings of latitude”.

---

## Numerical results (December 2021, hourly LNSP)

Dataset: `era5_spectral_*_lnsp.gb` in `met/era5/spectral_hourly/`, **743** consecutive hourly pairs, grid **720 × 361**, spectral truncation **T = 359** (capped by file).

| Quantity | Value |
|----------|--------|
| Median `ρ_res` | **1.0000** |
| Mean `ρ_res` | **1.0000** |
| Median `ρ_lon` | **0.9891** |
| Mean `ρ_lon` | **0.9888** |

**Reading:** For this month and product, **hour-to-hour `Δps` is dominated by spatial patterns on the map** (`ρ_res ≈ 1`): the **global mass change** between times corresponds to a **small uniform component** relative to typical gridcell `Δps` in an RMS sense. At the same time, **`ρ_lon ≈ 1`**: the increments are **not** well described as latitude-only rings; **longitude structure** is almost as large as the total RMS. Together, these are **consistent with systematic local (lat–lon) redistribution** superposed on a **smaller** globally coherent pressure shift that drives the **integrated** `M(t)` steps.

The step-to-step values of `ρ_res` stay **pinned near 1** across the month (not a handful of outliers), which supports **systematic** behaviour of the decomposition rather than occasional glitches.

![ρ_res per consecutive pair (743 steps)](../assets/era5_lnsp_drift_rho_res_timeseries.png)

*Figure: `ρ_res` for every hourly step. Values remain at ~1 for essentially all pairs — removing the area-mean `Δps` does not reduce RMS in practice.*

---

## Maps and Hovmöller views

### Time-mean absolute residual `|r|`

If drift were **only** a global uniform `Δps`, **`r` would be identically zero**. The time mean of `|r_{ij}|` is **non-zero everywhere** with **large regional contrast**, i.e. **preferred locations** for mass exchange between columns (still consistent with near-cancellation when integrated).

![Time-mean \|Δps − area-mean(Δps)\| (Pa)](../assets/era5_lnsp_drift_mean_abs_residual.png)

*Figure: Mean over all hourly pairs of `|r_{ij}|`. Brighter regions are where **systematic local redistribution** (relative to a global mean increment) is largest on average.*

### Zonal-mean `Δps` vs time

A Hovmöller of zonal-mean `Δps` shows **latitude–time structure** (propagating features, bands, cycle-related modulation). This is **not** the signature of a pure global offset (which would appear as a **flat** line in latitude at each time).

![Zonal-mean Δps (Pa): latitude vs hours since start](../assets/era5_lnsp_drift_hovmoller_zonal_dps.png)

*Figure: Zonal-mean `Δps` as a function of latitude and time (hours since first sample).*

### RMS over time of zonal-mean `Δps` vs latitude

Some latitude bands show **larger** typical zonal-mean increments than others — again **inconsistent** with “only a global scalar `Δps` every hour.”

![RMS over time of zonal-mean Δps vs latitude](../assets/era5_lnsp_drift_zonal_rms_vs_lat.png)

*Figure: `sqrt( mean_k zonal_mean(\Delta ps)_j^2 )` vs latitude.*

---

## Reproducibility

Scripts (Julia, project environment):

- Global `M(t)` and `ΔM/Δt`:  
  `julia --project=. scripts/diagnostics/spectral_lnsp_total_mass_timeseries.jl <spectral_lnsp_dir> [--out figure.png]`
- Spatial metrics and figures (loads full spectral coefficient stack; **~1.5 GiB RAM** for one month of hourly T359):  
  `julia --project=. scripts/diagnostics/spectral_lnsp_spatial_drift.jl <spectral_lnsp_dir> [--out-prefix path/prefix]`

Shared spectral/GRIB helpers: `scripts/diagnostics/spectral_lnsp_grib_utils.jl`.

Figures committed under `docs/src/assets/` for this documentation:

- `era5_lnsp_global_mass_timeseries.png`
- `era5_lnsp_drift_mean_abs_residual.png`
- `era5_lnsp_drift_hovmoller_zonal_dps.png`
- `era5_lnsp_drift_zonal_rms_vs_lat.png`
- `era5_lnsp_drift_rho_res_timeseries.png`

To refresh figures after re-running diagnostics, copy outputs from your chosen `--out-prefix` directory into `docs/src/assets/` with the names above (or regenerate with matching names).

---

## Limitations

- **One month / one configuration** is shown here; numbers will change for other periods or cadences (e.g. 6-hourly LNSP).
- Metrics characterize **LNSP-derived `ps`** on a **specific grid/transform**; they are not a statement about **TM5** or **AtmosTransport** mass fixers.
- **Cause** of steps (assimilation, analysis times, digital filtering) requires **ECMWF / ERA5 documentation**, not inferred from these plots alone.

---

## Summary

Diagnostic decomposition of hourly **ERA5 spectral LNSP** on the v4 preprocessor grid shows:

1. **Global** integrated mass **`M(t)`** exhibits **large, structured** changes in time (figure 1).
2. **`ρ_res ≈ 1`** and **`ρ_lon ≈ 1`** on **every** hourly step imply those changes are **not** explained as a **spatially uniform `Δps`** field at map RMS; **local (lat–lon) increments are dominant** and **repeat systematically** (`ρ_res` time series).
3. **Time-mean `|r|`** maps and **zonal-mean Hovmöller** plots show **persistent spatial and temporal organisation** — **consistent with systematic local redistribution** superposed on a **smaller** globally coherent component that moves **`M`**.

This supports treating ERA5 LNSP-driven **global mass drift** in transport models as **more than a single scalar “global pump”**: **spatial structure in `ps` increments is the norm**, not the exception, for this dataset.
