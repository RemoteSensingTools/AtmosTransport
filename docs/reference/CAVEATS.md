# Caveats and Troubleshooting

This page documents important operational caveats and known gotchas discovered
during model development and validation. Read this before running production
simulations — several of these issues can silently produce wrong results without
any error messages.

## Met Data Caveats

### GEOS Mass Flux Accumulation Time (GEOS-IT AND GEOS-FP)

**Severity: Critical — 8× transport speed error if ignored**

Both GEOS-IT C180 and GEOS-FP C720 files store mass fluxes (MFXC, MFYC)
accumulated over the **dynamics timestep** (~450 s), *not* the full 1-hour met
interval. The `tavg` prefix means time-averaged: MFXC is the per-dynamics-step
accumulated mass flux, averaged over the 1-hour output interval. Without
correction, the model reads these as 1-hour fluxes, making transport ~8× too slow.

**Diagnosis:** Compare derived wind speeds against expected climatology.
- GEOS-IT C180: CX-derived winds vs A3dyn U/V ratio = 0.13 (without fix) → 1.0 (with fix)
- GEOS-FP C720: surface RMS wind = 0.87 m/s (without fix) → 6.9 m/s (with fix)

**Fix:** Set `mass_flux_dt` in the TOML config:

```toml
[met_data]
mass_flux_dt = 450   # GEOS dynamics timestep in seconds (both GEOS-IT and GEOS-FP)
```

The default is 450 s. All GEOS run configs must include this.

**Stability note:** With the corrected fluxes, upper-atmosphere CFL can exceed 1.
Use `merge_levels_above_Pa = 3000` to collapse thin stratospheric levels, or
reduce `dt` to ~150 s.

**Binary reprocessing:** If you previously preprocessed GEOS-FP/IT data to
binary format with the wrong `mass_flux_dt` (3600), those binaries produce
incorrect transport. Delete them and reprocess with `mass_flux_dt = 450`.

### GEOS-IT Vertical Level Ordering

**Severity: Critical — completely wrong fields if levels are inverted**

GEOS-IT C180 files store vertical levels **bottom-to-top** (k=1 = surface,
k=72 = TOA), whereas GEOS-FP C720 uses **top-to-bottom** (k=1 = TOA,
k=72 = surface). Using the wrong convention silently corrupts the B-ratio
and column-mass computation, leading to extreme CFL values and NaN.

**Auto-detection:** The cubed-sphere reader (`geosfp_cubed_sphere_reader.jl`)
automatically detects the ordering by comparing DELP at k=1 versus k=Nz. If
the surface is at k=1, it flips with `reverse!(raw, dims=4)`.

**No user action required** — the auto-flip handles both conventions. This
caveat is documented here so that anyone adding a new met reader or
preprocessing script is aware of the convention difference.

### GEOS-FP Cubed-Sphere Panel Convention

**Severity: Critical — wrong transport at panel boundaries if panels are misordered**

The GEOS-FP/GEOS-IT native cubed-sphere files use a panel numbering convention
that differs from the standard gnomonic projection:

| Panel | File convention (nf=1..6) | Gnomonic convention |
|-------|---------------------------|---------------------|
| 1–2 | Equatorial, standard orientation | Equatorial (panels 1–4) |
| 3 | North pole | Equatorial (cont.) |
| 4–5 | Equatorial, rotated 90° CW | North pole (panel 5) |
| 6 | South pole | South pole (panel 6) |

The panel connectivity in `panel_connectivity.jl` follows the **file
convention** (fixed Feb 2026). At rotated panel boundaries, the mass flux
components swap: `cgrid_to_staggered_panels` handles the MFXC ↔ MFYC
exchange and sign flips automatically.

**Impact on preprocessed data:** Binary files generated before the Feb 2026
fix use the old (incorrect) boundary fluxes. Either re-preprocess or use
`netcdf_dir` to read raw NetCDF files at runtime.

**Impact on output/regridding:** Runtime output, visualization, and
ConservativeRegridding tree construction now use the explicit
`panel_convention` carried by `CubedSphereMesh` / binary headers. Native
GEOS-FP/IT files must select `GEOSNativePanelConvention()` (or
`panel_convention="geos_native"` in metadata); ERA5-derived CS binaries use
the default `GnomonicPanelConvention()`. Cell areas are identical across
panels for these two unstretched conventions, so air-mass computation is
unaffected by the reordering.

### ERA5: Spectral vs. Gridpoint Mass Fluxes

Two ERA5 mass-flux pipelines are available:

1. **Spectral (recommended):** `preprocess_spectral_massflux.jl` converts ERA5
   spectral harmonics (VO, D, LNSP) to mass-conserving mass fluxes, following
   TM5's approach (Bregman et al. 2003). This achieves near-zero mass drift.
   Config: `config/preprocessing/spectral_june2023.toml`.

2. **Gridpoint (stopgap):** `preprocess_mass_fluxes.jl` derives mass fluxes
   from gridpoint u/v winds. This introduces ~0.9% mass drift per simulation
   month. Use only if spectral GRIB data is unavailable.

**Recommendation:** Always use the spectral pipeline for production runs.
The gridpoint approach is retained for quick prototyping with readily available
ERA5 gridpoint data.

### Advection: Lin-Rood vs Strang Splitting (Cubed-Sphere)

**Severity: Medium — directional bias artifacts at panel boundaries**

Standard Strang splitting (X→Y→Z→Z→Y→X) introduces directional bias at
cubed-sphere panel boundaries, visible as wave artifacts at high latitudes
where panels meet at oblique angles. The bias arises because X-advection
always "sees" the field before Y-advection within a horizontal sweep.

**Fix:** Enable Lin-Rood cross-term splitting in the TOML config:

```toml
[advection]
scheme   = "ppm"
linrood  = true
```

This implements FV3's `fv_tp_2d` algorithm (Lin & Rood, 1996): for each
horizontal sweep, it computes both X-first and Y-first PPM orderings from
the original field and averages the result. This eliminates the directional
bias at the cost of extra halo exchanges and corner ghost-cell fills.

**Performance:** ~2–3× GPU time compared to standard Strang splitting due
to three halo exchanges and corner fills per horizontal sweep (vs one each
for Strang). For GEOS-IT C180 CATRINE with 4 tracers, GPU time increases
from ~0.76 s/win to ~1.3 s/win.

**Recommendation:** Use `linrood = true` for all cubed-sphere PPM production
runs. The cost is modest and eliminates a systematic error source.

### Mass Conservation on Cubed-Sphere Grids

**Severity: High — mass drift degrades tracer correlations without fixers**

Cubed-sphere advection has three levels of mass conservation enforcement:

1. **Pressure fixer** (`_cm_pressure_fixer_kernel!`): Incorporates the DELP
   tendency into the vertical mass flux `cm`, so that `m_ref` advances along
   the prescribed pressure trajectory across sub-steps. This is the primary
   conservation mechanism and reduces mass drift from ~% to ~ppm level.

2. **Per-cell mass fixer** (`mass_fixer = true` in `[advection]`, default):
   After each advection step, scales tracer mass as `rm *= m_ref / m_evolved`
   to correct any residual drift. Without this, R² against reference data
   drops from ~0.5 to ~0.01 in 8 days.

3. **Global mass fixer** (`_apply_global_mass_fixer!`): Uniform correction
   across all cells to match pre-advection total mass. Corrects Strang
   splitting drift at CS panel boundaries (~3 ppm/window). GPU time overhead:
   ~0.2 s/win.

```toml
[advection]
mass_fixer = true    # default; disable only for diagnostics
```

### Dry-Air Transport

**Severity: Low — automatic, no user configuration needed**

When specific humidity (QV) is available from the met data, the model
automatically converts all mass-related fields to a dry-air basis:
DELP, am, bm, CMFMC, and DTRAIN are multiplied by `(1 - QV)` on the GPU.
This ensures tracer mixing ratios are computed with respect to dry air mass,
consistent with atmospheric chemistry conventions.

The conversion is gated on the `qv_loaded` flag — when QV is unavailable
(e.g., quickstart mode), the model falls back to wet-air transport
transparently. CMFMC/DTRAIN dry corrections are only applied when freshly
uploaded (not `:cached`) to avoid double-correction.

## Configuration Caveats

### Diffusion Required for Realistic Column-Mean Output

**Severity: High — visually misleading output without diffusion**

Without boundary-layer diffusion enabled, surface-emitted CO₂ remains trapped
in the bottom model level. The column-mean diagnostic then dilutes the signal
by a factor of ~72 (number of vertical levels), making transport appear
invisible for the first several days.

**Fix:** Always enable diffusion in the TOML config when computing column-mean
diagnostics:

```toml
[diffusion]
type    = "boundary_layer"
Kz_max  = 100.0
H_scale = 8.0
```

This adds negligible GPU cost (<1% per window) but is essential for physically
meaningful column-mean or satellite-comparable output.

### Output File Appending

NetCDF output files are opened in **append mode** (`NCDataset(path, "a")`).
Running the model twice without deleting the output file will concatenate the
new timesteps after the old ones, producing a file with (e.g.) 720 timesteps
for a 30-day run that was executed twice.

This can produce confusing artifacts in animations: the "day 23" frame may
actually be the start of a second run overlaid on the first.

**Fix:** Always delete the output file before a fresh run:

```bash
rm ~/data/output/my_output.nc
julia --project=. scripts/run.jl config/runs/my_config.toml
```

### TOML Output Field Naming

When using the string shorthand for output fields, the species name defaults
to the output variable name:

```toml
# This sets species = :co2, which is correct:
co2 = "column_mean"

# But this sets species = :co2_column_mean, which will error
# (no tracer named co2_column_mean):
co2_column_mean = "column_mean"
```

To use a variable name that differs from the tracer name, use the dict form:

```toml
[output.fields]
co2_column_mean = {type = "column_mean", species = "co2"}
co2_surface     = {type = "surface_slice", species = "co2"}
co2_800hPa      = {type = "sigma_level", species = "co2", sigma = 0.8}
```

### Layer Merging for Thin Upper Levels

The upper atmosphere has very thin model levels that require small timesteps.
The `merge_levels_above_Pa` option merges these into coarser layers:

```toml
[grid]
merge_levels_above_Pa = 1000.0   # merge levels above ~10 hPa
```

This reduces the effective number of vertical levels and allows larger
timesteps in the upper atmosphere. Met data is regridded vertically by summing
DELP, air mass, and mass fluxes within merged groups.

**Limitation:** Only supported for NetCDF mode (not binary preprocessed files).

## Developer Caveats

### KernelAbstractions: No `return` in Kernels

`@kernel` functions (KernelAbstractions.jl) do not allow `return` statements.
Use boolean flag variables instead:

```julia
# WRONG — will error at compile time:
@kernel function my_kernel!(out, data, Nz)
    i, j = @index(Global, NTuple)
    for k in 1:Nz
        if condition(data, i, j, k)
            out[i, j] = data[i, j, k]
            return  # ← NOT ALLOWED
        end
    end
end

# CORRECT — use a flag:
@kernel function my_kernel!(out, data, Nz)
    i, j = @index(Global, NTuple)
    found = false
    for k in 1:Nz
        if !found && condition(data, i, j, k)
            out[i, j] = data[i, j, k]
            found = true
        end
    end
end
```

### GPU Extension Loading Order

`using CUDA` or `using Metal` must appear **before** `using AtmosTransport` in
custom run scripts to trigger the weakdep extension and avoid Julia world-age
issues around GPU array methods. The canonical runner
(`scripts/run_transport.jl`) handles this automatically based on
`[architecture] use_gpu` and `backend`. Metal runs require
`[numerics] float_type = "Float32"`.

### Cubed-Sphere Tracer Storage

Cubed-sphere tracers live in local `rm_panels` (NTuple{6} of haloed 3D arrays),
not in `model.tracers`. When calling `write_output!`, pass them explicitly:

```julia
write_output!(writer, model; tracers=cs_tracers)
```

### Safe Division Guards

Any kernel that divides tracer mass (`rm`) by air mass (`m`) must guard against
division by zero:

```julia
mixing_ratio = _m > zero(FT) ? rm / _m : zero(FT)
```

This pattern appears at 9+ sites in the cubed-sphere advection and diagnostic
kernels. Always use it when computing mixing ratios from mass fields.

## References

- Bregman et al. (2003): Mass-conserving wind fields. ACP 3, 447–457.
- Martin et al. (2022): GCHP v13 — native CS mass fluxes. GMD 15, 8731–8748.
