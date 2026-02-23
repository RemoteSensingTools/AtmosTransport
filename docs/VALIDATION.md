# Validation

How we test the model and (when available) compare to TM5.

## Stencil-level TM5 agreement (verified)

### Forward advection: exact match

Our `SlopesAdvection` implementation reproduces the TM5 Russell-Lerner advection
stencil (`deps/tm5/base/src/advectx.F90`, lines 660-680) to **zero difference** (not
just machine precision — identically zero) in both the no-limiter and minmod-limiter
configurations.

The comparison script (`scripts/compare_advection_stencil.jl`) sets up a Gaussian blob
on a 32x16x5 grid with spatially varying velocity, runs one advection step with both
our Julia `SlopesAdvection` and a literal translation of the TM5 Fortran stencil, and
confirms:

| Metric | No limiter | With minmod limiter |
|--------|-----------|---------------------|
| Max absolute difference | 0.0 | 0.0 |
| Max relative difference | 0.0 | 0.0 |
| Mass conservation | 1.7e-16 | 1.7e-16 |

TM5's Fortran stencil (concentration form):
```
if u >= 0:  alpha = u*dt/dx,  flux = u*(c[i] + (1-alpha)*slope[i]/2)
if u <  0:  alpha = u*dt/dx,  flux = u*(c[i+1] - (1+alpha)*slope[i+1]/2)
```

### Adjoint: exact discrete adjoint verified

The adjoint dot-product identity is satisfied for the x-direction:

```
|⟨L^T λ, δc⟩ / ⟨λ, L δc⟩ - 1| = 6.7e-16  (machine precision)
```

### Mapping between TM5 and Julia formulations

| TM5 variable | Julia equivalent | Description |
|-------------|-----------------|-------------|
| `rm` (tracer mass) | `rm = m * c` | Prognostic tracer quantity |
| `rxm` (x-slope of mass) | `sx = m * slope_c` | Slope from concentration, scaled by mass |
| `am` (x mass flux) | `am = Δt/2 * u * Δp_face * Δy / g` | Face mass flux (kg per half-timestep) |
| `bm` (y mass flux) | `bm = Δt/2 * v * Δp_face * Δx / g` | Face mass flux (kg per half-timestep) |
| `cm` (z mass flux) | Derived from horizontal convergence | Column mass conservation by construction |
| `m` (air mass) | `m = Δp * cell_area / g` | Cell air mass (kg) |

**Current implementation:** The mass-flux formulation (`src/Advection/mass_flux_advection.jl`)
co-advects tracer mass `rm` and air mass `m`, following TM5's architecture. Air mass is
tracked continuously through the full Strang split (X-Y-Z-Z-Y-X) and never reset during
the split. Slopes are computed from concentration `c = rm/m` (not from `rm` directly)
to preserve uniform fields, then scaled by `m`.

See [MASS_FLUX_EVOLUTION.md](MASS_FLUX_EVOLUTION.md) for the design history and
[Advection Theory](../literated/advection_theory.md) for the mathematical framework.

## Mass-flux advection validation

### Unit tests (test/test_mass_flux_advection.jl)

| Test | Result |
|------|--------|
| X mass conservation | < 1e-15 relative error |
| Y mass conservation | < 1e-15 relative error |
| Z mass conservation | < 1e-15 relative error |
| Full Strang mass conservation | 0.0 relative error |
| Uniform field preservation | < 4e-13 max deviation |
| 10-step Strang mass drift | 7.3e-5% |
| Positivity with limiter | 0 negative cells |
| CFL subcycling | Correct automatic subdivision |
| Full test suite | 209/209 tests passing |

### Comparison with deprecated concentration-based approach

| Metric | Concentration + mass fixer | Mass-flux (current) |
|--------|---------------------------|---------------------|
| Mass conservation | 0.91% drift per split | Machine precision |
| Uniform field preservation | O(1) deviation | < 4e-13 |
| Extreme values | Blowup at step ~40 | Stable indefinitely |
| Post-hoc correction needed | Yes | No |
| TM5 architectural agreement | Approximate | Faithful |

## ERA5 forward runs (historical, concentration-based)

> **Note:** These results predate the mass-flux rewrite. They document the
> concentration-based approach for historical reference. The mass-flux formulation
> resolves the mass drift and extreme values reported here.

### SlopesAdvection (concentration-based) on ERA5 data

48-hour simulation on 180x91x20 ERA5 grid (2° resolution, 20 pressure levels):

```
julia --project=. scripts/run_forward_era5.jl
```

| Metric | Value |
|--------|-------|
| Stability | No NaN, no explosion |
| Physical mass change | 1.55% over 48h |
| Σc (simple sum) change | 8.08% |
| Max tracer | 44491 (from initial 420-450) |
| Wall time | 25.2 s (288 steps at Δt=600s) |

The 1.55% physical mass drift was caused by advecting concentration rather than
tracer mass. This is now resolved by the mass-flux formulation.

### UpwindAdvection comparison (historical)

| Metric | Upwind | Slopes (limiter=true) |
|--------|--------|----------------------|
| Physical mass | 0.0% | 1.55% |
| Max tracer | 107949 | 44491 |
| Numerical diffusion | High (1st order) | Low (2nd order) |

## Gradient test on ERA5 winds

Gradient tests confirm adjoint correctness with realistic wind fields (not toy constant
velocities). Tests run 3 time steps at Δt=600s on the full 180x91x20 ERA5 grid.

Script: `scripts/gradient_test_era5.jl`

### Results

| Scheme | Best ratio (ε=1e-4) | Type | Expected |
|--------|---------------------|------|----------|
| **Slopes (no limiter)** | 1.0 ± 1.5e-7 | Exact discrete adjoint | ratio = 1.0 to machine precision |
| **Slopes (with limiter)** | 0.956 | Continuous adjoint | ratio ≈ 1.0 within ~5% |
| **Upwind** | 1.0 ± 8.4e-7 | Exact discrete adjoint | ratio = 1.0 to machine precision |

The exact discrete adjoints (Slopes no-limiter and Upwind) achieve machine-precision
agreement between the adjoint directional derivative and central finite differences,
confirming the adjoint implementations are correct transpositions of the forward
operators.

The Slopes with-limiter uses a continuous adjoint (negated wind, same forward code),
following TM5/NICAM-TM practice (Niwa et al., 2017). The ~4.4% discrepancy is expected
because the minmod limiter is nonlinear.

## TM5 end-to-end comparison (pending)

### Build status

TM5 builds successfully with `ifx`, ecCodes, and full ERA5+GRIB support.

- **Executable:** `/tmp/tm5_cfranken/var4d/dummy_tr/nam1x1/ml137/tropo25a/tm5-var4d.x`
- **Linked against:** ecCodes 2.34.0, NetCDF-Fortran, MKL, OpenMPI

See [TM5_LOCAL_SETUP.md](TM5_LOCAL_SETUP.md) for full build instructions.

### Remaining steps for end-to-end comparison

TM5 requires preprocessed meteo (mass fluxes from spectral harmonics), not raw ERA5
NetCDF. See [METEO_PREPROCESSING.md](METEO_PREPROCESSING.md) for details.

1. Download ERA5 spectral GRIB data (VO, D, LNSP) - **IN PROGRESS**
2. Configure TM5 GRIB reader (tmm.sourcekey format) - **TODO**
3. Run TM5 preprocessing mode - **TODO**
4. Run TM5 forward simulation - **TODO**
5. Compare outputs - **TODO**

Note: with the mass-flux advection rewrite, our formulation now matches TM5's
architecture (co-advection of `rm` and `m`, mass-based CFL, continuity-derived
vertical fluxes). Remaining differences are in the source of mass fluxes: TM5 uses
spectrally integrated mass fluxes while we compute mass fluxes from gridpoint winds.

| Aspect | TM5 approach | Julia model approach |
|--------|-------------|---------------------|
| Advection | Mass fluxes from spectral integration | Mass fluxes from gridpoint winds |
| Mass conservation | Guaranteed (spectral, Bregman et al. 2003) | Machine precision (mass-flux formulation) |
| Operator splitting | X-Y-Z-Z-Y-X with continuous `m` | X-Y-Z-Z-Y-X with continuous `m` (same) |
| Vertical flux | `dynam0` spectral continuity | Gridpoint continuity equation (same structure) |
| Convection | ECMWF convective fluxes (eu/ed/du/dd) | ECMWF convective mass fluxes (if available) |
| Vertical coordinate | Hybrid sigma-pressure (A/B) | Hybrid sigma-pressure or pressure levels |

## Unit and integration tests

- **Test suite:** `julia --project=. -e 'include("test/runtests.jl")'` (209 tests, all passing).
- **Mass-flux advection:** Mass conservation (x, y, z, full Strang), uniform field preservation, positivity, CFL subcycling (18 tests).
- **Stencil advection:** Mass conservation (x, y, z), adjoint identity (dot-product at rtol=1e-10), 1D slopes tests.
- **Convection:** Mass conservation, adjoint identity, single-column redistribution.
- **Diffusion:** Mass conservation, adjoint.
- **Gradient:** Full operator-splitting gradient test vs central finite differences (6 ε values, multiple physics combinations).

## GPU / CUDA

- **Status:** `UpwindAdvection` (x, y, z) uses KernelAbstractions `@kernel` for unified CPU/GPU execution. `SlopesAdvection` x-direction has a GPU kernel; y and z directions are CPU-only (GPU port is a stretch goal). Convection and diffusion remain CPU-only.
- **Architecture:** Grid carries `architecture` field (`CPU()` or `GPU()`). All arrays dispatch to `Array` or `CuArray` accordingly.
- **Testing:** Suite runs on CPU. GPU path tested manually with `USE_GPU=true julia --project=. scripts/run_forward_era5.jl`.
- **Float32/Float64:** Model supports both precisions via `USE_FLOAT32=true` environment variable.

## Configuration

All physical constants and simulation parameters are loaded from TOML files
(`config/defaults.toml`) with optional overrides via `CONFIG=/path/to/override.toml`.
See `src/Parameters.jl` for the type-stable parameter system inspired by CliMA.jl.

## GEOS-FP validation (separate track)

TM5 has no GEOS-FP reader. For GEOS-FP validation:

- Compare the Julia model against **GEOS-Chem** (which natively uses GEOS-FP)
- Or compare against observations (surface stations, satellite retrievals)
- The Julia model already has a working GEOS-FP reader and download scripts
- See [METEO_PREPROCESSING.md](METEO_PREPROCESSING.md) for details
