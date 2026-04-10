# 24-Hour Plan: Cross-Grid ERA5 Validation (LatLon + ReducedGaussian + CubedSphere)

**Date**: 2026-04-10
**Goal**: 2-day ERA5 transport runs on all 3 grids × CPU/GPU × F32/F64
with Catrine natural CO2 IC + GridFED fossil CO2 fluxes.
All grids should agree within expected numerical precision.

---

## Target Matrix

| Grid | CPU F64 | CPU F32 | GPU F64 (curry) | GPU F32 (wurst) |
|------|---------|---------|-----------------|-----------------|
| LatLon 0.5° | Primary | Cast | Primary | Cast |
| ReducedGaussian N320 | Primary | Cast | Primary | Cast |
| CubedSphere C90 | Primary | Cast | Primary | Cast |

"Cast" = read F64 binary, convert to F32 at load time.
"Primary" = native precision run.
GPU F64 only on curry (A100). GPU F32 on either server.

---

## Work Split — Two Parallel Streams

### STREAM A: Codex — Runtime, ReducedGaussian, Surface Fluxes

**Owns these files** (no one else touches them):
- `src_v2/Operators/Advection/StrangSplitting.jl`
- `src_v2/Models/DrivenSimulation.jl`
- `scripts/run_transport_binary_v2.jl`
- `src_v2/MetDrivers/TransportBinaryDriver.jl`
- `test_v2/test_driven_simulation.jl`
- `test_v2/test_run_transport_binary_v2.jl`
- `test_v2/test_basis_explicit_core.jl`
- `test_v2/test_transport_binary_reader.jl`
- All existing config files in `config/runs/era5_*_v2_*.toml`

**Tasks (in priority order)**:

1. **Commit current RG + runtime work** — get the accumulated fixes
   (subcycling, no-reset, file IC, GPU adapt, legacy binary compat) into
   a clean commit.

2. **Validate ReducedGaussian through 48h** — run the N320 binary for
   Dec 1-2 with Catrine CO2 IC, verify stability and mass conservation.
   Export snapshots at t=0, 6h, 12h, 24h, 48h.

3. **Wire surface flux injection into DrivenSimulation**:
   - Add a `SurfaceFluxSource` type (or simple struct) that holds a 2D
     flux field [kg CO₂/m²/s] on the model grid.
   - In the substep loop, after advection, call
     `_inject_source_kernel!` from `CellKernels.jl` to add
     `flux × cell_area × dt` to the bottom-level tracer mass.
   - Support two tracers: `natural_co2` (IC from Catrine, no flux) and
     `fossil_co2` (IC = 0, flux from GridFED).

4. **Add F32 cast layer**:
   - In `run_transport_binary_v2.jl`, after loading window data, if
     config says `float_type = "Float32"`, convert all arrays via
     `Float32.(array)` before building the model.
   - The transport binary stays F64 on disk; only the runtime arrays
     are F32.

5. **GPU smoke test on wurst** — run LatLon + RG for 1 window on GPU
   to verify the Adapt pathway works. Use `[architecture] use_gpu = true`
   in config.

6. **Produce LatLon + RG run configs** for the full matrix:
   - `era5_latlon_v2_catrine_2day_f64.toml`
   - `era5_latlon_v2_catrine_2day_f32.toml`
   - `era5_rg_v2_catrine_2day_f64.toml`
   - `era5_rg_v2_catrine_2day_f32.toml`
   Each config specifies: binary path, Catrine IC, GridFED fossil flux,
   2 tracers, 48 windows (2 days).

### STREAM B: Claude — CS Preprocessing, CS Driver, Visualization

**Owns these files** (no one else touches them):
- `src_v2/Grids/PanelConnectivity.jl`
- `src_v2/Grids/CubedSphereMesh.jl`
- `src_v2/Operators/Advection/HaloExchange.jl`
- `src_v2/Operators/Advection/CubedSphereStrang.jl`
- `test_v2/test_cubed_sphere_advection.jl`
- `scripts/preprocessing/regrid_latlon_to_cs_binary_v2.jl` (NEW)
- `scripts/visualization/plot_cross_grid_comparison.py` (NEW)
- All new CS-specific config files

**Tasks (in priority order)**:

1. **Build ERA5→C90 binary generator**:
   - Read 0.5° LatLon F64 transport binary (existing Dec 1-2 files).
   - Regrid cell-center winds (U, V) and surface pressure (ps) to C90
     panels using conservative regridding from `regrid_utils.jl`.
   - Compute per-panel face mass fluxes: `am = u_face × dp × Δy/(g) × dt/2`.
   - Diagnose cm from horizontal flux divergence (continuity).
   - Write CS transport binary (v5 format, 6 panels).
   - Target: 2-day C90 binary for Dec 1-2 2021.

2. **Build CS DrivenSimulation pathway**:
   - New file: `src_v2/Models/CubedSphereDrivenSimulation.jl`
   - Loads CS transport binary → fills per-panel flux arrays.
   - Calls `strang_split_cs!` per substep.
   - Handles Catrine CO2 IC interpolation to CS panels.
   - Handles GridFED fossil flux regridding to CS panels.
   - Writes output snapshots (panel → lat-lon regrid for viz).

3. **Build comparison visualization**:
   - Python script using matplotlib + cartopy.
   - Reads snapshot NetCDFs from all 3 grids.
   - Panel plots: column-mean CO₂ at t=0, 24h, 48h for each grid.
   - Difference plots: (grid_X - LatLon) at each time.
   - Summary stats: global mean, std, min, max per grid per time.
   - Meridional/zonal cross-sections.

4. **CS GPU test** — verify CS kernels work on GPU via Adapt pathway.

5. **Produce CS run configs**:
   - `era5_cs_c90_v2_catrine_2day_f64.toml`
   - `era5_cs_c90_v2_catrine_2day_f32.toml`

### SHARED (coordinate before touching):
- `src_v2/AtmosTransportV2.jl` — additive exports only, ping before editing
- `src_v2/Grids/Grids.jl` — additive includes only
- `src_v2/Operators/Operators.jl` — additive exports only
- `src_v2/State/FaceFluxState.jl` — additive types only

---

## Data Dependencies

```
ERA5 spectral GRIB (existing)
    ↓ [existing v4 preprocessor]
0.5° LatLon F64 binary (existing: Dec 1-2)
    ├──→ LatLon runs (direct)
    ├──→ ReducedGaussian binary (existing: Dec 1-2)
    │       └──→ RG runs (direct)
    └──→ [NEW: regrid_latlon_to_cs_binary_v2.jl]
            └──→ C90 CS binary (to be generated)
                    └──→ CS runs

Catrine IC (existing: startCO2_202112010000.nc, 1° 79-level)
    └──→ bilinear interp to each model grid at runtime

GridFED fossil (existing: GCP-GridFEDv2024.0_2021.short.nc, 0.1° monthly)
    └──→ regrid to model grid at startup, inject per substep
```

---

## Timeline (approximate)

### Hour 0-4: Foundation
- [ ] Codex: commit accumulated RG + runtime work
- [ ] Claude: start CS binary generator script

### Hour 4-8: Core pipelines
- [ ] Codex: wire surface flux injection, validate RG 48h
- [ ] Claude: generate C90 Dec 1-2 binaries, build CS driver

### Hour 8-12: Integration
- [ ] Codex: F32 cast layer, GPU smoke tests (LL + RG)
- [ ] Claude: CS DrivenSimulation, CS GPU test

### Hour 12-18: Full runs
- [ ] Both: launch all 12 runs (3 grids × 2 precisions × CPU; GPU later)
- [ ] Claude: build visualization script while runs execute

### Hour 18-24: Analysis and polish
- [ ] Claude: generate comparison plots
- [ ] Both: review cross-grid agreement, debug any discrepancies
- [ ] Both: GPU runs on curry (F64) and wurst (F32)

---

## Success Criteria

1. **All 12 CPU runs complete without crash** (3 grids × 2 FT × 2 tracers)
2. **Mass conservation**: `|Σrm(48h) - Σrm(0)| / Σrm(0) < 1e-10` (F64),
   `< 1e-5` (F32) for natural CO₂ (no sources)
3. **Cross-grid agreement**: global-mean column CO₂ within 0.5 ppm across
   all 3 grids at t=48h (natural CO₂, no sources)
4. **Fossil CO₂ plausible**: surface fossil CO₂ shows expected continental
   emission patterns; global total emission matches GridFED integral
5. **GPU = CPU**: GPU runs match CPU to machine precision (F64) or within
   1 ULP (F32)
6. **Plots**: clean panel comparison showing all grids + difference maps

---

## Key File Inventory

### Existing data
- LatLon binary: `~/data/AtmosTransport/met/era5/0.5x0.5/transport_binary_v2_tropo34_dec2021_f64/`
- RG binary: `~/data/AtmosTransport/met/era5/N320/transport_binary_v2_tropo34_dec2021_f64/`
- Catrine CO2 IC: `~/data/AtmosTransport/catrine/InitialConditions/startCO2_202112010000.nc`
- GridFED fossil: `~/data/AtmosTransport/catrine/Emissions/gridfed/GCP-GridFEDv2024.0_2021.short.nc`

### To be generated
- CS C90 binary: `~/data/AtmosTransport/met/era5/C90/transport_binary_v2_tropo34_dec2021_f64/`

### Existing code (reuse)
- Conservative regridding: `src/Sources/regrid_utils.jl`
- Injection kernel: `src_v2/Kernels/CellKernels.jl:_inject_source_kernel!`
- CS advection: `src_v2/Operators/Advection/CubedSphereStrang.jl`
- CS halo exchange: `src_v2/Operators/Advection/HaloExchange.jl`
