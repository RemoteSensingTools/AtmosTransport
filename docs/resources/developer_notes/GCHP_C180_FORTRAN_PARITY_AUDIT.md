# GCHP C180 Fortran Parity Audit for `catrine_geosit_c180_gchp.toml`

## Scope

This audit checks whether the AtmosTransport run path for:

- `config/runs/catrine_geosit_c180_gchp.toml`

matches what GCHP Fortran actually does from meteorology input through offline tracer advection and remap/scaling.

Primary code paths inspected:

- AtmosTransport Julia:
  - `src/IO/configuration.jl`
  - `src/IO/geosfp_cs_met_driver.jl`
  - `src/IO/binary_readers.jl`
  - `src/Models/physics_phases.jl`
  - `src/Advection/cubed_sphere_fvtp2d_gchp.jl`
  - `src/Advection/vertical_remap.jl`
- GCHP Fortran:
  - `src/GCHP_GridComp/GCHPctmEnv_GridComp/GCHPctmEnv_GridCompMod.F90`
  - `src/GCHP_GridComp/FVdycoreCubed_GridComp/fvdycore/model/fv_tracer2d.F90`
  - `src/GCHP_GridComp/FVdycoreCubed_GridComp/fvdycore/model/tp_core.F90`
  - `src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/run/shared/settings/geosit/advection_met/geosit.raw_1hr_c180_mass_flux.txt`

## Run Config Activations

From `config/runs/catrine_geosit_c180_gchp.toml`:

- `driver = "geosfp_cs"`
- `product = "geosit_c180"`
- `preprocessed_dir = "/temp1/catrine/met/geosit_c180/massflux_v3"`
- `surface_data_bin_dir = "/temp1/catrine/met/geosit_c180/surface_bin"`
- `dt = 450`, `met_interval = 3600`, `mass_flux_dt = 450`
- advection:
  - `scheme = "ppm"`, `gchp = true`, `vertical_remap = true`, `dry_correction = true`

The Julia builder and run loop do activate the GCHP path:

- `src/IO/configuration.jl:339-367` (`GEOSFPCubedSphereMetDriver`)
- `src/IO/configuration.jl:843-857` (`PPMAdvection(... use_gchp=true, use_vertical_remap=true)`)
- `src/Models/physics_phases.jl:455-592` (GCHP advection/remap branch)

## Data Input Parity

## 1) Mass flux / Courant / area-flux ingestion

### GCHP

For GEOS-IT C180 mass-flux mode, settings import:

- `MFXC;MFYC`
- `CXC;CYC`
- `PS1;PS2`
- `SPHU1;SPHU2`

from 1-hour CTM files (`geosit.raw_1hr_c180_mass_flux.txt:14-19`).

In Fortran bridge:

- imports are copied in `prepare_massflux_exports` (`GCHPctmEnv_GridCompMod.F90:1008-1014`)

### AtmosTransport

`load_met_window!` in binary mode reads:

- `DELP`, `AM`, `BM` (always)
- `CX`, `CY` if header says `has_courant`
- `XFX`, `YFX` if header says `has_area_flux`

Relevant code:

- `src/IO/geosfp_cs_met_driver.jl:1419-1496`
- `src/IO/binary_readers.jl:172-175` (`has_courant`, `has_area_flux`)
- `src/IO/binary_readers.jl:221-263` (CX/CY and XFX/YFX loaders)

On-disk file checked for this run (`/temp1/catrine/met/geosit_c180/massflux_v3/geosfp_cs_20211201_float32.bin`) has:

- `"version": 3`
- `"include_courant": true`
- `"include_area_flux": true`
- `"dt_met_seconds": 450.0`

So input field availability is aligned for this config.

## 2) Specific humidity / pressure time-level semantics

### GCHP

In C180 mass-flux settings:

- `PS1/PS2` and `SPHU1/SPHU2` come from CTM instantaneous 1-hour files (`...mass_flux.txt:16-19`).

Fortran uses them to build pressure edges:

- `prepare_ple_exports` computes `PLE0/PLE1` and `DryPLE0/DryPLE1` (`GCHPctmEnv_GridCompMod.F90:782-811`)
- dry/total selection via `USE_TOTAL_AIR_PRESSURE_IN_ADVECTION` (`GCHPctmEnv_GridCompMod.F90:560-569`)

`geosit.raw_1hr_c180_mass_flux.txt:5` sets:

- `RUNDIR_USE_TOTAL_AIR_PRESSURE_IN_ADVECTION=0` (dry-pressure advection)

### AtmosTransport

Humidity loading is GCHP-aligned at source choice:

- tries `CTM_I1` hourly QV first, falls back to I3 (`src/IO/geosfp_cs_met_driver.jl:1000-1018`, `1063-1114`)

But advection state uses:

- source `dp` from `DELP` in mass-flux binary (`src/Models/physics_phases.jl:475-479`)
- target scaling against next-window `DELP` (`src/Models/physics_phases.jl:540-546`)

There is no active use of `PS1/PS2` + `SPHU1/SPHU2` to build `DryPLE0/DryPLE1` in the GCHP branch.  
(`compute_dry_ple!` exists but is not called.)

Result: not exact parity with GCHP pressure-edge input semantics.

## 3) Humidity correction of mass flux

### GCHP

Fortran default:

- `CORRECT_MASS_FLUX_FOR_HUMIDITY` default is `1` (`GCHPctmEnv_GridCompMod.F90:572-577`)
- when enabled: `MFX/MFY /= (1 - SPHU0)` (`GCHPctmEnv_GridCompMod.F90:1029-1031`)

### AtmosTransport

No equivalent correction is applied to `AM/BM` before horizontal advection.  
`dry_correction=true` only affects air-mass basis from `DELP` and QV (`src/Models/physics_phases.jl:273-285`).

Result: not exact parity unless GCHP run is configured with humidity correction disabled.

## Advection Core Parity

## 4) Horizontal finite-volume kernel

### Matched pieces

- area-based pre-advection and final mass-flux averaging are consistent with `tp_core.F90`:
  - Fortran: `tp_core.F90:167-175`, `190`, `200-213`
  - Julia: `src/Advection/cubed_sphere_fvtp2d_gchp.jl:331-340`, `379-383`

- dp continuity update form is consistent:
  - Fortran: `fv_tracer2d.F90:517-519`
  - Julia: `src/Advection/cubed_sphere_fvtp2d_gchp.jl:798-802`

## 5) Subcycling (`nsplt`) mismatch

### GCHP

`tracer_2d` supports per-level split count:

- `ksplt(k) = int(1 + cmax(k))` unless compiled with `GLOBAL_CFL` (`fv_tracer2d.F90:467-471`)
- loop condition `if (it <= ksplt(k))` (`fv_tracer2d.F90:513`)

### AtmosTransport

Current Julia path uses a single global `nsplt` for all levels:

- explicit comment: "For simplicity, use global nsplt" (`src/Advection/cubed_sphere_fvtp2d_gchp.jl:767-769`)

Result: exact only if target GCHP build uses `GLOBAL_CFL`; otherwise this differs.

## Remap / Scaling Parity

The Julia GCHP branch closely follows `offline_tracer_advection` structure:

- source PE from evolved dp
- hybrid target PE with same surface edge
- remap
- global scaling factor to next met DELP

References:

- Fortran `offline_tracer_advection`: `fv_tracer2d.F90:993-1006`, `1037-1070`, `1142-1186`
- Julia:
  - `src/Models/physics_phases.jl:519-546`
  - `src/Advection/vertical_remap.jl:1107-1167`
  - `src/Advection/vertical_remap.jl:1431-1462`

This part is algorithmically aligned.

## Additional Robustness Risk

`CX/CY/XFX/YFX` buffers are allocated with `undef` when `use_gchp=true`:

- `src/IO/met_buffers.jl:185-188`

and only populated if binary header flags indicate availability:

- `src/IO/geosfp_cs_met_driver.jl:1491-1496`

For this run (`massflux_v3`) flags are present, so this is safe in practice.  
But if a v1/v2 file were accidentally used with this config, the GCHP branch condition could still pass and consume uninitialized data.

## Verdict

`catrine_geosit_c180_gchp.toml` is **close but not exactly identical** to GCHP Fortran behavior.

High-confidence matches:

- mass-flux + Courant + area-flux transport numerics (core kernels)
- remap/scaling flow
- hourly QV source preference (`CTM_I1` first)

Material differences from exact GCHP:

1. No active `PS1/PS2 + SPHU1/SPHU2 -> DryPLE0/DryPLE1` pressure-edge construction in advection.
2. No `MFX/MFY /= (1-SPHU0)` humidity correction (GCHP default behavior is enabled).
3. Global `nsplt` simplification vs GCHP per-level `ksplt(k)` unless `GLOBAL_CFL` is used.

For strict 1:1 parity, these three items need to be aligned.
