# GEOS-IT C180 Met Variable Mapping: GCHP 14.7.0 vs AtmosTransport

## Scope

This note documents what meteorology variable names are read for GEOS-IT C180 in:

1. GCHP 14.7.0 (GEOS-Chem submodule settings/registry)
2. AtmosTransportModel (current `geosfp_cs` GEOS-IT path)

Focus is on transport-relevant fields (horizontal mass fluxes, pressure, humidity, and key physics companions).

## GCHP 14.7.0: GEOS-IT C180 advection imports

Source:

- `~/code/gitHub/GCHP/src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/run/shared/settings/geosit/advection_met/geosit.raw_1hr_c180_mass_flux.txt`
- `~/code/gitHub/GCHP/src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/Interfaces/GCHP/Registry/Chem_Registry.rc`

From `geosit.raw_1hr_c180_mass_flux.txt`:

- `UA;VA <- U;V`
- `MFXC;MFYC <- MFXC;MFYC`
- `CXC;CYC <- CX;CY`
- `PS1 <- PS`
- `PS2 <- PS`
- `SPHU1 <- QV`
- `SPHU2 <- QV`

Additional advection metadata in that file:

- `RUNDIR_IMPORT_MASS_FLUX_FROM_EXTDATA=.true.`
- `RUNDIR_USE_TOTAL_AIR_PRESSURE_IN_ADVECTION=0`
- Top/down flags for mass flux, wind, humidity are set true.

`Chem_Registry.rc` confirms these import state names are expected (`PS1/PS2`, `SPHU1/SPHU2`, `UA/VA`, `CMFMC`, `DTRAIN`, etc.).

## AtmosTransport: GEOS-IT C180 fields currently read

Primary source files:

- `src/IO/geosfp_cubed_sphere_reader.jl`
- `src/IO/geosfp_cs_met_driver.jl`
- `scripts/preprocessing/convert_surface_cs_to_binary.jl`

### Core transport (CTM_A1)

From `read_geosfp_cs_timestep`:

- `MFXC`
- `MFYC`
- `DELP`
- `PS`

The reader auto-detects bottom-to-top files and flips vertical level order to TOA-first.

### Humidity

From `load_qv_window!`:

- `QV` from co-located `I3` file (`GEOSIT.YYYYMMDD.I3.C180.nc`)

### Convection companions

From `read_geosfp_cs_cmfmc` / `read_geosfp_cs_dtrain`:

- `CMFMC` from `A3mstE`
- `DTRAIN` from `A3dyn`

### Surface fields used in physics path

From `read_geosfp_cs_surface_fields` and binary converter:

- `PBLH`
- `USTAR`
- `HFLUX`
- `T2M`
- optional `TROPPT`
- optional `PS` (pulled from co-located `CTM_A1` during A1 binary conversion)

## Side-by-side mapping (transport-focused)

| Topic | GCHP 14.7.0 (GEOS-IT mass-flux settings) | AtmosTransport (current GEOS-IT C180 path) |
|---|---|---|
| Horizontal transport fluxes | `MFXC;MFYC <- MFXC;MFYC` | `MFXC`, `MFYC` from `CTM_A1` |
| Courant fields | `CXC;CYC <- CX;CY` | Not read/used |
| Wind fields in advection import set | `UA;VA <- U;V` | Not used in CS mass-flux advection path |
| Pressure fields for transport | `PS1`, `PS2 <- PS` (dual time-level import names) | Single `PS`; transport state primarily uses `DELP` + flux continuity |
| Humidity fields for dry/wet conversion | `SPHU1`, `SPHU2 <- QV` (dual time-level import names) | Single `QV` from `I3` |
| Layer thickness / mass representation | Via imported pressure/humidity fields and model internals | Explicit `DELP` ingestion from `CTM_A1` |
| Convective mass flux support | `CMFMC`, `DTRAIN` imported in non-advection met set | `CMFMC` (`A3mstE`) + `DTRAIN` (`A3dyn`) loaded when needed |

## Key alignment and differences

1. Main horizontal mass-flux names align (`MFXC`, `MFYC`).
2. AtmosTransport does not use GCHP-style duplicate advection time-level aliases (`PS1/PS2`, `SPHU1/SPHU2`).
3. AtmosTransport currently does not consume `CX/CY` in this path.
4. AtmosTransport CS transport path uses direct `MFXC/MFYC + DELP` and continuity closure; it does not depend on `U/V` ingestion for CS advection.
5. Horizontal GEOS mass fluxes are treated as already dry-basis mass fluxes in current interpretation.

## Config context in this repo

Representative run configs:

- `config/runs/geosit_c180_june2023.toml` uses NetCDF GEOS-IT C180 path.
- `config/runs/catrine_geosit_c180.toml` uses preprocessed mass-flux binaries plus `surface_data_bin_dir`.

Both use:

- `product = "geosit_c180"`
- `driver = "geosfp_cs"`
- `mass_flux_dt = 450`

## Source pointers (line-level)

- GCHP advection mapping:
  - `.../geosit.raw_1hr_c180_mass_flux.txt` lines 13-19
- GCHP import registry:
  - `.../Chem_Registry.rc` lines 69-94
- Atmos CTM_A1 reads:
  - `src/IO/geosfp_cubed_sphere_reader.jl` lines 180-183
- Atmos QV read:
  - `src/IO/geosfp_cs_met_driver.jl` line 1034
- Atmos surface binary var mapping:
  - `src/IO/geosfp_cs_met_driver.jl` lines 450-462
  - `scripts/preprocessing/convert_surface_cs_to_binary.jl` lines 120-130
