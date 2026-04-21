# GCHP Advection Timing Notes (C720/C180)

## Scope

This note captures how GCHP handles timing for:

- mass fluxes (`MFXC`, `MFYC`, `CXC`, `CYC`)
- surface pressure (`PS1`, `PS2`)
- specific humidity (`SPHU1`, `SPHU2`)

based on local code inspection of the `GCHP` checkout at:

- `main` @ `3216f28`
- tag `14.7.0`

## Main Findings

1. Mass fluxes and Courant numbers are imported as **time-averaged** fields in the C720 mass-flux settings (`F0;003000`).
2. `PS` and `SPHU` are imported as **instantaneous before/after** fields:
   - `PS1`, `SPHU1`: start of interval (`0`)
   - `PS2`, `SPHU2`: end offset (`0;HHMMSS`)
3. In the `GCHPctmEnv` component, these are explicitly consumed as before/after advection states:
   - `PS1/SPHU1` -> `PLE0/DryPLE0` (before)
   - `PS2/SPHU2` -> `PLE1/DryPLE1` (after)
4. The effective `PS2/SPHU2` offset in generated run directories is auto-synced to the transport timestep by `setCommonRunSettings.sh.template`.

## Where This Is Defined

### C720 advection settings

- [geosfp.raw_1hr_c720_mass_flux_PS_SPHU_C.txt](../../../../GCHP/src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/run/shared/settings/geosfp/advection_met/geosfp.raw_1hr_c720_mass_flux_PS_SPHU_C.txt)
  - `MFXC;MFYC` and `CXC;CYC` use `F0;003000`
  - `PS1/SPHU1` use `0`
  - `PS2/SPHU2` use `0;001000` (in this template)

Reference lines:

- [mass flux/Courant entries](../../../../GCHP/src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/run/shared/settings/geosfp/advection_met/geosfp.raw_1hr_c720_mass_flux_PS_SPHU_C.txt#L8)
- [PS/SPHU entries](../../../../GCHP/src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/run/shared/settings/geosfp/advection_met/geosfp.raw_1hr_c720_mass_flux_PS_SPHU_C.txt#L12)

### GEOS-IT C180 advection settings (same pattern)

- [geosit.raw_1hr_c180_mass_flux.txt](../../../../GCHP/src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/run/shared/settings/geosit/advection_met/geosit.raw_1hr_c180_mass_flux.txt)
- [geosit.preprocessed_1hr_c180_mass_flux.txt](../../../../GCHP/src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/run/shared/settings/geosit/advection_met/geosit.preprocessed_1hr_c180_mass_flux.txt)
- [geosit.preprocessed_3hr_c180_wind.txt](../../../../GCHP/src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/run/shared/settings/geosit/advection_met/geosit.preprocessed_3hr_c180_wind.txt)

## ExtData Timing Semantics Used Here

From MAPL ExtData parser:

- `F...` means "do not time-interpolate this field"
  - [ExtDataGridCompMod.F90#L604](../../../../GCHP/src/MAPL/gridcomps/ExtData/ExtDataGridCompMod.F90#L604)
- `0;HHMMSS` is parsed as a time shift on base `"0"`
  - [ExtDataGridCompMod.F90#L903](../../../../GCHP/src/MAPL/gridcomps/ExtData/ExtDataGridCompMod.F90#L903)
- non-interpolated fields use the left bracket sample directly
  - [ExtDataGridCompMod.F90#L3426](../../../../GCHP/src/MAPL/gridcomps/ExtData/ExtDataGridCompMod.F90#L3426)

Bracket/update flow:

- [ExtData run loop (update left/right brackets)](../../../../GCHP/src/MAPL/gridcomps/ExtData/ExtDataGridCompMod.F90#L1327)
- [CheckUpdate time shift handling](../../../../GCHP/src/MAPL/gridcomps/ExtData/ExtDataGridCompMod.F90#L4198)

## How GCHPctmEnv Uses PS/SPHU

Imports are declared as before/after:

- [PS1/PS2/SPHU1/SPHU2 import specs](../../../../GCHP/src/GCHP_GridComp/GCHPctmEnv_GridComp/GCHPctmEnv_GridCompMod.F90#L151)

Comments and usage:

- [ExtData before/after comment for PS](../../../../GCHP/src/GCHP_GridComp/GCHPctmEnv_GridComp/GCHPctmEnv_GridCompMod.F90#L742)
- [compute `PLE0` from `PS1`](../../../../GCHP/src/GCHP_GridComp/GCHPctmEnv_GridComp/GCHPctmEnv_GridCompMod.F90#L782)
- [compute `PLE1` from `PS2`](../../../../GCHP/src/GCHP_GridComp/GCHPctmEnv_GridComp/GCHPctmEnv_GridCompMod.F90#L787)
- [compute `DryPLE0` with `SPHU1`](../../../../GCHP/src/GCHP_GridComp/GCHPctmEnv_GridComp/GCHPctmEnv_GridCompMod.F90#L797)
- [compute `DryPLE1` with `SPHU2`](../../../../GCHP/src/GCHP_GridComp/GCHPctmEnv_GridComp/GCHPctmEnv_GridCompMod.F90#L805)

Mass-flux path and optional humidity correction:

- [prepare_massflux_exports](../../../../GCHP/src/GCHP_GridComp/GCHPctmEnv_GridComp/GCHPctmEnv_GridCompMod.F90#L999)
- [`MFX/MFY` correction by `/(1-SPHU0)`](../../../../GCHP/src/GCHP_GridComp/GCHPctmEnv_GridComp/GCHPctmEnv_GridCompMod.F90#L1029)

## Timestep-Dependent Rewrite in Run Setup

`setCommonRunSettings.sh.template` updates generated `ExtData.rc` so `PS2/SPHU2/TMPU2` offset matches transport timestep:

- [update function](../../../../GCHP/src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/run/GCHP/setCommonRunSettings.sh.template#L642)
- [applied to `PS2/SPHU2/TMPU2`](../../../../GCHP/src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/run/GCHP/setCommonRunSettings.sh.template#L783)

Template defaults in that script:

- C180-and-coarser branch: `TransConv_Timestep_sec=600`, `HHMMSS=001000`
  - [setCommonRunSettings.sh.template#L197](../../../../GCHP/src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/run/GCHP/setCommonRunSettings.sh.template#L197)
- finer branch: `TransConv_Timestep_sec=300`, `HHMMSS=000500`
  - [setCommonRunSettings.sh.template#L201](../../../../GCHP/src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/run/GCHP/setCommonRunSettings.sh.template#L201)

So in practice, `PS2/SPHU2` timing in the final run directory may differ from raw template defaults.

