# PBL and Convection Comparison Overview

## Scope

This note compares how **AtmosTransport**, **GCHP (GEOS-Chem in GCHP 14.7 lineage)**, and **TM5** implement:

1. Convective transport
2. PBL/turbulent vertical mixing
3. Dry vs wet mass basis and tracer representation
4. Operator ordering and timestep strategy

The focus is model behavior when using offline meteorological inputs.

## Executive summary

- AtmosTransport currently mirrors GEOS-Chem-style RAS transport structure (CMFMC + DTRAIN) and offers both local and non-local PBL diffusion variants.
- GCHP has the most tightly coupled physics path for convection + wet scavenging and uses either full PBL mixing or non-local VDIFF.
- TM5 formulates convection and diffusion as column matrices/linear solves, with an optional coupled conv+diff operator.
- The largest cross-model differences are not just in equations, but in:
  - dry-vs-wet air-mass basis,
  - when convection is applied relative to advection,
  - and whether wet scavenging is integrated into the convection operator.

## Side-by-side matrix

| Topic | AtmosTransport | GCHP (GEOS-Chem) | TM5 |
|---|---|---|---|
| Convection schemes | `TiedtkeConvection` (CMFMC-only), `RASConvection` (CMFMC + DTRAIN) | RAS or Grell-Freitas branch (`DO_RAS_CLOUD_CONVECTION` / `DO_GF_CLOUD_CONVECTION`) | Matrix-based convection from `eu/ed/du/dd`; optional coupled `convdiff` |
| Convection numerics | Column kernels; RAS uses two-pass cloud/environment update with CFL subcycling | Column physics with internal substeps (`NS`) and integrated scavenging paths | Build convection matrix, LU/solve per column; optional coupled convection+diffusion matrix |
| Wet scavenging in convection | Trait framework exists; current path effectively inert/no scavenging | Fully integrated soluble removal and precipitation coupling in convection routines | Species-dependent removal via `lbdcv * cvsfac` during apply |
| PBL options | Static exponential BLD, met-driven local PBL, met-driven non-local PBL | Full PBL mixing (`TurbDay`) or non-local VDIFF | LTG-style BL diffusion (`bldiff`) + implicit tridiagonal solve |
| Non-local PBL counter-gradient | Yes (`NonLocalPBLDiffusion`, Holtslag-Boville style) | Yes (`vdiff_mod`, Holtslag/Beljaars constants incl. `fak`, `sffrac`) | BL physics computes non-local effects in coefficient construction before diffusion solve |
| Tracer basis in these operators | CS path typically stores tracer mass `rm`; converts to mixing ratio in-kernel | Dry mixing ratio centric in chemistry state with dry-mass terms in convection/full PBL mixing path | Uses tracer mass arrays with mass-field coupling in transport operators |
| Dry/wet mass handling (key practical point) | Explicit moist-air mass (`m_wet`) use for convection/PBL conversions in CS path | Convection uses dry air mass term (`BMASS` from `DELP_DRY`) in column solver paths | Uses pressure-derived mass and met-conversion path for conv fields; no explicit GEOS-style `DELP_DRY` branch in these routines |
| Operator ordering | Advection substeps, convection once per window, then post-advection diffusion | Chunk phase executes convection and turbulence as separate physics phases | Split-operator loop (`v` for convection, `d` for diffusion) unless compiled with coupled `with_convdiff` |

## Deep dive

### 1. Convection approach

AtmosTransport RAS is an offline transport formulation patterned after GEOS-Chem RAS logic (entrainment diagnosed from CMFMC and DTRAIN, updraft concentration pass, then environment update pass). It is computationally lean and designed for GPU kernels.

GCHP convection is broader-physics: transport plus scavenging and precipitation-linked terms are solved together in the convection routine. It also conditionally switches parameterization branch (RAS vs GF) based on met/source logic.

TM5 treats convection as a linear algebra operator per column. This is structurally different from tendency-form kernels: it assembles a transport matrix, then solves/applies it, and can optionally merge diffusion in the same solve.

### 2. PBL approach

AtmosTransport has the most modular interface at runtime (static, local LTG-like, non-local Holtslag-Boville style). Both local and non-local options use implicit column solves and are GPU-oriented.

GCHP bifurcates turbulence behavior by configuration:

- Non-local VDIFF path (`LTURB && LNLPBL`)
- Full PBL mixing path (`LTURB && !LNLPBL`) with complete mixing under PBL top

TM5 computes BL diffusivity fields (`kvh`) and exchange coefficients (`dkg`) from met and applies an implicit tridiagonal solve. This is closest in spirit to AtmosTransport local PBL diffusion.

### 3. Timestep and splitting implications

AtmosTransport currently applies convection once per transport window (not every advection substep), explicitly to avoid repeated over-mixing behavior.

GCHP applies convection and turbulence through its model phase sequencing at physics cadence.

TM5 sequencing is explicit in split operators (`v`, `d`) with compile-time option to solve convection and diffusion together (`with_convdiff`). This matters for parity experiments because changing split order can shift results even with identical met inputs.

### 4. Dry vs wet basis and mixing-ratio interpretation

This is a major source of mismatch risk.

- AtmosTransport CS path explicitly distinguishes and uses moist mass (`m_wet`) in convection and diffusion conversions where those operators are on moist met fields.
- GCHP convection/full-PBL paths use dry-basis mass terms in core transport equations (notably via `DELP_DRY`/`BMASS` in convection routines).
- TM5 convection/diffusion routines shown here are pressure/mass-field based without a direct GEOS-style `DELP_DRY` branch in these modules; behavior depends on the met-conversion chain that prepares the operator inputs.

For cross-model comparisons, dry/wet normalization choices should be treated as first-order configuration differences, not post-processing details.

### 5. Practical parity implications

To isolate algorithmic differences:

1. Align operator ordering (convection timing relative to advection and diffusion).
2. Align basis choices (dry vs wet mass for `rm <-> q` conversions).
3. Disable or standardize scavenging behavior when comparing pure transport.
4. Keep subcycling/CFL policies as close as possible.
5. Compare single-column diagnostics first, then 3D fields.

## Code anchor points

### AtmosTransport

- `src/Convection/tiedtke_convection.jl`
- `src/Convection/ras_convection.jl`
- `src/Diffusion/pbl_diffusion.jl`
- `src/Diffusion/nonlocal_pbl_diffusion.jl`
- `src/Models/physics_phases.jl`

### GCHP / GEOS-Chem

- `src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/GeosCore/convection_mod.F90`
- `src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/GeosCore/mixing_mod.F90`
- `src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/GeosCore/pbl_mix_mod.F90`
- `src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/GeosCore/vdiff_mod.F90`
- `src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/GeosCore/input_mod.F90`
- `src/GCHP_GridComp/GEOSChem_GridComp/geos-chem/Interfaces/GCHP/gchp_chunk_mod.F90`

### TM5

- `deps/tm5/base/src/convection.F90`
- `deps/tm5/base/src/tm5_conv.F90`
- `deps/tm5/base/src/tm5_convdiff.F90`
- `deps/tm5/base/src/tm5_diff.F90`
- `deps/tm5/base/src/diffusion.F90`
- `deps/tm5/base/src/meteo.F90`
- `deps/tm5/base/src/tmm.F90`
- `deps/tm5/base/src/phys_convec_ec2tm.F90`
- `deps/tm5/base/src/modelIntegration.F90`
