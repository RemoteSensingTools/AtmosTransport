# Diffusion And Convection Physics

This note is the production contract for vertical mixing and convective
transport. Names in TOML should describe the source algorithm, not just the
fact that an operator is "PBL" or "convection".

## Production Choices

For GEOS and ERA binaries regridded to cubed sphere, the current production
configuration is:

```toml
[diffusion]
kind = "tm5_beljaars_viterbo_local_kz"
surface_flux_boundary = true

[convection]
kind = "cmfmc"  # GEOS native source
# or
kind = "tm5"    # ERA/TM5 four-field convection source
```

`surface_flux_boundary = true` means configured surface fluxes enter before
the implicit vertical diffusion solve. This is equivalent to treating emissions
as the lower-boundary source for the column mixer:

```text
S(dt) -> V(dt)
```

The legacy split placement is still available with
`surface_flux_boundary = false`:

```text
V(dt/2) -> S(dt) -> V(dt/2)
```

Use the boundary placement for production full-physics runs unless a parity
experiment explicitly needs the old split.

For GEOS/GCHP comparison experiments with binaries that carry the optional
VDIFF payload, the cubed-sphere runtime also accepts:

```toml
[diffusion]
kind = "geoschem_holtslag_boville_vdiff"
surface_flux_boundary = true
```

## Diffusion Closures

### `tm5_beljaars_viterbo_local_kz`

This is the current production PBL diffusivity closure. It derives Kz from
window surface fields:

- `pblh`
- `ustar`
- sensible heat flux (`pbl_hflux` in binaries, exposed as `hflux` at runtime)
- `t2m`
- dry air mass and grid-cell area

The formulas live in `PBLPhysicsParameters` and `WindowPBLKzField`. They are a
Beljaars-Viterbo / revised Louis-Tiedtke-Geleyn local Kz closure matching the
TM5-style constants already used by the legacy PBL path. The runtime then uses
the common `ImplicitVerticalDiffusion` backward-Euler tridiagonal column solve.

`kind = "pbl"` remains a legacy alias for this closure, but new configs should
use the explicit name.

### `geoschem_holtslag_boville_vdiff`

This is the GEOS/GCHP VDIFF-derived local-Kz runtime path for cubed-sphere
binaries. It uses the GEOS VDIFF source fields for virtual temperature,
hydrostatic column geometry, and wind-shear/stability enhancement, while using
the same `ImplicitVerticalDiffusion` tridiagonal column solve as the other
runtime diffusion closures.

Important scope boundary: this path is not yet full GEOS-Chem VDIFF parity.
The nonlocal counter-gradient term is not applied. The name records the source
lineage and required payload; campaign use should still be validated against
GCHP diagnostics before treating it as equivalent to GEOS-Chem's complete VDIFF
module.

GEOS-IT binaries carry the required source fields through the optional
`gchp_vdiff` payload:

- `vdiff_u`, `vdiff_v` from `A3dyn` (`U`, `V`; 3-hourly hold-constant)
- `vdiff_t` from `I3` (`T`; 3-hourly hold-constant)
- `vdiff_qv` from the CTM_I1 left endpoint (`QV`; hourly)
- PBL surface payload from `A1` (`PBLH`, `USTAR`, `HFLUX`, `T2M`)

The preprocessor switch is `[source] include_vdiff_fields = true`, and the
download descriptor is `config/downloads/geosit_c180_gchp_vdiff.toml`.
Runtime selection of `kind = "geoschem_holtslag_boville_vdiff"` must require
`binary_capabilities(reader).gchp_vdiff === true`.

## ERA Best Practice

ERA preprocessing can read single-level surface fields through
`era5_surface_reader.jl` and can carry the PBL section through the binary and
LL-to-CS regridding path. For ERA runs that have those sections, the best
current production choice is still:

```toml
[diffusion]
kind = "tm5_beljaars_viterbo_local_kz"
surface_flux_boundary = true
```

Structured lat-lon and reduced-Gaussian runtime binaries can use constant Kz
today. Met-derived PBL Kz for those topologies should be added through the same
typed closure name and binary capability check, not by adding topology-local
branches in runners.

## Convection Closures

### `cmfmc`

Use this for GEOS-native CMFMC/DTRAIN binaries. It is a forward column update
with a local CFL guard. It is cheap on GPU and matches the GEOS source fields
currently carried in GEOS-IT/GEOS-FP binaries.

### `tm5`

Use this for ERA/TM5 four-field convection binaries carrying:

- `entu`
- `detu`
- `entd`
- `detd`

The solver builds the TM5 backward-Euler column transport matrix and solves it
per active convective column. Rows above the effective cloud window are
identity and the lower-left quadrant is zero; the production solver now skips
that dead part for both matrix assembly and LU/solve while preserving the full
matrix contract for tests and adjoint replay.

Convection is applied once per meteorological window in driven runs. Advection
and diffusion run on the binary's substep schedule; convection is a physics
cadence choice, not part of the advection CFL schedule.

## Implementation Rules

- Operator names must encode algorithm lineage: `tm5_beljaars_viterbo_local_kz`
  is acceptable; generic `pbl` is only a compatibility alias.
- Binary capability checks must fail early if a selected operator lacks its
  required sections.
- Surface emissions are represented by surface-flux fields and a runtime
  toggle; they are not hidden inside the diffusion operator.
- Topology-specific refresh code belongs in typed field/cache objects and
  dispatch methods. Runners should not grow source/topology `if` trees.
- GCHP VDIFF counter-gradient support must extend
  `GCHPHoltslagBovilleKzField` or add a sibling typed field/operator, with
  tests against single-column reference cases before campaign use.
