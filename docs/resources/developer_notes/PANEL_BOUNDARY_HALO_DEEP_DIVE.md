# Panel Boundary and Halo Deep Dive (GCHP vs AtmosTransport)

## Scope

This note compares how **GCHP (FV3 transport path)** and **AtmosTransport** handle:

1. Cubed-sphere panel edge exchange (halos)
2. Corner ghost values used by high-order PPM
3. CFL subcycling interactions with halo refresh
4. Where halo behavior can leak into small residual biases

Focus is horizontal transport with prescribed mass fluxes.

## High-level summary

- Both models use 3-cell halos for cubed-sphere high-order stencils.
- Both models use explicit corner formulas (FV3 `copy_corners`) for cross-term/Lin-Rood style transport.
- **Key difference**: GCHP refreshes halos every tracer substep, while AtmosTransport split x/y sweeps currently refresh halos once per sweep (not per internal CFL substep).
- In AtmosTransport, this creates a panel-edge-localized discrepancy when internal sweep subcycling (`n_sub > 1`) is triggered.

## GCHP behavior

### 1) Edge halos and panel boundaries

- GCHP uses FV3/MPP halo infrastructure (`mpp_update_domains`, group halo updates) for tile boundaries and MPI subdomain boundaries.
- Tracer transport enters each split step with completed halo exchange:
  - `complete_group_halo_update(q_pack, domain)` in `fv_tracer2d.F90`.
- Offline advection path fills C-grid boundaries for Courant/mass flux arrays with `mpp_get_boundary(..., gridtype=CGRID_NE)` and then updates domains.

### 2) Corner handling

- FV3 `copy_corners` is called inside `fv_tp_2d` before directional PPM reconstructions:
  - `dir=2` before Y-first branch
  - `dir=1` before X-first branch
- This matches Putman & Lin/FV3 corner rotation formulas.

### 3) Subcycling and halos

- FV3 computes `nsplt` from Courant constraints and loops over substeps.
- Halo updates are completed each substep (`complete_group_halo_update` / `start_group_halo_update`), so each substep sees consistent edge states.

## AtmosTransport behavior

### 1) Edge halos and panel boundaries

- AtmosTransport uses static GEOS-native panel connectivity and explicit scalar halo copy:
  - `fill_panel_halos!` in `src/Grids/halo_exchange.jl`.
- Panel mapping is encoded in `src/Grids/panel_connectivity.jl` (GEOS `nf=1..6` convention).
- C-grid mass fluxes are converted to staggered panel faces using explicit west/south neighbor extraction:
  - `cgrid_to_staggered_panels` in `src/IO/geosfp_cubed_sphere_reader.jl`.

### 2) Corner handling

- AtmosTransport has FV3-derived corner rotations in `copy_corners!` (`src/Grids/halo_exchange.jl`).
- Corner fill is used in Lin-Rood cross-term path (`src/Advection/cubed_sphere_fvtp2d.jl`), not in split x/y sweeps.

### 3) Subcycling and halos (split sweeps)

- In split mass-flux sweeps (`src/Advection/cubed_sphere_mass_flux.jl` and `src/Advection/cubed_sphere_mass_flux_ppm.jl`):
  - halos are filled once at sweep start;
  - internal CFL subcycles then run without halo refresh between substeps.
- This differs from FV3/GCHP substep semantics.

## Side-by-side comparison

| Topic | GCHP (FV3 path) | AtmosTransport |
|---|---|---|
| Edge halo exchange | `mpp_update_domains` / group halo updates | `fill_panel_halos!` explicit panel copy |
| Corner treatment | `copy_corners` in `fv_tp_2d` | `copy_corners!` available; used in Lin-Rood path |
| Substep halo refresh | Yes (per `nsplt` iteration) | Split x/y sweeps: no (once per sweep) |
| Panel mapping source | MPP/FV3 tile/domain metadata | Static GEOS-native connectivity table |
| C-grid boundary faces | `mpp_get_boundary(..., CGRID_NE)` | Explicit neighbor edge extraction in `cgrid_to_staggered_panels` |

## Concrete no-edit validation in this repo

I ran local scripts (temporary `/tmp` scripts, no repository file changes) comparing:

- **Path A**: current `_sweep_x!` / `_sweep_y!` behavior
- **Path B**: same advection kernels but with halo refresh before each internal substep

Observed:

1. If `n_sub = 1`, paths are identical (`L1 diff = 0`).
2. If `n_sub = 2`, paths diverge (non-zero L1 difference).
3. For random fields on C48, the difference is confined to panel-edge bands; interior away from edges is unchanged in that test.
4. Same pattern is present in PPM x-sweep path (`_sweep_x_ppm!`) when `n_sub > 1`.

Additional quantitative check (C48, `n_sub=6`, no code edits, temporary `/tmp` scripts):

1. X-sweep (`_sweep_x!`) current vs per-substep halo refresh:
   - Edge-band L1 (4-cell edge mask): `7.24e-7`
   - Interior L1: `2.47e-11`
   - Edge/interior L1 ratio: `~2.93e4`
2. Y-sweep (`_sweep_y!`) current vs per-substep halo refresh:
   - Edge-band L1 (4-cell edge mask): `6.71e-7`
   - Interior L1: `6.71e-12`
   - Edge/interior L1 ratio: `~9.99e4`
3. Distance-to-edge profile (x-sweep):
   - `d=1` cell from edge: `83.0%` of total L1
   - `d=2` cells: `16.2%` of total L1
   - cumulative by `d<=2`: `99.2%`
4. Distance-to-edge profile (y-sweep):
   - `d=1` cell from edge: `98.0%` of total L1
   - `d=2` cells: `1.71%` of total L1
   - cumulative by `d<=2`: `99.7%`
5. Mapping sanity checks (no code edits, randomized fields):
   - `cgrid_to_staggered_panels` seam consistency check across all panel interfaces: `max abs diff = 0.0`
   - `fill_panel_halos!` edge-copy consistency check across all edges/halo depths: `max abs diff = 0.0`

Interpretation:

- Current split sweeps are effectively using stale neighbor halos for later internal substeps.
- This is a plausible mechanism for small panel-boundary residual biases.

## Practical implications for your bias question

Most likely halo-related residual channel in AtmosTransport today:

1. CFL-triggered internal x/y subcycling with no per-substep halo refresh in split sweeps.
2. This channel is effectively inactive when sweeps run with `n_sub=1` (the two paths are identical in no-edit tests).

Less likely/secondary channels:

1. Corner treatment differences if using split PPM path instead of Lin-Rood cross-term path.
2. Any analysis workflow that accidentally includes halo cells (core diagnostics here already use interior slices).

## Recommended mitigation path

### Priority 1 (algorithmic parity with FV3/GCHP)

Refresh panel halos per internal substep in:

1. `src/Advection/cubed_sphere_mass_flux.jl`: `_sweep_x!`, `_sweep_y!`
2. `src/Advection/cubed_sphere_mass_flux_ppm.jl`: `_sweep_x_ppm!`, `_sweep_y_ppm!`

Pattern:

1. Determine `n_sub` from CFL
2. Scale fluxes by `1/n_sub`
3. For each substep:
   - `fill_panel_halos!(rm_panels, grid)`
   - `fill_panel_halos!(m_panels, grid)`
   - panel kernel loop
4. Restore flux magnitude

### Priority 2 (robust regression checks)

Add targeted tests that compare:

1. `n_sub=1` vs `n_sub=2` behavior on panel-edge perturbations
2. edge-band vs interior error localization metrics
3. seam flux symmetry diagnostics across known panel interfaces

### Priority 3 (short-term run-mode workaround)

For strict comparison experiments before code change:

1. Keep sweep CFL below threshold so internal sweep `n_sub` stays 1, or
2. Prefer Lin-Rood cross-term path where feasible for boundary robustness.

## Code anchors

### GCHP / FV3

- `src/GCHP_GridComp/FVdycoreCubed_GridComp/fvdycore/model/fv_tracer2d.F90`
- `src/GCHP_GridComp/FVdycoreCubed_GridComp/fvdycore/model/tp_core.F90`
- `src/GCHP_GridComp/FVdycoreCubed_GridComp/fvdycore/geos_utils/ghost_cubsph.F90`

### AtmosTransport

- `src/Grids/panel_connectivity.jl`
- `src/Grids/halo_exchange.jl`
- `src/IO/geosfp_cubed_sphere_reader.jl`
- `src/Advection/cubed_sphere_mass_flux.jl`
- `src/Advection/cubed_sphere_mass_flux_ppm.jl`
- `src/Advection/cubed_sphere_fvtp2d.jl`
