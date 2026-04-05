# TM5 grid and advection references

This document points to TM5 sources and how this model aligns with TM5 conventions for lat/lon grids, cell centers, and pole/CFL handling.

## TM5 repositories and docs

- **TM5 on SourceForge (Cycle 3 4DVAR)**  
  https://sourceforge.net/p/tm5/cy3_4dvar/ci/default/tree/  
  Contains: `base/`, `analysis/`, `input/`, `proj/`, `rc/`, `user/`, `vpp/`. Core transport is in `base/`.

- **TM5 physics (advection, grid)**  
  https://tm5.sourceforge.net/about-tm5/physics/  
  Advection: slopes scheme (Russell and Lerner, 1987); convection: Tiedtke (1989). Operator splitting for advection, convection, sources, chemistry.

- **KNMI TR-294: "Towards advection on a full reduced grid for TM5"**  
  P.B. Hooghiemstra, 2006.  
  https://www.knmi.nl/research/publications/towards-advection-on-a-full-reduced-grid-for-tm5  
  PDF: KNMI TR-294. Describes:
  - **Regular grid**: same number of zonal cells on every latitude -> cells shrink near poles -> CFL violation risk unless time step is reduced.
  - **Reduced grid**: poleward cells are combined (e.g. one cell at the pole) so cell sizes are more uniform; ECMWF data is on a fully reduced grid.
  - **Cell properties**: cell center coordinates (x_i, y_i), cell length/width (dx_i, dy_i) per cell; linear subgrid tracer distribution; symmetric operator splitting X(dt/2), Y(dt/2), Y(dt/2), X(dt/2).

- **UU TM5 overview**  
  https://www.projects.science.uu.nl/tm5/TM5_overview.html  
  Resolutions: global 6x4 deg, intermediate 3x2 deg, zoom 1x1 deg (and 0.5x0.25 deg in high-res); vertical layers from ECMWF.

## Alignment in this model

- **Vertical coordinate**: All met sources use **hybrid sigma-pressure
  coordinates** (`p = A + B * p_surface`). A/B coefficients are stored in
  TOML files (`config/era5_L137_coefficients.toml`, `config/geos_L72_coefficients.toml`).
  The `HybridSigmaPressure` type provides a universal interface regardless of
  whether the met data originated from spectral harmonics (ECMWF) or
  gridded output (GEOS, MERRA).

- **Grid**: Regular latitude-longitude grid. All horizontal coordinates and
  spacings use **cell-center** convention: tracers and diagnostics at
  cell centers; Dx = R*cos(lat)*Dlon, Dy uniform.

- **Poles / CFL**: TM5-style **reduced grid** for x-advection (CPU) or
  **CFL-adaptive subcycling** (GPU and fallback). Polar cells are clustered
  to avoid CFL violations from shrinking zonal spacing. Subcycling also
  handles y and z CFL adaptively.

- **Source/observation placement**: Point sources and lon/lat -> grid indexing
  use **cell-center** coordinates via `find_nearest_ij(grid, lon_deg, lat_deg)`.

- **NetCDF output**: Written lon/lat are the grid cell-center arrays.

## Differences from TM5

- **Reduced grid**: TM5 runs on a fully reduced grid; we support TM5-style
  reduced grids on CPU and fall back to subcycling on GPU.
- **Advection scheme**: Same family -- **Russell & Lerner (1981) slopes scheme**
  (see `src/Advection/slopes_advection.jl`). GPU kernels via KernelAbstractions.jl
  for x, y, and z directions. Optional minmod flux limiter; Strang splitting.
  TM5 also offers the second-order moments (Prather) scheme; we now expose an
  experimental Prather path, but the LL production baseline remains `scheme = "slopes"`.
- **Mass fluxes**: TM5 computes mass-conserving fluxes via spectral integration.
  We use gridpoint winds directly. Comparison and validation are ongoing.
- **Met data**: TM5 requires ECMWF spectral GRIB data. Our model reads standard
  NetCDF from ERA5, GEOS-FP, or MERRA-2 via TOML-configured readers. Adding a
  new met source requires only a TOML config file and (if different) a
  coefficient TOML -- no Julia code changes.
