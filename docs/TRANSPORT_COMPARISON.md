# Transport Comparison: AtmosTransport.jl vs TM5 vs GEOS-Chem

This document compares the transport algorithms and design choices of our Julia
model against TM5 (KNMI/SRON, Krol et al. 2005) and GEOS-Chem / GCHP (Harvard,
Martin et al. 2022). The comparison covers six areas critical for accurate offline
chemical transport: advection, polar/CFL handling, mass conservation, operator
splitting, vertical transport, and meteorological preprocessing.

## 1. Advection Scheme Family

### TM5

TM5 uses the Russell & Lerner (1981) **slopes scheme** — a mass-coordinate
upstream method that tracks both the mean concentration and its spatial gradient
(slope) within each cell. The prognostic variables are tracer mass `rm`, air
mass `m`, and three slope components (`rxm`, `rym`, `rzm`).

Key features:

- **Slopes are prognostic**: carried forward in time and advected along with the
  tracer. This preserves sub-grid structure across time steps.
- **Minmod limiter**: applied to slopes to prevent negative concentrations:
  `s = minmod(s, 2*(c_R - c), 2*(c - c_L))`.
- **Mass-based Courant number**: `alpha = am / m_donor` where `am` is the mass
  flux and `m_donor` is the air mass in the donor cell.
- **Mass-flux form**: the flux formula is
  `F = alpha * (rm + (1 - alpha) * sxm / 2)` for positive flow, which is
  equivalent to Russell & Lerner's least-squares fit formula (see Russell &
  Lerner 1981, Eq. 16-22).

Source: `advectx.F90`, `advecty.F90`, `advectz.F90` in TM5.

### GEOS-Chem (Classic and GCHP)

GEOS-Chem Classic uses the **TPCORE** advection scheme (Lin & Rood 1996),
which is a Piecewise Parabolic Method (PPM, Colella & Woodward 1984) variant on
a regular lat-lon grid. GCHP uses **FV3** (Lin 2004), a finite-volume scheme on
the cubed-sphere grid.

Key features:

- **PPM reconstruction**: sub-cell tracer is a parabola (not linear as in slopes).
  Third-order accuracy in smooth regions vs second-order for slopes.
- **Monotonicity constraint**: equivalent to slope limiting but applied to the
  parabolic reconstruction.
- **GCHP/FV3**: native cubed-sphere avoids the polar singularity entirely.
  Advection uses a Lagrangian vertical coordinate with periodic remapping.
- **Mass-flux form**: FV3 computes fluxes from Courant numbers and sub-cell
  reconstructions, similar in spirit to TM5 but with higher-order reconstruction.

Source: Lin & Rood (1996), Lin (2004), Putman & Lin (2007).

### AtmosTransport.jl (this model)

We implement two advection scheme families:

1. **Slopes** (Russell & Lerner 1981): mass-flux form, matching TM5's formulation.
   Diagnostic slopes recomputed each step from `c = rm / m`. 2nd-order accuracy.

2. **PPM** (Putman & Lin 2007): Piecewise Parabolic Method with configurable
   polynomial order (4, 5, 6, or 7). 3rd-order accuracy in smooth regions.
   Implemented for both lat-lon and cubed-sphere grids.

Both schemes use KernelAbstractions.jl for unified CPU/GPU execution. The PPM
implementation follows Putman & Lin (2007) with sub-grid distribution polynomials
computed from cell-averaged values.

The mathematical comparison:

| Property | TM5 | GEOS-Chem Classic | GCHP/FV3 | This Model |
|----------|-----|-------------------|----------|------------|
| Sub-cell reconstruction | Linear (slopes) | Parabolic (PPM) | Parabolic (PPM) | Linear (slopes) or PPM |
| Formal accuracy | 2nd order | 3rd order (smooth) | 3rd order | 2nd (slopes) or 3rd (PPM) |
| Prognostic slopes | Yes | No (recomputed) | No | No (diagnostic) |
| Grid | Lat-lon (reduced) | Lat-lon | Cubed-sphere | Lat-lon + cubed-sphere |
| GPU support | No | No | Limited (ESMF) | Yes (KA) |

## 2. Polar / CFL Handling

The polar singularity in lat-lon grids is a fundamental challenge: as latitude
approaches +/-90 degrees, the zonal grid spacing `Dx = R * cos(phi) * Dlambda`
approaches zero, causing the CFL number to blow up for any finite wind speed.

### TM5: Reduced Grid

TM5 uses a **reduced grid** (Hooghiemstra 2006, KNMI TR-294) that clusters
adjacent zonal cells at high latitudes. For a 1x1 degree grid:

- At the equator: 360 zonal cells, each ~111 km wide
- At 60N: 180 reduced cells (cluster size 2), each ~111 km wide
- At 80N: 36 reduced cells (cluster size 10), each ~111 km wide
- At the pole: 1 cell

The effective zonal spacing stays approximately constant. Before x-advection,
the model:

1. **Reduces**: averages concentrations and sums masses across `r` adjacent cells
2. **Advects**: runs the slopes scheme on the reduced row
3. **Expands**: distributes changes back to the fine cells

This is implemented in TM5's `advectm_cfl.F90` (`uni2red`, `red2uni`). The
reduced grid eliminates x-direction CFL violations without subcycling, so each
row requires exactly one advection pass.

For y-advection, TM5 also clusters at high latitudes. The z-direction has no
polar issue.

### GEOS-Chem Classic: Polar Cap Treatment

GEOS-Chem Classic on lat-lon uses a **polar cap** approach: the polar rows are
averaged into a single cell (or a small number of cells) and treated with a
simplified flux computation. TPCORE also uses a cross-polar flux formulation.

### GCHP: No Polar Problem

GCHP's cubed-sphere grid has no polar singularity. All cells have roughly
uniform area (~1:1.4 ratio between largest and smallest). This is one of the
primary motivations for the cubed-sphere.

### This Model: Reduced Grid (CPU + GPU) + Cubed-Sphere

We implement three strategies:

1. **Reduced grid (CPU + GPU)**: TM5-style reduced grid via `ReducedGridSpec`.
   Cluster sizes per latitude are constrained to divisors of Nx. On GPU, the
   reduce/advect/expand cycle runs entirely on device. This reduces CFL subcycling
   from ~7x to ~1x near the poles, giving a ~9x speedup for ERA5 lat-lon runs.

2. **CFL-adaptive subcycling (GPU fallback)**: When reduced grid is not used,
   the maximum CFL across all latitudes determines `n_sub = ceil(max_cfl / 0.95)`.
   Mass fluxes are divided by `n_sub` and advection is repeated.

3. **Cubed-sphere (GPU)**: No polar singularity. All cells have roughly uniform
   area. GEOS-FP C720 and GEOS-IT C180 run natively on the cubed-sphere grid
   with panel-boundary flux exchange handled by `panel_connectivity.jl`.

## 3. Mass Conservation

### TM5: Guaranteed by Spectral Integration

TM5 derives mass fluxes from ECMWF spectral harmonics (vorticity, divergence,
log surface pressure) via the TMM module (`tmm.F90`). The computation is
performed in spectral space, where the continuity equation is satisfied exactly
at the truncation level. This guarantees:

- `sum(am_in - am_out + bm_in - bm_out + cm_top - cm_bot) = 0` per column
- Total atmospheric mass is conserved to machine precision
- No post-hoc mass fixers are needed

Reference: Bregman et al. (2003), "On the use of mass-conserving wind fields in
chemistry-transport models."

### GEOS-Chem Classic: Pressure Fixer

GEOS-Chem Classic reads archived gridpoint winds and computes mass fluxes via
TPCORE. Because gridpoint winds do not exactly satisfy the discrete continuity
equation, a **pressure fixer** is applied after each advection step to reconcile
the model's surface pressure with the meteorological surface pressure.

This is a known source of error. Martin et al. (2022) showed that directly
ingesting mass fluxes from native cubed-sphere archives (available since GEOS-FP
March 2021) reduces surface pressure tendency error by 15x.

### GCHP: Native Mass Fluxes

Since GCHP v13 (Martin et al. 2022), GCHP directly reads hourly C720
cubed-sphere mass fluxes archived by GEOS-FP. These are computed by the parent
GCM's dynamical core and satisfy the continuity equation exactly on the native
grid. No restaggering or regridding is needed.

### This Model: Multiple Mass Flux Sources

We support three mass flux pipelines with different conservation properties:

1. **ERA5 spectral** (`preprocess_spectral_massflux.jl`): Converts ERA5 spectral
   harmonics (VO, D, LNSP) to mass-conserving mass fluxes, following TM5's
   approach. Achieves near-zero mass drift. This is the recommended ERA5 pipeline.

2. **ERA5 gridpoint** (`preprocess_mass_fluxes.jl`): Derives mass fluxes from
   gridpoint u/v winds. ~0.9% mass drift per 30 days. Retained as a stopgap.

3. **GEOS-FP/IT cubed-sphere**: Reads native C720/C180 mass fluxes (MFXC, MFYC)
   directly from archived NetCDF. These satisfy the continuity equation on the
   native grid — no restaggering errors.

All pipelines reconstruct vertical mass fluxes from horizontal convergence:

```
cm[k+1] = cm[k] + (am_in - am_out + bm_in - bm_out)[k] - bt[k] * pit
```

**Key insight (from our development)**: the conserved quantity in mass-flux
advection is `sum(rm)` (total tracer mass), not `sum(c)` (sum of concentrations).
See `MASS_FLUX_EVOLUTION.md` for details.

## 4. Operator Splitting

### TM5: Symmetric Strang Splitting

TM5 uses a symmetric Strang split over a 6-hour window:

```
X(dt/2) → Y(dt/2) → Z(dt/2) → Z(dt/2) → Y(dt/2) → X(dt/2)
```

where `dt = 6h / n_steps` is the advection sub-step. Sources, convection, and
diffusion are applied between advection windows. The order within the Strang
split alternates between windows (XYZZYX, then YXZZXY) to reduce splitting
error over long integrations.

Source: TM5 `advect_tools.F90` (`dynam1`, lines 280-320).

### GEOS-Chem Classic: Sequential Splitting

GEOS-Chem Classic applies operators sequentially within each time step:

```
Emissions → Chemistry → Convection → Advection (TPCORE) → Boundary Layer Mixing
```

Advection itself uses directional splitting (X-Y-Z with alternating order).
The overall scheme is first-order in splitting error.

### GCHP: Process Splitting via MAPL

GCHP uses the MAPL framework for operator splitting. Transport (FV3 advection)
and chemistry/emissions are coupled through ESMF. The splitting order and
frequency are configurable.

### This Model: Symmetric Strang Splitting (TM5-aligned)

We use the same Strang split as TM5:

```
X(dt/2) → Y(dt/2) → Z(dt/2) → Z(dt/2) → Y(dt/2) → X(dt/2)
```

within `strang_split_massflux!()`. The air mass `m` is passed through the
entire split without resetting, matching TM5's continuous mass tracking.

Sources (EDGAR emissions) are injected once per meteorological window (before
the advection sub-steps), while convection and diffusion are applied after
advection in the same window. This matches TM5's operator-level splitting.

## 5. Vertical Transport: Convection and Boundary Layer Mixing

### TM5

**Convection**: TM5 uses the Tiedtke (1989) mass-flux convection scheme with
ECMWF-archived convective mass fluxes (updraft/downdraft mass flux, entrainment,
detrainment). The convective tendency is:

```
dq/dt = g/Dp * (F[k+1] - F[k])
```

where `F[k] = M_net[k] * q_upwind` with upwind selection based on the sign
of the net mass flux.

**Boundary layer mixing**: TM5 uses a first-order closure (K-diffusion) scheme
with diffusivity `Kz` derived from ECMWF boundary layer diagnostics (BL height,
surface fluxes). An implicit tridiagonal solve ensures stability for large `Kz`.

Both schemes are applied column-by-column after the advection Strang split.

### GEOS-Chem

**Convection**: GEOS-Chem uses archived convective mass fluxes from the parent
GCM (GEOS-FP provides updraft, downdraft, entrainment, detrainment). The
convection module handles deep convection (Relaxed Arakawa-Schubert or Zhang &
McFarlane, depending on GEOS version) and shallow convection separately.

GEOS-Chem also applies wet scavenging within the convection module — a feature
not present in TM5 or our model.

**Boundary layer mixing**: GEOS-Chem uses GEOS-FP archived turbulent diffusivity
profiles or a non-local PBL scheme (VDIFF module). The implementation uses an
implicit tridiagonal solve, similar to TM5.

### This Model

**Convection**: Fully implemented Tiedtke (1989) mass-flux scheme
(`tiedtke_convection.jl`) with discrete adjoint (`tiedtke_convection_adjoint.jl`).
The forward operator is linear in tracer concentration when mass fluxes are
prescribed, so the adjoint is exact (dot-product test passes to machine
precision).

Currently, convection requires `conv_mass_flux` in the meteorological input.
ERA5 provides convective mass flux fields (MUMF param 20, MDMF param 21) but
our ERA5 configuration does not yet include them. GEOS-FP provides them
natively and the GEOS-FP reader already supports them.

**Boundary layer mixing**: Fully implemented first-order K-diffusion with
implicit tridiagonal solve (`boundary_layer_diffusion.jl`) and adjoint
(`boundary_layer_diffusion_adjoint.jl`). Falls back to an exponential `Kz`
profile when met data does not provide diffusivity.

Status: both modules are implemented, tested, and have adjoints, but are not
yet integrated into the ERA5 forward run script. The GEOS-FP scripts
(`run_forward_geosfp.jl`, `run_point_sources_geosfp.jl`) already include
convection and diffusion.

## 6. Meteorological Preprocessing

### TM5: Spectral Processing via TMM

TM5's meteo module (TMM) reads raw ECMWF data as **spectral harmonic
coefficients** (vorticity, divergence, LNSP) and computes mass-conserving
mass fluxes via integration in spectral space. This is the most
mathematically rigorous approach, guaranteeing exact mass conservation at the
truncation level.

The pipeline: `ECMWF GRIB (spectral) → TMM → mass fluxes (am, bm, cm) + ps`.

TM5 can run in preprocessing mode (`tmm.output: T`) to generate preprocessed
NetCDF files, then subsequent runs read from this archive — analogous to our
`preprocess_mass_fluxes.jl` / `run.jl` with a preprocessed config workflow.

Required raw fields: spectral vorticity (VO, param 138), divergence (D, param
155), log surface pressure (LNSP, param 152). For convection: MUMF, MDMF, EU,
ED, DU, DD.

### GEOS-Chem Classic: Direct Wind Reading

GEOS-Chem Classic reads gridpoint winds directly from archived GEOS-FP 0.25-deg
NetCDF files and computes mass fluxes via TPCORE. A pressure fixer corrects
mass conservation errors.

### GCHP v13+: Native Mass Flux Archives

Since March 2021, GEOS-FP operationally archives hourly C720 cubed-sphere mass
fluxes (`MFXC`, `MFYC`). GCHP reads these directly, avoiding all
restaggering/regridding errors. This is the gold standard for offline transport.

GEOS-IT provides C180 archives (1998-present) for the same purpose.

### This Model: Multiple Pipelines

Three met data pipelines:

**ERA5 spectral (recommended):**
```
ERA5 GRIB (spectral VO, D, LNSP)
    → preprocess_spectral_massflux.jl (spectral integration)
    → NetCDF mass fluxes (am, bm, cm, m, ps)
    → forward run
```

**ERA5 gridpoint (stopgap):**
```
ERA5 NetCDF (gridpoint u, v, lnsp)
    → preprocess_mass_fluxes.jl (gridpoint winds × Dp/g × dx)
    → NetCDF or binary mass fluxes
    → forward run
```

**GEOS-FP/IT cubed-sphere:**
```
GEOS-FP/IT NetCDF (native C720/C180 MFXC, MFYC, DELP)
    → (optional) preprocess_geosfp_cs.jl → flat binary
    → forward run (reads NetCDF or binary directly)
```

### Comparison Summary

| Aspect | TM5 | GEOS-Chem Classic | GCHP v13+ | This Model |
|--------|-----|-------------------|-----------|------------|
| Input format | Spectral GRIB | Gridpoint NetCDF | Cubed-sphere NetCDF | Spectral GRIB, gridpoint NetCDF, or CS NetCDF |
| Mass flux source | Spectral integration | TPCORE from winds | Native GCM archive | Spectral integration, gridpoint winds, or native archive |
| Conservation | Machine precision | Pressure fixer | Machine precision | Near-zero (spectral), ~0.9%/mo (gridpoint), native (CS) |
| Preprocessing | TMM (offline option) | None (online) | None (direct read) | Offline (spectral or gridpoint) or direct read (CS) |
| Vertical fluxes | From spectral div | From continuity | Native (consistent) | From continuity |
| Grid support | Lat-lon (reduced) | Lat-lon | Cubed-sphere | Lat-lon (reduced grid) + cubed-sphere |

## Roadmap: Bridging the Gaps

1. ~~**Reduced-grid mass-flux advection**~~: **DONE.** TM5-style reduced grid
   on GPU reduces CFL subcycling from ~7x to ~1x (9x ERA5 speedup).

2. **ERA5 convective mass fluxes**: Add MUMF/MDMF download and config to enable
   Tiedtke convection in ERA5 runs (already working for GEOS-FP).

3. **Boundary layer diffusion with met-driven Kz**: Add ERA5 boundary layer
   height (`blh`, param 159) to the config and derive Kz profiles from it.

4. ~~**Spectral mass flux preprocessing**~~: **DONE.** `preprocess_spectral_massflux.jl`
   converts ERA5 spectral VO/D/LNSP to mass-conserving mass fluxes.

5. ~~**Native cubed-sphere support**~~: **DONE.** Full GPU pipeline for GEOS-FP C720
   and GEOS-IT C180 with panel-boundary flux exchange, PPM advection, BL diffusion,
   and output regridding to lat-lon.

6. ~~**PPM advection option**~~: **DONE.** Putman & Lin (2007) PPM with orders 4-7,
   implemented for both lat-lon and cubed-sphere grids.

## References

- Russell, G.L. and Lerner, J.A. (1981). "A new finite-differencing scheme for
  the tracer transport equation." J. Appl. Meteorol., 20, 1483-1498.
- Tiedtke, M. (1989). "A comprehensive mass flux scheme for cumulus
  parameterization in large-scale models." Mon. Wea. Rev., 117, 1779-1800.
- Lin, S.-J. and Rood, R.B. (1996). "Multidimensional flux-form semi-Lagrangian
  transport schemes." Mon. Wea. Rev., 124, 2046-2070.
- Lin, S.-J. (2004). "A 'vertically Lagrangian' finite-volume dynamical core for
  global models." Mon. Wea. Rev., 132, 2293-2307.
- Putman, W.M. and Lin, S.-J. (2007). "Finite-volume transport on various
  cubed-sphere grids." J. Comput. Phys., 227, 55-78.
- Krol, M., Houweling, S., Bregman, B., et al. (2005). "The two-way nested
  global chemistry-transport zoom model TM5: algorithm and applications."
  Atmos. Chem. Phys., 5, 417-432.
- Bregman, B., Segers, A., Krol, M., Meijer, E., and van Velthoven, P. (2003).
  "On the use of mass-conserving wind fields in chemistry-transport models."
  Atmos. Chem. Phys., 3, 447-457.
- Hooghiemstra, P.B. (2006). "Towards advection on a full reduced grid for TM5."
  KNMI Technical Report TR-294.
- Martin, S.T., et al. (2022). "Improved advection, resolution, performance, and
  community access in the new generation model GCHP version 13." Geosci. Model
  Dev., 15, 8731-8748.
- Williams, J.E., et al. (2017). "The global chemistry transport model TM5:
  description and evaluation of the tropospheric chemistry version 3.1."
  Geosci. Model Dev., 10, 721-764.
- Prather, M.J. (1986). "Numerical advection by conservation of second-order
  moments." J. Geophys. Res., 91, 6671-6681.
- Colella, P. and Woodward, P.R. (1984). "The Piecewise Parabolic Method (PPM)
  for gas-dynamical simulations." J. Comput. Phys., 54, 174-201.
