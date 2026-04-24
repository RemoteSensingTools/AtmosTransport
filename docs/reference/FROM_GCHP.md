# Coming from GCHP

Quick reference for GEOS-Chem/GCHP users switching to AtmosTransport.jl.

## Concept mapping

| GCHP (Fortran/C) | AtmosTransport.jl | Notes |
|---|---|---|
| `fv_tp_2d` (Lin-Rood) | `SlopesScheme` / `PPMScheme` on CS | Same stencil mathematics |
| `fv_tracer2d.F90` | `strang_split_cs!` | Panel-based Strang split |
| Halo exchange (ESMF) | `fill_panel_halos!` | Per-direction edge + corner fill |
| `MFXC`, `MFYC` (dry mass flux) | `am_panels`, `bm_panels` | Same physical quantity |
| `DELP` (moist) | `m_panels` (mass) | We use mass = dp * area / g |
| `ak`, `bk` (hybrid coord) | `HybridSigmaPressure` | Same A+B*ps formula |
| C180/C720 panels | `CubedSphereMesh(Nc=180)` | Same gnomonic projection |
| Panel ordering 1-6 | `GnomonicPanelConvention()` | Matches GCHP/FV3 default |
| `calcScalingFactor` | Not needed — direct cumsum PE | See CLAUDE.md on hybrid PE bug |

## Key differences

### Panel convention
AtmosTransport supports two CS conventions:
- `GnomonicPanelConvention()` — standard mathematical (+X, +Y, +Z, -X, -Y, -Z)
- `GEOSNativePanelConvention()` — GEOS-FP/IT file ordering

GCHP uses the FV3 convention. Set `convention=GnomonicPanelConvention()` for
compatibility with FV3/GCHP algorithms.

### Moist vs dry basis
GCHP runs on MOIST basis (`Use_Total_Air_Pressure > 0`). AtmosTransport
supports both via `TransportPolicy`:
- `mass_basis = :moist` — GCHP-compatible path
- `mass_basis = :dry` — native dry-air transport

### Vertical remap
GCHP's `map1_q2` conservative PPM remap is implemented in
`src/Operators/Advection/vertical_remap.jl`. Uses direct cumsum PE
(not hybrid formula) to avoid the 250 Pa mismatch between DELP and ak+bk*ps.

### No ESMF dependency
GCHP uses ESMF for regridding and halo exchange. AtmosTransport uses:
- `ConservativeRegridding.jl` for LL→CS regridding
- Custom `fill_panel_halos!` for CS halo exchange (same algorithm, no ESMF)

## Data pipeline

### Using GEOS-FP/IT data directly
GEOS-FP C720 and GEOS-IT C180 data can be used with AtmosTransport:
```toml
[grid]
type = "cubed_sphere"
Nc = 720  # or 180 for GEOS-IT

[met_data]
source = "geosfp"  # or "geosit"
mass_flux_dt = 450  # CRITICAL: dynamics timestep in seconds
```

### Using ERA5 spectral data on CS
AtmosTransport's unique feature: spectral→CS preprocessing pipeline.
```toml
[grid]
type = "cubed_sphere"
Nc = 90  # or any resolution

[input]
spectral_dir = "~/data/AtmosTransport/met/era5/spectral_hourly"
```

### Regridding existing LL binaries to CS
```julia
using AtmosTransport.Preprocessing
grid = build_target_geometry(Dict("type"=>"cubed_sphere", "Nc"=>90), Float64)
regrid_ll_binary_to_cs("era5_ll.bin", grid, "era5_cs_c90.bin";
                        met_interval=3600.0, dt=900.0)
```

## Advection schemes on CS

| Scheme | GCHP equivalent | Hp | Config |
|--------|----------------|-----|--------|
| `UpwindScheme()` | 1st order upwind | 1 | `scheme = "upwind"` |
| `SlopesScheme()` | van Leer (GCHP default) | 2 | `scheme = "slopes"` |
| `PPMScheme()` | Putman-Lin PPM | 3 | `scheme = "ppm"` |

The `Hp` (halo padding) must be set when constructing `CubedSphereMesh`:
```julia
mesh = CubedSphereMesh(Nc=90, Hp=3)  # for PPM
```

## File locations

| What | GCHP | AtmosTransport |
|------|------|----------------|
| Horizontal advection | `fv_tp_2d.F90` | `src/Operators/Advection/CubedSphereStrang.jl` |
| Vertical remap | `fv_mapz.F90` | `src/Operators/Advection/vertical_remap.jl` |
| Halo exchange | ESMF `halo` | `src/Operators/Advection/HaloExchange.jl` |
| Panel connectivity | `CubedSphereGridComp` | `src/Grids/PanelConnectivity.jl` |
| CS mesh geometry | FV3 grid generation | `src/Grids/CubedSphereMesh.jl` |
| Met driver | `HEMCO` / `ExtData` | `src/MetDrivers/` |
| CS preprocessing | `offline_tracer_advection` | `src/Preprocessing/transport_binary/cubed_sphere_spectral.jl` |

## Quick start

See [QUICKSTART.md](QUICKSTART.md) for end-to-end setup.

For GCHP-equivalent CS runs:
1. Preprocess: `julia scripts/preprocessing/preprocess_transport_binary.jl config/preprocessing/era5_cs_c90_transport_binary.toml --day 2021-12-01`
2. Run: `julia scripts/run.jl config/runs/your_cs_config.toml`
3. Use `scheme = "slopes"` for GCHP-equivalent advection (or `"ppm"` for higher accuracy)
