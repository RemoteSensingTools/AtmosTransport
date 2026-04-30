# Grid Conventions

This note records the native grid names and panel-order conventions that the
refactor should preserve in `src` instead of collapsing into generic
"regular grid" labels.

## Horizontal grid naming

| Family | Canonical name | Meaning | Notes for AtmosTransport |
|--------|----------------|---------|--------------------------|
| Regular lat-lon | `0.5°×0.5°`, `1.0°×1.0°`, or explicit `(Nx, Ny)` | Uniform longitude/latitude spacing | There is no single archive-wide prefix used everywhere. Keep spacing and cell counts explicit. |
| Regular lon-lat (Atlas notation) | `L<NLON>x<NLAT>` | Regular longitude-latitude grid in degrees | Useful external reference, but note that archive point counts and transport cell counts are not always the same object. |
| Regular Gaussian | `F<N>` | Gaussian grid with `2N` latitude circles and `4N` longitudes | ECMWF regular Gaussian naming. |
| Classic reduced Gaussian | `N<N>` | Reduced Gaussian grid with `2N` latitude circles and tabulated `PL` array | ECMWF classic reduced Gaussian naming. |
| Octahedral reduced Gaussian | `O<N>` | Reduced Gaussian grid with `2N` latitude circles and octahedral `PL` rule | Used by newer IFS products. |
| Cubed sphere | `C<Nc>` | `Nc × Nc` cells on each of 6 panels | GEOS/FV3 naming. `C180` means 180 cells per panel edge. |

## ECMWF Gaussian numbers

| Symbol | Meaning |
|--------|---------|
| `N` in `F<N>`, `N<N>`, `O<N>` | Gaussian number = number of latitude circles between the pole and equator |
| `PL` array | Number of longitude points on each latitude circle in a reduced Gaussian grid |
| `Txxx` / `TLxxx` / `TCOxxx` | Spectral truncation, not the grid itself |

Examples:

| Product | Native naming | Interpretation |
|--------|---------------|----------------|
| ERA5 HRES spectral core | `TL639` | Triangular spectral truncation 639 |
| ERA5 HRES reduced Gaussian companion grid | `N320` | Classic reduced Gaussian with 320 latitude circles from pole to equator |
| Newer IFS HRES | `O1280` / `TCO1279` | Octahedral reduced Gaussian with matching spectral truncation family |

Important: not every integer `N` is a standard classic ECMWF reduced Gaussian
grid name. Atlas tabulates only a discrete set of classic `N` values
(`16, 24, 32, 48, 64, ..., 320, 400, 512, 576, 640, 800, 1024, 1280, ...`),
so a label like `N360` should not be assumed valid without checking the source
dataset.

## GEOS / FV3 cubed-sphere names

| Name | Cells per panel edge | Approximate horizontal scale |
|------|----------------------|------------------------------|
| `C90` | 90 | ~100 km |
| `C180` | 180 | ~50 km |
| `C360` | 360 | ~25 km |
| `C720` | 720 | ~12.5 km |

## Cubed-sphere panel-order conventions

AtmosTransport distinguishes the cubed-sphere **definition** from the native
GEOS file order. A definition is the combination of coordinate law, center law,
panel convention, and longitude offset.

Two coordinate/center pairs are available:

| Definition | Coordinate law | Center law | Main use |
|------------|----------------|------------|----------|
| `EquiangularCubedSphereDefinition()` | `EquiangularGnomonic` | `AngularMidpointCenter` | Synthetic/legacy CS targets |
| `GMAOCubedSphereDefinition()` | `GMAOEqualDistanceGnomonic` | `FourCornerNormalizedCenter` | GEOS-IT C180 and GEOS-FP C720 |

The GMAO edge coordinate law is

```math
r = 1/\sqrt{3}, \quad \alpha_0 = \sin^{-1}(r), \quad
\beta_s = -\frac{\alpha_0}{N_c}(N_c + 2 - 2s),
\quad \xi_s = \frac{\tan(\beta_s)\cos(\alpha_0)}{r}.
```

The GMAO center law is GEOS/FV `cell_center2`:

```math
v_c = \frac{v_1 + v_2 + v_3 + v_4}
           {\lVert v_1 + v_2 + v_3 + v_4\rVert}.
```

This is why native GEOS C180 meridional spacing on panel 1 is about `0.42°`
near panel edges and `0.55°` near the panel center, rather than the uniform
`0.50°` spacing of a simple equiangular midpoint construction.

Panel conventions then describe only file order and panel orientation.

In words:

- Classical gnomonic order: panels 1-4 are equatorial, panel 5 is north-pole, panel 6 is south-pole.
- GEOS native file order: panels 1-2 are equatorial, panel 3 is north-pole, panels 4-5 are equatorial, panel 6 is south-pole.

More explicitly:

| Panel index | Classical gnomonic role | GEOS native file role |
|-------------|-------------------------|------------------------|
| 1 | Equatorial (`x_plus`) | Equatorial |
| 2 | Equatorial (`y_plus`) | Equatorial |
| 3 | Equatorial (`x_minus`) | North pole |
| 4 | Equatorial (`y_minus`) | Equatorial |
| 5 | North pole | Equatorial |
| 6 | South pole | South pole |

`src/Grids/CubedSphereMesh.jl` carries this distinction explicitly via
`CubedSphereDefinition`, `GnomonicPanelConvention`, and
`GEOSNativePanelConvention`. Coordinate helpers such as
`panel_cell_center_lonlat(mesh, panel)` and `panel_cell_corner_lonlat(mesh,
panel)` honor the full definition, so regridding, local wind rotation, CS face
connectivity, Poisson balance, diagnostic NetCDF output, and visualization all
consume the same geometry.

GEOS-native panels 4 and 5 keep GEOS file order: local `Xdim` runs mostly
southward and local `Ydim` runs eastward. Internal edge constants
(`EDGE_NORTH`, `EDGE_SOUTH`, etc.) therefore refer to local index boundaries
(`+Y`, `-Y`, `+X`, `-X`), not always geographic directions. This is
intentional: closure, mirror signs, and replay validation must follow file
indices.

## Sources

- ECMWF Atlas grid naming and Gaussian-number definitions:
  https://sites.ecmwf.int/docs/atlas/design/grid/
- ECMWF relationship between spectral truncation and Gaussian grids:
  https://confluence.ecmwf.int/display/UDOC/What%2Bis%2Bthe%2Bconnection%2Bbetween%2Bthe%2Bspectral%2Btruncation%2Band%2Bthe%2BGaussian%2Bgrids%2B-%2BMetview%2BFAQ
- Existing project cubed-sphere notes:
  `docs/reference/CAVEATS.md`
  `docs/reference/METEO_PREPROCESSING.md`
  `src/Grids/CubedSphereMesh.jl`
