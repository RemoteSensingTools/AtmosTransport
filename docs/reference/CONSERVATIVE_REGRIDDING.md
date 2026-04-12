# Conservative Regridding in AtmosTransport

This document describes how AtmosTransport performs truly conservative
(mass-preserving) regridding between different horizontal mesh types,
using [ConservativeRegridding.jl](https://github.com/JuliaGeo/ConservativeRegridding.jl)
(CR.jl) as the weight computation engine.

## What "conservative" means

Conservative regridding preserves the **area-weighted integral** of a field.
If `f` is a scalar field and `A_i` is the area of cell `i`, then:

    sum(f_src[i] * A_src[i]) == sum(f_dst[j] * A_dst[j])

to machine precision. This is critical for tracer mass budgets: a field
representing kg/m^2 of CO2 should have the same global mass before and after
regridding.

The alternative (bilinear interpolation) does NOT preserve this integral.
Bilinear is fine for smooth intensive fields like temperature or wind, but
for extensive quantities (mass, flux densities) it introduces artificial
sources and sinks.

## How it works: spherical polygon intersection

CR.jl computes the exact area of intersection between every source and
destination cell pair on the unit sphere:

```
1. Treeify:   Convert source and destination grids into spatial trees
              (quadtree cursors over CellBasedGrid vertices)

2. Dual DFS:  Walk both trees simultaneously, using SphericalCap bounding
              regions to prune cell pairs that cannot intersect. This
              reduces the O(N_src * N_dst) brute-force search to
              O(N * log N) in practice.

3. Clip:      For each candidate pair, compute the intersection polygon
              using ConvexConvexSutherlandHodgman clipping on the sphere.

4. Area:      Compute the spherical area of the intersection polygon
              (spherical excess / Girard's theorem via GeometryOps.area).

5. Assemble:  Store intersection areas in a sparse matrix A[dst, src].
              Compute cell areas: src_areas = sum(A, dims=1),
              dst_areas = sum(A, dims=2).
```

The regridding operation is then a simple sparse matrix-vector multiply:

    dst_field = (A * src_field) ./ dst_areas

## Supported mesh types

### LatLonMesh

Regular latitude-longitude grid with uniform spacing. Face vectors `lambda_f`
(longitude) and `phi_f` (latitude) define the cell edges.

**Tree construction:**
An `(Nx+1) x (Ny+1)` matrix of cell-face corners is built as
UnitSphericalPoints (3D Cartesian on the unit sphere). This is wrapped in
`CellBasedGrid{Spherical}` and `TopDownQuadtreeCursor` for O(log N) pruning.
Full-sphere meshes (360-degree lon, +/-90 lat) get `KnownFullSphereExtentWrapper`
to skip root-level extent computation.

**Longitude convention:** The mesh's `lambda_f` vector may use `[-180, 180]`
(default) or `[0, 360]`. The UnitSphericalPoint representation is unambiguous
regardless of convention.

**Cell indexing:** Column-major: `linear_idx = i + (j-1) * Nx`, where `i` is
the longitude index and `j` is the latitude index. Matches
`vec(reshape(field, Nx, Ny))`.

### CubedSphereMesh

Gnomonic equidistant cubed-sphere with 6 panels of `Nc x Nc` cells.

**Tree construction:**
For each panel, an `(Nc+1) x (Nc+1)` corner-point matrix is generated from
the analytical gnomonic projection (`_gnomonic_xyz`). Each panel is wrapped
in `CellBasedGrid{Spherical}` and `IndexOffsetQuadtreeCursor` with offset
`(p-1) * Nc^2`, so global cell indices span `1 : 6*Nc^2`. All 6 panels are
assembled into a `CubedSphereToplevelTree`.

**Panel conventions:**
- `GnomonicPanelConvention`: panels 1-6 = X+, Y+, X-, Y-, N pole, S pole
- `GEOSNativePanelConvention`: panels 1-6 = equatorial 1, equatorial 2,
  north pole, equatorial 4, equatorial 5, south pole. Remapped internally
  via `_gnomonic_panel_id`.

**Cell indexing:** Global flat index = `(panel-1) * Nc^2 + i + (j-1) * Nc`,
where `i` = xi-index, `j` = eta-index (column-major within each panel).

**Known limitation:** Uses analytical gnomonic coordinates. Real GEOS-FP
data has ~1-2 degree corner offsets and a 90-degree CW local-axis rotation
on panels 4 and 5. For bit-exact parity with production GEOS-FP binaries,
GMAO coordinate loading needs to be ported to v2.

### ReducedGaussianMesh

Variable-resolution latitude rings (native ERA5/IFS format). Each ring has
a different number of longitude cells.

**Tree construction:**
Each latitude ring `j` is an independent `CellBasedGrid{Spherical}` with
shape `(nlon+1, 2)` (longitude face edges x 2 latitude faces). The per-ring
trees are `IndexOffsetQuadtreeCursor`s with offset `ring_offsets[j] - 1`,
combined into a `MultiTreeWrapper`. The whole thing is wrapped in
`KnownFullSphereExtentWrapper`.

**Longitude convention:** Within each ring, cells span `[0, 360)` uniformly
with spacing `dlon = 360 / nlon`. Cell `i` covers
`[(i-1)*dlon, i*dlon]`.

**Latitude convention:** Ring boundaries from `mesh.lat_faces` (south to
north, `lat_faces[1] = -90`, `lat_faces[end] = +90`). Polar face latitudes
are clamped by 0.001 degrees to avoid degenerate polygons where all
pole-side corners collapse to a single point on the unit sphere (which
causes NaN in the SphericalCap extent computation).

**Cell indexing:** Ring-flattened south-to-north: `global_idx =
ring_offsets[j] + in_ring_idx - 1`. Matches `cell_index(mesh, i, j)`.

## The CR.jl bug fix

During development, we discovered and fixed two bugs in CR.jl's
`circle_from_four_corners` function (ConservativeRegridding.jl commit
`b9f1d6f`):

1. **UnitSphericalPoint double-conversion:** The function unconditionally
   ran `UnitSphereFromGeographic()` on all input points, including points
   that were already UnitSphericalPoints. This reinterpreted Cartesian
   `(x, y, z)` as geographic `(lon, lat)` — producing nonsensical
   coordinates.

2. **Degenerate center vector:** When the four corners of a cell-range
   extent are symmetrically placed around the origin (e.g., a latitude
   band straddling the equator with exactly 180 degrees of longitude
   span), their Cartesian sum is `(0, 0, 0)`, and `normalize((0,0,0))`
   produces NaN. This silently poisoned SphericalCap extents, causing the
   dual DFS to prune entire subtrees (NaN comparisons always return false).

Both bugs surfaced only when `CellBasedGrid` stores UnitSphericalPoint
vertices directly (which happens for ReducedGaussianMesh and CubedSphereMesh
treeify). The fix (a `_to_unit_sphere` pass-through + fallback centroid
computation) is a candidate for an upstream PR to JuliaGeo/ConservativeRegridding.jl.

## Weight caching and serialization

### JLD2 cache

`build_regridder(src, dst; cache_dir="/path")` generates a SHA-1 cache key
from the source and destination mesh parameters (grid dimensions, face
coordinates, radius, panel convention). On first call, the regridder is
built and saved to `cache_dir/regridder_<sha1>.jld2`. Subsequent calls with
identical meshes reload the cached weights in milliseconds.

The cache key includes only geometry-affecting fields (not halo widths,
precomputed metric terms, etc.) and a version tag
(`_REGRIDDER_CACHE_VERSION`) to invalidate stale caches when the algorithm
changes.

### ESMF NetCDF export

`save_esmf_weights(path, regridder)` writes the weights in the standard ESMF
offline-weights format consumed by xESMF, ESMF_RegridWeightGen, and GCHP:

| Variable | Dimension | Description |
|----------|-----------|-------------|
| `S`      | `n_s`     | Weight = `intersection_area / dst_cell_area` |
| `row`    | `n_s`     | Destination cell index (1-based) |
| `col`    | `n_s`     | Source cell index (1-based) |
| `frac_a` | `n_a`     | Fraction of source cell covered by destination grid |
| `frac_b` | `n_b`     | Fraction of destination cell covered by source grid |
| `area_a` | `n_a`     | Source cell areas (m^2) |
| `area_b` | `n_b`     | Destination cell areas (m^2) |

For full-sphere-to-full-sphere pairs, `frac_a` and `frac_b` are 1.0 to
machine precision.

## Preprocessing script

`scripts/preprocessing/preprocess_era5_cs_conservative_v2.jl` is a
drop-in counterpart to the existing bilinear
`regrid_latlon_to_cs_binary_v2.jl`. It replaces every bilinear
cell-center interpolation with CR.jl's conservative regridding for
cell-center fields (air mass `m`, surface pressure `ps`, recovered
winds `u`, `v`). The face-flux reconstruction, Poisson balancing, and
vertical-flux diagnosis from continuity are identical to the bilinear
script.

Usage:

```bash
julia --project=. scripts/preprocessing/preprocess_era5_cs_conservative_v2.jl \
    --input <latlon_binary.bin> --output <cs_binary.bin> \
    --Nc 90 [--cache-dir <path>]
```

The output binary is drop-in compatible with `CubedSphereBinaryReader` and
uses the gnomonic panel convention. The header tags
`regrid_method="conservative_crjl"` for provenance.

Both this conservative wrapper and the legacy bilinear wrapper
`scripts/preprocessing/regrid_latlon_to_cs_binary_v2.jl` now route through the
same stable transport-binary target interface documented in
[`PREPROCESSING_PHILOSOPHY.md`](PREPROCESSING_PHILOSOPHY.md). The difference is
the target kind and the regridding method recorded in the output header.

## Verification results

All tests in `test/regridding/runtests.jl` (564 tests, ~24 seconds):

### Mass conservation (constant field = 1.0)

| Direction                    | Relative error       |
|------------------------------|----------------------|
| LatLon(72x36) -> C12         | 3.7e-16             |
| C4 -> LatLon(36x18)          | 4.9e-16             |
| RG(5-ring) -> LatLon(24x12)  | 6.2e-16             |
| RG(5-ring) -> C4              | ~1e-15              |
| LatLon(24x12) -> RG(5-ring)  | coverage ratio 0.987 (polar clamp) |

### Smooth fields

| Field                | Direction            | Relative error       |
|----------------------|----------------------|----------------------|
| cos(lat)             | LatLon(36x18) -> C4  | 7.8e-16             |
| cos(lat)*sin(2*lon)  | LatLon(72x36) -> C12 | ~1e-12 * area_scale |

### ESMF export

| Metric     | Value range               |
|------------|---------------------------|
| frac_a     | [1 - 3e-15, 1 + 1e-15]   |
| frac_b     | [1 - 3e-15, 1 + 1e-15]   |
| S * dst_areas reconstruction | 1.2e-4 abs (float roundoff at 1e12 scale) |

### CubedSphere panel centers

Both `GnomonicPanelConvention` and `GEOSNativePanelConvention` place panel
centers at the correct physical locations on the unit sphere (verified to
1e-12 absolute tolerance against the known Cartesian coordinates of each
cube face pole).

## File inventory

| File | Purpose |
|------|---------|
| `src/Regridding/Regridding.jl` | Module shell, exports |
| `src/Regridding/treeify_meshes.jl` | `Trees.treeify` for all 3 mesh types + `cubed_sphere_face_corners` |
| `src/Regridding/weights_io.jl` | `build_regridder`, JLD2 save/load, ESMF export, `apply_regridder!` |
| `scripts/preprocessing/preprocess_era5_cs_conservative_v2.jl` | ERA5 LL -> CS binary producer |
| `test/regridding/runtests.jl` | Test suite entry point (564 tests) |
| `test/regridding/test_conservation.jl` | Mass conservation across grid pairs |
| `test/regridding/test_cubed_sphere_corners.jl` | Panel geometry golden tests |
| `test/regridding/test_transpose.jl` | Forward + reverse direction round-trip |
| `test/regridding/test_serialization.jl` | JLD2 + ESMF NetCDF round-trip |
| `test/regridding/test_reduced_gaussian_stub.jl` | ReducedGaussianMesh conservation tests |
| `docs/CONSERVATIVE_REGRIDDING.md` | This document |

## Dependencies

Added to `Project.toml` (direct deps):
- `ConservativeRegridding` (local `Pkg.develop` checkout)
- `JLD2` (weight serialization)
- `GeometryOps`, `GeometryOpsCore`, `GeoInterface` (polygon geometry)
- `StaticArrays` (polygon vertex containers)

These are preprocessing-only — the runtime transport core does not call
into the Regridding module.
