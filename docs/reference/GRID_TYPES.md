# Grid Types

AtmosTransport supports three horizontal grid types, each with different
tradeoffs for resolution, accuracy, and computational cost.

## Overview

| Grid | Type | Resolution | Cells | Scheme support | GPU | Use case |
|------|------|-----------|-------|---------------|-----|----------|
| LatLon (LL) | Structured | 0.5° typical | 720×361 | All (upwind, slopes, PPM) | Yes | Reference, ERA5 native |
| Reduced Gaussian (RG) | Unstructured | N320 typical | ~210k | Upwind only | Yes | ERA5 native resolution |
| Cubed Sphere (CS) | Panel-structured | C90 typical | 6×90² | All + LinRood | Yes | GEOS data, uniform resolution |

## LatLon (LL) — `LatLonMesh`

Regular latitude-longitude grid with uniform spacing in both directions.

**Pros**:
- Simplest data structure (3D arrays)
- All advection schemes supported
- Direct compatibility with ERA5 gridpoint data
- Best for quick prototyping and debugging

**Cons**:
- Polar singularity (cells converge at poles)
- CFL restriction from thin polar cells
- Non-uniform cell area (polar cells ~cos(lat)× smaller)

**Config**: no explicit grid section needed — inferred from binary header.

**Implementation**: `src/Grids/LatLonMesh.jl`, kernels in `structured_kernels.jl`

## Reduced Gaussian (RG) — face-indexed

Gaussian latitudes with fewer longitude points near the poles. Matches
ERA5's native T1279 grid (N640) or subsets like N320.

**Pros**:
- No polar singularity (cells are roughly equal area)
- Matches ERA5 spectral truncation → no interpolation artifacts
- Much fewer cells than equivalent LatLon resolution
- Excellent GPU performance with face-indexed kernels

**Cons**:
- Only upwind advection currently supported (Slopes/PPM planned)
- Unstructured topology → face-indexed kernels with `@atomic` scatter
- Irregular connectivity → harder to implement higher-order stencils

**Implementation**: `src/Grids/ReducedGaussianGrid.jl`, kernels in
`StrangSplitting.jl` (face-indexed path)

## Cubed Sphere (CS) — `CubedSphereMesh`

Six gnomonic panels, each an Nc × Nc structured grid. Panels connect
at edges with rotated coordinate systems.

**Pros**:
- Quasi-uniform resolution globally (no polar singularity)
- Native format for GEOS-FP/IT data
- All schemes including Lin-Rood (eliminates panel-boundary artifacts)
- Scalable to very high resolution (C720 = ~12.5 km)

**Cons**:
- Panel boundaries require halo exchange + coordinate rotation
- Higher-order schemes need larger halo padding (Hp=3 for PPM/LinRood)
- More complex data structures (NTuple{6, Array{FT,3}})

**Config**: `halo_padding = 3` in `[run]` for PPM/LinRood schemes.

**Implementation**: `src/Grids/CubedSphereMesh.jl`, panel sweeps in
`CubedSphereStrang.jl`, halo exchange in `HaloExchange.jl`

## Panel connectivity (CS)

Panels are numbered 1-6 in the GEOS file convention:
```
        ┌───┐
        │ 5 │ (North pole)
    ┌───┼───┼───┬───┐
    │ 4 │ 1 │ 2 │ 3 │ (Equatorial belt)
    └───┼───┼───┴───┘
        │ 6 │ (South pole)
        └───┘
```

At panel boundaries, fluxes are rotated to maintain correct
east/north orientation. Lin-Rood's cross-term averaging handles the
coordinate discontinuity at panel edges.

## Choosing a grid

- **Start with LL** for debugging and prototyping (simplest, all schemes work)
- **Use RG** when you need ERA5 native resolution without interpolation
- **Use CS** for GEOS data, high-resolution global runs, or when panel-boundary
  accuracy matters (enable `linrood` for best results)

## Resolution equivalences

| LL | RG | CS | Approx. spacing |
|----|----|----|----------------|
| 96×48 | N24 | C24 | ~4° |
| 360×181 | N90 | C90 | ~1° |
| 720×361 | N160-N180 | C180 | ~0.5° |
| 1440×721 | N320 | C360 | ~0.25° |
| — | N640 | C720 | ~0.125° |
