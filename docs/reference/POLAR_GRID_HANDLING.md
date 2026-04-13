# Polar Grid Handling: LatLon vs Reduced Gaussian vs TM5

## The Problem

At high latitudes on a regular latitude-longitude grid, cells become very narrow
in longitude. A cell at 89.75°N on a 0.5° grid has longitudinal width
`Δx ≈ 0.5° × cos(89.75°) ≈ 0.004°` — about 450 meters. The zonal mass flux
through this cell is comparable to mid-latitude values, but the cell mass is
tiny. The ratio `flux / mass` (the CFL number) becomes extreme:

| Grid | Typical polar X CFL | Status |
|------|---------------------|--------|
| LL 0.5° (720×361) | 19–47 | Crashes without special handling |
| LL 1° (360×181) | 5–9 | Works (n_sub ≈ 5–9) |
| LL 3.75° (96×48) | 0.6 | Works (n_sub = 1) |
| RG N24 (4608 cells) | 1.6–2.5 | Works (n_sub ≈ 2) |
| RG N320 (production) | ~10–15 (est.) | Should work with subcycling |

The reduced Gaussian grid avoids this by design: polar rings have fewer cells,
so each cell is wider and has more mass relative to the flux.

## Three Approaches

### 1. TM5: Reduced Grid on LatLon ("cluster cells")

TM5 operates on a regular lat-lon grid but **merges polar cells at runtime**
before X-advection. The implementation is in `redgridZoom.F90` (Edwin Spee, CWI):

**Setup** (`initredgrid`):
- Read from config which latitude bands to reduce and by how much
- Example at 3°×2°: rows j=1,2 (south pole) get `clustsize=3` → merge 3 cells
  into 1, reducing 120 cells to 40 per ring
- Config-driven: `region.glb6x4.redgrid.nh.comb : 3 2` means "merge 3 at
  row j=1, merge 2 at row j=2" in the northern hemisphere

**Before X-advection** (`uni2red_mf`, `uni2red`):
- Mass and tracers: sum `clustsize` adjacent cells into one super-cell
  (`m_red[i] = Σ m[is:ie]`, `rm_red[i] = Σ rm[is:ie]`)
- X mass flux: pick every `clustsize`-th face flux
  (`am_red[i] = am[i * clustsize]`)
- Y mass flux: average the `clustsize` faces
- Vertical cm: sum the `clustsize` cells

**X-advection**: operates on the reduced grid with fewer, wider cells → lower CFL

**After X-advection** (`red2uni`):
- Redistribute the super-cell mass back to the original cells proportionally
  (`m[i_orig] = m_red[i_red] × fraction[i_orig]` where fraction is the original
  mass share of each sub-cell)
- Tracer moments also redistributed

**Key property**: the reduction is lossless for mass (sum is preserved) and
approximately lossless for tracer gradients (the redistribution preserves
the original sub-cell mass fractions).

### 2. Our LatLon Path: CFL Subcycling

Instead of merging cells, we subdivide the time step. If the CFL at a cell
exceeds 1, we apply the flux in `n_sub` smaller pieces:

```
n_sub = ceil(max_CFL / cfl_limit)
flux_per_substep = flux / n_sub
```

This works well up to CFL ~10 (n_sub = 10). At CFL 35+ (0.5° polar cells),
the X sweep drains a polar cell's mass, and the subsequent Y sweep sees
near-zero mass → Y CFL becomes infinite → crash.

**Current status**: Works at 1° and coarser. Fails at 0.5° Day 2 (CFL ~47).
The legacy `src_legacy/` code had TM5-style clustering for this — not yet
ported to `src/`.

### 3. Reduced Gaussian Grid: No Polar Problem by Design

ECMWF's native ERA5 grid is octahedral reduced Gaussian (O1280), not lat-lon.
Each latitude ring has a number of cells proportional to `cos(lat)`. Near the
poles, rings have few cells (e.g., 20 cells at 89°) instead of hundreds.

**Advantage**: The CFL never exceeds ~3 even at the poles because the cell
size is matched to the physical scale of the flow. No clustering or extreme
subcycling needed.

**Disadvantage**: Face-indexed (unstructured) topology — requires atomic scatter
kernels on GPU instead of structured stencils. Currently only UpwindScheme is
implemented for RG (SlopesScheme needs face-indexed reconstruction).

## Comparison Table

| Feature | TM5 LatLon | Our LatLon | Our RG |
|---------|-----------|------------|--------|
| Grid storage | Regular LL | Regular LL | Face-indexed |
| Polar handling | Cluster cells at runtime | CFL subcycling | Inherent (fewer cells) |
| Max polar CFL | ~1 (after clustering) | 5 (1°), 35+ (0.5°) | 2.5 |
| X scheme order | Slopes (2nd) | Slopes (2nd) | Upwind (1st) only |
| GPU kernel | Structured | Structured | Atomic scatter |
| 0.5° production | Works (with clustering) | Needs clustering or mass_fixer | N/A (use N320) |

## Recommendation

For ERA5 production at ~0.5° equivalent:
- **Short term**: Use RG N320 (native ERA5 grid, no polar problem)
- **Medium term**: Port TM5 clustering to `src/` for LL 0.5°
- **Long term**: Implement SlopesScheme on RG for 2nd-order accuracy

The clustering approach is ~200 lines of code (see `redgridZoom.F90`). Our
`AdvectionWorkspace` already carries `cluster_sizes::Vector{Int32}` per latitude
row — the infrastructure exists, only the runtime uni2red/red2uni transforms
need implementation.
