# Visualization

AtmosTransport visualization is split into a lightweight data layer and an
optional Makie rendering layer.

The data layer lives in `AtmosTransport.Visualization` and loads with the core
package. It understands current snapshot NetCDF output from LL, RG, and CS
runs, exposes topology-native field views, and can rasterize fields for fast
debug maps.

Makie plotting methods load only after a Makie backend is loaded:

```julia
using CairoMakie
using AtmosTransport

snap = open_snapshot("run_snapshot.nc")
fig = snapshot_grid(snap, :co2; unit = :ppm, cols = 4)
save("co2_grid.png", fig)
```

Use `CairoMakie` for deterministic PNG/PDF/SVG output and `GLMakie` for fast
interactive debugging. These backends are intentionally optional; install one
in the active Julia environment before using the plotting API.

## Core API

```julia
snap = open_snapshot("run_snapshot.nc")
available_variables(snap)
snapshot_times(snap)

field = fieldview(snap, :co2;
                  transform = :column_mean,
                  time = 24.0,
                  unit = :ppm)

raster = as_raster(field; resolution = (360, 181))
```

Supported transforms:

| Transform | LL/RG snapshots | CS snapshots |
|-----------|-----------------|--------------|
| `:column_mean` | yes | air-mass-weighted vertical mean |
| `:level_slice` | no | one model level, requires `level` |
| `:surface_slice` | no | bottom level by default |
| `:column_sum` | no | vertical sum |

## Plotting API

```julia
using CairoMakie
using AtmosTransport

snap = open_snapshot("run_snapshot.nc")

fig = mapplot(fieldview(snap, :co2; time = 24.0, unit = :ppm))
save("co2_24h.png", fig)

fig = snapshot_grid(snap, :co2;
                    transform = :column_mean,
                    times = 0:6:72,
                    unit = :ppm,
                    cols = 4)
save("co2_grid.png", fig)

movie(snap, :co2, "co2.gif";
      transform = :column_mean,
      times = 0:6:72,
      unit = :ppm,
      fps = 6)
```

For multi-panel movies:

```julia
specs = [
    PlotSpec(:co2; transform = :column_mean, unit = :ppm, title = "column mean"),
    PlotSpec(:co2; transform = :surface_slice, unit = :ppm, title = "surface"),
]

movie_grid(snap, specs, "co2_panels.mp4"; times = 0:6:72, fps = 6)
```

## CLI

```bash
julia --project=. scripts/visualization/atmos_viz.jl \
    --input  ~/data/AtmosTransport/output/run_snapshot.nc \
    --tracer co2 \
    --kind   grid \
    --ppm \
    --out    artifacts/visualization/co2_grid.png
```

Movie output uses the same field and topology logic:

```bash
julia --project=. scripts/visualization/atmos_viz.jl \
    --input  ~/data/AtmosTransport/output/run_snapshot.nc \
    --tracer co2 \
    --kind   movie \
    --times  0,6,12,18,24 \
    --ppm \
    --out    artifacts/visualization/co2.gif
```

## Topology Policy

The debug path is intentionally fast:

- LL snapshots plot directly on their lon-lat grid.
- RG snapshots carry native cell-indexed fields plus the ring-aware diagnostic
  lon-lat raster written for quick plots.
- CS snapshots are conservatively regridded to a regular lon-lat raster and
  cache the CS-to-LL geometry across frames. Both `gnomonic` and
  `geos_native` panel conventions use the same convention-aware mesh/treeify
  path as runtime output.

The extension point for future stretched-CS or zoom meshes is the same as the
transport stack: add a topology type, implement `fieldview`/`as_raster` or a
native-cell plot recipe, and keep the public plotting API unchanged.
