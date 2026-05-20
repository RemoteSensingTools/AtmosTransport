# Preprocessing Philosophy

Preprocessing is where AtmosTransport turns source-specific meteorology and
inventories into runtime-friendly artifacts. The goal is simple: keep the
forward model focused on transport, and push source-specific mass-flux
construction, conservative remapping, and file-shaping offline.

Use this page as the preprocessing front door. The deeper science and
format-specific details stay in the existing reference docs.

## Why preprocessing exists

- Keep runtime I/O simple and fast: flat binaries load with mmap and avoid
  repeated NetCDF/GRIB decoding.
- Make source-specific work explicit: spectral integration, topology changes,
  and conservative remapping happen once, not inside every run.
- Preserve a stable contract: different preprocessors can feed the same
  transport-binary family when they agree on semantics.

## Choose a path

| Family | Input form | Output artifact | Runtime consumer | Preferred use case |
|--------|------------|-----------------|------------------|--------------------|
| Raw-source acquisition | Remote GRIB / NetCDF archives | Local raw files | Downstream preprocessors | You are staging ERA5, GEOS-FP/IT, or emission inventories |
| Meteorology preprocessing | Raw met products on native grids | Mass-flux NetCDF or transport-ready arrays | Transport-binary shapers or direct runtime readers | You need hybrid-coordinate mass fields and fluxes derived from source products |
| Transport-binary shaping | Structured or face-indexed transport fields | `.bin` transport binary | `preprocessed_latlon` driver or CS binary readers | You want fast repeated runtime ingestion |
| Conservative / regridding paths | Fields already defined on one grid | Same-family artifact on a new grid | Target-grid met/emission readers | You are crossing LL, reduced-Gaussian, or cubed-sphere topologies |
| Emissions preprocessing | Gridded inventory NetCDF | Emission binary on model grid | Surface-flux loaders | You are running CS simulations with lat-lon inventories |

## Stable Transport-Binary API

The canonical config-driven entrypoint is:

```bash
julia -t8 --project=. scripts/preprocessing/preprocess_transport_binary.jl \
    config/preprocessing/<config>.toml --day 2021-12-01
```

Native-source configs declare `[source].toml` plus `[source].root_dir`; ERA5
spectral configs use the historical `[input].spectral_dir` shape. Both paths
converge on `AtmosTransport.Preprocessing.process_day`.

The supported binary-in / binary-out bridge is the LL-to-CS regrid CLI:

```bash
julia -t16 --project=. scripts/preprocessing/regrid_ll_transport_binary_to_cs.jl \
    --input era5_latlon.bin --output era5_cs.bin --Nc 90
```

New source families should add an `AbstractMetSettings` loader and route through
`process_day`. New binary regrid pairs should extend
`Preprocessing.regrid_transport_binary` instead of adding another parallel
runner.

## Config-Driven ERA5 Spectral Path

ERA5 spectral preprocessing now uses the same `preprocess_transport_binary.jl`
entrypoint as the other supported source and topology pairs:

```bash
julia -t8 --project=. scripts/preprocessing/preprocess_transport_binary.jl \
    config/preprocessing/era5_latlon_transport_binary_v2.toml --day 2021-12-01
```

The historical `*_v2.toml` config names are still common because they name the
transport-binary product family, not a separate runner. For new target grids,
add the target geometry under `[grid]` and route the work through
`process_day(date, grid, settings, vertical; ...)`.

## Go Deeper

- [QUICKSTART.md](QUICKSTART.md): end-to-end download, preprocess, and run flow
- [METEO_PREPROCESSING.md](METEO_PREPROCESSING.md): met-source deep dive, hybrid coordinates, and TM5 comparison
- [CONSERVATIVE_REGRIDDING.md](CONSERVATIVE_REGRIDDING.md): conservative LL/RG/CS remapping details and the CS conservative transport path
- [BINARY_FORMAT.md](BINARY_FORMAT.md): topology-generic transport-binary family and record model
- [EMISSION_REGRIDDING.md](EMISSION_REGRIDDING.md): inventory-to-model-grid emission preprocessing

This page should stay short. Put science derivations, validation notes, and
format edge cases in the deeper references above.
