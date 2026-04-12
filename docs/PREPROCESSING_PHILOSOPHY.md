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

Stable repo entrypoints:

```julia
build_transport_binary_v2_target(kind::Symbol, argv; FT=Float64)
run_transport_binary_v2_preprocessor(target)
target_summary(target)
```

The public entrypoint is `kind::Symbol`. The internal `Val(...)` builders stay
behind the dispatch seam and should not be the primary user-facing API.

Current binary-in / binary-out target kinds:
- `:cubed_sphere_bilinear`
- `:cubed_sphere_conservative`

Extension contract for new targets:

| Hook | Purpose |
|------|---------|
| `prepare_transport_binary_v2_target` | Build reusable context after opening the source binary |
| `collect_transport_binary_v2_windows` | Produce output windows for the target |
| `build_transport_binary_v2_header` | Build the output header dictionary |
| `write_transport_binary_v2_output` | Write the final artifact and return bytes written |

CLI example:

```bash
julia --project=. scripts/preprocessing/preprocess_era5_cs_conservative_v2.jl \
    --input era5_latlon.bin --output era5_cs.bin --Nc 90
```

Programmatic example:

```julia
include("scripts/preprocessing/transport_binary_v2_dispatch.jl")
include("scripts/preprocessing/transport_binary_v2_cs_conservative.jl")

target = build_transport_binary_v2_target(
    :cubed_sphere_conservative,
    ["--input", "era5_latlon.bin", "--output", "era5_cs.bin", "--Nc", "90"],
)

println(target_summary(target))
run_transport_binary_v2_preprocessor(target)
```

Minimal new-target skeleton:

```julia
struct MyTarget <: AbstractTransportBinaryV2Target
    input_path::String
    output_path::String
end

target_input_path(t::MyTarget) = t.input_path
target_output_path(t::MyTarget) = t.output_path
target_float_type(::MyTarget) = Float64

prepare_transport_binary_v2_target(target, reader) = ...
collect_transport_binary_v2_windows(target, ctx, reader) = ...
build_transport_binary_v2_header(target, ctx, reader, windows) = Dict(...)
write_transport_binary_v2_output(target, ctx, reader, header, windows) = ...
```

## Go Deeper

- [QUICKSTART.md](QUICKSTART.md): end-to-end download, preprocess, and run flow
- [METEO_PREPROCESSING.md](METEO_PREPROCESSING.md): met-source deep dive, hybrid coordinates, and TM5 comparison
- [CONSERVATIVE_REGRIDDING.md](CONSERVATIVE_REGRIDDING.md): conservative LL/RG/CS remapping details and the CS conservative transport path
- [BINARY_FORMAT.md](BINARY_FORMAT.md): topology-generic transport-binary family and record model
- [EMISSION_REGRIDDING.md](EMISSION_REGRIDDING.md): inventory-to-model-grid emission preprocessing

This page should stay short. Put science derivations, validation notes, and
format edge cases in the deeper references above.
