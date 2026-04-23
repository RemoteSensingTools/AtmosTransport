# Catrine C48 10-day test configs

Three physics-ladder configs for `scripts/run_cs_driven.jl` at C48, 10 days
(Dec 1–10 2021), ERA5 source met.

`run_cs_driven.jl` is a new driven-simulation CS runner that uses
`CubedSphereTransportDriver` + `CubedSphereState` + `TransportModel` +
`DrivenSimulation`, exposing diffusion and convection via TOML. The
pre-existing `run_cs_transport.jl` (low-level panel-native, advection-only)
is untouched and remains the fast path for pure-advection work.

| Config | Physics | Runs today? |
|---|---|---|
| `advonly.toml` | advection | ✅ yes |
| `advdiff.toml` | + `ImplicitVerticalDiffusion` (constant Kz) | ✅ yes |
| `advdiffconv.toml` | + `TM5Convection` | ⚠️ needs TM5 sections in binary |

## Prerequisites

### 1. C48 transport binaries (required for all three configs)

Neither ERA5 nor GEOS-IT ship C48 binaries. Generate from existing
LL 720×361 v4 F64 binaries via:

```bash
for d in 01 02 03 04 05 06 07 08 09 10; do
  julia -t16 --project=. scripts/preprocessing/regrid_latlon_to_cs_binary_v2.jl \
    --input  ~/data/AtmosTransport/met/era5/ll720x361_v4/transport_binary_v2_tropo34_dec2021_f64/era5_transport_202112${d}_merged1000Pa_float64.bin \
    --output ~/data/AtmosTransport/met/era5/cs_c48/transport_binary_v2_tropo34_dec2021_f64/era5_transport_202112${d}_merged1000Pa_float64.bin \
    --Nc 48
done
```

Runtime ≈ 2–4 min per day at Nc=48. Total ≈ 30 min for 10 days.

### 2. (Only for `advdiffconv.toml`) TM5 convection sections in the binary

`TM5Convection` reads entu/detu/entd/detd from the transport binary.
These are emitted by the ERA5 LL preprocessor when
`[tm5_convection] enable = true` is set (plan 24 Commit 4).

The LL→C48 regridder (`regrid_latlon_to_cs_binary_v2.jl`) does **not**
currently carry the TM5 sections across. Options:

- **(a) Extend the regridder** to pass-through TM5 sections from the LL
  source when they exist. Small work; once done, regenerate the C48
  binaries with a TM5-enabled LL source and `advdiffconv.toml` runs.
- **(b) Switch to `kind = "cmfmc"`** if you have a GEOS-style C48 binary
  with native CMFMC+DTRAIN sections. No C48 GEOS-IT binary exists today;
  would need either a C180→C48 coarsen or C48 preprocessing from source.

With a TM5-less plain C48 binary, `kind = "tm5"` will fail loudly at
the first window load (missing sections detected by the operator).

## How run_cs_driven.jl wires physics

The TOML → operator mapping is:

```toml
[advection]  scheme = "upwind" | "slopes" | "ppm"  (+ ppm_order for ppm)
[diffusion]  kind   = "none" | "constant"          (+ value: m²/s for constant)
[convection] kind   = "none" | "tm5" | "cmfmc"
```

Under the hood:

```julia
scheme     = UpwindScheme() | SlopesScheme() | PPMScheme(order=...)
diffusion  = NoDiffusion()
              | ImplicitVerticalDiffusion(kz_field = CubedSphereField(
                    ntuple(_ -> ConstantField{FT,3}(value), 6)))
convection = NoConvection() | TM5Convection() | CMFMCConvection()
```

All three are attached at model construction:

```julia
model = TransportModel(state, fluxes, grid, scheme;
                        diffusion  = diffusion,
                        convection = convection)
```

and `DrivenSimulation` auto-threads the per-window convection forcing
from the binary into the operator each window.

## Remaining gaps (not blocked by the three configs above)

| Gap | Notes |
|---|---|
| Profile/precomputed/derived Kz from TOML | `ImplicitVerticalDiffusion` supports them; need per-kind TOML schema + builder dispatch. Constant Kz covers today's needs. |
| `[tracers.*.surface_flux]` (emissions) for CS | `DrivenSimulation` supports `surface_sources=(SurfaceFluxSource(...), ...)` kwarg, but `run_cs_driven.jl` doesn't yet build those from TOML — fossil tracer stays at zero. |
| File-based CS initial conditions (`kind = "file"`) | The LL runner has `build_initial_mixing_ratio` with bilinear remap; a CS analog would need cube-sphere regridding of the IC. Plain `uniform` + `catrine_co2` (flat 411 ppm) work today. |
| TM5 section carry-through in LL→CS regrid | Blocker for `advdiffconv.toml` specifically; see Prerequisite #2 above. |
