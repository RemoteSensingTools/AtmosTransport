> **Archived research note.** This dated investigation is preserved for
> scientific provenance. Paths, run status, and implementation details may
> have changed; use the maintained manual for current workflows.
# Catrine Dec 2021 validation protocol

Side-by-side validation of two preprocessing paths against the
GEOS-Chem Catrine snapshots at
`~/data/AtmosTransport/catrine-geoschem-runs/`. The reference is C180 × L72
cubed-sphere, instantaneous output every 3 h.

## Reference data layout

`GEOSChem.CATRINE_inst.YYYYMMDD_HHMMz.nc4` files carry the canonical
`Met_*` fields:

| Variable     | Shape                     | Units | Notes                       |
|--------------|---------------------------|-------|-----------------------------|
| `Met_PSC2WET`| `(180, 180, 6, 1)`        | hPa   | Moist surface pressure      |
| `Met_PSC2DRY`| `(180, 180, 6, 1)`        | hPa   | Dry surface pressure        |
| `Met_AD`     | `(180, 180, 6, 72, 1)`    | kg    | Dry air mass per cell-layer |
| `Met_PMIDDRY`| `(180, 180, 6, 72, 1)`    | hPa   | Mid-layer dry pressure      |
| `Met_T`      | `(180, 180, 6, 72, 1)`    | K     | Temperature                 |
| `Met_U`      | `(180, 180, 6, 72, 1)`    | m/s   | Eastward wind               |
| `Met_V`      | `(180, 180, 6, 72, 1)`    | m/s   | Northward wind              |
| `Met_BXHEIGHT`| `(180, 180, 6, 72, 1)`   | m     | Box height                  |
| `Met_PBLH`   | `(180, 180, 6, 1)`        | m     | PBL depth                   |
| `Met_AREAM2` | `(180, 180, 6, 1)`        | m²    | Cell area (constant)        |

## Two production paths under validation

### Path 1 — GEOS-IT C180 native (passthrough)

`config/preprocessing/geosit_c180_dec2021_catrine_f32.toml` reads
`raw_catrine/YYYYMMDD/GEOSIT.*.C180.nc` directly, no horizontal regrid
needed (source mesh == target mesh). Produces a full v4 transport binary
with PBL + CMFMC + DTRAIN + GCHP VDIFF (U/V/T/Q) payload sections.

**Status:** running detached on this workstation (PID 2530910). Smoke
2021-12-01 succeeded: replay rel error 1.56e-6, positivity gate max 0.949.

### Path 2 — ERA5 N320 → C180 (this branch)

`config/preprocessing/era5_n320_to_c180_dec2021_catrine_f32.toml` reads
the native ERA5 GRIB streams, synthesises spectral T / U / V / LNSP,
reads Q + UDMF / DDMF / UDRF / DDRF on the reduced_gg N320 mesh, derives
dry-basis layer mass on the source mesh, and conservatively regrids to
the C180 cubed-sphere target.

**Status (after breakpoint F):** the per-window pipeline produces
`pipeline.c180_fields` (PS, U, V, T, Q regridded to C180) plus
`pipeline.window_fields`, `pipeline.dry_fields`, and
`pipeline.convection_fields` on the N320 source mesh. Binary writing
(mass-flux reconstruction, panel-local wind rotation, Poisson balance)
is a follow-on commit.

## Comparison protocol

### Full-physics smoke run vs GEOS-Chem

For the C180 GEOS-IT full-physics smoke run, use:

```bash
julia --project=. scripts/run_transport.jl \
  config/runs/catrine_geosit_c180_v4_fullphys_dec2021_smoke3d.toml

julia --project=. scripts/diagnostics/compare_at_vs_geoschem_c180.jl \
  --at  ~/data/AtmosTransport/output/catrine_geosit_c180_v4_fullphys_gchp_dec2021_smoke3d.nc \
  --gc  ~/data/AtmosTransport/catrine-geoschem-runs \
  --out ~/data/AtmosTransport/output/catrine_geosit_c180_v4_fullphys_gchp_dec2021_smoke3d_metrics.nc
```

The diagnostics compare dry mole fraction (`mol mol-1 dry`) for
`co2_natural`, `co2_fossil`, `sf6`, and `rn222`. AtmosTransport snapshots
store levels top-to-surface; GEOS-Chem CATRINE files store L72
surface-to-top, so the comparison script takes the common surface-aligned
levels and reverses the GEOS-Chem vertical axis before computing metrics.

The runtime state stores dry mole fraction times dry-air mass. Physical
surface inventories are loaded and checked as kg species/s, then converted to
the model storage basis before injection:

```text
model_storage_rate = species_mass_rate * dry_air_molar_mass / species_molar_mass
```

This conversion is required for the per-layer snapshot fields to remain dry
mole fractions when `tracer_storage / dry_air_mass` is written.

Use recomputed species mass for column/global burden checks:

```text
species_mass = sum(dry_vmr * dry_air_mass * species_molar_mass / dry_air_molar_mass)
```

Do not use GEOS-Chem `ColumnMass_FossilCO2` as the fossil CO2 truth field for
this comparison. In the Dec 2021 CATRINE output it has a constant semantic
bias relative to the profile-derived mass from `SpeciesConcVV_FossilCO2` and
`Met_AD`.

Flux totals should be compared on the same calendar basis as the source
inventory. GridFED fossil CO2 monthly fields are `kgCO2/month/m2`; December
2021 must be divided by `31 * 86400` seconds, not an average month length.
With the CATRINE C180 areas this gives about `1.2294e6 kg/s`, matching the
embedded GEOS-Chem fossil CO2 surface-flux total.

The map/curtain animation can be regenerated with:

```bash
python scripts/visualization/animate_catrine_map_curtains.py \
  --at ~/data/AtmosTransport/output/catrine_geosit_c180_v4_fullphys_gchp_dec2021_smoke3d.nc \
  --gc ~/data/AtmosTransport/catrine-geoschem-runs \
  --species co2_fossil
```

It uses a Robinson world map with C180 cell corners, three continuous
longitude-pressure curtains at 40 N, the equator, and 40 S, linear pressure
with high pressure at the bottom, and symlog concentration colors. Defaults
are 0-8 ppm for the column maps and 0-40 ppm for the curtains.

### Snapshot comparison (today)

For every 3-hourly Catrine snapshot in Dec 2021:

1. Read the Catrine NetCDF for `(date, hour)`.
2. Run `process_era5_n320_window!` for the same `(date, hour)`.
3. Compare on the shared C180 horizontal grid:

   | Field      | Source path 1 (GEOS-IT)              | Source path 2 (ERA5)                |
   |------------|--------------------------------------|-------------------------------------|
   | PS (Pa)    | binary inspector `ps` slice          | `pipeline.c180_fields.ps[p]`        |
   | T (K)      | binary inspector `t` slice           | `pipeline.c180_fields.t[p]`         |
   | U (m/s)    | binary inspector `u` slice           | `pipeline.c180_fields.u[p]`         |
   | V (m/s)    | binary inspector `v` slice           | `pipeline.c180_fields.v[p]`         |
   | Q (kg/kg)  | binary inspector `qv` slice          | `pipeline.c180_fields.qv[p]`        |
   | PS_dry (Pa)| derive from binary's PS, QV          | derive from C180 PS + Q (follow-on) |
   | m_dry (kg) | binary inspector `m` slice           | derive from C180 DELP × area / g    |

4. Statistics per panel: mean, RMS, min/max of the
   `pipeline_value − catrine_value` residual. Acceptance bands:
   - PS: |mean error| < 100 Pa, RMS < 200 Pa
   - T (mid-troposphere): |mean error| < 0.5 K, RMS < 1.5 K
   - U, V: |mean error| < 0.3 m/s, RMS < 2 m/s
   - PS_dry / PS_total ≈ 0.985-0.995 (water vapor mass fraction)

The script
`scripts/diagnostics/compare_era5_n320_pipeline_vs_catrine.jl` runs the
per-window slice of this protocol against the actual N320 archive.

### Vertical merge

Catrine is L72; the ERA5 pipeline is L137 native. Use
`MergeAbovePressure(0.25 hPa)` (the GEOS L72 cap convention) plus the
existing `apply_vertical!` plumbing to bring the ERA5 pipeline output to
L72 for full-column comparison. The merged-binary path is the natural
follow-on once the C180 mass-flux reconstruction lands.

### Outstanding follow-ons before a full Dec 2021 run

1. Mass-flux reconstruction on the C180 cubed sphere from regridded
   cell-center U / V (panel-local wind rotation + Arakawa-C face flux
   reconstruction). Reuses the bulk of
   `src/Preprocessing/transport_binary/cubed_sphere_regrid.jl`.
2. Re-derive dry mass on C180 from regridded PS + Q so
   `Σ_k DELP_dry == PS_dry` holds to roundoff on the target mesh.
3. Poisson balance of the reconstructed C180 fluxes + `cm` diagnosis.
4. v4 transport-binary writer for the ERA5-N320 source — slot the
   pipeline output into the existing
   `open_streaming_cs_transport_binary` / `write_streaming_cs_window!`
   path, with the convection conversion (UDMF/DDMF/UDRF/DDRF → CMFMC +
   DTRAIN-equivalent or TM5 entu/entd) wired through.
5. Process-day driver dispatching on
   `(::ERA5N320Settings, ::CubedSphereTargetGeometry)` so the unified
   CLI in `scripts/preprocessing/preprocess_transport_binary.jl` picks
   up the new config without extra plumbing.

   **Note:** this requires either implementing the generic
   `read_window!(::RawWindow, ::ERA5N320Settings, ::ERA5GRIBDayHandles, …)`
   for `AbstractMetReader` compatibility, or routing the N320 pipeline
   through a dedicated `process_day` overload that bypasses
   `AbstractMetReader` and consumes `ERA5N320ToC180Pipeline` directly.
   The pipeline-level API on this branch deliberately stays outside the
   `read_window!` contract — it produces structured per-window output on
   both the source and target meshes rather than the single-grid
   `RawWindow` shape, so a thin adapter is the cleanest wiring.
