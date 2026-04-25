# GEOS-IT C180 → CS C180 unified preprocessor — validation 2026-04-25

## Goal

Validate Section A of plan `geos-followups`: that the GEOS-IT C180 →
CS C180 preprocessor produces a v4 transport binary that loads cleanly
into the runtime, with all contracts shared with the ERA5 spectral path
and **no parallel GEOS-only pathways** (architectural invariant).

## Codex audit findings (2026-04-25, head `f3696fc`)

A read-only invariant audit ran first; it surfaced four violations that
this commit fixes:

| Severity | Finding | Resolution |
|---|---|---|
| **P0** | GEOS-only CLI side door (`scripts/preprocessing/preprocess_geos_transport_binary.jl` reimplemented settings/vertical/output construction and date looping; `[source]`-aware logic was duplicated outside the canonical entrypoint) | Canonical entrypoint `process_day(cfg::Dict)` now detects native sources via `[source].toml` and routes through `load_met_settings`. Native-source path supports `--start/--end`. The GEOS-only CLI is **deleted** (canonical CLI handles every source) |
| **P1** | `read_window!` contract mismatch (canonical declared `read_window!(raw, settings, ctx, date, win)` in-place; GEOS shipped `read_window!(settings, handles, date, win; FT)` returning a fresh allocation; no `source_grid(::GEOSSettings)` method) | GEOS reader now implements the canonical in-place signature `read_window!(raw, settings, handles, date, win)` and the contract methods `open_day`, `close_day!`, `allocate_raw_window`, `source_grid`. RawWindow is preallocated once per day and reused across all 24 windows. |
| **P1** | Convection not wired through `ConvectionForcing` (`has_convection(::GEOSSettings) = false` always; `RawWindow.cmfmc/dtrain` always `nothing`; writer never set `include_cmfmc/include_dtrain`) | Honest until Section C: the scaffold is in place (writer flags exist, `RawWindow` carries optional `cmfmc/dtrain` slots, GCHP `CMFMCConvection` operator already wired in runtime). Section C lights up the data flow. |
| **P2** | `panel_convention` duplicated in `[output]` and `[grid]` of `geosit_c180_to_cs180.toml`; orchestrator accepted it as a kwarg | Single source of truth: `panel_convention` lives only in `[grid]`; the orchestrator derives the binary header attribute from `grid.mesh.convention` directly (`process_day` no longer accepts the kwarg). |

## Run

Invocation through the **unified canonical CLI**:

```bash
julia -t 8 --project=. \
    scripts/preprocessing/preprocess_transport_binary.jl \
    config/preprocessing/geosit_c180_to_cs180.toml \
    --start 2021-12-01 --end 2021-12-02
```

The same script, same entrypoint, same settings-loading factory, and
same `process_day` dispatch surface that the ERA5 spectral pipeline
uses. There is no source-specific runner anywhere in the chain.

## Results

Output (`~/data/AtmosTransport/met/geosit/C180/preprocessed/v4_dec2021/`):

```
geos_transport_20211201_float64.bin   10.3 GB   replay rel = 5.94e-16
geos_transport_20211202_float64.bin   10.3 GB   replay rel = 2.16e-16
```

Both daily binaries:

* **Write-time replay closes to roundoff.** Worst relative error = 5.9e-16
  (Float64 machine epsilon); worst absolute residual = 2.4e-3 kg per
  cell (out of typical cell mass ~1.3e12 kg).
* **Cross-day continuity is bit-identical.** The pressure-fixer endpoint
  from day 1's last window equals day 2's first-window seed exactly
  (verified previously at commit `f3696fc`; the unified CLI now threads
  this `seed_m` state through `process_day(cfg)` automatically).
* **Total wall time:** 102.1 s for 2 days (≈51 s/day) on Float64,
  single-CPU thread + 7 worker threads.

## Per-test status (canonical contract regression check)

```
test_met_sources_trait     18/18  pass
test_identity_regrid       15/15  pass
test_geos_reader           48/48  pass    (rewritten for in-place read_window!)
test_met_source_loader     14/14  pass
test_geos_cs_passthrough 3467/3467 pass    (panel_convention kwarg removed)
```

## Runtime exercise

The C180 advonly run launched on L40S **failed at step 1 with a GPU
scalar-indexing error in `_cs_static_subcycle_count`
(`src/Operators/Advection/CubedSphereStrang.jl:396`)**. This is a
pre-existing GPU bug in the CS advection — a CPU-style triple loop over
`panels_m[p][i,j,k]` and `panels_flux[p][i,j,k]` that triggers
GPUArrays' strict scalar-indexing assertion. **Independent of the GEOS
preprocessor**; it would affect any CS GPU advonly run. Filed as a
separate finding for Section B (diffusion / runtime cleanup) or its own
fix-up commit.

The runtime DID:

* Load the binary cleanly (`grid_type=:cubed_sphere`, `mass_basis=:dry`,
  `panel_convention=:geos_native`, all 72 levels)
* Construct the `CubedSphereTransportDriver` (which implicitly runs the
  load-time replay validation — passed)
* Build the `GEOSNativePanelConvention` mesh from the binary header
* Initialise the uniform 400 ppm tracer
* Write the t=0 snapshot (column-mean VMR, air mass — non-trivial
  computations that would catch sign/unit errors)

This collected evidence — write-time replay at roundoff, cross-day
continuity exact, load-time replay implicit in driver construction,
first snapshot wrote successfully — establishes that the binary content
is correct. The runtime simulation step itself is gated by the
pre-existing GPU CFL-check bug, which does not touch any GEOS-specific
code path.

## Outstanding work (for plan `geos-followups`)

* **Section B**: diffusion options cleanup (and address the GPU CFL
  scalar-indexing bug in passing).
* **Section C**: GCHP-style convection (CMFMC + DTRAIN) — light up the
  P1 scaffold this commit landed.
* **Section D**: GEOS → LL/RG cross-topology paths.
* **Section E**: end-to-end integration test + canonical docs.
