# Plan 41 — Unified Transport-Binary Preprocessor

## Status

Drafted 2026-05-15 (carried over from the CS-contract-round-3 session on
2026-05-14).

Executor state, 2026-05-15:
- P0a / P0b shipped: typed met-reader and vertical-transform surfaces.
- P1 shipped: typed Axis-3 contract surfaces for CS / LL / RG, with
  lazy contract-owned scratch.
- P2a shipped in the current Codex working tree: LL, RG, CS spectral,
  and CS GEOS production paths construct typed contracts and update the
  contract accumulator from production windows. LL/RG now receive the
  positivity policy kwargs passed by `entrypoint.jl`. `cubed_sphere_regrid.jl`
  remains out of scope for Plan 41 per foot-gun (F) / Plan 42.
- P2b shipped in the current Codex working tree: additive
  `ReadyWindow{G, FT}`, `PreprocessorRunCache{G, FT}`, and bare
  workspace/readiness generics (`allocate_window_workspace`,
  `ingest_window!`, `drain_ready_windows!`, `flush_final_windows!`).
- P2c shipped: concrete LL spectral, RG spectral, CS spectral, and
  GEOS-native CS window workspaces. Spectral entrypoint now owns a
  run-level `PreprocessorRunCache`; RG reuses compressed Laplacians
  across days and CS spectral reuses LL→CS regridders across days.
- P3a shipped in the current Codex working tree: LL, RG, and CS binary
  writer adapters now implement the typed `AbstractBinaryWriter{G, FT, Basis}`
  surface plus close/promote/quarantine hooks. Production call sites are
  unchanged; focused tests pin topology-dispatch mismatches as `MethodError`.
- P3b shipped in the current Codex working tree: additive
  `UnifiedPreprocessorDay` + `run_unified_preprocessor_day!` driver shell,
  with migration hooks for window count, ingest, drain, flush, and post-write
  advancement. It normalizes both future raw `ReadyWindow`s and current
  preverified `(ready, contract)` events.
- P3c shipped in the current Codex working tree: GEOS-native CS has the first
  real production opt-in (`unified_driver = true`). The default legacy loop is
  unchanged, and the synthetic GEOS passthrough test now byte-compares legacy
  vs unified outputs.
- P3d shipped in the current Codex working tree: TOML native-source entrypoint
  reads `[preprocessor].unified = true` and threads it only to supported
  native GEOS -> cubed-sphere runs. Default config behavior remains legacy.
- P3e shipped in the current Codex working tree: ERA5 spectral -> reduced
  Gaussian has a `unified_driver = true` opt-in and the spectral TOML
  entrypoint threads `[preprocessor].unified = true` only for RG targets.
  A fake decoded spectral-cache fixture byte-compares legacy vs unified RG
  binaries.
- P3f shipped in the current Codex working tree: ERA5 spectral -> cubed
  sphere has a `unified_driver = true` opt-in, the spectral TOML entrypoint
  now allows RG and CS targets, and a fake decoded spectral-cache fixture
  byte-compares legacy vs unified CS binaries. LL remains last because its
  fixed header needs post-flush pressure-offset metadata before the writer
  adapter can open safely.
- P3g shipped in the current Codex working tree: ERA5 spectral -> lat-lon has
  a `unified_driver = true` opt-in using a deferred-header writer that opens
  only after post-flush pressure-offset metadata exists. The full-day LL
  contract remains owned by `flush_final_windows!`, so the driver receives
  preverified ready events rather than replaying the gate a second time.
- P3h shipped in the current Codex working tree: added
  `scripts/diagnostics/compare_preprocessors.jl` for the
  required side-by-side legacy vs unified bakes. It runs one config into
  separate output directories, compares file sets, then reports exact byte
  equality or header-normalized equality with payload byte equality (needed
  for LL's volatile `creation_time` header). P3j made this diagnostic
  chunk-stream large binaries instead of reading whole files into memory, so
  C180 native GEOS comparisons are feasible. P3k added
  `--warn-only-positivity` so known positivity-gate policy investigations can
  still compare legacy vs unified bytes without editing the source TOML.
- P3i in progress: real side-by-side bakes. `era5_ll72x37_advresln_dec2021_f32.toml`
  for 2021-12-01 passes with header-normalized equality and identical payload
  bytes (`creation_time` is the only intentionally ignored header field).
  `era5_cs_c24_transport_binary_f32.toml` for 2021-12-01 passes exact
  byte-for-byte equality (`58136576` bytes).
- RG real smoke blocker: `era5_synthetic_rgN24_transport_binary_v2.toml`
  for 2021-12-01 fails before unified comparison because the legacy path trips
  the write-time RG replay gate at window 1 (`rel=3.78e-3 > 1e-10` at cell
  `(1051, 28)`). Treat as a pre-existing config/data contract failure rather
  than a Plan 41 unified-driver regression.
  Current code audit points at the RG mass-closure null space: the compressed
  face-Laplacian can only remove mean-zero per-level divergence, while the
  vertical `cm` closure forces the bottom interface back to zero by
  redistributing the remaining column residual. If that residual is not already
  negligible, the stored fluxes are surface-closed but no longer replay to the
  raw next-window mass. Do **not** weaken the replay gate to pass this config;
  either make RG mass endpoints chained/replay-consistent or regenerate with a
  source/mass-fix setup whose residual is below the replay tolerance.
- Native GEOS strict-gate blocker: `geosit_c180_native_dec2021_f32.toml` for
  2021-12-01 fails before unified comparison because the legacy path trips the
  CS substep-positivity gate at window 24 (`outgoing/m=1.046 > 0.95`) in the
  thin upper layer at cell `(4, 53, 24, 61)`. This is the same gate-policy
  issue already parked for the C180 L72 mesospheric regen.
  With `--warn-only-positivity`, legacy and unified outputs compare exact
  byte-for-byte for the 2021-12-01 F32 binary (`9551547392` bytes); replay
  stayed active and matched worst rel `2.63e-7` / abs `5.20e5` at win 13.
- GEOS gate-policy decision: mirror the full-archive C180 regen config in the
  one-day smoke config. Replay remains mandatory; substep positivity is
  log-only (`require_substep_positivity = false`) until GEOS-native vertical
  merging can remove the thin-L72-mesosphere false blocker.
- RG blocker decision: keep replay mandatory and carry the real N24 dry config
  as an inadmissible legacy-input blocker, not a unified-driver blocker. The
  fake decoded spectral-cache RG fixture remains the byte-parity cutover test
  until a follow-up fixes RG mass closure for real dry ERA5.
- Remaining P3/P4 work: remove legacy opt-in scaffolding in small topology
  commits, preserving the fake-fixture byte-parity tests and the LL/CS/GEOS
  real-bake evidence above.
- P4a shipped: TOML entrypoint defaults to the unified driver.
- P4b in progress: remove the TOML-level unified/legacy opt-out entirely.
  Direct `process_day(...; unified_driver = false)` remains temporarily for
  parity tests and local bisection, but users no longer select the legacy loop
  from config.
- P4c shipped: removed the RG spectral hand-written sliding-window writer.
- P4d in progress: remove the GEOS-native CS inline legacy loop; the outer
  dispatch now validates inputs and calls `_process_day_geos_cs_unified`.
- P4e in progress: remove the CS spectral inline sliding-window loop; the
  `process_day` body now uses `run_unified_preprocessor_day!` directly.

**Read [DESIGN.md](DESIGN.md) first** for the typed three-axis rationale,
the anti-pattern audit with file:line citations, and the foot-gun
closure table. This NOTES.md is the executor punch-list; DESIGN.md is
the design contract.

Triggered by user observations:
- 2026-05-15 morning: "binary preprocessor uses very different paths
  for the different variations of met data and target grids; this
  should be unified with multi-dispatch."
- 2026-05-15 afternoon: "it seems haphazard and dangerous right now."
- 2026-05-15 afternoon: "layer merging should be part of the path for
  ALL pathways, so doing this now will fix two problems in one go."

Layer merging is **in scope** (Axis 2: `AbstractVerticalTransform`),
not "future fix". The 2026-05-14 thin-mesospheric-layer regen issue
becomes a config choice (`merge_layers_thinner_than`,
`merge_above_pressure`, or audited `merge_by_index`) once Axis-2 lands.

## Problem statement

The transport-binary preprocessor currently spreads four near-identical
per-window driving loops across four sibling files in
`src/Preprocessing/transport_binary/` (plus one more entry point in
`src/Preprocessing/reduced_transport_helpers.jl`), branched by source-
kind in `entrypoint.jl`, and only the cubed-sphere paths carry a
per-substep positivity contract.

Concretely, the surface looks like this today:

| File | `process_day` signature dispatches on | Source assumption |
|---|---|---|
| `transport_binary/latlon_spectral.jl:10` | `grid::LatLonTargetGeometry` | ERA5 spectral |
| `transport_binary/cubed_sphere_spectral.jl:10` | `grid::CubedSphereTargetGeometry` | ERA5 spectral |
| `transport_binary/cubed_sphere_geos.jl:377` | `grid::CubedSphereTargetGeometry, settings::AbstractGEOSSettings` | GEOS native NetCDF |
| `Preprocessing/reduced_transport_helpers.jl:1250` | `grid::ReducedGaussianTargetGeometry` | ERA5 spectral |
| `transport_binary/cubed_sphere_regrid.jl` | (not `process_day`) | LL binary → CS |

Top-level driver in `transport_binary/entrypoint.jl` is split by
**source-kind, not topology**:

- `_process_day_native` (line 116) handles typed `AbstractMetSettings`
  factories (currently only GEOS-IT). Reads `[source].toml`, builds
  settings, owns its own date loop.
- `_process_day_spectral` (line 205) handles legacy NamedTuple settings
  (ERA5 spectral). Owns its own date loop, different kwarg surface.

Per-window contract calls (replay + CS positivity) are made by each
`process_day` method *independently* — the call sites live inside four
sibling driving loops that all look like:

```julia
for win in 1:Nwindows
    ...build (m, am, bm, cm, m_next)...
    ...call write-time replay gate (LL: verify_window_continuity_ll;
                                    RG: verify_window_continuity_rg;
                                    CS: verify_cs_window_contract!)...
    ...write window to streaming binary...
    ...accumulate diagnostics...
end
...post-loop summary...
```

The d1f50b6/45b87f3/9b1ceda series consolidated *the contract calls
themselves* into `cubed_sphere_contracts.jl` (one source of truth for
the CS gates). But the **outer scaffolding** around those calls is
still copy-pasted four times.

### Concrete duplication smell

1. **Date-loop scaffolding duplicated.** Every method opens its own
   `for d in dates / for win in 1:Nwindows` and re-implements `[idx/N]
   $(d) → $(out_path)` logging.
2. **Writer open/close duplicated**, including the `try`-`finally` plus
   the per-window `bytes_per_window` accounting added in the round-2
   review.
3. **Positivity gate is CS-only.** LL and RG paths have no analogue of
   `verify_substep_positivity_*!`; whether this is a real gap or a
   "doesn't apply" depends on whether LL/RG fluxes can violate
   substep positivity. This plan answers that explicitly.
4. **`out_path` lifecycle inconsistent.** The CS-GEOS path stages to
   `.tmp` and atomically renames; LL/RG paths write in place.
5. **Kwarg surface differs.** GEOS path takes 9 explicit kwargs
   (`out_path`, `dt_met_seconds`, `FT`, `mass_basis`, `replay_tol`,
   `positivity_cfl_limit`, `require_substep_positivity`, `chain_mass`,
   `seed_m`); CS-spectral takes 3; LL/RG take 1.

## End-state architecture

Three orthogonal axes, all registered through multi-dispatch:

### Axis A — Source (where the per-window state comes from)

A new abstract type and trait surface:

```julia
abstract type AbstractMetReader{FT, MetSettings} end

# Open all handles for one calendar day. Returns a typed reader that
# carries everything the per-window step needs (file handles, vertical
# coordinate, chained-mass seed, …).
open_day(settings::MetSettings, date::Date,
         ::Type{FT}; next_day_handle::Bool, seed::Any) where {FT}
    → AbstractMetReader{FT, MetSettings}

# Number of write-windows produced per day. Currently 24 (1-hour
# windows). Per-source so a 3-hourly source would say 8.
windows_per_day(reader)::Int

# Read window `w` into preallocated source-shape buffers `dst`. `dst`
# is owned by the target workspace and reused across windows.
read_window!(dst, reader, w::Int) → nothing

# Cross-day carry (e.g., GEOS pressure-fixer endpoint) returned from
# the last window. Used by the next day's `open_day(... ; seed=...)`.
end_of_day_seed(reader) → Any

close_day!(reader)
```

Concrete readers:
- `ERA5SpectralReader` — owns spectral synth + LL regrid (collapses
  the `_process_day_spectral`-side `Workspace` bookkeeping).
- `GEOSNativeReader` — owns `open_geos_day` + CTM_A1/CTM_I1 + PF
  chaining (replaces the inner half of `cubed_sphere_geos.jl::process_day`).
- (future) `MERRANativeReader`, `GEOSFPSpectralReader`, etc.

### Axis B — Vertical transform (source levels → output levels)

`AbstractVerticalTransform` is the in-scope layer-merging axis. P0b
defines `IdentityVertical`, `MergeByIndex`, `MergeLayersThinnerThan`,
`MergeAbovePressure`, `LevelSelection`, and `PressureOverlap`; the
target workspace receives a `VerticalPlan{FT,T}` rather than a
duck-typed `NamedTuple`.

### Axis C — Target topology (how the window is shaped + contract)

Already-typed `AbstractTargetGeometry` is the dispatch key. We tighten
the surface so EVERY method below MUST exist for a new topology:

```julia
# Allocate per-day workspace buffers (haloed panel tuples for CS,
# face-indexed arrays for RG, structured arrays for LL).
allocate_window_workspace(grid::AbstractTargetGeometry, vertical, reader, FT;
                          cache = nothing)
    → NamedTuple

# Ingest one read window into the target workspace. This may queue the
# current window immediately (GEOS-native CS), queue the previous window
# after a one-window lookahead (CS/RG spectral), or defer until final
# endpoint/balance work is available (LL full-day path).
ingest_window!(workspace, reader, win_idx, vertical)

# Drain completed target windows ready for contract verification/write,
# then finish any last-window endpoint/fallback work at end of day.
drain_ready_windows!(workspace) → iterator of ReadyWindow
flush_final_windows!(workspace, reader, vertical) → iterator of ReadyWindow

# Single-window contract: replay + positivity (where applicable). Each
# topology has its own contract module and accumulator type.
verify_window_contract!(ready_window, grid, contract_state;
                        replay_tol, positivity_cfl_limit, halo_width=0)
    → (replay, positivity)  # positivity = (direction, ratio, ok)

# Streaming writer for the topology's binary format.
open_streaming_binary(grid, out_path, header) → writer
write_window!(writer, ready_window) → bytes
close_streaming_binary!(writer)
promote_streaming_binary!(writer)

# Initialize, update, and summarize the per-day contract accumulator.
init_contract_accumulator(grid)
update_contract_accumulator(grid, worst, diag, win)
summarize_contract_status(grid, worst; cfl_limit, steps_per_window,
                          require_substep_positivity, quarantine_path)
```

### Unified driver

One driver function in `transport_binary/driver.jl`, callable for any
`(source, target)` combination:

```julia
function process_day(cfg::AbstractDict;
                     day_override = nothing,
                     start_date = nothing,
                     end_date = nothing)
    FT       = _resolve_float_type(cfg)
    grid     = build_target_geometry(cfg["grid"], FT)
    settings = build_met_settings(cfg["source"], FT)
    vertical = build_vertical_setup(cfg, settings, grid, FT)
    dates    = _resolve_dates(cfg, settings; day_override, start_date, end_date)

    contract_kwargs = _resolve_contract_kwargs(cfg)
    run_cache = init_preprocessor_run_cache(grid, settings, vertical, FT)
    seed = nothing

    for (idx, d) in enumerate(dates)
        out_path = _output_path(cfg, settings, d, FT)
        reader = open_day(settings, d, FT;
                          next_day_handle = idx < length(dates) ||
                                            _has_next_day(settings, d),
                          seed = seed)
        try
            workspace = allocate_window_workspace(grid, vertical, reader, FT;
                                                  cache = run_cache)
            worst = init_contract_accumulator(grid)
            writer = open_streaming_binary(grid, out_path * ".tmp",
                                            header_from(reader, grid, vertical))
            local writer_closed = false
            try
                for w in 1:windows_per_day(reader)
                    read_window!(workspace.source, reader, w)
                    ingest_window!(workspace, reader, w, vertical)
                    for ready in drain_ready_windows!(workspace)
                        diag = verify_window_contract!(
                            ready, grid, workspace.contract; contract_kwargs...)
                        worst = update_contract_accumulator(grid, worst,
                                                             diag.positivity,
                                                             ready.index)
                        write_window!(writer, ready)
                    end
                end
                for ready in flush_final_windows!(workspace, reader, vertical)
                    diag = verify_window_contract!(
                        ready, grid, workspace.contract; contract_kwargs...)
                    worst = update_contract_accumulator(grid, worst,
                                                         diag.positivity,
                                                         ready.index)
                    write_window!(writer, ready)
                end
                close_streaming_binary!(writer); writer_closed = true
            finally
                writer_closed || close_streaming_binary!(writer)
            end
            summarize_contract_status(grid, worst;
                                       quarantine_path = out_path * ".tmp",
                                       contract_kwargs...)
            promote_streaming_binary!(writer)
            seed = end_of_day_seed(reader)
        finally
            close_day!(reader)
        end
    end
end
```

This is ~80 lines and replaces TWO `_process_day_*` entrypoints plus
FOUR `process_day(date, grid, settings, vertical)` methods.

## Commit-by-commit migration

Each commit ships green, with all tests passing. Migration is
**additive** until the last commit so we can compare new ↔ legacy
behavior diff-by-diff.

### P0 — Reader trait + Vertical-transform trait (no behavior change)

Two independent additions, can ship as a single commit OR two
adjacent commits.

**P0a** — Define `AbstractMetReader{FT, MetSettings, ChainPolicy}` +
the trait functions (`open_day`, `windows_per_day`, `read_window!`,
`end_of_day_seed`, `native_vertical`, `window_metadata`, `close_day!`)
in `src/Preprocessing/met_readers.jl`. Concrete `ERA5SpectralReader`
and `GEOSNativeReader` wrap the *existing* `open_geos_day` / spectral
machinery — no rewrites, just a thin typed façade. `ChainPolicy` is
either `NoChain` or `ChainedMass{T}` (`T` is the seed array type).

**P0b** — Define `AbstractVerticalTransform` + `VerticalPlan{FT}` in
`src/Preprocessing/vertical_transforms.jl`. The concrete transform
surface is:

- `IdentityVertical`
- `MergeByIndex` — explicit native-level groups (`57:58`, `59:61`, …)
  for audited production reruns.
- `MergeLayersThinnerThan` — typed form of today's
  `merge_thin_levels`, greedily coarsening adjacent layers until each
  output layer exceeds `min_thickness_Pa`.
- `MergeAbovePressure` — upper-atmosphere coarsening: merge layers
  whose midpoint pressure is lower than a configured pressure cutoff
  (e.g. mesosphere above 100 Pa), with a target minimum output
  thickness. This is the GEOS-IT L72 regen escape hatch when we do not
  need to resolve the mesosphere.
- `LevelSelection`
- `PressureOverlap`

`plan_vertical` and `apply_vertical!(_, _, plan, ::FieldKind)` are the
trait surface. The existing `merge_thin_levels`
(`vertical_coordinates.jl:11`) and `select_levels_echlevs` become
`MergeLayersThinnerThan` and `LevelSelection` implementations.

Definition of done:
- (P0a) A 1-day GEOS smoke `julia --project=. -e 'reader =
  open_day(geos_settings, Date("2021-12-02"), Float32); for w in
  1:windows_per_day(reader); read_window!(dst, reader, w); end'`
  produces the same `(m, am, bm, cm, m_next, …)` arrays as today's
  GEOS path, bit-for-bit. Add focused test `test_met_readers.jl`.
- (P0b) Round-trip: starting from the existing `build_vertical_setup`
  output, construct the equivalent `AbstractVerticalTransform` from
  config, plan it, and verify `merged_vc` / `merge_map` /
  `Nz_output` agree to bit-exactness. Add focused test
  `test_vertical_transforms.jl` covering all transforms plus the
  mandatory field-kind rules: `MassField`, `TracerMassField`,
  `MassFluxField`, `PressureFluxField`, `ConvectionInterfaceFlux`
  (`cmfmc`), `ConvectionTendencyField` (`dtrain`),
  `IntensiveCenterField`, and `SurfaceField`.

### P1 — Target dispatch tightening

Add the three missing topology contract surfaces:

- `latlon_contracts.jl` → add `verify_ll_window_contract!`,
  `init_ll_positivity_accumulator`, …, mirroring `cubed_sphere_contracts.jl`.
  **Decision needed during P1**: do LL fluxes have a substep-positivity
  contract? Probe by running a representative ERA5 day's stored
  (am, bm, cm) through a probe equivalent of `verify_substep_positivity_cs!`
  and see whether ratios exceed 0.95 anywhere. If yes, ship the gate;
  if no, ship a stub that always returns `ok = true` and document the
  invariant.
- `reduced_gaussian_contracts.jl` → ditto for the RG path.
- Hoist any other duplicated contract helpers into per-topology
  modules.

Definition of done: ship the additive typed surfaces and focused
contract tests without changing binary output. Add focused
`test_ll_preprocessor_contract.jl` and `test_rg_preprocessor_contract.jl`
mirroring `test_cs_preprocessor_contract.jl`.

P2a handles the production call-site cutover so P1 can remain
bit-exact/additive.

### P2a — Production contract wiring

Wire production preprocessors into the P1 contract lifecycle without
changing driver control flow:

- LL spectral accepts `positivity_cfl_limit` and
  `require_substep_positivity`, constructs `LatLonContract{FT}`, runs
  replay + positivity across balanced storage, then summarizes before
  writing.
- RG spectral accepts the same policy kwargs, constructs
  `ReducedGaussianContract{FT}`, runs boundary-stub -> replay ->
  positivity on the Float64 balanced work buffers before each write,
  and summarizes after writer close.
- CS spectral and CS GEOS replace the free accumulator helpers with
  `CubedSphereContract{FT}` plus `verify_window!` /
  `update_accumulator!` / `summarize_status!`.
- Keep `ATMOSTR_NO_WRITE_REPLAY_CHECK=1` as a replay-only diagnostic
  bypass; positivity still runs.
- Remove obvious per-window scratch allocation while touching RG:
  `balance_window!` receives reusable `dm_target_work`.

Definition of done: CS / LL / RG contract suites and adjacent P0/P1
regressions pass; no binary-format or math changes.

### P2b — Additive workspace/readiness skeleton

Add the typed nouns needed by the later driver cutover:

- `ReadyWindow{G, FT, P}` wraps a ready payload and forwards payload
  properties so existing contract methods accept it.
- `PreprocessorRunCache{G, FT}` provides a typed per-run cache for
  artifacts such as CS regridders and RG compressed Laplacians.
- Bare generic hooks: `allocate_window_workspace`, `reset_workspace!`,
  `ingest_window!`, `drain_ready_windows!`, `flush_final_windows!`.

Definition of done: skeleton exports are tested, and `ReadyWindow`
round-trips through at least one real contract method.

### P2c — Concrete workspaces + readiness methods

Pull every method's "allocate per-day arrays" block into
`allocate_window_workspace(grid, vertical, reader, FT; cache)` per
topology. Pull the "consume one source window and maybe make one or
more target windows ready" logic into `ingest_window!`,
`drain_ready_windows!`, and `flush_final_windows!` per topology.

Also add a `PreprocessorRunCache` hook for expensive run-invariant
objects. At minimum, the CS spectral LL→CS conservative regridder and
the RG compressed Laplacian should be build-once-per-run when source
geometry and target geometry are unchanged across days.

This is mechanical surgery; the math doesn't change. Definition of
done for P2c: every production topology has a concrete workspace and
the existing drivers call the workspace/readiness hooks for allocation,
ingest, and ready-window packaging. A multi-day CS spectral run reuses
the LL→CS regridder from `PreprocessorRunCache`; a multi-day RG run
reuses the compressed Laplacian. The old "drop each process_day to
<50 lines" target moves to P3, because that requires the unified
driver opt-in and writer abstraction rather than just workspace
extraction.

### P3 — Unified driver

P3a adds concrete writer adapters first. They wrap the current LL v4
full-day writer and the existing RG/CS `StreamingTransportBinaryWriter`
without changing byte layout or production call sites. This gives the
unified driver a typed writer surface before any driver code is moved.

P3b adds the driver shell without wiring it into production. It owns the
common lifecycle (ingest -> drain -> verify/update -> write -> close ->
summarize -> promote; close/quarantine on failure) and exposes small
migration hooks so existing workspace methods can be adapted one topology
at a time.

P3c wires the first production opt-in: GEOS-native CS gets
`process_day(...; unified_driver=true)`. The legacy path stays default.
The parity gate is byte equality between the legacy and unified binaries
on the synthetic GEOS passthrough fixture.

P3d exposes that opt-in at the TOML entrypoint as
`[preprocessor].unified = true` for native GEOS -> cubed-sphere only.

P3e wires the first spectral opt-in: ERA5 spectral -> reduced Gaussian gets
`process_day(...; unified_driver=true)`, and `[preprocessor].unified = true`
is accepted for spectral RG configs. The parity gate uses a tiny decoded
spectral-cache fixture and compares the legacy and unified binaries byte for
byte.

Add `transport_binary/driver.jl` with the ~80-line driver above.
**Don't** wire it in yet — `entrypoint.jl::process_day` still routes
through `_process_day_native` / `_process_day_spectral`.

Add a new opt-in code path: if `cfg["preprocessor"]["unified"] = true`,
call the new driver; otherwise route through legacy. Run side-by-side
on 1-day smoke configs for ERA5 spectral × LL, ERA5 spectral × CS,
ERA5 spectral × RG, GEOS native × CS, and compare binaries byte-for-byte.

Definition of done: the 4 side-by-side smokes produce bit-identical
binaries (or document any FP-rounding-tier difference). Adds
`scripts/diagnostics/compare_preprocessors.jl`.

### P4 — Cut over

Switch `entrypoint.jl::process_day` to call `driver.process_day`
unconditionally. Move the four old `process_day(date, grid, settings,
vertical)` methods + `_process_day_native` + `_process_day_spectral`
into `src_legacy/Preprocessing/`. Delete the unified-vs-legacy opt-in
flag.

Definition of done: every regression test that exercises preprocessing
passes. The 1-day smoke configs from P3 produce bit-identical binaries
to the pre-cut-over commit. `scripts/preprocessing/preprocess_transport_binary.jl`
needs no change (it already takes a single TOML).

### P5 — Add MERRA or GEOS-FP native reader (validation)

The proof-of-extensibility test: add one new source by implementing
`open_day` + `read_window!` + `windows_per_day` + `close_day!` only —
no new `process_day` method, no driver edits. If this commit is
~150 lines plus tests, the unification did its job.

## Notes for the executor

- `cubed_sphere_geos.jl` is the only file today with `process_day`
  dispatching on `settings::AbstractGEOSSettings`. That dispatch
  collapses naturally into `GEOSNativeReader` (settings axis lives
  on the reader, not on `process_day`).
- `cubed_sphere_regrid.jl::regrid_ll_binary_to_cs` is a **different
  pipeline** (LL binary → CS binary), not a `process_day` variant.
  Out of scope for this plan. After P4 it becomes a topology
  transformer that could optionally use the new reader surface for
  its LL input — leave that for Plan 42.
- The d1f50b6 / 45b87f3 / 9b1ceda work (CS contract + tests) is the
  template for what `latlon_contracts.jl` and `reduced_gaussian_contracts.jl`
  should look like under this plan. Mirror `test_cs_preprocessor_contract.jl`
  one-to-one in the new topology test files.
- The `merge_thin_levels` machinery
  (`Preprocessing/vertical_coordinates.jl:11`,
  `Preprocessing/configuration.jl:295`) is currently wired only into
  the ERA5 spectral `build_vertical_setup`. The unified driver should
  expose layer merging as a per-source option so GEOS-IT can opt in
  (relevant to the v4 thin-mesospheric-layer regen issue documented
  in [cs_contract_round3_regen_blocked_2026_05_14.md](../../../.claude/projects/-home-cfranken-code-gitHub-AtmosTransportModel/memory/cs_contract_round3_regen_blocked_2026_05_14.md)).
  This becomes natural in P2 because `build_vertical_setup` moves
  next to `allocate_window_workspace`.

## Hard constraints (must not break)

1. **Bit-exact binaries** for every existing TOML config through P4.
2. **Existing contract surface is the floor.** The CS round-3 gate
   semantics (round-2's `Inf`/`NaN` fix, round-3's
   `typemax(Int)` + `cfl_limit` validation, the
   `require_substep_positivity = false` escape hatch) stay verbatim.
3. **GEOS chained-mass seed plumbing.** The pressure-fixer endpoint
   handoff in `_process_day_native` is load-bearing; the
   `end_of_day_seed(reader)` / `seed=` round-trip must be tested with
   the existing 5-day chained-mass smoke before P4 lands.
4. **Boundary-day error handling.** The 2023-12-31 boundary failure
   in the previous regen (no next-day CTM_I1 endpoint) must still
   surface with a comprehensible error, not a generic "missing file".

## Out of scope

- `cubed_sphere_regrid.jl::regrid_ll_binary_to_cs` (Plan 42).
- TM5 convection preprocessing (`tm5_convection_conversion.jl`).
- Runtime I/O — that's Plan 40, already shipped.
- Adding new target topologies beyond LL / RG / CS.

**No longer out of scope (was, until 2026-05-15 afternoon):**
- Layer merging for GEOS-IT is **in scope** as Axis-2 of the typed
  design (see DESIGN.md). The 2026-05-14 thin-mesospheric-layer regen
  issue becomes a `[vertical].transform = "merge_layers_thinner_than"`,
  `"merge_above_pressure"`, or `"merge_by_index"` config choice once
  P0b ships.
