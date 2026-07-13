# Binary Preprocessing Architecture and Philosophy

**Status:** Living architecture note. This document describes how the
transport-binary preprocessing path is intended to fit together and why the
major contract boundaries exist. The byte-level file format is specified in
[`BINARY_FORMAT.md`](BINARY_FORMAT.md); this page is the higher-level design
and operating philosophy.

## Scope

Binary preprocessing converts source meteorology into transport-ready files
that the runtime can map, validate, and step without rebuilding expensive
source-specific artifacts. A transport binary is not just a cache of arrays. It
is a signed numerical contract between the preprocessor and the model runtime:

- what source product was ingested;
- what vertical transform was applied;
- what target grid and mass basis were written;
- how each met window is to be substepped;
- what replay, mass-continuity, boundary, and positivity gates passed.

The current production convention is one binary file per source day, containing
`nwindow` met windows. For hourly GEOS-native products this is normally 24
windows per file. The runtime schedule is per window, so the number of
substeps can vary inside a day.

## Design Principles

### Binaries are contracts, not opaque caches

The runtime should be able to reject an unsafe or obsolete file before the
first transport step. Header fields therefore carry format version, target
geometry, mass basis, vertical coordinate, payload manifest, timing semantics,
and the required substep schedule. If a field affects transport semantics, it
belongs in the header or in a typed payload section, not in a filename or a
comment.

The current transport-binary format version is `4`. Readers reject every
other version. Summary scalars such as `steps_per_window` remain only for
display; the runtime contract is the full
per-window schedule `steps_per_window_by_window`.

### Source, vertical transform, and target are orthogonal axes

Preprocessing used to grow by copying whole day loops for each combination of
source product and target grid. That made critical knobs drift across paths.
The unified direction is three typed axes:

- **Source:** how native meteorology is opened, read, timed, and chained.
- **Vertical transform:** how native levels become output levels.
- **Target:** how a window is regridded, verified, and written for a topology.

Adding a new source should not require a new positivity contract. Adding a new
vertical policy should not be possible for only one target by accident. Adding
a new target topology should force it to declare its own contract surface.

### Numerical gates run before data is promoted

The preprocessor writes through a staging file. Replay and positivity contracts
are evaluated before promotion. A failed hard gate should leave no final binary
that the runtime can accidentally consume. Warning-only gates may be useful for
diagnostics, but production products should be generated with the hard contract
enabled.

### Runtime stepping follows the binary

The runtime does not infer a safe timestep from a global user knob. It reads
the schedule produced by the preprocessor. That keeps C90, C180, C360, merged,
and unmerged products scalable without requiring users to know the safe
substep count for each grid and vertical coordinate. The binary is the source
of truth.

The schedule is an **advection/transport-block** schedule. A v4 binary
stores the number of substeps needed for the met-window flux palindrome to pass
the replay and positivity gates. It is not a request to run every physics
operator at that frequency. In the driven runtime, the transport block
(advection with midpoint diffusion and surface emissions) consumes each stored
substep; convection and chemistry run once at the met-window boundary with the
full window duration.

## Pipeline Overview

At a high level the intended pipeline is:

```text
raw met files
  -> AbstractMetReader
  -> source window buffers
  -> VerticalPlan
  -> AbstractWindowWorkspace
  -> ReadyWindow or PreverifiedWindow
  -> AbstractWindowContract
  -> AbstractBinaryWriter
  -> format-v4 transport binary
  -> runtime reader and driven simulation
```

The generic day loop is represented by `UnifiedPreprocessorDay`. It owns a
reader, workspace, contract, writer, and optional adapter context. The generic
lifecycle is:

1. Open or reset the source reader for a day.
2. Allocate or reset the target workspace.
3. Ingest each source/met window into the workspace.
4. Drain windows that became ready.
5. Verify each ready window through the typed contract.
6. Write verified windows through the typed writer.
7. Flush any final cross-day windows.
8. Patch final metadata, close the staging file, summarize gates, and promote.
9. Quarantine the staging file on any hard failure.

The names that define this surface are:

- `AbstractMetReader`
- `AbstractVerticalTransform` and `VerticalPlan`
- `AbstractWindowWorkspace`
- `ReadyWindow`
- `PreverifiedWindow`
- `AbstractWindowContract`
- `AbstractBinaryWriter`
- `PreprocessorRunCache`
- `run_unified_preprocessor_day!`

## Ownership Boundaries

The split between config, binary header, and runtime must stay strict. The
run TOML chooses a product family and the model physics. It should not restate
facts that are already determined by the binary. In particular, a binary-driven
run should read these from the first binary header:

- target topology and horizontal dimensions;
- vertical level count and hybrid coefficients;
- vertical transform metadata;
- mass basis;
- payload manifest and optional physics capabilities;
- met-window duration and per-window substep schedule;
- runtime substep contract and preprocessor contract string.

The run TOML may carry temporary forensic assertions while debugging a stale
directory, but normal production should not duplicate header facts such as
`expected_nlevel`, `required_preprocessor_contract`, or
`require_adaptive_substeps`. The level count, preprocessor contract, and
adaptive substep schedule are consequences of the selected preprocessing
product and must come from the binary.

This is also the generalization rule for new sources and grids: the runtime
loads one typed binary product and then dispatches from header-owned facts.
Adding a new source, vertical policy, or target topology should add a concrete
reader/workspace/contract/writer implementation, not another parallel run
configuration schema.

## Axis 1: Source Readers

A source reader owns the source-product contract: file layout, native
coordinates, native timing, and any cross-day state. The intended abstract
shape is:

```julia
AbstractMetReader{FT, MetSettings, ChainPolicy}
```

`ChainPolicy` distinguishes readers that need cross-day mass carry from those
that do not. A chained native source should make the next-day seed explicit in
the reader type instead of hiding it in a loose `NamedTuple`.

Examples:

- `GEOSNativeReader` reads GEOS-native files and carries the state needed for
  pressure/mass chaining across day boundaries.
- `ERA5SpectralReader` represents the spectral ERA5 path; parts of that path
  are still being collapsed into the same unified lifecycle.

Source readers should answer:

- how many windows exist for the day;
- what the native vertical coordinate is;
- how air mass and fluxes are sampled in time;
- what source buffers are needed;
- whether cross-day state must be returned.

## Axis 2: Vertical Transforms

Vertical transformation is a first-class numerical decision. It should be
visible in config, encoded in the binary header, and implemented uniformly for
all source and target paths.

Current transform policies:

- `IdentityVertical`: preserve the native level grid.
- `MergeByIndex`: merge explicit native-level groups.
- `MergeLayersThinnerThan`: greedily merge adjacent layers thinner than a
  pressure-thickness threshold.
- `MergeAbovePressure`: preserve lower levels and coarsen layers above an
  isobar.
- `LevelSelection`: select native half-levels with the existing `echlevs`
  style.
- `PressureOverlap`: map onto an independent target hybrid coordinate by
  pressure-thickness overlap.

For GEOS C180 L72, the upper mesosphere can have very thin layers with little
column mass but large per-substep CFL pressure. `MergeAbovePressure` is the
preferred explicit policy for that case. For example, merging layers above
`0.25 hPa` preserves the lower atmosphere while removing the tiny upper-layer
constraint from the runtime schedule.

The important rule is that merging is never a hidden rescue path. It is a
declared transform with a planned output coordinate and field-specific
reduction rules:

- mass and layer-integrated fluxes are summed;
- intensive center fields are mass-weighted;
- interface fluxes select or combine interfaces according to the transform;
- surface fields pass through unchanged.

## Axis 3: Target Workspaces, Contracts, and Writers

The target axis is where topology-specific details belong. A lat-lon grid,
cubed sphere, and reduced Gaussian mesh should share lifecycle concepts, but
they should not pretend to have identical horizontal array shapes.

The typed surface is:

```julia
AbstractWindowWorkspace{G, FT}
AbstractWindowContract{G, FT}
AbstractBinaryWriter{G, FT, Basis}
```

`G` is the target geometry type. `FT` is the on-disk float type. `Basis` is the
mass basis tag, reusing the same `DryBasis` and `MoistBasis` nominals as the
runtime. That avoids a writer/runtime mass-basis mismatch becoming a late
header surprise.

Responsibilities:

- A workspace owns reusable per-day arrays and regridder state.
- A contract owns replay tolerance, positivity limits, boundary checks, and
  worst-window accumulators.
- A writer owns the staging path, fixed payload ordering, final header patch,
  close, promote, and quarantine behavior.

## Transport-Binary Version 4

Format v4 is the sole current transport-binary contract. It carries the
fields needed to make per-window substeps and replay scaling explicit:

- `steps_per_window_by_window`
- `time_step_schedule`
- `poisson_balance_target_scale_by_window`
- `poisson_balance_target_semantics`

`steps_per_window` and `poisson_balance_target_scale` summarize the maximum
schedule entry and its corresponding scale for inspection. They do not drive
runtime stepping.

A v4 reader validates at load time:

- the magic and exact format version;
- the schedule length and positivity;
- `steps_per_window == maximum(steps_per_window_by_window)`;
- the declared `time_step_schedule`;
- the per-window Poisson target scale;
- expected dimensions and payload sections;
- target grid and vertical metadata;
- mass basis and required preprocessor contract, when requested by run config.

The header is authoritative; filenames and directory names do not define the
binary contract.

## Adaptive Per-Window Transport Substeps

Per-window substeps are selected during preprocessing from a CFL/positivity
contract. The schedule is an integer count for each met window. There is no
requirement that the count be a power of two; the only hard requirement is
that the met-window duration is divided into an integer number of transport
substeps.

The intended policy is:

- compute the minimum safe `n_sub` for each met window from the verified
  positivity/CFL gate;
- apply a safety target below the hard limit;
- store the resulting schedule in the v4 header;
- scale stored substep flux amounts consistently with that schedule;
- declare `runtime_substep_contract = "binary_schedule"` when the binary has
  passed the write-time CFL/positivity gates;
- make runtime, diagnostics, replay, and adjoints read the schedule from the
  binary instead of re-piloting subcycles dynamically.

At runtime, `runtime_substep_contract = "binary_schedule"` also means the CS
transport kernel does not re-pilot an additional internal CFL subcycle. The
binary schedule already owns that decision. Convection and chemistry are then
applied once per met window, consistent with TM-style/GCHP-style physics
cadences where convection fields are sampled on the met/physics interval and
the convection operator owns any internal stability subcycling.

For GEOS C180, the current target is the runtime positivity limit
(`substep_cfl_target = 0.95`). Merged upper layers should reduce the required
schedule. A coarser horizontal grid should naturally produce smaller `n_sub`
values without the user changing a separate timestep knob.

### CFD and adjoint caveats

Adaptive transport substeps are a stability and positivity policy, not a proof
of equal accuracy for all schedules. The current binary-driven runtime keeps
convection and chemistry at the met-window cadence, so those operators are not
multiplied by an hour's transport substep count. Transport-center operators
such as diffusion and surface emissions still live inside the transport block.
If advection limiters, operator splitting, diffusion, emissions, or diagnostics
are nonlinear in the timestep, a run with per-window optimized transport
substeps is not expected to be bit-identical to a run that uses the global
finest substep count everywhere.

The contract we want is stronger and more useful:

- for a fixed binary, forward and adjoint runs replay the same schedule;
- replay and positivity gates are evaluated with the same per-window schedule;
- if exact parity is required for a study, freeze and reuse the same binary;
- any adjoint path that assumes a single global substep count must reject v4
  variable-schedule binaries until it is schedule-aware.

## Gate Philosophy

The contract layer should be conservative. A file that passes should be safe
to run without manual inspection. A file that fails should fail before
promotion or at runtime load.

Current gate families:

- **Replay/mass continuity:** window fluxes reproduce the expected mass change
  within tolerance.
- **Substep positivity:** outgoing mass from any cell stays within the chosen
  CFL/positivity limit for the stored schedule.
- **Reduced-Gaussian boundary stubs:** nonzero flux into discarded boundary
  stubs is a writer bug and should fail before downstream replay errors.
- **Header contract checks:** required schedule, Poisson scaling, mass basis,
  dimensions, payload manifest, and format version.
- **Run-config checks:** optional requirements such as expected contract
  string or required adaptive schedule.

Warnings are for exploration. Production binaries should pass hard gates.

## Performance Philosophy

The preprocessor is allowed to spend work that prevents repeated runtime work,
but it should not rebuild expensive artifacts inside inner loops. The long-term
shape is:

- build source readers once per day;
- build target regridders, compressed Laplacians, and geometry caches once per
  run where possible;
- keep window workspaces allocated and reused;
- keep contract scratch arrays lazy and reused by identity;
- stream binary payloads to staging files instead of retaining full days in
  memory;
- promote only after gates summarize cleanly;
- keep the runtime reader mmap-friendly and branch-light.

`PreprocessorRunCache` exists for artifacts that should live across days, such
as LL-to-CS regridders or reduced-Gaussian compressed operators. Per-day
workspace arrays belong in `AbstractWindowWorkspace` implementations. Per-gate
scratch belongs in the concrete `AbstractWindowContract`.

## Operational Rules

Recommended operating discipline:

- Do not mix old and new binary products in the same directory.
- Delete or quarantine obsolete files before regenerating a product family.
- Treat the header as the source of truth for version, level count, schedule,
  vertical transform, and contract string.
- Give product directories names that encode source, grid, vertical policy,
  float type, and adaptive schedule.
- Use run configs to require the expected binary contract and adaptive schedule.
- Prefer regeneration over compatibility shims when a contract becomes
  ambiguous.

For example, a GEOS C180 Float32 product with upper-layer merging and adaptive
substeps should be distinguishable from an older C180 L72 fixed-step product
by both directory name and header fields. The runtime should reject the older
product even if the filename pattern still matches.

## Extension Checklist

When adding a new source, target, or vertical policy, update the typed surface
first and then the production path:

1. Add or extend the source settings type.
2. Implement an `AbstractMetReader` with explicit timing and chain policy.
3. Select or implement an `AbstractVerticalTransform`.
4. Materialize a `VerticalPlan` and field-kind reductions.
5. Implement the target `AbstractWindowWorkspace`.
6. Implement or reuse the target `AbstractWindowContract`.
7. Implement the `AbstractBinaryWriter` with a fixed payload manifest.
8. Add format/header fields for any new transport semantics.
9. Add replay, positivity, boundary, and header tests.
10. Add a small smoke config and an inspection command.
11. Update this document and the byte-level format spec if semantics changed.

If a new knob must be propagated through multiple topology-specific functions,
that is a design smell. It should probably be a field on a settings, reader,
contract, workspace, or writer type.

## Known Open Work

The architecture is intentionally ahead of some migration details. Known
follow-up areas:

- finish collapsing spectral source paths into the same reader/workspace
  lifecycle;
- keep pruning legacy day-loop code once the unified driver owns each
  production path;
- bring the cubed-sphere regridder pathway under the same contract surface;
- make all adjoint and footprint paths explicitly schedule-aware;
- add compact manifest/index files for generated binary directories;
- keep docs and run configs aligned with the exact format version the runtime
  accepts.

## Cross References

- [`BINARY_FORMAT.md`](BINARY_FORMAT.md): byte-level transport-binary family
  spec.
- [`PREPROCESSING_PHILOSOPHY.md`](PREPROCESSING_PHILOSOPHY.md): older
  preprocessing notes that this document is intended to supersede over time.
- [`../src/preprocessing/overview.md`](../src/preprocessing/overview.md):
  Documenter overview for user-facing preprocessing docs.
- [`../plans/41_UNIFIED_PREPROCESSOR/DESIGN.md`](../plans/41_UNIFIED_PREPROCESSOR/DESIGN.md):
  typed-axis migration history.
