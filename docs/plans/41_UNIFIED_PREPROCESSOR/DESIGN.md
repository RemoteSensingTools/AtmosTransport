# Plan 41 — Design Rationale (Typed Three-Axis Preprocessor)

Companion to [NOTES.md](NOTES.md). NOTES is the executor punch-list;
this file is the design contract — what the type system buys, which
foot-guns dispatch closes, and what each abstract type's method-set
guarantees.

User constraint, 2026-05-15: "the binary preprocessor uses very
different paths for the different variations of met data and target
grids, this should be unified with multi-dispatch ... it seems
haphazard and dangerous right now ... [and] layer merging should be
part of the path for ALL pathways."

The "dangerous" word is load-bearing. Below: what is dangerous,
specifically, and how typed dispatch closes each foot-gun.

## Anti-pattern audit (real file:line citations)

These are the structural dangers in `src/Preprocessing/transport_binary/`
today — each one shippable as a silently-wrong binary unless the
operator manually notices.

**A. Per-method kwarg drift; missing kwargs are silently absent**

The three `process_day(date, grid, settings, vertical; …)` methods
that share a topology-dispatch contract have inconsistent kwargs:

| Path | Replay tol | Positivity CFL | Require positivity | Mass basis | Chain mass | Cross-day seed | Next-day endpoint |
|---|---|---|---|---|---|---|---|
| LatLon spectral ([latlon_spectral.jl:14](src/Preprocessing/transport_binary/latlon_spectral.jl#L14)) | derived | **absent** | **absent** | from settings | absent | absent | yes |
| CS spectral ([cubed_sphere_spectral.jl:14](src/Preprocessing/transport_binary/cubed_sphere_spectral.jl#L14)) | derived | yes | yes | from settings | absent | absent | yes |
| CS GEOS-native ([cubed_sphere_geos.jl:384](src/Preprocessing/transport_binary/cubed_sphere_geos.jl#L384)) | kwarg | yes | yes | kwarg `:dry` | kwarg | kwarg | accepted but unused |

A new contract knob (the round-2 `require_substep_positivity = false`
escape hatch) only lands if its kwarg is plumbed into each method.
**The LL path can't gate substep positivity at all today** — not
because LL is exempt, but because nobody propagated the knob there.

**B. `vertical::NamedTuple` is duck-typed across pathways with
divergent semantics**

[entrypoint.jl:161-163](src/Preprocessing/transport_binary/entrypoint.jl#L161-L163)
(native path) constructs:
```julia
vc = load_hybrid_coefficients(...)
vertical = (merged_vc = vc, Nz = Nz, Nz_native = Nz)   # identity, no merge
```

[entrypoint.jl:209](src/Preprocessing/transport_binary/entrypoint.jl#L209)
(spectral path) constructs:
```julia
vertical = build_vertical_setup(...)   # may merge or pressure-overlap
```

Both shapes type-check as `::NamedTuple`. `merged_vc` is a misnomer in
case A — it's not merged, it's identity. Downstream code that
inspects `vertical.merge_map` (length 0 vs length Nz_native) silently
takes different paths.

This is exactly the bug that fired in the 2026-05-14 v4 regen smoke:
the GEOS-IT pathway gives the L72 mesospheric thin layers (~14 Pa at
k=61) no vertical-merging option at all. The smoke surfaced
`outgoing/m = 1.356` at cell `(4, 47, 31, 61)` — a physical signal,
but one that the spectral path could have routed around with
`merge_thin_levels(min_thickness_Pa = 50)`, had the path existed.

**C. `mass_basis::Symbol` is a runtime contract, not a dispatch axis**

[cubed_sphere_geos.jl:384](src/Preprocessing/transport_binary/cubed_sphere_geos.jl#L384):
```julia
mass_basis::Symbol = :dry
```

The writer producing dry-basis output and the runtime reader
expecting dry-basis are paired via the binary header at load time —
NOT at compile time. A mass-basis mismatch is detectable only after
the binary is on disk. Since dry-basis is the runtime contract
([CLAUDE.md "Dry basis is the default runtime contract"](../../../CLAUDE.md)),
this is exactly the foot-gun that drove the project_cs_moist_basis_bug
RESOLVED-2026-04-14 incident.

**D. Cross-day seed plumbing is implicit and source-specific**

[cubed_sphere_geos.jl:349](src/Preprocessing/transport_binary/cubed_sphere_geos.jl#L349)
takes `seed_m = nothing` and returns `final_m` in a NamedTuple. The
entrypoint at [entrypoint.jl:191](src/Preprocessing/transport_binary/entrypoint.jl#L191)
threads this with `seed_m = get(result, :final_m, nothing)`. If a
future native source needs a 2-array seed (e.g., GEOS-FP's pressure
fixer + spectral-residual state), the protocol has no place to put
it without changing the function signature in every caller.

**E. Per-window write-time gates each call their own contract module**

Each of the 4 process_day implementations contains its own
"open writer / loop / replay gate / positivity gate / close / promote"
sequence. The CS positivity homogenization (d1f50b6) consolidated the
**contract calls** but not the **loop scaffolding**. A new topology
or a new gate type still requires a 4-way fan-out edit.

**F. `cubed_sphere_regrid.jl::regrid_ll_binary_to_cs` is a fifth
preprocessor that doesn't dispatch through `process_day`**

It consumes already-preprocessed LL binaries and re-emits CS binaries.
It lives in the same directory and uses the same contract module, but
it doesn't share the driving-loop scaffold. **It's out of scope for
Plan 41** (Plan 42 territory), but it should be cited as a sibling
the unified design will eventually subsume.

## Design — three orthogonal axes, all typed

Three orthogonal axes, each with its own abstract type. The driver
dispatches on the **product** of the three.

### Axis 1 — Source (read native met into typed buffers)

```julia
abstract type AbstractMetReader{FT <: AbstractFloat,
                                 MetSettings <: AbstractMetSettings,
                                 ChainPolicy <: AbstractChainPolicy} end
```

Concrete types: `ERA5SpectralReader`, `GEOSNativeReader`, etc.

The `ChainPolicy` type parameter encodes whether cross-day mass-state
carry happens at all (`NoChain` vs `ChainedMass{T}`). This makes the
seed-threading protocol type-system-mandatory for chained-mass
readers and a compile-time no-op for non-chained — closing foot-gun
(D).

Required methods:

```julia
open_day(reader_type, settings::MetSettings, date::Date,
         ::Type{FT}; seed::Union{Nothing, ChainSeed{T}}) where {FT, T}
    → reader::AbstractMetReader

windows_per_day(reader)::Int

# Allocates source-shape buffers `dst`; reads window `w` into them.
read_window!(dst, reader, w::Int) → nothing

# Returns the chained-mass seed for the next day, or `nothing`. Type
# is statically known from the reader's ChainPolicy parameter — a
# `NoChain` reader returns `nothing` and the next day's open_day is
# typed to refuse a non-nothing seed.
end_of_day_seed(reader::AbstractMetReader{FT, MS, ChainedMass{T}}) where {FT, MS, T} → T
end_of_day_seed(reader::AbstractMetReader{FT, MS, NoChain}) where {FT, MS} = nothing

# Native vertical-coordinate metadata (the source's level grid).
native_vertical(reader) → HybridSigmaPressure{FT}

# Native window timing metadata. A 3-hourly source returns 8.
# A 1-hourly source returns 24. Window-substep cadence
# (mass_flux_dt for GEOS) lives here too.
window_metadata(reader) → NamedTuple{(:windows, :substeps, :dt_substep, …)}

close_day!(reader) → nothing
```

**What this closes:**
- (A) → all "do I need to plumb this kwarg" decisions move into the
  reader's struct fields, which are checked at construction time.
- (D) → `ChainSeed{T}` is a typed nominal, not a NamedTuple field.
  Cross-source seed shapes are statically distinguishable.

### Axis 2 — Vertical transform (native source levels → output levels)

**This is the new axis** that closes the layer-merging gap (B) and
the "GEOS-IT can't merge mesospheric layers" foot-gun the user
flagged. Today's `build_vertical_setup` already implements three
strategies for the spectral path — they become first-class types:

```julia
abstract type AbstractVerticalTransform end

struct IdentityVertical <: AbstractVerticalTransform end

struct ThinLevelMerge <: AbstractVerticalTransform
    min_thickness_Pa :: Float64  # default 1000.0 for spectral, 50.0 for GEOS-IT mesosphere
end

struct LevelSelection <: AbstractVerticalTransform
    echlevs :: Vector{Int}  # 0-based half-level indices, bottom-up
end

struct PressureOverlap <: AbstractVerticalTransform
    target_coeff_path :: String  # e.g. config/geos_L72_coefficients.toml
end
```

Required methods:

```julia
# Plan once per day (or per run if reader.native_vertical is invariant).
plan_vertical(transform::AbstractVerticalTransform,
              native_vc::HybridSigmaPressure{FT}) where FT
    → VerticalPlan{FT}  # holds merged_vc, mapping (merge_map | overlap_coefs | nothing),
                        #            Nz_output, transform_kind ∈ {:identity, :merge_map, :pressure_overlap}

# Apply to one window in-place. Source-shape `buf_in` (Nz_native
# levels) → target-shape `buf_out` (Nz_output levels).
apply_vertical!(buf_out, buf_in, plan::VerticalPlan, ::FieldKind)
    → nothing
```

`FieldKind` is a singleton-type tag (`MassField`, `MassFluxField`,
`PressureFluxField`) that selects the right interpolation rule (sum
vs face-averaging vs surface-pinning). Each topology MAY add its own
`FieldKind` overloads but the default ones close 90% of cases.

**What this closes:**
- (B) → `vertical` is no longer a duck-typed NamedTuple; it's a
  typed `VerticalPlan{FT}` whose `transform_kind` is statically
  known.
- The "GEOS-IT can't merge thin levels" gap becomes a config choice:
  set `[vertical].transform = "thin_level_merge"` with
  `min_thickness_Pa = 50` and the L72 mesospheric layers merge into
  ~50 Pa output layers automatically.
- The thin-layer regen issue from 2026-05-14 becomes a different
  decision (which transform + what min_thickness) rather than a
  "we'd need to refactor first" blocker.

### Axis 3 — Target topology (write topology-shaped binaries with
typed contract)

`AbstractTargetGeometry` already exists. We tighten it by REQUIRING
each topology to register a complete dispatch surface — currently
only CS has the full set:

```julia
# Per-topology workspace (haloed panel tuples for CS, structured for
# LL, face-indexed for RG).
abstract type AbstractWindowWorkspace{G <: AbstractTargetGeometry, FT} end

allocate_window_workspace(grid::AbstractTargetGeometry,
                           vertical::VerticalPlan,
                           reader::AbstractMetReader)
    → AbstractWindowWorkspace

# Regrid one window from source-shape to target-shape after vertical
# transform has been applied.
regrid_window!(workspace::AbstractWindowWorkspace{G, FT},
                reader::AbstractMetReader, w::Int,
                vertical::VerticalPlan) where {G, FT}
    → nothing

# Per-topology window contract. CS today has both replay AND
# positivity; LL/RG today have replay only. The plan REQUIRES each
# topology to answer the positivity question explicitly (yes-with-
# implementation, or no-with-documentation).
abstract type AbstractWindowContract{G <: AbstractTargetGeometry, FT} end

verify_window!(workspace::AbstractWindowWorkspace{G, FT},
                contract::AbstractWindowContract{G, FT},
                w::Int) where {G, FT}
    → (replay::ReplayDiag, positivity::PositivityDiag)

update_accumulator!(contract::AbstractWindowContract, positivity::PositivityDiag, w::Int)

summarize_status!(contract::AbstractWindowContract;
                   quarantine_path::Union{Nothing, AbstractString})
    → nothing  # may error or warn per contract.policy

# Per-topology streaming writer. The MassBasis type parameter
# closes foot-gun (C): a writer producing dry-basis output is a
# statically different type from one producing moist-basis.
abstract type AbstractBinaryWriter{G <: AbstractTargetGeometry, FT,
                                    Basis <: AbstractMassBasis} end

open_streaming_binary(grid, out_path::AbstractString, header,
                       ::Type{Basis}) where Basis
    → AbstractBinaryWriter{G, FT, Basis}

write_window!(writer::AbstractBinaryWriter{G, FT, Basis},
                workspace::AbstractWindowWorkspace{G, FT}, w::Int)
    → bytes::Int

close_streaming_binary!(writer; commit::Bool)
    # commit=true: atomic rename .tmp → final
    # commit=false: leave .tmp for quarantine inspection
```

Where:

```julia
abstract type AbstractMassBasis end
struct DryMass   <: AbstractMassBasis end
struct MoistMass <: AbstractMassBasis end
```

**What this closes:**
- (A) → contract kwargs aren't drift-prone kwargs anymore; each
  topology constructs its own `AbstractWindowContract{G, FT}` from
  config, with whatever knobs IT needs. Adding `cfl_limit` to LL's
  contract is a 5-line struct extension in `LatLonContract`, not
  a 4-method edit.
- (C) → mass-basis mismatch becomes a compile-time `MethodError`,
  not a runtime header check.
- (E) → the driver opens the writer exactly once, the per-window
  loop body is bit-identical across topologies.

### The driver (one ~80-line method, no topology branches)

```julia
function process_day(reader::AbstractMetReader{FT, S, CP},
                     vertical::VerticalPlan{FT},
                     grid::G,
                     contract::AbstractWindowContract{G, FT},
                     workspace::AbstractWindowWorkspace{G, FT},
                     writer::AbstractBinaryWriter{G, FT, Basis};
                     out_path::AbstractString) where {FT, S, CP, G, Basis}
    nw = windows_per_day(reader)
    local writer_closed = false
    try
        for w in 1:nw
            read_window!(workspace.source, reader, w)
            apply_vertical!(workspace.source_remapped, workspace.source,
                             vertical, MassField())
            regrid_window!(workspace, reader, w, vertical)
            diag = verify_window!(workspace, contract, w)
            update_accumulator!(contract, diag.positivity, w)
            write_window!(writer, workspace, w)
        end
        close_streaming_binary!(writer; commit = true); writer_closed = true
    finally
        if !writer_closed
            close_streaming_binary!(writer; commit = false)
        end
    end
    summarize_status!(contract; quarantine_path = out_path * ".tmp")
    return end_of_day_seed(reader)
end
```

Every type used in the signature is dispatched on. Adding ORD=8
(hypothetically), GEOS-FP, a new vertical merge strategy, or
Reduced-Gaussian positivity: none of them require editing the driver.

## Foot-gun closure table

| Foot-gun | Today | Closed by |
|---|---|---|
| (A) kwarg drift | LL spectral has no `positivity_cfl_limit` kwarg; can't gate substep positivity even if user wants | `AbstractWindowContract{G, FT}` struct owns its kwargs; constructed once from cfg per topology |
| (B) `vertical::NamedTuple` duck typing | `entrypoint.jl:161-163` fakes `merged_vc = vc` for GEOS-IT path | `VerticalPlan{FT}` typed nominal; `transform_kind` statically known; `IdentityVertical` is its own type |
| (B') GEOS-IT can't merge thin levels | thin-layer mesospheric `outgoing/m=1.356` blocks the v4 regen | config selects `ThinLevelMerge(min_thickness_Pa=50)`; the L72 path merges layers like the ERA5 spectral path already does |
| (C) `mass_basis::Symbol` is a runtime header check | dry/moist mismatch detectable only post-hoc | `AbstractBinaryWriter{G, FT, Basis<:AbstractMassBasis}`; pairing is a compile-time `MethodError` |
| (D) cross-day seed plumbing | `seed_m::Union{Nothing, NamedTuple}` threaded by name lookup | `ChainPolicy` type parameter on `AbstractMetReader`; `end_of_day_seed` return type is statically known per reader |
| (E) 4-way fan-out for new gate | adding `require_substep_positivity` had to land in 3 files | one driver loop; per-topology contract owns its own policy |
| (F) `cubed_sphere_regrid.jl` is a fifth pathway | distinct entry point, same contract module | out of scope (Plan 42); design admits it as a `Sourceless` reader that reads from an LL `AbstractBinaryWriter`-produced binary |

## State-flow for one window

```
                                         ┌──────────────────────────────┐
                                         │ AbstractMetReader            │
                                         │  read_window!(src_buf, w)    │   native
                                         └────────────┬─────────────────┘   shape
                                                      ▼                     ─────
                                         ┌──────────────────────────────┐
                                         │ VerticalPlan                 │
                                         │  apply_vertical!(remap, src) │   output
                                         └────────────┬─────────────────┘   levels
                                                      ▼                     ─────
                                         ┌──────────────────────────────┐
                                         │ AbstractWindowWorkspace{G}   │
                                         │  regrid_window!(ws, …)       │   target
                                         └────────────┬─────────────────┘   shape
                                                      ▼                     ─────
                          ┌─────────────────────────────────────────┐
                          │ AbstractWindowContract{G, FT}           │
                          │  verify_window! → (replay, positivity)  │
                          │  update_accumulator!                    │
                          └────────────┬────────────────────────────┘
                                       ▼
                          ┌─────────────────────────────────────────┐
                          │ AbstractBinaryWriter{G, FT, Basis}      │
                          │  write_window!(writer, ws, w)           │
                          └────────────┬────────────────────────────┘
                                       ▼
                       ┌────────────────────────────────────────────┐
                       │ end-of-day:                                │
                       │   close_streaming_binary!(writer; commit)  │
                       │   summarize_status!(contract)              │
                       │   end_of_day_seed(reader) → next day       │
                       └────────────────────────────────────────────┘
```

Every horizontal slice is a typed dispatch site. The diagram is
isomorphic to the driver function above.

## Type-system invariants this design enforces

These are facts the type system makes IMPOSSIBLE to violate. Each
one corresponds to a class of bug the current path admits.

1. **A mass-flux writer for grid type G is only constructable from a
   workspace of grid type G.** `write_window!(writer::AbstractBinaryWriter{G,…}, workspace::AbstractWindowWorkspace{G,…}, …)` —
   if G doesn't match, no method matches.

2. **Cross-day mass seed flows only between readers that opted into
   `ChainedMass{T}`.** A `NoChain` reader returns `nothing`;
   `open_day(reader_t, settings, date, FT; seed::T)` for a `NoChain`
   reader_t fails dispatch because `seed::Nothing` is the only
   admissible signature.

3. **Vertical transform output shape is statically known.** The
   `VerticalPlan{FT}` carries `Nz_output` as a value but its type is
   stable. The downstream `regrid_window!` allocates against
   `plan.Nz_output` from a single typed accessor.

4. **A contract's policy fields are construction-time validated.**
   E.g. `CubedSphereContract(; positivity_cfl_limit = 0.0)` errors
   in the inner constructor — round-3's
   `_resolve_positivity_cfl_limit` validation moves from a free
   function into `CubedSphereContract`'s `function CubedSphereContract(…)`.

5. **Mass-basis pairing is type-checked.** A writer with
   `Basis = DryMass` cannot accept input from a workspace that
   labels its tracer arrays as `MoistMass` (the workspace also
   carries a basis type parameter; mismatch fails dispatch).

## Migration sequencing (vs NOTES.md)

This DESIGN doc supersedes NOTES.md's P0 sketch by promoting
**Vertical transform** from "out of scope (separate plan)" to an
in-scope axis. Concretely:

- **P0 (was: "reader trait surface")** → adds Axis-1 (`AbstractMetReader`)
  AND Axis-2 (`AbstractVerticalTransform`). The two are independent
  and can ship side-by-side without touching the existing driver.

- **P1 (was: "target dispatch tightening")** → adds Axis-3
  (`AbstractWindowContract` + `AbstractBinaryWriter` + `AbstractWindowWorkspace`)
  with concrete LL/RG/CS implementations. Forces the LL/RG positivity
  question to be answered explicitly (yes-with-implementation or
  no-with-justification).

- **P2 (was: "workspace + regrid trait")** → integration: the
  per-method driving loops in `latlon_spectral.jl`,
  `cubed_sphere_spectral.jl`, `cubed_sphere_geos.jl`,
  `reduced_transport_helpers.jl` now delegate to the trait calls.

- **P3 (was: "unified driver")** → the ~80-line driver above lives
  behind a `[preprocessor].unified = true` opt-in flag. Side-by-side
  smokes verify bit-exact binaries against the legacy paths for ERA5
  spectral × LL, ERA5 spectral × CS, GEOS native × CS.

- **P4 (was: "cut over")** → the legacy `process_day(date, grid,
  settings, vertical)` methods move to `src_legacy/Preprocessing/`.
  The opt-in flag is removed.

- **P5 (was: "validation")** → add a new source reader (e.g.,
  MERRA-2 native) in ≤150 lines of `AbstractMetReader` implementation.
  If P0–P4 worked, no other file changes are needed for the new
  source to land.

## What this is NOT trying to do

- **Not rewriting the runtime.** Runtime I/O is Plan 40 (done). This
  is preprocessor-only.
- **Not changing binary format.** v4 binaries produced before and
  after the migration are bit-identical for the same inputs.
- **Not adding new physics.** Convection, diffusion, advection: all
  unchanged.
- **Not changing the runtime contract.** Dry-basis remains the
  default; CS positivity gate semantics (round-2 + round-3) preserved.
- **Not solving the regrid-LL-to-CS gap.** `cubed_sphere_regrid.jl`
  stays out of scope (Plan 42).

## Hard constraints

1. **Bit-exact binaries** for every existing TOML config through P4.
2. **CS contract semantics floor** — round-2 + round-3 gates and the
   `require_substep_positivity = false` escape hatch preserved
   verbatim.
3. **GEOS chained-mass cross-day handoff** continues to work; tested
   with the existing 5-day chained-mass smoke before P4 lands.
4. **Boundary-day error legibility** — the 2023-12-31 boundary
   failure in the previous regen (no next-day CTM_I1 endpoint) must
   still surface with a comprehensible error.
5. **No silent behavior change for valid inputs.** Side-by-side
   smokes in P3 must produce byte-identical binaries.

## Open design questions (decide during P0–P1)

1. **`AbstractMassBasis` granularity** — do we want
   `DryMass <: AbstractMassBasis` as a singleton type, or a typed
   nominal `MassBasis{:dry}`/`MassBasis{:moist}` (`Val`-style)?
   The former is more Julian; the latter compresses to fewer files.
   No strong preference; P1 commit picks one.

2. **`VerticalPlan{FT}` mutability** — is it allocated once per day
   (because `native_vc` could drift across days if a reader updates
   its native vertical mid-run) or once per run? Today no reader
   does the former, but the trait should support it. Likely:
   constructed lazily on `open_day` and cached on the reader.

3. **`AbstractWindowWorkspace` sharing across windows** — today
   every `process_day` allocates per-day. The new `process_day` can
   keep this OR hoist allocation to once-per-run with reset
   semantics. The latter is faster for multi-day runs and matters
   for GPU runs where allocation pressure is real. Likely: per-run
   with `reset_workspace!(ws)` between days.

4. **What about Recipe-level metadata** (e.g., `mass_flux_dt = 450`
   for GEOS, `T_target` for spectral)? These are reader-specific.
   Probably: stored as reader fields, accessed via `window_metadata`
   only when needed by the writer's header.

5. **Should the unified driver also subsume the existing
   `_process_day_spectral` legacy NamedTuple-settings path?** The
   answer is yes — `ERA5SpectralReader` wraps the old
   `resolve_runtime_settings` and exposes the same per-window
   interface as `GEOSNativeReader`. The legacy `NamedTuple` lives
   inside the reader, not in any public API.
