# Plan 18 — Runtime Architecture Brief for v5

**Purpose:** GPT-5.4 review identified four architectural defects
in v4 ([review text in conversation]). This brief specifies the
correct runtime data flow so the plan agent can produce
`18_CONVECTION_PLAN_v5.md` that is actually executable.

**Status of v4:** decision structure, commit sequence, validation
approach, and Fortran analysis are all good and should carry
forward. The broken piece is the **runtime data flow from
window forcing to operator execution** — v4 short-circuits it
in a way that doesn't match the existing runtime.

**Scope of v5:** same as v4 + the corrections below. No other
content changes. Keep all decisions 1-22 (revised per §2 below).
Keep the 11-commit structure (with adjustments to Commit 6, 7,
8). Keep the three-tier validation. Keep the adjoint preservation
requirements. Keep the Fortran reference notes.

---

## 1. The core architectural mistake

V4 assumed that convection forcing reaches the operator via an
`AbstractTransportWindow` argument passed to `apply!` at step
time. But in the actual runtime:

- **Window forcing lives in `DrivenSimulation.window`** — not in
  `TransportModel`.
- **`TransportModel` holds `fluxes`** — a separate per-step
  container populated by `DrivenSimulation._refresh_forcing!`,
  which copies/interpolates from `sim.window.fluxes`.
- **`TransportModel.step!` takes `meteo = sim.driver`** — the
  driver passes through for time queries only, not for forcing
  data. The driver doesn't carry window forcing.

V4's operator signature `apply!(state, met::AbstractTransportWindow, ...)`
and its Commit 6 sketch passing `meteo` into `apply_convection!`
both assume data that isn't there at the call site. This breaks
end-to-end.

The fix is straightforward and matches how `fluxes` already
works: **convection forcing becomes a `TransportModel` field,
populated by `_refresh_forcing!` at each substep.**

## 2. The corrected data flow

### 2.1 Runtime container hierarchy

```
DrivenSimulation
├── model :: TransportModel
│   ├── state :: CellState             ← prognostic
│   ├── fluxes :: FaceFluxState        ← per-step forcing (existing)
│   ├── convection_forcing :: ConvectionForcing  ← NEW per-step forcing
│   ├── grid :: AtmosGrid
│   ├── advection :: AbstractAdvection
│   ├── diffusion :: AbstractDiffusionOperator
│   ├── emissions :: AbstractSurfaceFluxOperator
│   ├── chemistry :: AbstractChemistryOperator
│   ├── convection :: AbstractConvectionOperator   ← operator type
│   └── workspace :: ...
├── window :: AbstractTransportWindow  ← per-window forcing
│   ├── air_mass
│   ├── fluxes
│   ├── qv_start / qv_end
│   ├── deltas
│   └── convection :: Union{Nothing, ConvectionForcing}  ← NEW window payload
└── driver :: AbstractMetDriver        ← reader + timing, stateless
```

Two `ConvectionForcing` instances:
- **On the window** (populated by `load_transport_window!`): the
  window's-worth of forcing, read once per window.
- **On the model** (populated by `_refresh_forcing!`): the
  per-step forcing, copied/interpolated from the window each step.

For convection specifically, forcing is constant within a window
(no sub-step interpolation of CMFMC), so `_refresh_forcing!` just
takes a reference or does a no-op copy — no interpolation needed.
But the pattern matches advection for uniformity, and future
sub-window-varying convection (if it ever happens) uses the same
slot.

### 2.2 `ConvectionForcing` struct

Lives in the `MetDrivers` (or a shared location). Definition:

```julia
struct ConvectionForcing{CM, DT, TM}
    cmfmc      :: CM   # ::Union{Nothing, AbstractArray}  — see §3.2 for shape
    dtrain     :: DT   # ::Union{Nothing, AbstractArray}
    tm5_fields :: TM   # ::Union{Nothing, NamedTuple{(:entu,:detu,:entd,:detd)}}
end

# Default: everything nothing, used when convection is not active
ConvectionForcing() = ConvectionForcing(nothing, nothing, nothing)
```

Invariants (enforced by constructor):
- Either `(cmfmc, dtrain)` are both non-nothing OR `tm5_fields`
  is non-nothing OR all three are nothing. Not a mix.
- `dtrain` can be `nothing` even when `cmfmc` is non-nothing
  (Tiedtke-style single-flux fallback per v4 Decision 2).

### 2.3 The `apply!` signature

**Don't take an `AbstractTransportWindow`.** Take a
`ConvectionForcing` directly:

```julia
function apply!(
    state::CellState{B},
    forcing::ConvectionForcing,
    grid::AtmosGrid,
    op::Union{CMFMCConvection, TM5Convection},
    dt::Real;
    workspace,
) where {B <: AbstractMassBasis}
    ...
end
```

This is cleaner than v4's `met::AbstractTransportWindow` — the
operator only needs forcing arrays, not the whole window. The
state→grid→op signature mirrors plan 16b/17 exactly.

**Note type name:** use `AtmosGrid`, not `AbstractGrid`. The
latter doesn't exist; v4 used it incorrectly.

### 2.4 The `_refresh_forcing!` extension

In `DrivenSimulation.jl`, extend the existing `_refresh_forcing!`:

```julia
function _refresh_forcing!(sim::DrivenSimulation, substep::Int)
    λ = _substep_fraction(substep, sim.steps_per_window, typeof(sim.Δt), sim.use_midpoint_forcing)
    if sim.interpolate_fluxes_within_window
        interpolate_fluxes!(sim.model.fluxes, sim.window, λ)
    else
        copy_fluxes!(sim.model.fluxes, sim.window.fluxes)
    end
    expected_air_mass!(sim.expected_air_mass, sim.window, λ)
    if sim.qv_buffer !== nothing
        interpolate_qv!(sim.qv_buffer, sim.window, λ)
    end

    # NEW: refresh convection forcing (constant within window)
    if sim.model.convection_forcing !== nothing && sim.window.convection !== nothing
        copy_convection_forcing!(sim.model.convection_forcing, sim.window.convection)
    end

    return λ
end
```

`copy_convection_forcing!` copies arrays in place (or no-ops if
they reference the same storage). It's called every substep but
cheap — just references or shallow copies since the data doesn't
change mid-window.

Invalidation of the CMFMC CFL cache (Decision 21) happens in
`_maybe_advance_window!`, not here, because the cache only needs
to be invalidated when the cmfmc arrays actually change
(window boundaries):

```julia
function _maybe_advance_window!(sim, substep)
    if sim.iteration > 0 && substep == 1
        # ... existing window load ...

        # NEW: invalidate CFL cache when new window arrives
        if sim.model.workspace isa CMFMCWorkspace
            invalidate_cmfmc_cache!(sim.model.workspace)
        end
    end
end
```

### 2.5 Step orchestration

`TransportModel.step!` uses `model.convection_forcing` as the
forcing source, NOT `meteo`:

```julia
function step!(model::TransportModel, dt; meteo = nothing)
    # Transport block (existing — plan 16b+17 palindrome)
    apply!(model.state, model.fluxes, model.grid, model.advection, dt;
           workspace = model.workspace,
           diffusion_op = model.diffusion,
           emissions_op = model.emissions,
           meteo = meteo)

    # Convection block (NEW)
    if !(model.convection isa NoConvection)
        apply!(model.state, model.convection_forcing, model.grid,
               model.convection, dt;
               workspace = model.workspace.convection_ws)
    end

    # Chemistry block
    chemistry_block!(model.state, meteo, model.grid, model.chemistry, dt)
    return nothing
end
```

The `meteo` kwarg is threaded only to operators that need
`current_time(meteo)` for time-varying fields (plan 17 emissions).
Convection doesn't need it — forcing is already in
`model.convection_forcing`.

The `if !(model.convection isa NoConvection)` branch is
compile-time removed for default configs (`NoConvection`
singleton type).

## 3. Array-level entry point — handle both topologies

### 3.1 The shape problem

- **Structured `CellState`:** `tracers_raw` is `(Nx, Ny, Nz, Nt)`,
  `air_mass` is `(Nx, Ny, Nz)`.
- **Face-indexed `CellState`:** `tracers_raw` is `(ncells, Nz, Nt)`,
  `air_mass` is `(ncells, Nz)`.

V4's array-level entry only accepts `q_raw::AbstractArray{FT,4}`
— no face-indexed path.

### 3.2 Scope decision: face-indexed convection for plan 18?

**Recommendation: defer face-indexed convection to a follow-up
plan.**

Rationale:
- CATRINE production runs use structured lat-lon (verified
  against your workflow description)
- Face-indexed convection adds kernel complexity (different
  indexing, different reduction patterns) for no immediate user
- Plan 18 is already substantial; scope containment wins
- Follow-up "Plan 18b: Face-indexed convection" is a small
  addition once the structured path is validated

**If face-indexed convection IS needed for plan 18, say so
explicitly in the brief to me and I'll expand the design.**
Until then, plan 18 is structured-only; face-indexed is a
follow-up candidate.

### 3.3 Structured-only array-level signature

```julia
function apply_convection!(
    q_raw::AbstractArray{FT,4},        # (Nx, Ny, Nz, Nt)
    air_mass::AbstractArray{FT,3},     # (Nx, Ny, Nz)
    forcing::ConvectionForcing,        # fields shaped (Nx, Ny, Nz) / (Nx, Ny, Nz+1)
    op::AbstractConvectionOperator,
    dt::Real,
    workspace,
    grid::AtmosGrid,
) where FT
    ...
end
```

Both `q_raw` AND `air_mass` are passed. V4 left `air_mass` as a
placeholder; that's a real omission.

### 3.4 Face-indexed error stub

For safety, add an error stub on face-indexed state that tells
the user face-indexed convection isn't in plan 18:

```julia
function apply!(
    state::CellState{B, A, Raw, Names},
    forcing::ConvectionForcing,
    grid::AtmosGrid{<:AbstractFaceIndexedMesh},
    op::Union{CMFMCConvection, TM5Convection},
    dt::Real;
    workspace,
) where {B, A, Raw <: AbstractArray{T,3} where T, Names}
    throw(ArgumentError(
        "Face-indexed convection is not in plan 18 scope. " *
        "Structured lat-lon is supported. File follow-up plan " *
        "18b for face-indexed support."
    ))
end
```

Dispatch on `Raw <: AbstractArray{T,3}` catches the 3D packed
storage (face-indexed) vs 4D (structured). Verify this dispatch
pattern against current `CellState` in Commit 0.

## 4. `TransportBinaryDriver` / `TransportBinaryReader` format extension

### 4.1 The gap

V4 assumes `TransportBinaryDriver.reader` has `has_cmfmc`,
`load_cmfmc_window!`, etc. It doesn't — those live on
`ERA5BinaryReader`. The generic `TransportBinaryReader` (in
`TransportBinary.jl`) has no convection payload metadata at all.

For production CATRINE runs using the transport-binary format,
convection forcing can't be loaded with the current binary
schema.

### 4.2 Required infrastructure

Three chunks of work, all must land together:

**Chunk A: Header extension in `TransportBinary.jl`.**

Add to `TransportBinaryHeader`:
```julia
struct TransportBinaryHeader
    ... existing fields ...
    include_cmfmc    :: Bool
    include_tm5conv  :: Bool
    n_cmfmc          :: Int   # nelements per window, 0 if absent
    n_tm5conv        :: Int   # nelements per window per field (×4 in total)
end
```

Add corresponding serialization/deserialization in the
header-reading code (follow `include_qv` / `n_qv` as template).

**Chunk B: Reader methods in `TransportBinary.jl`.**

Add to `TransportBinaryReader`:
```julia
has_cmfmc(r::TransportBinaryReader) = r.header.include_cmfmc
has_tm5conv(r::TransportBinaryReader) = r.header.include_tm5conv

function load_cmfmc_window!(r::TransportBinaryReader, win::Int; cmfmc = ...)
    ...
end

function load_dtrain_window!(r::TransportBinaryReader, win::Int; dtrain = ...)
    ...
end

function load_tm5conv_window!(r::TransportBinaryReader, win::Int;
                              entu = ..., detu = ..., entd = ..., detd = ...)
    ...
end
```

Follow `load_qv_window!` / `load_surface_window!` as templates
(even though those names might be on ERA5 reader only — check
and mirror the correct template from current `TransportBinary.jl`).

**Note on DTRAIN:** this is the opportunity to add DTRAIN to
the binary format. V4 correctly identified that the current
ERA5 binary stores only CMFMC, not DTRAIN. In the new transport-
binary layout, include both CMFMC and DTRAIN as separate blocks.

**Chunk C: Preprocessing pipeline writes convection blocks.**

`src/Preprocessing/binary_pipeline.jl` (or wherever the transport
binary is generated from raw met) must be extended to propagate
convection blocks from the ERA5 binary (or source met files) to
the transport binary.

Inspect current pipeline to see where it reads from ERA5 and
writes the transport binary. Extend to:

```julia
if include_cmfmc_in_output
    cmfmc = load_cmfmc_window!(era5_reader, win)
    dtrain = compute_or_load_dtrain(...)   # see note below
    write_cmfmc_block(output_stream, cmfmc)
    write_dtrain_block(output_stream, dtrain)
end
if include_tm5conv_in_output
    tm5_fields = load_tm5conv_window!(era5_reader, win)
    write_tm5conv_block(output_stream, tm5_fields)
end
```

**DTRAIN sourcing question.** The ERA5 binary currently stores
only CMFMC. For a CMFMC+DTRAIN-capable transport binary, DTRAIN
must come from somewhere. Options:

1. **Extend ERA5 binary first** to include DTRAIN. Then transport
   binary reads both from ERA5 binary.
2. **Derive DTRAIN from cloud model** at preprocessing time. TM5
   stores entu/detu/entd/detd which encode detrainment; DTRAIN
   can be derived from `detu`. If the ERA5 binary has tm5conv,
   use that as the DTRAIN source.
3. **Accept DTRAIN=nothing** in plan 18 initial ship. The
   Tiedtke-style fallback (v4 Decision 2) means `CMFMCConvection`
   still runs; it just uses single-flux tendency. Full
   CMFMC+DTRAIN operation deferred until DTRAIN is in the
   binary.

**Recommendation for v5:** specify option 2 as the approach.
DTRAIN comes from the tm5conv `detu` field when the source
has tm5conv; otherwise stays `nothing` (fallback path). This
matches legacy behavior and makes the CMFMC+DTRAIN full-path
testable against both ERA5 (via tm5conv's detu) and GCHP (via
imported DTRAIN fields when that support arrives).

### 4.3 Scheduling in v5

Commit 7 must expand further than v4 specified. Recommended
sub-commits within Commit 7:

- **Commit 7.1:** `TransportBinary.jl` header extension + reader
  methods (~0.5 day)
- **Commit 7.2:** Preprocessing pipeline writes convection blocks
  (~1 day)
- **Commit 7.3:** `TransportBinaryDriver` convection integration:
  `supports_convection` override, `load_transport_window!`
  extension (~0.5 day)
- **Commit 7.4:** End-to-end integration test — full pipeline
  from raw ERA5 → transport binary with convection → driver →
  window → model → operator apply (~0.5 day)

Total Commit 7: ~2.5 days. Still one logical commit, but can be
split at session boundaries.

## 5. `current_time` — sim-level, not driver-level

### 5.1 The mismatch

`PRE_PLAN_18_FIXES.md` Fix A3 has `current_time(::TransportBinaryDriver)`
using `d.current_window_index`. That field doesn't exist on the
driver — it's on `DrivenSimulation`. The driver is stateless
(reader + grid only).

### 5.2 The correction

**`current_time` is a sim-level query.** The existing `meteo`
kwarg threads through operators, but `meteo = sim.driver` loses
the time context. Two fixes needed:

**Fix 5.2.1: Update the meteo arg to pass sim, not driver.**

In `DrivenSimulation.step!`:

```julia
# was: step!(sim.model, sim.Δt; meteo = sim.driver)
step!(sim.model, sim.Δt; meteo = sim)
```

This gives operators access to both `sim.driver` (for grid,
basis, reader methods) AND `sim.current_window_index` / `sim.time`
(for time queries).

**Fix 5.2.2: `current_time(::DrivenSimulation)` implementation.**

```julia
function current_time(sim::DrivenSimulation)
    return sim.time   # already tracked, updated per step
end
```

No driver changes needed. `sim.time` is already an existing field
(DrivenSimulation.jl:26).

**Fix 5.2.3: Fallback stub.**

For operators that need `current_time` but don't have a sim
(e.g., unit tests), keep the `nothing` fallback:

```julia
current_time(::Nothing) = 0.0
```

And optionally deprecate the old driver stub:

```julia
current_time(::AbstractMetDriver) = 0.0
# TODO(plan 18): remove; use current_time(sim) instead
```

### 5.3 Impact on PRE_PLAN_18_FIXES.md A3

**A3 must be rewritten in v5 (or as a prerequisite update).**
Current text says "implement `current_time(::TransportBinaryDriver)`
via `d.current_window_index`" — that's wrong because the driver
doesn't have that field. Correct text: "implement
`current_time(::DrivenSimulation)` via `sim.time`, and update
`DrivenSimulation.step!` to pass `sim` (not `sim.driver`) as
the `meteo` kwarg."

A3 becomes smaller, not bigger — `sim.time` already exists.

## 6. Summary of v5 changes vs v4

### Changes to §2 (Design decisions)

**Decision 20 (basis handling):** keep as-is. Already correct.

**Decision 21 (CFL sub-cycling):** keep as-is. Add §4.2 note that
cache invalidation hooks in `_maybe_advance_window!`, not
`_refresh_forcing!`.

**Decision 22 (convection forcing as window field):** EXTEND.
Convection forcing lives in BOTH the window (read from driver
per window) AND the model (per-step, refreshed via
`_refresh_forcing!`). The window field stays `ConvectionForcing`;
the model field is `ConvectionForcing`. Same struct, different
lifecycle slots. This mirrors how `fluxes` works.

**New Decision 23: Runtime data flow.** Convection forcing in
`TransportModel` is populated by `DrivenSimulation._refresh_forcing!`
each substep, analogous to how `model.fluxes` is populated.
`TransportModel.step!` reads from `model.convection_forcing`,
not from a window or driver argument. The `apply!` signature
takes `ConvectionForcing` directly, not `AbstractTransportWindow`.

**New Decision 24: `meteo` is the sim, not the driver.**
`DrivenSimulation.step!` passes `meteo = sim`, giving operators
access to both driver methods and time state. `current_time(sim)`
returns `sim.time`. Driver stays stateless.

**New Decision 25: Face-indexed convection is out of scope.**
Plan 18 is structured-only. Face-indexed is a follow-up candidate.
Error stub prevents accidental face-indexed use.

### Changes to §3 (Commit sequence)

**Commit 2 (window struct extension):** unchanged on window side.
Plus: add `convection_forcing :: ConvectionForcing` field to
`TransportModel` with default construction (all-nothing).

**Commit 6 (TransportModel wiring):** substantial updates:
- New `model.convection_forcing` field
- New `apply!` signature takes `ConvectionForcing`, not
  `AbstractTransportWindow`
- `step!` orchestration reads from `model.convection_forcing`

**Commit 7 (driver/window integration):** EXPANDED to include:
- Commit 7.1: `TransportBinary.jl` header + reader methods
- Commit 7.2: Preprocessing pipeline writes convection blocks
- Commit 7.3: `TransportBinaryDriver` convection methods
- Commit 7.4: End-to-end integration test

Total Commit 7 effort: ~2.5 days (was ~1.5).

**Commit 8 (DrivenSimulation integration):** EXPANDED to include:
- `_refresh_forcing!` extension that copies window →
  model.convection_forcing each substep
- `_maybe_advance_window!` extension that invalidates CMFMC cache
  on window roll
- `DrivenSimulation.step!` update: `meteo = sim` (not `sim.driver`)
- Sim-level `current_time(::DrivenSimulation)` override

Total Commit 8 effort: ~1 day (was sketched as minor).

### Changes to PRE_PLAN_18_FIXES.md

**Fix A3 rewrite:** `current_time` is a sim-level query. Remove
the `TransportBinaryDriver` override (it was impossible anyway —
the driver doesn't have `current_window_index`). Replace with:
- `current_time(::DrivenSimulation) = sim.time`
- `DrivenSimulation.step!` passes `meteo = sim` instead of
  `meteo = sim.driver`
- Keep `current_time(::AbstractMetDriver) = 0.0` fallback for
  non-sim contexts

Effort on A3 drops from 0.5-1 day to ~2 hours.

### No changes needed to

- All decisions 1-19 from v4 (including all the substantive ones
  on basis, sub-cloud, scavenging, adjoint preservation)
- The three-tier validation discipline
- The Fortran reference notes
- Known pitfalls (though a new one should be added: "assuming
  operator takes a window argument — it takes ConvectionForcing
  directly")
- The follow-up plan candidates

## 7. Verification checklist for v5

When producing v5, verify each of these against current `src/`:

- [ ] `DrivenSimulation` has fields `model`, `driver`, `window`,
      `time`, `iteration`, `current_window_index` (lines 17-38
      of src/Models/DrivenSimulation.jl)
- [ ] `DrivenSimulation._refresh_forcing!` copies `window.fluxes`
      → `model.fluxes` — use as template (lines 119-131)
- [ ] `DrivenSimulation.step!` passes `meteo = sim.driver` today;
      v5 changes this to `meteo = sim` (line 275)
- [ ] `TransportBinaryDriver` struct has only `reader` and `grid`
      (lines 109-112); no window index, no time field
- [ ] `TransportBinaryReader` constructed from `TransportBinary.jl`,
      NOT `ERA5BinaryReader.jl` (line 261); convection methods
      must be added to `TransportBinary.jl`
- [ ] `TransportBinaryHeader` has no convection payload fields
      (src/MetDrivers/TransportBinary.jl:35-); v5 adds them
- [ ] `CellState` structured has `tracers_raw::(Nx,Ny,Nz,Nt)`;
      face-indexed has `tracers_raw::(ncells,Nz,Nt)` (line 23-25)
- [ ] `air_mass` structured `(Nx,Ny,Nz)`; face-indexed `(ncells,Nz)`
- [ ] `AbstractGrid` does NOT exist in src; use `AtmosGrid`

## 8. Effort and scope

V5 net effect vs v4:
- Design decisions: +3 (Decisions 23, 24, 25)
- Commit sizing: Commit 7 grows from ~1.5d to ~2.5d; Commit 8
  grows from minor to ~1d
- PRE_PLAN_18_FIXES: A3 shrinks (correct scope is smaller)
- Total plan 18 effort: +1-1.5 days vs v4

Still fits in ~3-4 week timeline. No follow-up plans needed for
core convection — face-indexed is a separate follow-up, but
production CATRINE runs work without it.

## 9. What v5 does NOT change

- Three-tier validation is authoritative
- Cross-scheme tolerance is 5% (same basis, §Decision 10 corrected
  from v4)
- Inline helpers with dispatch-ready signatures for future
  scavenging
- Well-mixed sub-cloud layer added vs legacy
- Adjoint-identity Tier A test per operator
- No positivity clamp in the kernel
- All Fortran-reference line citations in the v4 notes remain
  valid

---

## Handoff instructions to plan agent

You (plan agent) produce `18_CONVECTION_PLAN_v5.md` that:

1. Starts from v4 as the base
2. Applies the corrections in §2-5 above
3. Keeps everything else unchanged
4. Verifies every interface claim against current `src/` per §7
5. Adds new Decisions 23, 24, 25 with the text suggested above
6. Updates Commits 6, 7, 8 per §6 changes
7. Also produces an updated `PRE_PLAN_18_FIXES.md` with Fix A3
   rewritten (`current_time` on sim, not driver)

Total expected effort: ~2-3 hours, structured passes.

After v5 is produced, the execution agent gets:
- `18_CONVECTION_PLAN_v5.md` (the authoritative plan)
- `PRE_PLAN_18_FIXES.md` (updated)
- `18_CONVECTION_UPSTREAM_FORTRAN_NOTES.md` (unchanged)
- `18_CONVECTION_UPSTREAM_GCHP_NOTES.md` (unchanged)

---

**End of architectural brief.**
