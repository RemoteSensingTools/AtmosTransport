# Pre-Plan-18 Fixes — Executable Specification

**Purpose:** three current-tree bugs identified by GPT-5.4 review
that block plan 18 execution. This document specifies each fix as
a standalone deliverable with acceptance criteria. Ship these
before starting plan 18 Commit 0.

**Recommended sequencing:** single PR containing all three fixes,
or three small PRs in order A1 → A3 → A2 (A2 is independent). No
operator changes; these are infrastructure fixes to the runtime
contract and test harness.

---

## Fix A1: Face-indexed `apply!` contract catchup

**Severity:** HIGH — currently breaks reduced-Gaussian runtime
entirely.

### Problem

`TransportModel.step!` (src/Models/TransportModel.jl:162-170)
forwards `diffusion_op`, `emissions_op`, `meteo` through `apply!`
since plans 16b/17:

```julia
function step!(model::TransportModel, dt; meteo = nothing)
    apply!(model.state, model.fluxes, model.grid, model.advection, dt;
           workspace = model.workspace,
           diffusion_op = model.diffusion,
           emissions_op = model.emissions,
           meteo = meteo)
    chemistry_block!(model.state, meteo, model.grid, model.chemistry, dt)
    return nothing
end
```

The structured lat-lon path at StrangSplitting.jl:1008-1015
accepts these kwargs. The face-indexed path at
StrangSplitting.jl:1035-1039 does NOT:

```julia
@eval function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                      grid::AtmosGrid{<:AbstractHorizontalMesh},
                      scheme::$scheme_type, dt;
                      workspace::AdvectionWorkspace,
                      cfl_limit::Real = one(eltype(state.air_mass))) where {B <: AbstractMassBasis}
```

Calling `TransportModel.step!` on a face-indexed (reduced-Gaussian)
model throws `MethodError` — the `diffusion_op`/`emissions_op`/`meteo`
kwargs are unrecognized.

Observable regression: `test_transport_binary_reader.jl:201`
("TransportBinaryReader reduced-Gaussian path" testset) fails.

### What's missing

Face-indexed `apply!` body at StrangSplitting.jl:1035-1073
implements only `H → V → V → H` — four sweeps, no palindrome
center, no diffusion, no emissions:

```julia
_sweep_horizontal_face_subcycled!(rm_tracer, m, hflux, grid.horizontal, scheme, workspace, cfl_limit_ft)
_sweep_vertical_face_subcycled!(rm_tracer, m, cm, scheme, workspace, cfl_limit_ft)
_sweep_vertical_face_subcycled!(rm_tracer, m, cm, scheme, workspace, cfl_limit_ft)
_sweep_horizontal_face_subcycled!(rm_tracer, m, hflux, grid.horizontal, scheme, workspace, cfl_limit_ft)
```

Plans 16b/17 extended the structured path to the full palindrome
`X Y Z [V(dt/2) S(dt) V(dt/2)] Z Y X` (strang_split_mt! at
StrangSplitting.jl:1199-1246) but never touched face-indexed.

### Deliverable

**Step 1: Extend signature.**

Modify both `@eval` blocks at StrangSplitting.jl:1031-1074 and
:1076-1086 to accept the same kwargs as structured path:

```julia
for (scheme_type, h_sweep, v_sweep) in (
    (:AbstractConstantScheme,    :sweep_horizontal!, :sweep_vertical!),
)
    @eval function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                          grid::AtmosGrid{<:AbstractHorizontalMesh},
                          scheme::$scheme_type, dt;
                          workspace::AdvectionWorkspace,
                          cfl_limit::Real = one(eltype(state.air_mass)),
                          diffusion_op::AbstractDiffusionOperator = NoDiffusion(),
                          emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                          meteo = nothing) where {B <: AbstractMassBasis}
        # ... body ...
    end
end
```

Error stubs at :1077-1086 just need `; kwargs...` to swallow the
extra args gracefully (they already throw `ArgumentError`, so no
additional work).

**Step 2: Implement palindrome in face-indexed body.**

Replace the 4-sweep body with a full palindrome matching the
structured path's semantics. Face-indexed has only one horizontal
"direction" (topology is unstructured), so the palindrome is
`H Z [V(dt/2) S(dt) V(dt/2)] Z H` — five sweeps + center,
equivalent to the structured `X Y Z [...] Z Y X` under the
constraint that H is a single face-flux sweep:

```julia
@eval function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                      grid::AtmosGrid{<:AbstractHorizontalMesh},
                      scheme::$scheme_type, dt;
                      workspace::AdvectionWorkspace,
                      cfl_limit::Real = one(eltype(state.air_mass)),
                      diffusion_op::AbstractDiffusionOperator = NoDiffusion(),
                      emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                      meteo = nothing) where {B <: AbstractMassBasis}
    m = state.air_mass
    hflux, cm = fluxes.horizontal_flux, fluxes.cm
    cfl_limit_ft = convert(eltype(m), cfl_limit)
    
    n_tr = ntracers(state)
    n_tr == 0 && return nothing
    
    m_save = n_tr > 1 ? similar(m) : m
    if n_tr > 1
        copyto!(m_save, m)
    end
    
    raw = state.tracers_raw
    last_dim = ndims(raw)
    
    for idx in 1:n_tr
        if idx > 1
            copyto!(m, m_save)
        end
        rm_tracer = selectdim(raw, last_dim, idx)
        
        # Forward half: H → Z
        _sweep_horizontal_face_subcycled!(rm_tracer, m, hflux, grid.horizontal, scheme, workspace, cfl_limit_ft)
        _sweep_vertical_face_subcycled!(rm_tracer, m, cm, scheme, workspace, cfl_limit_ft)
        
        # Palindrome center — mirrors strang_split_mt! lines 1228-1241
        if emissions_op isa NoSurfaceFlux
            apply_vertical_diffusion_face!(rm_tracer, diffusion_op, workspace, dt, meteo)
        else
            half_dt = dt === nothing ? nothing : dt / 2
            apply_vertical_diffusion_face!(rm_tracer, diffusion_op, workspace, half_dt, meteo)
            apply_surface_flux_face!(rm_tracer, emissions_op, workspace, dt, meteo, grid;
                                     tracer_name = state.tracer_names[idx])
            apply_vertical_diffusion_face!(rm_tracer, diffusion_op, workspace, half_dt, meteo)
        end
        
        # Reverse half: Z → H
        _sweep_vertical_face_subcycled!(rm_tracer, m, cm, scheme, workspace, cfl_limit_ft)
        _sweep_horizontal_face_subcycled!(rm_tracer, m, hflux, grid.horizontal, scheme, workspace, cfl_limit_ft)
    end
    
    return nothing
end
```

**Step 3: Implement helper functions.**

`apply_vertical_diffusion_face!` and `apply_surface_flux_face!`
are the face-indexed analogs of the structured versions used by
`strang_split_mt!`. Check whether the current `ImplicitVerticalDiffusion.apply!`
and `SurfaceFluxOperator.apply!` dispatch on `CellState{B}`
regardless of horizontal topology, or whether they need
topology-specific methods:

- If they dispatch on `state::CellState{B}` without topology
  constraints, the existing structured helpers likely work
  directly on face-indexed state (vertical diffusion is columnar;
  surface flux is per-column point-local). In that case,
  `apply_vertical_diffusion_face!` is a thin wrapper renaming
  the call; may not even need a separate function.

- If they take a 4D `tracers_raw` slice, face-indexed is 3D
  `(ncells, Nz)` so a different slice shape. Operator needs a
  method for both shapes, or face-indexed wrapper extracts a
  column and calls per-column.

Execution agent: inspect `src/Operators/Diffusion/operators.jl`
and `src/Operators/SurfaceFlux/operators.jl` to determine which
path is needed. Likely the former — plan 16b shipped
`ImplicitVerticalDiffusion` as a generic column operator that
should work on both topologies. If so, work is mechanical.

**Per-tracer loop caveat.** Face-indexed keeps a per-tracer loop
(comment at StrangSplitting.jl:1052-1056 notes multi-tracer
fusion on unstructured grids is out of scope for plan 14).
Surface flux expects per-tracer index resolution; pass
`state.tracer_names[idx]` (singular) rather than the whole tuple.
Verify this signature matches what `apply_surface_flux_face!`
expects; if not, construct a single-element tuple.

### Acceptance criteria

1. `test_transport_binary_reader.jl` passes end-to-end including
   the `reduced-Gaussian path` testset at line 201.

2. No regression in structured lat-lon tests
   (`test_transport_binary_v2_dispatch.jl`,
   `test_real_era5_latlon_e2e.jl`, etc.).

3. Plan 17's `test_transport_model_emissions.jl` passes unchanged
   for structured (should be bit-exact) AND for face-indexed if
   a face-indexed emission test exists; if not, add one:
   construct a reduced-Gaussian model with `SurfaceFluxOperator`,
   run one step, verify mass balance.

4. Plan 16b's `test_transport_model_diffusion.jl` similarly:
   face-indexed diffusion test passes or is added.

5. Bit-exact regression for structured runs: the structured path
   should be untouched by this fix. Any structured regression
   means a mistake.

### Time estimate

1-2 days, depending on whether the diffusion/emissions operators
need face-indexed dispatch methods (if yes, up to half the time
is in that plumbing).

### What NOT to do

- Do NOT change the structured path. It works; leave it alone.
- Do NOT refactor `strang_split_mt!` to "generalize" across
  topologies. Different topologies, different sweep semantics;
  keep them separate functions.
- Do NOT add face-indexed multi-tracer fusion. Out of scope for
  this fix (and for plan 18). File as follow-up.

---

## Fix A2: Stale v1-vs-v2 test harness cleanup

**Severity:** MEDIUM — pollutes CI with unreliable tests, segfault
on rerun.

### Problem

After the `src_v2 → src` promotion, two test files still contain
legacy "also load v1" include lines that now double-include the
same module:

- `test/test_dry_flux_interface.jl:23-24`:
  ```julia
  include("../src/AtmosTransport.jl")   # already loaded
  include("../src/AtmosTransport.jl")   # duplicate from v1-vs-v2 era
  ```
  
- `test/test_real_era5_v1_vs_v2.jl:29-30`: same pattern.

GPT-5.4 review reports segfault on second execution of
`test_dry_flux_interface.jl` around the zonal-flow / continuity
section (~line 428). Duplicate include is the likely cause but
not proven; regardless, the harness is stale.

### Deliverable

**Option 1 (recommended): delete `test_real_era5_v1_vs_v2.jl` entirely.**

The v1/v2 distinction no longer exists. The file can't possibly
be doing a meaningful v1-vs-v2 comparison on a post-promotion
tree. Delete it and remove its entry from `test/runtests.jl` if
present.

**Option 2: remove duplicate includes from both files.**

If `test_dry_flux_interface.jl` contains other useful tests
beyond the v1-vs-v2 comparison, keep the file but delete the
duplicate `include` at line 24. Likewise for
`test_real_era5_v1_vs_v2.jl:30`.

Execution agent: check what each file actually tests. If the
file only exists for v1-vs-v2 comparison, delete it. If it also
has unrelated tests, deduplicate the include.

### Acceptance criteria

1. `julia --project=. test/test_dry_flux_interface.jl` runs to
   completion without segfault, TWICE in a row (exercising the
   state that segfaulted previously).

2. `test/test_real_era5_v1_vs_v2.jl` is either deleted or runs
   cleanly.

3. `test/runtests.jl` includes only files that exist.

### Time estimate

1-2 hours.

### What NOT to do

- Do NOT try to "make v1-vs-v2 work". The distinction is gone.
- Do NOT preserve the duplicate include "in case it's needed
  later" — it's not.

---

## Fix A3: `current_time` sim-level override (rewritten in v5)

**Severity:** MEDIUM — silent wrong-answer for time-varying
forcing.

**Status:** Rewritten in v5. The original A3 (in v4 PRE_PLAN_18_FIXES)
proposed `current_time(::TransportBinaryDriver)` using
`d.current_window_index` — that field **does not exist on the driver**.
`TransportBinaryDriver` is stateless (struct at
`src/MetDrivers/TransportBinaryDriver.jl:109-112` has only `reader`
and `grid`). The canonical time state is on `DrivenSimulation`, not
on the driver. This rewrite targets sim-level overrides instead.

### Problem

`src/MetDrivers/AbstractMetDriver.jl:85` defines:

```julia
current_time(::AbstractMetDriver) = 0.0
```

as a stub. No concrete overrides exist. Any
`StepwiseField` / `AbstractTimeVaryingField` that reads
`current_time(meteo)` silently sees `t = 0.0` throughout the run,
so time-varying forcing collapses to the first window's value.

Critically, `DrivenSimulation.step!` (line 275) passes
`meteo = sim.driver`. Even if the driver had a real `current_time`
override, the driver is stateless — it has no current window index,
no elapsed time. The sim does (`sim.time` at line 26, advanced at
line 276: `sim.time += sim.Δt`).

This doesn't currently break any shipped test — `ConstantField`
doesn't care about time, and plan 17's existing tests exercise
`StepwiseField` with contrived single-window setups. But any
real multi-window time-varying forcing is silently wrong, and
plan 17's emission-rate `StepwiseField` pathway has no working
sim-level clock source.

### Deliverable

The fix is smaller than v4's version because `sim.time` already
exists — we just need to expose it as `current_time` and update
the call site. Three steps, no unit decisions needed.

**Step 1: Add `current_time(::DrivenSimulation)` override.**

In `src/Models/DrivenSimulation.jl` (near the existing accessors
around line 258-260 where `window_index`, `substep_index`,
`current_qv` are defined):

```julia
"""
    current_time(sim::DrivenSimulation) -> FT

Simulation time [s] at the start of the next step. Advances by
`sim.Δt` at the end of each `step!(sim)` call.

Threaded to operators via `meteo = sim` (see `step!(sim)`).
Operators that consume time (`StepwiseField` emission rates,
future time-varying Kz fields) read `current_time(meteo)` and
get `sim.time`.
"""
current_time(sim::DrivenSimulation) = sim.time
```

Export from `AtmosTransport.jl` if not already.

**Step 2: Update `DrivenSimulation.step!` to pass `sim`, not `sim.driver`.**

At `src/Models/DrivenSimulation.jl:275`, change:

```julia
# Was:
step!(sim.model, sim.Δt; meteo = sim.driver)

# To:
step!(sim.model, sim.Δt; meteo = sim)
```

Update the surrounding comment (lines 270-274) to reflect the new
meteo contract:

```julia
# Plan 17 Commit 6 / Plan 18 A3: step!(model) runs the full operator
# suite in one call. `meteo = sim` threads `current_time(sim) = sim.time`
# so time-varying fields (e.g., StepwiseField emission rates) refresh
# from the simulation's running time. Operators that need driver
# capabilities access them via `meteo.driver`.
```

**Step 3: Add `Nothing` fallback for sim-less contexts.**

In `src/MetDrivers/AbstractMetDriver.jl` or near the existing stub
at line 85:

```julia
current_time(::Nothing) = 0.0
```

Keeps the existing `current_time(::AbstractMetDriver) = 0.0` stub
in place for backward compatibility with any caller that still
passes a driver directly. Add a docstring note that the canonical
pattern is `meteo = sim` for production and `meteo = nothing` for
unit tests that don't care about time:

```julia
"""
    current_time(meteo) -> Float64

... existing docstring ...

# Canonical usage

- Production: `meteo = sim::DrivenSimulation`; returns `sim.time`.
- Unit tests without a sim: `meteo = nothing`; returns `0.0`.

The `::AbstractMetDriver` stub returning `0.0` is retained for
backward compatibility but should not be relied upon — the driver
is stateless and cannot provide real time information.
"""
```

**Step 4: Add regression test.**

New file `test/test_current_time.jl` or additions to existing
`test_driven_simulation.jl`:

```julia
@testset "current_time(sim) advances with steps" begin
    # Build a minimal sim. Exact construction depends on test
    # fixtures; mirror existing test_driven_simulation.jl setup.
    sim = _make_minimal_sim(; FT = Float64)

    # At sim construction
    @test current_time(sim) ≈ 0.0

    # After one step
    step!(sim)
    @test current_time(sim) ≈ sim.Δt rtol = 1e-14

    # After many steps (up to but not across a window)
    n = min(10, sim.steps_per_window - 1)
    for _ in 1:n
        step!(sim)
    end
    @test current_time(sim) ≈ (1 + n) * sim.Δt rtol = 1e-14
end

@testset "current_time threads through step! as meteo = sim" begin
    # Install a StepwiseField emission rate with boundaries that
    # cross between windows. Run through the boundary; verify the
    # StepwiseField samples the right period on each side.
    # Contrast with pre-fix behavior (would sample the first period
    # forever because meteo = sim.driver returned 0.0).
end

@testset "current_time(nothing) returns 0.0" begin
    @test current_time(nothing) == 0.0
end

@testset "current_time(driver) returns 0.0 (legacy stub, deprecated)" begin
    # Retain existing stub behavior; document that production uses sim.
    driver = _make_minimal_driver()
    @test current_time(driver) == 0.0
end
```

### Acceptance criteria

1. `current_time(sim::DrivenSimulation)` returns `sim.time` and
   advances per step.

2. `DrivenSimulation.step!` passes `meteo = sim` (not
   `meteo = sim.driver`).

3. `DrivenSimulation` runs that use `StepwiseField` with multiple
   windows observe the correct sample per window (regression test
   Step 4).

4. No regression in shipped plan-17 tests — `ConstantField`-based
   tests continue to be bit-exact (they don't read time, only the
   sample). Plan 17's existing `StepwiseField` tests with single-
   window setups continue to pass.

5. `current_time(nothing) = 0.0` fallback works for tests without a
   sim.

6. Legacy `current_time(::AbstractMetDriver) = 0.0` stub retained
   for backward compatibility; test verifies behavior.

### Time estimate

~2 hours. Smaller than the original A3 (0.5-1 day) because
`sim.time` already exists and only needs to be exposed + threaded.

### What NOT to do

- Do NOT implement `current_time(::TransportBinaryDriver)` using a
  `current_window_index` field — that field does NOT exist on the
  driver. Verified against `src/MetDrivers/TransportBinaryDriver.jl:109-112`.
  The driver is stateless.

- Do NOT remove the `current_time(::AbstractMetDriver) = 0.0` stub.
  Some external callers may still pass a driver directly; retain
  backward compatibility.

- Do NOT pass `meteo = sim.driver` from any new code. The canonical
  pattern is `meteo = sim`. Operators that need driver capabilities
  access them via `meteo.driver`.

- Do NOT decide a unit convention — `sim.time` uses whatever unit
  `sim.Δt` already uses (seconds since run start, by existing sim
  convention at `DrivenSimulation.jl:26, 276`). No new unit contract.

---

## Combined acceptance

All three fixes shipped; verified by:

1. Full test suite passes: `julia --project=. test/runtests.jl`
   runs to completion without segfault, all testsets pass
   (modulo any preexisting `@test_skip` or `@test_broken`).

2. Face-indexed physics fully wired: a reduced-Gaussian run with
   non-trivial `diffusion_op` and `emissions_op` executes and
   conserves mass.

3. `current_time` non-trivial: a multi-window run with
   `StepwiseField` produces observably different sample values
   across windows (verifiable by logging or a round-trip test).

4. `test_dry_flux_interface.jl` runs twice in a row without
   segfault.

## Sequencing

Recommended order:
1. A2 first (trivial, unblocks reliable CI for the other two)
2. A1 next (unblocks reduced-Gaussian runtime; most code)
3. A3 last (depends on being able to run end-to-end tests
   reliably, which A1+A2 ensure)

Could parallelize A1 and A3 if two people are working. A2
must precede both because its absence makes local test runs
unreliable.

**Total estimated effort: 2-4 days.**

---

## What these fixes do NOT cover

- **Plan 18 itself.** Separate document (`18_CONVECTION_PLAN.md`
  + corrections addendum + adjoint addendum).

- **Other signature drift between structured and face-indexed
  paths.** If any exists beyond `apply!`, flag it but defer to
  a separate audit. This document scopes A1 to the specific
  `apply!` regression GPT-5.4 identified.

- **Performance regressions.** A1's face-indexed palindrome may
  be slower than the 4-sweep path for reduced-Gaussian runs
  without diffusion/emissions (more sweeps, more buffer ops).
  If that matters, file as follow-up optimization. Correctness
  first.

- **Cubed-sphere face-indexed support.** CubedSphereMesh is
  metadata-only in `src` (per StrangSplitting.jl:1028 error
  stub). Out of scope.

---

**End of pre-plan-18 fixes specification.**

Ship these three, verify the combined acceptance criteria, then
proceed to plan 18 Commit 0.
