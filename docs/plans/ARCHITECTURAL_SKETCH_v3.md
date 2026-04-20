# Architectural Sketch — AtmosTransport.jl (v3)

**Status:** Reference document. Captures the architecture as of
plans 11-17 shipped.
**Updated:** 2026-04-19, post-plan-17.
**Prior versions:** v1 (post-plan-14, advection only), v2 (post-plan-16b,
added chemistry + diffusion + TimeVaryingField).

## 1. Purpose

This document sketches the shape of the AtmosTransport operator
suite after the plans 11-17 refactor. It's intended as a quick
orientation for:

- New contributors to understand the current structure
- Plan agents writing plans 18+ to see the foundation they build on
- Future reviewers deciding where new operators fit

For detailed rationale on specific design choices, see:

- `OPERATOR_COMPOSITION.md` — how operators compose in the
  step-level palindrome
- `TIME_VARYING_FIELD_MODEL.md` — the `AbstractTimeVaryingField`
  abstraction for input fields
- `REFERENCE_MODEL_COMPARISON.md` — how AtmosTransport relates to
  TM5, GEOS-Chem, IFS, FV3, ClimaAtmos, Oceananigans
- `17_SURFACE_EMISSIONS_PLAN/ordering_study_results.md` — measured
  V/S palindrome ordering comparison
- Plan NOTES files for per-plan retrospective findings

---

## 2. Storage model

`CellState` holds all tracer data in a single 4D array:

```julia
struct CellState{FT, ...}
    air_mass::Array{FT, 3}              # (Nx, Ny, Nz)
    tracers_raw::Array{FT, 4}           # (Nx, Ny, Nz, Nt)
    tracer_names::NTuple{Nt, Symbol}    # type-stable, compile-time Nt
end
```

**Access patterns:**

- Kernels: dispatch on `state.tracers_raw` directly
- Non-kernel code: uses the accessor API (storage-agnostic)
- Property-style access preserved via `getproperty`: `state.tracers.CO2`
  returns a view through `TracerAccessor`

**Accessor API** (in `src/State/Tracers.jl`):

```julia
ntracers(state)                   # Int
tracer_index(state, name::Symbol) # Int or nothing
tracer_name(state, idx::Int)      # Symbol
get_tracer(state, name)           # view, by name or index
eachtracer(state)                 # iterator over (name, view) pairs
```

**Julia memory-order note:** `(Nx, Ny, Nz, Nt)` means i is
fastest-varying (column-major). Spatial access `Q[:, j, k, t]` is
contiguous; per-tracer 3D views `view(Q, :, :, :, t)` are
contiguous blocks. Matches structured-kernel coalesced-access.

**Why this shape:** plan 14 validated with measured 10-270% GPU
speedup at production settings (A100 F64, Nt=30). Multi-tracer
kernels read/write mass and flux fields once across all tracers,
eliminating the per-tracer loop's redundant bandwidth.

---

## 3. Time-varying input fields

`AbstractTimeVaryingField{FT, N}` is the abstraction for any input
data that varies in time. Concrete types encode HOW the field is
obtained; operators consume through a uniform interface.

**Shipped concrete types:**

| Type | Rank | Shipped in | Purpose |
|---|---|---|---|
| `ConstantField{FT, N}` | any | 16a | scalar at any rank; chemistry rates, test cases |
| `ProfileKzField{FT, V}` | 3 | 16b | per-level `AbstractVector{FT}`, horizontally invariant |
| `PreComputedKzField{FT, A}` | 3 | 16b | 3D array wrapper; caller owns storage |
| `DerivedKzField{FT, ...}` | 3 | 16b | Beljaars-Viterbo from surface fields |
| `StepwiseField{FT, N, A, B, W}` | any | **17** | piecewise-constant in time; CATRINE monthly inventories |

**Module organization:**

`src/State/Fields/` is foundational. Depends only on `Base` + `Adapt`.
Concrete types specific to a downstream operator (e.g. future
convective mass flux fields) may live in that operator's module.
`Fields` must not depend on `Grids`, `Mesh`, or any operator.

**Interface** (all concrete types implement):

```julia
field_value(f, idx::NTuple{N, Int}) → FT        # kernel-callable
update_field!(f, t::Real)                       # CPU-only, may cache
```

Optional (only `StepwiseField` implements, as of plan 17):

```julia
integral_between(f, t1, t2, idx) → FT           # time integral
```

**Pattern:** `update_field!(f, t)` does expensive work on CPU.
`field_value(f, idx)` is cheap, stateless, kernel-safe.

**GPU dispatch pattern:** concrete types with array storage use
`Adapt.adapt_structure(to, f)` to convert backing arrays at
kernel-launch time. `Vector{FT}` → `CuArray{FT, 1}`,
`Array{FT, 3}` → `CuArray{FT, 3}`, transparently. Plan 16b Commit 6
validated this for `ProfileKzField` and `PreComputedKzField`; plan
17 Commit 1 validated for `StepwiseField` including the
`Val(:unchecked)` inner-constructor bypass for host-only validation.

**Mutable scalar pattern** (new in plan 17): kernel-facing structs
that need a mutable integer (e.g., `StepwiseField.current_window`)
store it as a 1-element `Vector{Int}` rather than `Base.RefValue{Int}`,
because `Adapt` converts `Vector` → `CuArray{Int, 1}` cleanly and
kernels can read the index as a device-local memory access.

---

## 4. Operator hierarchy

Four operator families as of plan 17, each with an abstract
supertype and concrete implementations. All conform to the same
`apply!` interface.

### 4.1 Advection

```
AbstractAdvectionScheme
├── UpwindScheme
├── SlopesScheme{L}           # L = limiter type
└── PPMScheme{L, ORD}         # ORD ∈ {4, 5, 6, 7}
```

Location: `src/Operators/Advection/`.

Structured via `strang_split!` → `strang_split_mt!` (multi-tracer)
on the 4D `tracers_raw` array. Six sweeps per step (X Y Z Z Y X)
with palindrome center accepting optional diffusion V call
(plan 16b Commit 4) + optional surface flux S call
(**plan 17 Commit 5**).

### 4.2 Diffusion

```
AbstractDiffusionOperator
├── NoDiffusion                        # no-op (default)
└── ImplicitVerticalDiffusion{FT, KzF} # backward Euler, Thomas solve
```

Location: `src/Operators/Diffusion/`.

Column-implicit Thomas solve with caller-supplied `w_scratch` and
`dz_scratch` via `AdvectionWorkspace`. Coefficients `(a, b, c, d)`
as named locals per level (not pre-factored), preserving
transposability for future adjoint port.

Array-level entry point `apply_vertical_diffusion!(q_raw, op, ws,
dt, meteo = nothing)` — plan 17 Commit 4 added the `meteo` trailing
argument for `current_time` threading.

### 4.3 Surface flux (plan 17 — new in v3)

```
AbstractSurfaceFluxOperator
├── NoSurfaceFlux                       # no-op (default)
└── SurfaceFluxOperator{M}              # wraps a PerTracerFluxMap
```

Location: `src/Operators/SurfaceFlux/`.

Applies `rm[i, j, Nz, tracer_idx] += rate[i, j] × dt` at the surface
layer (`k = Nz`), one KA kernel launch per emitting tracer.
`SurfaceFluxSource{RateT}` carries per-tracer `cell_mass_rate` in
**kg/s per cell** (already area-integrated; see plan 17 Decision 1).
`PerTracerFluxMap{S <: Tuple}` is an NTuple-backed map of sources
keyed by `tracer_name`; `flux_for(map, :name)` returns the source
or `nothing`.

Array-level entry point `apply_surface_flux!(q_raw, op, ws, dt,
meteo, grid; tracer_names)` — used by the palindrome on a raw
ping-pong buffer.

### 4.4 Chemistry

```
AbstractChemistryOperator
├── NoChemistry                          # no-op (default)
├── ExponentialDecay{FT, N, R}           # pointwise Q *= exp(-k·dt)
└── CompositeChemistry                   # sequential composition
```

Location: `src/Operators/Chemistry/`.

Plan 15 established the operator interface with chemistry as the
first non-advection operator. Decay rates are
`NTuple{N, AbstractTimeVaryingField{FT, 0}}` (retrofitted in 16a
Commit 2 from raw scalars).

---

## 5. Operator interface

All operators conform to:

```julia
apply!(state::CellState,
       meteo,
       grid::AbstractGrid,
       op,
       dt::Real;
       workspace) -> state
```

**Caveats by operator family:**

- Advection: takes flux state as a separate argument path; consumes
  `diffusion_op`, `emissions_op`, `meteo` kwargs for palindrome hooks
- Chemistry: `meteo = nothing` acceptable; pure decay ignores it
- Diffusion: requires real workspace (for `w_scratch`,
  `dz_scratch`); `meteo = nothing` OK for `ConstantField` / `ProfileKzField` /
  `PreComputedKzField` Kz, but NOT for `DerivedKzField`
- Surface flux: `workspace` unused; `meteo` used to thread
  `current_time` for time-varying flux fields (plan 17 wired
  through; `cell_mass_rate` is currently a static array, so time
  doesn't yet matter operationally)

**For column-level / array-level entry:**

Plan 16b + plan 17 introduced array-level entry points that
operate on a raw 4D tracer buffer rather than a `CellState`:

- `apply_vertical_diffusion!(q_raw, op, ws, dt, meteo = nothing)`
- `apply_surface_flux!(q_raw, op, ws, dt, meteo, grid; tracer_names)`

These are called inside the palindrome where `q_raw` is whichever
ping-pong buffer currently holds the tracer state — not necessarily
`state.tracers_raw`. The state-level `apply!` delegates to the
array-based entry.

---

## 6. Step-level composition

### 6.1 Current structure (plans 11-17)

```
step!(model, dt; meteo = nothing)
├── transport_block:
│   └── apply!(state, fluxes, grid, advection, dt;
│              workspace, diffusion_op, emissions_op, meteo)
│       └── strang_split_mt!(...)
│           ├── X (dt/2)
│           ├── Y (dt/2)
│           ├── Z (dt/2)
│           ├── palindrome-center dispatch:
│           │   if emissions_op isa NoSurfaceFlux
│           │       V(dt)                             ← bit-exact plan 16b
│           │   else
│           │       V(dt/2) → S(dt) → V(dt/2)         ← plan 17
│           ├── Z (dt/2)
│           ├── Y (dt/2)
│           └── X (dt/2)
└── chemistry_block:
    └── chemistry_block!(state, meteo, grid, chemistry, dt)
```

**Palindrome** (plan 17 Commit 5):

- Default (`emissions_op = NoSurfaceFlux()`, `diffusion_op = NoDiffusion()`)
  → every component is a dead branch; result bit-exact equivalent
  to pre-refactor advection.
- Default with active V only → `X Y Z V(dt) Z Y X` (plan 16b,
  unchanged).
- Active emissions → `X Y Z V(dt/2) S(dt) V(dt/2) Z Y X`
  (plan 17). Backward-Euler V gives V(dt/2) ∘ V(dt/2) ≠ V(dt) to
  O((dt·D)²), so results differ from the single-V(dt) path at that
  order; confirmed by the plan 17 ordering study.

### 6.2 Chemistry ordering workaround (plan 15 → resolved plan 17)

Plan 15 held chemistry at the sim level to preserve the TM5 order
`advection → emissions → chemistry` while emissions still lived
outside the palindrome. Plan 17 Commit 5 integrated S into the
palindrome, so `DrivenSimulation` now delegates entirely to
`step!(model)` — no sim-level chemistry hack, no post-step
`_apply_surface_sources!` call. The order is preserved naturally
because S is the last composed operator in the transport block
and chemistry runs after.

### 6.3 Future palindrome expansion (plan 18)

```
Plan 18: X Y Z V C S C V Z Y X   # convection C wrapped around S
```

Plan 17's ordering study recommended arrangement A (symmetric)
based on 2nd-order Strang accuracy; plan 18 convection C should
compose cleanly with A. See
`17_SURFACE_EMISSIONS_PLAN/ordering_study_results.md` §4 for the
compositional argument.

---

## 7. TransportModel and DrivenSimulation

`TransportModel` carries four operator families (plan 17 added
`emissions`):

```julia
struct TransportModel{StateT, FluxT, GridT, SchemeT, WorkspaceT,
                      ChemT, DiffT, EmT}
    state     :: StateT
    fluxes    :: FluxT
    grid      :: GridT
    advection :: SchemeT
    workspace :: WorkspaceT
    chemistry :: ChemT
    diffusion :: DiffT
    emissions :: EmT         # plan 17 — NoSurfaceFlux() by default
end
```

Constructors accept kwargs with sensible defaults:

```julia
TransportModel(state, fluxes, grid, advection;
               workspace  = AdvectionWorkspace(state),
               chemistry  = NoChemistry(),
               diffusion  = NoDiffusion(),
               emissions  = NoSurfaceFlux())
```

Helper constructors:

- `with_chemistry(model, chem)` — swap only chemistry
- `with_diffusion(model, diff)` — swap only diffusion
- `with_emissions(model, em)` — swap only emissions (plan 17)

All defaults preserve pre-refactor behavior bit-exactly.

`DrivenSimulation` wraps a `TransportModel` and runs it against
time-varying meteorology. Post plan 17 Commit 6 the plan-15
chemistry workaround is removed; the sim installs
`surface_sources` into the wrapped model via `with_emissions` and
delegates `step!(sim)` entirely to `step!(model; meteo = sim.driver)`.

---

## 8. Workspace

`AdvectionWorkspace` carries all scratch storage across operators.
Plan 11 introduced ping-pong 4D buffers; plan 14 finalized shapes;
plan 16b added diffusion scratch. Plan 17 adds no new workspace
fields — the surface-flux kernel needs no scratch beyond reading
the per-source `cell_mass_rate` arrays that live on the operator.

Current fields:

- `rm_4d_A`, `rm_4d_B` — ping-pong tracer buffers (plan 11)
- `m_A`, `m_B` — ping-pong mass buffers (plan 11)
- `am`, `bm`, `cm` — ping-pong flux buffers (plan 13)
- `w_scratch::Array{FT, 3}` — diffusion Thomas forward-elimination
  factor (plan 16b)
- `dz_scratch::Array{FT, 3}` — layer thickness input (caller-owned;
  hydrostatic filler is still plan 17-deferred)

Workspace is grid-sized at construction. On face-indexed grids
(reduced Gaussian), the 4D and 3D buffers are 0-sized since those
paths use `selectdim` views.

---

## 9. Testing discipline

Codified in top-level `CLAUDE.md §Testing`. Plan 17 adds no new
rules — follows the established contract:

- Observe post-operator state via `get_tracer(state, name)` or
  `state.tracers.name`, never through input arrays.
- Default-kwarg paths must be `==` bit-exact to the
  explicit-no-op path (plan 17 test_emissions_palindrome's first
  testset is the pattern for plan 18+).
- 77 pre-existing failures baseline (2 `test_basis_explicit_core`
  + 3 `test_structured_mesh_metadata` + 72 `test_poisson_balance`)
  preserved. Plan 17 shipped with 0 regressions.

Cumulative new tests plan 17: 152 (43 StepwiseField + 36
PerTracerFluxMap + 38 SurfaceFluxOperator + 12 emissions_palindrome
+ 14 TransportModel_emissions + 9 ordering_study).

---

## 10. Deferred work (visible in code)

Several items remain deferred through plan 17 with clear resumption
triggers:

- **`dz_scratch` operational filler.** Caller-owned; operational
  hydrostatic `delp → dz` helper lands with DrivenSimulation
  meteorology integration.
- **Adjoint kernel.** Tridiagonal structure preserved (plan 16b
  Commit 2); adjoint port is mechanical when needed.
- **Non-local (counter-gradient) PBL diffusion.** Legacy has
  `NonLocalPBLDiffusion` (Holtslag-Boville). Sibling concrete type
  to `ImplicitVerticalDiffusion` when needed.
- **Diffusion perf tuning.** Soft target exceeded at large grids
  (65-76% overhead vs 30% target).
- **Per-area flux variant** for surface emissions
  (`kg/m²/s × cell_area`). Plan 17 kept kg/s-per-cell; add as
  sibling when an inventory path needs it.
- **Stack emissions** (3D source fields, tall stacks / aviation /
  volcanic).
- **Dedicated `DepositionOperator`.** Plan 17 covers dry deposition
  as negative flux in `SurfaceFluxOperator`; a first-class operator
  is a follow-up plan.
- **`AbstractLayerOrdering{TopDown, BottomUp}`** — cross-cutting
  refactor so operators can dispatch on vertical-layer convention
  rather than assuming `k = Nz` surface. Motivated by
  GEOS-FP vs GEOS-IT conventions; touches every vertically-indexed
  kernel.
- **Remove dead `_inject_source_kernel!`** at
  `src/Kernels/CellKernels.jl:43` (orphaned since pre-plan-17).

---

## 11. File-level map

```
src/
├── AtmosTransport.jl            # top-level module, re-exports
├── State/
│   ├── CellState.jl             # 4D storage, getproperty, TracerAccessor
│   ├── Tracers.jl               # accessor API (ntracers, get_tracer, ...)
│   └── Fields/
│       ├── Fields.jl            # module, AbstractTimeVaryingField
│       ├── ConstantField.jl     # scalar at any rank
│       ├── ProfileKzField.jl    # vertical profile (rank-3)
│       ├── PreComputedKzField.jl
│       ├── DerivedKzField.jl    # Beljaars-Viterbo
│       └── StepwiseField.jl     # piecewise-constant in time (plan 17)
├── Operators/
│   ├── Operators.jl             # include order: Diffusion → SurfaceFlux → Advection → Chemistry
│   ├── Advection/
│   │   ├── Advection.jl
│   │   ├── StrangSplitting.jl   # strang_split!, strang_split_mt! (palindrome)
│   │   ├── structured_kernels.jl
│   │   ├── schemes.jl           # UpwindScheme, SlopesScheme, PPMScheme
│   │   └── ppm_subgrid_distributions.jl
│   ├── Diffusion/
│   │   ├── Diffusion.jl
│   │   ├── thomas_solve.jl      # solve_tridiagonal!, build_diffusion_coefficients
│   │   ├── diffusion_kernels.jl # _vertical_diffusion_kernel!
│   │   └── operators.jl         # ImplicitVerticalDiffusion, apply!, apply_vertical_diffusion!
│   ├── SurfaceFlux/             # ──────────── NEW in plan 17 ─────────
│   │   ├── SurfaceFlux.jl
│   │   ├── sources.jl           # SurfaceFluxSource + migrated helpers
│   │   ├── PerTracerFluxMap.jl
│   │   ├── surface_flux_kernels.jl  # _surface_flux_kernel!
│   │   └── operators.jl         # SurfaceFluxOperator, apply!, apply_surface_flux!
│   └── Chemistry/
│       ├── Chemistry.jl
│       ├── chemistry_kernels.jl
│       └── operators.jl         # ExponentialDecay, CompositeChemistry
├── Models/
│   ├── TransportModel.jl        # + emissions field (plan 17)
│   └── DrivenSimulation.jl      # plan-15 chem workaround REMOVED (plan 17)
└── MetDrivers/
    └── AbstractMetDriver.jl     # current_time stub — threaded through operators (plan 17 Commit 4)
```

---

## 12. Version history

**v1** (post-plan-14): described advection refactor only. Storage
model (4D tracers_raw), scheme hierarchy (Upwind/Slopes/PPM),
basic operator interface sketch.

**v2** (post-plan-16b): extended to cover chemistry (plan 15),
`TimeVaryingField` abstraction (plan 16a), vertical diffusion
(plan 16b), palindrome structure. Full file-level map. Testing
discipline formalized.

**v3** (this document, post-plan-17): added surface emissions
(plan 17), `StepwiseField` concrete type, `SurfaceFluxOperator` +
`PerTracerFluxMap`, palindrome `V(dt/2) S V(dt/2)` at center when
emissions active. Plan-15 chemistry workaround resolved.
`current_time(meteo)` threading finalized. Ordering-study-based
recommendation for V/S arrangement.

**v4** (future, post-plan-18): will add convection C wrapped around
the plan-17 `S` position. May revise palindrome rationale if a
real-meteorology convection scheme reveals ordering effects not
visible in the plan 17 ordering study.

---

**End of document.**
