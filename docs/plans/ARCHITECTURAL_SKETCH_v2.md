# Architectural Sketch — AtmosTransport.jl (v2)

**Status:** Reference document. Captures the architecture as of
plans 11-16b shipped.
**Updated:** 2026-04-19, post-plan-16b.
**Prior version:** v1 (written post-plan-14, covered advection only).

## 1. Purpose

This document sketches the shape of the AtmosTransport operator
suite after the plans 11-16b refactor. It's intended as a quick
orientation for:

- New contributors to understand the current structure
- Plan agents writing plans 17+ to see the foundation they build on
- Future reviewers deciding where new operators fit

For detailed rationale on specific design choices, see:

- `OPERATOR_COMPOSITION.md` — how operators compose in the
  step-level palindrome
- `TIME_VARYING_FIELD_MODEL.md` — the `AbstractTimeVaryingField`
  abstraction for input fields
- `REFERENCE_MODEL_COMPARISON.md` — how AtmosTransport relates to
  TM5, GEOS-Chem, IFS, FV3, ClimaAtmos, Oceananigans
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

**Module organization:**

`src/State/Fields/` is foundational. Depends only on `Base`.
Concrete types specific to a downstream operator (e.g., future
convective mass flux fields) may live in that operator's module.
`Fields` must not depend on `Grids`, `Mesh`, or any operator.

**Interface** (all concrete types implement):

```julia
field_value(f, idx::NTuple{N, Int}) → FT        # kernel-callable
update_field!(f, t::Real)                       # CPU-only, may cache
value_at(f, t::Real) → AbstractArray{FT, N}    # full spatial array
integral_between(f, t1, t2) → array             # for time integration
time_bounds(f) → (t_start, t_end)
```

**Pattern:** `update_field!(f, t)` does expensive work on CPU.
`field_value(f, idx)` is cheap, stateless, kernel-safe.

**GPU dispatch pattern:** concrete types with array storage use
`Adapt.adapt_structure(to, f)` to convert backing arrays at
kernel-launch time. `Vector{FT}` → `CuArray{FT, 1}`,
`Array{FT, 3}` → `CuArray{FT, 3}`, transparently. Plan 16b Commit 6
validated this for `ProfileKzField` and `PreComputedKzField`.

**Not yet shipped** (future plans):

- `StepwiseField{FT, N}` — piecewise-constant window averages
  (CATRINE-style monthly/annual inventories). Plan 17 scope.
- `LinearInterpolatedField{FT, N}` — linear between instantaneous
  snapshots. Future.
- `IntegralPreservingField{FT, N}` — smooth + preserves window
  integrals. Future.
- `ScaledField`, `MaskedField` — wrappers. Future.

---

## 4. Operator hierarchy

Three operator families, each with an abstract supertype and
concrete implementations. All conform to the same `apply!`
interface.

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
with palindrome center accepting an optional diffusion V call
(plan 16b Commit 4).

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

`NoDiffusion` is a literal dead branch on Julia dispatch — zero
floating-point work. Pre-16b behavior bit-exact preserved.

### 4.3 Chemistry

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

- Advection: `meteo = nothing` acceptable; takes flux state as a
  separate argument path
- Chemistry: `meteo = nothing` acceptable; pure decay ignores it
- Diffusion: requires real workspace (for `w_scratch`,
  `dz_scratch`); `meteo = nothing` OK for `ConstantField` / `ProfileKzField` /
  `PreComputedKzField` Kz, but NOT for `DerivedKzField` (needs
  surface fields via `field_value`)

**For column-level entry:**

Plan 16b Commit 4 also introduced `apply_vertical_diffusion!(q_raw, op, ws, dt)`
as an array-based lower-level entry point. This is called inside
the palindrome where `q_raw` is whichever ping-pong buffer
currently holds the tracer state — not necessarily
`state.tracers_raw`. The state-level `apply!` delegates to this
array-based entry.

---

## 6. Step-level composition

### 6.1 Current structure (plans 11-16b)

```
step!(model, dt)
├── transport_block:
│   └── strang_split!(state, fluxes, grid, advection_op, dt;
│                     workspace, diffusion_op = model.diffusion)
│       ├── X (dt/2)
│       ├── Y (dt/2)
│       ├── Z (dt/2)
│       ├── V (dt)  ← ImplicitVerticalDiffusion or NoDiffusion
│       ├── Z (dt/2)
│       ├── Y (dt/2)
│       └── X (dt/2)
└── chemistry_block:
    └── apply!(state, meteo, grid, model.chemistry, dt; workspace)
```

**Palindrome:** `X Y Z V Z Y X` with `V(dt)` as a single call at
center. `V` is linear with time-constant Kz, so one `V(dt)` call
equals the composition `V(dt/2) ∘ V(dt/2)` to leading order
(for Backward Euler, they agree to O((dt·D)²)).

### 6.2 Chemistry ordering workaround (plan 15 → plan 17)

Currently, `TransportModel.step!` runs `transport → chemistry`.
But the TM5 order when emissions enter is
`advection → emissions → chemistry`. Plan 15 resolved this with a
workaround: `DrivenSimulation` calls
`with_chemistry(model, NoChemistry())` on the wrapped model, then
applies chemistry at sim level after emissions.

Plan 17 resolves this cleanly by folding emissions into the
Z-sweep (surface flux BC), after which
`TransportModel.step!`'s `transport → chemistry` becomes the
right order without workarounds.

### 6.3 Future palindrome expansion (plans 17-18)

```
Plan 17: X Y Z V S V Z Y X       # emissions S at palindrome center,
                                 # wrapped by two V half-steps
Plan 18: X Y Z V C S C V Z Y X   # convection C similarly wrapped
```

See `OPERATOR_COMPOSITION.md` §3 for full palindrome rationale.

---

## 7. TransportModel and DrivenSimulation

`TransportModel` carries all three operator families:

```julia
struct TransportModel{Adv, Diff, ChemT, ...}
    advection::Adv
    diffusion::Diff
    chemistry::ChemT
    # ... state, fluxes, workspace, grid
end
```

Constructors accept kwargs with sensible defaults:

```julia
TransportModel(; advection = SlopesScheme(...),
                 diffusion = NoDiffusion(),
                 chemistry = NoChemistry(), ...)
```

Helper constructors:

- `with_chemistry(model, chem)` — swap only chemistry
- `with_diffusion(model, diff)` — swap only diffusion
- `with_advection(model, adv)` — swap only advection

All defaults preserve pre-refactor behavior bit-exactly.

`DrivenSimulation` wraps a `TransportModel` and runs it against
time-varying meteorology. Currently uses the plan-15 chemistry
workaround (strip chemistry from model, apply at sim level).

---

## 8. Workspace

`AdvectionWorkspace` carries all scratch storage across operators.
Plan 11 introduced ping-pong 4D buffers; plan 14 finalized shapes;
plan 16b added diffusion scratch.

Current fields:

- `rm_4d_A`, `rm_4d_B` — ping-pong tracer buffers (plan 11)
- `m_A`, `m_B` — ping-pong mass buffers (plan 11)
- `am`, `bm`, `cm` — ping-pong flux buffers (plan 13)
- `w_scratch::Array{FT, 3}` — diffusion Thomas forward-elimination
  factor (plan 16b)
- `dz_scratch::Array{FT, 3}` — layer thickness input (caller-owned;
  plan 16b Commit 5 defers the hydrostatic filler)

Workspace is grid-sized at construction. On face-indexed grids
(reduced Gaussian), the 4D and 3D buffers are 0-sized since those
paths use `selectdim` views.

---

## 9. Testing discipline

**Test contract** (plan 14 lesson, codified): observe post-operator
state through `get_tracer(state, name)` or `state.tracers.name`,
NEVER through caller-provided input arrays. Plan 14's 4D refactor
broke helpers that cached `caller_array === state.tracers.name`;
accessor-API observations survive.

**CPU/GPU dispatch**: `parent(arr) isa Array`, NOT `arr isa Array`.
Views of 4D `tracers_raw` are `SubArray`, not `Array`. Bare
`isa Array` misroutes to GPU path.

**Baseline failure count:** 77 pre-existing failures across 7 test
files, stable since plan 12. Every plan preserves this count.
Known distribution:

- `test_basis_explicit_core.jl`: 2 (metadata-only CubedSphere API)
- `test_structured_mesh_metadata.jl`: 3 (panel conventions)
- `test_poisson_balance.jl`: 72 (mirror_sign inflow/outflow)

Plans that would fix any of these need a separate scope.

---

## 10. Deferred work (visible in code)

Several items are deferred through plan 16b with clear resumption
triggers:

- **Threading `current_time(meteo)` through `apply!`.** Accessor
  exists (plan 16b Commit 5) but operators still use `t = zero(FT)`.
  Lands when first end-to-end `DerivedKzField` test needs real time.
- **`dz_scratch` operational filler.** Caller-owned; operational
  hydrostatic `delp → dz` helper lands with DrivenSimulation
  meteorology integration.
- **Adjoint kernel.** Tridiagonal structure preserved (plan 16b
  Commit 2); adjoint port is mechanical when needed.
- **Non-local (counter-gradient) PBL diffusion.** Legacy has
  `NonLocalPBLDiffusion` (Holtslag-Boville). Sibling concrete type
  to `ImplicitVerticalDiffusion` when needed.
- **Diffusion perf tuning.** Soft target exceeded at large grids
  (65-76% overhead vs 30% target). Three optimization paths noted:
  multi-tracer fusion in kernel, shared-memory Thomas, persistent
  `w_scratch`. All within-kernel; interface unchanged.

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
│       └── DerivedKzField.jl    # Beljaars-Viterbo
├── Operators/
│   ├── Operators.jl             # include order: Diffusion → Advection → Chemistry
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
│   │   └── operators.jl         # ImplicitVerticalDiffusion, apply!
│   └── Chemistry/
│       ├── Chemistry.jl
│       ├── chemistry_kernels.jl
│       └── operators.jl         # ExponentialDecay, CompositeChemistry
├── Models/
│   ├── TransportModel.jl        # diffusion + advection + chemistry
│   └── DrivenSimulation.jl      # meteorology-driven runner
└── MetDrivers/
    └── AbstractMetDriver.jl     # current_time stub (plan 16b Commit 5)
```

---

## 12. Version history

**v1** (post-plan-14): described advection refactor only. Storage
model (4D tracers_raw), scheme hierarchy (Upwind/Slopes/PPM),
basic operator interface sketch.

**v2** (this document, post-plan-16b): extended to cover chemistry
(plan 15), `TimeVaryingField` abstraction (plan 16a), vertical
diffusion (plan 16b), palindrome structure. Full file-level map.
Testing discipline formalized.

**v3** (future, post-plan-18): will add surface emissions (plan 17)
and convection (plan 18). May revise palindrome rationale if
ordering study reveals physics-level issues.

---

**End of document.**
