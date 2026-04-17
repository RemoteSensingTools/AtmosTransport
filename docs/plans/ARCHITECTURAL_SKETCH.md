# Target End State — Architectural Sketch

*A map of where we're going, so each refactor can check its direction.*

This document describes what the advection subsystem SHOULD look like
after the full sequence of refactors ships:

1. **Ping-pong buffers** (`11_PINGPONG_REFACTOR_PLAN.md`)
2. **Scheme type consolidation** (`12_SCHEME_CONSOLIDATION_PLAN.md`)
3. **Sync removal + CFL pilot unification + rename** (`13_SYNC_AND_CLEANUP_PLAN.md`)
4. **Single advection pipeline** (`14_SINGLE_PIPELINE_PLAN.md`)

It's not an implementation plan — it's the north star each plan aims
at. When executing any of the four refactors, check your diff against
this sketch to make sure you're converging toward it, not drifting.

---

# What a New Contributor Sees on Day 1 (Target)

Someone opens `src/Operators/Advection/` for the first time. They
read CLAUDE.md's pointer that says "this is where advection lives."
What should they find?

## Directory structure (target)

```
src/Operators/Advection/
├── Advection.jl                     — module entry, re-exports
├── schemes.jl                       — scheme type hierarchy (single)
├── limiters.jl                      — branchless @inline limiters
├── reconstruction.jl                — @inline face-flux functions
├── ppm_subgrid_distributions.jl     — PPM ORD=4,5,6,7 helpers
├── MassCFLPilot.jl                  — ONE max_cfl algorithm per direction
├── structured_kernels.jl            — @kernel shells for structured grids
├── StrangSplitting.jl               — orchestrator: ONE strang_split!, ONE apply!
├── LinRood.jl                       — cubed-sphere cross-term path (unchanged)
└── CubedSphereStrang.jl             — cubed-sphere Strang dispatch (unchanged)
```

### What's NOT in the target structure

- **No `Upwind.jl`** — `UpwindAdvection` legacy type is deleted; only `UpwindScheme` remains
- **No `RussellLerner.jl`** — `RussellLernerAdvection` legacy type is deleted; only `SlopesScheme` remains
- **No `multitracer_kernels.jl`** — merged into `structured_kernels.jl` (one set of shells, parameterized on tracer count)
- **No `HaloExchange.jl` in advection/** — if it exists, moved to a shared location, since halos aren't advection-specific

## Module surface (target)

What `using AtmosTransport` exposes at the advection level:

```julia
# Scheme types (ONE hierarchy)
AbstractAdvectionScheme
UpwindScheme
SlopesScheme{L}           # L is a limiter
PPMScheme{L}

# Limiters
AbstractLimiter
NoLimiter, MonotoneLimiter, PositivityLimiter

# Orchestration
AdvectionWorkspace        # ONE struct, clean names (rm_A/rm_B/m_A/m_B)
strang_split!             # ONE function, handles multi-tracer natively
apply!                    # ONE dispatch path per grid topology

# Utilities
max_cfl_x, max_cfl_y, max_cfl_z
```

No `UpwindAdvection`. No `RussellLernerAdvection`. No
`FirstOrderUpwindAdvection`. No `strang_split_mt!`. The set of
public names is smaller and each name has an unambiguous meaning.

## The call chain for a structured-grid advection step (target)

```
User code:
  apply!(state, fluxes, grid, scheme, dt; workspace=ws)
         │
         ▼
StrangSplitting.apply!    (dispatches on grid type: LatLon)
         │
         ▼
strang_split!             (the ONLY entry point)
         │
         ├─ CFL pilot (ONE algorithm, works on CPU and GPU):
         │    n_x = _cfl_pass_count(am, m, cfl_limit)
         │    n_y = _cfl_pass_count(bm, m, cfl_limit)
         │    n_z = _cfl_pass_count(cm, m, cfl_limit)
         │
         ├─ Pack tracers into 4D view (zero-copy via 4D layout in CellState)
         │
         ├─ Six-sweep palindrome, ping-pong buffers:
         │    sweep_x!(rm4d_A, rm4d_B, m_A, m_B, am, scheme, ws)  # × n_x
         │    sweep_y!(rm4d_B, rm4d_A, m_B, m_A, bm, scheme, ws)  # × n_y
         │    sweep_z!(...)                                      # × n_z
         │    sweep_z!(...)                                      # × n_z
         │    sweep_y!(...)                                      # × n_y
         │    sweep_x!(...)                                      # × n_x
         │    (swap buffer parity after each; final result in A
         │     if total parity is even, else B. Copy back if needed.)
         │
         └─ ONE synchronize(backend) at the end, if caller expects
            results visible immediately
```

That's the entire flow. No per-tracer outer loop. No `m_save`. No
`copyto!` between sweeps. No legacy sweep variants. No dual CFL
algorithm. No sync after every sweep.

## Data structures (target)

### `CellState{Basis, A}` — simplified

```julia
struct CellState{Basis <: AbstractMassBasis, A <: AbstractArray}
    air_mass    :: A                                 # 3D (Nx,Ny,Nz)
    tracers_data:: AbstractArray{eltype(A), 4}       # 4D (Nx,Ny,Nz,Nt)
    tracer_names:: NTuple{N, Symbol} where N
end
```

Construction (user-facing API unchanged):
```julia
state = CellState(m; CO2=rm_co2, SF6=rm_sf6)
# Internally packs into tracers_data, remembers names.
```

Access (user-facing API unchanged via `getproperty`):
```julia
state.tracers        # returns a lazy NamedTuple-like object with views
state.tracers.CO2    # returns view(state.tracers_data, :, :, :, 1)
pairs(state.tracers) # iterator of (name, view) pairs
```

The key difference from today: `tracers` is NOT a NamedTuple of
`Array{FT,3}` but a 4D `Array{FT,4}` with a name lookup. All
existing caller code works unchanged thanks to `getproperty`
returning views.

### `AdvectionWorkspace{FT, A, A4}` — clean names

```julia
struct AdvectionWorkspace{FT, A <: AbstractArray{FT,3},
                          A4 <: AbstractArray{FT,4},
                          V <: AbstractVector{Int32}}
    # Ping-pong pair for 3D air mass
    m_A           :: A
    m_B           :: A
    # Ping-pong pair for 4D tracer mass
    rm_A          :: A4
    rm_B          :: A4
    # CFL pilot scratch (separate from ping-pong)
    cfl_scratch   :: A
    # Grid metadata
    cluster_sizes :: V       # reduced-grid clustering (all-ones for uniform)
    face_left     :: V       # face connectivity (reduced Gaussian)
    face_right    :: V
end
```

No more `rm_buf`. No more `m_buf`. No more `m_save`. No more
conditional 4D buffer. The 4D buffers are always allocated because
multi-tracer is the only path.

Memory impact at C180 × 72 × 30 tracers Float64:
- `m_A` + `m_B`: 2 × 112 MB = 224 MB
- `rm_A` + `rm_B`: 2 × 3.4 GB = 6.8 GB
- `cfl_scratch`: 112 MB
- Total workspace: ~7.2 GB

That's about 2× today's allocation. The doubling buys
- Ping-pong → 30-40% per-step speedup
- Single path → clean codebase
- Elimination of the per-tracer loop overhead

If 7 GB workspace is too much at full scale, the answer is batching
(advect N tracers at a time, reuse 4D buffers) — which is a later
optimization (tier-3).

### `StructuredFaceFluxState{Basis}` — unchanged

No change planned. Still holds `am, bm, cm`.

## Module surface: function count (target)

Approximate line counts:

| File                         | Today   | Target | Notes                                                  |
|------------------------------|---------|--------|--------------------------------------------------------|
| StrangSplitting.jl           | 1413    | ~600   | Delete legacy sweeps, dual CFL, per-tracer loop        |
| structured_kernels.jl        | 157     | ~200   | Absorbs multitracer_kernels via tracer count dispatch  |
| multitracer_kernels.jl       | 147     | 0      | Merged into structured_kernels.jl                      |
| reconstruction.jl            | 753     | 753    | No change — this is the hot path, already clean        |
| limiters.jl                  | 211     | 211    | No change                                              |
| schemes.jl                   | 285     | 285    | No change (the hierarchy was already modern)           |
| Upwind.jl                    | 120     | 0      | Legacy type deleted                                    |
| RussellLerner.jl             | 299     | 0      | Legacy type deleted                                    |
| MassCFLPilot.jl              | 77      | 77     | ONE algorithm lives here, used by both CPU and GPU     |
| LinRood.jl                   | 856     | 856    | Not in scope                                           |
| CubedSphereStrang.jl         | ~600    | ~600   | Not in scope                                           |
| ppm_subgrid_distributions.jl | 203     | 203    | No change                                              |
| **Total advection/**         | ~5100   | ~3800  | ~25% reduction, most in StrangSplitting.jl             |

The biggest win is `StrangSplitting.jl` going from 1,413 lines to
around 600. Most of what's removed is duplication: two sweep
dispatch systems (legacy + new), two CFL algorithms (CPU + GPU),
two Strang orchestrators (per-tracer + fused).

## Invariants (target)

These must hold after the refactors. They're what makes the code
correct and makes future contributors confident:

1. **Branchless on GPU hot paths.** Every `@inline` function called
   from a `@kernel` uses `ifelse`, never `if/else`. Unchanged from today.

2. **Ping-pong buffer contract.** Kernel writes to destination, reads
   from source. Source and destination are different arrays. No
   `copyto!` between sweeps.

3. **Mass conservation is bit-identical.** Total mass preserved to
   machine precision per step, per tracer. Tested at ≤1e-12 for
   Float64, ≤5e-5 for Float32.

4. **CPU and GPU agree to ≤4 ULP (1-step), ≤16 ULP (4-step).**
   Unchanged. Floating-point FMA differences only.

5. **Multi-tracer fused = per-tracer sequential, bit-identical.**
   After single-pipeline refactor, this becomes a self-test of
   the kernel only (the per-tracer path no longer exists, but the
   kernel's arithmetic is unchanged; a golden regression file can
   document expected values).

6. **One scheme type per reconstruction order.** `UpwindScheme` for
   constant, `SlopesScheme{L}` for linear, `PPMScheme{L}` for
   quadratic. No legacy aliases.

7. **ONE entry point per grid topology.** `apply!` dispatches on
   `grid::AtmosGrid{<:LatLonMesh}`, `AtmosGrid{<:CubedSphereMesh}`,
   `AtmosGrid{<:AbstractHorizontalMesh}` — and each dispatches to
   a single well-defined function (no parallel "legacy path").

## What this sketch does NOT cover (out of scope for these four refactors)

- **Cubed-sphere integration with scheme dispatch.** Today LinRood.jl
  is entirely separate from the structured path. Unifying them is
  tier-3 work — genuinely hard because cross-term averaging is
  topology-specific. Not in this four-plan sequence.

- **Face-indexed (reduced Gaussian) support for SlopesScheme /
  PPMScheme.** Only UpwindScheme works for face-indexed today. Tier-3.

- **Float32 production path.** Validated Float32 end-to-end with
  conservation diagnostics. Tier-4.

- **In-kernel subcycling for z-direction.** Saves HBM reads for
  subcycled z sweeps. Tier-4.

These are real improvements but they build on the cleaner base the
four refactors produce.

## How to use this document

When executing any of the four refactors:

1. **Before starting:** read §1 ("directory structure") and §3 ("call
   chain for a structured-grid advection step"). If your refactor's
   diff moves the codebase AWAY from this sketch, pause and ask.

2. **During work:** if you encounter a design choice not covered by
   your plan, check this sketch. Does your choice converge toward
   or diverge from the target? Prefer convergent choices.

3. **Review before committing:** at the end of each refactor, the
   code should look more like this sketch and less like today's
   code. If the diff feels "sideways" — moves code around without
   simplifying — something is wrong.

## Sequencing

The four refactors have a partial order:

```
ping-pong (11) ──────┐
                     ├──► sync removal + rename (13)
scheme consol. (12) ─┘                │
                                      ▼
                              single pipeline (14)
```

- **Ping-pong (11)** can ship first, independent of everything else.
- **Scheme consolidation (12)** can ship in parallel with ping-pong,
  since they touch different things (types vs. workspace).
- **Sync removal + cleanup (13)** depends on BOTH 11 and 12 being
  done — it removes the legacy `@eval` sweep loops, which only
  makes sense after the legacy types are gone, AND it renames
  `rm_buf`→`rm_A`, which only makes sense after ping-pong defines
  those fields.
- **Single pipeline (14)** depends on 13 being done — it touches
  `strang_split!` and `apply!`, which need to be in their
  cleaned-up-post-13 form first.

Recommended order: 11, 12 (parallel), then 13, then 14.

## Closing note

This sketch is prescriptive about the target but not about the path.
Each of the four implementation plans specifies its own sequence of
commits, tests, rollback procedures. This document exists so those
plans have a shared vision — so they compose into a coherent endgame
rather than four independent local improvements.

If anything in this sketch becomes wrong as the refactors progress
(e.g., if memory budget forces a different workspace design, or if
a test reveals something about `CellState` that makes Option C
unworkable), update this document FIRST, then update the affected
plans. The sketch is the source of truth for target shape; the
plans are the source of truth for target implementation.
