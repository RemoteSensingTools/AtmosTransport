# State & basis

Once the [Grids](@ref) are set, prognostic state lives in a
**state object** that carries air mass and one or more tracers. There
are two state types — one for `LatLon` (structured) and `Reduced
Gaussian` (face-indexed / unstructured) sharing the same `CellState`
layout, and one for the panel-native cubed-sphere — but they share
the same accessor API and the same dry-basis contract.

## State types

| State | Topology | Storage shape per tracer |
|---|---|---|
| `CellState{Basis, A, Raw, Names}` | LatLon (structured), ReducedGaussian (face-indexed) | `(Nx, Ny, Nz)` (LL) or `(ncells, Nz)` (RG) |
| `CubedSphereState{Basis, A3, Raw4, Names}` | CubedSphere | `NTuple{6, (Nc + 2Hp, Nc + 2Hp, Nz)}` per panel (halo-padded) |

Both types are **GPU-aware**: their array fields are parametric, so
`Adapt.adapt_structure` ships them onto a `CuArray` (or another
backend) without copying any logic.

```mermaid
classDiagram
    class CellState {
      +air_mass     :: A
      +tracers_raw  :: Raw
      +tracer_names :: NTuple{Nt, Symbol}
      +mass_basis() :: AbstractMassBasis
    }
    class CubedSphereState {
      +air_mass     :: NTuple{6, A3}
      +tracers_raw  :: NTuple{6, Raw4}
      +tracer_names :: NTuple{Nt, Symbol}
      +halo_width   :: Int
    }
    class AbstractMassBasis
    class DryBasis
    class MoistBasis
    AbstractMassBasis <|-- DryBasis
    AbstractMassBasis <|-- MoistBasis
    CellState ..> AbstractMassBasis        : Basis param
    CubedSphereState ..> AbstractMassBasis : Basis param
```

### `CellState{Basis, A, Raw, Names}`

```julia
struct CellState{Basis <: AbstractMassBasis, A, Raw, Names}
    air_mass     :: A         # per-cell mass (kg) on the chosen basis
    tracers_raw  :: Raw       # packed mass storage (size(air_mass)..., Nt)
    tracer_names :: Names     # NTuple{Nt, Symbol}
end
```

User-facing surface:
- `state.air_mass` — per-cell mass, shape `(Nx, Ny, Nz)` or `(ncells, Nz)`.
- `state.tracers.<name>` — non-allocating view of one tracer's raw mass.
- `state.tracer_names` — tuple of symbols.
- `mass_basis(state)` — `DryBasis()` or `MoistBasis()`.

### `CubedSphereState{Basis, A3, Raw4, Names}`

```julia
struct CubedSphereState{Basis, A3, Raw4, Names}
    air_mass     :: NTuple{6, A3}     # per panel (Nc+2Hp, Nc+2Hp, Nz)
    tracers_raw  :: NTuple{6, Raw4}   # per panel (Nc+2Hp, Nc+2Hp, Nz, Nt)
    tracer_names :: Names
    halo_width   :: Int               # Hp from mesh
end
```

User-facing surface mirrors `CellState`. The halo padding is exposed in
`halo_width`; advection sweeps and panel-edge flux rotation use it.

## The dry-basis contract

By default, `state.air_mass` carries **dry-air mass** and every tracer
is interpreted on a **dry-VMR contract**. This is the single most
important runtime invariant in the project, encoded in
[CLAUDE.md](https://github.com/RemoteSensingTools/AtmosTransport/blob/main/CLAUDE.md):

> Dry basis is the default runtime contract. Trace-gas VMRs are always
> dry VMRs, including column averages.

A subtle point: **what's stored in `state.tracers_raw` is tracer mass,
not VMR.** The "dry-VMR contract" describes the user-facing semantics
(initial conditions, snapshot output, column means) — the storage
representation is mass for numerical reasons (it composes naturally
with mass-conserving advection). The conversion happens at the
boundaries:

- **Initial conditions.** A TOML entry `co2.uniform_value = 4.0e-4`
  is read as a *dry VMR* and converted to mass via
  `χ × air_mass` at construction time
  (`set_uniform_mixing_ratio!`).
- **In the loop.** Operators consume `state.tracers_raw` as mass; the
  mass-conservation contract holds because both `air_mass` and the
  tracer-mass slices are pinged through the same flux divergence.
- **Snapshot output.** `<tracer>_column_mean` is `column-integrated
  tracer mass / column-integrated air mass` — a dry VMR by
  construction.
- **Programmatic readout.** `mixing_ratio(state, :CO2)` (in
  `CellState.jl`) gives you the dry VMR directly:
  `get_tracer(state, :CO2) ./ state.air_mass`.

What this means for the binary side:

- The transport binary's `mass_basis = :dry` header says the
  preprocessor already converted `DELP_moist → DELP_dry × (1 − qv)`.
- `state.air_mass` comes from that dry mass.
- Snapshot output writes `<tracer>_column_mean` etc. consistently on
  that dry basis.

Operators dispatch on the basis tag so a `MoistBasis` state would
automatically take a different code path; in practice the runtime is
exclusively dry today and the moist path exists for legacy / future
diagnostic comparisons. **Mixing a moist binary with a dry runtime
contract is a load-time error**, not a silent corruption.

## Basis types

```
AbstractMassBasis  (Basis.jl)
├── DryBasis()    — air_mass is dry-air mass; tracers are dry VMR
└── MoistBasis()  — air_mass is total (moist) air mass; tracers are moist VMR
```

Query the basis at runtime with `mass_basis(state)`. The default
constructor `CellState(air_mass, ...)` produces a `DryBasis` state.

If you find yourself reaching for the moist basis, double-check by
looking at the binary header — under the dry-basis contract, the
preprocessor ships `mass_basis = :dry` and the runtime relies on it.

## Accessor API

All access goes through a small set of helpers in `src/State/Tracers.jl`.
**Tests should observe state through these helpers, not through input
arrays cached before construction** (per the project's testing rules).

### Reading a tracer

```julia
co2_mass = get_tracer(state, :CO2)        # SubArray view of tracer MASS
ch4_mass = state.tracers.CH4              # property access; same view

co2_vmr  = mixing_ratio(state, :CO2)      # CO2 mass / air mass — dry VMR
```

`get_tracer` returns a view into the last dimension of `tracers_raw`
— **tracer mass**, not VMR. On `CellState` this is a 3-D / 2-D view of
the right shape; on `CubedSphereState` it returns one slice per panel.
For dry VMR use `mixing_ratio(state, name)` (or compute the ratio
yourself if you need a backend-specialized variant).

### Iterating

```julia
for (name, χ) in eachtracer(state)
    @info name extrema(χ) sum(χ .* state.air_mass)
end
```

### Counts and indices

```julia
ntracers(state)                 # how many tracers
tracer_index(state, :CO2)       # 1-based index, or nothing
```

### Writing

```julia
set_uniform_mixing_ratio!(state, :CO2, 4.0e-4)   # sets mass = χ × air_mass
```

For more involved initialization (loading from a NetCDF file, regridding
from a different mesh), the runtime's `InitialConditionIO.jl` is the
canonical entry point; user-facing IC kinds are described in the
[First run](@ref) and [Quickstart with example data](@ref) pages.

## GPU residency

When `[architecture] use_gpu = true`, the runtime constructs
`state.air_mass` as a `CuArray` (or the equivalent for the chosen
backend) and asserts residency before stepping. The check fails loudly
if the dispatch chain accidentally fell back to CPU storage:

```text
[ Info: [gpu verified] backend=cuda backing=CuArray device=NVIDIA L40S
```

If you need to round-trip state to and from the host (e.g. for a
diagnostic), use `Array(state.air_mass)` — but **don't** do it inside
the simulation loop; that defeats the GPU.

## Time-varying meteorological fields

For physics blocks that need a time-varying input (e.g. Kz for
diffusion, surface fields for convection), the runtime exposes
`AbstractTimeVaryingField` subtypes. They share a small interface:
`field_value(f, idx)` (kernel-safe) and `update_field!(f, t)`
(host-side cache refresh, called once per met window). Concrete
types currently in the tree:

| Type | Use |
|---|---|
| `ConstantField{FT, N}` | Scalar broadcast to a fixed value. |
| `ProfileKzField{FT, V, N}` | Fixed vertical profile, uniform horizontally. |
| `PreComputedKzField{FT, N, A}` | Wrap a precomputed spatial field. |
| `DerivedKzField{FT, SF, DELP, A, P}` | Beljaars-Viterbo Kz derived from surface fields (u\*, T\*). |
| `StepwiseField{FT, N, A, B, W}` | Piecewise-constant in time (read from binary). |

!!! note "TOML wiring is partial today"
    The runtime recipe currently auto-builds only
    `[diffusion] kind = "none"`, `kind = "constant"`, and
    cubed-sphere `kind = "pbl"` when the binary carries raw
    `pblh/ustar/hflux/t2m` surface sections. Other field types above
    remain building blocks unless a runtime recipe wires them.

## What's next

- *Operators* (Advection, Convection, Diffusion, Sources) and
  *Binary format* — covered in Phase 3B of the documentation overhaul;
  pages will land in a follow-up commit.
