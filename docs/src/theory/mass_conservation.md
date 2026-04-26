# Mass conservation

This page derives the discrete conservation contract that holds end-to-end
across the AtmosTransport pipeline (preprocessor → binary → runtime →
snapshot output) and quantifies the floating-point tolerances at which it
holds.

## The contract

For every cell `c` and every advection sub-step:

```math
m^{n+1}_c \;=\; m^n_c \;+\; \sum_f \sigma_{c,f}\,F^n_f
```

where `F^n_f` is the per-substep mass-amount through face `f` (units: kg per
substep, the binary's `flux_kind = :substep_mass_amount`) and `σ_{c,f} ∈
{+1, -1}` is the inflow / outflow sign convention for face `f` at cell `c`.
The same identity holds for every tracer mass `μ^n_{c, t}`:

```math
\mu^{n+1}_{c,t} \;=\; \mu^n_{c,t} \;+\; \sum_f \sigma_{c,f}\,\Phi^n_{f,t}
```

with `Φ^n_{f, t}` the tracer-mass flux through face `f`. Because `m` and
the tracer-mass slice live on the same staggering and use the same `F`,
`Φ` triple-pairing, the **mixing ratio `χ = μ / m` of a passive tracer
that starts uniform stays uniform exactly** under advection — modulo
floating-point.

## Telescoping divergence — why advection conserves total mass

For a closed periodic domain (lat-lon with periodic longitude) summing the
update over all cells gives:

```math
\sum_c m^{n+1}_c \;=\; \sum_c m^n_c \;+\; \sum_c \sum_f \sigma_{c,f} F^n_f
```

Each interior face appears in the sum twice — once with `+F` (the cell on
the inflow side) and once with `−F` (the cell on the outflow side) — so
the double-sum collapses to **boundary fluxes only**:

```math
\sum_c \sum_f \sigma_{c,f} F^n_f \;=\; \sum_{f \in \partial \Omega} F^n_f
```

For a closed sphere there is no boundary, so the right-hand side
identically vanishes mathematically — and to floating-point in the
implementation, modulo the accumulation noise of summing many
fluxes. This holds for every advection scheme that is written in
flux form (the X / Y / Z sweep kernels in
`src/Operators/Advection/structured_kernels.jl`). It does NOT hold
for schemes written in advective form (e.g. `χ ∂_t = -u ∂_x χ`),
which is why every advection scheme in this repository is flux-form.

The same telescoping argument extends to:

- The cubed sphere via panel-edge halo synchronisation
  (`PanelConnectivity` in `src/Grids/PanelConnectivity.jl`) which carries
  the canonical face flux from one panel onto the halo of the neighbour
  panel with the right sign / orientation.
- The reduced-Gaussian grid via the LCM-based ring-boundary face
  segmentation in `_boundary_counts(nlon_per_ring)` in
  `src/Grids/ReducedGaussianMesh.jl`. Each ring-pair boundary is split
  into `lcm(nlon[j], nlon[j+1])` segments so the outflow from ring `j`
  exactly equals the inflow to ring `j+1`.

## Vertical closure

The discrete continuity equation for the vertical mass flux `cm` is:

```math
m^{n+1}_c \;=\; m^n_c \;-\; 2\,\text{steps}\,(am_{e} - am_{w} \;+\; bm_{n} - bm_{s} \;+\; cm_{k+1} - cm_{k})
```

with the surface boundary `cm` at `k = N_z + 1` pinned to zero (no mass
flux through the ground). Two paths produce a `cm` field that satisfies
this closure:

1. **Diagnose `cm` from `(am, bm, dm)`** — used by the LL / RG spectral
   preprocessor's `recompute_cm_from_dm_target!` after Poisson balance
   has corrected `am, bm`.
2. **The FV3 pressure-fixer `cm`** — used by the GEOS native CS
   preprocessor's `compute_cs_cm_pressure_fixer!`. Per-column scan that
   computes `cm[k+1] = cm[k] + (C_k − ΔB[k] · pit)` where
   `pit = Σ_k (am_inflow + bm_inflow)` is the column-integrated horizontal
   convergence. Closes `cm[N_z + 1] = 0` **exactly** by construction —
   no residual to redistribute.

Either way, the binary lands with a `cm` field that closes the explicit-
`dm` continuity equation to floating-point tolerance — the
**write-time replay gate** then verifies it before the binary file is
committed to disk.

## Replay-gate tolerance

Conservation in floating-point is verified per-window via:

```math
\frac{\|m_{\text{evolved}} - m_{\text{stored}, n+1}\|}{\|m_{\text{stored}, n+1}\|} \;\le\; \tau(\text{FT})
```

`replay_tolerance(FT)` is defined in
`src/MetDrivers/ReplayContinuity.jl`:

| FT | `τ(FT)` |
|---|---|
| `Float64` | `1e-10` |
| `Float32` | `1e-4`  |

The Float32 tolerance reflects the noise floor of single-precision
arithmetic at production resolutions (Float32's 23-bit mantissa,
giving ~7 decimal digits of precision per operation, accumulates to
roughly `1e-5` per substep on a 720×361 grid; the per-window gate is
relaxed to `1e-4` to absorb the per-window accumulation).

The gate fires twice in the lifecycle of a binary:

1. **Write-time** (always on, in the preprocessor). A binary that fails
   is rejected at write time — the preprocessor errors out rather than
   producing a known-bad file. The binary committed to disk is, by
   construction, replay-clean.
2. **Load-time** (opt-in, in the runtime). Set
   `[met_data] validate_replay = true` in the run config or
   `ATMOSTR_REPLAY_CHECK=1` in the environment. Off by default because
   it doubles binary load time; recommended for any new binary
   configuration before a long production simulation.

## Dry-basis vs moist-basis

The pipeline ships **dry-basis by default** (`mass_basis = :dry` in the
binary header). The mathematical content of the conservation law is
identical on either basis — what changes is the meaning of the stored `m`:

| `mass_basis` | `m` represents | Tracer VMR semantics |
|---|---|---|
| `:dry` | `m_dry = m_moist · (1 − qv)` per cell | dry VMR (`χ_dry = μ / m_dry`) |
| `:moist` | total air mass per cell | moist VMR (`χ_moist = μ / m_moist`) |

Conversions happen at the boundaries:

- **Preprocessing.** `apply_dry_basis_native!` in
  `src/Preprocessing/mass_support.jl` multiplies cell-centered
  `m, dp, ps` and face-averaged `am, bm` by `(1 − qv_face)` with the
  appropriate face-averaging convention. After this step every payload
  field in the binary is dry.
- **Runtime.** `state.air_mass` carries dry mass. Tracer storage in
  `state.tracers_raw` is **mass**, not VMR; the dry-VMR contract is
  enforced at the IC boundary (uniform-value initial conditions
  interpret `4.0e-4` as dry VMR and convert to mass via `χ × m_dry` at
  construction) and at the snapshot-output boundary
  (`<tracer>_column_mean = column-integrated tracer mass /
  column-integrated air mass` is dry by construction).
- **Convection forcing.** The CMFMC and DTRAIN fields shipped by
  GMAO are moist-basis. The GEOS reader's
  `_moist_to_dry_cmfmc!` / `_moist_to_dry_dtrain!` apply the
  `(1 − qv_face)` correction so the convection operator consumes
  forcing on the same basis as `state.air_mass`.

**Mixing a moist binary with a dry-basis runtime is rejected at
`DrivenSimulation` construction time** — not at raw binary open, but
well before any windows actually step.

## Where the math meets the code

| Concept | File:line |
|---|---|
| X-sweep kernel (flux-form telescoping) | `src/Operators/Advection/structured_kernels.jl:78–95` |
| Strang palindrome (X→Y→Z / V/S/V / Z→Y→X) | `src/Operators/Advection/StrangSplitting.jl:1363–1410` |
| FV3 pressure-fixer `cm` | `src/Preprocessing/cs_transport_helpers.jl::compute_cs_cm_pressure_fixer!` |
| Diagnose `cm` from `dm` target (LL / RG / CS) | shared implementation `src/MetDrivers/ReplayContinuity.jl::recompute_cm_from_dm_target!` (function definition starts at line 94); LL preprocessor calls it from `latlon_contracts.jl:146` after Poisson balance |
| Write-time replay gate | `src/MetDrivers/ReplayContinuity.jl::verify_window_continuity_*!` |
| `replay_tolerance(FT)` | `src/MetDrivers/ReplayContinuity.jl:25–26` |
| Dry-basis correction | `src/Preprocessing/mass_support.jl::apply_dry_basis_native!` |
| Basis-mismatch enforcement | `src/Models/DrivenSimulation.jl` (construction) |

## What's next

- [Advection schemes](@ref) — per-scheme order, monotonicity, CFL.
- [Conservation budgets](@ref) — what the test suite actually asserts.
- [Validation status](@ref) — what's been validated against external
  reference data; what hasn't.
