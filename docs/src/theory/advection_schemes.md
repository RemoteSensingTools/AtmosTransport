# Advection schemes

The runtime ships **four** flux-form advection schemes, each behind the
abstract type `AbstractAdvectionScheme` declared in
`src/Operators/Advection/schemes.jl`. They share the same kernel-call
signature and the same Strang palindrome composition; differences are
in reconstruction order, monotonicity, and panel-edge handling.

| Scheme | Smooth-flow accuracy | Monotone? | Positive? | LL | CS | RG |
|---|---|---|---|---|---|---|
| `UpwindScheme` | 1st order (donor cell) | yes (trivially) | yes (trivially) | yes | yes | **yes — RG's only option today** |
| `SlopesScheme{L}` | 2nd order in smooth regions (van Leer / Russell-Lerner) | yes if `L = MonotoneLimiter` (default) | only with `L = PositivityLimiter`, and even then only weakly (see [Limiters](@ref Limiters) below) | yes | yes | no (rejected at `StrangSplitting.jl:1237`) |
| `PPMScheme{L}` | 3rd order in smooth regions (Colella-Woodward 1984) | yes with `MonotoneLimiter`, may oscillate without | as `Slopes` | yes | yes — covered by `test/test_cubed_sphere_advection.jl` | no (same rejection) |
| `LinRoodPPMScheme{ORD}` | piecewise-parabolic; `ORD ∈ {5, 7}` selects the boundary stencil | yes (FV3 piecewise-parabolic limiter) | yes | n/a | yes — uses FV3 cross-term advection (`fv_tp_2d_cs!`) | n/a |

The "order" column reports the spatial accuracy of the **per-face
reconstruction** in smooth regions; near limiters / discontinuities
all four schemes drop to first order locally. `LinRoodPPMScheme`'s
`ORD ∈ {5, 7}` selects the **boundary stencil** for cross-panel
faces; the in-panel reconstruction is the same FV3 PPM either way.

## Russell-Lerner slopes (`SlopesScheme`)

The default scheme. Per-cell linear reconstruction with optional limiter:

```math
χ_c(x) \;\approx\; \chi_c \;+\; s_c \,(x - x_c)
```

where `s_c` is the slope estimate. The face flux through the right-of-cell-`c`
face uses the **Courant-fraction formula** (Russell & Lerner, 1981):

```math
F^{n+1/2}_{c+1/2} \;=\; F^n_{c+1/2}\,(\chi^n_c + s^n_c \cdot (1 - \alpha_{c+1/2})/2)
```

with `α = F / m` the local Courant fraction (`F` is the per-substep
mass amount stored in the binary, so the `Δt` factor is already
absorbed into `F` — see [Mass conservation](@ref) for the
`flux_kind = :substep_mass_amount` convention). This is implemented
unchanged from TM5's `advectx__slopes` / `advecty__slopes` routines
(`src/Operators/Advection/reconstruction.jl::_slopes_face_flux`,
declared at line 266 with the canonical formula derivation in the
docstring at lines 213-265).

**Properties:**
- 2nd order in smooth regions even with `NoLimiter` (the centered
  slope itself is a 2nd-order reconstruction; what `MonotoneLimiter`
  buys you is monotonicity at discontinuities, at the cost of
  dropping locally to 1st order at limiter saturation).
- Monotonic with `MonotoneLimiter` (van Leer minmod); does not
  produce new extrema across faces.
- Mass-conservative to floating-point via the flux-form telescoping
  argument (see [Mass conservation](@ref) for the precise statement
  and round-off bounds).

## Putman-Lin PPM (`PPMScheme`)

Piecewise-parabolic reconstruction (Colella & Woodward 1984) with a
parabolic-edge profile:

```math
χ_c(x) \;\approx\; \chi^L_c \;+\; \xi(\Delta\chi_c \;+\; \chi^{(6)}_c (1 - \xi))
```

where `ξ = (x - x_{c-1/2})/Δx_c`, `Δχ_c = χ^R_c − χ^L_c`, and
`χ^{(6)}_c = 6(χ_c − ½(χ^L_c + χ^R_c))` is the curvature parameter.
The face flux is the integral of this parabola over the swept region
`[x_face − u Δt, x_face]`.

**Properties:**
- 3rd order in smooth regions (one above Slopes); local accuracy
  drops to first order at limiter saturation.
- Monotonicity controlled by the limiter parameter; without a
  limiter PPM is **not monotone** and can produce small oscillations
  near discontinuities.
- Shipped on lat-lon AND cubed-sphere structured layouts (the CS
  case is exercised by `test/test_cubed_sphere_advection.jl:374-455`).
  Face-connected PPM for the **reduced-Gaussian** topology is not
  currently wired — and neither is `SlopesScheme` on RG. The
  face-indexed Strang path at `StrangSplitting.jl:1237` rejects
  every advection scheme except `UpwindScheme`, so **`UpwindScheme`
  is the only advection option for RG production runs today**.

For cubed-sphere production runs that need the FV3 cross-term
advection at panel edges, `LinRoodPPMScheme` (next section) is the
preferred variant.

## Lin-Rood PPM with cross-term (`LinRoodPPMScheme{ORD}`)

The cubed-sphere variant. Extends PPM with the **two-step Lin-Rood
splitting** (`fv_tp_2d_cs!` in
`src/Operators/Advection/LinRood.jl`) so the X and Y sweeps see each
other's intermediate fluxes via the inner-edge flux-and-slope rotation
that FV3 uses internally. Two reconstruction orders are selectable:

| `ORD` | Reconstruction | Use case |
|---|---|---|
| `5` | Huynh-constrained PPM | smooth-flow CS runs; no panel-edge artifacts in cross-equator transport |
| `7` | Special boundary treatment for panel edges | runs where panel edges dominate the flow (cross-pole transport, equatorial jets crossing panel boundaries) |

`PPMScheme` is the strict-structured PPM; `LinRoodPPMScheme` is the
cross-term-aware CS-native variant. **Pick `LinRoodPPMScheme` for any CS
production run**; the bare `PPMScheme` is mainly for LL benchmarking
and per-direction kernel testing.

A divergence-damping (del-2) operator is layered on top
(`_divergence_damping_cs_kernel!` in `LinRood.jl`) to suppress the
small numerical noise that survives at the panel boundaries.

## Limiters

Three selectable limiter parameters on `SlopesScheme` and `PPMScheme`,
declared in `src/Operators/Advection/limiters.jl`:

| Limiter | What it enforces | Use case |
|---|---|---|
| `NoLimiter()` | unlimited centered slope / parabola | smooth-flow benchmarks where you want the order-N error rate without limiter clipping |
| `MonotoneLimiter()` (default) | van Leer minmod: `s_c = sign·min(|forward|, |backward|, 2|central|)`. TVD-monotone; never creates new extrema across faces. | production runs |
| `PositivityLimiter()` | one-sided clip that drops the slope where the reconstruction would go negative at a face. **Weaker than `MonotoneLimiter`**: positivity-only, may still create new local maxima from large gradients. | tracers that must stay non-negative (mole fractions, water vapor, aerosol concentrations) AND tolerate occasional new maxima |

Limiter primitives are written branchless (`ifelse(a*b > 0, ..., 0)`)
in `limiters.jl` so they don't trigger warp divergence on the GPU.

If you need both monotonicity and positivity, use `MonotoneLimiter`
on a non-negative initial condition under a non-negative-flux flow —
which the GEOS / spectral preprocessors produce by construction.

## CFL handling and subcycling

The binary's stored fluxes are **per-substep mass amounts**
(`flux_kind = :substep_mass_amount`), already pre-divided by the
substep count for the active window. The Courant number is therefore
a pure mass ratio — no `Δt` factor:

```math
\alpha_{c+1/2} \;=\; \frac{|F^n_{c+1/2}|}{m^n_c}
```

If the maximum `α` over the domain exceeds the active `cfl_limit`,
the runtime **subcycles** that direction:

```julia
n_sub = ceil(max_α / cfl_limit)
```

and runs the direction `n_sub` times with flux scaling `F → F / n_sub`.
The X / Y / Z subcycle counts can differ; all six sweeps in the
Strang palindrome use the same per-direction count to preserve time
symmetry.

`_subcycling_pass_count` in
`src/Operators/Advection/StrangSplitting.jl:620-622` is the
per-direction counter; the structured per-direction max-α helpers
are `_x_subcycling_pass_count` / `_y_subcycling_pass_count` /
`_z_subcycling_pass_count` near `StrangSplitting.jl:753`. The CS
analogue lives in `src/Operators/Advection/CubedSphereStrang.jl:389`.

The `cfl_limit` defaults are baked into the per-topology Strang
entry points:

| Topology | Default `cfl_limit` | Source |
|---|---|---|
| Lat-lon (structured-grid Strang) | `1.0` | `StrangSplitting.jl:964` |
| Reduced Gaussian (face-indexed Strang — separate code path from LL) | `1.0` | `StrangSplitting.jl:1182` |
| Cubed-sphere (`SlopesScheme` / `PPMScheme`) | `0.95` | `StrangSplitting.jl:1063` |
| Cubed-sphere (`LinRoodPPMScheme`) | not used — the LinRood path **does not consume** `cfl_limit` and therefore does not subcycle on the standard advection sub-step. The Lin-Rood operator is run once per sub-step regardless of α. | `LinRood.jl:926` |

If you're running `LinRoodPPMScheme` on a flow with locally large
Courant numbers, the recourse is to halve `dt` in the run config
rather than rely on the operator to subcycle internally.

## Strang palindrome — second-order time accuracy via symmetry

The full transport step is the time-symmetric composition

```text
S(Δt) = X(Δt/2) Y(Δt/2) Z(Δt/2) ∘ V(Δt) ∘ Z(Δt/2) Y(Δt/2) X(Δt/2)
```

(with `V(Δt) → V(Δt/2) S(Δt) V(Δt/2)` when surface flux is on). The
symmetry around the center is what gives the composition its
second-order time accuracy: each direction's truncation error has the
form `c · Δt² + O(Δt⁴)` and the leading `c · Δt²` cancels between the
forward and reverse halves.

Convection and chemistry are NOT inside the palindrome — they are
applied once per met window, post-palindrome — because their natural
cadence is the met window rather than the advection sub-step, and they
do not commute with advection at the per-substep level.

## Panel-edge halo treatment (cubed sphere)

`PanelConnectivity` (`src/Grids/PanelConnectivity.jl:55-96`) carries a
table of panel-edge mappings: for each panel `p` and each edge `e ∈
{west, east, south, north}`, what is the neighbour panel and what's the
edge orientation (`0 = aligned`, `2 = reversed`)? The default
connectivity (lines 82-96) hard-codes the GEOS-FP / GEOS-IT 6-panel
arrangement. The gnomonic alternative is in lines 118-132.

At the start of each X / Y sweep on the cubed sphere, the runtime
`_propagate_cs_outflow_to_halo!` walks the connectivity table and
copies each panel's canonical face flux into the corresponding
neighbour panel's halo, applying the orientation flip if needed. This
keeps the telescoping conservation argument intact across panel
boundaries.

## Reduced-Gaussian per-ring face segmentation

Adjacent rings on a reduced-Gaussian grid have different cell counts
(`nlon[j+1] ≠ nlon[j]`), so the boundary between them cannot be
covered by a one-to-one face mapping. `_boundary_counts(nlon_per_ring)`
in `src/Grids/ReducedGaussianMesh.jl:121-130` segments each ring-pair
boundary into `lcm(nlon[j], nlon[j+1])` mini-faces — every cell on
ring `j` contributes `lcm/nlon[j]` mini-faces, and every cell on ring
`j+1` receives `lcm/nlon[j+1]` mini-faces. Mass-flux pairing is then
exact and the telescoping argument holds across the whole grid.

The cost is a higher face count than the cell count would suggest:
adjacent rings with `nlon = 108, 112` produce `lcm = 3024` mini-faces
between them. Performance-tuning notes are inline at line 118.

## Where the schemes meet the code

| Concept | File:line |
|---|---|
| Scheme abstract root | `src/Operators/Advection/schemes.jl::AbstractAdvectionScheme` (line 67) |
| `UpwindScheme` | `schemes.jl:195` |
| `SlopesScheme{L}` | `schemes.jl:233` |
| `PPMScheme{L}` | `schemes.jl:260` |
| `LinRoodPPMScheme{ORD}` | `schemes.jl:288` |
| Limiter primitives | `src/Operators/Advection/limiters.jl` |
| Slopes face flux (Russell-Lerner formula) | `src/Operators/Advection/reconstruction.jl::_slopes_face_flux` (line 266) |
| Strang palindrome | `src/Operators/Advection/StrangSplitting.jl:1363-1410` |
| CFL subcycle counters | `StrangSplitting.jl::_subcycling_pass_count` (line 620), `_static_*_subcycle_count` (lines 679-694) |
| CS panel-edge halo sync | `src/Grids/PanelConnectivity.jl` + `cs_transport_helpers.jl::_propagate_cs_outflow_to_halo!` |
| Lin-Rood cross-term + del-2 damping | `src/Operators/Advection/LinRood.jl` |

## What's next

- [Conservation budgets](@ref) — the synthetic-fixture tests that
  verify the conservation contract bit-by-bit.
- [Validation status](@ref) — what we've actually validated end-to-end
  vs what's still on the to-do list.
