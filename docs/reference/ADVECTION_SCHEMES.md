# Advection Schemes

AtmosTransport implements four advection schemes, progressing from simple
to sophisticated. All are mass-conserving by construction (flux-form
telescoping).

## Quick comparison

| Scheme | Order | Stencil | Monotone | CS panels | GPU | Best for |
|--------|-------|---------|----------|-----------|-----|----------|
| Upwind | 1st | 1 cell | Yes (positive-definite) | Yes | Yes | Debugging, baseline |
| Slopes | 2nd | 2 cells | Yes (van Leer limiter) | Yes (Hp>=2) | Yes | Production LL/RG |
| PPM | 3rd | 3 cells | Configurable (ORD 4-7) | Yes (Hp>=3) | Yes | Low-diffusion transport |
| LinRood | 3rd | 3 cells | Configurable | CS only (Hp>=3) | Yes | CS production (eliminates splitting error) |

## Upwind (`scheme = "upwind"`)

**First-order donor cell.** The simplest scheme: face value equals the
upwind cell mean.

```
f_{i+1/2} = q_i     if flux > 0
           = q_{i+1} if flux < 0
```

- **Type**: `UpwindScheme <: AbstractConstantScheme`
- **Pros**: unconditionally stable, positive-definite, fast
- **Cons**: highly diffusive (spreads gradients over ~sqrt(N) cells)
- **Config**: `scheme = "upwind"` in `[run]`
- **Source**: `src/Operators/Advection/structured_kernels.jl`

## Slopes / van Leer (`scheme = "slopes"`)

**Second-order piecewise-linear** with van Leer minmod limiter
(Russell & Lerner 1981).

```
f_{i+1/2} = q_i + (1 - CFL)/2 * slope_i    (for positive flux)
slope_i = minmod(q_{i+1} - q_i, q_i - q_{i-1})
```

- **Type**: `SlopesScheme <: AbstractLinearScheme`
- **Pros**: much sharper gradients than upwind, still monotone
- **Cons**: limited to 2nd-order accuracy
- **Config**: `scheme = "slopes"`
- **Halo**: requires `Hp >= 2` for cubed-sphere panels
- **Source**: `src/Operators/Advection/structured_kernels.jl` (via `_xface_tracer_flux`)

## PPM (`scheme = "ppm"`)

**Piecewise Parabolic Method** (Colella & Woodward 1984, extended by
Putman & Lin 2007). Reconstructs a parabola within each cell.

The `ppm_order` parameter selects the reconstruction variant:

| ORD | Name | Monotonicity | Reference |
|-----|------|-------------|-----------|
| 4 | LR96 + minmod | Fully monotone | Putman & Lin Sec. 4 |
| 5 | Huynh constraint | Quasi-monotone | Huynh 1996 |
| 6 | Quasi-5th order | Non-monotone (~1% undershoot) | Suresh & Huynh 1997 |
| 7 | ORD=5 + CS face discontinuity | Quasi-monotone | Putman & Lin App. C |

The parabolic reconstruction:
```
q(x) = q_L + x*(q_R - q_L + (1-x)*q_6)    for x in [0, 1]
q_6 = 6*(q_c - (q_L + q_R)/2)              curvature coefficient
```

Face flux integral (FV3 `xppm` formula):
```
F = q_c + (1 - alpha) * (q_R - q_c - alpha * q_6)    (positive flow)
```

- **Type**: `PPMScheme <: AbstractQuadraticScheme`
- **Config**: `scheme = "ppm"`, optionally `ppm_order = 5` in `[run]`
- **Halo**: requires `Hp >= 3` for cubed-sphere panels
- **Source**: `src/Operators/Advection/ppm_subgrid_distributions.jl` (edge values),
  `src/Operators/Advection/structured_kernels.jl` (flux kernels)

## Lin-Rood (`scheme = "linrood"`)

**Cross-term advection** (Lin & Rood 1996, Putman & Lin 2007). The key
innovation for cubed-sphere grids: standard Strang splitting
(X-Y-Z-Z-Y-X) introduces directional bias at panel boundaries. Lin-Rood
eliminates this by computing both orderings from the original field and
averaging the fluxes.

Algorithm (FV3's `fv_tp_2d`):
1. Compute Y-PPM face values from original field → inner Y fluxes
2. Pre-advect in Y to get q_i (Y-then-X intermediate)
3. Compute X-PPM face values from original field → inner X fluxes
4. Compute X-PPM face values from q_i → outer X fluxes
5. Pre-advect in X to get q_j (X-then-Y intermediate)
6. Compute Y-PPM face values from q_j → outer Y fluxes
7. Update: average inner and outer fluxes, apply simultaneously

Full 3D: `LinRood_H → Z → Z → LinRood_H` (Strang split on vertical only).

Optional divergence damping (`damp_coeff = 0.02`) applies del-2 diffusion
on mixing ratio before the first horizontal sweep.

- **Config**: `scheme = "linrood"`, `halo_padding = 3`, `ppm_order = 5`
- **Grid**: cubed-sphere only (no LL/RG variant needed — splitting error is
  a CS-specific problem)
- **Source**: `src/Operators/Advection/LinRood.jl`

## Choosing a scheme

**For lat-lon grids**: start with `slopes` (good accuracy/cost ratio).
Use `ppm` for high-resolution or long integrations where numerical
diffusion matters.

**For cubed-sphere grids**: use `linrood` for production runs (eliminates
panel-boundary artifacts). Use `upwind` for debugging.

**For reduced-Gaussian grids**: only `upwind` is currently supported
(face-indexed kernel architecture).

## Vertical advection

All schemes use the same vertical advection: column-sequential upwind
with double-buffering for mass conservation. The vertical sweep is
independent of the horizontal scheme choice.

For FV3-compatible transport, enable `vertical_remap = true` in the config
to use conservative PPM remapping instead of explicit Z-advection.
