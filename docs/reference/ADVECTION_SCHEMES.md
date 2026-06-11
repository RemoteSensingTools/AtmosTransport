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
| 5 | 4th-order edge interp + extremum limiter | Quasi-monotone | Colella & Woodward 1984; Putman & Lin Sec. 4 |
| 6 | Unlimited 5th-order upwind | Non-monotone (small over/undershoot) | Suresh & Huynh 1997 |
| 7 | ORD=5 + CS face discontinuity | Quasi-monotone | Putman & Lin App. C |

ORD=5/7 use the 4th-order cell-edge interpolation
`q_{i∓1/2} = (7/12)(q_im+q_i) − (1/12)(q_imm+q_ip)`, with quasi-monotone extremum
flattening applied separately in the face kernels. ORD=6 uses the unlimited
5th-order upwind-biased stencil `(2,−13,47,27,−3)/60`.

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

- **Config**: `scheme = "linrood"`, `halo_padding = 3`, `ppm_order = 5`,
  `fillz = true`
- **Grid**: cubed-sphere only (no LL/RG variant needed — splitting error is
  a CS-specific problem)
- **Source**: `src/Operators/Advection/LinRood.jl`

### The `fillz` knob (`fillz = true | false`, default `true`)

Lin-Rood's high-order reconstruction transiently **undershoots** just
downwind of very sharp gradients, producing small negative mixing ratios.
GCHP repairs these with `fillz` (`fv_fill.F90`): negative cells are filled
by borrowing mass from adjacent levels in the same column. Our port runs it
four times per substep (after each sweep of the palindrome). The knob
controls whether it runs at all.

**`fillz = true` (default — GCHP-faithful)**

- *Pro*: output fields are strictly non-negative; transport behaves
  identically to GCHP at sharp gradients.
- *Con*: **fillz is the scheme's only mass non-conservation.** Although the
  level-borrowing is zero-sum in exact arithmetic, the
  `mass → VMR → fill → VMR → mass` round-trip is not Float32-exact, and each
  repair injects a small net mass. Measured on a sharp IC=0 tracer
  (co2_fossil, C180 F32, 1 day): the fillz-injected mass equals the run's
  entire conservation surplus to three decimals (`fillz/surplus = 1.000`).
  The effect is a spin-up transient — +5 % of emission at t=3 h while fresh
  plumes are sharp, saturating in absolute terms (~0.3 % of emission at
  month scale) as plumes smooth and undershoots stop firing. Smooth
  large-background tracers (e.g. natural CO₂) barely trigger it.

**`fillz = false` (exactly conservative)**

- *Pro*: flux-form advection then conserves tracer mass **exactly**
  (telescoping fluxes) — the right setting for conservation-critical budget
  and inversion work. It also removes the only non-differentiable operation
  from the forward sweep, which is strictly better for the 4D-Var adjoint.
  Negative transients are benign in this offline model: every production
  operator is sign-safe (the Lin-Rood monotonicity limiter works on
  differences, the vertical sweep and convection are linear in tracer mass,
  the implicit diffusion solve is linear with row-sum 1, and exponential
  decay is multiplicative) — the reference-state anomaly transport
  (plan 45) routinely runs signed fields through the identical palindrome.
- *Con*: output VMR goes negative in two distinct ways (measured, fossil
  1-day C180): (i) plume-edge undershoots reaching ≈ −14 % of the local
  peak while plumes are sharpest (−1.5 ppm against +10.7 ppm peaks at
  t=3 h, decaying to −0.9 ppm by 24 h) — visible in maps near strong
  sources; (ii) ±epsilon noise around zero covering a large fraction of
  the far field (~40 % of cells, amplitudes ~1e-9 of the peak) — harmless
  but it means "fraction of negative cells" is not a useful metric, look
  at amplitudes. Transport also no longer matches GCHP's exact behavior at
  sharp gradients, and the mode is **not safe if a nonlinear or
  positivity-requiring chemistry operator is ever added** (the only decay
  we run is linear and sign-safe).

**Validated A/B** (co2_fossil-only, C180 F32, 1 day): conservation surplus
+5.061 % of emission (fillz on) → **+0.0005 %** (fillz off — i.e. exact
flux-form conservation, ~80× tighter than even the split-PPM scheme's
+0.04 %), with the fillz-injection diagnostic reading exactly 0.

**Recommendation**: keep the default for GCHP intercomparison work; set
`fillz = false` for mass-budget or 4D-Var work with sharp localized
sources. Verify with the opt-in diagnostic `ATMOSTR_FILLZ_MASS_DIAG=1`,
which logs the cumulative net mass fillz injected over a run
(`[fillz diag] …` at run end) — it reads exactly zero with `fillz = false`.

Note: the split-sweep schemes (`upwind`/`slopes`/`ppm`) take no `fillz`
knob — they prevent negatives upstream (the PPM/Slopes moment limiter caps
the subgrid reconstruction so a cell cannot export more mass than it holds)
and are conservative without a fixer.

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
