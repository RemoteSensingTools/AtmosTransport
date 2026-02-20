# TM5 Code Alignment Checklist

Use this when comparing our implementation to TM5 source (`deps/tm5/base/src/`)
to ensure forward agreement. See also [MASS_FLUX_EVOLUTION.md](MASS_FLUX_EVOLUTION.md)
for the history of how the advection was aligned with TM5.

## Mass-flux advection (TM5-faithful) — `mass_flux_advection.jl`

- [x] **Prognostic variables:** Tracer mass `rm` and air mass `m` co-advected
      (matches TM5 `advectx.F90` lines 630, 706-716).
- [x] **Courant number:** Mass-based `α = am / m_donor`
      (matches TM5 `advectx.F90` line 663).
- [x] **Flux formula:** `f = α * (rm + (1-α) * sxm / 2)` for positive flow
      (Russell-Lerner, matches TM5).
- [x] **Air mass update:** `m_new = m + am_in - am_out` within each 1D step
      (matches TM5 `advectx.F90` line 630).
- [x] **Continuous m tracking:** `m` passed through entire Strang split
      without resetting (matches TM5 operator splitting).
- [x] **Vertical flux from continuity:** `cm` derived from horizontal
      convergence with B-coefficient weighting (matches TM5 `dynam0` in
      `advect_tools.F90`).
- [x] **CFL subcycling:** Mass flux divided by `n_loop` and advection repeated
      (matches TM5 `advectx_get_nloop` in `advectm_cfl.F90`).
- [x] **Operator split order:** X(Δt/2), Y(Δt/2), Z(Δt/2), Z(Δt/2), Y(Δt/2),
      X(Δt/2) — symmetric Strang split (matches TM5).
- [x] **Periodic x:** Wrapping boundary at i=1 and i=Nx (matches TM5).
- [x] **Boundary y:** Zero flux at j=1 and j=Ny+1 (matches TM5 pole handling).
- [x] **Boundary z:** Zero flux at k=1 (top) and k=Nz+1 (bottom)
      (matches TM5 `dynamw_1d`).

### Known differences from TM5

- **Slope handling:** TM5 evolves slopes as prognostic variables (`rxm`, `rym`,
  `rzm`). We compute slopes diagnostically from concentration `c = rm/m` each
  step, then scale by `m`. This is simpler and preserves uniform fields exactly
  but does not carry slope information across time steps. The impact on accuracy
  is minimal (uniform field preserved to < 4e-13).

- **Mass flux source:** TM5 derives mass fluxes from ECMWF spectral harmonics
  (`pu`, `pv`, `sd`) via `dynam0`, which guarantees exact mass conservation at
  the spectral truncation level. We derive mass fluxes from gridpoint winds via
  `compute_mass_fluxes`, which is slightly less constrained but uses the same
  continuity equation structure for `cm`.

- **Reduced grid:** TM5 uses a reduced grid near the poles to maintain CFL
  stability. We now implement TM5-style reduced-grid x-advection for CPU via
  `advect_x_massflux_reduced!`, which reduces rm, m, and am to coarser rows
  at high latitudes, advects on the reduced row, then expands back.
  On GPU, we fall back to global CFL subcycling (simple, efficient for GPU
  thread scheduling). The numerical behavior is equivalent but the
  implementation differs.

## Stencil-level advection (concentration-based) — `slopes_advection.jl`

- [x] **Flux formula:** Face flux = u * (c_upwind + (1 - α) * slope/2)
      (identical to TM5).
- [x] **Slope limiter:** minmod(s, 2*(c_R-c), 2*(c-c_L))
      (identical to TM5).
- [x] **Periodic x:** Same handling at i=1 and i=Nx.
- [x] **Boundary y/z:** Zero flux at poles / top and bottom.
- [x] **Reduced grid:** TM5-style reduced grid implemented for mass-flux
      advection (`advect_x_massflux_reduced!`) and concentration-based
      advection on CPU. GPU uses CFL subcycling.

## Convection (Tiedtke mass flux)

- [x] **Update:** q_new[k] = q_old[k] + Δt*g/Δp[k] * (F[k+1] - F[k])
      with F = M_net * q_upwind.
- [x] **Upwind:** M>0 => upwind from below; M<0 => upwind from above.
- [x] **BCs:** F[1] = F[Nz+1] = 0.

## Grid

- [x] **Cell centers:** Tracer at (λᶜ, φᶜ). TM5 cell-centered — same.
- [x] **Δx, Δy:** Δx = R*cos(φᶜ)*Δλ; Δy uniform. Same as TM5.
- [x] **Reduced grid:** TM5-style reduced grid for mass-flux x-advection on
      CPU. GPU uses regular grid + CFL subcycling.

## Files to compare

| Our file | TM5 file | Status |
|----------|----------|--------|
| `src/Advection/mass_flux_advection.jl` | `advectx.F90`, `advecty.F90`, `advectz.F90` | Aligned |
| `src/Advection/mass_flux_advection.jl` (compute_mass_fluxes) | `advect_tools.F90` (dynam0) | Aligned (different flux source) |
| `src/Advection/slopes_advection.jl` | `advectx.F90` (stencil only) | Verified identical |
| `src/Convection/tiedtke_convection.jl` | TM5 convection source | Aligned |
| `src/TimeSteppers/operator_splitting.jl` | TM5 main loop | Aligned (Strang split) |
