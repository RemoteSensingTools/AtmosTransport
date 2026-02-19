# TM5 code alignment checklist

Use this when comparing our implementation to TM5 source (e.g. `base/` in TM5 repo) to ensure forward agreement. Delegate the actual diff to a script or side-by-side read.

## Advection (Russell–Lerner slopes)

- [ ] **Flux formula:** Face flux = u * (c_upwind + (1 - u*Δt/Δx) * slope/2) for upstream cell; slope = (c_R - c_L)/2 or minmod-limited. Same in TM5?
- [ ] **Operator split order:** X(Δt/2), Y(Δt/2), Z(Δt/2), Z(Δt/2), Y(Δt/2), X(Δt/2). TM5 symmetric?
- [ ] **Slope limiter:** Minmod or other? We use minmod(s, 2*(c_R-c), 2*(c-c_L)).
- [ ] **Periodic x:** Same handling at i=1 and i=Nx.
- [ ] **Boundary y/z:** Zero flux at poles / top and bottom. Same in TM5?

## Convection (Tiedtke mass flux)

- [ ] **Update:** q_new[k] = q_old[k] + Δt*g/Δp[k] * (F[k+1] - F[k]) with F = M_net * q_upwind.
- [ ] **Upwind:** M>0 => upwind from below (layer k); M<0 => upwind from above (layer k-1).
- [ ] **BCs:** F[1] = F[Nz+1] = 0.

## Grid

- [ ] **Cell centers:** Tracer at (λᶜ, φᶜ). TM5 cell-centered?
- [ ] **Δx, Δy:** Δx = R*cos(φᶜ)*Δλ; Δy uniform. TM5 same?
- [ ] **Reduced grid:** We use regular grid + CFL limit; TM5 may use reduced. Document difference.

## CFL / poles

- [ ] **TM5:** Time step reduction or reduced grid at poles?
- [ ] **Ours:** Per-column u_max from Δx(1,j); at poles Δx→0 so u effectively zeroed.

## Files to compare

- Our advection: `src/Advection/slopes_advection.jl`
- Our convection: `src/Convection/tiedtke_convection.jl`
- TM5: `base/` (advection and convection source file names in TM5)
