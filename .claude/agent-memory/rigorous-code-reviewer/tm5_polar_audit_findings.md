---
name: TM5 cy3-4dvar polar audit findings
description: Key divergences between TM5 cy3-4dvar polar advection and our implementation, discovered during 2026-04-07 mass_fixer=false audit
type: reference
---

> **STALE — AtmosTransport claims may be stale; TM5 Fortran citations
> remain valid**
>
> Written before plans 18 and 22 shipped. TM5 Fortran references at
> `deps/tm5-cy3-4dvar/base/src/*.F90` are durable and still correct.
> AtmosTransport-side claims (citing `src/Advection/mass_flux_advection.jl`
> and similar) describe code that now lives in `src_legacy/` or has been
> refactored. Verify AtmosTransport-side claims against current `src/`
> before acting on them.
>
> For a current trace of the corresponding subsystem, prefer the
> relevant module README under `src/` over this memory file.

Audit target: explain why `mass_fixer=false` fails at pole-adjacent stratospheric cell (20, 2, 8) in Y direction after 5 halvings.

Source files (all read line-by-line during audit):
- TM5: `deps/tm5-cy3-4dvar/base/src/{advectm_cfl,advecty,advectx,advect_tools,meteo,grid_type_ll,grid_3d,redgridZoom,datetime,tmm_mf_tm5_nc}.F90`
- Ours: `src/Advection/mass_flux_advection.jl`, `src/Grids/reduced_grid.jl`, `scripts/preprocessing/preprocess_spectral_v4_binary.jl`

## Structural facts about TM5 cy3-4dvar polar handling

1. **TM5 has NO Y nloop refinement.** `dynamv` (`advecty.F90:477-657`) is a single-pass loop over vertical levels. Y CFL is handled only by the global `Check_CFL` pre-pass halving ndyn. We do more than TM5 by adding `advect_y_massflux_subcycled!` — this is a safe divergence.

2. **TM5's reduced grid is applied ONLY in X, not Y.** `dynamum` (`advectm_cfl.F90:1214`) uses `imred(j,region)` but `dynamvm`/`dynamv` do not. The reduced grid is set up in `redgridZoom.F90` and applied via `uni2red`/`red2uni` inside `advectx_work` only.

3. **TM5 trusts the preprocessor for polar bm.** `dynam0` (`advect_tools.F90:775-781`) computes `bm(i,j,l) = dtv*pv(i,j,l)` for j=1..jmr+1 inclusive. There's no zeroing of bm at j=1, j=2, j=jmr, or j=jmr+1 anywhere in the TM5 runtime. It trusts `tmpp` (the ECMWF preprocessor, outside the cy3-4dvar source tree).

4. **TM5's Check_CFL does NOT check for m<=0.** Both `dynamum` (X) and `dynamvm` (Y) check only `abs(alpha) >= 1`. If m has gone slightly negative from prior sweeps, TM5 continues silently.

5. **TM5's `max_global_iteration = 32`** (`advectm_cfl.F90:35`) with Sourish Basu comment "trying to run on ml137". This is a smoking gun that cy3-4dvar hits >5 halvings at ml137 in practice.

6. **TM5 reduces ndyn via `new_valid_timestep`**, which finds a divisor of `3*3600 = 10800s`. It is NOT strict halving. Typical sequence: 1800 → 1350 → 1200 → 1080 → 900 → 720 → 600 → 450 → ...

7. **TM5's reduced grid for 1°×1° is `[360, 360, 180, 180, 90, 90, 30, 30, 15, 5]`** (from `rc/include/tm5_regions.rc:48`). TM5 cy3-4dvar does NOT have a 0.5°×0.5° config in its rc tree — it may never have been run at 0.5° in this branch. **Our 0.5° cluster sizes are an extrapolation, not validated against TM5.**

8. **TM5's polar mixing formula in dynamv** (at j=2 face and j=jmr face):
   - For southward flux from pole cell to interior: uses the SIMPLIFIED `beta * rm[pole]` with no slope (because the slope at the polar row would be singular).
   - For northward flux into the pole cell: uses the REGULAR Russell-Lerner formula with rym slope at the donor row.
   - Our `_massflux_y_kernel!` matches this exactly (conditional `j-1 == 1` / `j+1 == Ny` branches).

9. **TM5 updates j=1 and j=jmr rm ONLY by the single adjacent flux**: `rm(1) -= f(2)` and `rm(jmr) += f(jmr)`. It does NOT update rxm/rym at the polar rows. Our kernel computes `rm_new[i,1,k] = rm[i,1,k] + 0 - flux_n` which is equivalent since flux_s=0 at j=1.

10. **TM5's SolvePoissonEq_zoom zeros the Poisson correction at pole faces** via the algebraic identity `v(i, 0) = v(i, jm)` → subtracting `row=v(:,0)` gives both zero. Our code skips pole faces entirely — equivalent.

## Divergences that could explain the failure

**Ranked by likelihood of being the root cause:**

1. **BLOCKING — `max_halvings=5` vs TM5's 32.** At `advect_tools.F90:35` TM5 allows 32 global halvings; ours allows 5. This is the single clearest smoking gun.

2. **BLOCKING — We check `m <= 0` in the pilot; TM5 doesn't.** `mass_flux_advection.jl:1175-1180` fails immediately on negative pilot mass. TM5 silently continues.

3. **MODERATE — Y CFL check uses start-of-substep m, not post-X m.** Ours: pre-X. TM5's dynamvm: post-X. Small effect at polar cells where X is bounded by reduced grid.

4. **MODERATE — Halving cadence differs.** Strict ÷2 (us) vs valid divisors of 10800s (TM5).

## What I cannot determine without running TM5

- Whether TM5 cy3-4dvar has EVER been successfully run at 0.5°×0.5° on ERA5 ml137. No such config in `deps/tm5-cy3-4dvar/rc/`.
- Whether TM5's TMPP preprocessor produces cleaner polar bm than our spectral reconstruction. Would require running TMPP on same GRIB data.
- Whether the cell (20, 2, 8) cumulative drainage over 100 Strang substeps is a preprocessor pathology or a genuine phenomenon.

## Recommendations

1. Raise `max_halvings` to 32. Low risk.
2. Remove or loosen the `m <= 0` check in the pilot. Test empirically.
3. Add preprocessor diagnostic: worst `|bm|/m` at j=2..5 and j=Ny-4..Ny-1 per level.
4. Document that 0.5° reduced grid cluster sizes are extrapolated, not TM5-validated.
5. If possible, run TMPP (external TM5 tool) on Dec 1 2021 data to compare polar bm with our preprocessor.
