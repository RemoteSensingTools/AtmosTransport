---
name: TM5 Check_CFL outer-loop design for ERA5 LL
description: Complete line-by-line trace of TM5 Check_CFL flow and mapping to Julia code — key for understanding global timestep refinement
type: reference
---

> **PARTIALLY STALE — TM5 Fortran section durable; Julia mapping table
> is stale**
>
> The TM5 Check_CFL Fortran trace at the top (citing `advectm_cfl.F90`
> and the `dynam0` / `dynamum` / `dynamvm` / `dynamwm` call chain) is
> durable external reference material and still correct.
>
> The "Julia Mapping" table below cites `run_loop.jl`, `physics_phases.jl`,
> `met_buffers.jl`, and `mass_flux_advection.jl` — all of which now live
> in `src_legacy/`, not `src/`. Line numbers are therefore stale and the
> architectural mapping predates plans 14, 18, and 22. Do not trust the
> mapping table's current-code claims without re-deriving them against
> `src/Operators/Advection/` and `src/Models/TransportModel.jl`.

## TM5 Check_CFL Flow (advectm_cfl.F90:154-302)

1. `ndyn = ndyn_max` (typically 3600s for global region)
2. `n = ceiling(|t2-t1| / ndyn)` — number of Strang cycles per met window
3. Per Strang cycle: `Setup_MassFlow(tr, ndyn)` → `dynam0(region, ndyn)`:
   - `dtu = ndyn / (2 * tref(region))` with tref(1)=1 for global
   - `am = dtu * pu`, `bm = dtv * pv`, `cm = -dtw * sd`
   - Mass fluxes scale LINEARLY with ndyn

4. Mass-only pilot: `determine_cfl_iter` → `do_steps` → advectm{x,y,z}zoom → dynamum/dynamvm/dynamwm
   - X (dynamum): inner per-row subcycling, handles CFL > 1 locally (max_nloop=10)
   - Y (dynamvm): NO subcycling, sets cfl_ok=false if |beta| >= 1 ANYWHERE
   - Z (dynamwm): NO subcycling, sets cfl_ok=false if |gamma| >= 1 ANYWHERE

5. On failure: `new_valid_timestep(ndyn, 3*3600)` reduces ndyn to next valid divisor
   - `n = nint(n * ndyn_old / ndyn)` — increase step count
   - `am_t *= ndyn_new/ndyn_old`, same for bm_t, cm_t
   - Restore masses, retry (up to max_global_iteration=32)

## Julia Mapping

| TM5 | Julia | File |
|-----|-------|------|
| ndyn | implicit: dt_sub * 2 * tref = dt_sub | run_loop.jl:36 |
| n (Strang cycles) | n_sub = steps_per_window(driver) | run_loop.jl:35 |
| am/bm/cm in dynam0 | Pre-computed, stored in LatLonMetBuffer.am/bm/cm | met_buffers.jl:48-52 |
| Check_CFL outer loop | NEW: outer CFL check in advection_phase! | physics_phases.jl:559+ |
| dynamum subcycling | advect_x_massflux_subcycled! (flux-remaining) | mass_flux_advection.jl:1138 |
| dynamvm CFL check | max_cfl_massflux_y (3-arg: uses cfl_arr scratch) | mass_flux_advection.jl:1089 |
| dynamwm CFL check | max_cfl_massflux_z (3-arg: uses cfl_arr scratch) | mass_flux_advection.jl:1104 |
| store_masses / restore | In-place scale + restore of gpu.am/bm/cm | proposed |
| MassFluxWorkspace.cfl_y | Scratch for Y CFL check (Nx, Ny+1, Nz) | mass_flux_advection.jl:559 |
| MassFluxWorkspace.cfl_z | Scratch for Z CFL check (Nx, Ny, Nz+1) | mass_flux_advection.jl:560 |

## Key Insight

TM5 needs a full mass-only pilot because Y and Z have NO inner subcycling — the only fix is global timestep reduction. Our code has inner flux-remaining subcycling on all 3 directions, so the outer check is primarily a PERFORMANCE optimization: pre-scaling fluxes to avoid expensive per-iteration CFL recomputation in the inner loop.
