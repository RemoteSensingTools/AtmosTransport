---
name: LL advection chain review
description: Full trace of LatLon advection from run loop to GPU kernels, comparing Julia vs TM5 F90. Key bugs found in CFL subcycling.
type: reference
---

## LL Advection Call Chain (2026-04-03)

### Call chain traced
```
run! → _run_loop! (run_loop.jl:27)
  → advection_phase! (physics_phases.jl:593) [LL dispatch]
    → _clamp_cm_cfl!(gpu.cm, gpu.m_ref, 0.95) (physics_phases.jl:560)
    → copyto!(gpu.m_dev, gpu.m_ref)
    → for _ in 1:n_sub:
        → _apply_advection_latlon!(tracers, gpu.m_dev, gpu.am, gpu.bm, gpu.cm, ...) (run_helpers.jl:218)
          → strang_split_massflux!(tracers, m, am, bm, cm, grid, true, ws; cfl_limit) (mass_flux_advection.jl:1267)
            → for each tracer:
                → copyto!(ws.rm, rm_tracer)
                → advect_x_massflux_subcycled!(rm_single, m, am, grid, true, ws; cfl_limit)
                → advect_y_massflux_subcycled!(rm_single, m, bm, grid, true, ws; cfl_limit)
                → advect_z_massflux_subcycled!(rm_single, m, cm, true, ws; cfl_limit) × 2
                → advect_y_massflux_subcycled!(rm_single, m, bm, grid, true, ws; cfl_limit)
                → advect_x_massflux_subcycled!(rm_single, m, am, grid, true, ws; cfl_limit)
                → copyto!(rm_tracer, ws.rm)
```

### Workspace (MassFluxWorkspace) lives in LatLonMetBuffer.ws
- `ws.rm`: (Nx, Ny, Nz) — single tracer working buffer
- `ws.rm_buf`: (Nx, Ny, Nz) — kernel output buffer
- `ws.m_buf`: (Nx, Ny, Nz) — kernel mass output buffer
- `ws.cfl_x`: (Nx+1, Ny, Nz) — dual-use: CFL scratch then scaled flux
- `ws.cfl_y`: (Nx, Ny+1, Nz) — same
- `ws.cfl_z`: (Nx, Ny, Nz+1) — same
- `ws.cluster_sizes`: Int32 vector length Ny

### Key findings
1. **No n_sub cap in CFL subcycling** — n_sub = ceil(Int, cfl/0.95) with no upper bound.
   If cfl is very large (small m after prior advection), n_sub → huge → effective hang.
2. **Only Z-CFL is pre-clamped** — _clamp_cm_cfl! clamps cm before advection but NOT am/bm.
3. **Pole rows zeroed in am** — process_met_after_upload! zeros am[:, 1, :] and am[:, Ny, :].
4. **Multi-outer-loop: same fluxes, evolving mass** — n_sub Strang cycles use same am/bm/cm.

### TM5 reference comparison
- TM5 advectx_get_nloop (advectm_cfl.F90:1354): PER-ROW CFL, simulates mass evolution
- TM5 dynamv (advecty.F90:334): NO Y-subcycling at all
- TM5 dynamw_1d (advectz.F90:409): NO Z-subcycling, allows gamma > 1 via prognostic slopes
- TM5 Check_CFL (advectm_cfl.F90:154): OUTER-LEVEL timestep adaptation (halves ndyn)
