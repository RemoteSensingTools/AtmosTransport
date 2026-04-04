---
name: CS transport full algorithmic trace
description: Complete execution trace of cubed-sphere transport from run! through advection/convection/diffusion/output with file:line refs for all three paths (Strang, LinRood, GCHP)
type: reference
---

Full trace document written to `docs/CS_TRANSPORT_TRACE.md`. Covers:

- Phase 0: Model construction (config, grid, driver with mass_flux_dt=450)
- Phase 1: Initialization (IOScheduler double-buffer, 6-panel allocations, PendingIC)
- Phase 2: Per-window loop (upload, physics, PS computation, flux scaling, air mass)
- Phase 3: Three advection paths:
  - **Strang:** X->Y->Z->Z->Y->X per tracer, per-cell mass fixer
  - **LinRood:** fv_tp_2d_cs! -> Z -> Z -> fv_tp_2d_cs!, 3-phase halo+corner protocol
  - **GCHP:** dry or moist basis, gchp_tracer_2d! + vertical_remap_cs!, calcScalingFactor
- Phase 4-8: Emissions (PBL injection), diffusion (DRY basis), chemistry, global mass fixer
- Phase 9: Output (CS regrid or native panels)

Key file locations:
- Main loop: `src/Models/run_loop.jl:27-252`
- CS-specific phases: `src/Models/physics_phases.jl` (1700+ lines)
- Advection dispatch: `src/Models/run_helpers.jl:179-214`
- cm computation: `src/Advection/cubed_sphere_mass_flux.jl:321-337` (_cm_column_cs_kernel!)
- Strang PPM: `src/Advection/cubed_sphere_mass_flux_ppm.jl:458-475`
- Lin-Rood: `src/Advection/cubed_sphere_fvtp2d.jl:562-645` (fv_tp_2d_cs!)
- Z-sweep: `src/Advection/cubed_sphere_mass_flux.jl:1282-1298` (double-buffer critical)
- Per-cell mass fixer: `src/Advection/cubed_sphere_mass_flux.jl:286-298`
- RAS convection: `src/Convection/ras_convection.jl:297-339`
- PBL diffusion: `src/Diffusion/pbl_diffusion.jl:335-359`
- Binary reader: `src/IO/binary_readers.jl:399-419` (load_cs_window! via mmap)
- Met driver: `src/IO/geosfp_cs_met_driver.jl:137-176` (mass_flux_dt field)

Critical CS invariants identified:
1. Hp=3 for PPM-7
2. Halo exchange before every directional sweep
3. Corner fill direction-dependent for Lin-Rood
4. Double buffering in Z-sweep (ws.rm_buf/ws.m_buf)
5. m reset per tracer in Strang path
6. MOIST convection vs DRY advection/diffusion
7. B-correction in cm closure
8. Global mass fixer target = pre-advection mass
