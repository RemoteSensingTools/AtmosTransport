Below is a handoff note for Claude.

**Handoff For Claude**

Goal: get a **TM5-faithful LL forward run in `Float64` on the A100 first**, then worry about `Float32` on L40S.

**What TM5 Actually Does**
- Original TM5 protects the mass path by checking CFL on the **evolving mass path**, then reducing timestep globally if needed:
  - [advectm_cfl.F90#L205](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5/base/src/advectm_cfl.F90#L205)
  - [advectm_cfl.F90#L255](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5/base/src/advectm_cfl.F90#L255)
  - [advectm_cfl.F90#L278](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5/base/src/advectm_cfl.F90#L278)
  - [advectm_cfl.F90#L287](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5/base/src/advectm_cfl.F90#L287)
- Original TM5 explicitly checks meridional `beta = bm/m` on evolving mass, including the pole-adjacent face:
  - [advectm_cfl.F90#L2194](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5/base/src/advectm_cfl.F90#L2194)
  - [advectm_cfl.F90#L2207](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5/base/src/advectm_cfl.F90#L2207)
- Original TM5 Y advection excludes pole cells from the interior sweep and handles poles separately:
  - [advecty.F90#L632](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5/base/src/advecty.F90#L632)
  - [advecty.F90#L646](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5/base/src/advecty.F90#L646)
- Newer TM5 forward code goes further and uses **local evolving-mass loop refinement** in X and Y:
  - [advectx__slopes.F90#L441](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5-mp-r1112/tm5-moguntia-r1112-revised/base/advectx__slopes.F90#L441)
  - [advectx__slopes.F90#L499](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5-mp-r1112/tm5-moguntia-r1112-revised/base/advectx__slopes.F90#L499)
  - [advecty__slopes.F90#L236](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5-mp-r1112/tm5-moguntia-r1112-revised/base/advecty__slopes.F90#L236)
  - [advecty__slopes.F90#L280](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5-mp-r1112/tm5-moguntia-r1112-revised/base/advecty__slopes.F90#L280)
  - [advectx.F90#L617](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5-cy3-4dvar/base/src/advectx.F90#L617)
- TM5 forward transport mass is **moist/total-air mass**, not dry mass:
  - [grid_3d.F90#L816](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5-cy3-4dvar/base/src/grid_3d.F90#L816)
  - [meteo.F90#L4846](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5-cy3-4dvar/base/src/meteo.F90#L4846)

**What Is Still Wrong In AtmosTransport**
- Current LL X/Y/Z subcycling still uses **one pre-loop CFL** and then uniform subdivision on evolving mass:
  - [mass_flux_advection.jl#L1443](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L1443)
  - [mass_flux_advection.jl#L1467](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L1467)
  - [mass_flux_advection.jl#L1498](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L1498)
- That is the main runtime mismatch. It is a better explanation for the current instability than the binary data.
- The current prognostic-slopes branch is **not** a valid TM5 path yet:
  - Z slope update is stubbed out in [_prognostic_z_kernel!](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L499)
  - the “prognostic” Strang path still uses diagnostic X/Y and only prognostic Z in [strang_split_prognostic!](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L1729)
- LL runtime still applies non-TM5 pole treatment by zeroing `am` at pole rows:
  - [physics_phases.jl#L346](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/physics_phases.jl#L346)
  - [physics_phases.jl#L757](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/physics_phases.jl#L757)
- LL runtime still clamps `cm` as a stabilization hack:
  - [physics_phases.jl#L744](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/physics_phases.jl#L744)
- Current LL output is **not TM5-faithful** if QV is loaded, because it dry-corrects output mass:
  - [run_loop.jl#L249](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/run_loop.jl#L249)
  - [physics_phases.jl#L1747](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/physics_phases.jl#L1747)
  - [physics_phases.jl#L1756](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/physics_phases.jl#L1756)

**Immediate Plan**
1. Make a new clean A100/F64 debug config.
- `Float64`
- `mass_fixer = false`
- `dry_ic = false`
- `prognostic_slopes = false` initially
- no emissions, no diffusion, no convection, no chemistry
- disable LL QV loading for this debug run so output stays moist `rm / m`
- one tracer only

2. Do not use the current “nofixer” test config as evidence.
- It is not clean:
  - [era5_f64_nofixer_2day.toml#L27](/home/cfranken/code/gitHub/AtmosTransportModel/config/runs/era5_f64_nofixer_2day.toml#L27)
  - [era5_f64_nofixer_2day.toml#L38](/home/cfranken/code/gitHub/AtmosTransportModel/config/runs/era5_f64_nofixer_2day.toml#L38)

3. Fix the mass-stepping logic before touching slopes.
- Implement TM5-style evolving-mass local loop refinement for X and Y first.
- Use the newer TM5 local-loop code as the implementation template, and the original `Check_CFL` code as the conceptual guardrail.
- The correct first target is: keep `m` positive without `mass_fixer`, with diagnostic slopes still in place.

4. Instrument the actual runtime per sweep.
- After each of `X1, Y1, Z1, Z2, Y2, X2` log:
  - `min(m)`
  - first nonpositive `m` location
  - max local `alpha/beta/gamma`
  - local loop count chosen
- Do this on one failing window, not a long run.

5. Only after the mass path is stable:
- revisit full prognostic slopes
- remove `cm` clamp
- revisit pole-row handling
- revisit exact TM5 level selection

**No-Gos**
- Do **not** keep iterating on the current hybrid prognostic-slopes branch.
- Do **not** add another mass fixer, `m_ref` reset, or output-side workaround.
- Do **not** treat dry output as TM5-faithful for the F64 debug run.
- Do **not** spend more time on binary/preprocessor correctness unless a new source-vs-binary mismatch appears at the failing hotspot.
- Do **not** let L40S/Float32 constraints drive the next design step. Use the A100/F64 run to prove the logic first.
- Do **not** chase the `137` vs `68` config mismatch. The grid is auto-rebuilt from the binary header in [configuration.jl#L217](/home/cfranken/code/gitHub/AtmosTransportModel/src/IO/configuration.jl#L217).

**Acceptance Criteria For The F64 Phase**
- Uniform IC: preserved to machine precision.
- Single localized off-pole source: no NaN, no negative `m`, no global striping.
- Real IC over a short run: no negative `m`, no NaN, tracer mass drift small and explainable.
- Output basis during this debug phase: moist `rm / m`, not dry VMR.

After that works in F64 on the A100, the F32 phase is separate:
- keep the same logic
- then address the remaining F32-sensitive spots like CPU reduced-grid accumulation and donor-mass safety margins.
