**Findings**
1. High: Claude’s claim that “the whole path is now TM5-consistent at `cfl_limit = 1.0`” is still false in the live LL path. X and Y now honor the caller threshold in [mass_flux_advection.jl#L1862](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L1862) and [mass_flux_advection.jl#L2123](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L2123), but the active caller still passes `0.95` in [physics_phases.jl#L792](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/physics_phases.jl#L792). So `dc90eac` fixed the plumbing bug, but not the runtime TM5-threshold mismatch.

2. High: the “`am` pole zeroing is TM5-equivalent” issue was not actually resolved in code, only re-labeled in comments. The runtime still zeros the first and last latitude rows in [physics_phases.jl#L362](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/physics_phases.jl#L362), [physics_phases.jl#L363](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/physics_phases.jl#L363), [physics_phases.jl#L776](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/physics_phases.jl#L776), and [physics_phases.jl#L777](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/physics_phases.jl#L777). TM5 itself just fills `am = dtu * pu` on all real rows in [advect_tools.F90#L767](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5/base/src/advect_tools.F90#L767) through [advect_tools.F90#L770](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5/base/src/advect_tools.F90#L770). The new comment in [physics_phases.jl#L341](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/physics_phases.jl#L341) overstates the case by treating pole-adjacent real cells as if their zonal flux were physically zero by construction.

3. Medium: the real code improvement in `dc90eac` is the X/Y threshold consistency fix. That part is good: X no longer hardcodes `1.0`, and Y no longer ignores the caller. The relevant change is in [mass_flux_advection.jl#L1851](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L1851) through [mass_flux_advection.jl#L1862](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L1862), and [mass_flux_advection.jl#L2111](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L2111) through [mass_flux_advection.jl#L2123](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L2123).

4. Medium: Claude’s option `(a)` is still too strong. A pass on coarser or non-merged data would not prove “the algorithm is correct and the binary is the cause.” It would only show the code survives a less stiff case. The live path still has known parity gaps beyond the binary, including disabled prognostic slopes in [run_helpers.jl#L223](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/run_helpers.jl#L223) and the custom merged debug binary in [era5_f64_debug_moist.toml#L31](/home/cfranken/code/gitHub/AtmosTransportModel/config/runs/era5_f64_debug_moist.toml#L31).

5. Low: the statement that TM5’s special pole-row Y layout would be bit-identical for the uniform-IC tracer test is plausible for the tracer-slope term, but it does not settle the air-mass path. I would treat that as an informed hypothesis, not a conclusion.

**My take**
`dc90eac` is a partial cleanup, not a resolution. The meaningful fix is the X/Y `cfl_limit` consistency. The pole-row `am` issue is still a hypothesis dressed up as a conclusion.

**What I’d do next**
- Make the active F64 debug path actually use `cfl_limit = 1.0` in [physics_phases.jl#L792](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/physics_phases.jl#L792), then rerun the same one-window case.
- Treat pole-row `am` zeroing as provisional until it is validated against the upstream preprocessed/intermediate data. If it is truly spectral garbage, fix or zero it in preprocessing; if not, remove the runtime zeroing.
- Only after that use a non-merged or coarser control. Use it as a stress-reduction experiment, not as proof of correctness.

**No-gos**
- Don’t claim TM5-equivalence for pole-row `am` yet.
- Don’t say the path is at TM5 `1.0` while [physics_phases.jl#L792](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/physics_phases.jl#L792) still forces `0.95`.
- Don’t use a coarse-data pass as evidence that the current algorithm is correct.

Static review only: I checked [claude-at5.txt](/home/cfranken/claude-at5.txt) and commit `dc90eac`, but I did not run the model.
