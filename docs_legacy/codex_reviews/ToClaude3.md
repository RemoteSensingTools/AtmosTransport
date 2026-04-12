**Findings**
- High: the two concrete bugs from my last review do look fixed now. The pilot no longer aliases its scaled-flux buffers with the CFL scratch buffers in [mass_flux_advection.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L1525), [mass_flux_advection.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L1585), and [mass_flux_advection.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L1648), and the X mass-only pilot now mirrors reduced-grid mass redistribution instead of using the old fine-grid-only update in [mass_flux_advection.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L432).
- Medium: removing the `cm` clamp in [physics_phases.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/physics_phases.jl#L740) was the right correction. That clamp was a non-TM5 stabilizer, and Claude’s “two-sided drain” objection is technically sound.
- High: Claude’s current root-cause statement is still too simplistic. “Strang doubles Y, so `0.82 + 0.82 > 1`” is not a sufficient explanation, because TM5 also uses `xyzzyx`. The real mismatch is still that Julia is doing **global direction-wide doubling** in [advect_y_massflux_subcycled!](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L1604) and [advect_x_massflux_subcycled!](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L1541), while TM5 uses **local evolving-mass `nloop` refinement** in Y/X:
  - [advecty__slopes.F90](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5-mp-r1112/tm5-moguntia-r1112-revised/base/advecty__slopes.F90#L236)
  - [advectx__slopes.F90](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5-mp-r1112/tm5-moguntia-r1112-revised/base/advectx__slopes.F90#L441)
- Medium: Claude is over-crediting TM5 pole exclusion as the primary fix. TM5’s pole exclusion in [advecty.F90](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5/base/src/advecty.F90#L632) affects the tracer/slope update structure, but TM5 still updates air mass through the Y sweep and relies on evolving-mass `nloop`. If your observed failure is negative `m`, then local `nloop` parity is the first thing to match, not pole skipping by itself.
- Medium: Julia already has pole-aware Y face formulas in [mass_flux_advection.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L293). So “implement pole handling” is not starting from zero; the missing part is matching TM5’s overall Y-step structure, not just adding a new pole special case.
- Medium: the prognostic-slope branch is still not a live test path. `prognostic_slopes` is parsed in [configuration.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src/IO/configuration.jl#L968), but the live LL dispatch still forces the diagnostic path in [run_helpers.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/run_helpers.jl#L223). So don’t treat any current run as evidence about prognostic slopes.

**My Take**
- The fixes in `0941316` and `f6bf967` are real improvements. Claude did correct two genuine bugs.
- The current diagnosis then over-rotates. The problem is probably **not** “Strang is fundamentally impossible near the poles”; TM5 disproves that. The problem is that the Julia implementation still does not match TM5’s **local** timestep refinement behavior in Y/X, and Claude is trying to explain a TM5-survives / Julia-fails difference without first matching that.
- I would **not** do “task #13 first” if that means an ad hoc pole exclusion patch. That risks masking the real stepping mismatch.

**Suggested Next Moves**
- Implement TM5-style **local Y `nloop`** first, using [advecty__slopes.F90](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5-mp-r1112/tm5-moguntia-r1112-revised/base/advecty__slopes.F90#L236) as the model.
- Keep the current F64 moist debug config and test one failing window only.
- After local Y `nloop` is in place, re-run the uniform case and the off-pole pulse case.
- Only if that still fails specifically at the poles should Claude restructure the Y update to match TM5’s pole-exclusion layout from [advecty.F90](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5/base/src/advecty.F90#L632).
- Leave prognostic slopes off until the mass path is stable.

**No-Gos**
- Don’t patch the preprocessor to damp polar `bm`.
- Don’t add another mass fixer or `m_ref` reset.
- Don’t treat “Y is applied twice” as the root cause; TM5 does that too.
- Don’t use the current prognostic-slope code as a debugging path yet.
- Don’t move back to L40S/F32 until the A100/F64 path is stable.

The package import check was still precompiling when I stopped polling, so this review is static rather than test-backed.
