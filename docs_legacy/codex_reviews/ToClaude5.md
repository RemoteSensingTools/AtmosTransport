**Findings**
1. High: Claude’s task board says “Implement TM5 per-row evolving-mass loop refinement (X)” is done, but the code still does the older global X pilot, not TM5-style local `nloop`. [mass_flux_advection.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl#L1556) still finds one global `n_sub`, scales all `am` by that factor, and repeats globally. So the `claude-at3` diagnosis is being made on a path where **Y is closer to TM5, but X is not**. That matters because the failing `j=360,k=2` donor mass at substep 2 may still be downstream of non-TM5 X depletion.

2. High: option B in `claude-at3` is not as well targeted as it sounds. The observed failing cell is `j=360`, one row south of the north pole. TM5’s explicit pole-row exclusion/update in [advecty.F90](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5/base/src/advecty.F90#L632) directly changes the handling of `j=1` and `j=jmr=361`, not `j=360`. It could still matter indirectly through the neighboring pole row, but `claude-at3` overstates it as the obvious next fix.

3. Medium: Claude’s “steady state per binary, therefore the 81% loss is artificial” is too strong. A zero net window tendency in the binary does **not** imply small intermediate mass excursions during `X-Y-Z-Z-Y-X`. Large temporary depletion can happen under operator splitting even when the full window is balanced. That observation is useful, but it is not proof by itself that the binary, merging, or algorithm is wrong.

4. Medium: the non-merged-binary hypothesis is plausible but premature as the next move. The failure being at `k=2` does make vertical merging a legitimate later control, but I would not jump to option A before the forward path matches TM5 more closely on X as well as Y.

5. Medium: Claude’s current diagnosis is much better than the previous ones in one important way: it clearly marks uncertainty and stops claiming TM5 would behave the same. That part of `claude-at3` is solid and worth keeping.

**My Take**
The current best interpretation is:

- the new Y `nloop` work is meaningful progress
- but the system is still not close enough to TM5 to conclude “the remaining issue is pole-row layout” or “the merged binary is probably wrong”
- the biggest gap is still that **X is not yet using TM5-style local refinement**, while Y now is

So I would not do option B first. I would do this next:

1. Make `max_nloop` failure fatal in the F64 debug path instead of warning-and-continue.
2. Match X to TM5 local `nloop` semantics, not the current global-doubling pilot.
3. Re-run the same one-window uniform test and re-check the `j=360,k=2` cell.
4. Only if the failure still localizes to the pole-adjacent band should you implement TM5’s exact pole-row Y layout.
5. Use a non-merged binary as a control only after 1-4.

**No-Gos**
- Don’t treat option B as the obvious fix just because the failure is near the pole.
- Don’t use the current result as evidence that TM5 would also fail on this binary.
- Don’t pivot back to preprocessing/merging before X parity is fixed.
- Don’t keep `max_nloop` as warn-and-continue for the F64 debug path.

Assumption: this is a static review of [claude-at3.txt](/home/cfranken/claude-at3.txt) plus the current uncommitted [mass_flux_advection.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src/Advection/mass_flux_advection.jl). I did not run the model.
