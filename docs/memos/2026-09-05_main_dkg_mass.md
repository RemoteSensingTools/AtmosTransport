# Precomputed Dkg mass conservation follow-up

Baseline: `5cb5a7f9` on `revamp/current-main`, containing current main `a0698dde`.
The [archived experiment](../../scripts/benchmarks/results/main_dkg_mass_v100_20260905/README.md)
contains scripts, final evidence, and the unsuccessful exploratory target.

Ablation after paired advection seams isolates precomputed diffusion as the
largest remaining Float32 drift source: disabling it lowers six-tracer daily
drift from 8.17e-7 to 5.05e-9. The new direct-mass backward-Euler factorization
reduces the full-physics result to 3.14e-7 while retaining small exchanges.
Float64 final compensated totals match initialization; worst hourly drift is
1.98e-16. The same maximum final Float32 drift occurs with 32 tracers.

## Reference and implementation

Before changing transport, inspected actual mass and Dkg array shapes, units,
and magnitudes, and read TM5's `tm5_diff.F90:36–129` tridiagonal construction.
The comparison identifies coefficient rounding and the mass/VMR round trip,
not a missing source or boundary exchange. Independent Float64 tridiagonal
mass solves validate the mathematical operator.

`conservative_dkg.jl` factors the mass matrix into two column-conservative
bidiagonal matrices. Their inverses retain mass locally and transfer the rest
to a neighbor. The [manual derivation](../src/theory/mass_conservation.md)
gives the ratios and boundary conventions. Small partitions are computed
directly, incoming sums carry a rounding residual, and a stationary background
can be removed for profiles of one sign. Exactly isolated layers bypass the
background transform. All six workspace panel shapes are checked before state
mutation. Packed/scalar paths share the column function and existing scratch.

The tracer adjoint transposes these same passes with fixed meteorology. Constant
mass-objective seeds are exact; analytic weak recipient sensitivities and
independent dot products pass. This is a transpose of the mathematical solve,
not differentiation of floating-point rounding decisions. The public array-level
VMR API retains Thomas elimination; other diffusion fields/geometries and the
surface source schedule retain their existing behavior.

## Rejected lower-drift candidate

Computing outgoing mass as input minus rounded retention reduced the daily
Float32 result to 9.75e-8 but deleted sub-ulp physical exchange into empty cells.
It also lost relative accuracy for weak Float64 recipients. That candidate was
rejected. The final ratio representation computes the smaller partition directly
and passes recipient-relative analytic tests down to dt*D/m=1e-14.

The exploratory 1e-7 daily target fails for four final Float32 tracers. The
failure log is retained and the final comparison reports the target as unmet.
No maintained numerical test tolerance was loosened. Worst sampled-column
Float32 mass error improves about 12-fold and independent field error about
7-fold; all six full-day column-mean fields are closer to the Float64 run.
Those checks, preserved weak exchanges, and improved totals justify the change.
There is no global or column normalization and no positivity clamp.

## Cost, review, and scope

The original 32-tracer daily benchmark takes 38.635 s median versus 34.574 s
with paired seams alone (11.7% more). Cumulative host allocation is 6.414 GB
versus 6.419 GB; no persistent factor or tracer buffers are added. Peak memory
is unmeasured. The extra arithmetic is accepted for the requested reduction
in mass drift. Future tuning must preserve small recipient values, stiff and
signed solutions, scalar/packed consistency, and adjoint sensitivities.

Critical Codex self-review checked include order, dispatch and source coupling,
bidiagonal factor algebra and transpose, fixed-meteorology assumptions,
nonpositive-carrier behavior, closed interfaces, exact no-exchange identity,
weak-transfer cancellation, scratch ownership, and the independent field and
full-day evidence. No additional agent was used. The full 120-file CPU core
suite plus regridding passes, including Aqua 10 and JET 152 against its updated
baseline (four known launch reports added by two new kernels). Focused CPU
checks pass 12,087 assertions on Julia 1.12.6 and 1.10.12; V100 checks pass
6,989,800 through 65 tracers. The strict documentation build and 160 local
documentation/link/map checks pass.

Conservation does not establish positivity: existing PPM column-mean negative
undershoots remain around -4.27e-11 mol/mol. The numerical experiment covers one
24-hour archive, not multi-day drift, independent forcing fidelity, or every GPU.
Further reductions below Float32 storage roundoff require a separately validated
precision strategy. Erasing small fluxes is not an acceptable substitute.
