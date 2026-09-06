# Cubed-sphere tracer mass drift and Lin–Rood seam correction

The earlier full-day PPM performance workload retained maximum compensated-total
drift of 2.566e-5. Disabling convection and diffusion separately, and repeating
in Float64, shows that horizontal advection contributes the dominant error.
The error persists in Float64; calling it Float32 accumulation was premature.

A small independent contact-map fixture supplies exactly mirrored air-mass
fluxes. Raising its Courant number from about 0.001 to 0.1 reveals tracer
imbalance in both split PPM and Lin–Rood, while air mass remains conserved.
The old small-flux test and its 1e-9 tolerance hid this truncation error.

## Change

Lin–Rood's two panels could calculate different final face mixing ratios from
their transverse predictors. Before either panel applies the final divergence,
the runtime now averages the four inner/outer estimates into one value on both
sides of each physical seam. Interior faces and vertical transport are unchanged.
This uses the connectivity map for reversed tangential indices; the mirrored
air-mass flux supplies the normal sign. There is no global normalization.

The FV3 reference `tp_core.F90:108–242` constructs inner/outer transverse
predictions through halo strips. AtmosTransport's predictors update panel
interiors, retaining original mixing ratios in halo strips. Explicitly sharing
the final estimate is a conservative interface coupling; this change does not
claim literal equivalence to FV3's halo-predictor implementation.

The averaging is a symmetric linear projection, so its transpose is the same
exchange. The tape preserves unmodified inner estimates for predictor reversal,
and the adjoint exchanges all six panels' final face seeds before reversing
the predictors. This adds temporary reverse-pass storage for six panels of
face gradients; it adds no persistent forward workspace.

## Full-day evidence

On tofu's authorized V100, C90 L66, 24 hourly windows, six pressure-layer
tracers, no emissions, TM5 convection and exact Dkg diffusion:

| Precision | Largest absolute final relative drift before | After |
|---|---:|---:|
| Float32 | 3.772281e-5 | 6.980537e-7 |
| Float64 | 3.800515e-5 | 7.931338e-16 |

These are compensated tracer totals, not a normalization target. Initial totals
match exactly. The Float32 improvement is about 54-fold. Float64 uses the
legacy convection solve; Float32 uses the collaborative solve. The forcing
archive is experimental (1-degree three-hour convection held against hourly
transport), so this measures numerical conservation rather than forcing fidelity.

The [reproduction scripts, hourly totals, and validation records](../../scripts/benchmarks/results/main_mass_seams_v100_20260905/README.md)
include the PPM ablation and Lin–Rood before/after runs.

The full horizontal tracer adjoint agrees with finite differences at fixed
meteorology. A joint perturbation of air and tracer mass also exposed the
existing frozen-donor-mass contract of the outer-face adjoints; gradients for
optimizing meteorology are not validated by these checks.

Output inspection finds negative column-mean undershoots in the positive
pressure-layer test both before and after this fix (minimum about -2.09e-10
mol/mol). Correct global totals do not resolve that separate accuracy issue.

## Remaining conservation work

Split PPM, Slopes, and Upwind need a separate seam treatment: a physical contact
can join an X edge to a Y edge, which the directional palindrome updates at
different stages. Merely sharing same-direction faces is insufficient. Any
replacement must preserve paired transfers, signed tracers, uniform mixing
ratios, CFL behavior, and the forward/adjoint identity, with CPU/V100 checks.

The existing q-space emergency cell-local CFL scaling is not conservative when
activated. The new q-space regression uses valid CFL inputs; correcting the
fallback requires conservative subcycling rather than claiming it is equivalent.

A separate sampled-column diffusion probe against a Float64 direct-mass
tridiagonal reference gives maximum Float32 step drift 1.232e-6 and Float64
drift 2.060e-15. No diffusion arithmetic was changed here. Residual Float32
whole-run error needs its own measured budget after the seam corrections.

## Split-PPM follow-up

A CPU prototype groups each physical seam with its canonical owner's axis and
applies paired transfers to both panels. It conserves total storage and uniform
mixing ratio to roundoff, but its timestep refinement is only first order on
the seam fixture (relative field errors 6.15e-5, 3.11e-5, 1.57e-5 for 1, 2,
4 substeps against a 512-substep reference). It remains an archived experiment,
not a production fix. Matching totals alone would have missed this limitation.
The next design needs consistent transverse predictor evolution at the shared
interfaces, along with the corresponding adjoint and a reference-field check.

The original PPM has first-order refinement on the same fixture as well
(errors 7.27e-5, 3.60e-5, 1.80e-5 at 1, 2, 4 substeps). The grouped candidate
has smaller error against the original's refined reference; this does not
establish a temporal regression from grouping. Adoption still requires the
full forward/adjoint, CFL, and real-input validation rather than a totals-only
decision.

The complete 118-file core collection plus regridding passes across the initial
run and the resume after repairing one README file-map entry. The final focused
CS file passes 660 checks, including the additional strong tracer-adjoint test.
The strict docs build passes with deployment disabled.

## Completed split-scheme validation

The [subsequent split-scheme report](2026-09-05_main_split_mass_seams.md)
records production adoption after the required forward/adjoint, CFL,
independent analytic-field, CPU, and V100 checks. The prototype-only status
above describes the earlier checkpoint. Both schemes' Float64 full-day totals
now remain at roundoff; the temporal-order limitation remains documented.
