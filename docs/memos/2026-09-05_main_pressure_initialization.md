# Pressure-layer startup allocation

A synthetic C90 L66 startup probe separated VMR construction, tracer-mass
packing, and packed-state construction. For 32 tracers, VMR construction alone
allocated 5.40 GB. `@code_warntype` located the cause: heterogeneous configuration
values remained `Any` after their floating-point constructor calls and carried
that uncertainty through the per-column, per-level pressure search.

Type assertions on the already converted fraction and molecule count reduce
that allocation to 0.42 GB. The layer search still rounds the fraction to model
precision, calculates logarithmic pressure midpoints in Float64, selects the
first closest layer, sums selected dry mass in the same order, and normalizes
the same dry VMR. Signed native initialization and other initializers are unchanged.
The current manual now explains these pressure-layer semantics.

On tofu's V100, the actual two-hour C90 L66 32-tracer workload improves from
15.675 s to 5.675 s median, while cumulative host allocation falls from 7.252 GB
to 2.275 GB. All 196 output arrays remain exactly equal. This is a startup-heavy,
warm-cache workload; it does not imply a 63.5% speedup in transport kernels or
long production runs. See the [scripts, measurements, and validation](../../scripts/benchmarks/results/main_pressure_init_v100_20260905/README.md).

Codex diff review checked conversion precision, lowest-layer short-circuiting,
loop order, ties, normalization, and error behavior. Focused independent layer
and molecule tests, exact comparisons with the preserved prechange function,
existing signed/native and dry/moist initialization tests, Aqua/JET, and the
real V100 output comparisons pass. No GPU kernel or tracer-batch change is part
of this fix.

The remaining synthetic startup allocations include about 467 MB each for
32-tracer mass packing and subsequent state packing. They are separate from the
removed scalar boxing and remain candidates for a measured follow-up.
