# Direct cubed-sphere initial-state packing

The CS driven runner previously converted each tracer's interior VMR into six
halo-padded mass arrays, retained them all in a dictionary, then copied them
into the final four-dimensional state panels. Startup now allocates the final
panels once and converts each tracer directly into its packed slot.

The shared private conversion routine zeros the destination halos and uses the
same dry/moist arithmetic as the public allocating packer. The runner calls the
dry path after its existing rejection of moist CS binaries (their windows do
not carry qv). The public packer continues to support moist conversion with qv.
A small name-to-index dictionary preserves the previous packed tracer order;
using the original configuration dictionary directly changes order at some
capacities, including the 17- and 64-tracer cases in the regression suite.

On a local Float32 C90 L66 probe, five alternating measured repetitions after
warmup reduce 32-tracer initialization allocation from 1.357 GB to 0.890 GB,
and median time from 0.721 s to 0.496 s. This compares the preceding construction
algorithm and the new direct algorithm in one process using the shared packer.
The final storage is six `(96,96,66,32)` panels. Accurate host reduction gives
9.999999507942511e34 molecules for the configured 1e35-molecule final tracer,
identically for both paths (Float32 initialization rounding).

The new focused suite passes 558 checks on Float32/Float64, zero and nonzero
halo widths, signed dry/moist conversion, independent slots and inputs, input
shape errors, and state equivalence at 1/6/7/17/32/64 tracers. The current manual,
runtime-flow map, and public packer docstring describe interior VMR versus
halo-padded storage and the direct initialization flow.

Codex diff review checked state ownership, halo initialization, slot/index
mapping, dry/moist semantics, scalar arithmetic order, signed values, and
preservation of the initial air-mass reference. Direct workspace allocation
on the selected device still follows host initialization.

The V100 two-hour 32-tracer workload improves from 5.675 s to 5.030 s median,
with cumulative host allocation reduced from 2.275 GB to 1.808 GB. All 196
output arrays remain exactly equal, and 75 GPU file-handoff physics checks pass.
The full CPU suite passes 83,556 checks with 22 existing skips/expected failures.
The new helper's explicit CS/six-panel signature removes two invalid LL paths
considered by static analysis; focused packing checks pass again and final JET
returns to 142 reports against the unchanged allowance of 144.

An isolated halo probe on the archived GEOS-native mesh measures about 1.68 ms
per 32-tracer exchange, including directional corners, and matches CPU results.
The much larger host halo-section totals include waiting for queued sweeps.
No halo/advection kernel changes were made; future profiling should separate
sweep execution from waits before selecting the next optimization.

Whole-run V100 results and complete CPU validation are recorded in the
[benchmark artifacts](../../scripts/benchmarks/results/main_direct_packing_v100_20260905/README.md).
