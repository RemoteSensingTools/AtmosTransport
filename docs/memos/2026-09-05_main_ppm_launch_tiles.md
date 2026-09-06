# CUDA PPM launch tiles

Synchronized profiling resolved the apparent halo bottleneck: much of the host
halo-section time was waiting for previously queued PPM sweeps. An isolated
launch-layout comparison then found that the scalar 256-thread workgroup
becomes `(256, 1, 1)` in KernelAbstractions. At C90, only 35.2% of those threads
address interior cells. A `(32, 2)` tile raises that fraction to 93.75% while
keeping each warp's `i` cells contiguous.

The CUDA extension now selects this tile for packed Float32 PPM sweeps. The
kernel arithmetic and serial tracer loop are unchanged; all six higher-order
packed wrappers (X/Y/Z, ping-pong/copy-back) use the same policy. CPU, Metal,
Float64 CUDA, and other schemes keep their defaults. A separate experiment
parallelizing tracers as a fourth launch dimension was slower and was rejected.

On tofu GPU 0 (V100), the isolated 32-tracer panel sweep medians change from
4.847 to 2.430 ms (X), 1.683 to 0.969 ms (Y), and 1.713 to 1.117 ms (Z).
The real two-hour C90 L66 workload changes from 5.030 to 4.466 s with 32 tracers
and from 1.753 to 1.591 s with six tracers. Host allocation is essentially
unchanged. These short warm-cache medians use two measured repetitions after
warmup; they do not establish performance on other grids or NVIDIA devices.

All 196 output arrays remain exactly equal, with unchanged compensated-total
mass drift. The new opt-in V100 diagnostic passes 540 checks through 65 tracers,
including signed storage, halos, CPU reference agreement, and both wrapper
paths. Focused CPU advection, Aqua, and JET checks pass 619 assertions; JET
remains 142/144. The full 83,556-check suite passed at the preceding checkpoint
and was not repeated for this launch-policy-only change.

Codex diff review checked extension dispatch, module include order, unchanged
per-cell arithmetic, launch coverage at nonmultiples of the tile dimensions,
input/output ownership, and both copy-back and ping-pong callers. Performance
claims are restricted to measured V100 workloads; active-thread fractions are
not presented as hardware occupancy measurements.

See the [reproduction scripts and results](../../scripts/benchmarks/results/main_ppm_tiles_v100_20260905/README.md).
