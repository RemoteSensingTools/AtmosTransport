# Seven-day mass drift and memory on V100

Seven consecutive ERA5 C90 L66 daily files (December 1–7, 2018) extend the
one-day conservation check across actual file handoffs. The conservative Dkg
solve reduces worst final Float32 tracer drift from 3.62e-6 to 5.64e-7, and
all six final column-mean fields are closer to the Float64 reference. The
32-tracer run has the same maximum final drift. Float64 remains at roundoff.

![Daily mass drift](../../../../docs/src/assets/weekly_mass_v100.svg)

`check_totals.jl` rechecks the small archived TOMLs locally and writes
`daily_drift.csv`; `plot.py` renders that table as an exportable SVG.

## Daily compensated-total measurements

Six pressure-layer tracers, PPM, TM5 convection, precomputed Dkg, no emissions,
and `preserve_tracer_mass` resets. There are 168 windows and 1,778 transport
substeps. All eight daily snapshots (hours 0 through 168) are complete.

| Day | Seam-only Float32 | Conservative Float32 | Conservative Float64 |
|---|---:|---:|---:|
| 1 | 8.168655e-7 | 3.138140e-7 | 0 |
| 2 | 1.382629e-6 | 4.113739e-7 | 0 |
| 3 | 1.847355e-6 | 4.603499e-7 | 0 |
| 4 | 2.291312e-6 | 4.968472e-7 | 0 |
| 5 | 2.724364e-6 | 5.233122e-7 | 1.982834e-16 |
| 6 | 3.193759e-6 | 5.500987e-7 | 1.982834e-16 |
| 7 | 3.621084e-6 | 5.642842e-7 | 1.982834e-16 |

Each entry is the largest absolute relative change from that tracer's initial
compensated total, sampled at the end of the day. These are not within-day
maximums. At day 7, five Float64 totals equal their initial totals and the sixth
differs by one representable total increment. No normalization is applied.
The six-tracer Float32 improvement is 6.4-fold at day 7; 32 tracers also finish
at a maximum 5.642841741785666e-7. The earlier exploratory 1e-7 target remains
unmet. This week does not define a universal or long-duration error bound.

## Fields and scientific scope

| Tracer | Final relative L2 to Float64, seam-only | Conservative Dkg |
|---|---:|---:|
| 1 | 3.237976e-6 | 1.155342e-6 |
| 2 | 3.728959e-6 | 7.174572e-7 |
| 3 | 3.542778e-6 | 6.967667e-7 |
| 4 | 3.623655e-6 | 6.064909e-7 |
| 5 | 3.800056e-6 | 5.479695e-7 |
| 6 | 3.070686e-6 | 5.300637e-7 |

The reference is a Float64 run of the same transport setup, using legacy
convection; Float32 uses collaborative LU. This supplements independent column
solves and analytic transport tests but is not independent atmospheric forcing
validation. Small negative column means remain, down to about -1.46e-11 mol/mol
among the conservative six-tracer daily snapshots. Positivity is a separate
unresolved property. The archived forcing holds three-hour convection against
hourly transport, as in the earlier experiments.

## Memory and timing scope

| Run | Peak process RSS | Sampled peak device usage |
|---|---:|---:|
| Seam-only Float32, 6 tracers | 5.617 GiB | 1,470 MiB |
| Conservative Float32, 6 tracers | 5.822 GiB | 1,502 MiB |
| Conservative Float64, 6 tracers | 6.300 GiB | 2,976 MiB |
| Conservative Float32, 32 tracers | 5.934 GiB | 2,334 MiB |

GNU `time -v` measures peak process RSS across startup, compilation, transport,
and output checks; it includes resident mmap pages and runtime/compiler memory.
`nvidia-smi` samples device usage every 500 ms, including the CUDA allocator
pool and context; brief peaks can be missed. Only this task uses GPU 0 during
each run. These are different from active array bytes and do not establish
memory scaling beyond this workload. The seven input files total about 21.8 GB;
the process does not retain all their payload pages simultaneously in these runs.

Whole-run times include first compilation in each process: 194.4 s for the
seam-only six-tracer Float32 baseline, 214.2 s for conservative Float32,
752.1 s for conservative Float64, and 310.3 s for conservative 32-tracer
Float32. Do not interpret these as warmed before/after speedups. The dedicated
[warmed diffusion benchmark](../main_dkg_parallel_v100_20260906/README.md)
measures the runtime change with matching outputs. Weekly section timers are
nested and can include synchronization waits; in the Float64 run, legacy
convection accounts for about 518 s of reported section time. Cumulative host
allocation is retained in each result TOML and is not peak RAM.

## Sources, checks, and reproduction

The seam-only export `/tmp/atmos-split-mass-seams` matches `5cb5a7f9`; its
advection-seam and diffusion-operator source hashes were checked explicitly.
The conservative export `/tmp/atmos-dkg-parallel` contains the Dkg launch change
committed as `5db7c780`, before initialization-buffer reuse. Float64 uses the
existing legacy convection solver in every result here.

All four runs pass their finite-output and snapshot-completion checks (30
assertions each for six tracers, 108 for 32). The cross-run validator passes
74 checks of initial totals, daily drift improvement, Float64 roundoff, and
all six final field errors. The current complete CPU suite also passes 120
core files and 628 regridding checks. GPU diffusion checks through 65 tracers
are retained in the preceding experiment.

`run.jl` takes `label precision tracer_count` and writes a distinct
`/tmp/atmos-weekly-drift-<label>-<precision>-<count>/` directory. The archived
`run_with_monitor.sh` locates that script beside itself and takes the source
export as its fourth argument:

```bash
bash /path/to/run_with_monitor.sh conservative Float32 6 /tmp/atmos-dkg-parallel
bash /path/to/run_with_monitor.sh conservative Float64 6 /tmp/atmos-dkg-parallel
bash /path/to/run_with_monitor.sh seam_only Float32 6 /tmp/atmos-split-mass-seams
bash /path/to/run_with_monitor.sh conservative Float32 32 /tmp/atmos-dkg-parallel
```

Runs are sequential on tofu GPU 0, V100-PCIE-16GB,
UUID `GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb`. Julia 1.12.6,
CUDA.jl 5.11.3/runtime 12.6, four Julia threads and one OpenBLAS thread;
GPU scalar indexing is disabled. Each daily input is the same dry format-4,
explicit-dm schema used by the [one-day experiment](../main_dkg_mass_v100_20260905/README.md).
No cache dropping or cold-storage throughput test is performed.

Run `check_outputs.jl` with GPUs hidden from a configured source export. Large
NetCDFs and raw 500 ms device logs stay on tofu; this directory retains config,
result and resource summaries, timing CSVs, and the cross-run check log.
Source review verified complete-file ranges, continuous run time, retained
tracer state across file changes, and input cleanup after prefetch drains.
