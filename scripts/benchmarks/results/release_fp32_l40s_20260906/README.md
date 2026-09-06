# Float32 release checks and day profiles on L40S, 2026-09-06

The release candidate is `2a19501c3e2292725007854aa97ace2e663d33a5` on
`revamp/current-main`. The software baseline is released **v0.3.0**,
`a0698dde6c97ea3e157ece6ca2d0eca7a0db3465`. Both run on wurst GPU 0, an
NVIDIA L40S, in isolated exports. A fresh current-code run on tofu GPU 0
(Tesla V100) supplies the hardware comparison. No production sources were
changed for these experiments.

## Correctness

All ten maintained GPU diagnostic files pass: packed PPM launches, conservative
Dkg diffusion, TM5 tracer batching, PPM seams and their adjoints, Lin-Rood seams,
snapshot reductions/streaming, transporting adjoints, startup prefetch,
prefetch failure cleanup, and cubed-sphere multi-file handoff.
`gpu-summary.json` lists the exact successful summaries. The maintained suites
exercise Float32 and Float64; the timing study below uses only Float32.
The 703 transporting-adjoint assertions include the standalone recording and
reverse wrappers. Multi-file handoff also agrees with its CPU reference.

The first run passed six files, then one Float64 adjoint case encountered
`LLVM error: No such file or directory` while loading the shared GPUCompiler
runtime cache. An independent job was compiling on GPU 1 at the time. The
cause was not proven; no numerical assertion failed in that run. The affected
adjoint file and remaining three files all pass with a private writable depot.
Both L40S timing processes also use distinct writable depots, with the existing
shared depot as the dependency fallback. The
original failure and successful continuation logs are retained separately.

## Workload and comparison rules

The input is the existing experimental dry format-4 C90 L66 ERA5 archive for
2018-12-01, with explicit endpoint `dm`, TM5 rates, and exact `dkg`. It combines
hourly transport with 1-degree, three-hour convection held between updates;
this is a runtime workload, not independent validation of its meteorological
preparation. Both hosts read an identical 3 GB input with SHA-256
`63b48515658f6b4e7dc8e545ebd6175b1ca1d7561c342e2601e73aceb7ff36ae`.

The loader probe reports Float32 `(96,96,66)` halo-padded panels, interior
level-33 air mass `2.829608e12 kg`, and first-cell surface pressure
`100869.164 Pa`. Each run covers 24 hourly windows and 255 steps, using PPM,
exact TM5 Dkg diffusion, full-column TM5 convection with no layer aggregation,
and 6 or 32 pressure-layer tracers. It writes column means at hours 0 and 24.

Every case has one discarded warmup and five measured repetitions. Separate
persistent Julia processes keep each code version warm; the L40S driver
alternates their order for measured samples. Garbage collection, allocator
reclamation and device synchronization precede each run. The timed region
includes the complete driven simulation and a final CUDA synchronization.
CUDA scalar indexing is disabled. Dependency versions match between code
versions and hosts: Julia 1.12.6, CUDA.jl 5.11.3, runtime 12.6, four Julia
threads and one OpenBLAS thread.

**The 32-tracer solver distinction is explicit.** v0.3.0 rejects
`use_collab_lu=true` above six tracers. Its 32-tracer baseline therefore uses
its supported `use_collab_lu=false` solver. The candidate uses collaborative
batching at both tracer counts. Both use all 66 levels and `n_merge=1`; no
vertical truncation or aggregation is introduced to make the older run fit.
The unsupported request and its error are preserved in the initial log.

Whole-run times include host work and I/O. Wurst has Xeon Platinum 8462Y+ CPUs;
tofu has Xeon Platinum 8168 CPUs. The cross-host ratio is an end-to-end machine
comparison, not a measurement of isolated GPU arithmetic throughput. Section
timers can nest or overlap and must not be summed into a wall-time partition.
Host allocation counts are cumulative bytes, not peak resident memory.

## Results

Five measured runs per cell; values are median seconds for one simulated day.

| Tracers | v0.3 on L40S | Current on L40S | Software speedup | Current on V100 | Wurst/tofu speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 7.350 | 4.541 | 1.62× | 9.670 | 2.13× |
| 32 | 47.468 | 13.841 | 3.43× | 29.164 | 2.11× |

The older 32-tracer row uses its legacy solver; its collaborative path is
unsupported at this tracer count. Six tracers use collaborative LU in both
versions. The speedup denominator is the current L40S median.

| Case | Measured range (s) | Median cumulative host allocation (GB) |
| --- | ---: | ---: |
| v0.3 / L40S, 6 tracers | 7.081–7.574 | 7.251 |
| v0.3 / L40S, 32 tracers | 47.054–47.584 | 13.613 |
| Current / L40S, 6 tracers | 4.471–5.097 | 5.487 |
| Current / L40S, 32 tracers | 13.739–14.178 | 6.019 |
| Current / V100, 6 tracers | 9.348–9.919 | 5.485 |
| Current / V100, 32 tracers | 29.055–29.376 | 6.017 |

`analysis.log` records 3,750 file-completeness/finite/repeatability/conservation
checks and 198 cross-host checks, all passing. Every current column-mean array
is numerically identical between L40S and V100 (maximum absolute difference
zero). The current maximum final mass drift is **3.13814e-7** at both tracer
counts and on both hosts. The older code gives **2.23890e-5** with six tracers
and **2.56564e-5** with 32. Old/new column fields differ by at most
**8.42886e-4 relative L2**; v0.3 predates the conservation fixes and is not an
exact numerical reference for the release candidate.

`analysis.json` also retains the individual timing samples, median GC time,
host allocation counts and section timing medians. Device sampling is retained
for the 32-tracer L40S continuation and the complete V100 run. The initial
L40S sampler was overwritten during the failed 32-tracer restart and is not
used as evidence for six-tracer resource peaks. L40S samples include both
resident worker contexts; they are not a single-model memory comparison.

## Reproduction

The isolated working directory on both hosts is
`/tmp/atmos-l40-release-20260906`. `setup.jl` prepares current and baseline
environments after exporting the two source commits into its `current/` and
`baseline/` directories. Exact environment files and source fingerprints are
retained here. `profile-server.jl` executes each case; `run-profile.py` runs
both versions sequentially on one device and alternates the measured order.
Use a fresh output directory when repeating. The final reproduction driver
rejects existing result files before opening logs or starting workers;
`profile-preflight-check.txt` verifies that it preserves existing logs.

```bash
export JULIA_LOAD_PATH='@:@stdlib' JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 ATMOSTR_TIMERS=1
export CUDA_VISIBLE_DEVICES=GPU-c353397f-bf7b-b51e-d882-9071fb19321c
export ATMOSTR_PROFILE_GPU_NAME=L40S
python3 run-profile.py baseline current
```

For the V100 comparison, select
`GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb`, set the expected name to `V100`, and
run only the `current` worker. The first L40S attempt completed all six-tracer
samples, then stopped at v0.3.0's 32-tracer guard. The continuation sets
`ATMOSTR_PROFILE_TRACERS=32` and `ATMOSTR_PROFILE_SUFFIX=-32`. Final reproduction
code explicitly selects the legacy solver for the older 32-tracer run.

The GPU roster runs with the current environment and an isolated writable
`JULIA_DEPOT_PATH`. `ATMOSTR_GPU_START_CASE=7` reproduces the successful
continuation after the first six files. `analyze.jl` verifies complete time
axes, finite output, within-version repeatability, current-code conservation,
and current L40S/V100 agreement; it also quantifies the old/new column-field
difference because v0.3.0 predates the conservation repairs. Large NetCDF
outputs remain in the temporary experiment directories; the small results,
checks, timing samples and configurations are archived here.
