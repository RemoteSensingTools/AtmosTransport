# Matrix convection on tofu's V100

Measured 2026-09-05 on `tofu.gps.caltech.edu`, GPU 0: Tesla V100-PCIE-16GB,
compute capability 7.0, driver 580.178.04. GPU 1 was not used. Both devices
were idle before the experiment. The user explicitly authorized this V100 run.

## Environment and comparison

Julia 1.12.6, CUDA.jl 5.11.3, KernelAbstractions 0.9.41, CUDA toolkit 12.6.
The initially selected CUDA 13.2 toolkit rejected compute capability 7.0 before
kernel compilation. Setting `CUDA.set_runtime_version!(v"12.6")` in each
isolated benchmark environment resolved that issue. The repository's normal
environment and the system driver were not changed.

Two clean source exports used the same benchmark script and dependencies:

- `3f8adc5e`: before tracer batching, structured LU, and the bidiagonal RHS solve.
- `9f5008db`: all three optimizations enabled through normal kernel selection.

Each case has 4,096 synthetic Float32 face-indexed columns; every column
convects, with cloud tops spread over levels 1–8. Separate cases have no
downdrafts or finite downdrafts in every column. There is no truncation or
vertical aggregation in the timings. The benchmark covers 60 and 85 levels
and 1, 6, 7, 12, 32, and 65 tracers.

The old kernel accepts at most six tracers per launch. For larger counts,
the baseline uses independent batches of at most six tracers and repeats
matrix construction/factorization for each launch. This is an explicit
workaround, not native support in that version. The new kernel handles every
tracer in one launch, retaining factors between its internal batches.

Reported times are CUDA-event medians of nine warmed single-step samples.
Each sample starts from the same RHS, restored outside the timed interval.
Allocation, compilation, host/device transfers, forcing derivation, and I/O
are excluded. Synchronized wall times and individual samples are also saved.
These are convection-kernel measurements, not end-to-end model speedups.

## Results at 85 levels

| Column forcing | Tracers | Old batches, ms | Current, ms | Speedup |
| --- | --- | --- | --- | --- |
| Updraft only | 1 | 21.33 | 15.71 | 1.36× |
| Updraft only | 6 | 21.38 | 15.85 | 1.35× |
| Updraft only | 7 | 42.74 | 16.97 | 2.52× |
| Updraft only | 32 | 128.35 | 22.13 | 5.80× |
| Updraft only | 65 | 235.71 | 28.59 | 8.24× |
| With downdrafts | 6 | 22.99 | 24.41 | 0.94× |
| With downdrafts | 32 | 138.13 | 35.94 | 3.84× |
| With downdrafts | 65 | 253.45 | 47.62 | 5.32× |

A second current-version sweep after the baseline changed all 85-level
medians by less than 0.5%. The six-tracer downdraft regression is repeatable:
about 6% more time than the old kernel. At 60 levels, six-tracer downdrafts
are about 4% slower. The earliest 60-level case varied by up to 9.5% between
sweeps, so short initial measurements are less stable than the 85-level cases.
GPU clocks were not locked; no power or clock settings were changed.

## Shared memory and remaining work

Compiled resource queries report 32,420 bytes of shared memory per 85-level
column, three resident 32-thread blocks per multiprocessor, and only 4.6875%
theoretical warp occupancy. At 60 levels, the values are 16,900 bytes, five
blocks, and 7.8125%. These are driver occupancy limits, not measured achieved
occupancy. Shared-memory allocation stays constant from one to 65 tracers.

The current kernel uses 176 registers per thread, compared with 105 in the
baseline. Both report 32 bytes of local memory per thread and identical shared
memory/occupancy limits. The higher register count alone does not establish
the cause of the small-tracer downdraft regression; that needs profiling.

The six-tracer total limit is removed. The remaining storage constraint is
the dense shared matrix, which limits resident warps even though LU arithmetic
is cheaper for updraft-only columns. Increasing the RHS batch capacity would
also increase shared memory and may reduce residency; it needs measurement.
The serial matrix build and the six active RHS lanes per batch are additional
profiling targets. No production kernel changes were made during this check.

## Correctness

The opt-in GPU suite passes 487 assertions on V100: deferred scratch adaptation
and all three topologies, cloud-free and convecting columns, no/finite/tiny
downdrafts, 1–65 tracers, inactive upper levels, CS halos, mass conservation,
and bitwise equality between one multi-tracer launch and separate <=6-tracer
launches. Device results agree with explicitly dense CPU LU within Float32
tolerance. The performance sweep additionally checks four dense-reference
columns per case and conservation/positivity over every column. Its maximum
normalized reference error is 1.80e-7 and maximum relative mass error 5.61e-8.

CUDA 12.6 Compute Sanitizer reports zero memory errors and zero race hazards
(errors or warnings). Each tool runs six deep-column fixtures: all three
topologies, no/finite downdrafts, 91 physical levels with depth 85, and 65
tracers, including independent split launches. Each run passes 40 assertions.
The full opt-in suite is skipped in these targeted sanitizer runs.
[Recorded validation output](../../scripts/benchmarks/results/matrix_convection_v100_20260905/validation.txt).

## Reproduction and artifacts

Use the [GPU benchmark script](../../scripts/benchmarks/bench_matrix_convection_gpu.jl)
in the source checkout being measured. In an isolated environment:

```bash
julia --project=. -e 'using CUDA; CUDA.set_runtime_version!(v"12.6")'
CUDA_VISIBLE_DEVICES=0 JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  ATMOSTR_MATRIX_GPU_NAME=V100 ATMOSTR_MATRIX_BENCH_REVISION=9f5008db \
  julia --project=. scripts/benchmarks/bench_matrix_convection_gpu.jl current.toml native
CUDA_VISIBLE_DEVICES=0 ATMOSTR_MATRIX_GPU_NAME=V100 \
  ATMOSTR_RUN_MATRIX_BATCH_GPU_TESTS=1 \
  julia --project=. test/diagnostic/test_tm5_tracer_batching_gpu.jl
```

For the old export, copy the same benchmark script into it, use revision
`3f8adc5e`, and replace `native` with `split`. A100 remains the default device
guard for these tools; the name override explicitly selects V100 validation.

For the sanitizer checks, run the following with `TOOL=memcheck`, then with
`TOOL=racecheck`, keeping `ATMOSTR_RUN_MATRIX_BATCH_GPU_TESTS` unset:

```bash
CUDA_VISIBLE_DEVICES=0 JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /usr/local/cuda-12.6/bin/compute-sanitizer --tool "$TOOL" --error-exitcode 1 \
  julia --project=. -e '
    using CUDA, Test
    include("test/diagnostic/test_tm5_tracer_batching_gpu.jl")
    @assert occursin("V100", CUDA.name(CUDA.device()))
    CUDA.allowscalar(false)
    @testset "V100 deep-column sanitizer fixtures" begin
        for topology in (:ll,:rg,:cs), downdrafts in (false,true)
            check_batches(batch_fixture(91,85,65,downdrafts),topology)
        end
    end'
```

Raw measurements: [current](../../scripts/benchmarks/results/matrix_convection_v100_20260905/current.toml),
[baseline](../../scripts/benchmarks/results/matrix_convection_v100_20260905/baseline.toml),
[current repeat](../../scripts/benchmarks/results/matrix_convection_v100_20260905/current_repeat.toml).
