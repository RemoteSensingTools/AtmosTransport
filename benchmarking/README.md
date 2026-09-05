# AtmosTransport Benchmarks

This directory contains the production benchmark harness for tracking
AtmosTransport performance over time. It follows the same broad pattern as the
Oceananigans benchmark dashboard: Julia produces raw benchmark records plus a
`github-action-benchmark` compatible JSON file; CI or a dedicated runner
publishes the history.

Example:

```bash
julia --project=benchmarking benchmarking/run_benchmarks.jl \
  --backend=cpu \
  --float-type=Float32,Float64 \
  --grid=C24,C48 \
  --levels=32 \
  --tracers=1,4 \
  --operator=advection,diffusion,convection,full,io,adjoint \
  --steps=5 \
  --repeats=5 \
  --output=benchmark_results.json
```

Backend labels are intentionally hardware-class labels:

- `CPU-AMD`
- `CPU-Intel`
- `CPU-Apple`
- `GPU-CUDA`
- `GPU-Metal`

The exact processor or GPU name is stored in each record's metadata and in the
dashboard record `extra` field. Metal runs are restricted to `Float32`.

The GitHub workflow in `.github/workflows/Benchmarks.yml` runs tiny CPU smokes
on pull requests and pushes to `main`, and can publish heavier manual-dispatch
runs under the Pages `benchmarks` directory.

GPU benchmark jobs are manual opt-in lanes. Register self-hosted runners with
these labels:

- CUDA Linux runner: `self-hosted`, `linux`, `gpu-cuda`
- Apple Silicon runner: `self-hosted`, `macOS`, `gpu-metal`

Then start the `Benchmarks` workflow manually and set `run_cuda=true` and/or
`run_metal=true`. CPU runs default to `Float32,Float64`; GPU runs default to
`Float32` and can be changed with the `gpu_float_types` workflow input when the
runner hardware supports it. The publish job merges CPU, CUDA, and Metal
artifacts into one dashboard input so concurrent backend jobs do not race while
pushing chart data.

## Binary-to-NetCDF pipeline

The `io` operator now measures real snapshot capture, NetCDF writing and reading,
including every tracer. It no longer measures Julia serialization of one tracer.

For a synthetic full runtime benchmark (LL, RG, CS; one/four tracers; full versus
column-only output):

```bash
julia --project=benchmarking benchmarking/run_pipeline_benchmarks.jl cpu pipeline_cpu.json
CUDA_VISIBLE_DEVICES=0 julia --project=benchmarking benchmarking/run_pipeline_benchmarks.jl cuda pipeline_a100.json
```

The CUDA command verifies an A100 by default. Set
`ATMOSTR_BENCH_GPU_NAME=V100` for an explicitly authorized V100 run. The V100
measurements used CUDA.jl 5.11.3 with runtime 12.6; a CUDA 13.2 runtime rejected
that device. Runtime selection belongs to the isolated benchmark environment.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `ATMOSTR_BENCH_NC` | `12` | CS panel width; LL/RG synthetic resolution uses the same size parameter. |
| `ATMOSTR_BENCH_NZ` | `16` | Vertical levels. |
| `ATMOSTR_BENCH_REPEATS` | `3` | Measured repetitions after warmup. |
| `ATMOSTR_BENCH_TRACERS` | `1,4` | Comma-separated positive tracer counts. |
| `ATMOSTR_BENCH_TOPOLOGIES` | `ll,rg,cs` | Grid layouts to measure. |
| `ATMOSTR_BENCH_FILES` | `1` | Repeated input files, each containing two one-hour windows. |
| `ATMOSTR_BENCH_REVISION` | `unspecified` | Revision label recorded with the measurement. |

Results record elapsed-time samples, host allocations, input/output bytes,
tracer count, selected fields, Julia/CUDA versions, and device. Every tracer
and output time is read back and checked. These are warm OS page-cache timings,
not cold NAS throughput or device-memory peaks. Fixtures use synthetic forcing
and do not establish campaign-scale throughput.
