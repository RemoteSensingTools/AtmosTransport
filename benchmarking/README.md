# AtmosTransport Benchmarks

This directory contains the production benchmark harness for tracking
AtmosTransport performance over time. It follows the same broad pattern as the
Oceananigans benchmark dashboard: Julia produces raw benchmark records plus a
`github-action-benchmark` compatible JSON file; CI or a dedicated runner
publishes the history.

Prepare the environment from the repository root, including on Julia 1.10
where `[sources]` is ignored:

```bash
julia --project=benchmarking -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

AtmosTransport is sourced from this checkout without a duplicated release
version bound, so the benchmark environment stays usable after version bumps.

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
