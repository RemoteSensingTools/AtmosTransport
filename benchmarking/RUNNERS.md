# Benchmark Runner Setup

The benchmark workflow supports CPU on GitHub-hosted runners and optional GPU
lanes on self-hosted runners. GPU jobs are not enabled automatically because
GitHub will queue forever if no matching runner exists.

## CUDA Runner

Register a Linux machine with an NVIDIA GPU as a self-hosted GitHub Actions
runner and add these labels:

```text
self-hosted
linux
gpu-cuda
benchmark
```

The runner should have:

- NVIDIA driver visible to `nvidia-smi`
- Julia installable by `julia-actions/setup-julia`
- enough disk for Julia package artifacts and benchmark outputs

The workflow command uses:

```bash
julia --project=benchmarking benchmarking/run_benchmarks.jl --backend=cuda
```

## Metal Runner

Register an Apple Silicon Mac as a self-hosted runner and add these labels:

```text
self-hosted
macOS
gpu-metal
benchmark
```

The runner should have:

- Apple Silicon GPU visible in `system_profiler SPDisplaysDataType`
- Julia installable by `julia-actions/setup-julia`
- Metal.jl functional in the benchmark environment

Metal benchmark runs use `Float32` only:

```bash
julia --project=benchmarking benchmarking/run_benchmarks.jl --backend=metal --float-type=Float32
```

## Running GPU CI

Use the GitHub Actions `Benchmarks` workflow with manual dispatch:

- `run_cuda=true` to run the CUDA lane
- `run_metal=true` to run the Metal lane
- `publish=true` to merge all available backend artifacts and push dashboard data

The CPU lane always runs. The publish job waits for CPU, CUDA, and Metal jobs,
then merges whichever benchmark artifacts were produced.
