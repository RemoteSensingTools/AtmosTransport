# Installation

This page covers getting AtmosTransport.jl onto a machine and verifying the
install with the synthetic-fixture test suite. No external met data is
required to complete this page.

## Prerequisites

| Requirement | Notes |
|---|---|
| **Julia** | 1.10 or newer; tested on 1.10 and 1.12. [juliaup](https://github.com/JuliaLang/juliaup) is the recommended installer. |
| **Git** | Standard. |
| **Disk** | ~2 GB for source + dependencies + precompiled artifacts. Real meteorological datasets are separate (tens to hundreds of GB). |
| **GPU backend** *(optional)* | NVIDIA CUDA or Apple Silicon Metal can enable GPU acceleration. The CPU backend works without any GPU drivers. |

Metal runs require `Float32` numerics. The runtime rejects Metal with
`float_type = "Float64"` before model construction.

## Get the code

```bash
git clone https://github.com/RemoteSensingTools/AtmosTransport.git
cd AtmosTransport
```

## Install Julia dependencies

The repository pins its dependencies via `Project.toml` and `Manifest.toml`.
Instantiate them once:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Dependencies include `KernelAbstractions`, `NCDatasets`, `FFTW`,
`StaticArrays`, and (as weak deps) `CUDA` and `Metal`. Plan on a ~5–10 minute
first-time precompile.

## Verify the install

The core test suite runs end-to-end on **synthetic fixtures** — no external
met data needed. It exercises the preprocessor, the runtime, and the
diagnostic tools across all three grid topologies.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected output (abbreviated):

```text
Test Summary:                | Pass  Total  Time
Met source trait             |   18     18  0.2s
IdentityRegrid passthrough   |   15     15  3.7s
GEOS reader                  |   48     48  8.3s
GEOS → CS passthrough        | 3467   3467 11.4s
GEOS convection wiring       |   26     26 12.8s
…
```

If `Pkg.test()` finishes green, the install is sound and you can proceed
to [First run](@ref).

## Optional: GPU stack

If you have a GPU and want runtime acceleration, use the backend selector in
your run TOML:

```toml
[architecture]
use_gpu = true
backend = "auto"   # or "cuda" / "metal"

[numerics]
float_type = "Float32"
```

The driver script (`scripts/run_transport.jl`) preloads CUDA or Metal before
AtmosTransport when needed.

To verify the GPU is visible to Julia:

```bash
julia --project=. -e 'using CUDA; CUDA.versioninfo()'
julia --project=. -e 'using Metal; Metal.versioninfo()'
```

CUDA can run either `Float32` or `Float64` depending on the hardware and
validation goal:

```toml
# in your run config TOML
[architecture]
use_gpu    = true
backend    = "cuda"

[numerics]
float_type = "Float32"   # or "Float64" on CUDA / CPU
```

Note that the L40S and most consumer / data-center GPUs lack first-class
F64 throughput; F32 is the recommended default for production runs.

## Optional: multi-threaded I/O

For double-buffered I/O overlap (disk reads in parallel with GPU compute),
start Julia with multiple threads:

```bash
julia --threads=2 --project=. scripts/run_transport.jl <config.toml>
```

The runtime detects available threads automatically.

## What's next

- [First run](@ref) — invoke the runtime on an existing config.
- [Inspecting output](@ref) — verify a transport binary or snapshot NetCDF.
