# Contributing to AtmosTransport.jl

Thank you for your interest in contributing! This document provides guidelines
for contributing to AtmosTransport.jl.

## Getting Started

### Prerequisites

- Julia 1.10 or later (install via [juliaup](https://github.com/JuliaLang/juliaup))
- Git
- (Optional) NVIDIA GPU with CUDA 12+ drivers for GPU testing

### Development Setup

```bash
git clone https://github.com/RemoteSensingTools/AtmosTransport.jl.git
cd AtmosTransport
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Running Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

### Building Documentation Locally

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
ATMOSTR_DOCS_BUILD_ONLY=1 julia --project=docs docs/make.jl
# Open docs/build/index.html in your browser
```

## Code Style

- Follow standard Julia conventions: `snake_case` for functions and variables,
  `CamelCase` for types
- Use multiple dispatch rather than if-else chains on type tags
- Keep functions short and focused; prefer composing small functions
- Add docstrings (using `DocStringExtensions`) to all exported functions and types

## Finding the scientific code

Start with `src/Models/TransportModel.jl` for operator ordering, then
`src/Operators/<process>/` for equations and kernels. `DrivenSimulation.jl`
refreshes meteorology; `DrivenRunner.jl` manages the binary-file/window loop.
Its `runner/` files separate configuration, model setup, progress, and output.
`InitialConditionIO.jl` handles initial conditions, while
`initial_conditions/surface_flux.jl` handles emission inventories.

The current interfaces are:

| Process | Interface | Examples |
| --- | --- | --- |
| Advection | `AbstractAdvectionScheme` | `UpwindScheme`, `SlopesScheme`, `PPMScheme`, `LinRoodPPMScheme` |
| Convection | `AbstractConvection` | `TM5Convection`, `CMFMCConvection` |
| Diffusion | `AbstractDiffusion` | `ImplicitVerticalDiffusion` |
| Chemistry | `AbstractChemistryOperator` | `ExponentialDecay`, `CompositeChemistry` |
| Horizontal geometry | `AbstractHorizontalMesh` | `LatLonMesh`, `ReducedGaussianMesh`, `CubedSphereMesh` |

Read [the topology support matrix](src/Operators/TOPOLOGY_SUPPORT.md) before
adding a configuration. Reduced-Gaussian advection currently supports upwind
and no advection; slopes and PPM are rejected at recipe construction.

## Adding a physics operator

[examples/custom_loss.jl](examples/custom_loss.jl) is an executable, minimal
chemistry extension using the current `apply!(state, meteo, grid, op, dt;
workspace)` interface:

```bash
julia --project=. examples/custom_loss.jl
```

For each operator, document its equation, units, array layout, vertical
ordering, and cadence. State whether a quantity is physical species mass or
model tracer storage: dry-basis transport stores **dry VMR × dry-air mass**;
converting an emission inventory in kg species/s requires molecular weights.
Document numerical approximations and unsupported topology/backend combinations.

Keep scratch buffers in a typed workspace on the state backend. Use
KernelAbstractions kernels or backend-supported array operations, and avoid
scalar indexing of GPU arrays. Add conservation, uniform-tracer, or analytic
solution checks in `test/core/`; include adjoint identity or finite-difference
checks when modifying an operator used by the inverse model. Test files use
`using AtmosTransport` and run in isolated modules, sharing the package cache.

For performance changes, report warm timings, allocations, backend, precision,
grid size, tracer count, and output selection. The benchmark harness lives in
`benchmarking/`; distinguish kernel timing from binary input, host transfer,
and actual snapshot writing. Run CUDA checks on the A100 on `curry`, selecting
it explicitly with `CUDA_VISIBLE_DEVICES=0` and verifying the device name.

## Submitting Changes

1. Fork the repository and create a feature branch
2. Make your changes with clear, focused commits
3. Ensure all tests pass: `julia --project=. -e 'using Pkg; Pkg.test()'`
4. Open a pull request with a clear description of what changed and why

## Reporting Issues

Please open an issue on GitHub with:
- A clear description of the problem
- Minimal reproducible example (if applicable)
- Julia version and OS information (`versioninfo()`)
