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
git clone https://github.com/RemoteSensingTools/AtmosTransport.git
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
julia --project=docs docs/make.jl
# Open docs/build/index.html in your browser
```

## Code Style

- Follow standard Julia conventions: `snake_case` for functions and variables,
  `CamelCase` for types
- Use multiple dispatch rather than if-else chains on type tags
- Keep functions short and focused; prefer composing small functions
- Add docstrings (using `DocStringExtensions`) to all exported functions and types

## Architecture Overview

AtmosTransport uses an Oceananigans.jl-inspired design with abstract type
hierarchies and multiple dispatch:

```
AbstractAdvection        →  SlopesAdvection, UpwindAdvection, ...
AbstractConvection       →  TiedtkeConvection, NoConvection, ...
AbstractDiffusion        →  BoundaryLayerDiffusion, NoDiffusion, ...
AbstractChemistry        →  NoChemistry, RadioactiveDecay, CompositeChemistry, ...
AbstractGrid             →  LatitudeLongitudeGrid, CubedSphereGrid
```

## Adding a New Physics Operator

To add a new advection scheme (for example):

1. **Define your type** in `src/Advection/`:
   ```julia
   struct MyAdvection <: AbstractAdvection end
   ```

2. **Implement the interface** — dispatch on your type:
   ```julia
   function advect!(tracers, grid, adv::MyAdvection, mass_fluxes, dt)
       # your implementation
   end
   ```

3. **Add the adjoint** (if supporting 4D-Var):
   ```julia
   function adjoint_advect!(adj_tracers, grid, adv::MyAdvection, mass_fluxes, dt)
       # adjoint of your implementation
   end
   ```

4. **Export** your type from the submodule.

5. **Add tests** in `test/` and verify with `Pkg.test()`.

The same pattern applies to convection (`AbstractConvection`), diffusion
(`AbstractDiffusion`), and chemistry (`AbstractChemistry`) operators.

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
