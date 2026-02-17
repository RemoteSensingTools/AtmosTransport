# # Design Principles: A Next-Generation Atmospheric Transport Model
#
# AtmosTransportModel.jl is designed from scratch to address limitations of
# legacy atmospheric transport models (TM5, GEOS-Chem, LMDZ, TM3, etc.)
# while preserving their proven scientific capabilities — particularly the
# hand-coded discrete adjoint for 4D-Var data assimilation.
#
# This document explains the design philosophy and why this model can serve
# as a community resource for atmospheric inverse modeling.

# ## Why a new transport model?
#
# Existing atmospheric transport models have served the community for decades
# but carry significant technical debt:
#
# | Limitation | TM5 | GEOS-Chem | Impact |
# |:---|:---|:---|:---|
# | Language | Fortran 90 | Fortran 90 | Hard to attract new developers; limited metaprogramming |
# | GPU support | None | Partial (via external tools) | Cannot exploit modern HPC hardware |
# | Precision | Float64 only | Float64 only | 2× slower on consumer GPUs (L40S, etc.) |
# | Met drivers | ERA-Interim/ERA5 only | GEOS-FP/MERRA-2 only | Hard to compare across reanalyses |
# | Grid flexibility | Lat-lon only | Lat-lon only | Pole singularity limits resolution |
# | Adjoint | Hand-coded (robust) | AD-based (fragile, slow) | Maintenance vs. performance trade-off |
# | Configuration | Fortran namelists | Input files | Hard to version-control and share |
# | Extensibility | Requires Fortran expertise | Monolithic codebase | High barrier to contribution |
#
# AtmosTransportModel.jl addresses all of these while building on the
# scientific strengths of TM5's operator-splitting framework and proven
# discrete adjoint.

# ## Core Design Principles

# ### 1. Julian Multiple Dispatch — Not Object-Oriented, Not Procedural
#
# Following the design philosophy of Oceananigans.jl (ocean modeling) and
# CliMA (climate modeling), we use Julia's multiple dispatch system to
# separate **what** from **how**:
#
# ```julia
# # The physics operator is defined by its TYPE, not a flag or string
# advect!(tracers, scheme::SlopesAdvection, grid::LatitudeLongitudeGrid, met, dt)
# advect!(tracers, scheme::SlopesAdvection, grid::CubedSphereGrid, met, dt)
# advect!(tracers, scheme::UpwindAdvection, grid::AbstractGrid, met, dt)
# ```
#
# This means new advection schemes, new grid types, or new physics operators
# are added by defining new types and methods — **no modification of existing
# code required**. Compare with Fortran where adding a new scheme typically
# requires touching `if/else` blocks throughout the codebase.

# ### 2. Configuration-Driven, Not Code-Driven
#
# All meteorological data source definitions live in TOML configuration files,
# not in Julia source code:
#
# ```
# config/
#   canonical_variables.toml    ← the "contract" physics code depends on
#   met_sources/
#     geosfp.toml               ← GEOS-FP: URLs, collections, variable names
#     merra2.toml               ← MERRA-2: same structure, different mappings
#     era5.toml                 ← ERA5: different grid, different API
# ```
#
# **Adding a new met data source requires zero Julia code changes** — just
# write a new TOML file. This is critical for enabling community contributions
# and systematic met driver intercomparison studies.

# ### 3. GPU/CPU Agnostic via KernelAbstractions.jl
#
# Every computational kernel is written once using KernelAbstractions.jl and
# runs on both CPU (for development/debugging) and GPU (for production).
# The architecture is selected at model construction time:
#
# ```julia
# model = TransportModel(grid, architecture=GPU())   # NVIDIA GPU
# model = TransportModel(grid, architecture=CPU())   # standard CPU
# ```
#
# No code duplication, no separate GPU branch.

# ### 4. Seamless Float32 / Float64 Precision
#
# The entire model is parametric on float type `FT`. On an NVIDIA A100
# (strong Float64), you can use `Float64` with minimal penalty. On consumer
# GPUs like the L40S where Float64 is 32× slower than Float32, simply:
#
# ```julia
# grid = LatitudeLongitudeGrid(Float32; Nx=360, Ny=180, Nz=72, ...)
# ```
#
# All physics operators, fields, and constants automatically use `Float32`.
# This can yield 2-10× speedup on appropriate hardware with negligible
# impact on transport accuracy.

# ### 5. Hand-Coded Discrete Adjoint — Not Automatic Differentiation
#
# For 4D-Var data assimilation, we need the adjoint (transpose) of the
# linearised transport operator. We use a **hand-coded discrete adjoint**
# following TM5's proven approach, rather than Automatic Differentiation:
#
# | Aspect | Hand-coded adjoint | AD (e.g., Enzyme.jl) |
# |:---|:---|:---|
# | Performance | Optimal — same cost as forward | Often 2-5× slower |
# | GPU support | Native (same kernels) | Fragile with GPU code |
# | Debugging | Transparent, testable | Black-box derivatives |
# | Maintenance | Must update with forward | Automatic but brittle |
#
# Every forward operator has a paired adjoint method:
# ```julia
# advect!(tracers, scheme, grid, met, dt)           # forward
# adjoint_advect!(adj_tracers, scheme, grid, met, dt) # adjoint
# ```
#
# Correctness is verified via the dot-product test (adjoint identity):
# ⟨Ax, y⟩ = ⟨x, Aᵀy⟩

# ### 6. Grid Agnostic — Lat-Lon and Cubed-Sphere
#
# The model supports multiple grid types through abstract grid interfaces:
#
# - **LatitudeLongitudeGrid**: Traditional, simple, compatible with most met data
# - **CubedSphereGrid**: No pole singularity, quasi-uniform resolution, ideal
#   for high-resolution global simulations
#
# Physics operators dispatch on grid type, so the same model setup works
# on either grid. When using GEOS-FP or MERRA-2 on a cubed-sphere grid,
# native cubed-sphere met output can be used directly without regridding.

# ### 7. Extensibility as a First-Class Design Goal
#
# The architecture uses Julia's abstract type hierarchy and package extensions
# to ensure that new capabilities can be added without modifying existing code:
#
# - **New advection scheme**: Define `struct MyScheme <: AbstractAdvectionScheme`
#   and implement `advect!` and `adjoint_advect!`
# - **New met data source**: Write a TOML config file
# - **New grid type**: Subtype `AbstractGrid` and implement accessor functions
# - **New physics operator**: Subtype `AbstractPhysicsOperator`
# - **New output format**: Subtype `AbstractOutputWriter`
#
# Package extensions (Julia 1.9+) keep optional dependencies like CUDA.jl
# out of the core package.

# ## Why This Can Be a Community Resource
#
# Legacy transport models have served the atmospheric science community
# extraordinarily well, but they face growing challenges:
#
# ### The GPU Transition
# HPC is moving decisively toward GPU-accelerated computing. Fortran-based
# models require massive rewrites (or wrapper approaches) to use GPUs.
# AtmosTransportModel.jl is GPU-native from day one.
#
# ### The Met Driver Lock-In Problem
# Most transport models are tightly coupled to a single meteorological
# data source (GEOS-Chem → GEOS-FP; TM5 → ECMWF ERA). This makes it
# scientifically difficult to disentangle transport model errors from met
# driver errors. Our TOML-driven approach lets researchers run **identical
# model physics** with ERA5, MERRA-2, or GEOS-FP and directly attribute
# differences to the meteorological driving data.
#
# ### The Adjoint Bottleneck
# Many groups need transport adjoints for flux inversions but cannot maintain
# the complex Fortran adjoint codes. By providing a clean, tested, well-
# documented discrete adjoint in Julia — with GPU support — we lower the
# barrier to entry for atmospheric inverse modeling.
#
# ### Reproducibility and Transparency
# Julia's package manager ensures reproducible environments. Literate.jl
# documentation means the docs are tested code. TOML configuration files
# are human-readable, version-controllable, and diff-able.
#
# ### Lower Barrier to Contribution
# Julia is easier to learn than Fortran for new graduate students.
# Multiple dispatch means contributors can add new physics without
# understanding the entire codebase. The abstract type system provides
# clear interfaces and contracts.

# ## Comparison with Existing Models
#
# | Feature | AtmosTransportModel.jl | TM5 | GEOS-Chem |
# |:---|:---|:---|:---|
# | Language | Julia | Fortran 90 | Fortran 90 |
# | GPU acceleration | Native (KernelAbstractions.jl) | No | Partial |
# | Float32 support | Seamless | No | No |
# | Met drivers | ERA5 + MERRA-2 + GEOS-FP | ERA only | GEOS only |
# | Met config | TOML (swappable) | Hard-coded | Hard-coded |
# | Adjoint | Hand-coded discrete | Hand-coded discrete | AD (Tapenade) |
# | Adjoint on GPU | Yes | N/A | No |
# | Grid types | Lat-lon + cubed-sphere | Lat-lon | Lat-lon |
# | Package manager | Julia Pkg (reproducible) | Make + manual deps | CMake |
# | Documentation | Literate.jl (tested) | PDF/wiki | ReadTheDocs |
# | Extensibility | Multiple dispatch + TOML | Fortran modules | Fortran modules |

# ## Getting Started
#
# See the [Quick Start guide](../README.md) for installation and basic usage.
# The [Met Driver Comparison](met_driver_comparison.jl) tutorial shows how
# to run the same simulation with different meteorological data sources.
