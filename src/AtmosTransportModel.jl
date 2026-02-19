"""
    AtmosTransportModel

A Julia-based atmospheric transport model for GPU and CPU.

Inspired by TM5 (Krol et al., 2005) and designed with Oceananigans.jl-style
multiple dispatch. Supports latitude-longitude and cubed-sphere grids,
multiple met data sources (ERA5, MERRA-2, GEOS-FP), hand-coded discrete
adjoint with Revolve checkpointing, and extensible physics operators.

See the README and `docs/` for full documentation.
"""
module AtmosTransportModel

# ---- Core infrastructure (order matters for dependencies) ----
include("Architectures.jl")
using .Architectures

include("Parameters.jl")
using .Parameters

include("Communications.jl")
using .Communications

# ---- Grids ----
include("Grids/Grids.jl")
using .Grids

# ---- Fields ----
include("Fields/Fields.jl")
using .Fields

# ---- Physics operators ----
include("Advection/Advection.jl")
using .Advection

include("Convection/Convection.jl")
using .Convection

include("Diffusion/Diffusion.jl")
using .Diffusion

include("Chemistry/Chemistry.jl")
using .Chemistry

# ---- Time stepping ----
include("TimeSteppers/TimeSteppers.jl")
using .TimeSteppers

# ---- Adjoint infrastructure ----
include("Adjoint/Adjoint.jl")
using .Adjoint

# ---- Callbacks ----
include("Callbacks/Callbacks.jl")
using .Callbacks

# ---- I/O ----
include("IO/IO.jl")
using .IO

# ---- Regridding ----
include("Regridding/Regridding.jl")
using .Regridding

# ---- Sources / Emissions ----
include("Sources/Sources.jl")
using .Sources

# ---- Models (depends on everything above) ----
include("Models/Models.jl")
using .Models

end # module AtmosTransportModel
