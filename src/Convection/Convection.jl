"""
    Convection

Convective transport parameterizations with paired discrete adjoints.

# Interface contract

    convect!(tracers, met, grid, conv::AbstractConvection, Δt)
    adjoint_convect!(adj_tracers, met, grid, conv::AbstractConvection, Δt)
"""
module Convection

using DocStringExtensions

using ..Grids: AbstractGrid
using ..Fields: AbstractField

export AbstractConvection, TiedtkeConvection, NoConvection, RASConvection
export convect!, adjoint_convect!, invalidate_ras_cfl_cache!
export AbstractTracerSolubility, InertTracer, SolubleTracer
export tracer_solubility, wet_scavenge_fraction

"""
$(TYPEDEF)

Supertype for convective transport parameterizations.

Available concrete types:
- [`NoConvection`](@ref): no-op pass-through
- [`TiedtkeConvection`](@ref): simplified upwind mass-flux (CMFMC only)
- [`RASConvection`](@ref): Relaxed Arakawa-Schubert with entrainment/detrainment (CMFMC + DTRAIN)

# Adding a new scheme

1. Define `struct MyScheme <: AbstractConvection end`
2. Implement `convect!` methods for `CubedSphereGrid` and `LatitudeLongitudeGrid`
3. Add `_needs_convection(::MyScheme) = true` in `run_implementations.jl`
4. If the scheme needs DTRAIN: add `_needs_dtrain(::MyScheme) = true`
5. Add a `type = "myscheme"` branch in `_build_convection` (configuration.jl)
"""
abstract type AbstractConvection end

"""
$(TYPEDEF)

No convection (pass-through). Adjoint is also a no-op.
"""
struct NoConvection <: AbstractConvection end

convect!(tracers, met, grid, ::NoConvection, Δt) = nothing
adjoint_convect!(adj_tracers, met, grid, ::NoConvection, Δt) = nothing

"""
$(TYPEDEF)

Tiedtke (1989) mass-flux convection scheme, as used in TM5.
Mass fluxes come from met data (fixed), so the operator is linear in tracers
and the adjoint is the transpose of the mass-flux redistribution matrix.

The forward operator uses prescribed convective mass fluxes from met data
(`met.conv_mass_flux`) to redistribute tracers vertically via upwind
mass-flux transport. See `tiedtke_convection.jl` for implementation.
"""
struct TiedtkeConvection <: AbstractConvection end

"""
$(TYPEDEF)

Relaxed Arakawa-Schubert (RAS) convection scheme (Moorthi & Suarez 1992),
as implemented in GEOS-Chem (`convection_mod.F90`).

Uses both CMFMC (updraft mass flux at interfaces) and DTRAIN (detraining
mass flux at layer centers) from GEOS met data. Entrainment is diagnosed
from the mass balance: ENTRN = max(0, CMFMC + DTRAIN - CMFMC_below).

If DTRAIN data is unavailable at runtime, falls back to Tiedtke-style
CMFMC-only transport with a warning.

See `ras_convection.jl` for the full algorithm and references.
"""
struct RASConvection <: AbstractConvection end

include("scavenging.jl")
include("tiedtke_convection.jl")
include("tiedtke_convection_adjoint.jl")
include("ras_convection.jl")

end # module Convection
