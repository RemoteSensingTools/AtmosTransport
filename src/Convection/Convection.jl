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

export AbstractConvection, TiedtkeConvection, NoConvection
export convect!, adjoint_convect!

"""
$(TYPEDEF)

Supertype for convective transport parameterizations.
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

include("tiedtke_convection.jl")
include("tiedtke_convection_adjoint.jl")

end # module Convection
