# ---------------------------------------------------------------------------
# Abstract operator hierarchy for the basis-explicit transport architecture.
#
# All physics operators dispatch on these abstract types. The universal
# interface is:
#
#   apply!(state, fluxes::AbstractFaceFluxState, grid, op, dt; kwargs...)
#
# Transport operators receive only CellState + AbstractFaceFluxState +
# AtmosGrid. They never see raw winds, humidity, or met-specific structs.
#
# The operator contract is face-oriented at the mathematical level.
# Concrete flux storage and kernel strategy are selected by dispatch on
# the mesh's flux topology:
#
#   structured mesh  → AbstractStructuredFaceFluxState  → cell-loop kernels
#   unstructured mesh → AbstractUnstructuredFaceFluxState → face-loop kernels
#
# Diffusion and convection roots are declared here because those modules
# extend them directly. Advection, chemistry, and surface-flux modules declare
# their own family roots as subtypes of AbstractOperator. Each family has one
# root and one apply! protocol.
# ---------------------------------------------------------------------------

"""
    AbstractOperator

Root type for all physics operators in the transport model.
"""
abstract type AbstractOperator end

"""
    AbstractDiffusion <: AbstractOperator

Root type for vertical diffusion operators. Concrete subtypes live in
`src/Operators/Diffusion/operators.jl` (`NoDiffusion`,
`ImplicitVerticalDiffusion`, …).
"""
abstract type AbstractDiffusion <: AbstractOperator end

"""
    AbstractConvection <: AbstractOperator

Root type for convective-transport operators. Concrete subtypes live in
`src/Operators/Convection/` (`NoConvection`, `CMFMCConvection`,
`TM5Convection`).
"""
abstract type AbstractConvection <: AbstractOperator end

# ---------------------------------------------------------------------------
# Default error stubs
# ---------------------------------------------------------------------------

function apply! end

export AbstractOperator
export AbstractDiffusion, AbstractConvection
export apply!
