# ---------------------------------------------------------------------------
# Abstract operator hierarchy for the basis-explicit transport architecture
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
# ---------------------------------------------------------------------------

"""
    AbstractOperator

Root type for all physics operators in the transport model.
"""
abstract type AbstractOperator end

"""
    AbstractDiffusion <: AbstractOperator

Vertical diffusion operators (implicit Thomas solver, PBL schemes).
Phase 2+ — stub.
"""
abstract type AbstractDiffusion <: AbstractOperator end

"""
    AbstractConvection <: AbstractOperator

Convective transport operators (Tiedtke, RAS, host-driven).
Phase 2+ — stub.
"""
abstract type AbstractConvection <: AbstractOperator end

"""
    AbstractSourceSink <: AbstractOperator

Source/sink operators (emissions, deposition, chemistry).
Phase 2+ — stub.
"""
abstract type AbstractSourceSink <: AbstractOperator end

# ---------------------------------------------------------------------------
# Default error stubs
# ---------------------------------------------------------------------------

function apply! end

export AbstractOperator
export AbstractDiffusion
export AbstractConvection, AbstractSourceSink
export apply!
