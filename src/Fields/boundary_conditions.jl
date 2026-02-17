# ---------------------------------------------------------------------------
# Boundary conditions
#
# Boundary conditions are attached to Fields and used during halo filling.
# The abstract type allows different BCs per field (e.g. zero-flux, periodic,
# prescribed value).
# ---------------------------------------------------------------------------

"""
    AbstractBoundaryCondition

Supertype for field boundary conditions.
"""
abstract type AbstractBoundaryCondition end

"""No boundary condition (halo fill handles everything via grid topology)."""
struct DefaultBC <: AbstractBoundaryCondition end

"""Zero-flux (Neumann) boundary condition."""
struct ZeroFluxBC <: AbstractBoundaryCondition end

"""Prescribed value (Dirichlet) boundary condition."""
struct ValueBC{T} <: AbstractBoundaryCondition
    value :: T
end

export AbstractBoundaryCondition, DefaultBC, ZeroFluxBC, ValueBC
