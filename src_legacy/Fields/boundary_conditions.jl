# ---------------------------------------------------------------------------
# Boundary conditions
#
# Boundary conditions are attached to Fields and used during halo filling.
# The abstract type allows different BCs per field (e.g. zero-flux, periodic,
# prescribed value).
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Supertype for field boundary conditions.
"""
abstract type AbstractBoundaryCondition end

"""
$(TYPEDEF)

No boundary condition (halo fill handles everything via grid topology).
"""
struct DefaultBC <: AbstractBoundaryCondition end

"""
$(TYPEDEF)

Zero-flux (Neumann) boundary condition.
"""
struct ZeroFluxBC <: AbstractBoundaryCondition end

"""
$(TYPEDEF)

Prescribed value (Dirichlet) boundary condition.

$(FIELDS)
"""
struct ValueBC{T} <: AbstractBoundaryCondition
    "the prescribed boundary value"
    value :: T
end

export AbstractBoundaryCondition, DefaultBC, ZeroFluxBC, ValueBC
