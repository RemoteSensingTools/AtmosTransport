# ---------------------------------------------------------------------------
# 4DVar cost function
#
# J(x₀) = ½(x₀ - xᵇ)ᵀ B⁻¹ (x₀ - xᵇ) + ½ Σᵢ (yᵢ - H[M(x₀)])ᵀ R⁻¹ (yᵢ - H[M(x₀)])
#
# where:
#   x₀  = control variables (e.g. surface fluxes)
#   xᵇ  = background (prior) estimate
#   B    = background error covariance
#   yᵢ   = observations at time i
#   H    = observation operator
#   M    = forward model
#   R    = observation error covariance
# ---------------------------------------------------------------------------

using LinearAlgebra: dot

"""
$(TYPEDEF)

Supertype for cost functions used in variational data assimilation.
"""
abstract type AbstractCostFunction end

"""
$(TYPEDEF)

Supertype for observation operators that map model state to observation space.

# Interface contract

    observe(op, model_state, time, location) → simulated observation
    adjoint_observe!(adj_state, op, innovation, time, location) → accumulate adjoint forcing
"""
abstract type AbstractObservationOperator end

"""
$(TYPEDEF)

Standard 4DVar cost function.

$(FIELDS)
"""
struct CostFunction4DVar{FT, B, R, H} <: AbstractCostFunction
    "prior estimate of control variables"
    x_background :: Vector{FT}
    "inverse background error covariance (operator or matrix)"
    B_inv        :: B
    "inverse observation error covariance"
    R_inv        :: R
    "observation operator (`AbstractObservationOperator`)"
    obs_operator :: H
    "vector of (time, location, value) tuples"
    observations :: Vector{@NamedTuple{time::Float64, location::Any, value::FT}}
end

"""
$(SIGNATURES)

Evaluate the cost function. Returns scalar cost `J`.
Implementation stub — requires forward model integration.
"""
function evaluate(J::CostFunction4DVar, model)
    error("CostFunction4DVar.evaluate not yet implemented")
end
