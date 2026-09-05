using AtmosTransport
using AtmosTransport.Operators.Chemistry: AbstractChemistryOperator
import AtmosTransport.Operators: apply!

"""
    UniformLoss(rate)

First-order loss, dq/dt = -rate*q, applied to every tracer. `rate` is in s⁻¹
and `dt` in seconds. Tracers store dry VMR × dry-air mass, so multiplication
by exp(-rate*dt) is valid without a molecular-weight conversion. Air mass is
unchanged. The same elementwise operation handles LL/RG arrays and CS panels;
vertical index 1 is the top of the atmosphere.
"""
struct UniformLoss{T} <: AbstractChemistryOperator
    rate::T
    function UniformLoss(rate::T) where {T<:AbstractFloat}
        isfinite(rate) && rate >= 0 || throw(ArgumentError("rate must be finite and non-negative"))
        new{T}(rate)
    end
end

function apply!(state::Union{CellState,CubedSphereState}, meteo, grid,
                op::UniformLoss, dt; workspace=nothing)
    factor = exp(-op.rate * dt)
    arrays = state isa CubedSphereState ? state.tracers_raw : (state.tracers_raw,)
    for values in arrays
        values .*= factor
    end
    return state
end

# A one-half-life check with no external meteorological data.
state = CellState(DryBasis, ones(2,2,2); co2=fill(400e-6,2,2,2))
apply!(state, nothing, nothing, UniformLoss(log(2.0)/3600), 3600.0)
@assert all(isapprox.(get_tracer(state,:co2),200e-6))
