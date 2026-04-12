"""
    Callbacks

Callback and forcing system for user-defined extensions without modifying core code.

Callbacks are checked and executed at defined points in the time-stepping loop.
Forcing functions provide additional source terms.

# Types

- `DiscreteCallback` — fired when `condition(model, t)` returns true
- `Forcing` — a function applied as a source term each time step
"""
module Callbacks

using DocStringExtensions

export AbstractCallback, DiscreteCallback, Forcing
export execute_callbacks!

"""
$(TYPEDEF)

Supertype for all callbacks.
"""
abstract type AbstractCallback end

"""
$(TYPEDEF)

Callback that fires when `condition(model, t)` returns `true`.

$(FIELDS)
"""
struct DiscreteCallback{C, A} <: AbstractCallback
    "`(model, t) → Bool`"
    condition :: C
    "`(model) → nothing` (mutates model state)"
    affect!   :: A
end

"""
$(TYPEDEF)

User-defined source/forcing term applied every time step.

$(FIELDS)
"""
struct Forcing{F}
    "`(x, y, z, t, params...) → value` to be added to tracer tendency"
    func :: F
end

"""
$(SIGNATURES)

Check and execute all callbacks whose conditions are met.
"""
function execute_callbacks!(callbacks, model, t)
    for cb in callbacks
        if cb isa DiscreteCallback && cb.condition(model, t)
            cb.affect!(model)
        end
    end
    return nothing
end

end # module Callbacks
