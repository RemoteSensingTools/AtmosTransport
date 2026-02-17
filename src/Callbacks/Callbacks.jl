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

export AbstractCallback, DiscreteCallback, Forcing
export execute_callbacks!

"""Supertype for all callbacks."""
abstract type AbstractCallback end

"""
    DiscreteCallback{C, A} <: AbstractCallback

Callback that fires when `condition(model, t)` returns `true`.

# Fields
- `condition` — `(model, t) → Bool`
- `affect!`   — `(model) → nothing` (mutates model state)
"""
struct DiscreteCallback{C, A} <: AbstractCallback
    condition :: C
    affect!   :: A
end

"""
    Forcing{F}

User-defined source/forcing term applied every time step.

# Fields
- `func` — `(x, y, z, t, params...) → value` to be added to tracer tendency
"""
struct Forcing{F}
    func :: F
end

"""
    execute_callbacks!(callbacks, model, t)

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
