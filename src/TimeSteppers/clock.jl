# ---------------------------------------------------------------------------
# Clock — tracks simulation time and iteration count
# ---------------------------------------------------------------------------

"""
    Clock{FT}

Tracks simulation time and iteration count.

# Fields
- `time :: FT` — current simulation time [seconds]
- `iteration :: Int` — current iteration number
- `Δt :: FT` — current time step [seconds]
"""
mutable struct Clock{FT}
    time      :: FT
    iteration :: Int
    Δt        :: FT
end

Clock(FT::Type = Float64; time = zero(FT), Δt = zero(FT)) =
    Clock{FT}(time, 0, Δt)

"""Advance the clock by `Δt` seconds."""
function tick!(clock::Clock, Δt)
    clock.time += Δt
    clock.iteration += 1
    return nothing
end

"""Reverse the clock by `Δt` seconds (for adjoint runs)."""
function tick_backward!(clock::Clock, Δt)
    clock.time -= Δt
    clock.iteration -= 1
    return nothing
end
