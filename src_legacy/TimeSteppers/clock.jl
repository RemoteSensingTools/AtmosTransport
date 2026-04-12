# ---------------------------------------------------------------------------
# Clock — tracks simulation time and iteration count
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Tracks simulation time and iteration count.

$(FIELDS)
"""
mutable struct Clock{FT}
    "current simulation time [seconds]"
    time      :: FT
    "current iteration number"
    iteration :: Int
    "current time step [seconds]"
    Δt        :: FT
end

Clock(FT::Type = Float64; time = zero(FT), Δt = zero(FT)) =
    Clock{FT}(time, 0, Δt)

"""
$(SIGNATURES)

Advance the clock by `Δt` seconds.
"""
function tick!(clock::Clock, Δt)
    clock.time += Δt
    clock.iteration += 1
    return nothing
end

"""
$(SIGNATURES)

Reverse the clock by `Δt` seconds (for adjoint runs).
"""
function tick_backward!(clock::Clock, Δt)
    clock.time -= Δt
    clock.iteration -= 1
    return nothing
end
