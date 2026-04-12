# ---------------------------------------------------------------------------
# Temporal interpolation between met data snapshots
#
# Met data is available at discrete times (hourly, 3-hourly).
# The model may need fields at intermediate times.
# Linear interpolation between adjacent snapshots.
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Manages two adjacent met data snapshots and interpolates between them.

$(FIELDS)
"""
mutable struct TemporalInterpolator{M <: AbstractMetData}
    "the met data reader"
    met    :: M
    "time of the earlier snapshot"
    t_prev :: Float64
    "time of the later snapshot"
    t_next :: Float64
end

"""
$(SIGNATURES)

Return weight `α ∈ [0, 1]` for linear interpolation:
`field(t) = (1 - α) * field(t_prev) + α * field(t_next)`
"""
function interpolation_weight(interp::TemporalInterpolator, t::Float64)
    dt = interp.t_next - interp.t_prev
    dt > 0 || error("t_next must be > t_prev")
    α = (t - interp.t_prev) / dt
    return clamp(α, 0.0, 1.0)
end

export TemporalInterpolator, interpolation_weight
