# ---------------------------------------------------------------------------
# Halo operations
#
# Halo filling dispatches on BOTH grid type and comms type, so that:
#   - LatitudeLongitudeGrid + SingletonComms → local periodic/extrapolation fill
#   - CubedSphereGrid + SingletonComms → local panel-edge exchange
#   - Any grid + MPIComms (future) → MPI halo exchange
#
# These are stubs; implementations filled by parallel agents.
# ---------------------------------------------------------------------------

using ..Communications: AbstractComms, SingletonComms
using ..Grids: LatitudeLongitudeGrid, CubedSphereGrid

import ..Communications: fill_halo!

"""
    fill_halo!(field::Field, comms::AbstractComms)

Fill halo regions of `field` using the appropriate method for the grid and comms.
"""
function fill_halo!(field::Field, comms::AbstractComms)
    return fill_halo!(field, grid(field), comms)
end

# Lat-lon + singleton: periodic in lon, extrapolate in lat
function fill_halo!(field::Field, ::LatitudeLongitudeGrid, ::SingletonComms)
    # TODO: implement periodic lon wrap + lat boundary handling
    return nothing
end

# Cubed-sphere + singleton: panel-edge exchange
function fill_halo!(field::Field, ::CubedSphereGrid, ::SingletonComms)
    # TODO: implement panel connectivity halo exchange
    return nothing
end
