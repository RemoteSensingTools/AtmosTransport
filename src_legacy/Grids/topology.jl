# ---------------------------------------------------------------------------
# Topology types
#
# Topology describes the connectivity of each dimension. Physics code and
# halo exchange dispatch on topology to handle wrapping, walls, or
# cubed-sphere panel boundaries correctly.
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Supertype for grid dimension topologies.
"""
abstract type AbstractTopology end

"""
$(TYPEDEF)

Periodic dimension (e.g. longitude on a lat-lon grid).
"""
struct Periodic   <: AbstractTopology end

"""
$(TYPEDEF)

Bounded dimension with walls (e.g. latitude on a lat-lon grid).
"""
struct Bounded    <: AbstractTopology end

"""
$(TYPEDEF)

Cubed-sphere panel connectivity (edges connect to neighboring panels).
"""
struct CubedPanel <: AbstractTopology end

"""
$(TYPEDEF)

Flat (degenerate) dimension — single grid point, used to reduce dimensionality.
"""
struct Flat       <: AbstractTopology end
