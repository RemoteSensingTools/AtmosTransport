# ---------------------------------------------------------------------------
# Topology types
#
# Topology describes the connectivity of each dimension. Physics code and
# halo exchange dispatch on topology to handle wrapping, walls, or
# cubed-sphere panel boundaries correctly.
# ---------------------------------------------------------------------------

"""
    AbstractTopology

Supertype for grid dimension topologies.
"""
abstract type AbstractTopology end

"""Periodic dimension (e.g. longitude on a lat-lon grid)."""
struct Periodic   <: AbstractTopology end

"""Bounded dimension with walls (e.g. latitude on a lat-lon grid)."""
struct Bounded    <: AbstractTopology end

"""Cubed-sphere panel connectivity (edges connect to neighboring panels)."""
struct CubedPanel <: AbstractTopology end

"""Flat (degenerate) dimension — single grid point, used to reduce dimensionality."""
struct Flat       <: AbstractTopology end
