# ---------------------------------------------------------------------------
# Staggered grid location types
#
# Defined in Grids (not Fields) because grid accessor functions dispatch on them.
# Fields re-exports these for convenience.
#
# Tracers/pressure: (Center, Center, Center)
# u-velocity:       (Face,   Center, Center)
# v-velocity:       (Center, Face,   Center)
# w-velocity:       (Center, Center, Face)
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Supertype for staggered-grid location tags.
"""
abstract type AbstractLocationType end

"""
$(TYPEDEF)

Cell center location.
"""
struct Center <: AbstractLocationType end

"""
$(TYPEDEF)

Cell face location.
"""
struct Face <: AbstractLocationType end
