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
    AbstractLocationType

Supertype for staggered-grid location tags.
"""
abstract type AbstractLocationType end

"""Cell center location."""
struct Center <: AbstractLocationType end

"""Cell face location."""
struct Face <: AbstractLocationType end
