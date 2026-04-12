# ---------------------------------------------------------------------------
# Re-export location types from Grids
#
# Center and Face are defined in Grids (since grid accessors dispatch on them).
# We re-export here so users can do `using .Fields: Center, Face`.
# ---------------------------------------------------------------------------

using ..Grids: AbstractLocationType, Center, Face

"""Location type for tracers: (Center, Center, Center)."""
const TracerLocation = Tuple{Center, Center, Center}
