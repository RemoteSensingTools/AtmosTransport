# ---------------------------------------------------------------------------
# Abstract field type
#
# Interface contract for all fields:
#   data(field)               → underlying array (with halos)
#   interior(field)           → view of interior data (no halos)
#   set!(field, value)        → fill field with a scalar, function, or array
#   grid(field)               → the grid this field lives on
#   location(field)           → (LX, LY, LZ) location tuple
#   architecture(field)       → CPU or GPU
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Supertype for all field types. Parametric on:
- `LX, LY, LZ`: location types (Center or Face) per dimension
- `G`: grid type
"""
abstract type AbstractField{LX, LY, LZ, G} end

"""Return the underlying data array (including halos)."""
function data end

"""Return a view of interior data (excluding halos)."""
function interior end

"""Return the grid associated with the field."""
function grid end

"""Return the location tuple `(LX(), LY(), LZ())`."""
location(::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = (LX(), LY(), LZ())
