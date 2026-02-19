# ---------------------------------------------------------------------------
# Tracer collection
#
# Tracers are stored as a NamedTuple of Fields, following Oceananigans:
#   tracers = (CO2 = Field(...), CH4 = Field(...))
#
# This allows arbitrary numbers and names of tracers without code changes.
# ---------------------------------------------------------------------------

"""
$(SIGNATURES)

Create a `NamedTuple` of `Field`s at `(Center, Center, Center)` for each tracer name.

# Example

    tracers = TracerFields((:CO2, :CH4), grid)
    tracers.CO2  # a Field{Center, Center, Center}
"""
function TracerFields(names::NTuple{N, Symbol}, grid) where N
    fields = ntuple(i -> Field(Center(), Center(), Center(), grid), N)
    return NamedTuple{names}(fields)
end

export TracerFields
