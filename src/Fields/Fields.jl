"""
    Fields

Field types for storing tracer concentrations and meteorological data on grids.

Fields carry their grid, location (Center/Face per dimension), data (with halos),
and boundary conditions. Architecture-aware: data lives on CPU or GPU arrays
depending on the grid's architecture.

# Key types

- `Field{LX, LY, LZ}` — a 3D field at a specified staggered location
- `Center`, `Face` — location tags for staggered grids

# Key functions

- `set!(field, value)` — initialize a field
- `interior(field)` — view of interior data (no halos)
- `data(field)` — full data including halos
"""
module Fields

using ..Architectures: AbstractArchitecture, array_type
import ..Architectures: architecture
using ..Grids: AbstractGrid, AbstractLocationType, Center, Face
using ..Grids: grid_size, halo_size, total_size, xnode, ynode, znode

export AbstractLocationType, Center, Face
export AbstractField, Field
export set!, interior, data

include("location_types.jl")
include("abstract_field.jl")
include("field.jl")
include("tracer_collection.jl")
include("halo_operations.jl")
include("boundary_conditions.jl")

end # module Fields
