# ---------------------------------------------------------------------------
# Concrete Field type
# ---------------------------------------------------------------------------

using OffsetArrays: OffsetArray

"""
$(TYPEDEF)

A 3D field on a grid at a specified staggered location.

Data is stored as an `OffsetArray` to include halo regions with negative/zero
indices, so that stencil operations can index `field[i-1, j, k]` without
bounds checking in the interior.

$(FIELDS)
"""
struct Field{LX, LY, LZ, G, D, B} <: AbstractField{LX, LY, LZ, G}
    "the grid this field lives on"
    grid :: G
    "underlying data array (`OffsetArray` including halos)"
    _data :: D
    "boundary conditions attached to this field"
    boundary_conditions :: B
end

function Field(::LX, ::LY, ::LZ,
               grid::AbstractGrid{FT, Arch};
               boundary_conditions = nothing) where {LX <: AbstractLocationType,
                                                      LY <: AbstractLocationType,
                                                      LZ <: AbstractLocationType,
                                                      FT, Arch}
    arch = architecture(grid)
    AT = array_type(arch)
    sz = total_size(grid)
    raw = AT{FT}(undef, sz...)
    fill!(raw, zero(FT))

    # Wrap in OffsetArray so interior starts at index 1
    gs = grid_size(grid)
    hs = halo_size(grid)
    # For lat-lon: axes (1-Hx):(Nx+Hx) so interior is 1:Nx, 1:Ny, 1:Nz
    if hasproperty(hs, :Hx)  # LatitudeLongitudeGrid
        offset_data = OffsetArray(raw, (1-hs.Hx):(gs.Nx+hs.Hx),
                                       (1-hs.Hy):(gs.Ny+hs.Hy),
                                       (1-hs.Hz):(gs.Nz+hs.Hz))
    else
        offset_data = raw  # CubedSphere — offset scheme TBD
    end

    return Field{LX, LY, LZ, typeof(grid), typeof(offset_data),
                 typeof(boundary_conditions)}(grid, offset_data, boundary_conditions)
end

# ---------------------------------------------------------------------------
# Interface implementations
# ---------------------------------------------------------------------------

data(f::Field)     = f._data
grid(f::Field)     = f.grid

function interior(f::Field)
    g = f.grid
    gs = grid_size(g)
    if hasproperty(gs, :Nx)
        return view(f._data, 1:gs.Nx, 1:gs.Ny, 1:gs.Nz)
    else
        return view(f._data, :, :, :, :)  # CubedSphere placeholder
    end
end

"""
$(SIGNATURES)

Set field interior to `value`. `value` can be:
- A scalar (fills uniformly)
- A function `f(x, y, z)` evaluated at cell centers
- An array matching the interior size
"""
function set!(f::Field, value::Number)
    fill!(interior(f), value)
    return nothing
end

function set!(f::Field, func::Function)
    g = f.grid
    gs = grid_size(g)
    int = interior(f)
    for k in 1:gs.Nz, j in 1:gs.Ny, i in 1:gs.Nx
        x = xnode(i, j, g, Center())
        y = ynode(i, j, g, Center())
        z = znode(k, g, Center())
        int[i, j, k] = func(x, y, z)
    end
    return nothing
end

architecture(f::Field) = architecture(f.grid)
