# ---------------------------------------------------------------------------
# LatLonMesh — regular latitude-longitude horizontal mesh
#
# Implements the AbstractStructuredMesh interface with efficient (i,j) indexing.
# Geometry is derived from uniform Δλ, Δφ spacing and planet radius.
# ---------------------------------------------------------------------------

"""
    LatLonMesh{FT} <: AbstractStructuredMesh

Regular latitude-longitude mesh on the sphere.

# Fields
- `Nx`, `Ny` — number of cells in longitude and latitude
- `Δλ`, `Δφ` — uniform spacing [degrees]
- `λᶜ`, `λᶠ` — cell-center and cell-face longitudes [degrees]
- `φᶜ`, `φᶠ` — cell-center and cell-face latitudes [degrees]
- `radius`   — planet radius [m]
"""
struct LatLonMesh{FT <: AbstractFloat} <: AbstractStructuredMesh
    Nx :: Int
    Ny :: Int
    Δλ :: FT
    Δφ :: FT
    λᶜ :: Vector{FT}
    λᶠ :: Vector{FT}
    φᶜ :: Vector{FT}
    φᶠ :: Vector{FT}
    radius :: FT
end

function LatLonMesh(;
        FT::Type{<:AbstractFloat} = Float64,
        Nx::Int, Ny::Int,
        longitude = (-180, 180),
        latitude  = (-90, 90),
        radius    = FT(6.371e6))

    λ_west, λ_east  = FT.(longitude)
    φ_south, φ_north = FT.(latitude)

    Δλ = (λ_east - λ_west) / Nx
    Δφ = (φ_north - φ_south) / Ny

    λᶠ = collect(range(λ_west,  λ_east,  length=Nx+1) .|> FT)
    λᶜ = FT[(λᶠ[i] + λᶠ[i+1]) / 2 for i in 1:Nx]
    φᶠ = collect(range(φ_south, φ_north, length=Ny+1) .|> FT)
    φᶜ = FT[(φᶠ[j] + φᶠ[j+1]) / 2 for j in 1:Ny]

    return LatLonMesh{FT}(Nx, Ny, Δλ, Δφ, λᶜ, λᶠ, φᶜ, φᶠ, FT(radius))
end

# ---- AbstractStructuredMesh interface ----

nx(m::LatLonMesh) = m.Nx
ny(m::LatLonMesh) = m.Ny
ncells(m::LatLonMesh) = m.Nx * m.Ny
nfaces(m::LatLonMesh) = (m.Nx + 1) * m.Ny + m.Nx * (m.Ny + 1)

# ---- Geometry: cell metrics ----

"""Cell area [m²] at structured index (i, j)."""
function cell_area(m::LatLonMesh{FT}, i::Integer, j::Integer) where FT
    R = m.radius
    return R^2 * deg2rad(m.Δλ) * abs(sind(m.φᶠ[j+1]) - sind(m.φᶠ[j]))
end
cell_area(m::LatLonMesh, c::Tuple{<:Integer, <:Integer}) = cell_area(m, c[1], c[2])

"""Cell area vector (precomputed, one per latitude band)."""
function cell_areas_by_latitude(m::LatLonMesh{FT}) where FT
    R = m.radius
    return FT[R^2 * deg2rad(m.Δλ) * abs(sind(m.φᶠ[j+1]) - sind(m.φᶠ[j])) for j in 1:m.Ny]
end

# ---- Internal structured-only metric helpers (not exported) ----

"""x-spacing [m] at latitude j. Clamped to Δy near poles."""
function dx(m::LatLonMesh{FT}, j::Integer) where FT
    R = m.radius
    dx_val = R * cosd(m.φᶜ[j]) * deg2rad(m.Δλ)
    dy_val = R * deg2rad(m.Δφ)
    return max(dx_val, dy_val)
end

"""y-spacing [m] — uniform."""
function dy(m::LatLonMesh{FT}) where FT
    return m.radius * deg2rad(m.Δφ)
end

"""x-face length [m] (meridional edges) at latitude band j."""
_face_length_x(m::LatLonMesh{FT}, j::Integer) where FT = m.radius * deg2rad(m.Δφ)

"""y-face length [m] (zonal edges) at face latitude j (1:Ny+1)."""
function _face_length_y(m::LatLonMesh{FT}, j::Integer) where FT
    return m.radius * cosd(m.φᶠ[j]) * deg2rad(m.Δλ)
end

# ---- Universal face_length(mesh, f) implementation ----

"""
    face_length(m::LatLonMesh, f) -> FT

Length [m] of face `f` using the universal face-indexed API.

X-faces (f ∈ 1:(Nx+1)*Ny) return the meridional edge length.
Y-faces (f ∈ (Nx+1)*Ny+1 : nfaces) return the zonal edge length.
"""
function face_length(m::LatLonMesh{FT}, f::Integer) where FT
    n_xfaces = (m.Nx + 1) * m.Ny
    if f <= n_xfaces
        j = div(f - 1, m.Nx + 1) + 1
        return _face_length_x(m, j)
    else
        fi = f - n_xfaces
        j = div(fi - 1, m.Nx) + 1
        return _face_length_y(m, j)
    end
end

"""
    face_normal(m::LatLonMesh, f) -> (nx, ny)

Unit normal of face `f`. X-faces → (1, 0), Y-faces → (0, 1).
"""
function face_normal(m::LatLonMesh{FT}, f::Integer) where FT
    n_xfaces = (m.Nx + 1) * m.Ny
    return ifelse(f <= n_xfaces, (one(FT), zero(FT)), (zero(FT), one(FT)))
end

# ---- Face connectivity (structured shortcut) ----

"""
For structured LatLon, faces are implicitly indexed:
- X-faces: index f in 1:(Nx+1)*Ny  → face between (i-1,j) and (i,j), periodic in x
- Y-faces: index f in 1:Nx*(Ny+1)  → face between (i,j-1) and (i,j), bounded in y
Direct (i,j,k) array access is preferred over these for performance.
"""
function face_cells(m::LatLonMesh, f::Integer)
    Nx = m.Nx
    n_xfaces = (Nx + 1) * m.Ny
    if f <= n_xfaces
        j = div(f - 1, Nx + 1) + 1
        i = mod(f - 1, Nx + 1) + 1
        left  = i == 1      ? Nx : i - 1
        right = i == Nx + 1 ? 1  : i
        return (left + (j-1)*Nx, right + (j-1)*Nx)
    else
        fi = f - n_xfaces
        j = div(fi - 1, Nx) + 1
        i = mod(fi - 1, Nx) + 1
        below = j == 1        ? 0 : i + (j-2)*Nx
        above = j == m.Ny + 1 ? 0 : i + (j-1)*Nx
        return (below, above)
    end
end

"""
    cell_faces(m::LatLonMesh, c) -> (west, east, south, north)

Indices of the 4 faces bounding cell `c`. Accepts either a flat cell index
(column-major: `c = i + (j-1)*Nx`) or a tuple `(i, j)`.

Returns face indices consistent with the face_length/face_normal/face_cells
numbering: X-faces 1:(Nx+1)*Ny, then Y-faces (Nx+1)*Ny+1 : nfaces.
"""
function cell_faces(m::LatLonMesh, c::Integer)
    Nx = m.Nx
    i = mod(c - 1, Nx) + 1
    j = div(c - 1, Nx) + 1
    return _cell_faces_ij(m, i, j)
end

function cell_faces(m::LatLonMesh, c::Tuple{<:Integer, <:Integer})
    return _cell_faces_ij(m, c[1], c[2])
end

function _cell_faces_ij(m::LatLonMesh, i::Integer, j::Integer)
    Nx = m.Nx
    n_xfaces = (Nx + 1) * m.Ny
    west  = (j - 1) * (Nx + 1) + i
    east  = (j - 1) * (Nx + 1) + i + 1
    south = n_xfaces + (j - 1) * Nx + i
    north = n_xfaces + j * Nx + i
    return (west, east, south, north)
end

export LatLonMesh, cell_areas_by_latitude
