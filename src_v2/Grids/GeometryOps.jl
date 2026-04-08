# ---------------------------------------------------------------------------
# Geometry operations — the universal face/cell API
#
# Every horizontal mesh must implement these functions. Structured meshes
# provide efficient (i,j)-indexed specializations; unstructured meshes use
# explicit connectivity tables.
# ---------------------------------------------------------------------------

# ---- Cell queries ----

"""
    ncells(mesh::AbstractHorizontalMesh) -> Int

Total number of horizontal cells in the mesh.
"""
function ncells end

"""
    cell_area(mesh::AbstractHorizontalMesh, c) -> FT

Area [m²] of cell `c`. For structured meshes, `c` can be `(i, j)`.
"""
function cell_area end

"""
    cell_faces(mesh::AbstractHorizontalMesh, c) -> indices

Indices of faces bounding cell `c`. For structured meshes this
returns the 4 (or 6 for hex) face indices; for unstructured meshes
it uses the CSR adjacency.
"""
function cell_faces end

# ---- Face queries ----

"""
    nfaces(mesh::AbstractHorizontalMesh) -> Int

Total number of horizontal faces in the mesh.
"""
function nfaces end

"""
    face_length(mesh::AbstractHorizontalMesh, f) -> FT

Length [m] of face `f` (or area of the face cross-section for 3D).
"""
function face_length end

"""
    face_normal(mesh::AbstractHorizontalMesh, f) -> (nx, ny)

Outward unit normal of face `f` in the local tangent plane.
For structured lat-lon grids, x-faces have normal (1,0), y-faces (0,1).
"""
function face_normal end

"""
    face_cells(mesh::AbstractHorizontalMesh, f) -> (left, right)

The two cells sharing face `f`. Convention: flux from left to right
is positive. Boundary faces use `0` or a sentinel for the exterior cell.
"""
function face_cells end

# ---- Structured mesh convenience ----

"""
    nx(mesh::AbstractStructuredMesh) -> Int
    ny(mesh::AbstractStructuredMesh) -> Int

Logical grid dimensions for structured meshes.
"""
function nx end
function ny end

export ncells, nfaces, cell_area, cell_faces
export face_length, face_normal, face_cells
export nx, ny
