# ---------------------------------------------------------------------------
# Grid utility functions
# ---------------------------------------------------------------------------

"""
    total_size(grid)

Return the total array size including halo regions for memory allocation.

For LatitudeLongitudeGrid: `(Nx + 2*Hx, Ny + 2*Hy, Nz + 2*Hz)` — interior
dimensions plus halo on both sides of each dimension.
"""
function total_size end

total_size(g::LatitudeLongitudeGrid) = (g.Nx + 2g.Hx, g.Ny + 2g.Hy, g.Nz + 2g.Hz)

function total_size(g::CubedSphereGrid)
    Nt = g.Nc + 2g.Hp
    return (6, Nt, Nt, g.Nz + 2g.Hz)
end

"""
    interior_indices(grid)

Return the range of interior (non-halo) indices for each dimension.
"""
function interior_indices end

function interior_indices(g::LatitudeLongitudeGrid)
    return (g.Hx+1 : g.Hx+g.Nx,
            g.Hy+1 : g.Hy+g.Ny,
            g.Hz+1 : g.Hz+g.Nz)
end

function interior_indices(g::CubedSphereGrid)
    return (1:6,
            g.Hp+1 : g.Hp+g.Nc,
            g.Hp+1 : g.Hp+g.Nc,
            g.Hz+1 : g.Hz+g.Nz)
end

export total_size, interior_indices
