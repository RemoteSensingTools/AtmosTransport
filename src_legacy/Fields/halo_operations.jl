# ---------------------------------------------------------------------------
# Halo operations
#
# Halo filling dispatches on BOTH grid type and comms type, so that:
#   - LatitudeLongitudeGrid + SingletonComms → local periodic/extrapolation fill
#   - CubedSphereGrid + SingletonComms → local panel-edge exchange
#   - Any grid + MPIComms (future) → MPI halo exchange
#
# These are stubs; implementations filled by parallel agents.
# ---------------------------------------------------------------------------

using ..Communications: AbstractComms, SingletonComms
using ..Grids: LatitudeLongitudeGrid, CubedSphereGrid, fill_panel_halos!

import ..Communications: fill_halo!

"""
$(SIGNATURES)

Fill halo regions of `field` using the appropriate method for the grid and comms.
"""
function fill_halo!(field::Field, comms::AbstractComms)
    return fill_halo!(field, grid(field), comms)
end

function fill_halo!(field::Field, grid::LatitudeLongitudeGrid, ::SingletonComms)
    arr = data(field)
    gs = grid_size(grid)
    hs = halo_size(grid)
    Nx, Ny, Nz = gs.Nx, gs.Ny, gs.Nz
    Hx, Hy, Hz = hs.Hx, hs.Hy, hs.Hz

    @inbounds for k in 1:Nz
        for j in 1:Ny
            for i in 1:Hx
                arr[1-i, j, k] = arr[Nx+1-i, j, k]
                arr[Nx+i, j, k] = arr[i, j, k]
            end
        end
    end

    @inbounds for k in 1:Nz
        for j in 1:Hy
            for i in (1-Hx):(Nx+Hx)
                arr[i, 1-j, k] = arr[i, 1, k]
                arr[i, Ny+j, k] = arr[i, Ny, k]
            end
        end
    end

    return nothing
end

function fill_halo!(field::Field, grid::CubedSphereGrid, ::SingletonComms)
    arr = data(field)
    panels = ntuple(6) do p
        view(arr, p, :, :, :)
    end
    fill_panel_halos!(panels, grid)
    return nothing
end
