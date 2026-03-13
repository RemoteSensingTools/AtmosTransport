# ---------------------------------------------------------------------------
# Cubed-sphere panel halo exchange
#
# Fills halo regions for 3D tracer fields stored as NTuple{6, Array{FT, 3}}
# with extended dimensions (Nc + 2*Hp) × (Nc + 2*Hp) × Nz per panel.
# Interior data lives at indices [Hp+1:Hp+Nc, Hp+1:Hp+Nc, :].
#
# Edge convention (GEOS standard):
#   1 = North (top,    j = Nc)
#   2 = South (bottom, j = 1)
#   3 = East  (right,  i = Nc)
#   4 = West  (left,   i = 1)
#
# Orientation codes (from connectivity table):
#   0 = aligned     → along-edge same direction
#   1 = 90° CW      → along-edge same direction
#   2 = 180°        → along-edge reversed
#   3 = 90° CCW     → along-edge reversed
#
# The key subtlety: when Panel 2's north connects to Panel 5, it connects
# to Panel 5's EAST edge (not south). The reciprocal edge is found by
# inverse lookup in the connectivity table.
#
# References:
#   Putman & Lin (2007) — FV3 cubed-sphere grid connectivity
#   Martin et al. (2022, GMD) — GCHP panel exchange
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend

"""
$(SIGNATURES)

Allocate a cubed-sphere 3D field as `NTuple{6, Array{FT, 3}}` with extended
dimensions `(Nc + 2*Hp) × (Nc + 2*Hp) × Nz` per panel, initialized to zero.
Interior indices are `[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]`.
"""
function allocate_cubed_sphere_field(grid::CubedSphereGrid{FT},
                                     Nz::Int = grid.Nz) where FT
    Nc = grid.Nc
    Hp = grid.Hp
    n = Nc + 2 * Hp
    AT = array_type(architecture(grid))
    return ntuple(6) do _
        AT(zeros(FT, n, n, Nz))
    end
end

"""
$(SIGNATURES)

Fill halo regions of a cubed-sphere 3D field by copying interior data from
neighboring panels with the correct edge-to-edge mapping and orientation.

`data` is `NTuple{6, Array{FT, 3}}` with each panel `(Nc + 2*Hp) × (Nc + 2*Hp) × Nz`.
"""
function fill_panel_halos!(data::NTuple{6, A},
                           grid::CubedSphereGrid) where {A <: AbstractArray}
    Nc = grid.Nc
    Hp = grid.Hp
    conn = grid.connectivity

    for p in 1:6
        for e in 1:4  # 1=north, 2=south, 3=east, 4=west
            nb = conn.neighbors[p][e]
            q = nb.panel
            orient = nb.orientation
            q_e = _reciprocal_edge_index(conn, p, e)
            _fill_edge!(data[p], data[q], e, q_e, orient, Nc, Hp)
        end
    end
    # Single sync after all 24 edge kernels are queued
    synchronize(get_backend(data[1]))
    return nothing
end

"""Find which edge of the neighbor panel connects back to panel `p`."""
function _reciprocal_edge_index(conn::PanelConnectivity, p::Int, e::Int)
    q = conn.neighbors[p][e].panel
    for eq in 1:4
        conn.neighbors[q][eq].panel == p && return eq
    end
    error("Broken panel connectivity: no reciprocal edge for P$p edge $e → P$q")
end

"""
Copy `Hp` layers of interior data from source panel near edge `q_e`
into the halo of destination panel outside edge `e`, applying the
along-edge orientation mapping.

Dispatches to a GPU kernel for device arrays, or a CPU loop for host arrays.
"""
function _fill_edge!(dst::AbstractArray{T, 3}, src::AbstractArray{T, 3},
                     e::Int, q_e::Int, orient::Int,
                     Nc::Int, Hp::Int) where T
    Nk = size(dst, 3)
    flip = orient >= 2

    backend = get_backend(dst)
    if backend isa KernelAbstractions.CPU
        @inbounds for k in 1:Nk
            for d in 1:Hp
                for s in 1:Nc
                    s_src = flip ? (Nc + 1 - s) : s
                    i_src, j_src = _edge_interior_ij(q_e, d, s_src, Nc, Hp)
                    i_dst, j_dst = _edge_halo_ij(e, d, s, Nc, Hp)
                    dst[i_dst, j_dst, k] = src[i_src, j_src, k]
                end
            end
        end
    else
        k! = _fill_edge_kernel!(backend, 256)
        k!(dst, src, e, q_e, flip, Nc, Hp; ndrange=(Nc, Hp, Nk))
    end
    return nothing
end

@kernel function _fill_edge_kernel!(dst, @Const(src), e, q_e, flip, Nc, Hp)
    s, d, k = @index(Global, NTuple)
    @inbounds begin
        s_src = flip ? (Nc + 1 - s) : s
        i_src, j_src = _edge_interior_ij(q_e, d, s_src, Nc, Hp)
        i_dst, j_dst = _edge_halo_ij(e, d, s, Nc, Hp)
        dst[i_dst, j_dst, k] = src[i_src, j_src, k]
    end
end

"""Interior array index (i, j) for the source panel at edge `q_e`, depth `d`,
along-edge position `s`. Depth 1 = boundary row/column."""
@inline function _edge_interior_ij(q_e::Int, d::Int, s::Int, Nc::Int, Hp::Int)
    if     q_e == 1  # north: read from top rows going inward
        return (Hp + s, Hp + Nc + 1 - d)
    elseif q_e == 2  # south: read from bottom rows going inward
        return (Hp + s, Hp + d)
    elseif q_e == 3  # east: read from right columns going inward
        return (Hp + Nc + 1 - d, Hp + s)
    else             # west: read from left columns going inward
        return (Hp + d, Hp + s)
    end
end

"""Halo array index (i, j) for destination panel at edge `e`, depth `d`,
along-edge position `s`. Depth 1 = immediately outside the interior."""
@inline function _edge_halo_ij(e::Int, d::Int, s::Int, Nc::Int, Hp::Int)
    if     e == 1  # north halo: above interior top
        return (Hp + s, Hp + Nc + d)
    elseif e == 2  # south halo: below interior bottom
        return (Hp + s, Hp + 1 - d)
    elseif e == 3  # east halo: right of interior right
        return (Hp + Nc + d, Hp + s)
    else           # west halo: left of interior left
        return (Hp + 1 - d, Hp + s)
    end
end

# ---------------------------------------------------------------------------
# Corner halo filling (cube vertex ghost cells)
#
# At each cube vertex, 3 panels meet. The edge halos (filled by
# fill_panel_halos!) cover the 4 edges but leave the Hp×Hp corner regions
# uninitialized. These corners are needed for the Lin-Rood cross-term
# correction where the inner PPM operates on the full halo extent.
#
# The rotation formulas are derived from FV3's `copy_corners` in tp_core.F90.
# They are direction-dependent: dir=1 for X-sweep, dir=2 for Y-sweep.
# The formulas are purely local (operate on a single panel's array using
# already-filled edge halo data) and are the same for all 6 panels due
# to the cube's rotational symmetry.
#
# Reference: Putman & Lin (2007), tp_core.F90 lines 246-321
# ---------------------------------------------------------------------------

"""
$(SIGNATURES)

Fill corner ghost cells for a cubed-sphere 3D field. Must be called AFTER
`fill_panel_halos!` (which fills edge halos that corner rotations read from).

`dir` selects the rotation:
- `dir=1`: for X-direction PPM (rotates Y-edge data into corners)
- `dir=2`: for Y-direction PPM (rotates X-edge data into corners)
"""
function copy_corners!(data::NTuple{6, A},
                       grid::CubedSphereGrid, dir::Int) where {A <: AbstractArray}
    Nc = grid.Nc
    Hp = grid.Hp
    N  = Nc + 2 * Hp  # total array extent per dimension

    for p in 1:6
        _fill_corners_panel!(data[p], Nc, Hp, N, dir)
    end
    synchronize(get_backend(data[1]))
    return nothing
end

"""Fill all 4 corner regions of a single panel array."""
function _fill_corners_panel!(q::AbstractArray{T, 3},
                              Nc::Int, Hp::Int, N::Int, dir::Int) where T
    Nk = size(q, 3)
    backend = get_backend(q)

    if backend isa KernelAbstractions.CPU
        @inbounds for k in 1:Nk
            for dj in 1:Hp
                for di in 1:Hp
                    _set_corner_cells!(q, di, dj, k, Nc, Hp, N, dir)
                end
            end
        end
    else
        k! = _copy_corners_kernel!(backend, 256)
        k!(q, Nc, Hp, N, dir; ndrange=(Hp, Hp, Nk))
    end
    return nothing
end

@kernel function _copy_corners_kernel!(q, Nc, Hp, N, dir)
    di, dj, k = @index(Global, NTuple)
    @inbounds _set_corner_cells!(q, di, dj, k, Nc, Hp, N, dir)
end

"""Set the 4 corner cells at offset (di, dj) within each corner region."""
@inline function _set_corner_cells!(q, di::Int, dj::Int, k::Int,
                                     Nc::Int, Hp::Int, N::Int, dir::Int)
    # Corner cell indices (1-based offsets within the Hp×Hp corner block)
    # SW corner: oi = Hp+1-di, oj = Hp+1-dj  (di=1 → nearest to interior)
    # SE corner: oi = Hp+Nc+di, oj = Hp+1-dj
    # NE corner: oi = Hp+Nc+di, oj = Hp+Nc+dj
    # NW corner: oi = Hp+1-di, oj = Hp+Nc+dj
    oi_sw = Hp + 1 - di;  oj_sw = Hp + 1 - dj
    oi_se = Hp + Nc + di; oj_se = Hp + 1 - dj
    oi_ne = Hp + Nc + di; oj_ne = Hp + Nc + dj
    oi_nw = Hp + 1 - di;  oj_nw = Hp + Nc + dj

    if dir == 1
        # X-direction: rotate Y-edge halos into corners
        # FV3 tp_core.F90 formulas translated to our indexing (oi = fi + Hp):
        #   SW: q(fi,fj) = q(fj, 1-fi)       → q[oi,oj] = q[oj, 2Hp+1-oi]
        #   SE: q(fi,fj) = q(npy-fj, fi-npx+1) → q[oi,oj] = q[N+1-oj, oi-Nc]
        #   NE: q(fi,fj) = q(fj, 2npx-1-fi)  → q[oi,oj] = q[oj, 2(Nc+Hp)+1-oi]
        #   NW: q(fi,fj) = q(npy-fj, fi-1+npx) → q[oi,oj] = q[N+1-oj, oi+Nc]
        q[oi_sw, oj_sw, k] = q[oj_sw, 2*Hp + 1 - oi_sw, k]
        q[oi_se, oj_se, k] = q[N + 1 - oj_se, oi_se - Nc, k]
        q[oi_ne, oj_ne, k] = q[oj_ne, 2*(Nc + Hp) + 1 - oi_ne, k]
        q[oi_nw, oj_nw, k] = q[N + 1 - oj_nw, oi_nw + Nc, k]
    else
        # Y-direction: rotate X-edge halos into corners
        # FV3 tp_core.F90 formulas translated:
        #   SW: q(fi,fj) = q(1-fj, fi)       → q[oi,oj] = q[2Hp+1-oj, oi]
        #   SE: q(fi,fj) = q(npy+fj-1, npx-fi) → q[oi,oj] = q[Nc+oj, N+1-oi]
        #   NE: q(fi,fj) = q(2npy-1-fj, fi)  → q[oi,oj] = q[2(Nc+Hp)+1-oj, oi]
        #   NW: q(fi,fj) = q(fj+1-npx, npy-fi) → q[oi,oj] = q[oj-Nc, N+1-oi]
        q[oi_sw, oj_sw, k] = q[2*Hp + 1 - oj_sw, oi_sw, k]
        q[oi_se, oj_se, k] = q[Nc + oj_se, N + 1 - oi_se, k]
        q[oi_ne, oj_ne, k] = q[2*(Nc + Hp) + 1 - oj_ne, oi_ne, k]
        q[oi_nw, oj_nw, k] = q[oj_nw - Nc, N + 1 - oi_nw, k]
    end
    return nothing
end
