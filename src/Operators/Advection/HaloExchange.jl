# ---------------------------------------------------------------------------
# Cubed-sphere panel halo exchange for src
#
# Fills halo regions for 3D fields stored as NTuple{6, Array{FT,3}}
# with extended dimensions (Nc + 2*Hp) × (Nc + 2*Hp) × Nz per panel.
# Interior data lives at indices [Hp+1:Hp+Nc, Hp+1:Hp+Nc, :].
#
# Edge convention:
#   1 = North (top,    j = Nc)
#   2 = South (bottom, j = 1)
#   3 = East  (right,  i = Nc)
#   4 = West  (left,   i = 1)
#
# Orientation codes (from PanelConnectivity):
#   0 = aligned  → along-edge same direction
#   2 = reversed → along-edge reversed
#
# Ported from src/Grids/halo_exchange.jl, adapted for src types.
#
# References:
#   Putman & Lin (2007) — FV3 cubed-sphere grid connectivity
#   Martin et al. (2022, GMD) — GCHP panel exchange
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend, CPU as KA_CPU

# ---------------------------------------------------------------------------
# Edge indexing helpers
# ---------------------------------------------------------------------------

"""Map `(edge, depth, along)` to the interior (i, j) that provides source data."""
@inline function _edge_interior_ij(q_e::Int, d::Int, s::Int, Nc::Int, Hp::Int)
    if     q_e == EDGE_NORTH  # read from top rows going inward
        return (Hp + s, Hp + Nc + 1 - d)
    elseif q_e == EDGE_SOUTH  # read from bottom rows going inward
        return (Hp + s, Hp + d)
    elseif q_e == EDGE_EAST   # read from right columns going inward
        return (Hp + Nc + 1 - d, Hp + s)
    else  # EDGE_WEST         # read from left columns going inward
        return (Hp + d, Hp + s)
    end
end

"""Map `(edge, depth, along)` to the halo (i, j) that receives data."""
@inline function _edge_halo_ij(e::Int, d::Int, s::Int, Nc::Int, Hp::Int)
    if     e == EDGE_NORTH  # above interior top
        return (Hp + s, Hp + Nc + d)
    elseif e == EDGE_SOUTH  # below interior bottom
        return (Hp + s, Hp + 1 - d)
    elseif e == EDGE_EAST   # right of interior
        return (Hp + Nc + d, Hp + s)
    else  # EDGE_WEST       # left of interior
        return (Hp + 1 - d, Hp + s)
    end
end

# ---------------------------------------------------------------------------
# Edge fill (CPU + GPU kernel)
# ---------------------------------------------------------------------------

"""Fill one edge's halo from the neighbor panel's interior."""
function _fill_edge!(dst::AbstractArray{T, 3}, src::AbstractArray{T, 3},
                     e::Int, q_e::Int, orient::Int,
                     Nc::Int, Hp::Int) where T
    Nk = size(dst, 3)
    flip = orient >= 2

    backend = get_backend(dst)
    if backend isa KA_CPU
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
        synchronize(backend)
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

# ---------------------------------------------------------------------------
# Corner fill — FV3 tp_core.F90 rotation formulas
# ---------------------------------------------------------------------------

"""
    _set_corner_cells!(q, di, dj, k, Nc, Hp, N, dir)

Fill all 4 corner cells at offset `(di, dj)` within each `Hp × Hp` corner block.

`dir = 1` for X-sweep (rotates Y-edge halos into corners).
`dir = 2` for Y-sweep (rotates X-edge halos into corners).

Formulas translated from FV3 `tp_core.F90` via `src/Grids/halo_exchange.jl`.
"""
@inline function _set_corner_cells!(q, di::Int, dj::Int, k::Int,
                                     Nc::Int, Hp::Int, N::Int, dir::Int)
    oi_sw = Hp + 1 - di;  oj_sw = Hp + 1 - dj
    oi_se = Hp + Nc + di; oj_se = Hp + 1 - dj
    oi_ne = Hp + Nc + di; oj_ne = Hp + Nc + dj
    oi_nw = Hp + 1 - di;  oj_nw = Hp + Nc + dj

    @inbounds if dir == 1
        # X-direction: rotate Y-edge halos into corners
        q[oi_sw, oj_sw, k] = q[oj_sw, 2*Hp + 1 - oi_sw, k]
        q[oi_se, oj_se, k] = q[N + 1 - oj_se, oi_se - Nc, k]
        q[oi_ne, oj_ne, k] = q[oj_ne, 2*(Nc + Hp) + 1 - oi_ne, k]
        q[oi_nw, oj_nw, k] = q[N + 1 - oj_nw, oi_nw + Nc, k]
    else
        # Y-direction: rotate X-edge halos into corners
        q[oi_sw, oj_sw, k] = q[2*Hp + 1 - oj_sw, oi_sw, k]
        q[oi_se, oj_se, k] = q[Nc + oj_se, N + 1 - oi_se, k]
        q[oi_ne, oj_ne, k] = q[2*(Nc + Hp) + 1 - oj_ne, oi_ne, k]
        q[oi_nw, oj_nw, k] = q[oj_nw - Nc, N + 1 - oi_nw, k]
    end
    return nothing
end

@kernel function _copy_corners_kernel!(q, Nc, Hp, N, dir)
    di, dj, k = @index(Global, NTuple)
    @inbounds _set_corner_cells!(q, di, dj, k, Nc, Hp, N, dir)
end

"""Fill corners for all 6 panels of a single field."""
function _fill_corners!(panels::NTuple{6}, Nc::Int, Hp::Int, dir::Int)
    Hp == 0 && return nothing
    N = Nc + 2 * Hp
    for p in 1:6
        q = panels[p]
        Nk = size(q, 3)
        backend = get_backend(q)
        if backend isa KA_CPU
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
            synchronize(backend)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    fill_panel_halos!(panels::NTuple{6}, mesh::CubedSphereMesh; dir=0)

Fill halo regions of a 6-panel cubed-sphere field by copying interior data
from neighboring panels with correct edge-to-edge orientation mapping.

Each `panels[p]` must be `(Nc + 2Hp) × (Nc + 2Hp) × Nz` with interior at
`[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]`.

If `dir` is 1 or 2, corner fill is performed for the given sweep direction
(1=X, 2=Y) using the FV3 tp_core rotation formulas.
"""
function fill_panel_halos!(panels::NTuple{6, A},
                           mesh::CubedSphereMesh;
                           dir::Int = 0) where {A <: AbstractArray}
    Nc, Hp = mesh.Nc, mesh.Hp
    Hp == 0 && return nothing

    conn = mesh.connectivity
    # Fill all 24 edges (6 panels × 4 edges)
    for p in 1:6
        for e in 1:4
            nb = conn.neighbors[p][e]
            q_e = reciprocal_edge(conn, p, e)
            _fill_edge!(panels[p], panels[nb.panel], e, q_e, nb.orientation, Nc, Hp)
        end
    end

    # Corner rotation if a sweep direction is specified
    if dir in (1, 2)
        _fill_corners!(panels, Nc, Hp, dir)
    end

    return nothing
end

export fill_panel_halos!
