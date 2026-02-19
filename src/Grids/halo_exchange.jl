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
"""
function _fill_edge!(dst::AbstractArray{T, 3}, src::AbstractArray{T, 3},
                     e::Int, q_e::Int, orient::Int,
                     Nc::Int, Hp::Int) where T
    Nk = size(dst, 3)
    flip = orient >= 2  # orient 0,1 → same direction; orient 2,3 → reversed

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
    return nothing
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
