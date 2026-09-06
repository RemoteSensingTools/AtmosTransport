# A physical cubed-sphere face must carry one tracer flux. The two panels'
# transverse predictors can produce different face mixing ratios, even when
# their air-mass fluxes are mirrored exactly. Share the mean of both panels'
# inner/outer estimates before either panel applies the flux divergence.
#
# This four-value averaging is a symmetric linear projection: its transpose
# is the same exchange, used by the Lin-Rood reverse pass. Mixing ratios are
# scalars, so only the edge ordering reverses; the oriented air-mass flux
# supplies the sign. Interior faces are untouched.

@inline function _lr_edge_face_index(edge, s, Nc)
    edge == EDGE_NORTH && return (s, Nc + 1)
    edge == EDGE_SOUTH && return (s, 1)
    edge == EDGE_EAST && return (Nc + 1, s)
    return (1, s)
end

@kernel function _share_lr_edge_kernel!(a_in, a_out, b_in, b_out,
                                       edge_a, edge_b, reverse_order, Nc)
    s, k = @index(Global, NTuple)
    t = reverse_order ? Nc + 1 - s : s
    i, j = _lr_edge_face_index(edge_a, s, Nc)
    u, v = _lr_edge_face_index(edge_b, t, Nc)
    half = eltype(a_in)(0.5)
    @inbounds begin
        common = half * (half * (a_in[i, j, k] + a_out[i, j, k]) +
                         half * (b_in[u, v, k] + b_out[u, v, k]))
        a_in[i, j, k] = common
        a_out[i, j, k] = common
        b_in[u, v, k] = common
        b_out[u, v, k] = common
    end
end

function _share_lr_seam_faces!(fx_in::NTuple{6, <:AbstractArray{FT, 3}},
                                fx_out::NTuple{6, <:AbstractArray{FT, 3}},
                                fy_in::NTuple{6, <:AbstractArray{FT, 3}},
                                fy_out::NTuple{6, <:AbstractArray{FT, 3}},
                                mesh::CubedSphereMesh) where FT
    Nc = mesh.Nc
    Nz = size(fx_in[1], 3)
    backend = get_backend(fx_in[1])
    kernel! = _share_lr_edge_kernel!(backend, 64)
    conn = mesh.connectivity
    for p in 1:6, edge in (EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST)
        neighbor = conn.neighbors[p][edge]
        q = neighbor.panel
        p < q || continue
        other_edge = reciprocal_edge(conn, p, edge)
        a_in, a_out = edge in (EDGE_EAST, EDGE_WEST) ?
            (fx_in[p], fx_out[p]) : (fy_in[p], fy_out[p])
        b_in, b_out = other_edge in (EDGE_EAST, EDGE_WEST) ?
            (fx_in[q], fx_out[q]) : (fy_in[q], fy_out[q])
        kernel!(a_in, a_out, b_in, b_out, edge, other_edge,
                neighbor.orientation != 0, Nc; ndrange=(Nc, Nz))
    end
    synchronize(backend)
    return nothing
end
