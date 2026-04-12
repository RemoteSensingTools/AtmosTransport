# ---------------------------------------------------------------------------
# Global multi-panel Poisson mass-flux balance for cubed-sphere grids.
#
# Unlike the per-panel FFT approach (which treats each panel as doubly-
# periodic and ignores cross-panel continuity), this solver operates on
# a GLOBAL face table that includes all Nc cross-panel boundary faces.
# It uses the same Jacobi-preconditioned CG solver as the reduced-
# Gaussian Poisson balance, applied to the global graph Laplacian.
#
# On a 6-panel CS with Nc cells per edge:
#   - 6 × Nc² = total cells
#   - Interior faces per panel: (Nc-1)×Nc (x-dir) + Nc×(Nc-1) (y-dir)
#     = 2Nc² - 2Nc per panel
#   - Cross-panel faces: 12 edges × Nc faces per edge = 12Nc
#     (but each edge is shared by exactly 2 panels, so 12Nc/2 = 6Nc
#      unique cross-panel faces... actually, each of the 12 panel-edges
#      contributes Nc faces, and each cross-panel connection is a PAIR
#      of panel edges, giving 12 edges / 2 = 6 connections × Nc = 6Nc
#      unique cross-panel faces)
#   - Total faces: 12 × Nc².
#     Per panel: 2Nc(Nc-1) strictly interior faces + 4Nc boundary faces.
#     6 panels × 2Nc(Nc-1) = 12Nc² - 12Nc interior faces.
#     12 unique cube edges × Nc faces per edge = 12Nc cross-panel faces.
#     Total: 12Nc² - 12Nc + 12Nc = 12Nc².
#     Degree: every cell has exactly 4 neighbors → degree = 4 everywhere.
#
# The graph Laplacian L = D - A with degree 4 everywhere has a 1-D
# constant null space (the sphere is a closed surface). The solver
# uses mean-zero projection, identical to the RG path.
#
# References:
#   - reduced_transport_helpers.jl: balance_reduced_horizontal_fluxes!
#   - PanelConnectivity.jl: default_panel_connectivity()
# ---------------------------------------------------------------------------

using LinearAlgebra: dot

# ---------------------------------------------------------------------------
# Global cell and face indexing
# ---------------------------------------------------------------------------

"""
    _global_cell(i, j, p, Nc) -> Int

Map panel-local cell `(i, j)` on panel `p` to a global cell index in `1:6Nc²`.
Column-major within each panel: cell `(i, j, p)` → `(p-1)*Nc² + (j-1)*Nc + i`.
"""
@inline _global_cell(i::Int, j::Int, p::Int, Nc::Int) = (p - 1) * Nc * Nc + (j - 1) * Nc + i

"""
    CSGlobalFaceTable

Global face table for a 6-panel cubed sphere.

# Fields
- `face_left  :: Vector{Int32}` — global cell index on the "left" side of each face
- `face_right :: Vector{Int32}` — global cell index on the "right" side of each face
- `nf :: Int` — total number of faces (= 12 × Nc²)
- `nc :: Int` — total number of cells (= 6 × Nc²)
- `Nc :: Int` — cells per panel edge

## Per-face back-mapping to per-panel am/bm arrays

Each face in the global table corresponds to an entry in some panel's `am` or `bm`
array. To apply the correction `δψ[right] - δψ[left]` back to the per-panel arrays,
we store the **canonical** location of each face:

- `face_panel :: Vector{Int32}` — which panel owns this face (1-6)
- `face_dir   :: Vector{Int32}` — direction: 1 = x-face (stored in am), 2 = y-face (stored in bm)
- `face_idx_i :: Vector{Int32}` — the `i` index into `am[i, j, k]` or `bm[i, j, k]`
- `face_idx_j :: Vector{Int32}` — the `j` index

For cross-panel faces, there's also a **mirror** entry on the neighbor panel:

- `mirror_panel :: Vector{Int32}` — 0 for interior faces; neighbor panel for cross-panel faces
- `mirror_dir   :: Vector{Int32}` — direction of the mirror entry
- `mirror_idx_i :: Vector{Int32}` — the `i` index of the mirror entry
- `mirror_idx_j :: Vector{Int32}` — the `j` index of the mirror entry

Both canonical and mirror entries store the same physical flux (up to sign).
The sign convention is: flux positive → mass flows from `face_left` to `face_right`.
For interior x-faces, `am[i, j]` flows from cell `(i-1, j)` to `(i, j)`, so
`face_left = global(i-1, j, p)` and `face_right = global(i, j, p)`.
"""
struct CSGlobalFaceTable
    face_left   :: Vector{Int32}
    face_right  :: Vector{Int32}
    face_panel  :: Vector{Int32}
    face_dir    :: Vector{Int32}
    face_idx_i  :: Vector{Int32}
    face_idx_j  :: Vector{Int32}
    mirror_panel :: Vector{Int32}
    mirror_dir   :: Vector{Int32}
    mirror_idx_i :: Vector{Int32}
    mirror_idx_j :: Vector{Int32}
    nf :: Int
    nc :: Int
    Nc :: Int
end

"""
    build_cs_global_face_table(Nc, conn) -> CSGlobalFaceTable

Build the global face table for a C`Nc` cubed sphere with panel connectivity
`conn`. Enumerates all `12Nc²` unique faces (interior + cross-panel).

For cross-panel faces, the canonical entry is on the panel with the lower
panel index (tie-breaking ensures each cross-panel face appears exactly once).

The cross-panel cell pairing uses `PanelConnectivity` to determine which
cell on the neighbor panel shares the boundary face. The `orientation` field
(0 = aligned, 2 = reversed) controls the along-edge index mapping.

## Cross-panel edge geometry

Consider panel `p` with its north edge (`j = Nc`) connecting to neighbor
panel `q` at some edge `eq` (north/south/east/west). The cells along
panel `p`'s north edge are `(s, Nc)` for `s = 1:Nc`. The corresponding
cells on panel `q` depend on which edge of `q` is connected and whether
the orientation is aligned or reversed:

| Neighbor edge `eq` | Aligned (ori=0)      | Reversed (ori=2)         |
|--------------------|----------------------|--------------------------|
| south (j=1)        | (s, 1)               | (Nc+1-s, 1)             |
| north (j=Nc)       | (s, Nc)              | (Nc+1-s, Nc)            |
| west (i=1)         | (1, s)               | (1, Nc+1-s)             |
| east (i=Nc)        | (Nc, s)              | (Nc, Nc+1-s)            |

For the canonical face, we store the face as a bm entry on panel `p`
at `bm[s, Nc+1, k]` (the north boundary of cell `(s, Nc)`). The
mirror entry is the corresponding face on panel `q`.
"""
function build_cs_global_face_table(Nc::Int, conn::PanelConnectivity)
    # Maximum possible faces: 12Nc²
    max_nf = 12 * Nc^2
    fl = Vector{Int32}(undef, max_nf)
    fr = Vector{Int32}(undef, max_nf)
    fp = Vector{Int32}(undef, max_nf)
    fd = Vector{Int32}(undef, max_nf)
    fi = Vector{Int32}(undef, max_nf)
    fj = Vector{Int32}(undef, max_nf)
    mp = zeros(Int32, max_nf)
    md = zeros(Int32, max_nf)
    mi = zeros(Int32, max_nf)
    mj = zeros(Int32, max_nf)

    nf = 0

    # --- Phase 1: Interior faces (within each panel) ---
    for p in 1:6
        # Interior x-faces: am[i, j] for i ∈ 2:Nc, j ∈ 1:Nc
        # Flux flows from cell (i-1, j) to cell (i, j)
        for j in 1:Nc, i in 2:Nc
            nf += 1
            fl[nf] = _global_cell(i - 1, j, p, Nc)
            fr[nf] = _global_cell(i,     j, p, Nc)
            fp[nf] = Int32(p)
            fd[nf] = Int32(1)   # x-face → am
            fi[nf] = Int32(i)
            fj[nf] = Int32(j)
        end
        # Interior y-faces: bm[i, j] for i ∈ 1:Nc, j ∈ 2:Nc
        # Flux flows from cell (i, j-1) to cell (i, j)
        for j in 2:Nc, i in 1:Nc
            nf += 1
            fl[nf] = _global_cell(i, j - 1, p, Nc)
            fr[nf] = _global_cell(i, j,     p, Nc)
            fp[nf] = Int32(p)
            fd[nf] = Int32(2)   # y-face → bm
            fi[nf] = Int32(i)
            fj[nf] = Int32(j)
        end
    end

    n_interior = nf

    # --- Phase 2: Cross-panel faces ---
    #
    # Each of the 12 panel edges has Nc boundary faces. Each edge connects
    # to exactly one neighbor edge, so there are 12/2 = 6 unique cross-panel
    # edges × Nc faces = 6Nc unique cross-panel faces.
    #
    # Edge numbering: 1=north (j=Nc), 2=south (j=1), 3=east (i=Nc), 4=west (i=1)
    #
    # Sign convention: face_left is the cell on the OUTGOING edge (north/east),
    # face_right is the cell on the INCOMING edge (south/west). This ensures
    # that positive flux in the per-panel array (which means "mass exits panel
    # through an outgoing edge" or "mass enters panel through an incoming edge")
    # consistently means "mass flows from face_left to face_right".
    #
    # On a cubed sphere, every cross-panel connection pairs an outgoing edge
    # with an incoming edge:
    #   north(out) ↔ south(in) or west(in)
    #   east(out)  ↔ south(in) or west(in)
    #   south(in)  ↔ north(out) or east(out)
    #   west(in)   ↔ north(out) or east(out)
    #
    # To avoid double-counting, we only create faces from the panel with the
    # OUTGOING edge (north or east). The canonical face entry is stored on
    # the outgoing panel's boundary, the mirror on the incoming panel's boundary.
    #
    # Outgoing edges: north (e=1) and east (e=3).

    for p in 1:6
        for e in (EDGE_NORTH, EDGE_EAST)  # only outgoing edges
            q   = conn.neighbors[p][e].panel
            ori = conn.neighbors[p][e].orientation
            eq  = _reciprocal_edge(conn, p, e)

            for s in 1:Nc
                t = ori == 0 ? s : Nc + 1 - s

                # Cell on panel p at outgoing edge e, position s:
                cp = _edge_cell(p, e, s, Nc)
                # Cell on panel q at incoming edge eq, position t:
                cq = _edge_cell(q, eq, t, Nc)

                gp = _global_cell(cp[1], cp[2], p, Nc)
                gq = _global_cell(cq[1], cq[2], q, Nc)

                # Canonical: outgoing boundary face on panel p
                # Mirror: incoming boundary face on panel q
                can_dir, can_i, can_j = _edge_face_location(e, s, Nc)
                mir_dir, mir_i, mir_j = _edge_face_location(eq, t, Nc)

                nf += 1
                # face_left = outgoing cell on p, face_right = incoming cell on q
                fl[nf] = gp
                fr[nf] = gq
                fp[nf] = Int32(p)
                fd[nf] = Int32(can_dir)
                fi[nf] = Int32(can_i)
                fj[nf] = Int32(can_j)
                mp[nf] = Int32(q)
                md[nf] = Int32(mir_dir)
                mi[nf] = Int32(mir_i)
                mj[nf] = Int32(mir_j)
            end
        end
    end

    n_cross = nf - n_interior
    nc = 6 * Nc^2

    @assert nf == 12 * Nc^2 "Expected $(12*Nc^2) faces, got $nf " *
        "(interior=$n_interior, cross=$n_cross)"

    return CSGlobalFaceTable(
        fl[1:nf], fr[1:nf], fp[1:nf], fd[1:nf], fi[1:nf], fj[1:nf],
        mp[1:nf], md[1:nf], mi[1:nf], mj[1:nf],
        nf, nc, Nc,
    )
end

"""
    _edge_cell(p, edge, s, Nc) -> (i, j)

Return the cell indices `(i, j)` on panel `p` at position `s` along edge `edge`.
Edge 1=north, 2=south, 3=east, 4=west.
"""
@inline function _edge_cell(::Int, edge::Int, s::Int, Nc::Int)
    if edge == 1       # north: j = Nc, i = s
        return (s, Nc)
    elseif edge == 2   # south: j = 1, i = s
        return (s, 1)
    elseif edge == 3   # east: i = Nc, j = s
        return (Nc, s)
    else               # west: i = 1, j = s
        return (1, s)
    end
end

"""
    _edge_face_location(edge, s, Nc) -> (dir, i, j)

Return the per-panel am/bm index for the boundary face at position `s` along
edge `edge`.

- `dir = 1` → x-face stored in `am[i, j, k]`
- `dir = 2` → y-face stored in `bm[i, j, k]`

| Edge    | Face array   | Index         |
|---------|-------------|---------------|
| north   | bm[s, Nc+1] | dir=2, i=s, j=Nc+1 |
| south   | bm[s, 1]    | dir=2, i=s, j=1     |
| east    | am[Nc+1, s] | dir=1, i=Nc+1, j=s  |
| west    | am[1, s]    | dir=1, i=1, j=s      |
"""
@inline function _edge_face_location(edge::Int, s::Int, Nc::Int)
    if edge == 1       # north boundary
        return (2, s, Nc + 1)
    elseif edge == 2   # south boundary
        return (2, s, 1)
    elseif edge == 3   # east boundary
        return (1, Nc + 1, s)
    else               # west boundary
        return (1, 1, s)
    end
end

"""
    _reciprocal_edge(conn, p, e) -> Int

Find which edge of the neighbor panel connects back to panel `p` edge `e`.
"""
function _reciprocal_edge(conn::PanelConnectivity, p::Int, e::Int)
    q = conn.neighbors[p][e].panel
    for eq in 1:4
        conn.neighbors[q][eq].panel == p && return eq
    end
    error("Broken panel connectivity: no reciprocal edge for P$p edge $e → P$q")
end

# ---------------------------------------------------------------------------
# Graph Laplacian and CG solver (reused from reduced_transport_helpers.jl)
# ---------------------------------------------------------------------------

"""
    cs_cell_face_degree(ft::CSGlobalFaceTable) -> Vector{Int}

Compute the face degree for each global cell. On a closed cubed sphere
every cell has exactly 4 neighbors (no boundary stubs), so degree should
be 4 everywhere. This function verifies that invariant.
"""
function cs_cell_face_degree(ft::CSGlobalFaceTable)
    degree = zeros(Int, ft.nc)
    @inbounds for f in 1:ft.nf
        degree[ft.face_left[f]]  += 1
        degree[ft.face_right[f]] += 1
    end
    # Verify: every cell should have degree 4
    for c in 1:ft.nc
        degree[c] == 4 || @warn "Cell $c has degree $(degree[c]) (expected 4)"
    end
    return degree
end

"""
    _cs_graph_laplacian_mul!(out, psi, ft, degree)

Compute `out = L · psi` for the global CS graph Laplacian.
"""
function _cs_graph_laplacian_mul!(out::AbstractVector{Float64},
                                  psi::AbstractVector{Float64},
                                  ft::CSGlobalFaceTable,
                                  degree::Vector{Int})
    @inbounds for c in eachindex(out)
        out[c] = degree[c] * psi[c]
    end
    @inbounds for f in 1:ft.nf
        left  = Int(ft.face_left[f])
        right = Int(ft.face_right[f])
        out[left]  -= psi[right]
        out[right] -= psi[left]
    end
    return out
end

@inline function _project_mean_zero!(v::AbstractVector{Float64})
    s = sum(v) / length(v)
    @. v -= s
    return v
end

"""
    solve_cs_poisson_pcg!(psi, rhs, ft, degree, scratch; tol=1e-14, max_iter=20000)

Jacobi-Preconditioned CG for the global CS graph Laplacian `L · psi = rhs`.

L has a 1-D constant null space (closed surface), so we project `rhs`
and iterates to mean-zero. Returns `(residual_linfty, iterations)`.

Identical algorithm to `solve_graph_poisson_pcg!` in reduced_transport_helpers.jl,
but using the CS-specific Laplacian multiply.
"""
function solve_cs_poisson_pcg!(psi::AbstractVector{Float64},
                                rhs::AbstractVector{Float64},
                                ft::CSGlobalFaceTable,
                                degree::Vector{Int},
                                scratch;
                                tol::Float64=1e-14,
                                max_iter::Int=20000)
    r  = scratch.r
    p  = scratch.p
    Ap = scratch.Ap
    z  = scratch.z

    _project_mean_zero!(rhs)
    fill!(psi, 0.0)
    copyto!(r, rhs)

    # Jacobi preconditioner: z = M⁻¹ r, M = diag(degree)
    @inbounds for c in eachindex(r)
        z[c] = degree[c] > 0 ? r[c] / degree[c] : r[c]
    end
    copyto!(p, z)

    rz_old = dot(r, z)
    rhs_linf = maximum(abs, rhs) + eps()

    iter = 0
    r_linf = rhs_linf
    while iter < max_iter
        _cs_graph_laplacian_mul!(Ap, p, ft, degree)
        pAp = dot(p, Ap)
        pAp <= 0 && break
        alpha = rz_old / pAp
        @. psi += alpha * p
        @. r   -= alpha * Ap

        _project_mean_zero!(r)

        r_linf = maximum(abs, r)
        r_linf / rhs_linf < tol && break

        @inbounds for c in eachindex(r)
            z[c] = degree[c] > 0 ? r[c] / degree[c] : r[c]
        end
        _project_mean_zero!(z)
        rz_new = dot(r, z)
        rz_old == 0.0 && break
        beta = rz_new / rz_old
        @. p = z + beta * p
        rz_old = rz_new
        iter += 1
    end

    _project_mean_zero!(psi)
    return r_linf, iter
end

# ---------------------------------------------------------------------------
# Correction application: map global ψ to per-panel am/bm
# ---------------------------------------------------------------------------

"""
    apply_cs_flux_correction!(panels_am, panels_bm, psi, ft, k)

Apply the Poisson correction to per-panel `am` and `bm` arrays at level `k`.

For each face `f` in the global table, the correction is:

    δflux = psi[face_right[f]] - psi[face_left[f]]

This is added to the canonical am/bm entry. For cross-panel faces, the
mirror entry on the neighbor panel is set EQUAL to the corrected canonical
value (with appropriate sign). This is the key difference from the broken
approach of independently adding δψ to both entries: a cross-panel face
is ONE physical face stored in TWO arrays, so the corrected value must be
written to BOTH locations consistently.

## Sign convention for cross-panel mirrors

The canonical face has `face_left` on panel p and `face_right` on panel q.
The canonical am/bm entry is on panel p's boundary. The sign convention
of the canonical entry defines "outgoing from panel p" as positive.

The mirror entry on panel q stores the same flux but with the opposite
sign convention (positive = outgoing from panel q = INTO panel p). So:

    mirror_value = -canonical_value

For am entries: `am[i, j, k]` is positive when mass flows in the +x direction
(from cell (i-1, j) to cell (i, j)). At a west boundary (`am[1, j, k]`),
positive means mass flows INTO the panel. At an east boundary (`am[Nc+1, j, k]`),
positive means mass flows OUT of the panel.

For bm entries: analogous for +y direction.

The actual sign relationship between canonical and mirror depends on which
edges are connected. We handle this by directly computing the correction
for each entry from the global ψ values, rather than trying to infer a
sign flip.
"""
function apply_cs_flux_correction!(panels_am::NTuple{6, Array{FT, 3}},
                                    panels_bm::NTuple{6, Array{FT, 3}},
                                    psi::AbstractVector{Float64},
                                    ft::CSGlobalFaceTable,
                                    k::Int) where FT
    @inbounds for f in 1:ft.nf
        left  = Int(ft.face_left[f])
        right = Int(ft.face_right[f])
        delta = psi[right] - psi[left]

        # Apply to canonical entry
        p   = Int(ft.face_panel[f])
        dir = Int(ft.face_dir[f])
        i   = Int(ft.face_idx_i[f])
        j   = Int(ft.face_idx_j[f])

        if dir == 1  # x-face → am
            panels_am[p][i, j, k] += FT(delta)
        else         # y-face → bm
            panels_bm[p][i, j, k] += FT(delta)
        end

        # Apply to mirror entry (cross-panel faces only)
        mq = Int(ft.mirror_panel[f])
        mq == 0 && continue   # interior face, no mirror

        mdir = Int(ft.mirror_dir[f])
        mi   = Int(ft.mirror_idx_i[f])
        mj   = Int(ft.mirror_idx_j[f])

        # On a cubed sphere, every cross-panel edge pairs an "outgoing"
        # boundary (north/east: positive flux = mass exits panel) with
        # an "incoming" boundary (south/west: positive flux = mass enters
        # panel). This means the physical flux has the SAME sign in both
        # the canonical and mirror arrays:
        #
        #   canonical on p's outgoing edge: positive = mass exits p
        #   mirror on q's incoming edge:    positive = mass enters q
        #
        # Both describe "mass flows from p to q" with a positive value.
        # So we copy the corrected canonical value directly (no negation).
        #
        # Verification: for the global divergence at cell cq on panel q,
        # the face contribution through an incoming edge (south/west) is
        # +flux, and flux = canonical_value > 0 means mass enters q,
        # reducing q's outgoing divergence. This matches the global
        # face table's div[right] -= flux (since cq = face_right and
        # the flux is positive when mass flows from left to right).
        # Read the corrected canonical flux value (may be in am or bm
        # depending on canonical dir, not mirror dir).
        canonical_val = dir == 1 ? panels_am[p][i, j, k] : panels_bm[p][i, j, k]
        if mdir == 1  # x-face → am on neighbor
            panels_am[mq][mi, mj, k] = canonical_val
        else          # y-face → bm on neighbor
            panels_bm[mq][mi, mj, k] = canonical_val
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# High-level balance entry point
# ---------------------------------------------------------------------------

"""
    CSPoissonScratch

Pre-allocated work buffers for the CS global Poisson balance.
"""
struct CSPoissonScratch
    psi :: Vector{Float64}
    rhs :: Vector{Float64}
    r   :: Vector{Float64}
    p   :: Vector{Float64}
    Ap  :: Vector{Float64}
    z   :: Vector{Float64}
    div :: Vector{Float64}
end

function CSPoissonScratch(nc::Int)
    return CSPoissonScratch(
        zeros(Float64, nc), zeros(Float64, nc),
        zeros(Float64, nc), zeros(Float64, nc),
        zeros(Float64, nc), zeros(Float64, nc),
        zeros(Float64, nc),
    )
end

"""
    balance_cs_global_mass_fluxes!(panels_am, panels_bm, panels_m, panels_m_next,
                                    ft, degree, steps_per_window, scratch;
                                    tol=1e-14, max_iter=20000)

TM5-style global Poisson mass-flux balance for a 6-panel cubed sphere.

Corrects `panels_am[p]` and `panels_bm[p]` so that horizontal flux
convergence at every cell on every panel matches the prescribed
forward-window mass tendency per substep:

    dm_dt[c, k] = (m_next[c, k] - m_cur[c, k]) / (2 × steps_per_window)

Works level-by-level:
1. Compute current horizontal divergence from all 6 panels' am/bm
2. Form RHS = divergence - target dm_dt
3. Solve global graph Poisson: L · ψ = RHS
4. Correct all faces (interior + cross-panel) by δflux = ψ[right] - ψ[left]
5. Synchronize cross-panel mirror entries to match canonical (all levels)

Returns a diagnostic NamedTuple:
- `max_pre_residual` — max |RHS| before balance
- `max_post_residual` — max |divergence - target| after balance
- `max_cg_iter` — worst CG iteration count across levels
"""
function balance_cs_global_mass_fluxes!(
    panels_am::NTuple{6, Array{FT, 3}},
    panels_bm::NTuple{6, Array{FT, 3}},
    panels_m::NTuple{6, Array{FT, 3}},
    panels_m_next::NTuple{6, Array{FT, 3}},
    ft::CSGlobalFaceTable,
    degree::Vector{Int},
    steps_per_window::Int,
    scratch::CSPoissonScratch;
    tol::Float64=1e-14,
    max_iter::Int=20000,
) where FT

    Nc = ft.Nc
    Nz = size(panels_am[1], 3)
    nc = ft.nc
    inv_scale = 1.0 / (2.0 * steps_per_window)

    max_pre = 0.0
    max_post = 0.0
    max_it = 0
    max_rhs_mean = 0.0
    max_pre_proj = 0.0
    max_post_proj = 0.0

    cg_scratch = (r = scratch.r, p = scratch.p, Ap = scratch.Ap, z = scratch.z)

    for k in 1:Nz
        # 1. Compute current horizontal divergence at each global cell.
        #    div[c] = Σ_f (flux_out - flux_in) where the sign is determined
        #    by whether c is face_left (flux exits = positive contribution)
        #    or face_right (flux enters = negative contribution).
        #
        #    For the global face table, face_left is the "upstream" cell
        #    (flux positive → mass from left to right), so:
        #      div[left]  += flux     (mass leaves left)
        #      div[right] -= flux     (mass enters right)
        div = scratch.div
        fill!(div, 0.0)

        # Accumulate divergence from the canonical face entries
        @inbounds for f in 1:ft.nf
            p   = Int(ft.face_panel[f])
            dir = Int(ft.face_dir[f])
            i   = Int(ft.face_idx_i[f])
            j   = Int(ft.face_idx_j[f])

            flux = if dir == 1
                Float64(panels_am[p][i, j, k])
            else
                Float64(panels_bm[p][i, j, k])
            end

            left  = Int(ft.face_left[f])
            right = Int(ft.face_right[f])
            div[left]  += flux
            div[right] -= flux
        end

        # 2. RHS = divergence - target mass tendency
        rhs = scratch.rhs
        @inbounds for c in 1:nc
            p_idx = (c - 1) ÷ (Nc * Nc) + 1
            local_idx = (c - 1) % (Nc * Nc)
            j_local = local_idx ÷ Nc + 1
            i_local = local_idx % Nc + 1

            target = (Float64(panels_m_next[p_idx][i_local, j_local, k]) -
                      Float64(panels_m[p_idx][i_local, j_local, k])) * inv_scale
            rhs[c] = div[c] - target
        end

        # Diagnostics: raw pre-balance residual
        rhs_raw_linf = maximum(abs, rhs)
        rhs_raw_linf > max_pre && (max_pre = rhs_raw_linf)

        rhs_mean = sum(rhs) / nc
        abs(rhs_mean) > max_rhs_mean && (max_rhs_mean = abs(rhs_mean))

        pre_proj = 0.0
        @inbounds for c in 1:nc
            a = abs(rhs[c] - rhs_mean)
            a > pre_proj && (pre_proj = a)
        end
        pre_proj > max_pre_proj && (max_pre_proj = pre_proj)

        if pre_proj < tol
            continue  # already balanced on range(L)
        end

        # 3. Solve L · ψ = rhs
        psi = scratch.psi
        _, it = solve_cs_poisson_pcg!(psi, rhs, ft, degree, cg_scratch;
                                       tol=tol, max_iter=max_iter)
        it > max_it && (max_it = it)

        # Diagnostic: post-solve projected residual
        Lpsi = scratch.Ap
        _cs_graph_laplacian_mul!(Lpsi, psi, ft, degree)
        # Recompute projected rhs (since solve_cs_poisson_pcg! modifies rhs)
        fill!(div, 0.0)
        @inbounds for f in 1:ft.nf
            p   = Int(ft.face_panel[f])
            dir = Int(ft.face_dir[f])
            i   = Int(ft.face_idx_i[f])
            j   = Int(ft.face_idx_j[f])
            flux = dir == 1 ? Float64(panels_am[p][i, j, k]) :
                              Float64(panels_bm[p][i, j, k])
            left  = Int(ft.face_left[f])
            right = Int(ft.face_right[f])
            div[left]  += flux
            div[right] -= flux
        end
        @inbounds for c in 1:nc
            p_idx = (c - 1) ÷ (Nc * Nc) + 1
            local_idx = (c - 1) % (Nc * Nc)
            j_local = local_idx ÷ Nc + 1
            i_local = local_idx % Nc + 1
            target = (Float64(panels_m_next[p_idx][i_local, j_local, k]) -
                      Float64(panels_m[p_idx][i_local, j_local, k])) * inv_scale
            rhs[c] = (div[c] - target) - rhs_mean
        end
        post_proj = 0.0
        @inbounds for c in 1:nc
            a = abs(Lpsi[c] - rhs[c])
            a > post_proj && (post_proj = a)
        end
        post_proj > max_post_proj && (max_post_proj = post_proj)

        # 4. Apply correction to all faces (interior + cross-panel)
        apply_cs_flux_correction!(panels_am, panels_bm, psi, ft, k)

        # 5. Post-balance raw residual
        fill!(div, 0.0)
        @inbounds for f in 1:ft.nf
            p   = Int(ft.face_panel[f])
            dir = Int(ft.face_dir[f])
            i   = Int(ft.face_idx_i[f])
            j   = Int(ft.face_idx_j[f])
            flux = dir == 1 ? Float64(panels_am[p][i, j, k]) :
                              Float64(panels_bm[p][i, j, k])
            left  = Int(ft.face_left[f])
            right = Int(ft.face_right[f])
            div[left]  += flux
            div[right] -= flux
        end
        post_raw = 0.0
        @inbounds for c in 1:nc
            p_idx = (c - 1) ÷ (Nc * Nc) + 1
            local_idx = (c - 1) % (Nc * Nc)
            j_local = local_idx ÷ Nc + 1
            i_local = local_idx % Nc + 1
            target = (Float64(panels_m_next[p_idx][i_local, j_local, k]) -
                      Float64(panels_m[p_idx][i_local, j_local, k])) * inv_scale
            r = abs(div[c] - target)
            r > post_raw && (post_raw = r)
        end
        post_raw > max_post && (max_post = post_raw)
    end

    # 6. Synchronize ALL cross-panel mirror entries at ALL levels.
    #    This is done unconditionally (not just for levels that needed CG
    #    correction) because levels skipped by the `pre_proj < tol` test
    #    still need their mirror entries set to the canonical values.
    _sync_cs_mirrors!(panels_am, panels_bm, ft, Nz)

    return (;
        max_pre_residual = max_pre,
        max_post_residual = max_post,
        max_rhs_mean = max_rhs_mean,
        max_pre_projected = max_pre_proj,
        max_post_projected = max_post_proj,
        max_cg_iter = max_it,
    )
end

"""
    _sync_cs_mirrors!(panels_am, panels_bm, ft, Nz)

Copy canonical boundary flux values to their cross-panel mirror entries
for all levels. This ensures the per-panel am/bm arrays are consistent
at panel boundaries after the global CG correction.
"""
function _sync_cs_mirrors!(panels_am::NTuple{6, Array{FT, 3}},
                            panels_bm::NTuple{6, Array{FT, 3}},
                            ft::CSGlobalFaceTable,
                            Nz::Int) where FT
    @inbounds for f in 1:ft.nf
        mq = Int(ft.mirror_panel[f])
        mq == 0 && continue  # interior face

        p    = Int(ft.face_panel[f])
        dir  = Int(ft.face_dir[f])
        i    = Int(ft.face_idx_i[f])
        j    = Int(ft.face_idx_j[f])
        mdir = Int(ft.mirror_dir[f])
        mi   = Int(ft.mirror_idx_i[f])
        mj   = Int(ft.mirror_idx_j[f])

        for k in 1:Nz
            canonical_val = dir == 1 ? panels_am[p][i, j, k] : panels_bm[p][i, j, k]
            if mdir == 1
                panels_am[mq][mi, mj, k] = canonical_val
            else
                panels_bm[mq][mi, mj, k] = canonical_val
            end
        end
    end
    return nothing
end

"""
    diagnose_cs_cm!(panels_cm, panels_am, panels_bm, panels_dm, panels_m, Nc, Nz)

Diagnose vertical mass flux `cm` from horizontal flux divergence and mass
tendency for all 6 panels. Same algorithm as `diagnose_cm!` but applied
per-panel.

This should be called AFTER `balance_cs_global_mass_fluxes!` so the
horizontal fluxes are globally consistent. After balance, the residual
at cm[Nc, Nc, Nz+1] should be at machine precision.
"""
function diagnose_cs_cm!(panels_cm::NTuple{6, Array{FT, 3}},
                          panels_am::NTuple{6, Array{FT, 3}},
                          panels_bm::NTuple{6, Array{FT, 3}},
                          panels_dm::NTuple{6, Array{FT, 3}},
                          panels_m::NTuple{6, Array{FT, 3}},
                          Nc::Int, Nz::Int) where FT
    for p in 1:6
        am = panels_am[p]
        bm = panels_bm[p]
        cm = panels_cm[p]
        dm = panels_dm[p]
        m  = panels_m[p]

        @inbounds for j in 1:Nc, i in 1:Nc
            # Top boundary: no flux through TOA
            cm[i, j, 1] = zero(FT)

            # Cumulative: cm[k+1] = cm[k] + div_h[k] - dm[k]
            for k in 1:Nz
                div_h = (am[i, j, k] - am[i + 1, j, k]) +
                        (bm[i, j, k] - bm[i, j + 1, k])
                cm[i, j, k + 1] = cm[i, j, k] + div_h - dm[i, j, k]
            end

            # Bottom boundary residual: should be ~0 after global balance.
            # Redistribute any remaining residual proportionally to m.
            residual = cm[i, j, Nz + 1]
            if abs(residual) > eps(FT)
                total_m = zero(FT)
                for k in 1:Nz
                    total_m += m[i, j, k]
                end
                if total_m > zero(FT)
                    cum_fix = zero(FT)
                    for k in 1:Nz
                        frac = m[i, j, k] / total_m
                        cum_fix += frac * residual
                        cm[i, j, k + 1] -= cum_fix
                    end
                end
            end
        end
    end
    return nothing
end
