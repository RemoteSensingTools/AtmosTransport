# ---------------------------------------------------------------------------
# Global multi-panel Poisson mass-flux balance for cubed-sphere grids.
#
# Ported from scripts_legacy/preprocessing/cs_global_poisson_balance.jl
# into the modern src/Preprocessing/ pipeline.
#
# Unlike the per-panel FFT approach (which treats each panel as doubly-
# periodic and ignores cross-panel continuity), this solver operates on
# a GLOBAL face table that includes all cross-panel boundary faces.
# It uses Jacobi-preconditioned CG on the global graph Laplacian.
#
# On a 6-panel CS with Nc cells per edge:
#   - 6 × Nc² total cells
#   - 12 × Nc² total faces (degree = 4 everywhere on the closed sphere)
#   - Graph Laplacian L = D - A has a 1-D constant null space
#   - Solver uses mean-zero projection (same as the RG path)
#
# References:
#   - ring_poisson_balance.jl: reduced-Gaussian CG balance
#   - PanelConnectivity.jl: default_panel_connectivity()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Global cell and face indexing
# ---------------------------------------------------------------------------

"""
    _cs_global_cell(i, j, p, Nc) -> Int

Map panel-local cell `(i, j)` on panel `p` to a global cell index in `1:6Nc²`.
Column-major within each panel: cell `(i, j, p)` → `(p-1)*Nc² + (j-1)*Nc + i`.
"""
@inline _cs_global_cell(i::Int, j::Int, p::Int, Nc::Int) =
    (p - 1) * Nc * Nc + (j - 1) * Nc + i

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

- `face_panel :: Vector{Int32}` — which panel owns this face (1-6)
- `face_dir   :: Vector{Int32}` — direction: 1 = x-face (am), 2 = y-face (bm)
- `face_idx_i :: Vector{Int32}` — the `i` index into `am[i, j, k]` or `bm[i, j, k]`
- `face_idx_j :: Vector{Int32}` — the `j` index

For cross-panel faces, there's also a **mirror** entry on the neighbor panel:

- `mirror_panel :: Vector{Int32}` — 0 for interior faces; neighbor panel for cross-panel
- `mirror_dir   :: Vector{Int32}` — direction of the mirror entry
- `mirror_idx_i :: Vector{Int32}` — the `i` index of the mirror entry
- `mirror_idx_j :: Vector{Int32}` — the `j` index of the mirror entry

Sign convention: flux positive → mass flows from `face_left` to `face_right`.
"""
struct CSGlobalFaceTable
    face_left    :: Vector{Int32}
    face_right   :: Vector{Int32}
    face_panel   :: Vector{Int32}
    face_dir     :: Vector{Int32}
    face_idx_i   :: Vector{Int32}
    face_idx_j   :: Vector{Int32}
    mirror_panel :: Vector{Int32}
    mirror_dir   :: Vector{Int32}
    mirror_idx_i :: Vector{Int32}
    mirror_idx_j :: Vector{Int32}
    nf :: Int
    nc :: Int
    Nc :: Int
end

"""
    _cs_edge_cell(edge, s, Nc) -> (i, j)

Return the cell indices `(i, j)` at position `s` along edge `edge`.
Edge 1=north, 2=south, 3=east, 4=west.
"""
@inline function _cs_edge_cell(edge::Int, s::Int, Nc::Int)
    if edge == EDGE_NORTH       # j = Nc, i = s
        return (s, Nc)
    elseif edge == EDGE_SOUTH   # j = 1, i = s
        return (s, 1)
    elseif edge == EDGE_EAST    # i = Nc, j = s
        return (Nc, s)
    else                        # i = 1, j = s
        return (1, s)
    end
end

"""
    _cs_edge_face_location(edge, s, Nc) -> (dir, i, j)

Return the per-panel am/bm index for the boundary face at position `s` along
edge `edge`.

| Edge    | Face array   | Index             |
|---------|-------------|-------------------|
| north   | bm[s, Nc+1] | dir=2, i=s, j=Nc+1 |
| south   | bm[s, 1]    | dir=2, i=s, j=1     |
| east    | am[Nc+1, s] | dir=1, i=Nc+1, j=s  |
| west    | am[1, s]    | dir=1, i=1, j=s      |
"""
@inline function _cs_edge_face_location(edge::Int, s::Int, Nc::Int)
    if edge == EDGE_NORTH       # north boundary
        return (2, s, Nc + 1)
    elseif edge == EDGE_SOUTH   # south boundary
        return (2, s, 1)
    elseif edge == EDGE_EAST    # east boundary
        return (1, Nc + 1, s)
    else                        # west boundary
        return (1, 1, s)
    end
end

"""
    build_cs_global_face_table(Nc, conn) -> CSGlobalFaceTable

Build the global face table for a C`Nc` cubed sphere with panel connectivity
`conn`. Enumerates all `12Nc²` unique faces (interior + cross-panel).

Cross-panel faces are created only from outgoing edges (north, east) to avoid
double-counting. The canonical entry is on the outgoing panel, the mirror on
the incoming panel.
"""
function build_cs_global_face_table(Nc::Int, conn::PanelConnectivity)
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
        for j in 1:Nc, i in 2:Nc
            nf += 1
            fl[nf] = _cs_global_cell(i - 1, j, p, Nc)
            fr[nf] = _cs_global_cell(i,     j, p, Nc)
            fp[nf] = Int32(p)
            fd[nf] = Int32(1)   # x-face → am
            fi[nf] = Int32(i)
            fj[nf] = Int32(j)
        end
        # Interior y-faces: bm[i, j] for i ∈ 1:Nc, j ∈ 2:Nc
        for j in 2:Nc, i in 1:Nc
            nf += 1
            fl[nf] = _cs_global_cell(i, j - 1, p, Nc)
            fr[nf] = _cs_global_cell(i, j,     p, Nc)
            fp[nf] = Int32(p)
            fd[nf] = Int32(2)   # y-face → bm
            fi[nf] = Int32(i)
            fj[nf] = Int32(j)
        end
    end

    n_interior = nf

    # --- Phase 2: Cross-panel faces ---
    # Only from outgoing edges (north, east) to avoid double-counting.
    for p in 1:6
        for e in (EDGE_NORTH, EDGE_EAST)
            q   = conn.neighbors[p][e].panel
            ori = conn.neighbors[p][e].orientation
            eq  = reciprocal_edge(conn, p, e)

            for s in 1:Nc
                t = ori == 0 ? s : Nc + 1 - s

                cp = _cs_edge_cell(e, s, Nc)
                cq = _cs_edge_cell(eq, t, Nc)

                gp = _cs_global_cell(cp[1], cp[2], p, Nc)
                gq = _cs_global_cell(cq[1], cq[2], q, Nc)

                can_dir, can_i, can_j = _cs_edge_face_location(e, s, Nc)
                mir_dir, mir_i, mir_j = _cs_edge_face_location(eq, t, Nc)

                nf += 1
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

# ---------------------------------------------------------------------------
# Graph Laplacian and CG solver
# ---------------------------------------------------------------------------

"""
    cs_cell_face_degree(ft::CSGlobalFaceTable) -> Vector{Int}

Compute face degree for each global cell. On a closed CS, every cell has
degree 4.
"""
function cs_cell_face_degree(ft::CSGlobalFaceTable)
    degree = zeros(Int, ft.nc)
    @inbounds for f in 1:ft.nf
        degree[ft.face_left[f]]  += 1
        degree[ft.face_right[f]] += 1
    end
    for c in 1:ft.nc
        degree[c] == 4 || @warn "Cell $c has degree $(degree[c]) (expected 4)"
    end
    return degree
end

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

@inline function _cs_project_mean_zero!(v::AbstractVector{Float64})
    s = sum(v) / length(v)
    @. v -= s
    return v
end

"""
    solve_cs_poisson_pcg!(psi, rhs, ft, degree, scratch; tol=1e-14, max_iter=20000)

Jacobi-Preconditioned CG for the global CS graph Laplacian `L · psi = rhs`.

L has a 1-D constant null space (closed surface), so we project `rhs`
and iterates to mean-zero. Returns `(residual_linfty, iterations)`.
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

    _cs_project_mean_zero!(rhs)
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

        _cs_project_mean_zero!(r)

        r_linf = maximum(abs, r)
        r_linf / rhs_linf < tol && break

        @inbounds for c in eachindex(r)
            z[c] = degree[c] > 0 ? r[c] / degree[c] : r[c]
        end
        _cs_project_mean_zero!(z)
        rz_new = dot(r, z)
        rz_old == 0.0 && break
        beta = rz_new / rz_old
        @. p = z + beta * p
        rz_old = rz_new
        iter += 1
    end

    _cs_project_mean_zero!(psi)
    return r_linf, iter
end

# ---------------------------------------------------------------------------
# Correction application: map global ψ to per-panel am/bm
# ---------------------------------------------------------------------------

"""
    apply_cs_flux_correction!(panels_am, panels_bm, psi, ft, k)

Apply the Poisson correction to per-panel `am` and `bm` arrays at level `k`.

For each face, the correction is `δflux = psi[face_right] - psi[face_left]`.
Cross-panel mirror entries are set equal to the corrected canonical value
(same sign — both entries describe "mass flows from p to q" as positive).
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

        p   = Int(ft.face_panel[f])
        dir = Int(ft.face_dir[f])
        i   = Int(ft.face_idx_i[f])
        j   = Int(ft.face_idx_j[f])

        if dir == 1
            panels_am[p][i, j, k] += FT(delta)
        else
            panels_bm[p][i, j, k] += FT(delta)
        end

        mq = Int(ft.mirror_panel[f])
        mq == 0 && continue   # interior face

        mdir = Int(ft.mirror_dir[f])
        mi   = Int(ft.mirror_idx_i[f])
        mj   = Int(ft.mirror_idx_j[f])

        canonical_val = dir == 1 ? panels_am[p][i, j, k] : panels_bm[p][i, j, k]
        if mdir == 1
            panels_am[mq][mi, mj, k] = canonical_val
        else
            panels_bm[mq][mi, mj, k] = canonical_val
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Mirror synchronization
# ---------------------------------------------------------------------------

"""
    _sync_cs_mirrors!(panels_am, panels_bm, ft, Nz)

Copy canonical boundary flux values to their cross-panel mirror entries
for all levels.
"""
function _sync_cs_mirrors!(panels_am::NTuple{6, Array{FT, 3}},
                            panels_bm::NTuple{6, Array{FT, 3}},
                            ft::CSGlobalFaceTable,
                            Nz::Int) where FT
    @inbounds for f in 1:ft.nf
        mq = Int(ft.mirror_panel[f])
        mq == 0 && continue

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

# ---------------------------------------------------------------------------
# High-level balance entry point
# ---------------------------------------------------------------------------

"""
    balance_cs_global_mass_fluxes!(panels_am, panels_bm, panels_m, panels_m_next,
                                    ft, degree, steps_per_window, scratch;
                                    tol=1e-14, max_iter=20000)

TM5-style global Poisson mass-flux balance for a 6-panel cubed sphere.

Corrects `panels_am[p]` and `panels_bm[p]` so that horizontal flux
convergence at every cell matches the prescribed mass tendency:

    dm_dt[c, k] = (m_next[c, k] - m_cur[c, k]) / (2 × steps_per_window)

Returns a diagnostic NamedTuple with pre/post residuals and CG iteration counts.
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
        # 1. Compute current horizontal divergence
        div = scratch.div
        fill!(div, 0.0)

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

        # 4. Apply correction to all faces
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

# ---------------------------------------------------------------------------
# Vertical mass flux diagnosis
# ---------------------------------------------------------------------------

"""
    diagnose_cs_cm!(panels_cm, panels_am, panels_bm, panels_dm, panels_m, Nc, Nz)

Diagnose vertical mass flux `cm` from balanced horizontal flux divergence and
mass tendency for all 6 panels. Call AFTER `balance_cs_global_mass_fluxes!`.
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
            cm[i, j, 1] = zero(FT)

            for k in 1:Nz
                div_h = (am[i, j, k] - am[i + 1, j, k]) +
                        (bm[i, j, k] - bm[i, j + 1, k])
                cm[i, j, k + 1] = cm[i, j, k] + div_h - dm[i, j, k]
            end

            # Redistribute any remaining residual proportionally to m
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
