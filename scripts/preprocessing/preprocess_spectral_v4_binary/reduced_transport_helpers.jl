struct ReducedSpectralThreadCache
    P_buf        :: Matrix{Float64}
    fft_buffers  :: Dict{Int, Vector{ComplexF64}}
    real_buffers :: Dict{Int, Vector{Float64}}
    u_spec       :: Matrix{ComplexF64}
    v_spec       :: Matrix{ComplexF64}
end

struct ReducedTransformWorkspace
    sp             :: Vector{Float64}
    lnsp           :: Vector{Float64}
    dp             :: Matrix{Float64}
    m_arr          :: Matrix{Float64}
    hflux_arr      :: Matrix{Float64}
    cm_arr         :: Matrix{Float64}
    cell_areas     :: Vector{Float64}
    face_left      :: Vector{Int32}
    face_right     :: Vector{Int32}
    face_degree    :: Vector{Int}
    div_scratch    :: Matrix{Float64}
    # Scratch buffers for the Poisson-balance CG solver (cell-indexed).
    balance_psi    :: Vector{Float64}
    balance_rhs    :: Vector{Float64}
    balance_r      :: Vector{Float64}
    balance_p      :: Vector{Float64}
    balance_Ap     :: Vector{Float64}
    balance_z      :: Vector{Float64}
    caches         :: Vector{ReducedSpectralThreadCache}
end

struct ReducedMergeWorkspace{FT}
    m_native_ft     :: Matrix{FT}
    hflux_native_ft :: Matrix{FT}
    m_merged        :: Matrix{FT}
    hflux_merged    :: Matrix{FT}
    cm_merged       :: Matrix{FT}
    div_scratch     :: Matrix{Float64}
end

struct ReducedWindowStorage{FT}
    all_m     :: Vector{Matrix{FT}}
    all_hflux :: Vector{Matrix{FT}}
    all_cm    :: Vector{Matrix{FT}}
    all_ps    :: Vector{Vector{FT}}
end

function allocate_reduced_transform_workspace(grid::ReducedGaussianTargetGeometry,
                                              T::Int,
                                              Nz_native::Int)
    mesh = grid.mesh
    nc = AtmosTransportV2.ncells(mesh)
    nf = AtmosTransportV2.nfaces(mesh)
    nt = Threads.nthreads()
    nt_max = max(nt, 2 * nt) + 4

    cell_areas = [AtmosTransportV2.cell_area(mesh, c) for c in 1:nc]
    buffer_lengths = sort!(unique(vcat(collect(mesh.nlon_per_ring), collect(mesh.boundary_counts))))
    face_left = Vector{Int32}(undef, nf)
    face_right = Vector{Int32}(undef, nf)
    for f in 1:nf
        left, right = AtmosTransportV2.face_cells(mesh, f)
        face_left[f] = Int32(left)
        face_right[f] = Int32(right)
    end

    caches = [ReducedSpectralThreadCache(
                 zeros(Float64, T + 1, T + 1),
                 Dict(n => zeros(ComplexF64, n) for n in buffer_lengths),
                 Dict(n => zeros(Float64, n) for n in buffer_lengths),
                 zeros(ComplexF64, T + 1, T + 1),
                 zeros(ComplexF64, T + 1, T + 1),
             ) for _ in 1:nt_max]

    face_degree = cell_face_degree(face_left, face_right, nc)

    return ReducedTransformWorkspace(
        zeros(Float64, nc),
        zeros(Float64, nc),
        zeros(Float64, nc, Nz_native),
        zeros(Float64, nc, Nz_native),
        zeros(Float64, nf, Nz_native),
        zeros(Float64, nc, Nz_native + 1),
        cell_areas,
        face_left,
        face_right,
        face_degree,
        zeros(Float64, nc, Nz_native),
        zeros(Float64, nc),   # balance_psi
        zeros(Float64, nc),   # balance_rhs
        zeros(Float64, nc),   # balance_r
        zeros(Float64, nc),   # balance_p
        zeros(Float64, nc),   # balance_Ap
        zeros(Float64, nc),   # balance_z (preconditioned residual)
        caches,
    )
end

function allocate_reduced_merge_workspace(grid::ReducedGaussianTargetGeometry,
                                          Nz_native::Int,
                                          Nz::Int,
                                          ::Type{FT}) where FT
    mesh = grid.mesh
    nc = AtmosTransportV2.ncells(mesh)
    nf = AtmosTransportV2.nfaces(mesh)
    return ReducedMergeWorkspace{FT}(
        zeros(FT, nc, Nz_native),
        zeros(FT, nf, Nz_native),
        zeros(FT, nc, Nz),
        zeros(FT, nf, Nz),
        zeros(FT, nc, Nz + 1),
        zeros(Float64, nc, Nz),
    )
end

function allocate_reduced_window_storage(Nt::Int, ::Type{FT}, ::ReducedGaussianTargetGeometry, ::Int) where FT
    return ReducedWindowStorage{FT}(
        Vector{Matrix{FT}}(undef, Nt),
        Vector{Matrix{FT}}(undef, Nt),
        Vector{Matrix{FT}}(undef, Nt),
        Vector{Vector{FT}}(undef, Nt),
    )
end

@inline function _fft_buffer!(cache::ReducedSpectralThreadCache, n::Int)
    return cache.fft_buffers[n]
end

@inline function _real_buffer!(cache::ReducedSpectralThreadCache, n::Int)
    return cache.real_buffers[n]
end

function spectral_to_ring!(dest::AbstractVector{Float64},
                           spec::AbstractMatrix{ComplexF64},
                           T::Int,
                           lat_deg::Float64,
                           cache::ReducedSpectralThreadCache;
                           lon_shift_rad::Float64 = 0.0)
    Nlon = length(dest)
    compute_legendre_column!(cache.P_buf, T, sind(lat_deg))
    fft_buf = _fft_buffer!(cache, Nlon)
    fill!(fft_buf, zero(ComplexF64))

    for m in 0:min(T, div(Nlon, 2))
        Gm = zero(ComplexF64)
        @inbounds for n in m:T
            Gm += spec[n + 1, m + 1] * cache.P_buf[n + 1, m + 1]
        end
        if lon_shift_rad != 0.0 && m > 0
            Gm *= exp(im * m * lon_shift_rad)
        end
        fft_buf[m + 1] = Gm
    end

    for m in 1:min(T, div(Nlon, 2) - 1)
        fft_buf[Nlon - m + 1] = conj(fft_buf[m + 1])
    end

    FFTW.bfft!(fft_buf)
    @inbounds for i in 1:Nlon
        dest[i] = real(fft_buf[i])
    end
    return dest
end

function spectral_to_reduced_scalar!(field::Vector{Float64},
                                     spec::AbstractMatrix{ComplexF64},
                                     T::Int,
                                     grid::ReducedGaussianTargetGeometry,
                                     cache::ReducedSpectralThreadCache;
                                     centered::Bool = true)
    mesh = grid.mesh
    @inbounds for j in 1:AtmosTransportV2.nrings(mesh)
        start = mesh.ring_offsets[j]
        stop = mesh.ring_offsets[j + 1] - 1
        shift = centered ? (pi / mesh.nlon_per_ring[j]) : 0.0
        spectral_to_ring!(@view(field[start:stop]), spec, T, grid.lats[j], cache; lon_shift_rad=shift)
    end
    return field
end

function spectral_to_reduced_boundary!(dest::AbstractVector{Float64},
                                       spec::AbstractMatrix{ComplexF64},
                                       T::Int,
                                       lat_deg::Float64,
                                       cache::ReducedSpectralThreadCache)
    spectral_to_ring!(dest, spec, T, lat_deg, cache; lon_shift_rad=pi / length(dest))
    return dest
end

function compute_reduced_dp_and_mass!(dp::Matrix{Float64},
                                      m_arr::Matrix{Float64},
                                      sp::Vector{Float64},
                                      cell_areas::Vector{Float64},
                                      dA,
                                      dB)
    nc = length(sp)
    Nz = length(dA)
    inv_g = 1.0 / GRAV
    @inbounds for k in 1:Nz, c in 1:nc
        dp_face = abs(dA[k] + dB[k] * sp[c])
        dp[c, k] = dp_face
        m_arr[c, k] = dp_face * cell_areas[c] * inv_g
    end
    return nothing
end

function compute_reduced_horizontal_fluxes!(hflux::AbstractVector{Float64},
                                            lnsp_center::Vector{Float64},
                                            u_spec::AbstractMatrix{ComplexF64},
                                            v_spec::AbstractMatrix{ComplexF64},
                                            T::Int,
                                            dA_k::Float64,
                                            dB_k::Float64,
                                            grid::ReducedGaussianTargetGeometry,
                                            half_dt::Float64,
                                            cache::ReducedSpectralThreadCache)
    mesh = grid.mesh
    R_g = mesh.radius / GRAV
    fill!(hflux, 0.0)

    @inbounds for j in 1:AtmosTransportV2.nrings(mesh)
        nlon = mesh.nlon_per_ring[j]
        ring_vals = _real_buffer!(cache, nlon)
        spectral_to_ring!(ring_vals, u_spec, T, grid.lats[j], cache; lon_shift_rad=0.0)
        ring_start = mesh.ring_offsets[j]
        cos_lat = cosd(grid.lats[j])
        dlat = deg2rad(mesh.lat_faces[j + 1] - mesh.lat_faces[j])
        for i in 1:nlon
            face = ring_start + i - 1
            left = ring_start + (i == 1 ? nlon - 1 : i - 2)
            right = ring_start + i - 1
            ps_face = exp((lnsp_center[left] + lnsp_center[right]) / 2)
            dp_face = abs(dA_k + dB_k * ps_face)
            hflux[face] = ring_vals[i] / cos_lat * dp_face * R_g * dlat * half_dt
        end
    end

    @inbounds for b in 2:AtmosTransportV2.nrings(mesh)
        nseg = mesh.boundary_counts[b]
        seg_vals = _real_buffer!(cache, nseg)
        spectral_to_reduced_boundary!(seg_vals, v_spec, T, mesh.lat_faces[b], cache)
        south_ring = b - 1
        north_ring = b
        nlon_s = mesh.nlon_per_ring[south_ring]
        nlon_n = mesh.nlon_per_ring[north_ring]
        dlon = 2pi / nseg
        face0 = mesh._ncells + mesh.boundary_offsets[b] - 1
        for seg in 1:nseg
            face = face0 + seg
            south_i = ((seg - 1) * nlon_s) ÷ nseg + 1
            north_i = ((seg - 1) * nlon_n) ÷ nseg + 1
            south = mesh.ring_offsets[south_ring] + south_i - 1
            north = mesh.ring_offsets[north_ring] + north_i - 1
            ps_face = exp((lnsp_center[south] + lnsp_center[north]) / 2)
            dp_face = abs(dA_k + dB_k * ps_face)
            hflux[face] = seg_vals[seg] * dp_face * R_g * dlon * half_dt
        end
    end

    return nothing
end

"""
    cell_face_degree(face_left, face_right, nc)

Return the per-cell **interior** face degree for the graph Laplacian:
the number of horizontal faces adjacent to each cell that connect to
*another cell* (i.e., both `face_left > 0` and `face_right > 0`).
Boundary-stub faces (where one side is 0 — the pole caps on a
reduced-Gaussian mesh) are NOT counted, because the flux correction
step only modifies interior-face fluxes. Counting the stubs here would
make the Laplacian's diagonal inconsistent with the discrete correction
operator and leave a `n_stubs · psi[c]` residual at pole cells.
"""
function cell_face_degree(face_left::Vector{Int32},
                          face_right::Vector{Int32},
                          nc::Int)
    degree = zeros(Int, nc)
    @inbounds for f in eachindex(face_left)
        left = Int(face_left[f])
        right = Int(face_right[f])
        if left > 0 && right > 0
            degree[left]  += 1
            degree[right] += 1
        end
    end
    return degree
end

"""
    _graph_laplacian_mul!(out, psi, face_left, face_right, degree)

Compute `out = L · psi` where `L = D - A` is the graph Laplacian on the
face-indexed mesh. `D[c,c] = degree[c]` (number of horizontal faces
incident to cell c); `A[c,d] = 1` for each shared face.
"""
function _graph_laplacian_mul!(out::AbstractVector{Float64},
                               psi::AbstractVector{Float64},
                               face_left::Vector{Int32},
                               face_right::Vector{Int32},
                               degree::Vector{Int})
    @inbounds for c in eachindex(out)
        out[c] = degree[c] * psi[c]
    end
    @inbounds for f in eachindex(face_left)
        left = Int(face_left[f])
        right = Int(face_right[f])
        if left > 0 && right > 0
            out[left]  -= psi[right]
            out[right] -= psi[left]
        end
    end
    return out
end

"""
    _project_mean_zero!(v)

Subtract the mean of `v` in-place. Used to keep Conjugate-Gradient
iterates in the range of the singular graph Laplacian (which has a
1-dimensional constant null space on the reduced-Gaussian mesh with
interior-only degree).
"""
@inline function _project_mean_zero!(v::AbstractVector{Float64})
    s = sum(v) / length(v)
    @. v -= s
    return v
end

"""
    solve_graph_poisson_pcg!(psi, rhs, face_left, face_right, degree, scratch;
                             tol=1e-14, max_iter=20000)

Solve the singular system `L · psi = rhs` via Jacobi-Preconditioned
Conjugate Gradient, where `L = D - A` is the graph Laplacian on the
face mesh using **interior-only** face degrees (boundary-stub faces at
the pole caps are NOT counted — matching the discrete correction
operator that only modifies interior fluxes). In this form L has a
1-dimensional constant null space, so:

* `rhs` must be in range(L) — we enforce this by subtracting its mean
  (solvability condition for a singular system).
* We project `rhs` and the initial `psi = 0` to mean-zero at the start.
* The CG iterates stay in range(L) up to rounding error.
* We project the final `psi` to mean-zero so it is the unique
  minimum-norm solution (for reproducibility).

The Jacobi preconditioner `M = diag(L) = diag(degree)` accelerates
convergence on this Laplacian.

`scratch` must expose preallocated buffers `r`, `p`, `Ap`, and `z`,
each of length `length(psi)`.

Returns `(residual_linfty, iterations)` — the final L∞ residual
`max|L·psi - rhs|` and the number of iterations performed.
"""
function solve_graph_poisson_pcg!(psi::AbstractVector{Float64},
                                  rhs::AbstractVector{Float64},
                                  face_left::Vector{Int32},
                                  face_right::Vector{Int32},
                                  degree::Vector{Int},
                                  scratch;
                                  tol::Float64=1e-14,
                                  max_iter::Int=20000)
    r  = scratch.r
    p  = scratch.p
    Ap = scratch.Ap
    z  = scratch.z

    # Enforce solvability: rhs must be in range(L) for the singular L.
    _project_mean_zero!(rhs)

    fill!(psi, 0.0)
    copyto!(r, rhs)

    # Jacobi preconditioner z = M^{-1} r, with M = diag(degree).
    @inbounds for c in eachindex(r)
        z[c] = degree[c] > 0 ? r[c] / degree[c] : r[c]
    end
    copyto!(p, z)

    rz_old = dot(r, z)
    rhs_linf = maximum(abs, rhs) + eps()

    iter = 0
    r_linf = rhs_linf
    while iter < max_iter
        _graph_laplacian_mul!(Ap, p, face_left, face_right, degree)
        pAp = dot(p, Ap)
        pAp <= 0 && break
        alpha = rz_old / pAp
        @. psi += alpha * p
        @. r   -= alpha * Ap

        # Keep r in range(L) = mean-zero subspace (correct for singular L,
        # harmless for non-singular L). Without this, roundoff drift of
        # r out of range(L) stalls convergence.
        _project_mean_zero!(r)

        r_linf = maximum(abs, r)
        r_linf / rhs_linf < tol && break

        @inbounds for c in eachindex(r)
            z[c] = degree[c] > 0 ? r[c] / degree[c] : r[c]
        end
        # Same projection for z so p stays in range(L).
        _project_mean_zero!(z)
        rz_new = dot(r, z)
        rz_old == 0.0 && break
        beta = rz_new / rz_old
        @. p = z + beta * p
        rz_old = rz_new
        iter += 1
    end

    # Return the minimum-norm solution: project out the null-space (constant)
    # component of psi. The flux correction `psi[right] - psi[left]` is
    # invariant under constant shifts, so this is cosmetic but keeps
    # `psi` bounded and makes results reproducible.
    _project_mean_zero!(psi)
    return r_linf, iter
end

"""
    balance_reduced_horizontal_fluxes!(hflux, m_cur, m_next, face_left, face_right,
                                       degree, steps_per_window, scratch;
                                       tol, max_iter)

TM5-style Poisson mass-flux balance adapted to the reduced-Gaussian face
mesh. Corrects `hflux[f, k]` so that the horizontal convergence at each
cell matches the prescribed forward-window mass tendency per substep:

    dm_dt[c, k] = (m_next[c, k] - m_cur[c, k]) / (2 * steps_per_window)

Uses a graph-Laplacian Conjugate-Gradient solve per level (constant
null-space projected out). After the correction, the recomputed cm from
`recompute_faceindexed_cm_from_divergence!` will be near machine zero,
matching the LL-path balanced behaviour. Returns a short diagnostic
`(; max_pre_residual, max_post_residual, max_cg_iter)`.
"""
function balance_reduced_horizontal_fluxes!(hflux::AbstractMatrix{Float64},
                                            m_cur::AbstractMatrix{Float64},
                                            m_next::AbstractMatrix{Float64},
                                            face_left::Vector{Int32},
                                            face_right::Vector{Int32},
                                            degree::Vector{Int},
                                            steps_per_window::Int,
                                            scratch;
                                            tol::Float64=1e-14,
                                            max_iter::Int=20000)
    nc, Nz = size(m_cur)
    size(hflux, 2) == Nz || error("hflux Nz mismatch with m")
    inv_scale = 1.0 / (2.0 * steps_per_window)

    max_pre = 0.0
    max_post = 0.0
    max_it = 0

    psi = scratch.psi
    rhs = scratch.rhs
    div = scratch.r  # reuse CG residual buffer for divergence

    for k in 1:Nz
        # 1. Compute current horizontal divergence at level k.
        fill!(div, 0.0)
        @inbounds for f in eachindex(face_left)
            flux = hflux[f, k]
            left  = Int(face_left[f])
            right = Int(face_right[f])
            left > 0 && (div[left]  += flux)
            right > 0 && (div[right] -= flux)
        end
        # 2. Target divergence = forward-window mass tendency per substep.
        @inbounds for c in 1:nc
            rhs[c] = div[c] - (m_next[c, k] - m_cur[c, k]) * inv_scale
        end
        pre_res = maximum(abs, rhs)
        pre_res > max_pre && (max_pre = pre_res)
        if pre_res < tol
            # Already balanced at this level.
            continue
        end
        # 3. Solve L · psi = rhs (psi is per-cell scalar correction potential).
        _, it = solve_graph_poisson_pcg!(psi, rhs, face_left, face_right, degree,
                                         scratch; tol=tol, max_iter=max_iter)
        it > max_it && (max_it = it)
        # 4. Apply flux correction: hflux[f] += psi[right] - psi[left].
        @inbounds for f in eachindex(face_left)
            left  = Int(face_left[f])
            right = Int(face_right[f])
            if left > 0 && right > 0
                hflux[f, k] += psi[right] - psi[left]
            end
        end
        # 5. Verify post-balance residual.
        fill!(div, 0.0)
        @inbounds for f in eachindex(face_left)
            flux = hflux[f, k]
            left  = Int(face_left[f])
            right = Int(face_right[f])
            left > 0 && (div[left]  += flux)
            right > 0 && (div[right] -= flux)
        end
        post_res = 0.0
        @inbounds for c in 1:nc
            r = abs(div[c] - (m_next[c, k] - m_cur[c, k]) * inv_scale)
            r > post_res && (post_res = r)
        end
        post_res > max_post && (max_post = post_res)
    end

    return (; max_pre_residual=max_pre, max_post_residual=max_post, max_cg_iter=max_it)
end

function recompute_faceindexed_cm_from_divergence!(cm::AbstractMatrix{FT},
                                                   hflux::AbstractMatrix{FT},
                                                   face_left::Vector{Int32},
                                                   face_right::Vector{Int32},
                                                   div_scratch::AbstractMatrix{Float64};
                                                   B_ifc::Vector{<:Real}=Float64[]) where FT
    nc = size(cm, 1)
    Nz = size(cm, 2) - 1
    fill!(cm, zero(FT))
    fill!(div_scratch, 0.0)

    @inbounds for k in 1:Nz, f in eachindex(face_left)
        flux = Float64(hflux[f, k])
        left = Int(face_left[f])
        right = Int(face_right[f])
        left > 0 && (div_scratch[left, k] += flux)
        right > 0 && (div_scratch[right, k] -= flux)
    end

    if !isempty(B_ifc) && length(B_ifc) == Nz + 1
        @inbounds for c in 1:nc
            pit = 0.0
            for k in 1:Nz
                pit += div_scratch[c, k]
            end
            acc = 0.0
            for k in 1:Nz
                acc = acc - div_scratch[c, k] + (Float64(B_ifc[k + 1]) - Float64(B_ifc[k])) * pit
                cm[c, k + 1] = FT(acc)
            end
        end
    else
        @inbounds for c in 1:nc
            acc = 0.0
            for k in 1:Nz
                acc = acc - div_scratch[c, k]
                cm[c, k + 1] = FT(acc)
            end
        end
    end
    return nothing
end

"""
    apply_reduced_poisson_balance!(storage, work, vertical, steps_per_window)

Post-process all stored windows by applying TM5-style Poisson balance to
the merged horizontal flux, then recomputing the merged vertical flux
`cm` from continuity on the balanced hflux. For the final window of a
day (where `m[win+1]` is unavailable) we target zero mass tendency,
matching the LL fallback in `fill_window_mass_tendency!`.

After this pass, `storage.all_cm[win]` should be at machine-zero vs
`storage.all_m[win]` (matching the LL-path behaviour).
"""
function apply_reduced_poisson_balance!(storage::ReducedWindowStorage{FT},
                                        work::ReducedTransformWorkspace,
                                        vertical,
                                        steps_per_window::Int;
                                        tol::Float64=1e-14,
                                        max_iter::Int=20000) where FT
    Nt = length(storage.all_m)
    Nt == 0 && return nothing
    nc, Nz = size(storage.all_m[1])

    div_scratch = zeros(Float64, nc, Nz)
    hflux_work = zeros(Float64, size(storage.all_hflux[1]))
    m_cur_work = zeros(Float64, nc, Nz)
    m_next_work = zeros(Float64, nc, Nz)
    scratch = (psi = work.balance_psi, rhs = work.balance_rhs,
               r = work.balance_r, p = work.balance_p, Ap = work.balance_Ap,
               z = work.balance_z)

    @info "  Applying Poisson mass flux balance (reduced-Gaussian)..."
    worst_pre = 0.0
    worst_post = 0.0
    worst_iter = 0

    for win in 1:Nt
        copyto!(hflux_work, storage.all_hflux[win])
        copyto!(m_cur_work, storage.all_m[win])
        if win < Nt
            copyto!(m_next_work, storage.all_m[win + 1])
        else
            copyto!(m_next_work, m_cur_work)  # zero tendency fallback
        end

        diag = balance_reduced_horizontal_fluxes!(
            hflux_work, m_cur_work, m_next_work,
            work.face_left, work.face_right, work.face_degree,
            steps_per_window, scratch;
            tol=tol, max_iter=max_iter,
        )
        worst_pre = max(worst_pre, diag.max_pre_residual)
        worst_post = max(worst_post, diag.max_post_residual)
        worst_iter = max(worst_iter, diag.max_cg_iter)

        # Store balanced hflux back as FT.
        storage.all_hflux[win] = FT.(hflux_work)

        # Recompute cm from balanced hflux.
        cm_buf = zeros(Float64, nc, Nz + 1)
        recompute_faceindexed_cm_from_divergence!(
            cm_buf, hflux_work, work.face_left, work.face_right, div_scratch;
            B_ifc=vertical.merged_vc.B,
        )
        storage.all_cm[win] = FT.(cm_buf)
    end

    @info "  Poisson balance complete for $Nt windows. " *
          "Residual: pre=$(round(worst_pre, sigdigits=4)) kg, " *
          "post=$(round(worst_post, sigdigits=4)) kg, " *
          "max_cg_iter=$worst_iter"
    return nothing
end

function spectral_to_native_fields!(work::ReducedTransformWorkspace,
                                    lnsp_spec::Matrix{ComplexF64},
                                    vo_hour::Array{ComplexF64, 3},
                                    d_hour::Array{ComplexF64, 3},
                                    T::Int,
                                    level_range::UnitRange{Int},
                                    ab,
                                    grid::ReducedGaussianTargetGeometry,
                                    half_dt::Float64)
    cache1 = work.caches[1]
    spectral_to_reduced_scalar!(work.lnsp, lnsp_spec, T, grid, cache1; centered=true)
    @. work.sp = exp(work.lnsp)
    compute_reduced_dp_and_mass!(work.dp, work.m_arr, work.sp, work.cell_areas, ab.dA, ab.dB)

    Nz = length(level_range)
    Threads.@threads :static for kk in 1:Nz
        level = level_range[kk]
        cache = work.caches[Threads.threadid()]
        vod2uv!(cache.u_spec, cache.v_spec,
                @view(vo_hour[:, :, level]),
                @view(d_hour[:, :, level]),
                T)
        compute_reduced_horizontal_fluxes!(@view(work.hflux_arr[:, kk]),
                                           work.lnsp,
                                           cache.u_spec,
                                           cache.v_spec,
                                           T,
                                           Float64(ab.dA[level]),
                                           Float64(ab.dB[level]),
                                           grid,
                                           half_dt,
                                           cache)
    end

    recompute_faceindexed_cm_from_divergence!(work.cm_arr, work.hflux_arr,
                                              work.face_left, work.face_right,
                                              work.div_scratch; B_ifc=ab.b_ifc)
    return nothing
end

function merge_field_2d!(merged::AbstractMatrix{FT}, native::AbstractMatrix{FT}, mm::Vector{Int}) where FT
    fill!(merged, zero(FT))
    @inbounds for k in 1:length(mm)
        @views merged[:, mm[k]] .+= native[:, k]
    end
    return nothing
end

function merge_reduced_window!(merged::ReducedMergeWorkspace{FT},
                               native::ReducedTransformWorkspace,
                               vertical) where FT
    @. merged.m_native_ft = FT(native.m_arr)
    @. merged.hflux_native_ft = FT(native.hflux_arr)
    merge_field_2d!(merged.m_merged, merged.m_native_ft, vertical.merge_map)
    merge_field_2d!(merged.hflux_merged, merged.hflux_native_ft, vertical.merge_map)
    recompute_faceindexed_cm_from_divergence!(merged.cm_merged,
                                              merged.hflux_merged,
                                              native.face_left,
                                              native.face_right,
                                              merged.div_scratch;
                                              B_ifc=vertical.merged_vc.B)
    return nothing
end

function store_reduced_window!(storage::ReducedWindowStorage{FT},
                               merged::ReducedMergeWorkspace{FT},
                               ps::AbstractVector,
                               win_idx::Int) where FT
    storage.all_m[win_idx] = copy(merged.m_merged)
    storage.all_hflux[win_idx] = copy(merged.hflux_merged)
    storage.all_cm[win_idx] = copy(merged.cm_merged)
    storage.all_ps[win_idx] = FT.(ps)
    return nothing
end
