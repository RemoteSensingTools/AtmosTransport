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
    nc = ncells(mesh)
    nf = nfaces(mesh)
    nt = Threads.nthreads()
    nt_max = max(nt, 2 * nt) + 4

    cell_areas = [cell_area(mesh, c) for c in 1:nc]
    buffer_lengths = sort!(unique(vcat(collect(mesh.nlon_per_ring), collect(mesh.boundary_counts))))
    face_left = Vector{Int32}(undef, nf)
    face_right = Vector{Int32}(undef, nf)
    for f in 1:nf
        left, right = face_cells(mesh, f)
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
    nc = ncells(mesh)
    nf = nfaces(mesh)
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
    @inbounds for j in 1:nrings(mesh)
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

    @inbounds for j in 1:nrings(mesh)
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

    @inbounds for b in 2:nrings(mesh)
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

Uses a graph-Laplacian Conjugate-Gradient solve per level. The graph
Laplacian with interior-only face degrees is singular (1-D constant
null space), so the solver projects `rhs` onto range(L) before
solving. Any part of `rhs` that lives in the null space (uniform
per-cell offset, from pole-cap boundary stubs or from a global
mass-tendency mismatch that isn't closed by the raw spectral fluxes)
**cannot** be corrected by interior-face flux adjustments and is
absorbed downstream by the continuity `cm` cumsum. The physics
truth is the post-balance `max(|cm|/m)` probe, not the raw
divergence residuals reported here.

Returns a diagnostic NamedTuple:
  `max_pre_raw_residual`    — `max(|rhs|)` BEFORE mean projection
                              (target - div(hflux) in raw units)
  `max_rhs_mean`            — `max(|mean(rhs)|)` across levels
                              (the null-space component magnitude)
  `max_pre_projected`       — `max(|rhs - mean(rhs)|)`, the part of
                              the residual in range(L)
  `max_post_projected`      — `max(|L·psi - (rhs - mean(rhs))|)`, the
                              actual PCG residual on range(L); this
                              is the number you want near zero
  `max_post_raw_residual`   — `max(|div(new hflux) - target|)`; will
                              include the null-space component and
                              so stays ~`max_rhs_mean` even when the
                              PCG solve is exact. Kept as a
                              backwards-compat proxy for the old
                              `max_post_residual` but do NOT use it
                              as a solver-convergence metric.
  `max_cg_iter`             — worst CG iteration count across levels

Also returned for backwards-compat:
  `max_pre_residual = max_pre_raw_residual`,
  `max_post_residual = max_post_raw_residual`.
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

    max_pre_raw = 0.0           # max(|rhs|) before mean projection
    max_rhs_mean = 0.0          # max(|mean(rhs)|) — null-space magnitude
    max_pre_proj = 0.0          # max(|rhs - mean(rhs)|)
    max_post_proj = 0.0         # max(|L*psi - (rhs - mean(rhs))|) on range(L)
    max_post_raw = 0.0          # max(|div(new hflux) - target|)
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

        # 2a. Raw (pre-projection) diagnostics.
        rhs_sum = 0.0
        rhs_raw_linf = 0.0
        @inbounds for c in 1:nc
            rhs_sum += rhs[c]
            a = abs(rhs[c])
            a > rhs_raw_linf && (rhs_raw_linf = a)
        end
        rhs_mean = rhs_sum / nc
        rhs_raw_linf > max_pre_raw && (max_pre_raw = rhs_raw_linf)
        a_mean = abs(rhs_mean)
        a_mean > max_rhs_mean && (max_rhs_mean = a_mean)

        # 2b. Projected `rhs` L∞ (magnitude of the part of rhs
        # that the solver can actually kill).
        pre_proj = 0.0
        @inbounds for c in 1:nc
            a = abs(rhs[c] - rhs_mean)
            a > pre_proj && (pre_proj = a)
        end
        pre_proj > max_pre_proj && (max_pre_proj = pre_proj)

        if pre_proj < tol
            # Projected residual already at tol: nothing left for the
            # solver to do on this level. The raw residual may still be
            # nonzero if rhs_mean is large, but that's null-space and
            # not fixable here.
            continue
        end

        # 3. Solve L · psi = rhs (psi is per-cell scalar correction potential).
        #    solve_graph_poisson_pcg! projects `rhs` to mean-zero in-place;
        #    `rhs` exits the call as the PCG residual on range(L).
        _, it = solve_graph_poisson_pcg!(psi, rhs, face_left, face_right, degree,
                                         scratch; tol=tol, max_iter=max_iter)
        it > max_it && (max_it = it)

        # 3a. Post-solve residual on range(L): max(|L*psi - rhs_projected|).
        # Compute L*psi explicitly to avoid trusting the CG internal metric.
        Lpsi = scratch.Ap  # reuse buffer
        _graph_laplacian_mul!(Lpsi, psi, face_left, face_right, degree)
        # `rhs` has been overwritten by the solver to (rhs_projected - alpha * L*p).
        # What we want is `max(|L*psi_final - rhs_projected|)`. Recompute
        # rhs_projected from the current hflux and the known mean:
        fill!(div, 0.0)
        @inbounds for f in eachindex(face_left)
            flux = hflux[f, k]
            left  = Int(face_left[f])
            right = Int(face_right[f])
            left > 0 && (div[left]  += flux)
            right > 0 && (div[right] -= flux)
        end
        @inbounds for c in 1:nc
            rhs[c] = (div[c] - (m_next[c, k] - m_cur[c, k]) * inv_scale) - rhs_mean
        end
        post_proj = 0.0
        @inbounds for c in 1:nc
            a = abs(Lpsi[c] - rhs[c])
            a > post_proj && (post_proj = a)
        end
        post_proj > max_post_proj && (max_post_proj = post_proj)

        # 4. Apply flux correction: hflux[f] += psi[right] - psi[left].
        @inbounds for f in eachindex(face_left)
            left  = Int(face_left[f])
            right = Int(face_right[f])
            if left > 0 && right > 0
                hflux[f, k] += psi[right] - psi[left]
            end
        end

        # 5. Raw post-balance residual (includes the irremovable
        # null-space component).
        fill!(div, 0.0)
        @inbounds for f in eachindex(face_left)
            flux = hflux[f, k]
            left  = Int(face_left[f])
            right = Int(face_right[f])
            left > 0 && (div[left]  += flux)
            right > 0 && (div[right] -= flux)
        end
        post_raw = 0.0
        @inbounds for c in 1:nc
            r = abs(div[c] - (m_next[c, k] - m_cur[c, k]) * inv_scale)
            r > post_raw && (post_raw = r)
        end
        post_raw > max_post_raw && (max_post_raw = post_raw)
    end

    return (;
        max_pre_raw_residual = max_pre_raw,
        max_rhs_mean = max_rhs_mean,
        max_pre_projected = max_pre_proj,
        max_post_projected = max_post_proj,
        max_post_raw_residual = max_post_raw,
        max_cg_iter = max_it,
        # Backwards-compat aliases (kept for older call sites):
        max_pre_residual = max_pre_raw,
        max_post_residual = max_post_raw,
    )
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
    worst_pre_raw = 0.0
    worst_rhs_mean = 0.0
    worst_pre_proj = 0.0
    worst_post_proj = 0.0
    worst_post_raw = 0.0
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
        worst_pre_raw  = max(worst_pre_raw,  diag.max_pre_raw_residual)
        worst_rhs_mean = max(worst_rhs_mean, diag.max_rhs_mean)
        worst_pre_proj = max(worst_pre_proj, diag.max_pre_projected)
        worst_post_proj = max(worst_post_proj, diag.max_post_projected)
        worst_post_raw = max(worst_post_raw, diag.max_post_raw_residual)
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

    @info @sprintf("  Poisson balance complete for %d windows.", Nt)
    @info @sprintf("    pre_raw=%.3e  pre_proj=%.3e  post_proj=%.3e  post_raw=%.3e kg",
                   worst_pre_raw, worst_pre_proj, worst_post_proj, worst_post_raw)
    @info @sprintf("    null-space component max|mean(rhs)|=%.3e kg  (not correctable by interior-face fluxes,",
                   worst_rhs_mean)
    @info "                                                      absorbed by downstream cm cumsum)"
    @info @sprintf("    max_cg_iter=%d. Physics truth is max(|cm|/m) after cm recomputation,", worst_iter)
    @info "                not the raw residuals above (see TransportBinaryDriver sanity guard)."
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

"""
    SlidingWindowBuffer{FT}

Two-slot circular buffer for streaming RG preprocessing.  Only two windows'
worth of `(m, hflux, cm, ps)` are kept in memory at any time, enabling
O160/O320 binary generation without OOM.
"""
struct SlidingWindowBuffer{FT}
    m     :: Vector{Matrix{FT}}     # length-2 circular buffer
    hflux :: Vector{Matrix{FT}}
    cm    :: Vector{Matrix{FT}}
    ps    :: Vector{Vector{FT}}
end

function allocate_sliding_window_buffer(nc::Int, nf::Int, Nz::Int, ::Type{FT}) where FT
    SlidingWindowBuffer{FT}(
        [zeros(FT, nc, Nz)     for _ in 1:2],
        [zeros(FT, nf, Nz)     for _ in 1:2],
        [zeros(FT, nc, Nz + 1) for _ in 1:2],
        [zeros(FT, nc)         for _ in 1:2],
    )
end

"""
    fill_buffer_slot!(buf, slot, merged, ps_vec, FT)

Copy merged results into the given slot (1 or 2) of the sliding buffer.
"""
function fill_buffer_slot!(buf::SlidingWindowBuffer{FT},
                           slot::Int,
                           merged::ReducedMergeWorkspace{FT},
                           ps_vec::AbstractVector) where FT
    copyto!(buf.m[slot],     merged.m_merged)
    copyto!(buf.hflux[slot], merged.hflux_merged)
    copyto!(buf.cm[slot],    merged.cm_merged)
    buf.ps[slot] .= ps_vec   # broadcast handles Float64→FT conversion in-place
    return nothing
end

# =========================================================================
# RG process_window! and process_day — mirrors the LL path in binary_pipeline.jl
# =========================================================================

"""
    process_window!(win_idx, hour, spec, grid::ReducedGaussianTargetGeometry,
                    vertical, settings, work, merged, storage, ps_offsets)

Process one analysis window on a reduced-Gaussian target grid.
Spectral synthesis → mass fluxes → level merge → store.
"""
function process_window!(win_idx::Int,
                         hour::Int,
                         spec,
                         grid::ReducedGaussianTargetGeometry,
                         vertical,
                         settings,
                         work::ReducedTransformWorkspace,
                         merged::ReducedMergeWorkspace{FT},
                         storage::ReducedWindowStorage{FT},
                         ps_offsets::Vector{Float64}) where FT
    t0 = time()

    spectral_to_native_fields!(work,
        spec.lnsp_all[hour], spec.vo_by_hour[hour], spec.d_by_hour[hour],
        spec.T, vertical.level_range, vertical.ab, grid, settings.half_dt)

    # Mass fix: pin global mean total ps (same formula as LL path)
    if settings.mass_fix_enable
        target_ps = settings.target_ps_dry_pa / (1.0 - settings.qv_global_climatology)
        area_sum = sum(work.cell_areas)
        mean_ps = dot(work.sp, work.cell_areas) / area_sum
        offset = target_ps - mean_ps
        work.sp .+= offset
        ps_offsets[win_idx] = offset
        # Recompute mass with fixed ps
        compute_reduced_dp_and_mass!(work.dp, work.m_arr, work.sp, work.cell_areas,
                                     vertical.ab.dA, vertical.ab.dB)
    end

    merge_reduced_window!(merged, work, vertical)
    store_reduced_window!(storage, merged, work.sp, win_idx)

    elapsed = round(time() - t0, digits=2)
    should_log_window(win_idx, length(storage.all_m)) &&
        @info(@sprintf("    Window %d/%d (hour %02d): %.2fs  ps_offset=%+.3f Pa",
                       win_idx, length(storage.all_m), hour, elapsed, ps_offsets[win_idx]))

    return nothing
end

"""
    synthesize_and_merge_window!(work, merged, hour, spec, grid, vertical,
                                 settings, ps_offsets, win_idx)

Spectral synthesis → native fields → mass fix → level merge for one window.
Results are left in `merged.m_merged`, `merged.hflux_merged`, `merged.cm_merged`
and `work.sp` (surface pressure).  No allocation.
"""
function synthesize_and_merge_window!(work::ReducedTransformWorkspace,
                                      merged::ReducedMergeWorkspace{FT},
                                      hour::Int,
                                      spec,
                                      grid::ReducedGaussianTargetGeometry,
                                      vertical,
                                      settings,
                                      ps_offsets::Vector{Float64},
                                      win_idx::Int) where FT
    spectral_to_native_fields!(work,
        spec.lnsp_all[hour], spec.vo_by_hour[hour], spec.d_by_hour[hour],
        spec.T, vertical.level_range, vertical.ab, grid, settings.half_dt)

    if settings.mass_fix_enable
        target_ps = settings.target_ps_dry_pa / (1.0 - settings.qv_global_climatology)
        area_sum = sum(work.cell_areas)
        mean_ps = dot(work.sp, work.cell_areas) / area_sum
        offset = target_ps - mean_ps
        work.sp .+= offset
        ps_offsets[win_idx] = offset
        compute_reduced_dp_and_mass!(work.dp, work.m_arr, work.sp, work.cell_areas,
                                     vertical.ab.dA, vertical.ab.dB)
    end

    merge_reduced_window!(merged, work, vertical)
    return nothing
end

"""
    balance_window!(hflux_work, m_cur_work, m_next_work, cm_work, div_scratch,
                    buf, slot, m_next, work, vertical, steps_per_window;
                    tol, max_iter)

Poisson-balance the horizontal fluxes in buffer `slot` using
`buf.m[slot]` as current mass and `m_next` as next-window mass target,
then recompute `cm` from the balanced fluxes.  Mutates `buf.hflux[slot]`
and `buf.cm[slot]` in-place.

All scratch arrays (`hflux_work`, `m_cur_work`, `m_next_work`, `cm_work`,
`div_scratch`) are preallocated Float64 buffers from the caller — no
per-call allocation.

Returns a diagnostics NamedTuple (from `balance_reduced_horizontal_fluxes!`).
"""
function balance_window!(hflux_work::Matrix{Float64},
                         m_cur_work::Matrix{Float64},
                         m_next_work::Matrix{Float64},
                         cm_work::Matrix{Float64},
                         div_scratch::Matrix{Float64},
                         buf::SlidingWindowBuffer{FT},
                         slot::Int,
                         m_next::AbstractMatrix{FT},
                         work::ReducedTransformWorkspace,
                         vertical,
                         steps_per_window::Int;
                         tol::Float64 = 1e-14,
                         max_iter::Int = 20000) where FT

    copyto!(hflux_work, buf.hflux[slot])
    copyto!(m_cur_work, buf.m[slot])
    copyto!(m_next_work, m_next)

    scratch = (psi = work.balance_psi, rhs = work.balance_rhs,
               r = work.balance_r, p = work.balance_p,
               Ap = work.balance_Ap, z = work.balance_z)

    diag = balance_reduced_horizontal_fluxes!(
        hflux_work, m_cur_work, m_next_work,
        work.face_left, work.face_right, work.face_degree,
        steps_per_window, scratch; tol=tol, max_iter=max_iter)

    # Store balanced hflux back as FT.
    buf.hflux[slot] .= FT.(hflux_work)

    # Recompute cm from balanced hflux (in Float64, then convert to FT).
    # Note: recompute_faceindexed_cm_from_divergence! fills cm_work and
    # div_scratch internally, so no fill! needed here.
    recompute_faceindexed_cm_from_divergence!(
        cm_work, hflux_work,
        work.face_left, work.face_right, div_scratch;
        B_ifc = vertical.merged_vc.B)
    buf.cm[slot] .= FT.(cm_work)

    return diag
end

"""
    process_day(date, grid::ReducedGaussianTargetGeometry, settings, vertical;
                next_day_hour0=nothing)

Streaming one-day preprocessing for reduced-Gaussian targets.

Uses a 2-window sliding buffer: at any time only two windows' worth of
`(m, hflux, cm, ps)` are held in memory.  Each window is Poisson-balanced
and written to disk before the next pair is computed.  This reduces peak
memory from `O(Nt)` to `O(1)` and enables O160/O320 binary generation.

Pipeline per window:
  spectral synthesis → mass fix → level merge → (wait for next window) →
  Poisson balance using (m_cur, m_next) → cm recomputation → stream-write
"""
function process_day(date::Date,
                     grid::ReducedGaussianTargetGeometry,
                     settings,
                     vertical;
                     next_day_hour0=nothing)
    FT = settings.output_float_type
    mesh = grid.mesh
    nc = ncells(mesh)
    nf = nfaces(mesh)
    Nz_native = vertical.Nz_native
    Nz = vertical.Nz
    steps_per_met = exact_steps_per_window(settings.met_interval, settings.dt)
    date_str = Dates.format(date, "yyyymmdd")

    vo_d_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_vo_d.gb")
    lnsp_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_lnsp.gb")

    if !isfile(vo_d_path) || !isfile(lnsp_path)
        @warn "Missing GRIB files for $date_str, skipping"
        return nothing
    end

    t_day = time()
    @info "  Reading spectral data for $date_str..."
    spec = read_day_spectral_streaming(vo_d_path, lnsp_path; T_target=settings.T_target)
    @info @sprintf("  Spectral data read: T=%d, %d hours (%.1fs)",
                   spec.T, spec.n_times, time() - t_day)

    Nt = spec.n_times
    mkpath(settings.out_dir)
    bin_path = output_binary_path(date, settings.out_dir, settings.min_dp, FT)

    # Allocate workspaces — shared across all windows (no per-window allocation)
    work   = allocate_reduced_transform_workspace(grid, spec.T, Nz_native)
    merged = allocate_reduced_merge_workspace(grid, Nz_native, Nz, FT)
    buf    = allocate_sliding_window_buffer(nc, nf, Nz, FT)
    ps_offsets = zeros(Float64, Nt)

    # Poisson-balance scratch (Float64, reused every window — no per-call allocation)
    hflux_work    = zeros(Float64, nf, Nz)
    m_cur_work    = zeros(Float64, nc, Nz)
    m_next_work   = zeros(Float64, nc, Nz)
    cm_work       = zeros(Float64, nc, Nz + 1)
    div_scratch_b = zeros(Float64, nc, Nz)

    # Open the streaming binary writer
    vc_merged = vertical.merged_vc
    transport_grid = AtmosGrid(mesh, vc_merged, CPU(); FT=FT)
    sample_window = (m = buf.m[1], hflux = buf.hflux[1],
                     cm = buf.cm[1], ps = buf.ps[1])

    writer = open_streaming_transport_binary(
        bin_path, transport_grid, Nt, sample_window;
        FT = FT,
        dt_met_seconds     = settings.met_interval,
        half_dt_seconds    = settings.half_dt,
        steps_per_window   = steps_per_met,
        source_flux_sampling = :window_start_endpoint,
        mass_basis = Symbol(settings.mass_basis),
        extra_header = Dict{String, Any}(
            "preprocessor"     => "preprocess_transport_binary.jl",
            "source_type"      => "era5_spectral",
            "target_type"      => "reduced_gaussian",
            "gaussian_number"  => grid.gaussian_number,
            "poisson_balanced" => true,
            "mass_fix_enabled" => settings.mass_fix_enable,
        ))

    bytes_per_window = writer.elems_per_window * sizeof(eltype(writer.pack_buffer))
    expected_total = writer.header_bytes + Nt * bytes_per_window
    @info @sprintf("  Output: %s (%.2f GB, %d windows, %d cells, %d faces)",
                   basename(bin_path), expected_total / 1e9, Nt, nc, nf)

    log_mass_fix_configuration(settings)
    @info "  Streaming: synthesize → balance → write (2-window sliding buffer)..."

    # Worst-case balance diagnostics across all windows
    worst_pre_raw  = 0.0
    worst_post_proj = 0.0
    worst_post_raw = 0.0
    worst_iter     = 0

    # cur/nxt are indices into the 2-slot buffer (swap instead of copy)
    cur = 1
    nxt = 2

    # ── Process first window into slot `cur` ──
    t0 = time()
    synthesize_and_merge_window!(work, merged, spec.hours[1], spec, grid,
                                 vertical, settings, ps_offsets, 1)
    fill_buffer_slot!(buf, cur, merged, work.sp)
    should_log_window(1, Nt) &&
        @info @sprintf("    Window  1/%d (hour %02d): synth %.2fs  offset=%+.3f Pa",
                       Nt, spec.hours[1], time() - t0, ps_offsets[1])

    # ── Sliding-buffer loop: windows 2..Nt ──
    for win in 2:Nt
        t0 = time()
        synthesize_and_merge_window!(work, merged, spec.hours[win], spec, grid,
                                     vertical, settings, ps_offsets, win)
        fill_buffer_slot!(buf, nxt, merged, work.sp)
        t_synth = time() - t0

        # Balance the PREVIOUS window using (m_cur, m_next)
        t_bal = time()
        diag = balance_window!(hflux_work, m_cur_work, m_next_work,
                               cm_work, div_scratch_b,
                               buf, cur, buf.m[nxt],
                               work, vertical, steps_per_met)
        t_bal = time() - t_bal

        worst_pre_raw   = max(worst_pre_raw,   diag.max_pre_raw_residual)
        worst_post_proj = max(worst_post_proj,  diag.max_post_projected)
        worst_post_raw  = max(worst_post_raw,   diag.max_post_raw_residual)
        worst_iter      = max(worst_iter,       diag.max_cg_iter)

        # Write the balanced previous window
        window_nt = (m = buf.m[cur], hflux = buf.hflux[cur],
                     cm = buf.cm[cur], ps = buf.ps[cur])
        write_streaming_window!(writer, window_nt)

        written_win = win - 1  # balance diagnostics are for the PREVIOUS window
        should_log_window(written_win, Nt) &&
            @info @sprintf("    Window %2d/%d: wrote (bal %.2fs pre_raw=%.2e post_proj=%.2e iter=%d) | synth %2d (%.2fs) offset=%+.3f Pa",
                           written_win, Nt, t_bal, diag.max_pre_raw_residual, diag.max_post_projected, diag.max_cg_iter, win, t_synth, ps_offsets[win])

        # Swap slots
        cur, nxt = nxt, cur
    end

    # ── Balance & write the LAST window (zero-tendency fallback) ──
    t_bal = time()
    diag = balance_window!(hflux_work, m_cur_work, m_next_work,
                           cm_work, div_scratch_b,
                           buf, cur, buf.m[cur],   # m_next = m_cur → zero tendency
                           work, vertical, steps_per_met)
    t_bal = time() - t_bal

    worst_pre_raw   = max(worst_pre_raw,   diag.max_pre_raw_residual)
    worst_post_proj = max(worst_post_proj,  diag.max_post_projected)
    worst_post_raw  = max(worst_post_raw,   diag.max_post_raw_residual)
    worst_iter      = max(worst_iter,       diag.max_cg_iter)

    window_nt = (m = buf.m[cur], hflux = buf.hflux[cur],
                 cm = buf.cm[cur], ps = buf.ps[cur])
    write_streaming_window!(writer, window_nt)

    @info @sprintf("    Window %2d/%d (last): bal %.2fs  pre_raw=%.2e post_proj=%.2e iter=%d",
                   Nt, Nt, t_bal, diag.max_pre_raw_residual,
                   diag.max_post_projected, diag.max_cg_iter)

    # ── Finalize ──
    close_streaming_transport_binary!(writer)

    if settings.mass_fix_enable
        @info @sprintf("  Mass-fix offsets (Pa) min/max/mean: %+.3f / %+.3f / %+.3f",
                       minimum(ps_offsets), maximum(ps_offsets),
                       sum(ps_offsets) / Nt)
    end

    @info @sprintf("  Poisson balance summary: pre_raw=%.3e  post_proj=%.3e  post_raw=%.3e  max_cg_iter=%d",
                   worst_pre_raw, worst_post_proj, worst_post_raw, worst_iter)

    actual = filesize(bin_path)
    @info @sprintf("  Done: %s (%.2f GB, %.1fs)", basename(bin_path),
                   actual / 1e9, time() - t_day)
    actual == expected_total ||
        @warn @sprintf("File size mismatch: expected %d bytes, got %d", expected_total, actual)

    return bin_path
end
