struct ReducedSpectralThreadCache
    P_buf        :: Matrix{Float64}
    fft_buffers  :: Dict{Int, Vector{ComplexF64}}
    real_buffers :: Dict{Int, Vector{Float64}}
    u_spec       :: Matrix{ComplexF64}
    v_spec       :: Matrix{ComplexF64}
end

struct ReducedTransformWorkspace
    sp          :: Vector{Float64}
    lnsp        :: Vector{Float64}
    dp          :: Matrix{Float64}
    m_arr       :: Matrix{Float64}
    hflux_arr   :: Matrix{Float64}
    cm_arr      :: Matrix{Float64}
    cell_areas  :: Vector{Float64}
    face_left   :: Vector{Int32}
    face_right  :: Vector{Int32}
    div_scratch :: Matrix{Float64}
    caches      :: Vector{ReducedSpectralThreadCache}
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
        zeros(Float64, nc, Nz_native),
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

function allocate_reduced_window_storage(Nt::Int, ::Type{FT}, grid::ReducedGaussianTargetGeometry, Nz::Int) where FT
    mesh = grid.mesh
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
