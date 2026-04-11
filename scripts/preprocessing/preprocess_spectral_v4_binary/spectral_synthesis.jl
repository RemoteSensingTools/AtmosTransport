"""
    compute_legendre_column!(P, T, sin_lat)

Fill the associated Legendre table used by the spherical-harmonic synthesis for
one latitude.
"""
function compute_legendre_column!(P::Matrix{Float64}, T::Int, sin_lat::Float64)
    cos_lat = sqrt(1.0 - sin_lat^2)

    fill!(P, 0.0)
    P[1, 1] = 1.0

    for m in 1:T
        P[m + 1, m + 1] = sqrt((2m + 1.0) / (2m)) * cos_lat * P[m, m]
    end

    for m in 0:(T - 1)
        P[m + 2, m + 1] = sqrt(2m + 3.0) * sin_lat * P[m + 1, m + 1]
    end

    for m in 0:T
        for n in (m + 2):T
            n2 = n * n
            m2 = m * m
            a = sqrt((4.0 * n2 - 1.0) / (n2 - m2))
            b = sqrt(((2n + 1.0) * (n - m - 1.0) * (n + m - 1.0)) /
                     ((2n - 3.0) * (n2 - m2)))
            P[n + 1, m + 1] = a * sin_lat * P[n, m + 1] - b * P[n - 1, m + 1]
        end
    end
    return nothing
end

"""
    _delta(m, n)

IFS/TM5 helper coefficient used in the vorticity/divergence to wind transform.
"""
@inline function _delta(m::Int, n::Int)
    n == 0 && return 0.0
    n2 = Float64(n * n)
    m2 = Float64(m * m)
    return -sqrt((n2 - m2) / (4n2 - 1)) / n
end

"""
    _sigma(m, n)

IFS/TM5 helper coefficient used in the vorticity/divergence to wind transform.
"""
@inline function _sigma(m::Int, n::Int)
    (n == 0 || m == 0) && return 0.0
    return -Float64(m) / (Float64(n) * (n + 1))
end

"""
    vod2uv!(u_spec, v_spec, vo_spec, d_spec, T)

Convert one level of spectral relative vorticity and divergence into spectral
zonal and meridional wind coefficients.
"""
function vod2uv!(u_spec::Matrix{ComplexF64}, v_spec::Matrix{ComplexF64},
                 vo_spec::AbstractMatrix{ComplexF64}, d_spec::AbstractMatrix{ComplexF64},
                 T::Int)
    fill!(u_spec, zero(ComplexF64))
    fill!(v_spec, zero(ComplexF64))

    for m in 0:T
        for n in m:T
            delta_mn = _delta(m, n)
            delta_mn1 = _delta(m, n + 1)
            sigma_mn = _sigma(m, n)

            vo_nm1 = n > m ? vo_spec[n, m + 1] : zero(ComplexF64)
            vo_np1 = n < T ? vo_spec[n + 2, m + 1] : zero(ComplexF64)
            d_nm1 = n > m ? d_spec[n, m + 1] : zero(ComplexF64)
            d_np1 = n < T ? d_spec[n + 2, m + 1] : zero(ComplexF64)
            vo_n = vo_spec[n + 1, m + 1]
            d_n = d_spec[n + 1, m + 1]

            u_spec[n + 1, m + 1] = R_EARTH * (delta_mn * vo_nm1 + im * sigma_mn * d_n - delta_mn1 * vo_np1)
            v_spec[n + 1, m + 1] = R_EARTH * (-delta_mn * d_nm1 + im * sigma_mn * vo_n + delta_mn1 * d_np1)
        end
    end
    return nothing
end

"""
    spectral_to_grid!(field, spec, T, lats, Nlon, P_buf, fft_buf; ...)

Synthesize a single spectral field onto the target regular lat-lon grid.
The optional longitude shift is used so face- and center-staggered quantities
can share the same spherical-harmonic path.
"""
function spectral_to_grid!(field::Matrix{Float64},
                           spec::AbstractMatrix{ComplexF64},
                           T::Int,
                           lats::Vector{Float64},
                           Nlon::Int,
                           P_buf::Matrix{Float64},
                           fft_buf::Vector{ComplexF64};
                           fft_out::Union{Nothing, Vector{ComplexF64}}=nothing,
                           bfft_plan=nothing,
                           lon_shift_rad::Float64=0.0)
    Nfft = Nlon

    for j in eachindex(lats)
        sin_lat = sind(lats[j])
        compute_legendre_column!(P_buf, T, sin_lat)

        fill!(fft_buf, zero(ComplexF64))
        for m in 0:min(T, div(Nfft, 2))
            Gm = zero(ComplexF64)
            @inbounds for n in m:T
                Gm += spec[n + 1, m + 1] * P_buf[n + 1, m + 1]
            end
            if lon_shift_rad != 0.0 && m > 0
                Gm *= exp(im * m * lon_shift_rad)
            end
            fft_buf[m + 1] = Gm
        end

        for m in 1:min(T, div(Nfft, 2) - 1)
            fft_buf[Nfft - m + 1] = conj(fft_buf[m + 1])
        end

        f_lon = if bfft_plan !== nothing && fft_out !== nothing
            mul!(fft_out, bfft_plan, fft_buf)
            fft_out
        else
            bfft(fft_buf)
        end

        @inbounds for i in 1:Nlon
            field[i, j] = real(f_lon[i])
        end
    end
    return nothing
end

"""
    stagger_winds!(u_stag, v_stag, u_cc, v_cc, Nlon, Nlat, Nz)

Convert cell-centered winds to the structured face staggering used by the
lat-lon flux kernels.
"""
function stagger_winds!(u_stag, v_stag, u_cc, v_cc, Nlon, Nlat, Nz)
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:Nlon
        u_stag[i, j, k] = u_cc[i, j, k]
    end
    @inbounds for k in 1:Nz, j in 1:Nlat
        u_stag[Nlon + 1, j, k] = u_cc[1, j, k]
    end

    @inbounds for k in 1:Nz, j in 2:Nlat, i in 1:Nlon
        v_stag[i, j, k] = (v_cc[i, j - 1, k] + v_cc[i, j, k]) / 2
    end
    v_stag[:, 1, :] .= 0
    v_stag[:, Nlat + 1, :] .= 0
    return nothing
end

"""
    compute_dp!(dp, ps, dA, dB, Nlon, Nlat, Nz)

Compute native-layer pressure thickness from hybrid A/B coefficients and
surface pressure.
"""
function compute_dp!(dp, ps, dA, dB, Nlon, Nlat, Nz)
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:Nlon
        dp[i, j, k] = abs(dA[k] + dB[k] * ps[i, j])
    end
    return nothing
end

"""
    compute_air_mass!(m_arr, dp, area, Nlon, Nlat, Nz)

Convert pressure thickness into total air mass per cell.
"""
function compute_air_mass!(m_arr, dp, area, Nlon, Nlat, Nz)
    inv_g = 1.0 / GRAV
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:Nlon
        m_arr[i, j, k] = dp[i, j, k] * area[i, j] * inv_g
    end
    return nothing
end

"""
    compute_mass_fluxes!(am, bm, cm, u_stag, v_stag, dp, ps, dA, dB, grid, half_dt, Nz)

Compute the native-level horizontal mass fluxes and diagnose the native-level
vertical flux from discrete continuity.

The output fluxes are scaled to `half_dt`, matching the v4 binary convention.
"""
function compute_mass_fluxes!(am, bm, cm, u_stag, v_stag, dp, ps,
                              dA, dB, grid::LatLonTargetGeometry, half_dt, Nz)
    Nlon, Nlat = nlon(grid), nlat(grid)
    R_g = R_EARTH / GRAV
    dlon = grid.dlon
    dlat = grid.dlat
    lnsp_center = log.(ps)

    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:(Nlon + 1)
        i_l = i == 1 ? Nlon : i - 1
        i_r = i <= Nlon ? i : 1
        ps_face = exp((lnsp_center[i_l, j] + lnsp_center[i_r, j]) / 2)
        dp_face = abs(dA[k] + dB[k] * ps_face)
        cos_lat = grid.cos_lat[j]
        if j == 1 || j == Nlat
            am[i, j, k] = 0
        else
            am[i, j, k] = u_stag[i, j, k] / cos_lat * dp_face * R_g * dlat * half_dt
        end
    end

    @inbounds for k in 1:Nz, j in 1:(Nlat + 1), i in 1:Nlon
        if j == 1 || j == Nlat + 1
            bm[i, j, k] = 0
        else
            j_s = j - 1
            j_n = j
            ps_face = exp((lnsp_center[i, j_s] + lnsp_center[i, j_n]) / 2)
            dp_face = abs(dA[k] + dB[k] * ps_face)
            bm[i, j, k] = v_stag[i, j, k] * dp_face * R_g * dlon * half_dt
        end
    end

    fill!(cm, zero(eltype(cm)))
    @inbounds for j in 1:Nlat, i in 1:Nlon
        pit = 0.0
        for k in 1:Nz
            pit += (Float64(am[i + 1, j, k]) - Float64(am[i, j, k])) +
                   (Float64(bm[i, j + 1, k]) - Float64(bm[i, j, k]))
        end
        acc = 0.0
        for k in 1:Nz
            div_h = (Float64(am[i + 1, j, k]) - Float64(am[i, j, k])) +
                    (Float64(bm[i, j + 1, k]) - Float64(bm[i, j, k]))
            acc = acc - div_h + Float64(dB[k]) * pit
            cm[i, j, k + 1] = eltype(cm)(acc)
        end
    end

    return nothing
end

"""
    spectral_to_native_fields!(...)

Transform one analysis hour of ERA5 spectral `lnsp`, `vo`, and `d` into native
gridpoint mass/flux fields on the configured target geometry.

This routine builds only the moist native fields. Dry-basis conversion and any
humidity-aware mass fix are applied later once the matching hourly `q` field is
available.
"""
function spectral_to_native_fields!(
    m_arr::Array{Float64, 3}, am::Array{Float64, 3}, bm::Array{Float64, 3},
    cm::Array{Float64, 3}, sp::Matrix{Float64},
    u_cc::Array{Float64, 3}, v_cc::Array{Float64, 3},
    u_stag::Array{Float64, 3}, v_stag::Array{Float64, 3},
    dp::Array{Float64, 3},
    lnsp_spec::Matrix{ComplexF64},
    vo_hour::Array{ComplexF64, 3}, d_hour::Array{ComplexF64, 3},
    T::Int, level_range::UnitRange{Int}, ab,
    grid::LatLonTargetGeometry, half_dt::Float64,
    P_buf::Matrix{Float64}, fft_buf::Vector{ComplexF64},
    field_2d::Matrix{Float64},
    P_buf_t, fft_buf_t, fft_out_t, u_spec_t, v_spec_t, field_2d_t,
    bfft_plans)

    Nlon = nlon(grid)
    Nlat = nlat(grid)
    Nz = length(level_range)

    # Compute spectral synthesis longitude phase shifts FROM THE MESH
    # coordinates, not from `dlon / 2`. The old `center_shift = dlon/2`
    # only aligned with a `longitude=(0,360)` mesh where the first cell
    # center is at `1.875°`. For the default `longitude=(-180,180)` mesh
    # the first cell center is at `-178.125°` and the first west face
    # is at `-180°`, so `spectral_to_grid!` needs a negative shift to
    # move the FFT's natural origin (physical lon = 0°) to the mesh's
    # first point. Prior bug: every LL v4 binary was 180° off in
    # longitude relative to its stored `λᶜ`. See 2026-04-11 AGENT_CHAT
    # entry for full diagnosis.
    #
    # `spectral_to_grid!` applies `Gm *= exp(im * m * lon_shift_rad)`,
    # which makes FFT output index 1 represent the field at physical
    # lon = `lon_shift_rad`. So `lon_shift_rad = deg2rad(λᶜ[1])` for
    # cell-centered scalars (sp, v_cc) and `deg2rad(λᶠ[1])` for the
    # west-face-located u_cc.
    sp_shift = deg2rad(grid.lons[1])
    u_edge_shift = deg2rad(first(grid.mesh.λᶠ))

    spectral_to_grid!(field_2d, lnsp_spec, T, grid.lats, Nlon, P_buf, fft_buf;
                      lon_shift_rad=sp_shift)
    @. sp = exp(field_2d)

    Threads.@threads for k in 1:Nz
        level = level_range[k]
        tid = Threads.threadid()

        vod2uv!(u_spec_t[tid], v_spec_t[tid],
                @view(vo_hour[:, :, level]),
                @view(d_hour[:, :, level]),
                T)

        # u_cc is located at the west face in longitude (cell edges)
        spectral_to_grid!(field_2d_t[tid], u_spec_t[tid], T,
                          grid.lats, Nlon, P_buf_t[tid], fft_buf_t[tid];
                          fft_out=fft_out_t[tid], bfft_plan=bfft_plans[tid],
                          lon_shift_rad=u_edge_shift)
        u_cc[:, :, k] .= field_2d_t[tid]

        # v_cc is located at cell center in longitude
        spectral_to_grid!(field_2d_t[tid], v_spec_t[tid], T,
                          grid.lats, Nlon, P_buf_t[tid], fft_buf_t[tid];
                          fft_out=fft_out_t[tid], bfft_plan=bfft_plans[tid],
                          lon_shift_rad=sp_shift)
        v_cc[:, :, k] .= field_2d_t[tid]
    end

    stagger_winds!(u_stag, v_stag, u_cc, v_cc, Nlon, Nlat, Nz)
    compute_dp!(dp, sp, ab.dA, ab.dB, Nlon, Nlat, Nz)
    compute_air_mass!(m_arr, dp, grid.area, Nlon, Nlat, Nz)
    compute_mass_fluxes!(am, bm, cm, u_stag, v_stag, dp, sp,
                         ab.dA, ab.dB, grid, half_dt, Nz)

    return nothing
end

"""
    recompute_native_mass_fields!(transform, ab, grid, half_dt)

Recompute native `dp`, moist cell mass, and mass fluxes after a uniform
surface-pressure adjustment has been applied to `transform.sp`.
"""
function recompute_native_mass_fields!(transform,
                                       ab,
                                       grid::LatLonTargetGeometry,
                                       half_dt::Float64)
    Nlon = nlon(grid)
    Nlat = nlat(grid)
    Nz = size(transform.m_arr, 3)

    compute_dp!(transform.dp, transform.sp, ab.dA, ab.dB, Nlon, Nlat, Nz)
    compute_air_mass!(transform.m_arr, transform.dp, grid.area, Nlon, Nlat, Nz)
    compute_mass_fluxes!(transform.am_arr, transform.bm_arr, transform.cm_arr,
                         transform.u_stag, transform.v_stag, transform.dp, transform.sp,
                         ab.dA, ab.dB, grid, half_dt, Nz)
    return nothing
end
