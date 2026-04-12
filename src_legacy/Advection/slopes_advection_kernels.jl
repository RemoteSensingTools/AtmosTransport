# KernelAbstractions kernels for Russell-Lerner slopes advection (GPU/CPU same code).
# Used when architecture(grid) is GPU; CPU path remains the existing host loop.
# advect_x_kernel is implemented; advect_y and advect_z can be ported the same way
# (precompute Δy_j or scalar Δy, Δz_k; same kernel pattern). Convection and diffusion
# can also be ported to @kernel for full GPU support.

using KernelAbstractions: @kernel, @index

# Device-safe minmod (no grid or non-const calls).
@inline function minmod_device(a, b, c)
    if a > 0 && b > 0 && c > 0
        return min(a, b, c)
    elseif a < 0 && b < 0 && c < 0
        return max(a, b, c)
    else
        return zero(a)
    end
end

@kernel function advect_x_kernel(
    c_new, @Const(c), @Const(u), @Const(Δx_j),
    Nx, Ny, Nz, @Const(Δt), use_limiter
)
    I = @index(Global, Cartesian)
    i, j, k = I[1], I[2], I[3]
    @inbounds begin
        Δx_ij = Δx_j[j]
        i_prev = i == 1 ? Nx : i - 1
        i_next = i == Nx ? 1 : i + 1
        i_next_next = i_next == Nx ? 1 : i_next + 1
        i_prev_prev = i_prev == 1 ? Nx : i_prev - 1

        s_i = (c[i_next, j, k] - c[i_prev, j, k]) / 2
        if use_limiter
            s_i = minmod_device(s_i, 2 * (c[i_next, j, k] - c[i, j, k]), 2 * (c[i, j, k] - c[i_prev, j, k]))
        end

        s_i_next = (c[i_next_next, j, k] - c[i, j, k]) / 2
        if use_limiter
            s_i_next = minmod_device(s_i_next, 2 * (c[i_next_next, j, k] - c[i_next, j, k]), 2 * (c[i_next, j, k] - c[i, j, k]))
        end

        s_i_prev = (c[i, j, k] - c[i_prev_prev, j, k]) / 2
        if use_limiter
            s_i_prev = minmod_device(s_i_prev, 2 * (c[i, j, k] - c[i_prev, j, k]), 2 * (c[i_prev, j, k] - c[i_prev_prev, j, k]))
        end

        u_right = u[i + 1, j, k]
        u_left = u[i, j, k]
        Δx_next = Δx_j[j]  # same column j
        Δx_prev = Δx_j[j]

        flux_right = u_right > 0 ?
            u_right * (c[i, j, k] + (1 - u_right * Δt / Δx_ij) * s_i / 2) :
            u_right * (c[i_next, j, k] - (1 + u_right * Δt / Δx_next) * s_i_next / 2)

        flux_left = u_left > 0 ?
            u_left * (c[i_prev, j, k] + (1 - u_left * Δt / Δx_prev) * s_i_prev / 2) :
            u_left * (c[i, j, k] - (1 + u_left * Δt / Δx_ij) * s_i / 2)

        c_new[i, j, k] = c[i, j, k] - Δt / Δx_ij * (flux_right - flux_left)
    end
end

@kernel function advect_y_kernel(
    c_new, @Const(c), @Const(v), @Const(Δy_val),
    Nx, Ny, Nz, @Const(Δt), use_limiter
)
    I = @index(Global, Cartesian)
    i, j, k = I[1], I[2], I[3]
    @inbounds begin
        # j boundaries: j=1 is south pole, j=Ny is north pole
        # Zero flux at poles
        # Slopes: zero at boundaries (j=1 or j=Ny)

        s_j = if j > 1 && j < Ny
            s_raw = (c[i, j+1, k] - c[i, j-1, k]) / 2
            if use_limiter
                minmod_device(s_raw, 2*(c[i, j+1, k] - c[i, j, k]), 2*(c[i, j, k] - c[i, j-1, k]))
            else
                s_raw
            end
        else
            zero(eltype(c))
        end

        s_j_next = if j < Ny && j+1 > 1 && j+1 < Ny
            s_raw = (c[i, j+2, k] - c[i, j, k]) / 2
            if use_limiter
                minmod_device(s_raw, 2*(c[i, j+2, k] - c[i, j+1, k]), 2*(c[i, j+1, k] - c[i, j, k]))
            else
                s_raw
            end
        else
            zero(eltype(c))
        end

        s_j_prev = if j > 1 && j-1 >= 1 && j-1 < Ny
            j_prev_prev = j - 2
            if j_prev_prev >= 1
                s_raw = (c[i, j, k] - c[i, j_prev_prev, k]) / 2
                if use_limiter
                    minmod_device(s_raw, 2*(c[i, j, k] - c[i, j-1, k]), 2*(c[i, j-1, k] - c[i, j_prev_prev, k]))
                else
                    s_raw
                end
            else
                zero(eltype(c))
            end
        else
            zero(eltype(c))
        end

        v_right = v[i, j+1, k]
        v_left = v[i, j, k]

        flux_right = if j < Ny
            if v_right > 0
                v_right * (c[i, j, k] + (1 - v_right * Δt / Δy_val) * s_j / 2)
            else
                v_right * (c[i, j+1, k] - (1 + v_right * Δt / Δy_val) * s_j_next / 2)
            end
        else
            v_right > 0 ? v_right * c[i, j, k] : zero(eltype(c))
        end

        flux_left = if j > 1
            if v_left > 0
                v_left * (c[i, j-1, k] + (1 - v_left * Δt / Δy_val) * s_j_prev / 2)
            else
                v_left * (c[i, j, k] - (1 + v_left * Δt / Δy_val) * s_j / 2)
            end
        else
            v_left <= 0 ? v_left * c[i, j, k] : zero(eltype(c))
        end

        c_new[i, j, k] = c[i, j, k] - Δt / Δy_val * (flux_right - flux_left)
    end
end

@kernel function advect_z_kernel(
    c_new, @Const(c), @Const(w), @Const(Δz_arr),
    Nx, Ny, Nz, @Const(Δt), use_limiter
)
    I = @index(Global, Cartesian)
    i, j, k = I[1], I[2], I[3]
    @inbounds begin
        Δz_k = Δz_arr[i, j, k]

        s_k = if k > 1 && k < Nz
            s_raw = (c[i, j, k+1] - c[i, j, k-1]) / 2
            if use_limiter
                minmod_device(s_raw, 2*(c[i, j, k+1] - c[i, j, k]), 2*(c[i, j, k] - c[i, j, k-1]))
            else
                s_raw
            end
        else
            zero(eltype(c))
        end

        s_k_next = if k < Nz && k+1 > 1 && k+1 < Nz
            s_raw = (c[i, j, k+2] - c[i, j, k]) / 2
            if use_limiter
                minmod_device(s_raw, 2*(c[i, j, k+2] - c[i, j, k+1]), 2*(c[i, j, k+1] - c[i, j, k]))
            else
                s_raw
            end
        else
            zero(eltype(c))
        end

        s_k_prev = if k > 1 && k-1 >= 1 && k-1 < Nz
            k_prev_prev = k - 2
            if k_prev_prev >= 1
                s_raw = (c[i, j, k] - c[i, j, k_prev_prev]) / 2
                if use_limiter
                    minmod_device(s_raw, 2*(c[i, j, k] - c[i, j, k-1]), 2*(c[i, j, k-1] - c[i, j, k_prev_prev]))
                else
                    s_raw
                end
            else
                zero(eltype(c))
            end
        else
            zero(eltype(c))
        end

        Δz_next = k < Nz ? Δz_arr[i, j, k+1] : Δz_k
        Δz_prev = k > 1 ? Δz_arr[i, j, k-1] : Δz_k

        w_top = w[i, j, k]
        w_bot = w[i, j, k+1]

        flux_top = if k > 1
            if w_top > 0
                w_top * (c[i, j, k-1] + (1 - w_top * Δt / Δz_prev) * s_k_prev / 2)
            else
                w_top * (c[i, j, k] - (1 + w_top * Δt / Δz_k) * s_k / 2)
            end
        else
            w_top <= 0 ? w_top * c[i, j, k] : zero(eltype(c))
        end

        flux_bot = if k < Nz
            if w_bot > 0
                w_bot * (c[i, j, k] + (1 - w_bot * Δt / Δz_k) * s_k / 2)
            else
                w_bot * (c[i, j, k+1] - (1 + w_bot * Δt / Δz_next) * s_k_next / 2)
            end
        else
            w_bot > 0 ? w_bot * c[i, j, k] : zero(eltype(c))
        end

        c_new[i, j, k] = c[i, j, k] - Δt / Δz_k * (flux_bot - flux_top)
    end
end
