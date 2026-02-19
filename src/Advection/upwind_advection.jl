# ---------------------------------------------------------------------------
# First-order upwind advection (reference scheme)
#
# Uses KernelAbstractions @kernel for unified CPU/GPU execution.
# No code duplication: the same kernels run on both backends.
#
# All physical constants (R, ε) are passed as kernel arguments from the grid,
# not hardcoded. This enables different planets, precision, etc.
#
# Tracers and velocities are NamedTuples of 3D arrays.
# u: (Nx+1, Ny, Nz), v: (Nx, Ny+1, Nz), w: (Nx, Ny, Nz+1)
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index

"""
$(TYPEDEF)

First-order upwind advection. Simple reference scheme for testing.
"""
struct UpwindAdvection <: AbstractAdvectionScheme end

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@kernel function _upwind_advect_x!(c_new, @Const(c), @Const(u),
                                   @Const(φᶜ), Δλ, Δφ, R, Nx, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        dx = R * cosd(φᶜ[j]) * deg2rad(Δλ)
        dx = max(dx, R * deg2rad(Δφ))

        i_prev = i == 1  ? Nx : i - 1
        i_next = i == Nx ? 1  : i + 1

        u_r = u[i + 1, j, k]
        u_l = u[i, j, k]
        fx_r = u_r > 0 ? u_r * c[i, j, k]      : u_r * c[i_next, j, k]
        fx_l = u_l > 0 ? u_l * c[i_prev, j, k]  : u_l * c[i, j, k]

        c_new[i, j, k] = c[i, j, k] - Δt / dx * (fx_r - fx_l)
    end
end

@kernel function _upwind_advect_y!(c_new, @Const(c), @Const(v),
                                   @Const(φᶠ), R, ε, Ny, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(c)

        cos_N = cosd(φᶠ[j + 1])
        cos_S = cosd(φᶠ[j])
        sin_N = sind(φᶠ[j + 1])
        sin_S = sind(φᶠ[j])
        inv_ds = one(FT) / max(abs(sin_N - sin_S), ε)

        v_n = v[i, j + 1, k]
        v_s = v[i, j, k]

        fy_n = if j < Ny
            v_n > 0 ? v_n * cos_N * c[i, j, k] : v_n * cos_N * c[i, j + 1, k]
        else
            v_n > 0 ? v_n * cos_N * c[i, j, k] : zero(FT)
        end

        fy_s = if j > 1
            v_s > 0 ? v_s * cos_S * c[i, j - 1, k] : v_s * cos_S * c[i, j, k]
        else
            v_s <= 0 ? v_s * cos_S * c[i, j, k] : zero(FT)
        end

        c_new[i, j, k] = c[i, j, k] - Δt / R * inv_ds * (fy_n - fy_s)
    end
end

@kernel function _upwind_advect_z!(c_new, @Const(c), @Const(w),
                                   @Const(Δz_arr), Nz, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        FT  = eltype(c)
        dpk = Δz_arr[k]

        w_t = w[i, j, k]
        w_b = w[i, j, k + 1]

        fz_t = if k > 1
            w_t > 0 ? w_t * c[i, j, k - 1] : w_t * c[i, j, k]
        else
            w_t <= 0 ? w_t * c[i, j, k] : zero(FT)
        end

        fz_b = if k < Nz
            w_b > 0 ? w_b * c[i, j, k] : w_b * c[i, j, k + 1]
        else
            w_b > 0 ? w_b * c[i, j, k] : zero(FT)
        end

        c_new[i, j, k] = c[i, j, k] - Δt / dpk * (fz_b - fz_t)
    end
end

# ---------------------------------------------------------------------------
# Dispatch wrappers
# ---------------------------------------------------------------------------

function _get_backend(grid)
    return device(grid.architecture)
end

function advect_x!(tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, ::UpwindAdvection, Δt)
    u  = velocities.u
    backend = _get_backend(grid)
    FT = floattype(grid)
    for (_, c) in pairs(tracers)
        c_new = similar(c)
        kernel! = _upwind_advect_x!(backend, 256)
        kernel!(c_new, c, u, grid.φᶜ, grid.Δλ, grid.Δφ, grid.radius,
                grid.Nx, FT(Δt), ndrange=(grid.Nx, grid.Ny, grid.Nz))
        copyto!(c, c_new)
    end
    return nothing
end

function advect_y!(tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, ::UpwindAdvection, Δt)
    v  = velocities.v
    backend = _get_backend(grid)
    FT = floattype(grid)
    for (_, c) in pairs(tracers)
        c_new = similar(c)
        kernel! = _upwind_advect_y!(backend, 256)
        kernel!(c_new, c, v, grid.φᶠ, grid.radius, FT(1e-30),
                grid.Ny, FT(Δt), ndrange=(grid.Nx, grid.Ny, grid.Nz))
        copyto!(c, c_new)
    end
    return nothing
end

function advect_z!(tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, ::UpwindAdvection, Δt)
    w  = velocities.w
    backend = _get_backend(grid)
    FT = floattype(grid)
    Δz_arr = FT[Δz(k, grid) for k in 1:grid.Nz]
    Δz_dev = array_type(grid.architecture)(Δz_arr)
    for (_, c) in pairs(tracers)
        c_new = similar(c)
        kernel! = _upwind_advect_z!(backend, 256)
        kernel!(c_new, c, w, Δz_dev, grid.Nz, FT(Δt),
                ndrange=(grid.Nx, grid.Ny, grid.Nz))
        copyto!(c, c_new)
    end
    return nothing
end
