# ---------------------------------------------------------------------------
# Face kernels — one thread per (face, level)
#
# Used for: horizontal advection reconstruction, face flux evaluation,
# face metric application. This is the most important kernel family
# for future reduced Gaussian grid support.
#
# For structured grids, face kernels are currently organized by direction
# (x-face, y-face) within the advection scheme modules (RussellLerner.jl).
# This file provides face-oriented helper kernels that are scheme-agnostic.
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, get_backend, synchronize

"""
    compute_mass_flux_from_wind!(am, u, dp, dy, Nx, half_dt, g)

Convert staggered wind to x-face mass flux:
am[i,j,k] = half_dt × u[i,j,k] × dp_face × dy[j] / g
"""
@kernel function _mass_flux_from_wind_x_kernel!(am, @Const(u), @Const(dp),
                                                  @Const(dy), Nx, half_dt, g)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        il = i == 1 ? Nx : i - 1
        ir = i > Nx ? 1 : i
        dp_f = (dp[il, j, k] + dp[ir, j, k]) / 2
        am[i, j, k] = half_dt * u[i, j, k] * dp_f * dy[j] / g
    end
end

"""
    compute_mass_flux_from_wind!(bm, v, dp, dx_face, Ny, half_dt, g)

Convert staggered wind to y-face mass flux:
bm[i,j,k] = half_dt × v[i,j,k] × dp_face × |dx_face[j]| / g
"""
@kernel function _mass_flux_from_wind_y_kernel!(bm, @Const(v), @Const(dp),
                                                  @Const(dx_face), Ny, half_dt, g)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        jb = max(j - 1, 1)
        ja = min(j, Ny)
        dp_f = (dp[i, jb, k] + dp[i, ja, k]) / 2
        bm[i, j, k] = half_dt * v[i, j, k] * dp_f * abs(dx_face[j]) / g
    end
end

export _mass_flux_from_wind_x_kernel!, _mass_flux_from_wind_y_kernel!
