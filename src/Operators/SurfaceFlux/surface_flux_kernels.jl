# All kernels use Kahan compensated addition to prevent F32 rounding loss when
# a small emission increment is added to a large background tracer field.
#
# Kahan update: y = x - c; t = s + y; c = (t - s) - y; s = t
# where s = current cell value, x = rate*dt increment, c = running compensation.
# Each kernel takes a `comp` array (same surface shape as `rate`) that persists
# across substeps, carrying the accumulated rounding debt forward.

"""
    _surface_flux_kernel!(q_raw, rate, comp, dt, tracer_idx, Nz)

KernelAbstractions kernel that adds a single source's surface flux to
one tracer slab inside the 4D `tracers_raw` buffer using Kahan compensated
addition.

For structured grids, `q_raw` has shape `(Nx, Ny, Nz, Nt)`. The kernel
is launched over `(Nx, Ny)` and every thread updates the surface layer
at `k = Nz` for the tracer at `tracer_idx`:

    Kahan: y = rate[i,j]*dt - comp[i,j]
           t = q_raw[i,j,Nz,tracer_idx] + y
           comp[i,j] = (t - q_raw[i,j,Nz,tracer_idx]) - y
           q_raw[i,j,Nz,tracer_idx] = t

`comp` has shape `(Nx, Ny)` and is zero-initialised at source construction;
it persists across all substeps so the rounding debt accumulates correctly.
"""
@kernel function _surface_flux_kernel!(q_raw, @Const(rate), comp, dt, tracer_idx, Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        x = rate[i, j] * dt
        s = q_raw[i, j, Nz, tracer_idx]
        c = comp[i, j]
        y = x - c
        t = s + y
        comp[i, j]                   = (t - s) - y
        q_raw[i, j, Nz, tracer_idx]  = t
    end
end

"""
    _surface_flux_face_kernel!(q_raw, rate, comp, dt, tracer_idx, Nz)

Face-indexed packed surface-flux kernel with Kahan compensation. `q_raw`
has shape `(ncells, Nz, Nt)` and `rate`/`comp` have shape `(ncells,)`.
"""
@kernel function _surface_flux_face_kernel!(q_raw, @Const(rate), comp, dt, tracer_idx, Nz)
    c_idx = @index(Global, Linear)
    @inbounds begin
        x = rate[c_idx] * dt
        s = q_raw[c_idx, Nz, tracer_idx]
        c = comp[c_idx]
        y = x - c
        t = s + y
        comp[c_idx]                    = (t - s) - y
        q_raw[c_idx, Nz, tracer_idx]   = t
    end
end

"""
    _surface_flux_face_single_kernel!(q_raw, rate, comp, dt, Nz)

Face-indexed single-tracer helper with Kahan compensation for a
`(ncells, Nz)` tracer slice. Used by the reduced-Gaussian advection
palindrome.
"""
@kernel function _surface_flux_face_single_kernel!(q_raw, @Const(rate), comp, dt, Nz)
    c_idx = @index(Global, Linear)
    @inbounds begin
        x = rate[c_idx] * dt
        s = q_raw[c_idx, Nz]
        c = comp[c_idx]
        y = x - c
        t = s + y
        comp[c_idx]   = (t - s) - y
        q_raw[c_idx, Nz] = t
    end
end

"""
    _surface_flux_cs_single_kernel!(q_raw, rate, comp, dt, Nz, Hp)

Cubed-sphere single-tracer surface-flux kernel with Kahan compensation.
`q_raw` is one halo-padded tracer panel `(Nc + 2Hp, Nc + 2Hp, Nz)`,
`rate` and `comp` are the interior `(Nc, Nc)` panel arrays.
"""
@kernel function _surface_flux_cs_single_kernel!(q_raw, @Const(rate), comp, dt, Nz, Hp)
    ii, jj = @index(Global, NTuple)
    @inbounds begin
        x = rate[ii, jj] * dt
        s = q_raw[ii + Hp, jj + Hp, Nz]
        c = comp[ii, jj]
        y = x - c
        t = s + y
        comp[ii, jj]                  = (t - s) - y
        q_raw[ii + Hp, jj + Hp, Nz]  = t
    end
end

"""
    _surface_flux_cs_single_interp_kernel!(q_raw, series, comp, w0, w1, i0, i1, dt, Nz, Hp)

Cubed-sphere single-tracer time-interpolated surface-flux kernel with
Kahan compensation. The blended increment `(w0·series[i0] + w1·series[i1])·dt`
is added via Kahan to `q_raw[ii+Hp, jj+Hp, Nz]`.
"""
@kernel function _surface_flux_cs_single_interp_kernel!(q_raw, @Const(series),
                                                        comp, w0, w1, i0, i1, dt, Nz, Hp)
    ii, jj = @index(Global, NTuple)
    @inbounds begin
        x = (w0 * series[ii, jj, i0] + w1 * series[ii, jj, i1]) * dt
        s = q_raw[ii + Hp, jj + Hp, Nz]
        c = comp[ii, jj]
        y = x - c
        t = s + y
        comp[ii, jj]                  = (t - s) - y
        q_raw[ii + Hp, jj + Hp, Nz]  = t
    end
end

"""
    _surface_flux_cs_kernel!(q_raw, rate, comp, dt, tracer_idx, Nz, Hp)

Packed cubed-sphere surface-flux kernel with Kahan compensation. `q_raw`
is one halo-padded panel `(Nc + 2Hp, Nc + 2Hp, Nz, Nt)` and `rate`/`comp`
are the interior `(Nc, Nc)` panel arrays.
"""
@kernel function _surface_flux_cs_kernel!(q_raw, @Const(rate), comp, dt, tracer_idx, Nz, Hp)
    ii, jj = @index(Global, NTuple)
    @inbounds begin
        x = rate[ii, jj] * dt
        s = q_raw[ii + Hp, jj + Hp, Nz, tracer_idx]
        c = comp[ii, jj]
        y = x - c
        t = s + y
        comp[ii, jj]                              = (t - s) - y
        q_raw[ii + Hp, jj + Hp, Nz, tracer_idx]  = t
    end
end

"""
    _surface_flux_cs_interp_kernel!(q_raw, series, comp, w0, w1, i0, i1, dt, tracer_idx, Nz, Hp)

Packed cubed-sphere time-interpolated surface-flux kernel with Kahan
compensation. The blended increment is applied to `tracer_idx` via Kahan.
"""
@kernel function _surface_flux_cs_interp_kernel!(q_raw, @Const(series),
                                                  comp, w0, w1, i0, i1, dt, tracer_idx, Nz, Hp)
    ii, jj = @index(Global, NTuple)
    @inbounds begin
        x = (w0 * series[ii, jj, i0] + w1 * series[ii, jj, i1]) * dt
        s = q_raw[ii + Hp, jj + Hp, Nz, tracer_idx]
        c = comp[ii, jj]
        y = x - c
        t = s + y
        comp[ii, jj]                              = (t - s) - y
        q_raw[ii + Hp, jj + Hp, Nz, tracer_idx]  = t
    end
end
