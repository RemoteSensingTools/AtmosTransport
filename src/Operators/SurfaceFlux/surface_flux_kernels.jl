"""
    _surface_flux_kernel!(q_raw, rate, dt, tracer_idx, Nz)

KernelAbstractions kernel that adds a single source's surface flux to
one tracer slab inside the 4D `tracers_raw` buffer.

For structured grids, `q_raw` has shape `(Nx, Ny, Nz, Nt)`. The kernel
is launched over `(Nx, Ny)` and every thread updates the surface layer
at `k = Nz` for the tracer at `tracer_idx`:

    q_raw[i, j, Nz, tracer_idx] += rate[i, j] * dt

One kernel launch per emitting source. For typical N ≤ 10 tracers the
launch overhead is negligible relative to the kernel's O(Nx · Ny) work.

# Unit convention

`rate[i, j]` is in kg/s per cell (plan 17 Decision 1 — already area-
integrated). `dt` is in seconds. Result is added mass in kg. The
kernel does NOT multiply by cell area; the caller is expected to have
pre-integrated the flux into per-cell rate.

# Surface layer convention

`k = Nz` is the surface (plan 17 Decision 2). This matches the LatLon
grid storage convention used everywhere in `src/`. A future
`AbstractLayerOrdering` abstraction can generalise this.
"""
@kernel function _surface_flux_kernel!(q_raw, @Const(rate), dt, tracer_idx, Nz)
    i, j = @index(Global, NTuple)
    @inbounds q_raw[i, j, Nz, tracer_idx] += rate[i, j] * dt
end

"""
    _surface_flux_face_kernel!(q_raw, rate, dt, tracer_idx, Nz)

Face-indexed packed surface-flux kernel. `q_raw` has shape
`(ncells, Nz, Nt)` and `rate` has shape `(ncells,)`. The kernel is
launched over `ncells` and updates the surface layer `k = Nz` for the
target tracer:

    q_raw[c, Nz, tracer_idx] += rate[c] * dt
"""
@kernel function _surface_flux_face_kernel!(q_raw, @Const(rate), dt, tracer_idx, Nz)
    c = @index(Global, Linear)
    @inbounds q_raw[c, Nz, tracer_idx] += rate[c] * dt
end

"""
    _surface_flux_face_single_kernel!(q_raw, rate, dt, Nz)

Face-indexed single-tracer helper for a `(ncells, Nz)` tracer slice.
Used by the reduced-Gaussian advection palindrome, which still loops
over tracers one slice at a time.
"""
@kernel function _surface_flux_face_single_kernel!(q_raw, @Const(rate), dt, Nz)
    c = @index(Global, Linear)
    @inbounds q_raw[c, Nz] += rate[c] * dt
end

"""
    _surface_flux_cs_single_kernel!(q_raw, rate, dt, Nz, Hp)

Cubed-sphere single-tracer surface-flux kernel. `q_raw` is one
halo-padded tracer panel `(Nc + 2Hp, Nc + 2Hp, Nz)` and `rate` is the
interior `(Nc, Nc)` panel source.
"""
@kernel function _surface_flux_cs_single_kernel!(q_raw, @Const(rate), dt, Nz, Hp)
    ii, jj = @index(Global, NTuple)
    @inbounds q_raw[ii + Hp, jj + Hp, Nz] += rate[ii, jj] * dt
end
