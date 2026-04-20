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
