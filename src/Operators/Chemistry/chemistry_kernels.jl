# ---------------------------------------------------------------------------
# Multi-tracer fused decay kernel
#
# One KernelAbstractions launch per `apply!(::ExponentialDecay)` call. The
# kernel iterates over every spatial cell (Cartesian index over all axes
# except the trailing tracer axis) and, for each of the operator's
# `Nt_op ≤ ntracers(state)` selected tracers, multiplies `tracers_raw[I,
# t_idx]` by `exp(-rate * dt)`.
#
# Design notes
# ------------
# - `indices::NTuple{Nt_op, Int32}` is resolved on the host in `apply!` via
#   `tracer_index(state, op.tracer_names[n])`, so the kernel does no symbol
#   lookups.
# - Operating on `tracers_raw` directly (the packed 4D/3D buffer from
#   CellState, plan 14) means one kernel launch handles every selected
#   tracer without the Julia-level per-tracer loop that the pre-plan-15
#   `apply_chemistry!` used.
# - The kernel is rank-agnostic: for structured `(Nx, Ny, Nz, Nt)` it
#   launches `ndrange = (Nx, Ny, Nz)`; for face-indexed `(ncells, Nz, Nt)`
#   it launches `ndrange = (ncells, Nz)`. The trailing tracer axis is
#   indexed with a scalar `t_idx`, which combined with the Cartesian
#   spatial index yields the correct N-dim access.
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const

"""
    _exp_decay_kernel!(tracers_raw, indices, rates, dt, Nt_op)

Apply exponential decay `c *= exp(-rate * dt)` in-place to a packed tracer
buffer. The kernel does NOT allocate; all inputs are read-only except
`tracers_raw`.
"""
@kernel function _exp_decay_kernel!(tracers_raw,
                                     @Const(indices),
                                     @Const(rates),
                                     dt,
                                     Nt_op)
    I = @index(Global, Cartesian)
    @inbounds for n in Int32(1):Nt_op
        t_idx = indices[n]
        rate  = rates[n]
        tracers_raw[I, t_idx] *= exp(-rate * dt)
    end
end
