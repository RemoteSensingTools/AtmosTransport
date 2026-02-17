"""
    Communications

Communication abstraction layer inspired by ClimaComms.jl.

All inter-process communication (halo exchange, reductions) goes through
`AbstractComms` so that single-process and MPI codes share the same physics.

# Interface contract

Any new comms subtype must implement:
- `fill_halo!(field, grid, comms)` — exchange halo data between neighbors
- `reduce_sum(x, comms)`           — global sum reduction
- `barrier(comms)`                  — synchronization point
"""
module Communications

export AbstractComms, SingletonComms
export fill_halo!, reduce_sum, barrier

# ---------------------------------------------------------------------------
# Abstract type
# ---------------------------------------------------------------------------

"""
    AbstractComms

Supertype for communication backends. `SingletonComms` is the default
(single-process, no-op). `MPIComms` will be added via the MPI extension.
"""
abstract type AbstractComms end

# ---------------------------------------------------------------------------
# Concrete: SingletonComms (single process)
# ---------------------------------------------------------------------------

"""
    SingletonComms()

No-op communication backend for single-process runs.
Halo fills are local copies; reductions are plain `sum`.
"""
struct SingletonComms <: AbstractComms end

# ---------------------------------------------------------------------------
# Interface stubs (default to SingletonComms behavior)
# ---------------------------------------------------------------------------

"""
    fill_halo!(field, grid, comms::AbstractComms)

Exchange halo regions. Dispatches on both grid type (for boundary topology)
and comms type (for single-process vs MPI).
"""
function fill_halo! end

"""
    reduce_sum(x, comms::AbstractComms)

Global sum across all processes. On `SingletonComms`, just returns `sum(x)`.
"""
reduce_sum(x, ::SingletonComms) = sum(x)

"""
    barrier(comms::AbstractComms)

Synchronization barrier. No-op on `SingletonComms`.
"""
barrier(::SingletonComms) = nothing

end # module Communications
