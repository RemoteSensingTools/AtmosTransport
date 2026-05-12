# ---------------------------------------------------------------------------
# Plan 26 Phase A.3 — checkpoint schedules for tape reduction.
#
# `AbstractCheckpointSchedule` lets the user trade compute for tape storage
# during the reverse pass. Two concrete schedules:
#
#   * `FullCheckpoint()` (default) — every step lives on the tape. Identical
#     behaviour to the pre-A.3 `cs_surface_emission_footprint`; zero
#     recomputation, peak tape disk ~ nsteps × per-step panel bytes.
#
#   * `StrideCheckpoint(K)` — save full `panels_m` snapshots every K steps;
#     during the reverse pass, replay each K-step window from its left-edge
#     checkpoint into a throwaway window tape, walk that window in reverse,
#     drop it, move to the previous window. Peak in-memory ops × K instead
#     of × nsteps; peak window-tape disk ~ K × per-step panel bytes.
#
# `RevolveCheckpoint` (Griewank-Walther binomial schedule) is reserved for
# a follow-up commit — its recursive replay structure deserves its own
# round of FD-identity tests.
# ---------------------------------------------------------------------------

abstract type AbstractCheckpointSchedule end

"""
    FullCheckpoint()

Default tape schedule. Every forward step contributes records to a single
tape; the reverse pass walks the tape once with no recomputation. Peak
storage is proportional to `nsteps`.
"""
struct FullCheckpoint <: AbstractCheckpointSchedule end

"""
    StrideCheckpoint(K)

Save a full `panels_m` checkpoint every `K` forward steps and recompute
the in-window tape lazily during the reverse pass. `K` must be a positive
integer; `K = 1` is degenerate (every step is a checkpoint and the
recompute factor matches `FullCheckpoint`); `K >= nsteps` collapses to a
single window (also degenerate but cheap on tape).

Peak in-memory ops count and per-window tape disk are both ~`K` times
the per-step cost; peak checkpoint memory is `cld(nsteps, K)` full
`panels_m` copies.
"""
struct StrideCheckpoint <: AbstractCheckpointSchedule
    K::Int
    function StrideCheckpoint(K::Integer)
        K >= 1 || throw(ArgumentError(
            "StrideCheckpoint stride must be >= 1; got $(K)"))
        return new(Int(K))
    end
end

"""
    checkpoint_window_count(schedule, nsteps) -> Int

Number of forward windows for `schedule` covering `nsteps` integration
steps. `FullCheckpoint` is 1 (the existing single tape); `StrideCheckpoint(K)`
is `cld(nsteps, K)`.
"""
checkpoint_window_count(::FullCheckpoint, nsteps::Integer) = 1
checkpoint_window_count(s::StrideCheckpoint, nsteps::Integer) = cld(Int(nsteps), s.K)

"""
    checkpoint_window_range(schedule, window_index, nsteps) -> UnitRange{Int}

Step range covered by `window_index` (1-indexed) for `schedule` and a run
of `nsteps` integration steps.
"""
checkpoint_window_range(::FullCheckpoint, w::Integer, nsteps::Integer) = 1:Int(nsteps)
function checkpoint_window_range(s::StrideCheckpoint, w::Integer, nsteps::Integer)
    nw = checkpoint_window_count(s, nsteps)
    1 <= w <= nw || throw(BoundsError(1:nw, w))
    lo = (Int(w) - 1) * s.K + 1
    hi = min(Int(w) * s.K, Int(nsteps))
    return lo:hi
end
