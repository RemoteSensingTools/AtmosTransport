# ---------------------------------------------------------------------------
# Footprint result + tape-byte estimate result types.
#
# `CSFootprintResult` is the user-facing return value of the CS reverse-mode
# surface-emission footprint API; `CSTapeByteEstimate` is the diagnostic
# returned by `cs_surface_emission_footprint(..., dry_run=true)` to size
# in-memory or on-disk tape allocations.
#
# Relocated from `src/Adjoints/Adjoints.jl` lines 73-114 unchanged in
# Plan 26 P0.3a; no semantic change. Loaded into the `Adjoints` module
# via an `include` from `src/Adjoints/Adjoints.jl` — the names remain
# accessible at the same call sites with no import changes.
# ---------------------------------------------------------------------------

"""
    CSFootprintResult

Reverse-mode footprint result for one scalar objective. `footprints[t]`
is an `NTuple{6}` of `(Nc, Nc)` arrays containing `dJ / dE`, where `E` is
the per-cell surface-emission rate [kg s^-1] applied at the midpoint of
model step `t`. `lag_steps[t] == nsteps - t`.
"""
struct CSFootprintResult{FT, O <: AbstractCSFootprintObjective, A2 <: AbstractArray{FT, 2}}
    objective::O
    footprints::Vector{NTuple{6, A2}}
    lag_steps::Vector{Int}
    dt::FT
    # Compatibility field from the earlier finite-difference prototype;
    # reverse-mode results set it to zero.
    epsilon::FT
    # Not evaluated by the reverse pass. Current built-in objectives only
    # need dJ/drm at final time, independent of final tracer mass.
    base_value::FT
end

"""
    CSTapeByteEstimate

Counts and byte estimate for the CS adjoint tape. `state_bytes` counts full
stored panel states: air-mass states for linear schemes, plus tracer branch
states for nonlinear limited schemes. Halo and midpoint records are scalar
metadata and are counted in `total_records` but not included in
`state_bytes`.
"""
struct CSTapeByteEstimate
    nsteps::Int
    sweep_records::Int
    halo_records::Int
    midpoint_records::Int
    diffusion_records::Int
    convection_records::Int
    state_records::Int
    total_records::Int
    bytes_per_state::Int
    state_bytes::Int
end
