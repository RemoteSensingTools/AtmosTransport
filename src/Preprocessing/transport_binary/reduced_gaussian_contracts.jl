# ===========================================================================
# Plan 41 P1 — per-window RG transport-binary contract surface.
#
# Mirrors `cubed_sphere_contracts.jl` and `latlon_contracts.jl` for the
# face-indexed reduced-Gaussian topology. Today the RG preprocessor calls
# only `verify_window_continuity_rg` (the replay gate); there is no
# analogue of `verify_substep_positivity_cs!` for RG fluxes. P1 closes
# that asymmetry: RG gets the same per-substep positivity gate, the same
# worst-window accumulator, and the same `require_substep_positivity`
# escape-hatch policy as CS. The gate is intentionally NOT wired into the
# RG `process_day` path yet — that's P2.
#
# RG array shapes (confirmed against `ReducedWindowStorage` and
# `verify_window_continuity_rg`):
#
#     m_cur      :: (nc, Nz)             # cell-centered mass
#     hflux      :: (nf, Nz)             # face mass-flux per substep
#     cm         :: (nc, Nz + 1)         # vertical interface flux
#     face_left  :: Vector{Int32}        # `face_left[f] = c` ⇒ cell c is on
#                                        #   the left side of face f
#     face_right :: Vector{Int32}        # same on the right side
#
# Sign convention: `hflux[f, k] > 0` means flux goes from
# `face_left[f]` to `face_right[f]`. So for cell `c`, the outgoing
# horizontal flux through face `f`:
#
#     face_left[f]  == c → outgoing = max(0,  hflux[f, k])
#     face_right[f] == c → outgoing = max(0, -hflux[f, k])
#
# Some faces are "boundary" (one of `face_left` / `face_right` is 0 or
# negative). The replay kernel only accumulates the interior faces; the
# positivity kernel does the same.
#
# Direction reported in the diagnostic is `:h` (horizontal) or `:z`
# (vertical); RG faces aren't axis-aligned, so there is no separate `:x`/
# `:y` decomposition. The CFL contract is per-direction in CS because
# each direction runs its own substep schedule, and the same coarsening
# applies to RG: the horizontal pass is one substep direction (mixed x/y)
# and the vertical pass is the second.
# ===========================================================================

"""
    verify_substep_positivity_rg!(m, hflux, cm, face_left, face_right;
                                  cfl_limit = 0.95)

Per-substep horizontal+vertical positivity scan for a face-indexed RG
window. Mirrors `verify_substep_positivity_cs!` / `..._ll!` but operates
on the face-indexed RG mass-flux representation.

For every cell `(c, k)`:
  1. `m > 0`. A non-positive cell mass is reported with `ratio = Inf`.
  2. Horizontal outgoing mass per substep ≤ `cfl_limit * m`. Horizontal
     outgoing is the sum across all RG faces that touch cell `c` of the
     per-face outflow (sign-aware, see file header).
  3. Vertical outgoing mass per substep ≤ `cfl_limit * m`. Same as
     CS / LL: `max(0, -cm[c, k]) + max(0, cm[c, k+1])`.

`NaN`/`Inf` cell mass and `NaN`/`Inf` fluxes are flagged as `ratio = Inf`
(matches the CS round-2 fix).

Returns `(direction, ratio, location, ok)` with:
  - `direction :: Union{Symbol, Nothing}` — `:h` / `:z` / `nothing`.
  - `ratio :: Float64` — worst `outgoing / m`.
  - `location :: NTuple{2, Int}` — `(cell, level)`.
  - `ok :: Bool` — `ratio ≤ cfl_limit`.
"""
function verify_substep_positivity_rg!(m::AbstractMatrix{FT},
                                        hflux::AbstractMatrix,
                                        cm::AbstractMatrix,
                                        face_left::AbstractVector{<:Integer},
                                        face_right::AbstractVector{<:Integer};
                                        cfl_limit::Real = 0.95) where FT
    nc, Nz = size(m)
    size(hflux, 2) == Nz ||
        error("verify_substep_positivity_rg!: hflux level count " *
              "$(size(hflux, 2)) != m level count $(Nz).")
    size(cm) == (nc, Nz + 1) ||
        error("verify_substep_positivity_rg!: cm shape $(size(cm)) " *
              "incompatible with m $(size(m)); expected ($(nc), $(Nz + 1)).")
    nf = size(hflux, 1)
    length(face_left)  == nf ||
        error("verify_substep_positivity_rg!: face_left length " *
              "$(length(face_left)) != nfaces $(nf).")
    length(face_right) == nf ||
        error("verify_substep_positivity_rg!: face_right length " *
              "$(length(face_right)) != nfaces $(nf).")

    # Accumulate per-cell horizontal outflow once for all levels. A NaN
    # or Inf flux at any face is propagated into the affected cells'
    # outflow so the worst-ratio scan reports the contamination.
    outgoing_h = zeros(Float64, nc, Nz)
    bad_h      = falses(nc, Nz)
    @inbounds for f in 1:nf
        cL = Int(face_left[f])
        cR = Int(face_right[f])
        cL_interior = 1 ≤ cL ≤ nc
        cR_interior = 1 ≤ cR ≤ nc
        for k in 1:Nz
            h = hflux[f, k]
            if !isfinite(h)
                cL_interior && (bad_h[cL, k] = true)
                cR_interior && (bad_h[cR, k] = true)
                continue
            end
            if cL_interior && h > zero(h)
                outgoing_h[cL, k] += Float64(h)
            end
            if cR_interior && h < zero(h)
                outgoing_h[cR, k] -= Float64(h)
            end
        end
    end

    worst_dir = nothing
    worst_ratio = 0.0
    worst_loc = (0, 0)
    @inbounds for k in 1:Nz, c in 1:nc
        mi = m[c, k]
        if !isfinite(mi) || mi <= zero(FT)
            if !isinf(worst_ratio)
                worst_ratio = Inf
                worst_dir = :h
                worst_loc = (c, k)
            end
            continue
        end
        # Horizontal first.
        if bad_h[c, k]
            if !isinf(worst_ratio)
                worst_ratio = Inf
                worst_dir = :h
                worst_loc = (c, k)
            end
        else
            ratio_h = outgoing_h[c, k] / Float64(mi)
            if ratio_h > worst_ratio
                worst_ratio = ratio_h
                worst_dir = :h
                worst_loc = (c, k)
            end
        end
        # Vertical second.
        fl = cm[c, k]
        fh = cm[c, k + 1]
        if !isfinite(fl) || !isfinite(fh)
            if !isinf(worst_ratio)
                worst_ratio = Inf
                worst_dir = :z
                worst_loc = (c, k)
            end
            continue
        end
        outgoing_z = max(zero(FT), -fl) + max(zero(FT), fh)
        ratio_z = Float64(outgoing_z) / Float64(mi)
        if ratio_z > worst_ratio
            worst_ratio = ratio_z
            worst_dir = :z
            worst_loc = (c, k)
        end
    end

    return (direction = worst_dir, ratio = worst_ratio,
            location = worst_loc, ok = worst_ratio <= cfl_limit)
end

"""
    verify_rg_window_contract!(m_cur, hflux, cm, m_next, face_left, face_right,
                               steps_per_window, win_idx;
                               replay_tol, positivity_cfl_limit = 0.95)

Single canonical per-window RG binary contract check. Runs the replay
gate (`verify_window_continuity_rg`, errors on failure) followed by the
per-substep positivity scan (`verify_substep_positivity_rg!`, returns a
diagnostic).

Returns `(; replay, positivity)`.
"""
function verify_rg_window_contract!(m_cur::AbstractMatrix{FT},
                                     hflux::AbstractMatrix,
                                     cm::AbstractMatrix,
                                     m_next::AbstractMatrix,
                                     face_left::AbstractVector{<:Integer},
                                     face_right::AbstractVector{<:Integer},
                                     steps_per_window::Integer,
                                     win_idx::Integer;
                                     replay_tol::Real,
                                     positivity_cfl_limit::Real = 0.95) where FT
    div_scratch = Array{Float64}(undef, size(m_cur))
    replay = verify_window_continuity_rg(m_cur, hflux, cm, m_next,
                                          face_left, face_right,
                                          div_scratch, steps_per_window)
    replay.max_rel_err <= replay_tol ||
        error("Write-time replay gate FAILED for RG window $(win_idx): " *
              "rel=$(replay.max_rel_err) > tol=$(replay_tol) at cell " *
              "$(replay.worst_idx) (abs=$(replay.max_abs_err) kg). Stored RG " *
              "fluxes do not integrate to the target mass endpoint under " *
              "palindrome continuity.")
    positivity = verify_substep_positivity_rg!(m_cur, hflux, cm,
                                                face_left, face_right;
                                                cfl_limit = positivity_cfl_limit)
    return (replay = replay, positivity = positivity)
end

"""
    init_rg_positivity_accumulator() -> NamedTuple

Zero-valued state for accumulating the worst per-window RG positivity
diagnostic across a preprocessing loop.
"""
init_rg_positivity_accumulator() = (ratio = 0.0,
                                     direction = :none,
                                     win = 0,
                                     location = (0, 0))

"""
    update_rg_positivity_accumulator(worst, diag, win_idx) -> NamedTuple

Return an updated RG accumulator from a fresh per-window diagnostic.
"""
function update_rg_positivity_accumulator(worst::NamedTuple, diag::NamedTuple,
                                            win_idx::Integer)
    diag.ratio > worst.ratio || return worst
    return (ratio = diag.ratio,
            direction = diag.direction === nothing ? :none : diag.direction,
            win = Int(win_idx),
            location = diag.location)
end

"""
    summarize_rg_positivity_status(worst; cfl_limit, steps_per_window,
                                   require_substep_positivity = true,
                                   quarantine_path = nothing)

Post-loop summary helper for the RG positivity accumulator. Mirrors
`summarize_cs_positivity_status` (CS round-2 + round-3 escape-hatch
semantics).
"""
function summarize_rg_positivity_status(worst::NamedTuple;
                                          cfl_limit::Real,
                                          steps_per_window::Integer,
                                          require_substep_positivity::Bool = true,
                                          quarantine_path::Union{Nothing, AbstractString} = nothing)
    msg = @sprintf("max outgoing/m=%.3f dir=%s win=%d cell=%s (limit=%.2f)",
                   worst.ratio, worst.direction, worst.win,
                   worst.location, cfl_limit)
    if worst.ratio <= cfl_limit
        @info "  Per-substep positivity gate: $msg"
        return nothing
    end

    max_safe_factor       = typemax(Int) ÷ max(Int(steps_per_window), 1)
    useful_recommendation = isfinite(worst.ratio) &&
                             isfinite(cfl_limit) && cfl_limit > 0 &&
                             worst.ratio / cfl_limit <= max_safe_factor

    detail = if useful_recommendation
        recommended = max(Int(steps_per_window),
                          ceil(Int, worst.ratio / cfl_limit) * Int(steps_per_window))
        "Per-substep positivity contract violated: $msg. " *
        "The binary stores fluxes that violate positivity at the recorded " *
        "`steps_per_window=$(steps_per_window)`. Re-run with " *
        "`steps_per_window=$(recommended)` (or higher) on the source " *
        "preprocessing config, or set `[numerics].require_substep_positivity = false` " *
        "to suppress."
    else
        "Per-substep positivity contract violated, and no representable " *
        "`steps_per_window` can rescue it: $msg. " *
        "The observed ratio is `Inf`/`NaN` (m<=0, NaN-mass/flux, or Inf-flux) " *
        "or finite but pathologically large; the configured `cfl_limit` may " *
        "also be non-positive. Either the source data has been corrupted " *
        "upstream of preprocessing, the pressure-fixer / endpoint-balance " *
        "pass drove a cell negative, or `[numerics].positivity_cfl_limit` is " *
        "misconfigured (must satisfy `0 < cfl_limit <= 1`). Investigate the " *
        "source field at the reported cell location, or set " *
        "`[numerics].require_substep_positivity = false` to record the " *
        "violation as a warning and keep the binary for diagnostic inspection."
    end
    if require_substep_positivity
        quarantine_path === nothing || (isfile(quarantine_path) && rm(quarantine_path; force = true))
        error(detail)
    else
        @warn detail
    end
    return nothing
end

# ---------------------------------------------------------------------------
# `ReducedGaussianContract{FT}` — typed Axis-3 concrete for the RG topology.
# Mirrors `CubedSphereContract{FT}` and `LatLonContract{FT}`.
#
# The contract carries the per-grid face connectivity (`face_left`/
# `face_right`) so the abstract `verify_window!` call site doesn't need
# to know it. The `process_day` orchestrator constructs the contract
# once per run with the grid's connectivity vectors and reuses it for
# every window.
# ---------------------------------------------------------------------------

"""
    ReducedGaussianContract{FT} <: AbstractWindowContract{ReducedGaussianTargetGeometry, FT}

Typed nominal owning an RG preprocessor's per-window gate policy and
worst-window positivity accumulator. Holds the face connectivity
(`face_left` / `face_right`) so the per-window call site doesn't need
to thread it through every call.

Construction validates `positivity_cfl_limit ∈ (0, 1]`,
`steps_per_window ≥ 1`, and `length(face_left) == length(face_right)`.
"""
mutable struct ReducedGaussianContract{FT} <:
                AbstractWindowContract{ReducedGaussianTargetGeometry, FT}
    replay_tol                  :: Float64
    positivity_cfl_limit        :: Float64
    require_substep_positivity  :: Bool
    steps_per_window            :: Int
    face_left                   :: Vector{Int32}
    face_right                  :: Vector{Int32}
    worst                       :: NamedTuple

    function ReducedGaussianContract{FT}(;
                                          replay_tol::Real,
                                          positivity_cfl_limit::Real = 0.95,
                                          require_substep_positivity::Bool = true,
                                          steps_per_window::Integer,
                                          face_left::AbstractVector{<:Integer},
                                          face_right::AbstractVector{<:Integer}) where FT
        isfinite(positivity_cfl_limit) && 0 < positivity_cfl_limit ≤ 1 ||
            error("ReducedGaussianContract: positivity_cfl_limit = " *
                  "$(positivity_cfl_limit); must be in (0, 1].")
        steps_per_window ≥ 1 ||
            error("ReducedGaussianContract: steps_per_window = " *
                  "$(steps_per_window); must be ≥ 1.")
        length(face_left) == length(face_right) ||
            error("ReducedGaussianContract: face_left length " *
                  "$(length(face_left)) != face_right length " *
                  "$(length(face_right)).")
        return new(Float64(replay_tol),
                   Float64(positivity_cfl_limit),
                   require_substep_positivity,
                   Int(steps_per_window),
                   Vector{Int32}(face_left),
                   Vector{Int32}(face_right),
                   init_rg_positivity_accumulator())
    end
end

@inline contract_replay_tolerance(c::ReducedGaussianContract)   = c.replay_tol
@inline contract_cfl_limit(c::ReducedGaussianContract)          = c.positivity_cfl_limit
@inline contract_require_positivity(c::ReducedGaussianContract) = c.require_substep_positivity

function verify_window!(window, contract::ReducedGaussianContract, win_idx::Integer)
    return verify_rg_window_contract!(window.m_cur, window.hflux,
                                       window.cm, window.m_next,
                                       contract.face_left, contract.face_right,
                                       contract.steps_per_window, Int(win_idx);
                                       replay_tol           = contract.replay_tol,
                                       positivity_cfl_limit = contract.positivity_cfl_limit)
end

function update_accumulator!(contract::ReducedGaussianContract, positivity_diag,
                              win_idx::Integer)
    contract.worst = update_rg_positivity_accumulator(contract.worst,
                                                       positivity_diag,
                                                       Int(win_idx))
    return nothing
end

function summarize_status!(contract::ReducedGaussianContract;
                            quarantine_path::Union{Nothing, AbstractString} = nothing)
    return summarize_rg_positivity_status(contract.worst;
                                            cfl_limit = contract.positivity_cfl_limit,
                                            steps_per_window = contract.steps_per_window,
                                            require_substep_positivity =
                                                contract.require_substep_positivity,
                                            quarantine_path = quarantine_path)
end
