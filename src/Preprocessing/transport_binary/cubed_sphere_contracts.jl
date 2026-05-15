# Cubed-sphere replay helpers shared by spectral and regrid CS preprocessors.

"""
    fill_cs_window_mass_tendency!(dm_panels, m_cur, m_next, steps_per_window)

Fill the CS Poisson-balance target for one window.

`m_cur` and `m_next` are explicit endpoint masses. The stored CS horizontal
fluxes are half-sweep amounts under Strang splitting, so the target is
`(m_next - m_cur) / (2 * steps_per_window)` per panel cell.
"""
@inline function fill_cs_window_mass_tendency!(dm_panels::NTuple{NP, <:AbstractArray{FT, 3}},
                                               m_cur::NTuple{NP, <:AbstractArray{FT, 3}},
                                               m_next::NTuple{NP, <:AbstractArray{FT, 3}},
                                               steps_per_window::Int) where {FT, NP}
    inv_two_steps = one(FT) / FT(2 * steps_per_window)
    for p in 1:NP
        @inbounds for idx in eachindex(dm_panels[p])
            dm_panels[p][idx] = (m_next[p][idx] - m_cur[p][idx]) * inv_two_steps
        end
    end
    return nothing
end

"""
    convert_cs_mass_target_to_delta!(m_target, m_cur)

Convert an in-place CS endpoint target into the on-disk `dm` payload.

The CS writer stores forward endpoint differences. Call this only after all
balance and replay checks that still need the absolute target endpoint.
"""
@inline function convert_cs_mass_target_to_delta!(m_target::NTuple{NP, <:AbstractArray{FT, 3}},
                                                  m_cur::NTuple{NP, <:AbstractArray{FT, 3}}) where {FT, NP}
    for p in 1:NP
        @inbounds for idx in eachindex(m_target[p])
            m_target[p][idx] -= m_cur[p][idx]
        end
    end
    return nothing
end

"""
    verify_write_replay_cs!(m_cur, am, bm, cm, m_next, steps_per_window, tol_rel, win_idx)

Run the CS write-time replay gate for one window and return its diagnostic.

The check integrates the stored panel-local fluxes from `m_cur` under the
runtime palindrome-continuity contract and verifies that the result matches the
explicit endpoint `m_next`. A failure here means the binary would produce a
runtime day-boundary or window-boundary mass inconsistency.
"""
function verify_write_replay_cs!(m_cur::NTuple{NP, <:AbstractArray{FT, 3}},
                                 am::NTuple{NP, <:AbstractArray},
                                 bm::NTuple{NP, <:AbstractArray},
                                 cm::NTuple{NP, <:AbstractArray},
                                 m_next::NTuple{NP, <:AbstractArray},
                                 steps_per_window::Int,
                                 tol_rel::Real,
                                 win_idx::Int) where {FT, NP}
    diag = verify_window_continuity_cs(m_cur, am, bm, cm, m_next, steps_per_window)
    diag.max_rel_err <= tol_rel ||
        error("Write-time replay gate FAILED for CS window $(win_idx): " *
              "rel=$(diag.max_rel_err) > tol=$(tol_rel) at cell $(diag.worst_idx) " *
              "(abs=$(diag.max_abs_err) kg). Stored CS fluxes do not integrate to " *
              "the target mass endpoint under palindrome continuity.")
    return diag
end

"""
    verify_substep_positivity_cs!(m, am, bm, cm; cfl_limit = 0.95, halo_width = 0)

Verify the per-substep horizontal+vertical positivity contract that the runtime's
`_cs_static_subcycle_count` depends on. For every interior cell on every panel:

  1. The cell air mass itself must be positive (`m > 0`). A non-positive cell
     mass is an immediate contract violation — the runtime divides by `m` and
     would produce `Inf` or `NaN` in the CFL scan. Such a cell is reported with
     `ratio = Inf` regardless of flux magnitude.
  2. The per-direction outgoing mass per substep must not exceed `cfl_limit * m`.

Returns a NamedTuple `(direction, ratio, location, ok)`:
* `direction :: Union{Symbol, Nothing}` — `:x`, `:y`, `:z`, or `nothing` when no
  cell was inspected.
* `ratio :: Float64` — worst observed `outgoing / m` over the window, or `Inf`
  if any cell had `m <= 0`.
* `location :: NTuple{4, Int}` — `(panel, i, j, k)` of the worst cell.
* `ok :: Bool` — `true` iff `ratio <= cfl_limit`.

The replay gate (`verify_write_replay_cs!`) only checks endpoint continuity. A
binary that drives a cell mass negative mid-sweep can still pass replay because
the cell re-fills from inflow before the window ends — but the runtime cannot
recover. This gate is the actual contract the runtime depends on.

`halo_width` defaults to `0` (panel arrays are stored unhaloed at preprocess
time); pass `> 0` to scan only the interior of a haloed buffer.
"""
function verify_substep_positivity_cs!(m::NTuple{NP, <:AbstractArray{FT, 3}},
                                       am::NTuple{NP, <:AbstractArray},
                                       bm::NTuple{NP, <:AbstractArray},
                                       cm::NTuple{NP, <:AbstractArray};
                                       cfl_limit::Real = 0.95,
                                       halo_width::Integer = 0) where {FT, NP}
    Hp = Int(halo_width)
    # All buffers share the same interior extent `(Nc, Nc, Nz)`; derive from m.
    Nc = size(m[1], 1) - 2Hp
    Nz = size(m[1], 3)
    iL = Hp + 1
    iH = Hp + Nc
    worst_dir = nothing
    worst_ratio = 0.0
    worst_loc = (0, 0, 0, 0)
    for p in 1:NP
        m_p = m[p]
        for (dir, F_lo_view, F_hi_view) in (
            (:x, view(am[p], iL    :iH,     iL:iH,     1:Nz),
                 view(am[p], iL + 1:iH + 1, iL:iH,     1:Nz)),
            (:y, view(bm[p], iL:iH,     iL    :iH,     1:Nz),
                 view(bm[p], iL:iH,     iL + 1:iH + 1, 1:Nz)),
            (:z, view(cm[p], iL:iH, iL:iH, 1    :Nz),
                 view(cm[p], iL:iH, iL:iH, 2:Nz + 1)),
        )
            m_int = view(m_p, iL:iH, iL:iH, 1:Nz)
            for k in 1:Nz, j in 1:Nc, i in 1:Nc
                mi = m_int[i, j, k]
                fl = F_lo_view[i, j, k]
                fh = F_hi_view[i, j, k]
                # Any non-finite cell mass, non-finite face flux, or
                # non-positive cell mass is an immediate contract violation
                # regardless of how the CFL ratio would round. NaN comparisons
                # all return `false` in Julia, so `mi <= 0` alone would let
                # `NaN`-mass cells slip through; `!isfinite` handles `NaN` and
                # `±Inf` consistently. Pin the report to the first such cell
                # encountered (subsequent ones cannot make `Inf` worse, and
                # `NaN > Inf` is false so the diagnostic stays stable).
                if !isfinite(mi) || mi <= zero(FT) ||
                   !isfinite(fl) || !isfinite(fh)
                    if !isinf(worst_ratio)
                        worst_ratio = Inf
                        worst_dir = dir
                        worst_loc = (p, i, j, k)
                    end
                    continue
                end
                outgoing = max(zero(FT), -fl) + max(zero(FT), fh)
                ratio = outgoing / mi
                if ratio > worst_ratio
                    worst_ratio = ratio
                    worst_dir = dir
                    worst_loc = (p, i, j, k)
                end
            end
        end
    end
    return (direction = worst_dir, ratio = Float64(worst_ratio),
            location = worst_loc, ok = worst_ratio <= cfl_limit)
end

"""
    verify_cs_window_contract!(m_cur, am, bm, cm, m_next, steps_per_window, win_idx;
                               replay_tol, positivity_cfl_limit = 0.95, halo_width = 0)

Single canonical per-window CS binary contract check. Runs the replay gate
(`verify_write_replay_cs!`, errors on failure) followed by the per-substep
positivity scan (`verify_substep_positivity_cs`, returns a diagnostic). Every
CS-producing preprocessor (spectral, regrid, GEOS-native) should call this so
no path can silently skip a gate.

Returns `(; replay, positivity)` with both diagnostics. Positivity is non-fatal
here — callers aggregate the worst window and pass it to
`summarize_cs_positivity_status` after the loop, where the run-level
`require_substep_positivity` policy decides whether to error or warn.
"""
function verify_cs_window_contract!(m_cur::NTuple{NP, <:AbstractArray{FT, 3}},
                                    am::NTuple{NP, <:AbstractArray},
                                    bm::NTuple{NP, <:AbstractArray},
                                    cm::NTuple{NP, <:AbstractArray},
                                    m_next::NTuple{NP, <:AbstractArray},
                                    steps_per_window::Int,
                                    win_idx::Int;
                                    replay_tol::Real,
                                    positivity_cfl_limit::Real = 0.95,
                                    halo_width::Integer = 0) where {FT, NP}
    replay = verify_write_replay_cs!(m_cur, am, bm, cm, m_next,
                                     steps_per_window, replay_tol, win_idx)
    positivity = verify_substep_positivity_cs!(m_cur, am, bm, cm;
                                               cfl_limit = positivity_cfl_limit,
                                               halo_width = halo_width)
    return (replay = replay, positivity = positivity)
end

"""
    init_cs_positivity_accumulator() -> NamedTuple

Zero-valued state for accumulating the worst per-window positivity diagnostic
across a preprocessing loop. Pair with `update_cs_positivity_accumulator!`.
"""
init_cs_positivity_accumulator() = (ratio = 0.0,
                                    direction = :none,
                                    win = 0,
                                    location = (0, 0, 0, 0))

"""
    update_cs_positivity_accumulator(worst, diag, win_idx) -> NamedTuple

Return an updated accumulator from a fresh per-window diagnostic.
"""
function update_cs_positivity_accumulator(worst::NamedTuple, diag::NamedTuple, win_idx::Int)
    diag.ratio > worst.ratio || return worst
    return (ratio = diag.ratio,
            direction = diag.direction === nothing ? :none : diag.direction,
            win = win_idx,
            location = diag.location)
end

"""
    summarize_cs_positivity_status(worst; cfl_limit, steps_per_window,
                                   require_substep_positivity = true,
                                   quarantine_path = nothing)

Post-loop summary helper. Logs the worst-window outcome, and if it exceeds
`cfl_limit`:
* deletes `quarantine_path` (if given) so a downstream consumer cannot pick up
  the half-written binary;
* errors when `require_substep_positivity = true`, otherwise warns.

The error message includes a recommended `steps_per_window` value that would
satisfy the gate, computed from the observed worst ratio.
"""
function summarize_cs_positivity_status(worst::NamedTuple;
                                        cfl_limit::Real,
                                        steps_per_window::Int,
                                        require_substep_positivity::Bool = true,
                                        quarantine_path::Union{Nothing, AbstractString} = nothing)
    msg = @sprintf("max outgoing/m=%.3f dir=%s win=%d cell=%s (limit=%.2f)",
                   worst.ratio, worst.direction, worst.win,
                   worst.location, cfl_limit)
    if worst.ratio <= cfl_limit
        @info "  Per-substep positivity gate: $msg"
        return nothing
    end

    # A useful integer step recommendation requires that (a) the observed
    # ratio is finite, (b) the configured `cfl_limit` is a sensible positive
    # finite divisor, and (c) `ceil(Int, ratio / cfl_limit) * steps_per_window`
    # fits in `Int`. Any of these failing — `Inf`/`NaN` ratio from m<=0 or
    # NaN-mass/flux, an invalid `cfl_limit = 0` that produces `Inf` from the
    # divide, or a finite-but-pathologically-large ratio like `1e308` — would
    # throw `InexactError` from `ceil(Int, ...)` BEFORE the intended
    # error/warn path runs, which would re-break the
    # `require_substep_positivity = false` escape hatch exactly when an
    # operator needs it most. Branch *before* the rescue arithmetic.
    max_safe_factor       = typemax(Int) ÷ max(steps_per_window, 1)
    useful_recommendation = isfinite(worst.ratio) &&
                             isfinite(cfl_limit) && cfl_limit > 0 &&
                             worst.ratio / cfl_limit <= max_safe_factor

    detail = if useful_recommendation
        recommended = max(steps_per_window,
                          ceil(Int, worst.ratio / cfl_limit) * steps_per_window)
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

# ===========================================================================
# Plan 41 P1 — typed CS contract concrete.
#
# `CubedSphereContract{FT}` wraps the existing CS gate state (policy fields
# + worst-window accumulator) in a typed nominal that participates in the
# unified `AbstractWindowContract{G, FT}` dispatch surface. The struct holds
# the policy at construction time (closes foot-gun A) and validates
# `positivity_cfl_limit` in its inner constructor so an invalid TOML value
# errors before any window runs.
#
# The trait-surface methods (`verify_window!`, `update_accumulator!`,
# `summarize_status!`) delegate to the existing NamedTuple-based helpers
# above so the per-window math is bit-exact identical to today's path —
# the typed surface is additive scaffolding for the P2 unified driver.
# ===========================================================================

"""
    CubedSphereContract{FT} <: AbstractWindowContract{CubedSphereTargetGeometry, FT}

Typed nominal owning a CS preprocessor's per-window gate policy and
worst-window positivity accumulator.

Fields:

  - `replay_tol::Float64` — relative replay tolerance.
  - `positivity_cfl_limit::Float64` — per-substep positivity CFL gate.
    Must satisfy `0 < limit ≤ 1`; validated at construction.
  - `require_substep_positivity::Bool` — whether `summarize_status!`
    errors (`true`) or warns (`false`) on a positivity violation.
  - `steps_per_window::Int` — for the recommended-steps message in the
    summary's escape-hatch detail. Must be ≥ 1.
  - `halo_width::Int` — passed through to `verify_substep_positivity_cs!`.
  - `worst::NamedTuple` — mutable accumulator (initially zero).

Construct with explicit kwargs; defaults match the CS round-2/round-3
production policy.
"""
mutable struct CubedSphereContract{FT} <:
                AbstractWindowContract{CubedSphereTargetGeometry, FT}
    replay_tol                  :: Float64
    positivity_cfl_limit        :: Float64
    require_substep_positivity  :: Bool
    steps_per_window            :: Int
    halo_width                  :: Int
    worst                       :: NamedTuple

    function CubedSphereContract{FT}(;
                                      replay_tol::Real,
                                      positivity_cfl_limit::Real = 0.95,
                                      require_substep_positivity::Bool = true,
                                      steps_per_window::Integer,
                                      halo_width::Integer = 0) where FT
        isfinite(positivity_cfl_limit) && 0 < positivity_cfl_limit ≤ 1 ||
            error("CubedSphereContract: positivity_cfl_limit = " *
                  "$(positivity_cfl_limit); must be in (0, 1].")
        steps_per_window ≥ 1 ||
            error("CubedSphereContract: steps_per_window = " *
                  "$(steps_per_window); must be ≥ 1.")
        halo_width ≥ 0 ||
            error("CubedSphereContract: halo_width = $(halo_width); " *
                  "must be ≥ 0.")
        return new(Float64(replay_tol),
                   Float64(positivity_cfl_limit),
                   require_substep_positivity,
                   Int(steps_per_window),
                   Int(halo_width),
                   init_cs_positivity_accumulator())
    end
end

# Convenience accessor shims so the abstract trait surface answers.
@inline contract_replay_tolerance(c::CubedSphereContract)    = c.replay_tol
@inline contract_cfl_limit(c::CubedSphereContract)           = c.positivity_cfl_limit
@inline contract_require_positivity(c::CubedSphereContract)  = c.require_substep_positivity

"""
    verify_window!(window, contract::CubedSphereContract, win_idx::Int)
        -> (; replay, positivity)

Run the per-window CS contract on a NamedTuple `window` with fields
`m_cur`, `am`, `bm`, `cm`, `m_next` (each a 6-tuple of panel arrays).
Delegates to `verify_cs_window_contract!`; the replay gate throws on
violation, the positivity gate is non-fatal here.
"""
function verify_window!(window, contract::CubedSphereContract, win_idx::Integer)
    return verify_cs_window_contract!(window.m_cur, window.am, window.bm,
                                       window.cm, window.m_next,
                                       contract.steps_per_window, Int(win_idx);
                                       replay_tol           = contract.replay_tol,
                                       positivity_cfl_limit = contract.positivity_cfl_limit,
                                       halo_width           = contract.halo_width)
end

"""
    update_accumulator!(contract::CubedSphereContract, positivity_diag, win_idx::Int)

Fold one window's positivity diagnostic into the CS contract's
worst-window accumulator. Mutates `contract.worst`.
"""
function update_accumulator!(contract::CubedSphereContract, positivity_diag,
                              win_idx::Integer)
    contract.worst = update_cs_positivity_accumulator(contract.worst,
                                                       positivity_diag,
                                                       Int(win_idx))
    return nothing
end

"""
    summarize_status!(contract::CubedSphereContract;
                       quarantine_path::Union{Nothing, AbstractString} = nothing)

Run the CS positivity post-loop summary using the contract's worst
accumulator and policy fields. May log, warn, or error depending on
the accumulator state and `require_substep_positivity`.
"""
function summarize_status!(contract::CubedSphereContract;
                            quarantine_path::Union{Nothing, AbstractString} = nothing)
    return summarize_cs_positivity_status(contract.worst;
                                           cfl_limit = contract.positivity_cfl_limit,
                                           steps_per_window = contract.steps_per_window,
                                           require_substep_positivity =
                                               contract.require_substep_positivity,
                                           quarantine_path = quarantine_path)
end
