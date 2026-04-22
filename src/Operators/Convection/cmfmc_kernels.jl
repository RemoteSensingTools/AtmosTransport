# ---------------------------------------------------------------------------
# CMFMCConvection kernel + inline helpers — plan 18 Commit 3.
#
# Ports from GEOS-Chem `convection_mod.F90:DO_RAS_CLOUD_CONVECTION`.
# Derivation + four-term-vs-two-term equivalence is captured in the
# plan-18 upstream notes; those were deleted during plan-21 cleanup,
# so reach them via git archaeology at commit 27e9d2e, path
# docs/plans/18_ConvectionPlan/18_CONVECTION_UPSTREAM_GCHP_NOTES.md.
# Medium cleanup from the earlier Julia port per Decision 15. Two
# deliberate departures from that legacy port:
#
#   1. **ADD well-mixed sub-cloud layer** (Decision 17) —
#      pressure-weighted below-cloud-base treatment from GCHP
#      convection_mod.F90:742-782. The legacy Julia port skipped this;
#      git commit ec2d2c0 preserves it at
#      src_legacy/Convection/ras_convection.jl for comparison.
#   2. **KEEP no positivity clamp** (Decision 11 — adjoint addendum §D).
#      Legacy already has no clamp (git commit ec2d2c0,
#      src_legacy/Convection/ras_convection.jl:208-214); plan 18
#      preserves this and adds a docstring pointer to the rationale.
#
# Convention (per CLAUDE.md Invariant 2): `k=1=TOA`, `k=Nz=surface`.
# CMFMC is stored at interfaces: `cmfmc[i, j, k]` = flux at the TOP
# of layer k (going UP), so `cmfmc[i, j, k+1]` = flux at the BOTTOM
# of layer k (from below). The pass directions reflect this:
#
#   Pass 1 (updraft, bottom-to-top): k = Nz down to 1, but in practice
#     only active between cloud base and cloud top.
#   Pass 2 (tendency, top-to-bottom): k = 1 up to Nz (our "top-down"
#     equals increasing k).
#
# No field type parameter on CMFMCConvection (plan 18 v5.1 §2.3
# Decision 20) — the operator is basis-polymorphic; the consumer
# contract is "CMFMC and DTRAIN must match state.air_mass basis".
#
# Imports come from the parent `Convection.jl` module; this file is
# `include`d into that module scope.
# ---------------------------------------------------------------------------

# =========================================================================
# Inline helpers (Decision 19 — dispatch-ready structure for future
# wet scavenging).
# =========================================================================

"""
    _cmfmc_updraft_mix(qc_below, q_env, cmfmc_below, entrn, cmout, tiny)
        -> (qc_post_mix, qc_scav)

Updraft mixing at one level: environment air (`q_env`) mixes with
updraft air from below (`qc_below`) in mass-weighted proportion.

# Inert-tracer version (plan 18)

Returns `(qc, zero(qc))` — `qc` is the post-mix concentration, the
scavenging fraction is identically zero. A future wet-deposition
plan adds a method that splits `qc` into `(qc_pres, qc_scav)` keyed
on a solubility trait parameter.

# Arguments

- `qc_below` — updraft concentration from the layer below (pre-mix).
- `q_env` — environment mixing ratio at the current layer.
- `cmfmc_below` — inflow mass flux from below [kg / m² / s].
- `entrn` — environment air entrained into the updraft
  [kg / m² / s], already capped at `cmout - cmfmc_below ≥ 0`.
- `cmout` — total outflow from the updraft [kg / m² / s].
- `tiny` — small-value threshold; below `cmout ≤ tiny` the formula
  is ill-conditioned and we fall back to `qc = q_env`.
"""
@inline function _cmfmc_updraft_mix(qc_below, q_env, cmfmc_below, entrn, cmout, tiny)
    if cmout > tiny
        qc = (cmfmc_below * qc_below + entrn * q_env) / cmout
    else
        qc = q_env
    end
    return qc, zero(qc)
end

"""
    _cmfmc_apply_tendency(q_env, q_above, qc_post_mix, cmfmc_above,
                          dtrain, bmass, dt)
        -> q_new

Apply one sub-step's tendency to the environment mixing ratio at the
current layer.

# Inert two-term form

The GCHP four-term tendency from `convection_mod.F90:DO_RAS_CLOUD_CONVECTION`
is algebraically equivalent to the two-term form below for inert tracers
(`QC_PRES = old_QC`). §5.3 of the plan-18 upstream notes works through
the simplification — see commit 27e9d2e,
`docs/plans/18_ConvectionPlan/18_CONVECTION_UPSTREAM_GCHP_NOTES.md`.

```
tsum = cmfmc_above * (q_above - q_env) + dtrain * (qc_post_mix - q_env)
q_new = q_env + (dt / bmass) * tsum
```

- `cmfmc_above * (q_above - q_env)` — compensating subsidence at the
  top interface: environment air from the layer above (q_above)
  descends to replace what the updraft removed.
- `dtrain * (qc_post_mix - q_env)` — in-cloud air at `qc_post_mix`
  detrained into the environment.

For the top-down tendency pass, `q_above` is the PRE-tendency value
at layer k-1 (saved by the caller before updating); simultaneous-
update semantics match GCHP.

A future wet-deposition plan adds a method on this helper accepting
`(qc_pres, qc_scav)` so the four-term form with scavenging is
restored without rewriting the kernel.

# Arguments

- `q_env` — environment mixing ratio at current layer (pre-update).
- `q_above` — environment mixing ratio at the layer above
  (PRE-tendency value).
- `qc_post_mix` — updraft mixing ratio at current layer (post-mix,
  from Pass 1).
- `cmfmc_above` — updraft mass flux at the TOP interface of the
  current layer (leaves the layer going up) [kg / m² / s].
- `dtrain` — detrainment from updraft to environment at the current
  layer [kg / m² / s].
- `bmass` — layer air mass per unit horizontal area [kg / m²].
- `dt` — sub-step length [s].

NO positivity clamp. Per plan 18 Decision 11 + adjoint addendum §D,
the kernel is linear in `q_env`, `q_above`, `qc_post_mix` — tiny
negativities are absorbed by the global mass fixer, not by a
nonlinear clamp that would break the adjoint-identity property.
"""
@inline function _cmfmc_apply_tendency(q_env, q_above, qc_post_mix,
                                        cmfmc_above, dtrain, bmass, dt)
    tsum = cmfmc_above * (q_above - q_env) +
           dtrain      * (qc_post_mix - q_env)
    return q_env + (dt / bmass) * tsum
end

"""
    _cmfmc_wellmix_subcloud!(q_view, delp_view, cldbase_k, q_cldbase,
                              cmfmc_at_cldbase, dt)

Pressure-weighted well-mixed treatment below cloud base.

Per GCHP `convection_mod.F90:742-782`. Plan 18 ADDS this treatment
relative to the earlier Julia port of RAS convection (git commit
ec2d2c0 contains `src_legacy/Convection/ras_convection.jl`, which
omits it — Decision 17 is a deliberate correction).

In our convention `k=1=TOA, k=Nz=surface`, "below cloud base" means
layers with `k > cldbase_k` (larger k = closer to surface). The
well-mixed treatment replaces the sub-cloud environment at layers
`cldbase_k+1 … Nz` with a single uniform mixing ratio `qc_mixed`
that combines a pressure-weighted average of the pre-step values
with an inflow contribution from the updraft flux at cloud base.

# Arguments

- `q_view` — view of the tracer-mass column at `(i, j, :)`, indexed
  by `k = 1..Nz`. Modified in place.
- `air_mass_view` — view of the air-mass column at `(i, j, :)`,
  indexed by `k = 1..Nz`. Read for mass-weighting.
- `cldbase_k` — cloud base level (smallest `k ∈ 1..Nz` with
  `cmfmc > tiny`; for our convention, the cloud is AT layer
  `cldbase_k`, and sub-cloud layers are `cldbase_k+1..Nz`).
- `q_cldbase` — environment mixing ratio at the cloud-base layer
  (`k = cldbase_k`), pre-step.
- `cmfmc_at_cldbase` — updraft flux entering the cloud base from
  below (the bottom-of-cloud-base-layer interface, i.e.
  `cmfmc[:, :, cldbase_k + 1]`) [kg / m² / s].
- `dt` — sub-step length [s].

# Formula

```
qb = (Σ_{k in sub} q_k · m_k) / Σ m_k           # pressure-weighted avg
mb = Σ m_k                                        # total sub-cloud mass
qc_mixed = (mb · qb + cmfmc_at_cldbase · q_cldbase · dt)
           / (mb + cmfmc_at_cldbase · dt)
q_k ← qc_mixed                                    # apply uniformly, all k > cldbase_k
```

Caller is responsible for guard conditions (cldbase_k < Nz, non-zero
sub-cloud thickness). No-op if `cldbase_k >= Nz` (no layers below).
"""
@inline function _cmfmc_wellmix_subcloud!(q_view, air_mass_view, cldbase_k,
                                           q_cldbase, cmfmc_at_cldbase, dt)
    cldbase_k >= length(q_view) && return nothing   # no sub-cloud layers
    FT = eltype(q_view)

    qb_num = zero(FT)
    mb     = zero(FT)
    @inbounds for k in (cldbase_k + 1):length(q_view)
        m_k = air_mass_view[k]
        q_k = m_k > FT(1e-30) ? q_view[k] / m_k : zero(FT)
        qb_num += q_k * m_k
        mb     += m_k
    end

    mb > zero(FT) || return nothing

    qb = qb_num / mb
    qc_mixed = (mb * qb + cmfmc_at_cldbase * q_cldbase * dt) /
               (mb + cmfmc_at_cldbase * dt)

    @inbounds for k in (cldbase_k + 1):length(q_view)
        q_view[k] = qc_mixed * air_mass_view[k]
    end
    return nothing
end

# =========================================================================
# CFL sub-cycling
# =========================================================================

"""
    _cmfmc_max_cfl(cmfmc, air_mass, cell_areas_y, dt) -> FT

Scan one window's CMFMC field and return the grid-maximum
`|cmfmc| · dt / bmass` ratio. `bmass = air_mass[i,j,k] / cell_area_y[j]`
has units kg/m², and CMFMC has units kg/m²/s, so the ratio is
dimensionless.

Returns the same floating-point type as the state (`FT`). The
convection path stays type-stable end to end; if `Float32` needs
better accumulation behavior, that should be handled explicitly in the
relevant reduction rather than by promoting the whole CFL scan.
"""
function _cmfmc_max_cfl(cmfmc::AbstractArray{FT, 3},
                        air_mass::AbstractArray{FT, 3},
                        cell_areas_y::AbstractVector,
                        dt::Real) where FT
    dt_ft = FT(dt)
    worst = zero(FT)
    Nx, Ny, Nz = size(air_mass)
    # cmfmc is (Nx, Ny, Nz+1) at interfaces
    @inbounds for k_iface in 1:Nz + 1, j in 1:Ny, i in 1:Nx
        # The relevant bmass for the interface sits adjacent to it.
        # For an interface with layers on both sides, we pessimize
        # against the thinner layer (smaller bmass → larger CFL).
        if k_iface == 1
            m_cell = air_mass[i, j, 1]
        elseif k_iface > Nz
            m_cell = air_mass[i, j, Nz]
        else
            m_cell = min(air_mass[i, j, k_iface - 1], air_mass[i, j, k_iface])
        end
        bmass = m_cell / cell_areas_y[j]
        bmass > zero(FT) || continue
        ratio = abs(cmfmc[i, j, k_iface]) * dt_ft / bmass
        worst = max(worst, ratio)
    end
    return worst
end

function _cmfmc_max_cfl(cmfmc::AbstractArray{FT, 2},
                        air_mass::AbstractMatrix{FT},
                        cell_areas::AbstractVector,
                        dt::Real) where FT
    dt_ft = FT(dt)
    worst = zero(FT)
    ncell, Nz = size(air_mass)
    @inbounds for k_iface in 1:(Nz + 1), c in 1:ncell
        if k_iface == 1
            m_cell = air_mass[c, 1]
        elseif k_iface > Nz
            m_cell = air_mass[c, Nz]
        else
            m_cell = min(air_mass[c, k_iface - 1], air_mass[c, k_iface])
        end
        bmass = m_cell / cell_areas[c]
        bmass > zero(FT) || continue
        ratio = abs(cmfmc[c, k_iface]) * dt_ft / bmass
        worst = max(worst, ratio)
    end
    return worst
end

function _cmfmc_max_cfl(cmfmc::NTuple{6, <:AbstractArray{FT, 3}},
                        air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                        cell_areas::NTuple{6, <:AbstractMatrix},
                        dt::Real) where FT
    dt_ft = FT(dt)
    worst = zero(FT)

    @inbounds for p in 1:6
        cmfmc_panel = cmfmc[p]
        air_panel = air_mass[p]
        area_panel = cell_areas[p]
        Nc_x, Nc_y = size(area_panel)
        Hp_x = div(size(air_panel, 1) - Nc_x, 2)
        Hp_y = div(size(air_panel, 2) - Nc_y, 2)
        Hp_x == Hp_y || throw(ArgumentError(
            "Cubed-sphere CMFMC air-mass halos must be symmetric; got ($(Hp_x), $(Hp_y))"))
        Nz = size(air_panel, 3)

        for k_iface in 1:(Nz + 1), j in 1:Nc_y, i in 1:Nc_x
            ii = i + Hp_x
            jj = j + Hp_y
            if k_iface == 1
                m_cell = air_panel[ii, jj, 1]
            elseif k_iface > Nz
                m_cell = air_panel[ii, jj, Nz]
            else
                m_cell = min(air_panel[ii, jj, k_iface - 1], air_panel[ii, jj, k_iface])
            end
            bmass = m_cell / area_panel[i, j]
            bmass > zero(FT) || continue
            ratio = abs(cmfmc_panel[i, j, k_iface]) * dt_ft / bmass
            worst = max(worst, ratio)
        end
    end

    return worst
end

"""
    _get_or_compute_n_sub!(ws, cmfmc, air_mass, cell_metrics, dt) -> Int

Return the cached CFL sub-step count, recomputing from the CMFMC
field if the cache is stale (first call after a window advance).

CFL rule (plan 18 v5.1 §2.8 Decision 21):

    n_sub = max(1, ceil(max_over_grid(cmfmc · dt / bmass) / cfl_safety))

with `cfl_safety = 0.5`. Cached on `ws.cached_n_sub[]` alongside
`ws.cache_valid[]`; `invalidate_cmfmc_cache!(ws)` sets the sentinel
false so the next call re-scans.
"""
# Safety ceiling for the CFL-derived sub-step count. If met data is
# pathologically inconsistent (e.g. `cmfmc` in kg/m²/s but `air_mass`
# accidentally in units that make `bmass` tiny — a common unit-scale
# bug), the naive formula can demand millions of sub-steps and make
# the runtime appear to hang. Cap at a production-reasonable 1024
# (typical CATRINE dt=1800s produces 5-15 sub-steps in deep
# convection) and error with an actionable message above that.
const _CMFMC_N_SUB_MAX = 1024

function _get_or_compute_n_sub!(ws::CMFMCWorkspace,
                                 cmfmc,
                                 air_mass,
                                 cell_metrics,
                                 dt::Real)
    if !ws.cache_valid[]
        worst = _cmfmc_max_cfl(cmfmc, air_mass, cell_metrics, dt)
        cfl_safety = typeof(worst)(0.5)
        n_sub = max(1, ceil(Int, worst / cfl_safety))
        if n_sub > _CMFMC_N_SUB_MAX
            throw(ArgumentError(
                "CMFMCConvection CFL sub-step count $(n_sub) exceeds " *
                "safety ceiling $(_CMFMC_N_SUB_MAX). Worst local " *
                "cmfmc·dt/bmass ratio = $(worst). Check that " *
                "`forcing.cmfmc` is in kg/m²/s on the same basis as " *
                "`state.air_mass`, and that `air_mass` is in kg per " *
                "cell (NOT kg/m²). Use a smaller `dt` if the ratio " *
                "is physically realistic (sustained CFL > $(cfl_safety * _CMFMC_N_SUB_MAX) is unusual)."
            ))
        end
        ws.cached_n_sub[] = n_sub
        ws.cache_valid[] = true
    end
    return ws.cached_n_sub[]
end

@kernel function _cmfmc_cs_panel_column_kernel!(
    tracers_raw,                 # (Nc+2Hp, Nc+2Hp, Nz, Nt), modified in place
    @Const(air_mass),            # (Nc+2Hp, Nc+2Hp, Nz)
    @Const(cmfmc),               # (Nc, Nc, Nz+1) at interfaces
    @Const(dtrain),              # (Nc, Nc, Nz) at centers
    @Const(cell_areas),          # (Nc, Nc)
    qc_scratch,                  # (Nc+2Hp, Nc+2Hp, Nz) — workspace
    Nz::Int,
    Nt::Int,
    dt,
    Hp::Int,
    ::Val{has_dtrain}
) where has_dtrain
    i, j = @index(Global, NTuple)

    FT = eltype(tracers_raw)
    tiny = FT(1e-30)
    ii = i + Hp
    jj = j + Hp
    cell_area = cell_areas[i, j]

    @inbounds for t_idx in 1:Nt
        cldbase_k = 0
        for k in 1:Nz
            cmfmc_bot_k = cmfmc[i, j, k + 1]
            if abs(cmfmc_bot_k) > tiny
                cldbase_k = k
                break
            end
        end

        if cldbase_k == 0
            continue
        end

        if cldbase_k < Nz
            m_cb = air_mass[ii, jj, cldbase_k]
            q_cldbase = m_cb > tiny ? tracers_raw[ii, jj, cldbase_k, t_idx] / m_cb : zero(FT)
            cmfmc_at_cldbase = cmfmc[i, j, cldbase_k + 1]
            if cmfmc_at_cldbase > tiny
                qb_num = zero(FT)
                mb = zero(FT)
                for k in (cldbase_k + 1):Nz
                    m_k = air_mass[ii, jj, k]
                    q_k = m_k > tiny ? tracers_raw[ii, jj, k, t_idx] / m_k : zero(FT)
                    qb_num += q_k * m_k
                    mb += m_k
                end
                if mb > zero(FT)
                    qb = qb_num / mb
                    qc_mixed = (mb * qb + cmfmc_at_cldbase * q_cldbase * dt) /
                               (mb + cmfmc_at_cldbase * dt)
                    for k in (cldbase_k + 1):Nz
                        tracers_raw[ii, jj, k, t_idx] = qc_mixed * air_mass[ii, jj, k]
                    end
                end
            end
        end

        qc_below = zero(FT)

        for k in Nz:-1:1
            m_k = air_mass[ii, jj, k]
            q_k = m_k > tiny ? tracers_raw[ii, jj, k, t_idx] / m_k : zero(FT)

            cmfmc_bot = k < Nz ? cmfmc[i, j, k + 1] : zero(FT)
            cmfmc_top = cmfmc[i, j, k]
            dtrain_k = has_dtrain ? dtrain[i, j, k] : zero(FT)

            cmout = cmfmc_top + dtrain_k
            cmfmc_bot_eff = min(cmfmc_bot, cmout)
            entrn = cmout - cmfmc_bot_eff

            qc, _qc_scav = _cmfmc_updraft_mix(qc_below, q_k,
                                              cmfmc_bot_eff, entrn, cmout, tiny)
            qc_scratch[ii, jj, k] = qc
            qc_below = qc
        end

        q_env_prev = zero(FT)

        for k in 1:Nz
            m_k = air_mass[ii, jj, k]
            q_k = m_k > tiny ? tracers_raw[ii, jj, k, t_idx] / m_k : zero(FT)

            bmass = m_k / cell_area
            cmfmc_top = cmfmc[i, j, k]
            dtrain_k = has_dtrain ? dtrain[i, j, k] : zero(FT)
            qc_post = qc_scratch[ii, jj, k]

            if k > 1 && bmass > tiny
                q_new = _cmfmc_apply_tendency(q_k, q_env_prev, qc_post,
                                              cmfmc_top, dtrain_k, bmass, dt)
            elseif bmass > tiny
                q_new = _cmfmc_apply_tendency(q_k, q_k, qc_post,
                                              zero(FT), dtrain_k, bmass, dt)
            else
                q_new = q_k
            end

            q_env_prev = q_k
            tracers_raw[ii, jj, k, t_idx] = q_new * m_k
        end
    end
end

# =========================================================================
# Main kernel — one thread per (i, j) column.
# =========================================================================

@kernel function _cmfmc_column_kernel!(
    tracers_raw,                 # (Nx, Ny, Nz, Nt), modified in place
    @Const(air_mass),            # (Nx, Ny, Nz)
    @Const(cmfmc),               # (Nx, Ny, Nz+1) at interfaces
    @Const(dtrain),              # (Nx, Ny, Nz) at centers (may be zeros for Tiedtke fallback)
    @Const(cell_areas_y),        # (Ny,)
    qc_scratch,                  # (Nx, Ny, Nz) — workspace
    Nz::Int,
    Nt::Int,
    dt,
    ::Val{has_dtrain}           # compile-time branch for Tiedtke fallback
) where has_dtrain
    i, j = @index(Global, NTuple)

    FT = eltype(tracers_raw)
    tiny = FT(1e-30)
    cell_area_j = cell_areas_y[j]

    @inbounds for t_idx in 1:Nt

        # ── Pass 0: cloud-base detection (plan 18 addition per Decision 17) ──
        # Smallest k with |cmfmc[k+1]| > tiny — this is the level that
        # first has updraft inflow from below.
        cldbase_k = 0
        for k in 1:Nz
            cmfmc_bot_k = cmfmc[i, j, k + 1]
            if abs(cmfmc_bot_k) > tiny
                cldbase_k = k
                break
            end
        end

        if cldbase_k == 0
            # No active convection in this column — nothing to do.
            continue
        end

        # ── Well-mixed sub-cloud layer (GCHP :742-782, missing in legacy) ──
        # Before Pass 1, uniformise the environment below cloud base so
        # the updraft entrains a well-mixed column.
        if cldbase_k < Nz
            m_cb = air_mass[i, j, cldbase_k]
            q_cldbase = m_cb > tiny ?
                tracers_raw[i, j, cldbase_k, t_idx] / m_cb : zero(FT)
            cmfmc_at_cldbase = cmfmc[i, j, cldbase_k + 1]
            if cmfmc_at_cldbase > tiny
                # Mirror of _cmfmc_wellmix_subcloud! inlined here because
                # we can't take views inside a KA @kernel body cleanly.
                qb_num = zero(FT); mb = zero(FT)
                for k in (cldbase_k + 1):Nz
                    m_k = air_mass[i, j, k]
                    q_k = m_k > tiny ? tracers_raw[i, j, k, t_idx] / m_k : zero(FT)
                    qb_num += q_k * m_k
                    mb     += m_k
                end
                if mb > zero(FT)
                    qb = qb_num / mb
                    qc_mixed = (mb * qb + cmfmc_at_cldbase * q_cldbase * dt) /
                               (mb + cmfmc_at_cldbase * dt)
                    for k in (cldbase_k + 1):Nz
                        tracers_raw[i, j, k, t_idx] = qc_mixed * air_mass[i, j, k]
                    end
                end
            end
        end

        # ── Pass 1: updraft concentration, bottom-to-top (Nz → 1) ──
        # The updraft rises from the surface upward. In our convention,
        # "rising" = decreasing k. At the base (k=Nz), no updraft from
        # below, so we start with qc_below = 0.
        qc_below = zero(FT)

        for k in Nz:-1:1
            m_k = air_mass[i, j, k]
            q_k = m_k > tiny ? tracers_raw[i, j, k, t_idx] / m_k : zero(FT)

            cmfmc_bot = k < Nz ? cmfmc[i, j, k + 1] : zero(FT)   # from below
            cmfmc_top = cmfmc[i, j, k]                            # going up
            dtrain_k  = has_dtrain ? dtrain[i, j, k] : zero(FT)

            cmout = cmfmc_top + dtrain_k
            # Cap inflow at outflow — safety against inconsistent met data
            # (no positivity clamp; this is a pre-kernel consistency
            # projection that keeps `entrn ≥ 0` without breaking
            # linearity in q).
            cmfmc_bot_eff = min(cmfmc_bot, cmout)
            entrn = cmout - cmfmc_bot_eff

            qc, _qc_scav = _cmfmc_updraft_mix(qc_below, q_k,
                                               cmfmc_bot_eff, entrn, cmout, tiny)
            qc_scratch[i, j, k] = qc
            qc_below = qc
        end

        # ── Pass 2: environment tendency, top-to-bottom (1 → Nz) ──
        # In our convention, "top-to-bottom" = increasing k. Subsidence
        # uses the PRE-tendency q at layer k-1 — saved in q_env_prev
        # before the layer-k update.
        q_env_prev = zero(FT)

        for k in 1:Nz
            m_k = air_mass[i, j, k]
            q_k = m_k > tiny ? tracers_raw[i, j, k, t_idx] / m_k : zero(FT)

            bmass = m_k / cell_area_j
            cmfmc_top = cmfmc[i, j, k]    # flux out the top, going up
            dtrain_k  = has_dtrain ? dtrain[i, j, k] : zero(FT)
            qc_post   = qc_scratch[i, j, k]

            if k > 1 && bmass > tiny
                q_new = _cmfmc_apply_tendency(q_k, q_env_prev, qc_post,
                                               cmfmc_top, dtrain_k, bmass, dt)
            elseif bmass > tiny
                # Top layer (k=1): no q_above — the subsidence term
                # reduces to zero (legacy `if k > 1` guard at :188-189).
                q_new = _cmfmc_apply_tendency(q_k, q_k, qc_post,
                                               zero(FT), dtrain_k, bmass, dt)
            else
                q_new = q_k
            end

            q_env_prev = q_k     # save PRE-update for next level's subsidence
            tracers_raw[i, j, k, t_idx] = q_new * m_k
        end
    end
end

@kernel function _cmfmc_faceindexed_column_kernel!(
    tracers_raw,                 # (ncells, Nz, Nt), modified in place
    @Const(air_mass),            # (ncells, Nz)
    @Const(cmfmc),               # (ncells, Nz+1) at interfaces
    @Const(dtrain),              # (ncells, Nz) at centers
    @Const(cell_areas),          # (ncells,)
    qc_scratch,                  # (ncells, Nz) — workspace
    Nz::Int,
    Nt::Int,
    dt,
    ::Val{has_dtrain}
) where has_dtrain
    c = @index(Global, Linear)

    FT = eltype(tracers_raw)
    tiny = FT(1e-30)
    cell_area = cell_areas[c]

    @inbounds for t_idx in 1:Nt
        cldbase_k = 0
        for k in 1:Nz
            cmfmc_bot_k = cmfmc[c, k + 1]
            if abs(cmfmc_bot_k) > tiny
                cldbase_k = k
                break
            end
        end

        if cldbase_k == 0
            continue
        end

        if cldbase_k < Nz
            m_cb = air_mass[c, cldbase_k]
            q_cldbase = m_cb > tiny ? tracers_raw[c, cldbase_k, t_idx] / m_cb : zero(FT)
            cmfmc_at_cldbase = cmfmc[c, cldbase_k + 1]
            if cmfmc_at_cldbase > tiny
                qb_num = zero(FT)
                mb = zero(FT)
                for k in (cldbase_k + 1):Nz
                    m_k = air_mass[c, k]
                    q_k = m_k > tiny ? tracers_raw[c, k, t_idx] / m_k : zero(FT)
                    qb_num += q_k * m_k
                    mb += m_k
                end
                if mb > zero(FT)
                    qb = qb_num / mb
                    qc_mixed = (mb * qb + cmfmc_at_cldbase * q_cldbase * dt) /
                               (mb + cmfmc_at_cldbase * dt)
                    for k in (cldbase_k + 1):Nz
                        tracers_raw[c, k, t_idx] = qc_mixed * air_mass[c, k]
                    end
                end
            end
        end

        qc_below = zero(FT)

        for k in Nz:-1:1
            m_k = air_mass[c, k]
            q_k = m_k > tiny ? tracers_raw[c, k, t_idx] / m_k : zero(FT)

            cmfmc_bot = k < Nz ? cmfmc[c, k + 1] : zero(FT)
            cmfmc_top = cmfmc[c, k]
            dtrain_k = has_dtrain ? dtrain[c, k] : zero(FT)

            cmout = cmfmc_top + dtrain_k
            cmfmc_bot_eff = min(cmfmc_bot, cmout)
            entrn = cmout - cmfmc_bot_eff

            qc, _qc_scav = _cmfmc_updraft_mix(qc_below, q_k,
                                              cmfmc_bot_eff, entrn, cmout, tiny)
            qc_scratch[c, k] = qc
            qc_below = qc
        end

        q_env_prev = zero(FT)

        for k in 1:Nz
            m_k = air_mass[c, k]
            q_k = m_k > tiny ? tracers_raw[c, k, t_idx] / m_k : zero(FT)

            bmass = m_k / cell_area
            cmfmc_top = cmfmc[c, k]
            dtrain_k = has_dtrain ? dtrain[c, k] : zero(FT)
            qc_post = qc_scratch[c, k]

            if k > 1 && bmass > tiny
                q_new = _cmfmc_apply_tendency(q_k, q_env_prev, qc_post,
                                              cmfmc_top, dtrain_k, bmass, dt)
            elseif bmass > tiny
                q_new = _cmfmc_apply_tendency(q_k, q_k, qc_post,
                                              zero(FT), dtrain_k, bmass, dt)
            else
                q_new = q_k
            end

            q_env_prev = q_k
            tracers_raw[c, k, t_idx] = q_new * m_k
        end
    end
end
