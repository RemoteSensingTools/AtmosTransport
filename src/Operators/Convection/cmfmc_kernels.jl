# ---------------------------------------------------------------------------
# CMFMCConvection kernel + inline helpers.
#
# Ports from GEOS-Chem `convection_mod.F90:DO_RAS_CLOUD_CONVECTION`.
# Two deliberate departures from the earlier Julia port:
#
#   1. **ADD well-mixed sub-cloud layer** — pressure-weighted
#      below-cloud-base treatment from GCHP
#      convection_mod.F90:742-782. The legacy Julia port skipped this;
#      git commit ec2d2c0 preserves it at
#      src_legacy/Convection/ras_convection.jl for comparison.
#   2. **KEEP no positivity clamp**. Legacy already has no clamp
#      (git commit ec2d2c0, src_legacy/Convection/ras_convection.jl:208-214);
#      preserving linearity is important for the adjoint path.
#
# Convention: `k=1=TOA`, `k=Nz=surface`.
# CMFMC is stored at interfaces: `cmfmc[i, j, k]` = flux at the TOP
# of layer k (going UP), so `cmfmc[i, j, k+1]` = flux at the BOTTOM
# of layer k (from below). The pass directions reflect this:
#
#   Pass 1 (updraft, bottom-to-top): k = Nz down to 1, but in practice
#     only active between cloud base and cloud top.
#   Pass 2 (tendency, top-to-bottom): k = 1 up to Nz (our "top-down"
#     equals increasing k).
#
# No field type parameter on CMFMCConvection: the operator is
# basis-polymorphic; the consumer contract is "CMFMC and DTRAIN must
# match state.air_mass basis".
#
# Imports come from the parent `Convection.jl` module; this file is
# `include`d into that module scope.
# ---------------------------------------------------------------------------

# =========================================================================
# Numerical tiny — type-dispatched "treat as zero" threshold for
# cmfmc/dtrain comparisons and column-mass guards. Two requirements:
#
#   (a) ABOVE the type's representation noise for typical cmfmc
#       magnitudes (`eps(FT) × max_scale` where max_scale ~ 1 kg/m²/s
#       in storage), so noise in a Float32 binary cannot spuriously
#       activate cloud-base detection or Pass 0.
#   (b) BELOW the smallest physically-meaningful cmfmc magnitude
#       (~1e-6 kg/m²/s for very weak convection), so real signals
#       are never silently dropped.
#
# That gives a target window (eps(FT) × ~1, 1e-6):
#
#   - Float32: `1f-6`  — about 8 × `eps(Float32) ≈ 1.19e-7`, at the
#     top of the safe window. Real GEOS-IT CMFMC values that fall
#     below this are essentially indistinguishable from Float32 round-off
#     in the binary anyway.
#   - Float64: `1e-14` — about 45 × `eps(Float64) ≈ 2.22e-16`, well
#     above noise and well below physical signal.
#
# Previously the kernels used `FT(1e-30)` (which on Float32 sat
# ~1e-23× BELOW `eps(Float32)` — i.e. inside the noise band and
# at risk of spurious activation on Float32 binaries). Centralising
# the constant here also makes it easy to retune without scanning
# three kernels.
# =========================================================================

@inline _cmfmc_tiny(::Type{Float32}) = 1f-6
@inline _cmfmc_tiny(::Type{Float64}) = 1e-14
@inline _cmfmc_tiny(::Type{T}) where {T <: AbstractFloat} = T(1e-14)

# =========================================================================
# Inline helpers, dispatch-ready for future wet scavenging.
# =========================================================================

"""
    _cmfmc_updraft_mix(qc_below, q_env, cmfmc_below, entrn, cmout, tiny)
        -> (qc_post_mix, qc_scav)

Updraft mixing at one level: environment air (`q_env`) mixes with
updraft air from below (`qc_below`) in mass-weighted proportion.

# Inert-tracer version

Returns `(qc, zero(qc))` — `qc` is the post-mix concentration, the
scavenging fraction is identically zero. A future wet-deposition
plan adds a method that splits `qc` into `(qc_pres, qc_scav)` keyed
on a solubility trait parameter.

# Arguments

- `qc_below` — updraft concentration from the layer below (pre-mix).
- `q_env` — environment mixing ratio at the current layer.
- `cmfmc_below` — inflow mass flux from below [kg / m² / s].
- `entrn` — environment air entrained into the updraft
  [kg / m² / s]. After the post-2026-05-24 audit (C3) the caller
  guards `entrn ≥ 0 && cmout > tiny` and falls back to
  `qc = qc_below` when the guard fails; this helper therefore only
  runs in the well-formed regime where `entrn ≥ 0` is guaranteed.
- `cmout` — total outflow from the updraft [kg / m² / s].
- `tiny` — small-value threshold; in the guarded calling pattern
  `cmout > tiny` is enforced by the caller, but the helper keeps
  the `cmout ≤ tiny → qc = q_env` fall-through for direct callers
  (e.g. unit tests that exercise the helper outside the kernel).
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
(`QC_PRES = old_QC`).

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

NO positivity clamp. The kernel is linear in `q_env`, `q_above`,
`qc_post_mix` — tiny
negativities are absorbed by the global mass fixer, not by a
nonlinear clamp that would break the adjoint-identity property.
"""
@inline function _cmfmc_apply_tendency(q_env, q_above, qc_post_mix,
                                        cmfmc_above, dtrain, bmass, dt)
    tsum = cmfmc_above * (q_above - q_env) +
           dtrain      * (qc_post_mix - q_env)
    return q_env + (dt / bmass) * tsum
end

# =========================================================================
# CFL sub-cycling
# =========================================================================

@inline _cmfmc_host_scan_array(a::Array) = a
@inline _cmfmc_host_scan_array(a) = Array(a)

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
    if !(cmfmc isa Array) || !(air_mass isa Array) || !(cell_areas_y isa Array)
        return _cmfmc_max_cfl(_cmfmc_host_scan_array(cmfmc),
                              _cmfmc_host_scan_array(air_mass),
                              _cmfmc_host_scan_array(cell_areas_y),
                              dt)
    end
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
        bmass = m_cell / FT(cell_areas_y[j])
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
    if !(cmfmc isa Array) || !(air_mass isa Array) || !(cell_areas isa Array)
        return _cmfmc_max_cfl(_cmfmc_host_scan_array(cmfmc),
                              _cmfmc_host_scan_array(air_mass),
                              _cmfmc_host_scan_array(cell_areas),
                              dt)
    end
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
        bmass = m_cell / FT(cell_areas[c])
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
    if any(p -> !(cmfmc[p] isa Array), 1:6) ||
       any(p -> !(air_mass[p] isa Array), 1:6) ||
       any(p -> !(cell_areas[p] isa Array), 1:6)
        return _cmfmc_max_cfl(map(_cmfmc_host_scan_array, cmfmc),
                              map(_cmfmc_host_scan_array, air_mass),
                              map(_cmfmc_host_scan_array, cell_areas),
                              dt)
    end
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
            bmass = m_cell / FT(area_panel[i, j])
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

CFL rule:

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
    tiny = _cmfmc_tiny(FT)
    ii = i + Hp
    jj = j + Hp
    cell_area = FT(cell_areas[i, j])

    @inbounds for t_idx in 1:Nt
        # Cloud base = largest k with `|cmfmc[k+1]| > tiny` (lowest
        # altitude with non-zero updraft inflow). See LL kernel above
        # for the GCHP convention reference (convection_mod.F90:625).
        cldbase_k = 0
        for k in Nz:-1:1
            cmfmc_bot_k = cmfmc[i, j, k + 1]
            if abs(cmfmc_bot_k) > tiny
                cldbase_k = k
                break
            end
        end

        if cldbase_k == 0
            continue
        end

        # Well-mixed sub-cloud, kg/m² accumulator + column-closing
        # cloud-base update — see LL kernel for the full derivation.
        if cldbase_k < Nz
            m_cb = air_mass[ii, jj, cldbase_k]
            q_cldbase = m_cb > tiny ? tracers_raw[ii, jj, cldbase_k, t_idx] / m_cb : zero(FT)
            cmfmc_at_cldbase = cmfmc[i, j, cldbase_k + 1]
            if cmfmc_at_cldbase > tiny
                qb_num     = zero(FT); qb_comp    = zero(FT)
                mb_pa      = zero(FT); mb_pa_comp = zero(FT)
                for k in (cldbase_k + 1):Nz
                    m_k = air_mass[ii, jj, k]
                    q_k = m_k > tiny ? tracers_raw[ii, jj, k, t_idx] / m_k : zero(FT)
                    m_k_pa = m_k / cell_area
                    qb_num, qb_comp    = _kahan_add(qb_num, qb_comp, q_k * m_k_pa)
                    mb_pa,  mb_pa_comp = _kahan_add(mb_pa,  mb_pa_comp, m_k_pa)
                end
                if mb_pa > zero(FT)
                    qb = qb_num / mb_pa
                    qc_mixed = (mb_pa * qb + cmfmc_at_cldbase * q_cldbase * dt) /
                               (mb_pa + cmfmc_at_cldbase * dt)
                    for k in (cldbase_k + 1):Nz
                        tracers_raw[ii, jj, k, t_idx] = qc_mixed * air_mass[ii, jj, k]
                    end
                    m_cb_pa = m_cb / cell_area
                    if m_cb_pa > tiny
                        q_cldbase_new = q_cldbase +
                            cmfmc_at_cldbase * dt * (qc_mixed - q_cldbase) / m_cb_pa
                        tracers_raw[ii, jj, cldbase_k, t_idx] = q_cldbase_new * m_cb
                    end
                end
            end
        end

        # Pass 1: GCHP-style guard — see LL kernel above for the
        # `entrn ≥ 0 .and. cmout > tiny` rationale and the
        # deliberate omission of the non-conservative `Q+DELQ<0` clamp.
        qc_below = zero(FT)

        for k in Nz:-1:1
            m_k = air_mass[ii, jj, k]
            q_k = m_k > tiny ? tracers_raw[ii, jj, k, t_idx] / m_k : zero(FT)

            cmfmc_bot = k < Nz ? cmfmc[i, j, k + 1] : zero(FT)
            cmfmc_top = cmfmc[i, j, k]
            dtrain_k = has_dtrain ? dtrain[i, j, k] : zero(FT)

            cmout = cmfmc_top + dtrain_k
            entrn = cmout - cmfmc_bot

            if entrn >= zero(FT) && cmout > tiny
                qc, _qc_scav = _cmfmc_updraft_mix(qc_below, q_k,
                                                  cmfmc_bot, entrn, cmout, tiny)
            else
                qc = qc_below
            end
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
    tiny = _cmfmc_tiny(FT)
    cell_area_j = FT(cell_areas_y[j])

    @inbounds for t_idx in 1:Nt

        # ── Pass 0: cloud-base detection ──
        # Cloud base = lowest altitude with non-zero updraft inflow.
        # In our TOA-first orientation (k=1=TOA, k=Nz=surface) that is
        # the LARGEST k with `|cmfmc[k+1]| > tiny`. Scan k=Nz → 1
        # (surface upward in altitude) and take the first hit.
        # This mirrors GCHP `convection_mod.F90:625`, where the
        # surface-up scan `DO K = 1, NLAY` selects the lowest level
        # with non-zero forcing as the cloud base.
        cldbase_k = 0
        for k in Nz:-1:1
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

        # ── Well-mixed sub-cloud layer (GCHP convection_mod.F90:742-782) ──
        # Before Pass 1, uniformise the environment below cloud base so
        # the updraft entrains a well-mixed column. "Below cloud base"
        # = larger k in our orientation = layers (cldbase_k+1):Nz.
        #
        # The mass-weighted mixing formula needs `mb` and `cmfmc·dt` in
        # the SAME units (kg/m²) so the two terms in the denominator
        # are commensurable. We accumulate `mb_pa` in kg/m² by
        # dividing each layer's per-cell mass by `cell_area_j`.
        #
        # Deliberate improvement over GCHP: GCHP's step leaves
        # `Q(CLDBASE)` unchanged, so the "extra mass" implicit in
        # `(mb_pa + cmfmc·dt)` is never debited to the cloud-base
        # layer — column tracer mass drifts by `cmfmc·dt·(q_new -
        # q_cldbase)·cell_area` per call. GCHP relies on its dynamics
        # core to absorb that residual; our convection-only operator
        # has no such absorber. We therefore close the budget locally
        # by updating `Q(CLDBASE) += cmfmc·dt·(qc_mixed - q_cldbase) /
        # m_cb_pa`, which by construction makes Pass 0 strictly
        # mass-conserving. The downstream Pass 1 entrainment then
        # reads the updated `q_cldbase`, which is the physically
        # correct value of the well-mixed air entering the updraft.
        if cldbase_k < Nz
            m_cb = air_mass[i, j, cldbase_k]
            q_cldbase = m_cb > tiny ?
                tracers_raw[i, j, cldbase_k, t_idx] / m_cb : zero(FT)
            cmfmc_at_cldbase = cmfmc[i, j, cldbase_k + 1]
            if cmfmc_at_cldbase > tiny
                qb_num = zero(FT); qb_comp = zero(FT)
                mb_pa  = zero(FT); mb_pa_comp = zero(FT)
                for k in (cldbase_k + 1):Nz
                    m_k = air_mass[i, j, k]
                    q_k = m_k > tiny ? tracers_raw[i, j, k, t_idx] / m_k : zero(FT)
                    m_k_pa = m_k / cell_area_j
                    qb_num, qb_comp     = _kahan_add(qb_num, qb_comp, q_k * m_k_pa)
                    mb_pa,  mb_pa_comp  = _kahan_add(mb_pa,  mb_pa_comp, m_k_pa)
                end
                if mb_pa > zero(FT)
                    qb = qb_num / mb_pa
                    qc_mixed = (mb_pa * qb + cmfmc_at_cldbase * q_cldbase * dt) /
                               (mb_pa + cmfmc_at_cldbase * dt)
                    for k in (cldbase_k + 1):Nz
                        tracers_raw[i, j, k, t_idx] = qc_mixed * air_mass[i, j, k]
                    end
                    # Close the column budget at the cloud-base layer.
                    m_cb_pa = m_cb / cell_area_j
                    if m_cb_pa > tiny
                        q_cldbase_new = q_cldbase +
                            cmfmc_at_cldbase * dt * (qc_mixed - q_cldbase) / m_cb_pa
                        tracers_raw[i, j, cldbase_k, t_idx] = q_cldbase_new * m_cb
                    end
                end
            end
        end

        # ── Pass 1: updraft concentration, bottom-to-top (Nz → 1) ──
        # The updraft rises from the surface upward. In our convention,
        # "rising" = decreasing k. At the base (k=Nz), no updraft from
        # below, so we start with qc_below = 0.
        #
        # Match GCHP `convection_mod.F90:917`: only update qc when
        # `entrn ≥ 0 .and. cmout > tiny`; otherwise keep qc unchanged.
        # We do NOT add the GCHP `Q + DELQ < 0 → DELQ = -Q(K)` clamp
        # (convection_mod.F90:1001-1004) because that clamp is
        # non-conservative and breaks linearity in q, which the
        # adjoint path requires. Negativity is the global mass fixer's
        # responsibility.
        qc_below = zero(FT)

        for k in Nz:-1:1
            m_k = air_mass[i, j, k]
            q_k = m_k > tiny ? tracers_raw[i, j, k, t_idx] / m_k : zero(FT)

            cmfmc_bot = k < Nz ? cmfmc[i, j, k + 1] : zero(FT)   # from below
            cmfmc_top = cmfmc[i, j, k]                            # going up
            dtrain_k  = has_dtrain ? dtrain[i, j, k] : zero(FT)

            cmout = cmfmc_top + dtrain_k
            entrn = cmout - cmfmc_bot

            if entrn >= zero(FT) && cmout > tiny
                qc, _qc_scav = _cmfmc_updraft_mix(qc_below, q_k,
                                                   cmfmc_bot, entrn, cmout, tiny)
            else
                qc = qc_below
            end
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
    tiny = _cmfmc_tiny(FT)
    cell_area = FT(cell_areas[c])

    @inbounds for t_idx in 1:Nt
        # Cloud base = largest k with `|cmfmc[k+1]| > tiny` (lowest
        # altitude with non-zero updraft inflow). See LL kernel above
        # for the GCHP convention reference (convection_mod.F90:625).
        cldbase_k = 0
        for k in Nz:-1:1
            cmfmc_bot_k = cmfmc[c, k + 1]
            if abs(cmfmc_bot_k) > tiny
                cldbase_k = k
                break
            end
        end

        if cldbase_k == 0
            continue
        end

        # Well-mixed sub-cloud, kg/m² accumulator + column-closing
        # cloud-base update — see LL kernel for the full derivation.
        if cldbase_k < Nz
            m_cb = air_mass[c, cldbase_k]
            q_cldbase = m_cb > tiny ? tracers_raw[c, cldbase_k, t_idx] / m_cb : zero(FT)
            cmfmc_at_cldbase = cmfmc[c, cldbase_k + 1]
            if cmfmc_at_cldbase > tiny
                qb_num     = zero(FT); qb_comp    = zero(FT)
                mb_pa      = zero(FT); mb_pa_comp = zero(FT)
                for k in (cldbase_k + 1):Nz
                    m_k = air_mass[c, k]
                    q_k = m_k > tiny ? tracers_raw[c, k, t_idx] / m_k : zero(FT)
                    m_k_pa = m_k / cell_area
                    qb_num, qb_comp    = _kahan_add(qb_num, qb_comp, q_k * m_k_pa)
                    mb_pa,  mb_pa_comp = _kahan_add(mb_pa,  mb_pa_comp, m_k_pa)
                end
                if mb_pa > zero(FT)
                    qb = qb_num / mb_pa
                    qc_mixed = (mb_pa * qb + cmfmc_at_cldbase * q_cldbase * dt) /
                               (mb_pa + cmfmc_at_cldbase * dt)
                    for k in (cldbase_k + 1):Nz
                        tracers_raw[c, k, t_idx] = qc_mixed * air_mass[c, k]
                    end
                    m_cb_pa = m_cb / cell_area
                    if m_cb_pa > tiny
                        q_cldbase_new = q_cldbase +
                            cmfmc_at_cldbase * dt * (qc_mixed - q_cldbase) / m_cb_pa
                        tracers_raw[c, cldbase_k, t_idx] = q_cldbase_new * m_cb
                    end
                end
            end
        end

        # Pass 1: GCHP-style guard — see LL kernel above for the
        # `entrn ≥ 0 .and. cmout > tiny` rationale and the
        # deliberate omission of the non-conservative `Q+DELQ<0` clamp.
        qc_below = zero(FT)

        for k in Nz:-1:1
            m_k = air_mass[c, k]
            q_k = m_k > tiny ? tracers_raw[c, k, t_idx] / m_k : zero(FT)

            cmfmc_bot = k < Nz ? cmfmc[c, k + 1] : zero(FT)
            cmfmc_top = cmfmc[c, k]
            dtrain_k = has_dtrain ? dtrain[c, k] : zero(FT)

            cmout = cmfmc_top + dtrain_k
            entrn = cmout - cmfmc_bot

            if entrn >= zero(FT) && cmout > tiny
                qc, _qc_scav = _cmfmc_updraft_mix(qc_below, q_k,
                                                  cmfmc_bot, entrn, cmout, tiny)
            else
                qc = qc_below
            end
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
