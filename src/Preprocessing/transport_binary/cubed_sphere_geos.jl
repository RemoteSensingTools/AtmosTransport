# ===========================================================================
# Native GEOS-IT/FP cubed-sphere → v4 transport binary preprocessing path.
#
# Source axis:  AbstractGEOSSettings (read native CTM_A1/CTM_I1 NetCDF)
# Target axis:  CubedSphereTargetGeometry, source mesh == target mesh
#               (passthrough — IdentityRegrid)
#
# Critical design choices:
#
#  1. **No Poisson balance.** FV3's native MFXC/MFYC are already discretely
#     conservative; running a CG projection on top would only absorb
#     floating-point noise at the cost of distorting the physics-consistent
#     fluxes. (User correction 2026-04-24.)
#
#  2. **Pressure-fixer cm + optional endpoint-mass chaining** (codex Option C,
#     validated 2026-04-25). FV3 conserves moist mass; per-column dry mass changes via
#     both horizontal MFXC divergence AND vertical moisture transport. The
#     raw GEOS DELP_dry endpoints don't satisfy strict per-level
#     `(m_next-m)/(2·steps) = -(div_h+div_v)` for any local `cm` choice.
#     The historical GEOS-FP runner (commit `76fa489::compute_cm_panel_cpu!`)
#     instead used FV3's pressure-fixer rule
#       `cm[k+1]-cm[k] = C_k - ΔB[k]·pit`,  pit = Σ_k C_k
#     which closes `cm[Nz+1] = 0` exactly without any per-cell residual
#     redistribution. Substituted into the v4 replay equation, the per-level
#     mass evolution is `Δm[k] = +2·steps · ΔB[k]·pit`. With
#     `chain_mass = true`, window 1 starts from raw GEOS DELP_dry and
#     subsequent windows take `m_cur = m_next_pf` from the previous window.
#     With `chain_mass = false`, every window starts from raw GEOS DELP_dry
#     and writes a local pressure-fixer tendency. Both modes are internally
#     self-consistent: replay closes to roundoff and the runtime tracer mass
#     evolves with the same fluxes that produced that window's `m_next_pf`.
#
#  3. **Window-by-window loop**:
#
#       read_window!(settings, handles, date, win)         # raw GEOS endpoints
#       geos_native_to_face_flux!(am_v4, bm_v4, ...)       # face-stagger + panel halos
#       compute_cs_cm_pressure_fixer!(cm_v4, am_v4, bm_v4, ΔB, ...)
#       evolve m_next_pf = m_cur + 2·steps · ΔB·pit         # closes replay exactly
#       fill dm = m_next_pf - m_cur, write window, m_cur ← m_next_pf
# ===========================================================================

"""
    _delp_pa_to_air_mass_kg!(m_kg, m_pa, cell_areas, inv_g) -> m_kg

In-place: convert pressure thickness in Pa to cell air mass in kg per
`m_kg[i, j, k] = m_pa[i, j, k] × cell_areas[i, j] × inv_g`. Cell areas are
in m² and apply identically to every CS panel by symmetry.
"""
function _delp_pa_to_air_mass_kg!(m_kg::AbstractArray{FT, 3},
                                  m_pa::AbstractArray{FT, 3},
                                  cell_areas::AbstractMatrix{FT},
                                  inv_g::FT) where {FT}
    Nx, Ny, Nz = size(m_kg)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        m_kg[i, j, k] = m_pa[i, j, k] * cell_areas[i, j] * inv_g
    end
    return m_kg
end

"""
    _evolve_mass_pressure_fixer!(m_next, m_cur, am_v4, bm_v4, ΔB,
                                 two_steps, Nc, Nz)

Per-cell column evolution under the FV3 pressure-fixer rule:

    pit       = Σ_k (am_inflow_k + bm_inflow_k)
    m_next[k] = m_cur[k] + two_steps · ΔB[k] · pit

This is the unique mass evolution that makes the replay equation close
exactly when the stored `cm` is the pressure-fixer's
`cm[k+1]-cm[k] = C_k - ΔB[k]·pit`. See module-header rationale for why
this differs from the raw GEOS DELP_dry endpoint tendency.
"""
function _evolve_mass_pressure_fixer!(
        m_next::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        m_cur::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        am_v4::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        bm_v4::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        ΔB::AbstractVector,
        two_steps::FT, Nc::Int, Nz::Int) where {FT}
    @inbounds for p in 1:CS_PANEL_COUNT
        am = am_v4[p]; bm = bm_v4[p]
        m  = m_cur[p]; mn = m_next[p]
        for j in 1:Nc, i in 1:Nc
            pit = zero(FT)
            for k in 1:Nz
                pit += (am[i, j, k] - am[i + 1, j, k]) +
                       (bm[i, j, k] - bm[i, j + 1, k])
            end
            for k in 1:Nz
                mn[i, j, k] = m[i, j, k] + two_steps * FT(ΔB[k]) * pit
            end
        end
    end
    return nothing
end

"""
    _ps_from_air_mass!(ps, m, area, g, Nc, Nz)

Set `ps[i,j] = (Σ_k m[i,j,k]) · g / area[i,j]` (Pa). Used to keep the
binary's stored `ps` consistent with the chained pressure-fixer mass.
"""
function _ps_from_air_mass!(ps::AbstractMatrix{FT},
                            m::AbstractArray{FT, 3},
                            cell_areas::AbstractMatrix{FT},
                            g::FT, Nc::Int, Nz::Int) where {FT}
    @inbounds for j in 1:Nc, i in 1:Nc
        s = zero(FT)
        for k in 1:Nz
            s += m[i, j, k]
        end
        ps[i, j] = s * g / cell_areas[i, j]
    end
    return ps
end

_validate_geos_native_panel_convention(::GEOSNativePanelConvention) = nothing
function _validate_geos_native_panel_convention(conv)
    error("GEOS-CS passthrough requires panel_convention=`geos_native` on " *
          "the target geometry; got $(typeof(conv)).")
end

_geos_next_endpoint_available(handles::GEOSDayHandles) =
    handles.next_ctm_i1 !== nothing
_geos_next_endpoint_available(handles::GEOSFPNativeDayHandles) =
    handles.next_ctm !== nothing

abstract type AbstractGEOSCSResolutionStrategy end
struct GEOSCSIdentityStrategy <: AbstractGEOSCSResolutionStrategy end
struct GEOSCSBlockCoarsenStrategy{R} <: AbstractGEOSCSResolutionStrategy end

_geos_cs_resolution_strategy(settings::AbstractGEOSSettings, grid::CubedSphereTargetGeometry) =
    _geos_cs_resolution_strategy(Val(settings.Nc), Val(grid.Nc))

_geos_cs_resolution_strategy(::Val{N}, ::Val{N}) where {N} =
    GEOSCSIdentityStrategy()

function _geos_cs_resolution_strategy(::Val{Nsrc}, ::Val{Ndst}) where {Nsrc, Ndst}
    (Nsrc > Ndst && Nsrc % Ndst == 0) ||
        throw(ArgumentError(
            "GEOS-CS conversion supports native passthrough or nested block coarsening only; " *
            "source Nc=$(Nsrc), target Nc=$(Ndst)."))
    return GEOSCSBlockCoarsenStrategy{Nsrc ÷ Ndst}()
end

_geos_cs_strategy_name(::GEOSCSIdentityStrategy) = "identity"
_geos_cs_strategy_name(::GEOSCSBlockCoarsenStrategy{R}) where {R} =
    "nested_block_coarsen_$(R)x$(R)"

function _coarsen_sum3!(dst::AbstractArray{FT, 3},
                        src::AbstractArray{FT, 3},
                        ::Val{R}) where {FT, R}
    Nc = size(dst, 1)
    Nz = size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
        s = zero(FT)
        for jj in ((j - 1) * R + 1):(j * R), ii in ((i - 1) * R + 1):(i * R)
            s += src[ii, jj, k]
        end
        dst[i, j, k] = s
    end
    return dst
end

function _coarsen_area_weighted2!(dst::AbstractMatrix{FT},
                                  src::AbstractMatrix{FT},
                                  src_area::AbstractMatrix{FT},
                                  ::Val{R}) where {FT, R}
    Nc = size(dst, 1)
    @inbounds for j in 1:Nc, i in 1:Nc
        num = zero(FT)
        den = zero(FT)
        for jj in ((j - 1) * R + 1):(j * R), ii in ((i - 1) * R + 1):(i * R)
            a = src_area[ii, jj]
            num += src[ii, jj] * a
            den += a
        end
        dst[i, j] = num / den
    end
    return dst
end

function _coarsen_area_weighted3!(dst::AbstractArray{FT, 3},
                                  src::AbstractArray{FT, 3},
                                  src_area::AbstractMatrix{FT},
                                  ::Val{R}) where {FT, R}
    Nc = size(dst, 1)
    Nz = size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
        num = zero(FT)
        den = zero(FT)
        for jj in ((j - 1) * R + 1):(j * R), ii in ((i - 1) * R + 1):(i * R)
            a = src_area[ii, jj]
            num += src[ii, jj, k] * a
            den += a
        end
        dst[i, j, k] = num / den
    end
    return dst
end

function _coarsen_xface_sum!(dst::AbstractArray{FT, 3},
                             src::AbstractArray{FT, 3},
                             ::Val{R}) where {FT, R}
    Nc = size(dst, 2)
    Nz = size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:(Nc + 1)
        fi = (i - 1) * R + 1
        s = zero(FT)
        for fj in ((j - 1) * R + 1):(j * R)
            s += src[fi, fj, k]
        end
        dst[i, j, k] = s
    end
    return dst
end

function _coarsen_yface_sum!(dst::AbstractArray{FT, 3},
                             src::AbstractArray{FT, 3},
                             ::Val{R}) where {FT, R}
    Nc = size(dst, 1)
    Nz = size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:(Nc + 1), i in 1:Nc
        fj = (j - 1) * R + 1
        s = zero(FT)
        for fi in ((i - 1) * R + 1):(i * R)
            s += src[fi, fj, k]
        end
        dst[i, j, k] = s
    end
    return dst
end

function _geos_strategy_workspace(::GEOSCSIdentityStrategy,
                                  settings::AbstractGEOSSettings,
                                  grid::CubedSphereTargetGeometry,
                                  ::Type{FT}, Nz::Int) where FT
    return nothing
end

function _geos_strategy_workspace(::GEOSCSBlockCoarsenStrategy{R},
                                  settings::AbstractGEOSSettings,
                                  grid::CubedSphereTargetGeometry,
                                  ::Type{FT}, Nz::Int) where {FT, R}
    source_mesh = source_grid(settings; FT = FT)
    Nsrc = settings.Nc
    Nc = grid.Nc
    panels_3d_src() = ntuple(_ -> zeros(FT, Nsrc, Nsrc, Nz), CS_PANEL_COUNT)
    panels_xface_src() = ntuple(_ -> zeros(FT, Nsrc + 1, Nsrc, Nz), CS_PANEL_COUNT)
    panels_yface_src() = ntuple(_ -> zeros(FT, Nsrc, Nsrc + 1, Nz), CS_PANEL_COUNT)
    panels_2d_dst() = ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT)
    panels_3d_dst(nlev) = ntuple(_ -> zeros(FT, Nc, Nc, nlev), CS_PANEL_COUNT)
    return (
        source_mesh = source_mesh,
        source_cell_areas = source_mesh.cell_areas,
        fine_m_kg = panels_3d_src(),
        fine_am_v4 = panels_xface_src(),
        fine_bm_v4 = panels_yface_src(),
        surface = settings.include_surface ? (
            pblh = panels_2d_dst(),
            ustar = panels_2d_dst(),
            hflux = panels_2d_dst(),
            t2m = panels_2d_dst(),
        ) : nothing,
        cmfmc = settings.include_convection ? panels_3d_dst(Nz + 1) : nothing,
        dtrain = settings.include_convection ? panels_3d_dst(Nz) : nothing,
    )
end

function _geos_fluxes_to_target!(::GEOSCSIdentityStrategy, _ws,
                                 am_v4, bm_v4, raw, grid,
                                 Nc::Int, Nz::Int, flux_scale)
    geos_native_to_face_flux!(am_v4, bm_v4, raw.am, raw.bm,
                              grid.mesh.connectivity, Nc, Nz, flux_scale)
    return nothing
end

function _geos_fluxes_to_target!(::GEOSCSBlockCoarsenStrategy{R}, ws,
                                 am_v4, bm_v4, raw, _grid,
                                 Nc::Int, Nz::Int, flux_scale) where R
    Nsrc = Nc * R
    geos_native_to_face_flux!(ws.fine_am_v4, ws.fine_bm_v4, raw.am, raw.bm,
                              ws.source_mesh.connectivity, Nsrc, Nz, flux_scale)
    for p in 1:CS_PANEL_COUNT
        _coarsen_xface_sum!(am_v4[p], ws.fine_am_v4[p], Val(R))
        _coarsen_yface_sum!(bm_v4[p], ws.fine_bm_v4[p], Val(R))
    end
    return nothing
end

function _geos_seed_mass!(::GEOSCSIdentityStrategy, _ws, m_cur, raw,
                          cell_areas, inv_g, Nc::Int, Nz::Int)
    for p in 1:CS_PANEL_COUNT
        _delp_pa_to_air_mass_kg!(m_cur[p], raw.m[p], cell_areas, inv_g)
    end
    return nothing
end

function _geos_seed_mass!(::GEOSCSBlockCoarsenStrategy{R}, ws, m_cur, raw,
                          _cell_areas, inv_g, _Nc::Int, _Nz::Int) where R
    for p in 1:CS_PANEL_COUNT
        _delp_pa_to_air_mass_kg!(ws.fine_m_kg[p], raw.m[p], ws.source_cell_areas, inv_g)
        _coarsen_sum3!(m_cur[p], ws.fine_m_kg[p], Val(R))
    end
    return nothing
end

_geos_surface_payload!(::GEOSCSIdentityStrategy, _ws, raw) = raw.surface

function _geos_surface_payload!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.surface === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        _coarsen_area_weighted2!(ws.surface.pblh[p], raw.surface.pblh[p], ws.source_cell_areas, Val(R))
        _coarsen_area_weighted2!(ws.surface.ustar[p], raw.surface.ustar[p], ws.source_cell_areas, Val(R))
        _coarsen_area_weighted2!(ws.surface.hflux[p], raw.surface.hflux[p], ws.source_cell_areas, Val(R))
        _coarsen_area_weighted2!(ws.surface.t2m[p], raw.surface.t2m[p], ws.source_cell_areas, Val(R))
    end
    return ws.surface
end

_geos_cmfmc_payload!(::GEOSCSIdentityStrategy, _ws, raw) = raw.cmfmc
_geos_dtrain_payload!(::GEOSCSIdentityStrategy, _ws, raw) = raw.dtrain

function _geos_cmfmc_payload!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.cmfmc === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        _coarsen_area_weighted3!(ws.cmfmc[p], raw.cmfmc[p], ws.source_cell_areas, Val(R))
    end
    return ws.cmfmc
end

function _geos_dtrain_payload!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.dtrain === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        _coarsen_area_weighted3!(ws.dtrain[p], raw.dtrain[p], ws.source_cell_areas, Val(R))
    end
    return ws.dtrain
end

mutable struct GEOSCubedSphereWindowWorkspace{FT, ST, SW, RAW, CA} <:
               AbstractWindowWorkspace{CubedSphereTargetGeometry, FT}
    strategy    :: ST
    strategy_ws :: SW
    raw         :: RAW
    am_v4       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    bm_v4       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    cm_v4       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    dm_v4       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    m_cur       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    m_next_pf   :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    ps_cur      :: NTuple{CS_PANEL_COUNT, Array{FT, 2}}
    ΔB          :: Vector{FT}
    g           :: FT
    inv_g       :: FT
    cell_areas  :: CA
    flux_scale  :: FT
    two_steps   :: FT
    chain_mass  :: Bool
end

function allocate_window_workspace(grid::CubedSphereTargetGeometry,
                                   settings::AbstractGEOSSettings,
                                   vertical,
                                   ::Type{FT};
                                   dt_met_seconds::Real,
                                   chain_mass::Bool = true,
                                   cache = nothing) where FT
    Nc = grid.Nc
    Nz = vertical.Nz
    strategy = _geos_cs_resolution_strategy(settings, grid)
    strategy_ws = _geos_strategy_workspace(strategy, settings, grid, FT, Nz)
    npanel = CS_PANEL_COUNT

    vc = vertical.merged_vc
    g = FT(GRAV)
    inv_g = inv(g)
    cell_areas = grid.mesh.cell_areas
    ΔB = FT[FT(vc.B[k + 1] - vc.B[k]) for k in 1:Nz]
    steps_per_met = round(Int, FT(dt_met_seconds) / FT(settings.mass_flux_dt))
    dt_factor = FT(settings.mass_flux_dt / 2)
    flux_scale = dt_factor / g
    two_steps = FT(2 * steps_per_met)

    am_v4 = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), npanel)
    bm_v4 = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), npanel)
    cm_v4 = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), npanel)
    dm_v4 = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel)
    m_cur = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel)
    m_next_pf = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel)
    ps_cur = ntuple(_ -> zeros(FT, Nc, Nc), npanel)
    raw = allocate_raw_window(settings; FT = FT, Nz = Nz)

    return GEOSCubedSphereWindowWorkspace{
        FT, typeof(strategy), typeof(strategy_ws), typeof(raw), typeof(cell_areas)}(
            strategy, strategy_ws, raw, am_v4, bm_v4, cm_v4, dm_v4,
            m_cur, m_next_pf, ps_cur, ΔB, g, inv_g, cell_areas,
            flux_scale, two_steps, chain_mass)
end

function ingest_window!(workspace::GEOSCubedSphereWindowWorkspace{FT},
                        reader::GEOSNativeReader{FT},
                        win::Int,
                        grid::CubedSphereTargetGeometry,
                        settings::AbstractGEOSSettings,
                        vertical) where FT
    Nc = grid.Nc
    Nz = vertical.Nz
    read_window!(workspace.raw, reader, win)

    _geos_fluxes_to_target!(workspace.strategy, workspace.strategy_ws,
                            workspace.am_v4, workspace.bm_v4,
                            workspace.raw, grid, Nc, Nz,
                            workspace.flux_scale)

    if win == 1 || !workspace.chain_mass
        if workspace.chain_mass && win == 1 && reader.seed !== nothing
            for p in 1:CS_PANEL_COUNT
                size(reader.seed[p]) == (Nc, Nc, Nz) ||
                    error("seed_m[$p] shape $(size(reader.seed[p])) ≠ ($Nc, $Nc, $Nz)")
                copyto!(workspace.m_cur[p], reader.seed[p])
            end
        else
            _geos_seed_mass!(workspace.strategy, workspace.strategy_ws,
                             workspace.m_cur, workspace.raw,
                             workspace.cell_areas, workspace.inv_g, Nc, Nz)
        end
        for p in 1:CS_PANEL_COUNT
            _ps_from_air_mass!(workspace.ps_cur[p], workspace.m_cur[p],
                               workspace.cell_areas, workspace.g, Nc, Nz)
        end
    end

    compute_cs_cm_pressure_fixer!(workspace.cm_v4, workspace.am_v4,
                                  workspace.bm_v4, workspace.ΔB, Nc, Nz)
    _evolve_mass_pressure_fixer!(workspace.m_next_pf, workspace.m_cur,
                                 workspace.am_v4, workspace.bm_v4,
                                 workspace.ΔB, workspace.two_steps, Nc, Nz)
    return nothing
end

function drain_ready_windows!(workspace::GEOSCubedSphereWindowWorkspace{FT},
                              contract::CubedSphereContract{FT},
                              win::Int,
                              grid::CubedSphereTargetGeometry,
                              settings::AbstractGEOSSettings,
                              steps_per_met::Int) where FT
    fill_cs_window_mass_tendency!(workspace.dm_v4, workspace.m_cur,
                                  workspace.m_next_pf, steps_per_met)
    contract_diag = verify_window!((m_cur = workspace.m_cur,
                                     am = workspace.am_v4,
                                     bm = workspace.bm_v4,
                                     cm = workspace.cm_v4,
                                     m_next = workspace.m_next_pf),
                                    contract, win)

    m_target = ntuple(p -> copy(workspace.m_next_pf[p]), CS_PANEL_COUNT)
    convert_cs_mass_target_to_delta!(m_target, workspace.m_cur)

    window_nt = (m = workspace.m_cur, am = workspace.am_v4,
                 bm = workspace.bm_v4, cm = workspace.cm_v4,
                 ps = workspace.ps_cur, dm = m_target)
    if settings.include_surface
        window_nt = merge(window_nt,
                          (surface = _geos_surface_payload!(
                               workspace.strategy, workspace.strategy_ws,
                               workspace.raw),))
    end
    if settings.include_convection
        window_nt = merge(window_nt,
                          (cmfmc = _geos_cmfmc_payload!(
                               workspace.strategy, workspace.strategy_ws,
                               workspace.raw),
                           dtrain = _geos_dtrain_payload!(
                               workspace.strategy, workspace.strategy_ws,
                               workspace.raw)))
    end
    ready = ReadyWindow{CubedSphereTargetGeometry, FT}(win, window_nt)
    return (ready = ready, contract = contract_diag)
end

function advance_window!(workspace::GEOSCubedSphereWindowWorkspace,
                         grid::CubedSphereTargetGeometry)
    workspace.chain_mass || return nothing
    Nc = grid.Nc
    Nz = size(workspace.m_cur[1], 3)
    for p in 1:CS_PANEL_COUNT
        copyto!(workspace.m_cur[p], workspace.m_next_pf[p])
        _ps_from_air_mass!(workspace.ps_cur[p], workspace.m_cur[p],
                           workspace.cell_areas, workspace.g, Nc, Nz)
    end
    return nothing
end

"""
    process_day(date, grid::CubedSphereTargetGeometry,
                settings::AbstractGEOSSettings, vertical;
                out_path,
                dt_met_seconds = 3600.0,
                FT = Float64,
                mass_basis = :dry,
                replay_tol = replay_tolerance(FT),
                seed_m = nothing,
                next_day_hour0 = nothing,
                chain_mass = true) -> NamedTuple

Build a v4 cubed-sphere transport binary at `out_path` from one UTC day of
native GEOS data. Source mesh and target mesh must match (CS passthrough).

Stored mass is the pressure-fixer evolution from the current window's
initial mass (see module header), not a naive raw-endpoint tendency. The
replay gate closes to roundoff by construction; the maximum absolute
residual goes down to floating-point noise instead of the ~1% column-residual
the naive `m=DELP_dry, cm=diagnose_cs_cm` path produced.

For multi-day preprocessing with `chain_mass = true`, `seed_m` carries the
pressure-fixer endpoint from the previous day so adjacent daily binaries share
a boundary mass: pass `nothing` (default) on day 1 to seed from raw GEOS
DELP_dry, and on day N+1 pass the `final_m` returned by day N's
`process_day`. With `chain_mass = false`, `seed_m` is ignored and every
window reinitializes from raw GEOS mass.

When `chain_mass = true`, the returned NamedTuple includes
`final_m::NTuple{6, Array{FT, 3}}`, the pressure-fixer state at the END of the
last window. With `chain_mass = false`, `final_m` is `nothing`.

`next_day_hour0` is part of the inherited topology-dispatch contract but
unused — the GEOS reader handles next-day endpoints internally via
`next_ctm_i1`.
"""
function process_day(date::Date,
                     grid::CubedSphereTargetGeometry,
                     settings::AbstractGEOSSettings,
                     vertical;
                     out_path::AbstractString,
                     dt_met_seconds::Real = 3600.0,
                     FT::Type{<:AbstractFloat} = Float64,
                     mass_basis::Symbol = :dry,
                     replay_tol::Real = replay_tolerance(FT),
                     positivity_cfl_limit::Real = 0.95,
                     require_substep_positivity::Bool = true,
                     chain_mass::Bool = true,
                     seed_m::Union{Nothing, NTuple{6, <:AbstractArray}} = nothing,
                     next_day_hour0 = nothing)
    # Reject configurations the path cannot honor:
    mass_basis === :dry ||
        error("GEOS-CS passthrough only supports mass_basis=:dry; got $(mass_basis). " *
              "GEOS MFXC/MFYC are already dry; the chained pressure-fixer is dry-basis.")
    _validate_geos_native_panel_convention(grid.mesh.convention)
    # Single source of truth: the binary's panel-convention attrib comes from
    # the target mesh's convention, not from a duplicated config key.
    panel_convention = "geos_native"

    Nc     = grid.Nc
    npanel = CS_PANEL_COUNT
    Nz     = vertical.Nz
    vc     = vertical.merged_vc

    steps_per_met = round(Int, FT(dt_met_seconds) / FT(settings.mass_flux_dt))
    # `read_window!` exposes GEOS MFXC/MFYC as a rate-like diagnostic
    # (`raw.am = MFXC / mass_flux_dt`). CTM_A1's MFXC/MFYC values are one
    # dynamics-step pressure-area transport amount, not an hourly accumulated
    # total. The v4 CS runtime uses the same window-constant transport for each
    # 450 s substep inside the hour, with two horizontal Strang half-sweeps per
    # substep, so each stored half-sweep face flux is `MFXC / (2g)`.
    # Because `raw.am` has already divided by `mass_flux_dt`, multiply by
    # `mass_flux_dt / (2g)` here.
    reader_seed = seed_m === nothing ? nothing :
        ntuple(p -> Array{FT, 3}(seed_m[p]), CS_PANEL_COUNT)
    reader = open_reader(settings, date, FT;
                         seed = reader_seed,
                         chain_mass = chain_mass,
                         next_day_handle = true)
    nw = windows_per_day(reader)
    workspace = allocate_window_workspace(grid, settings, vertical, FT;
                                          dt_met_seconds = dt_met_seconds,
                                          chain_mass = chain_mass)

    @info "GEOS → CS: $(date), source=$(settings) → $(out_path)"
    @info "  source_C=$(settings.Nc) target_C=$Nc  strategy=$(_geos_cs_strategy_name(workspace.strategy))"
    @info "  Nz=$Nz  windows=$nw  steps_per_met=$steps_per_met  flux_scale=$(workspace.flux_scale)"
    @info "  Level orientation: $(reader.handles.orientation)  (next-day endpoint: $(_geos_next_endpoint_available(reader.handles)))"

    mkpath(dirname(out_path))

    # Stage to `.tmp` so a mid-loop replay failure or post-loop positivity
    # quarantine never leaves a partial binary at `out_path`. Promote
    # `tmp_path -> out_path` only after all contract gates pass (or after the
    # summary warns under `require_substep_positivity = false`).
    tmp_path = out_path * ".tmp"
    isfile(tmp_path) && rm(tmp_path; force = true)

    # Writer-open is inside the protected region: if `open_…!` writes a header
    # then errors (e.g. truncated payload negotiation), the `finally` block
    # still removes the partial `tmp_path`. `writer = nothing` lets the
    # finally close-guard distinguish "never opened" from "opened but not yet
    # explicitly closed".
    writer        = nothing
    writer_closed = false
    mv_done       = false
    try
        writer = open_streaming_cs_transport_binary(
            tmp_path, Nc, npanel, Nz, nw, vc;
            FT = FT,
            dt_met_seconds = dt_met_seconds,
            steps_per_window = steps_per_met,
            mass_basis = mass_basis,
            include_flux_delta = true,
            include_surface    = settings.include_surface,
            include_cmfmc      = settings.include_convection,
            include_dtrain     = settings.include_convection,
            panel_convention   = panel_convention,
            cs_definition      = _cs_definition_tag(grid),
            cs_coordinate_law  = _cs_coordinate_law_tag(grid),
            cs_center_law      = _cs_center_law_tag(grid),
            longitude_offset_deg = longitude_offset_deg(cs_definition(grid.mesh)),
            extra_header = Dict{String, Any}(
                "source_Nc" => settings.Nc,
                "geos_cs_resolution_strategy" => _geos_cs_strategy_name(workspace.strategy),
            ),
        )

        worst_replay_rel = 0.0
        worst_replay_abs = 0.0
        worst_replay_win = 0
        window_contract = CubedSphereContract{FT}(
            replay_tol = replay_tol,
            positivity_cfl_limit = positivity_cfl_limit,
            require_substep_positivity = require_substep_positivity,
            steps_per_window = steps_per_met,
        )

        t_start = time()

        @inbounds for win in 1:nw
            ingest_window!(workspace, reader, win, grid, settings, vertical)
            ready_diag = drain_ready_windows!(workspace, window_contract,
                                              win, grid, settings, steps_per_met)
            contract_diag = ready_diag.contract
            if worst_replay_win == 0 || contract_diag.replay.max_rel_err > worst_replay_rel
                worst_replay_rel = contract_diag.replay.max_rel_err
                worst_replay_abs = contract_diag.replay.max_abs_err
                worst_replay_win = win
            end
            update_accumulator!(window_contract, contract_diag.positivity, win)
            write_streaming_cs_window!(writer, ready_diag.ready.payload, Nc, npanel)
            advance_window!(workspace, grid)
        end

        elapsed = time() - t_start
        @info @sprintf("  Done in %.1fs (%.2fs/window). Worst replay: rel=%.2e abs=%.2e at win=%d",
                       elapsed, elapsed / nw, worst_replay_rel, worst_replay_abs, worst_replay_win)

        # Close the writer before the positivity summary so a quarantine
        # delete (require_substep_positivity=true + violation) can remove the
        # binary cleanly without an open file handle on it.
        close_streaming_transport_binary!(writer)
        writer_closed = true
        summarize_status!(window_contract; quarantine_path = tmp_path)

        # Reached only when positivity passed, or when require_substep_positivity=false
        # turned a violation into a warning. Promote the staged file either way.
        mv(tmp_path, out_path; force = true)
        mv_done = true

        # Capture the pressure-fixer endpoint from the last window so the
        # caller can seed the next day's `process_day` and preserve cross-day
        # continuity (codex 2026-04-25 P2).
        final_m = chain_mass ? ntuple(p -> copy(workspace.m_cur[p]), npanel) : nothing
        set_end_of_day_seed!(reader, final_m)

        return (
            elapsed = elapsed,
            worst_replay_rel = worst_replay_rel,
            worst_replay_abs = worst_replay_abs,
            worst_replay_win = worst_replay_win,
            out_path = out_path,
            final_m = final_m,
        )
    finally
        # Guard on `writer !== nothing` so a failure in `open_…` itself
        # (before assignment) does not try to close a `nothing`.
        if writer !== nothing && !writer_closed
            close_streaming_transport_binary!(writer)
        end
        # On any non-clean exit (loop exception, replay error, quarantined
        # positivity violation), remove the staged file so it cannot be
        # mistaken for a finished binary on retry. `summarize_…` already
        # deleted it in the explicit quarantine case; this guard catches the
        # other failure modes.
        if !mv_done && isfile(tmp_path)
            rm(tmp_path; force = true)
        end
        close_reader!(reader)
    end
end
