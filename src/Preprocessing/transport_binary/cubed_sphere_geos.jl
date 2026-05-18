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
#  2. **Raw dry endpoint mass + diagnosed cm.** FV3's pressure-fixer rule is
#     useful for closing a local vertical flux, but its implied dry endpoint
#     can go negative in GEOS-IT's very thin upper layers even when the raw
#     next-hour GEOS dry mass is healthy. Plan-41 binary v3 therefore makes
#     the raw GEOS DELP_dry endpoint the written mass target. The native
#     horizontal fluxes are column-balanced to that target, then `cm` is
#     diagnosed from `(am, bm, dm)` so replay and endpoint positivity are both
#     checked against the same physical endpoint.
#
#  3. **Window-by-window loop**:
#
#       read_window!(settings, handles, date, win)         # raw GEOS endpoints
#       geos_native_to_face_flux!(am_v4, bm_v4, ...)       # face-stagger + panel halos
#       derive m_next_target from raw GEOS DELP_dry endpoint
#       choose per-window steps, scale fluxes, balance column fluxes
#       diagnose cm from dm = (m_next_target - m_cur)/(2·steps)
#       write window, m_cur ← m_next_target when chaining
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
                                  ::Type{FT}, Nz_native::Int,
                                  Nz_out::Int) where FT
    return nothing
end

function _geos_strategy_workspace(::GEOSCSBlockCoarsenStrategy{R},
                                  settings::AbstractGEOSSettings,
                                  grid::CubedSphereTargetGeometry,
                                  ::Type{FT}, Nz_native::Int,
                                  Nz_out::Int) where {FT, R}
    source_mesh = source_grid(settings; FT = FT)
    Nsrc = settings.Nc
    Nc = grid.Nc
    panels_3d_src() = ntuple(_ -> zeros(FT, Nsrc, Nsrc, Nz_native), CS_PANEL_COUNT)
    panels_xface_src() = ntuple(_ -> zeros(FT, Nsrc + 1, Nsrc, Nz_native), CS_PANEL_COUNT)
    panels_yface_src() = ntuple(_ -> zeros(FT, Nsrc, Nsrc + 1, Nz_native), CS_PANEL_COUNT)
    panels_2d_dst() = ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT)
    panels_3d_dst(nlev) = ntuple(_ -> zeros(FT, Nc, Nc, nlev), CS_PANEL_COUNT)
    return (
        source_mesh = source_mesh,
        source_cell_areas = source_mesh.cell_areas,
        fine_m_kg = panels_3d_src(),
        fine_am_v4 = panels_xface_src(),
        fine_bm_v4 = panels_yface_src(),
        cmfmc_native = settings.include_convection ? panels_3d_dst(Nz_native + 1) : nothing,
        dtrain_native = settings.include_convection ? panels_3d_dst(Nz_native) : nothing,
        surface = (settings.include_surface || settings.include_vdiff_fields) ? (
            pblh = panels_2d_dst(),
            ustar = panels_2d_dst(),
            hflux = panels_2d_dst(),
            t2m = panels_2d_dst(),
        ) : nothing,
        vdiff_native = settings.include_vdiff_fields ? (
            u = panels_3d_dst(Nz_native),
            v = panels_3d_dst(Nz_native),
            t = panels_3d_dst(Nz_native),
            qv = panels_3d_dst(Nz_native),
        ) : nothing,
        vdiff_weights_native = settings.include_vdiff_fields ?
            panels_3d_dst(Nz_native) : nothing,
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

function _geos_target_mass!(::GEOSCSIdentityStrategy, _ws, m_next, raw,
                            cell_areas, inv_g, Nc::Int, Nz::Int)
    for p in 1:CS_PANEL_COUNT
        _delp_pa_to_air_mass_kg!(m_next[p], raw.m_next[p], cell_areas, inv_g)
    end
    return nothing
end

function _geos_target_mass!(::GEOSCSBlockCoarsenStrategy{R}, ws, m_next, raw,
                            _cell_areas, inv_g, _Nc::Int, _Nz::Int) where R
    for p in 1:CS_PANEL_COUNT
        _delp_pa_to_air_mass_kg!(ws.fine_m_kg[p], raw.m_next[p], ws.source_cell_areas, inv_g)
        _coarsen_sum3!(m_next[p], ws.fine_m_kg[p], Val(R))
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

_geos_vdiff_native_target!(::GEOSCSIdentityStrategy, _ws, raw) = raw.vdiff
_geos_vdiff_native_weights!(::GEOSCSIdentityStrategy, _ws, raw,
                            _cell_areas, _inv_g) = raw.m

function _geos_vdiff_native_target!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.vdiff === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        _coarsen_area_weighted3!(ws.vdiff_native.u[p], raw.vdiff.u[p],
                                 ws.source_cell_areas, Val(R))
        _coarsen_area_weighted3!(ws.vdiff_native.v[p], raw.vdiff.v[p],
                                 ws.source_cell_areas, Val(R))
        _coarsen_area_weighted3!(ws.vdiff_native.t[p], raw.vdiff.t[p],
                                 ws.source_cell_areas, Val(R))
        _coarsen_area_weighted3!(ws.vdiff_native.qv[p], raw.vdiff.qv[p],
                                 ws.source_cell_areas, Val(R))
    end
    return ws.vdiff_native
end

function _geos_vdiff_native_weights!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw,
                                     _cell_areas, inv_g) where R
    for p in 1:CS_PANEL_COUNT
        _delp_pa_to_air_mass_kg!(ws.fine_m_kg[p], raw.m[p],
                                  ws.source_cell_areas, inv_g)
        _coarsen_sum3!(ws.vdiff_weights_native[p], ws.fine_m_kg[p], Val(R))
    end
    return ws.vdiff_weights_native
end

_geos_cmfmc_native_target!(::GEOSCSIdentityStrategy, _ws, raw) = raw.cmfmc
_geos_dtrain_native_target!(::GEOSCSIdentityStrategy, _ws, raw) = raw.dtrain

function _geos_cmfmc_native_target!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.cmfmc === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        _coarsen_area_weighted3!(ws.cmfmc_native[p], raw.cmfmc[p],
                                 ws.source_cell_areas, Val(R))
    end
    return ws.cmfmc_native
end

function _geos_dtrain_native_target!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.dtrain === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        _coarsen_area_weighted3!(ws.dtrain_native[p], raw.dtrain[p],
                                 ws.source_cell_areas, Val(R))
    end
    return ws.dtrain_native
end

function _geos_cmfmc_payload!(workspace)
    workspace.cmfmc_v4 === nothing && return nothing
    native = _geos_cmfmc_native_target!(workspace.strategy, workspace.strategy_ws,
                                        workspace.raw)
    native === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        apply_vertical!(workspace.cmfmc_v4[p], native[p], workspace.plan,
                        ConvectionInterfaceFlux())
    end
    return workspace.cmfmc_v4
end

function _geos_dtrain_payload!(workspace)
    workspace.dtrain_v4 === nothing && return nothing
    native = _geos_dtrain_native_target!(workspace.strategy, workspace.strategy_ws,
                                         workspace.raw)
    native === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        apply_vertical!(workspace.dtrain_v4[p], native[p], workspace.plan,
                        ConvectionTendencyField())
    end
    return workspace.dtrain_v4
end

function _geos_vdiff_payload!(workspace)
    workspace.vdiff_v4 === nothing && return nothing
    native = _geos_vdiff_native_target!(workspace.strategy, workspace.strategy_ws,
                                        workspace.raw)
    native === nothing && return nothing
    weights = _geos_vdiff_native_weights!(workspace.strategy, workspace.strategy_ws,
                                          workspace.raw, workspace.cell_areas,
                                          workspace.inv_g)
    for p in 1:CS_PANEL_COUNT
        apply_vertical!(workspace.vdiff_v4.u[p], native.u[p], workspace.plan,
                        IntensiveCenterField(), weights[p])
        apply_vertical!(workspace.vdiff_v4.v[p], native.v[p], workspace.plan,
                        IntensiveCenterField(), weights[p])
        apply_vertical!(workspace.vdiff_v4.t[p], native.t[p], workspace.plan,
                        IntensiveCenterField(), weights[p])
        apply_vertical!(workspace.vdiff_v4.qv[p], native.qv[p], workspace.plan,
                        IntensiveCenterField(), weights[p])
    end
    return workspace.vdiff_v4
end

mutable struct GEOSCubedSphereWindowWorkspace{FT, ST, SW, RAW, CA, VP, CV, DV, VD} <:
               AbstractWindowWorkspace{CubedSphereTargetGeometry, FT}
    strategy    :: ST
    strategy_ws :: SW
    raw         :: RAW
    plan        :: VP
    am_native_v4 :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    bm_native_v4 :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    m_native_kg  :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    am_v4       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    bm_v4       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    cm_v4       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    dm_v4       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    m_cur       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    m_next_target :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    ps_cur      :: NTuple{CS_PANEL_COUNT, Array{FT, 2}}
    cmfmc_v4    :: CV
    dtrain_v4   :: DV
    vdiff_v4    :: VD
    g           :: FT
    inv_g       :: FT
    cell_areas  :: CA
    base_flux_scale :: FT
    flux_scale  :: FT
    source_steps_per_met :: Int
    steps_current :: Int
    steps_schedule :: Vector{Int}
    adaptive_substeps :: Bool
    substep_cfl_target :: Float64
    min_steps_per_window :: Int
    max_steps_per_window :: Int
    chain_mass  :: Bool
end

const _GEOS_ADAPTIVE_SUBSTEP_MAX_REFINEMENTS = 8

function allocate_window_workspace(grid::CubedSphereTargetGeometry,
                                   settings::AbstractGEOSSettings,
                                   vertical,
                                   ::Type{FT};
                                   dt_met_seconds::Real,
                                   chain_mass::Bool = true,
                                   cache = nothing,
                                   adaptive_substeps::Bool = false,
                                   substep_cfl_target::Real = 0.95,
                                   min_steps_per_window::Integer = 1,
                                   max_steps_per_window::Integer = typemax(Int),
                                   windows_per_day::Integer = 0) where FT
    Nc = grid.Nc
    Nz = vertical.Nz
    Nz_native = vertical.Nz_native
    plan = if hasproperty(vertical, :plan)
        vertical.plan
    else
        Nz == Nz_native ||
            error("GEOS vertical setup with Nz=$(Nz), Nz_native=$(Nz_native) must carry a `plan` field.")
        vc_identity = HybridSigmaPressure(FT.(vertical.merged_vc.A),
                                          FT.(vertical.merged_vc.B))
        plan_vertical(IdentityVertical(), vc_identity)
    end
    strategy = _geos_cs_resolution_strategy(settings, grid)
    strategy_ws = _geos_strategy_workspace(strategy, settings, grid, FT,
                                           Nz_native, Nz)
    npanel = CS_PANEL_COUNT

    g = FT(GRAV)
    inv_g = inv(g)
    cell_areas = grid.mesh.cell_areas
    steps_per_met = round(Int, FT(dt_met_seconds) / FT(settings.mass_flux_dt))
    dt_factor = FT(settings.mass_flux_dt / 2)
    flux_scale = dt_factor / g
    target = Float64(substep_cfl_target)
    isfinite(target) && target > 0 ||
        error("substep_cfl_target must be finite and > 0; got $(substep_cfl_target)")
    min_steps = Int(min_steps_per_window)
    max_steps = Int(max_steps_per_window)
    1 <= min_steps <= max_steps ||
        error("invalid adaptive substep bounds: min=$(min_steps), max=$(max_steps)")
    schedule_len = Int(windows_per_day)
    schedule_len >= 0 ||
        error("windows_per_day must be non-negative, got $(windows_per_day)")

    am_native_v4 = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz_native), npanel)
    bm_native_v4 = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz_native), npanel)
    m_native_kg = ntuple(_ -> zeros(FT, Nc, Nc, Nz_native), npanel)
    am_v4 = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), npanel)
    bm_v4 = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), npanel)
    cm_v4 = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), npanel)
    dm_v4 = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel)
    m_cur = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel)
    m_next_target = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel)
    ps_cur = ntuple(_ -> zeros(FT, Nc, Nc), npanel)
    cmfmc_v4 = settings.include_convection ?
        ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), npanel) : nothing
    dtrain_v4 = settings.include_convection ?
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel) : nothing
    vdiff_v4 = settings.include_vdiff_fields ? (
        u  = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel),
        v  = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel),
        t  = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel),
        qv = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel),
    ) : nothing
    raw = allocate_raw_window(settings; FT = FT, Nz = Nz_native)

    return GEOSCubedSphereWindowWorkspace{
        FT, typeof(strategy), typeof(strategy_ws), typeof(raw), typeof(cell_areas),
        typeof(plan), typeof(cmfmc_v4), typeof(dtrain_v4), typeof(vdiff_v4)}(
            strategy, strategy_ws, raw, plan,
            am_native_v4, bm_native_v4, m_native_kg,
            am_v4, bm_v4, cm_v4, dm_v4,
            m_cur, m_next_target, ps_cur, cmfmc_v4, dtrain_v4, vdiff_v4,
            g, inv_g, cell_areas,
            flux_scale, flux_scale, steps_per_met, steps_per_met,
            fill(steps_per_met, schedule_len), Bool(adaptive_substeps),
            target, min_steps, max_steps, chain_mass)
end

@inline _geos_clamp_steps(n::Integer, lo::Integer, hi::Integer) =
    min(max(Int(n), Int(lo)), Int(hi))

function _scale_cs_flux_panels!(panels, factor)
    for p in 1:CS_PANEL_COUNT
        panels[p] .*= factor
    end
    return panels
end

function _geos_prepare_window_for_steps!(workspace::GEOSCubedSphereWindowWorkspace{FT},
                                         grid::CubedSphereTargetGeometry,
                                         steps::Int) where FT
    Nc = grid.Nc
    Nz = size(workspace.m_cur[1], 3)
    for p in 1:CS_PANEL_COUNT
        apply_vertical!(workspace.am_v4[p], workspace.am_native_v4[p],
                        workspace.plan, MassFluxField())
        apply_vertical!(workspace.bm_v4[p], workspace.bm_native_v4[p],
                        workspace.plan, MassFluxField())
    end
    if steps != workspace.source_steps_per_met
        factor = FT(workspace.source_steps_per_met / steps)
        _scale_cs_flux_panels!(workspace.am_v4, factor)
        _scale_cs_flux_panels!(workspace.bm_v4, factor)
    end
    workspace.flux_scale = workspace.base_flux_scale *
                           (workspace.source_steps_per_met / steps)

    bal_diag = balance_cs_column_mass_fluxes!(
        workspace.am_v4, workspace.bm_v4, workspace.m_cur,
        workspace.m_next_target, grid.face_table, grid.cell_degree, steps,
        grid.poisson_scratch)
    fill_cs_window_mass_tendency!(workspace.dm_v4, workspace.m_cur,
                                  workspace.m_next_target, steps)
    for p in 1:CS_PANEL_COUNT
        fill!(workspace.cm_v4[p], zero(FT))
    end
    diagnose_cs_cm!(workspace.cm_v4, workspace.am_v4, workspace.bm_v4,
                    workspace.dm_v4, workspace.m_cur, Nc, Nz)
    return bal_diag
end

function _geos_required_steps(steps::Int, ratio::Float64, cfl_target::Float64,
                              min_steps::Int, max_steps::Int)
    required = if isfinite(ratio)
        ceil(Int, steps * ratio / cfl_target)
    else
        max_steps
    end
    return _geos_clamp_steps(max(required, 1), min_steps, max_steps)
end

function _geos_select_steps_for_window!(workspace::GEOSCubedSphereWindowWorkspace,
                                        grid::CubedSphereTargetGeometry,
                                        win::Int)
    steps = _geos_clamp_steps(workspace.source_steps_per_met,
                              workspace.min_steps_per_window,
                              workspace.max_steps_per_window)
    bal_diag = nothing
    positivity = nothing
    prepared_steps = 0
    if workspace.adaptive_substeps
        for _ in 1:_GEOS_ADAPTIVE_SUBSTEP_MAX_REFINEMENTS
            bal_diag = _geos_prepare_window_for_steps!(workspace, grid, steps)
            prepared_steps = steps
            positivity = verify_substep_positivity_cs!(
                workspace.m_cur, workspace.am_v4, workspace.bm_v4,
                workspace.cm_v4; cfl_limit = workspace.substep_cfl_target,
                m_next = workspace.m_next_target)
            next_steps = _geos_required_steps(
                steps, Float64(positivity.ratio), workspace.substep_cfl_target,
                workspace.min_steps_per_window, workspace.max_steps_per_window)
            next_steps == steps && break
            steps = next_steps
        end
        if prepared_steps != steps
            bal_diag = _geos_prepare_window_for_steps!(workspace, grid, steps)
        end
    else
        bal_diag = _geos_prepare_window_for_steps!(workspace, grid, steps)
    end
    workspace.steps_current = steps
    1 <= win <= length(workspace.steps_schedule) ||
        throw(ArgumentError("GEOS steps_schedule length $(length(workspace.steps_schedule)) " *
                            "cannot record window $(win)."))
    workspace.steps_schedule[win] = steps
    return (steps = steps, balance = bal_diag, positivity = positivity)
end

function ingest_window!(workspace::GEOSCubedSphereWindowWorkspace{FT},
                        reader::GEOSNativeReader{FT},
                        win::Int,
                        grid::CubedSphereTargetGeometry,
                        settings::AbstractGEOSSettings,
                        vertical) where FT
    Nc = grid.Nc
    Nz = vertical.Nz
    Nz_native = vertical.Nz_native
    read_window!(workspace.raw, reader, win)
    workspace.flux_scale = workspace.base_flux_scale
    workspace.steps_current = workspace.source_steps_per_met

    _geos_fluxes_to_target!(workspace.strategy, workspace.strategy_ws,
                            workspace.am_native_v4, workspace.bm_native_v4,
                            workspace.raw, grid, Nc, Nz_native,
                            workspace.flux_scale)
    for p in 1:CS_PANEL_COUNT
        apply_vertical!(workspace.am_v4[p], workspace.am_native_v4[p],
                        workspace.plan, MassFluxField())
        apply_vertical!(workspace.bm_v4[p], workspace.bm_native_v4[p],
                        workspace.plan, MassFluxField())
    end

    if win == 1 || !workspace.chain_mass
        if workspace.chain_mass && win == 1 && reader.seed !== nothing
            for p in 1:CS_PANEL_COUNT
                size(reader.seed[p]) == (Nc, Nc, Nz) ||
                    error("seed_m[$p] shape $(size(reader.seed[p])) ≠ ($Nc, $Nc, $Nz)")
                copyto!(workspace.m_cur[p], reader.seed[p])
            end
        else
            _geos_seed_mass!(workspace.strategy, workspace.strategy_ws,
                             workspace.m_native_kg, workspace.raw,
                             workspace.cell_areas, workspace.inv_g, Nc, Nz_native)
            for p in 1:CS_PANEL_COUNT
                apply_vertical!(workspace.m_cur[p], workspace.m_native_kg[p],
                                workspace.plan, MassField())
            end
        end
        for p in 1:CS_PANEL_COUNT
            _ps_from_air_mass!(workspace.ps_cur[p], workspace.m_cur[p],
                               workspace.cell_areas, workspace.g, Nc, Nz)
        end
    end

    _geos_target_mass!(workspace.strategy, workspace.strategy_ws,
                       workspace.m_native_kg, workspace.raw,
                       workspace.cell_areas, workspace.inv_g, Nc, Nz_native)
    for p in 1:CS_PANEL_COUNT
        apply_vertical!(workspace.m_next_target[p], workspace.m_native_kg[p],
                        workspace.plan, MassField())
    end
    _geos_select_steps_for_window!(workspace, grid, win)
    return nothing
end

function drain_ready_windows!(workspace::GEOSCubedSphereWindowWorkspace{FT},
                              contract::CubedSphereContract{FT},
                              win::Int,
                              grid::CubedSphereTargetGeometry,
                              settings::AbstractGEOSSettings,
                              steps_per_met::Int) where FT
    steps = workspace.steps_current
    contract.steps_per_window = steps
    contract_diag = verify_window!((m_cur = workspace.m_cur,
                                     am = workspace.am_v4,
                                     bm = workspace.bm_v4,
                                     cm = workspace.cm_v4,
                                     m_next = workspace.m_next_target),
                                    contract, win)

    for p in 1:CS_PANEL_COUNT
        copyto!(workspace.dm_v4[p], workspace.m_next_target[p])
    end
    convert_cs_mass_target_to_delta!(workspace.dm_v4, workspace.m_cur)

    surface_payload = (settings.include_surface || settings.include_vdiff_fields) ?
        _geos_surface_payload!(workspace.strategy, workspace.strategy_ws,
                               workspace.raw) : nothing
    cmfmc_payload = settings.include_convection ? _geos_cmfmc_payload!(workspace) : nothing
    dtrain_payload = settings.include_convection ? _geos_dtrain_payload!(workspace) : nothing
    vdiff_payload = settings.include_vdiff_fields ? _geos_vdiff_payload!(workspace) : nothing
    window_nt = (m = workspace.m_cur, am = workspace.am_v4,
                 bm = workspace.bm_v4, cm = workspace.cm_v4,
                 ps = workspace.ps_cur, dm = workspace.dm_v4,
                 surface = surface_payload,
                 cmfmc = cmfmc_payload,
                 dtrain = dtrain_payload,
                 vdiff = vdiff_payload)
    ready = ReadyWindow{CubedSphereTargetGeometry, FT}(win, window_nt)
    return PreverifiedWindow(ready, contract_diag)
end

function advance_window!(workspace::GEOSCubedSphereWindowWorkspace,
                         grid::CubedSphereTargetGeometry)
    workspace.chain_mass || return nothing
    Nc = grid.Nc
    Nz = size(workspace.m_cur[1], 3)
    for p in 1:CS_PANEL_COUNT
        copyto!(workspace.m_cur[p], workspace.m_next_target[p])
        _ps_from_air_mass!(workspace.ps_cur[p], workspace.m_cur[p],
                           workspace.cell_areas, workspace.g, Nc, Nz)
    end
    return nothing
end

struct GEOSReplayStats
    worst_replay_rel :: Float64
    worst_replay_abs :: Float64
    worst_replay_win :: Int
end

GEOSReplayStats() = GEOSReplayStats(0.0, 0.0, 0)

struct GEOSCSUnifiedDriverContext{G, S, V}
    grid             :: G
    settings         :: S
    vertical         :: V
    steps_per_met    :: Int
    replay_stats     :: Base.RefValue{GEOSReplayStats}
end

GEOSCSUnifiedDriverContext(grid, settings, vertical, steps_per_met::Integer) =
    GEOSCSUnifiedDriverContext{typeof(grid), typeof(settings), typeof(vertical)}(
        grid, settings, vertical, Int(steps_per_met), Ref(GEOSReplayStats()))

function driver_ingest_window!(workspace::GEOSCubedSphereWindowWorkspace{FT},
                               reader::GEOSNativeReader{FT},
                               win::Int,
                               ctx::GEOSCSUnifiedDriverContext) where FT
    return ingest_window!(workspace, reader, win, ctx.grid, ctx.settings, ctx.vertical)
end

function driver_drain_ready_windows!(workspace::GEOSCubedSphereWindowWorkspace{FT},
                                     contract::CubedSphereContract{FT},
                                     win::Int,
                                     ctx::GEOSCSUnifiedDriverContext) where FT
    ready_diag = drain_ready_windows!(workspace, contract, win, ctx.grid,
                                      ctx.settings, ctx.steps_per_met)
    replay = ready_diag.contract.replay
    stats = ctx.replay_stats[]
    if stats.worst_replay_win == 0 || replay.max_rel_err > stats.worst_replay_rel
        ctx.replay_stats[] = GEOSReplayStats(replay.max_rel_err,
                                             replay.max_abs_err,
                                             win)
    end
    return ready_diag
end

function driver_flush_final_windows!(::GEOSCubedSphereWindowWorkspace,
                                     ::GEOSNativeReader,
                                     ::CubedSphereContract,
                                     ::GEOSCSUnifiedDriverContext)
    return ()
end

function driver_before_close_writer!(workspace::GEOSCubedSphereWindowWorkspace,
                                     _reader::GEOSNativeReader,
                                     _contract::CubedSphereContract,
                                     writer::CubedSphereBinaryWriter,
                                     _ctx::GEOSCSUnifiedDriverContext)
    set_streaming_steps_per_window_schedule!(writer.inner, workspace.steps_schedule)
    return nothing
end

function driver_after_write_window!(workspace::GEOSCubedSphereWindowWorkspace,
                                    _reader::GEOSNativeReader,
                                    _ready::ReadyWindow,
                                    ctx::GEOSCSUnifiedDriverContext)
    return advance_window!(workspace, ctx.grid)
end

function _process_day_geos_cs_unified(date::Date,
                                      grid::CubedSphereTargetGeometry,
                                      settings::AbstractGEOSSettings,
                                      vertical;
                                      out_path::AbstractString,
                                      dt_met_seconds::Real,
                                      FT::Type{<:AbstractFloat},
                                      mass_basis::Symbol,
                                      replay_tol::Real,
                                      positivity_cfl_limit::Real,
                                      require_substep_positivity::Bool,
                                      adaptive_substeps::Bool,
                                      substep_cfl_target::Real,
                                      min_steps_per_window::Integer,
                                      max_steps_per_window::Integer,
                                      chain_mass::Bool,
                                      seed_m::Union{Nothing, NTuple{6, <:AbstractArray}})
    Nc     = grid.Nc
    npanel = CS_PANEL_COUNT
    Nz     = vertical.Nz
    vc     = vertical.merged_vc
    panel_convention = "geos_native"

    steps_per_met = round(Int, FT(dt_met_seconds) / FT(settings.mass_flux_dt))
    reader_seed = seed_m === nothing ? nothing :
        ntuple(p -> Array{FT, 3}(seed_m[p]), CS_PANEL_COUNT)
    reader = open_reader(settings, date, FT;
                         seed = reader_seed,
                         chain_mass = chain_mass,
                         next_day_handle = true)
    driver_started = false
    inner_writer = nothing
    tmp_path = out_path * ".tmp"

    try
        nw = windows_per_day(reader)
        workspace = allocate_window_workspace(grid, settings, vertical, FT;
                                               dt_met_seconds = dt_met_seconds,
                                               chain_mass = chain_mass,
                                               adaptive_substeps = adaptive_substeps,
                                               substep_cfl_target = substep_cfl_target,
                                               min_steps_per_window = min_steps_per_window,
                                               max_steps_per_window = max_steps_per_window,
                                               windows_per_day = nw)

        @info "GEOS → CS: $(date), source=$(settings) → $(out_path) [unified]"
        @info "  source_C=$(settings.Nc) target_C=$Nc  strategy=$(_geos_cs_strategy_name(workspace.strategy))"
        @info "  Nz=$Nz  windows=$nw  steps_per_met=$steps_per_met  flux_scale=$(workspace.flux_scale)"
        adaptive_substeps &&
            @info "  Adaptive substeps: target CFL=$(Float64(substep_cfl_target)) bounds=$(Int(min_steps_per_window)):$(Int(max_steps_per_window))"
        @info "  Level orientation: $(reader.handles.orientation)  (next-day endpoint: $(_geos_next_endpoint_available(reader.handles)))"

        mkpath(dirname(out_path))
        isfile(tmp_path) && rm(tmp_path; force = true)

        inner_writer = open_streaming_cs_transport_binary(
            tmp_path, Nc, npanel, Nz, nw, vc;
            FT = FT,
            dt_met_seconds = dt_met_seconds,
            steps_per_window = steps_per_met,
            mass_basis = mass_basis,
            include_flux_delta = true,
            include_surface    = settings.include_surface || settings.include_vdiff_fields,
            include_cmfmc      = settings.include_convection,
            include_dtrain     = settings.include_convection,
            include_gchp_vdiff = settings.include_vdiff_fields,
            panel_convention   = panel_convention,
            cs_definition      = _cs_definition_tag(grid),
            cs_coordinate_law  = _cs_coordinate_law_tag(grid),
            cs_center_law      = _cs_center_law_tag(grid),
            longitude_offset_deg = longitude_offset_deg(cs_definition(grid.mesh)),
            extra_header = Dict{String, Any}(
                "preprocessor" => "geos_native_to_cs",
                "preprocessor_contract" => "plan41_variable_substeps",
                "runtime_substep_contract" => "binary_schedule",
                "geos_mass_endpoint" => "raw_dry_endpoint",
                "geos_horizontal_balance" => "column_poisson_to_raw_endpoint",
                "geos_vertical_flux" => "diagnosed_from_balanced_horizontal_and_raw_endpoint",
                "source_Nc" => settings.Nc,
                "geos_cs_resolution_strategy" => _geos_cs_strategy_name(workspace.strategy),
                "source_steps_per_window" => steps_per_met,
                "adaptive_substeps" => adaptive_substeps,
                "substep_cfl_target" => Float64(substep_cfl_target),
                "positivity_cfl_limit" => Float64(positivity_cfl_limit),
                "require_substep_positivity" => require_substep_positivity,
                "include_gchp_vdiff" => settings.include_vdiff_fields,
                "gchp_vdiff_source_fields" => settings.include_vdiff_fields ?
                    "A3dyn:U,V + I3:T + CTM_I1:QV + A1:PBLH,USTAR,HFLUX,T2M" : "none",
                "gchp_vdiff_sampling" => settings.include_vdiff_fields ?
                    "A3/I3 held constant over 3 hourly windows; QV uses left CTM_I1 endpoint" : "none",
                "vertical_transform" => String(Symbol(get(vertical, :vertical_mapping_method, :identity))),
                "vertical_Nz_native" => vertical.Nz_native,
                "vertical_Nz_output" => vertical.Nz,
            ),
        )
        writer = CubedSphereBinaryWriter(inner_writer, DryBasis();
                                         Nc = Nc, npanel = npanel,
                                         final_path = out_path)
        window_contract = CubedSphereContract{FT}(
            replay_tol = replay_tol,
            positivity_cfl_limit = positivity_cfl_limit,
            require_substep_positivity = require_substep_positivity,
            steps_per_window = steps_per_met,
        )
        ctx = GEOSCSUnifiedDriverContext(grid, settings, vertical, steps_per_met)

        t_start = time()
        driver_started = true
        driver_result = run_unified_preprocessor_day!(
            UnifiedPreprocessorDay(reader, workspace, window_contract, writer;
                                   context = ctx))
        elapsed = time() - t_start
        stats = ctx.replay_stats[]
        @info @sprintf("  Done in %.1fs (%.2fs/window). Worst replay: rel=%.2e abs=%.2e at win=%d",
                       elapsed, elapsed / nw, stats.worst_replay_rel,
                       stats.worst_replay_abs, stats.worst_replay_win)

        final_m = chain_mass ? ntuple(p -> copy(workspace.m_cur[p]), npanel) : nothing
        set_end_of_day_seed!(reader, final_m)

        return (
            elapsed = elapsed,
            worst_replay_rel = stats.worst_replay_rel,
            worst_replay_abs = stats.worst_replay_abs,
            worst_replay_win = stats.worst_replay_win,
            out_path = driver_result.out_path,
            steps_per_window_by_window = copy(workspace.steps_schedule),
            final_m = final_m,
        )
    finally
        if !driver_started
            if inner_writer !== nothing
                try
                    close_streaming_transport_binary!(inner_writer)
                catch err
                    @warn("Unified GEOS-CS: failed to close writer during cleanup",
                          exception = (err, catch_backtrace()))
                end
            end
            close_reader!(reader)
            isfile(tmp_path) && rm(tmp_path; force = true)
        end
    end
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

Stored mass targets the raw GEOS dry endpoint (`DELP_dry`) transformed to the
output vertical grid. The native horizontal fluxes are column-balanced to that
endpoint, then `cm` is diagnosed so the replay and positivity contracts are
checked against the same endpoint the runtime will see.

For multi-day preprocessing with `chain_mass = true`, `seed_m` carries the
raw endpoint from the previous day so adjacent daily binaries share a boundary
mass: pass `nothing` (default) on day 1 to seed from raw GEOS DELP_dry, and on
day N+1 pass the `final_m` returned by day N's `process_day`. With
`chain_mass = false`, `seed_m` is ignored and every window reinitializes from
raw GEOS mass.

When `chain_mass = true`, the returned NamedTuple includes
`final_m::NTuple{6, Array{FT, 3}}`, the raw-endpoint state at the END of the
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
                     adaptive_substeps::Bool = false,
                     substep_cfl_target::Real = positivity_cfl_limit,
                     min_steps_per_window::Integer = 1,
                     max_steps_per_window::Integer = typemax(Int),
                     chain_mass::Bool = true,
                     seed_m::Union{Nothing, NTuple{6, <:AbstractArray}} = nothing,
                     next_day_hour0 = nothing)
    # Reject configurations the path cannot honor:
    mass_basis === :dry ||
        error("GEOS-CS passthrough only supports mass_basis=:dry; got $(mass_basis). " *
              "GEOS MFXC/MFYC are already dry; the chained pressure-fixer is dry-basis.")
    _validate_geos_native_panel_convention(grid.mesh.convention)
    return _process_day_geos_cs_unified(
        date, grid, settings, vertical;
        out_path = out_path,
        dt_met_seconds = dt_met_seconds,
        FT = FT,
        mass_basis = mass_basis,
        replay_tol = replay_tol,
        positivity_cfl_limit = positivity_cfl_limit,
        require_substep_positivity = require_substep_positivity,
        adaptive_substeps = adaptive_substeps,
        substep_cfl_target = substep_cfl_target,
        min_steps_per_window = min_steps_per_window,
        max_steps_per_window = max_steps_per_window,
        chain_mass = chain_mass,
        seed_m = seed_m,
    )
end
