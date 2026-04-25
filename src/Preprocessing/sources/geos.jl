# ===========================================================================
# Native GEOS-IT / GEOS-FP NetCDF reader for the preprocessor.
#
# GEOS data lives on a cubed-sphere grid (C180 for GEOS-IT, C720 for GEOS-FP),
# 72 hybrid sigma-pressure levels, daily files split by collection:
#
#   CTM_A1   hourly  MFXC, MFYC, DELP   (window-averaged horizontal mass flux,
#                                       window-averaged DELP)
#   CTM_I1   hourly  PS, QV             (instantaneous endpoints)
#   A3dyn    3-hr    DTRAIN, U, V       (held constant in 3-hr blocks; convection)
#   A3mstE   3-hr    CMFMC              (convection, edge-based)
#
# Critical conventions baked into this reader:
#
#   * `mass_flux_dt = 450 s` is the FV3 dynamics timestep over which MFXC and
#     MFYC are accumulated — NOT the 1-hour archive cadence. Forgetting this
#     scaling produces an 8x slowdown of transport.
#
#   * GEOS-IT stores levels bottom-to-top (k=1 = surface). GEOS-FP stores them
#     top-to-bottom. Both are flipped (where needed) to AtmosTransport's
#     top-to-bottom convention. Auto-detection compares DELP[k=1] vs
#     DELP[k=Nz]; the surface side has the larger pressure thickness.
#
#   * DELP, MFXC, MFYC, CMFMC and DTRAIN in the GEOS archive are MOIST. PS is
#     also moist (total surface pressure). The reader converts to DRY here
#     so the rest of the pipeline (which is dry-basis by contract) consumes
#     consistent fields. MFXC and MFYC are converted layer-by-layer using the
#     window-averaged QV.
#
#   * `m` and `m_next` in the produced `RawWindow` are DELP_dry at the two
#     window endpoints, reconstructed from PS_dry via the hybrid coordinate.
#     This guarantees Σm_k = ps at every endpoint, which is what the v4
#     binary write-time replay gate needs.
# ===========================================================================

abstract type AbstractGEOSSettings <: AbstractMetSettings end

"""
    GEOSSettings{flavor} <: AbstractGEOSSettings

Settings for one of the two supported GEOS flavors:

- `flavor = :geosit` — GEOS-IT (file pattern `GEOSIT.{date}.{collection}.C{Nc}.nc`).
- `flavor = :geosfp` — GEOS-FP (file pattern `GEOSFP.{date}.{collection}.C{Nc}.nc`).

Auto-detection of level orientation runs at `open_geos_day` time when
`level_orientation = :auto`. Set explicitly to `:bottom_up` or `:top_down`
to skip the heuristic.
"""
Base.@kwdef struct GEOSSettings{flavor} <: AbstractGEOSSettings
    root_dir            :: String
    Nc                  :: Int
    mass_flux_dt        :: Float64 = 450.0
    level_orientation   :: Symbol  = :auto    # :auto, :bottom_up, :top_down
    include_convection  :: Bool    = false
    coefficients_file   :: String  = "config/geos_L72_coefficients.toml"
end

const GEOSITSettings = GEOSSettings{:geosit}
const GEOSFPSettings = GEOSSettings{:geosfp}

# ---------------------------------------------------------------------------
# File-naming dispatch on flavor.
# ---------------------------------------------------------------------------

_geos_prefix(::GEOSSettings{:geosit}) = "GEOSIT"
_geos_prefix(::GEOSSettings{:geosfp}) = "GEOSFP"

"""
    geos_collection_path(settings, date::Date, collection::AbstractString) -> String

Resolve the on-disk path of one collection for `date`. Searches a flat
`root_dir` and the per-day `root_dir/YYYYMMDD/` layout.
"""
function geos_collection_path(settings::GEOSSettings, date::Date, collection::AbstractString)
    datestr = Dates.format(date, "yyyymmdd")
    fname   = "$(_geos_prefix(settings)).$(datestr).$(collection).C$(settings.Nc).nc"
    flat    = joinpath(settings.root_dir, fname)
    daily   = joinpath(settings.root_dir, datestr, fname)
    isfile(flat)  && return flat
    isfile(daily) && return daily
    error("GEOS file not found: tried $flat and $daily")
end

# ---------------------------------------------------------------------------
# Day handles.
# ---------------------------------------------------------------------------

"""
    GEOSDayHandles

Open NCDataset handles for one UTC day's GEOS collections plus the resolved
level orientation and the hybrid coordinate (loaded once for endpoint-DELP
reconstruction). The orchestrator opens these once at the start of
`process_day` and closes them at the end.
"""
mutable struct GEOSDayHandles{V <: HybridSigmaPressure}
    ctm_a1      :: NCDataset
    ctm_i1      :: NCDataset
    next_ctm_i1 :: Union{Nothing, NCDataset}
    a3dyn       :: Union{Nothing, NCDataset}
    a3mste      :: Union{Nothing, NCDataset}
    orientation :: Symbol                          # :bottom_up or :top_down
    vc          :: V                               # hybrid sigma-pressure (top-down)
end

"""
    open_geos_day(settings, date; next_day_handle=true) -> GEOSDayHandles

Open per-collection NCDataset handles for `date`. When `next_day_handle` is
`true` and the next-day CTM_I1 file exists, opens it too so the last window
of `date` has its right endpoint available.
"""
function open_geos_day(settings::GEOSSettings, date::Date;
                       next_day_handle::Bool=true)
    ctm_a1 = NCDataset(geos_collection_path(settings, date, "CTM_A1"), "r")
    ctm_i1 = NCDataset(geos_collection_path(settings, date, "CTM_I1"), "r")

    next_ctm_i1 = nothing
    if next_day_handle
        try
            next_ctm_i1 = NCDataset(geos_collection_path(settings, date + Day(1), "CTM_I1"), "r")
        catch
            # Last day of the available archive — no next-day endpoint.
        end
    end

    a3dyn  = nothing
    a3mste = nothing
    if settings.include_convection
        a3dyn  = NCDataset(geos_collection_path(settings, date, "A3dyn"),  "r")
        a3mste = NCDataset(geos_collection_path(settings, date, "A3mstE"), "r")
    end

    orientation = settings.level_orientation === :auto ?
                  detect_level_orientation(ctm_a1) :
                  settings.level_orientation

    vc = load_hybrid_coefficients(expanduser(settings.coefficients_file))

    return GEOSDayHandles(ctm_a1, ctm_i1, next_ctm_i1, a3dyn, a3mste, orientation, vc)
end

"""Close all handles. Idempotent."""
function close_geos_day!(handles::GEOSDayHandles)
    close(handles.ctm_a1)
    close(handles.ctm_i1)
    handles.next_ctm_i1 === nothing || close(handles.next_ctm_i1)
    handles.a3dyn       === nothing || close(handles.a3dyn)
    handles.a3mste      === nothing || close(handles.a3mste)
    return nothing
end

# ---------------------------------------------------------------------------
# Level-orientation auto-detection.
# DELP[k=1] dominates DELP[k=Nz] when k=1 is the surface — bottom-to-top.
# ---------------------------------------------------------------------------

"""
    detect_level_orientation(ctm_a1::NCDataset) -> Symbol

Return `:bottom_up` if k=1 is the surface (mass-thicker) and `:top_down`
if k=1 is TOA. Heuristic is unambiguous: surface DELP is O(1000 Pa),
TOA DELP is O(1 Pa).
"""
function detect_level_orientation(ctm_a1::NCDataset)
    delp = ctm_a1["DELP"]
    Nz = size(delp, 4)
    delp_top    = mean(skipmissing(delp[:, :, :, 1, 1]))
    delp_bottom = mean(skipmissing(delp[:, :, :, Nz, 1]))
    return delp_top > delp_bottom ? :bottom_up : :top_down
end

# ---------------------------------------------------------------------------
# Per-panel array slicers.
# ---------------------------------------------------------------------------

"""Read one window of a 3D field as `NTuple{6, Array{FT,3}}`, level-flipped if needed."""
function _read_panels_3d(var, win_idx::Int, orientation::Symbol; FT::Type)
    raw = Array{FT}(var[:, :, :, :, win_idx])     # (Nc, Nc, 6, Nz)
    if orientation === :bottom_up
        return ntuple(p -> reverse(raw[:, :, p, :]; dims=3), 6)
    else
        return ntuple(p -> Array(raw[:, :, p, :]), 6)
    end
end

"""Read one window of a 2D field as `NTuple{6, Matrix{FT}}`."""
function _read_panels_2d(var, win_idx::Int; FT::Type)
    raw = Array{FT}(var[:, :, :, win_idx])        # (Nc, Nc, 6)
    return ntuple(p -> Array(raw[:, :, p]), 6)
end

# ---------------------------------------------------------------------------
# Endpoint dry-basis reconstruction.
# Given PS_total (Pa) and QV (kg/kg) at one hour, plus the hybrid
# coordinate, build (PS_dry, DELP_dry) consistent with Σ DELP_dry = PS_dry.
# ---------------------------------------------------------------------------

"""
    endpoint_dry_mass!(delp_dry, ps_dry, ps_total, qv, vc) -> (delp_dry, ps_dry)

Reconstruct dry DELP and dry PS at one endpoint hour from the moist PS and
QV provided by GEOS, using the hybrid coordinate `vc`. The output is on
top-down level convention (k=1 = TOA).

Algorithm:

    DELP_full[k] = ΔA[k] + ΔB[k] * PS_total
    DELP_dry[k]  = (1 - QV[k]) * DELP_full[k]
    PS_dry       = Σ DELP_dry[k]

In-place: writes into `delp_dry` and `ps_dry`.
"""
function endpoint_dry_mass!(delp_dry::AbstractArray{FT,3},
                            ps_dry::AbstractMatrix{FT},
                            ps_total::AbstractMatrix{FT},
                            qv::AbstractArray{FT,3},
                            vc::HybridSigmaPressure) where {FT}
    Nx, Ny, Nz = size(qv)
    @assert size(delp_dry) == size(qv)
    @assert size(ps_dry)   == size(ps_total) == (Nx, Ny)
    A = vc.A
    B = vc.B
    @assert length(A) == length(B) == Nz + 1

    @inbounds for j in 1:Ny, i in 1:Nx
        ps_total_ij = ps_total[i, j]
        ps_dry_ij = zero(FT)
        @simd for k in 1:Nz
            ΔA = FT(A[k+1] - A[k])
            ΔB = FT(B[k+1] - B[k])
            delp_full = ΔA + ΔB * ps_total_ij
            delp_dry_k = (1 - qv[i, j, k]) * delp_full
            delp_dry[i, j, k] = delp_dry_k
            ps_dry_ij += delp_dry_k
        end
        ps_dry[i, j] = ps_dry_ij
    end
    return delp_dry, ps_dry
end

"""Allocate (delp_dry, ps_dry) for one panel and run `endpoint_dry_mass!`."""
function endpoint_dry_mass(ps_total::AbstractMatrix{FT}, qv::AbstractArray{FT,3},
                           vc::HybridSigmaPressure) where {FT}
    delp_dry = similar(qv)
    ps_dry   = similar(ps_total)
    endpoint_dry_mass!(delp_dry, ps_dry, ps_total, qv, vc)
    return delp_dry, ps_dry
end

# ---------------------------------------------------------------------------
# Window-mean dry conversion of horizontal mass flux.
# MFXC and MFYC are moist (Pa·m² accumulated); convert per layer using the
# window-mean QV and divide by mass_flux_dt to get kg-equivalent /s.
#
# We use `(QV_left + QV_right)/2` as the window-mean approximation. The
# orchestrator could refine later.
# ---------------------------------------------------------------------------

function _scale_and_dry!(am::AbstractArray{FT,3},
                         qv_window_mean::AbstractArray{FT,3},
                         mfxc_raw::AbstractArray{FT,3},
                         inv_dt::FT) where {FT}
    @inbounds @simd for i in eachindex(am)
        am[i] = mfxc_raw[i] * (1 - qv_window_mean[i]) * inv_dt
    end
    return am
end

# ---------------------------------------------------------------------------
# AbstractMetSettings interface.
# ---------------------------------------------------------------------------

windows_per_day(::GEOSSettings, ::Date) = 24
has_convection(s::GEOSSettings) = s.include_convection

"""
    read_window!(settings, handles, date, win_idx; FT=Float64) -> RawWindow

Build a `RawWindow` for window `win_idx` of `date` from open `handles`.
Both endpoints (t_n, t_{n+1}) carry dry DELP + dry PS reconstructed from
PS_total via the hybrid coordinate, plus the original QV (used downstream
for tracer mass-basis conversion). Window-integrated `am`/`bm` are MFXC and
MFYC, converted to dry mass-equivalent and divided by `mass_flux_dt`.
"""
function read_window!(settings::GEOSSettings, handles::GEOSDayHandles,
                      date::Date, win_idx::Int;
                      FT::Type = Float64)
    nw = windows_per_day(settings, date)
    1 <= win_idx <= nw || error("window $win_idx out of range 1..$nw")

    or = handles.orientation
    vc = handles.vc

    # ---- Endpoints: PS (hPa→Pa), QV ----
    ps_n_total = _read_panels_2d(handles.ctm_i1, "PS", win_idx; FT=FT) |>
                 t -> ntuple(p -> t[p] .* FT(100), 6)
    qv_n       = _read_panels_3d(handles.ctm_i1, "QV", win_idx, or; FT=FT)

    if win_idx < nw
        ps_np1_total = _read_panels_2d(handles.ctm_i1, "PS", win_idx + 1; FT=FT) |>
                       t -> ntuple(p -> t[p] .* FT(100), 6)
        qv_np1       = _read_panels_3d(handles.ctm_i1, "QV", win_idx + 1, or; FT=FT)
    elseif handles.next_ctm_i1 !== nothing
        ps_np1_total = _read_panels_2d(handles.next_ctm_i1, "PS", 1; FT=FT) |>
                       t -> ntuple(p -> t[p] .* FT(100), 6)
        qv_np1       = _read_panels_3d(handles.next_ctm_i1, "QV", 1, or; FT=FT)
    else
        error("last window ($win_idx of $nw) has no next-day CTM_I1 endpoint; " *
              "open the day with `next_day_handle=true` and ensure the next-day file is on disk")
    end

    # ---- Endpoint dry mass (Σ DELP_dry = PS_dry guaranteed) ----
    m_n_panels       = ntuple(p -> begin
                                   d, _   = endpoint_dry_mass(ps_n_total[p], qv_n[p], vc); d
                               end, 6)
    ps_n_dry_panels  = ntuple(p -> begin
                                   _, ps  = endpoint_dry_mass(ps_n_total[p], qv_n[p], vc); ps
                               end, 6)
    m_np1_panels     = ntuple(p -> begin
                                   d, _   = endpoint_dry_mass(ps_np1_total[p], qv_np1[p], vc); d
                               end, 6)
    ps_np1_dry_panels = ntuple(p -> begin
                                   _, ps  = endpoint_dry_mass(ps_np1_total[p], qv_np1[p], vc); ps
                               end, 6)

    # ---- Window-integrated horizontal mass flux (dry, /mass_flux_dt) ----
    inv_dt = inv(FT(settings.mass_flux_dt))
    mfxc_raw = _read_panels_3d(handles.ctm_a1, "MFXC", win_idx, or; FT=FT)
    mfyc_raw = _read_panels_3d(handles.ctm_a1, "MFYC", win_idx, or; FT=FT)

    am_panels = ntuple(p -> begin
                            am = similar(mfxc_raw[p])
                            qv_avg = (qv_n[p] .+ qv_np1[p]) .* FT(0.5)
                            _scale_and_dry!(am, qv_avg, mfxc_raw[p], inv_dt)
                        end, 6)
    bm_panels = ntuple(p -> begin
                            bm = similar(mfyc_raw[p])
                            qv_avg = (qv_n[p] .+ qv_np1[p]) .* FT(0.5)
                            _scale_and_dry!(bm, qv_avg, mfyc_raw[p], inv_dt)
                        end, 6)

    return RawWindow{FT, typeof(ps_n_dry_panels), typeof(m_n_panels)}(
        m_n_panels,    ps_n_dry_panels,    qv_n,
        m_np1_panels,  ps_np1_dry_panels,  qv_np1,
        am_panels,     bm_panels,
        nothing, nothing,
        nothing, nothing,
    )
end

# Per-window 2D variable handle by name (NCDataset[name]).
_read_panels_2d(ds::NCDataset, name, win_idx; FT) =
    _read_panels_2d(ds[name], win_idx; FT=FT)

_read_panels_3d(ds::NCDataset, name, win_idx, orientation; FT) =
    _read_panels_3d(ds[name], win_idx, orientation; FT=FT)
