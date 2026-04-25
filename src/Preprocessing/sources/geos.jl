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
#   * DELP and PS in the GEOS archive are MOIST (total atmosphere). MFXC and
#     MFYC are ALREADY DRY mass fluxes per GMAO and the in-tree diagnostic
#     `compare_era5_geosit_met.jl` (`am_moist = MFXC / (g·dt_dyn) / (1−qv)`).
#     The reader converts DELP and PS to dry via the hybrid coordinate plus
#     QV; MFXC and MFYC pass through unchanged (only divided by `mass_flux_dt`).
#
#   * `m` and `m_next` in the produced `RawWindow` are DELP_dry at the two
#     window endpoints, reconstructed from PS_total via the hybrid coordinate.
#     `Σ m_k = ps_dry` at every endpoint to roundoff. The orchestrator does
#     NOT use these directly as the v4 binary's stored mass — for native
#     GEOS sources the stored mass is the FV3 pressure-fixer's chained
#     evolution from the first hour's DELP_dry, governed by the stored
#     fluxes (see `cubed_sphere_geos.jl`). The raw endpoint values remain
#     useful for diagnostics and for the first-window initialization.
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
#
# GEOS-IT C180 archive uses *daily* files: `GEOSIT.YYYYMMDD.<collection>.C180.nc`.
# GEOS-FP C720 native archive uses *hourly* files (one file per UTC hour):
# `GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.YYYYMMDD_HHMM.V01.nc4`. Wiring the
# GEOS-FP hourly file pattern into the day-handle abstraction is non-trivial
# (each window references a different physical file, so opening "the day" is
# really opening 24 files) and is deferred to a follow-up commit. The
# GEOSFPSettings dispatch below errors loudly so a misconfigured caller fails
# before silently producing wrong data.
# ---------------------------------------------------------------------------

"""
    geos_collection_path(settings::GEOSITSettings, date::Date, collection) -> String

Resolve the on-disk path of one GEOS-IT collection for `date`. Searches a
flat `root_dir` and the per-day `root_dir/YYYYMMDD/` layout.
"""
function geos_collection_path(settings::GEOSSettings{:geosit}, date::Date,
                              collection::AbstractString)
    datestr = Dates.format(date, "yyyymmdd")
    fname   = "GEOSIT.$(datestr).$(collection).C$(settings.Nc).nc"
    flat    = joinpath(settings.root_dir, fname)
    daily   = joinpath(settings.root_dir, datestr, fname)
    isfile(flat)  && return flat
    isfile(daily) && return daily
    error("GEOS-IT file not found: tried $flat and $daily")
end

function geos_collection_path(::GEOSSettings{:geosfp}, ::Date, ::AbstractString)
    error("GEOS-FP native file pattern (hourly `GEOS.fp.asm.tavg_1hr_ctm_*.nc4`) " *
          "is not yet implemented. Only GEOS-IT (daily files) is supported in this " *
          "commit; see plan indexed-baking-valiant for the GEOS-FP follow-up.")
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

"""
    _ps_pa_factor(var) -> FT scaling

GEOS-IT CTM_I1 stores PS in hPa; GEOS-FP native CTM stores PS in Pa. Read
the `units` attribute (case-insensitive) and return the multiplier needed
to land in Pa. Errors loudly on unrecognized units to prevent silent
100x errors.
"""
function _ps_pa_factor(var; FT::Type)
    units = lowercase(strip(get(var.attrib, "units", "")))
    if units == "pa"
        return FT(1)
    elseif units == "hpa" || units == "mbar" || units == "millibars"
        return FT(100)
    else
        error("Unrecognized PS units `$(units)` on GEOS NetCDF variable; expected " *
              "Pa or hPa/mbar")
    end
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
# Mass-flux scaling.
#
# MFXC and MFYC in GEOS-IT and GEOS-FP CTM files are ALREADY dry mass fluxes
# (per the in-tree diagnostic `compare_era5_geosit_met.jl` and GMAO docs:
# "GEOS am_moist = MFXC / (g × dt_dyn) / (1 − qv)"; getting MOIST from
# native MFXC requires *dividing* by `(1 − qv)`, so multiplying by it would
# double-dry). The reader therefore only divides by `mass_flux_dt` to convert
# the dynamics-step accumulated quantity to a window-mean rate; no humidity
# correction is applied here.
# ---------------------------------------------------------------------------

function _scale_flux!(am::AbstractArray{FT,3},
                      mfxc_raw::AbstractArray{FT,3},
                      inv_dt::FT) where {FT}
    @inbounds @simd for i in eachindex(am)
        am[i] = mfxc_raw[i] * inv_dt
    end
    return am
end

# ---------------------------------------------------------------------------
# AbstractMetSettings interface.
# ---------------------------------------------------------------------------

windows_per_day(::GEOSSettings, ::Date) = 24

# Per-source output-filename override: GEOS gets a clearer prefix.
_native_output_filename(::AbstractGEOSSettings, date::Date, FT::Type) =
    "geos_transport_$(Dates.format(date, "yyyymmdd"))_$(FT === Float32 ? "float32" : "float64").bin"

# `has_convection` reports the capability that downstream code can rely on,
# not the user's intent. Until Commit 8 populates `cmfmc` / `dtrain` in
# `read_window!`, this returns `false` regardless of `include_convection` so
# capability-driven dispatch (e.g. CMFMC operator wiring) does not silently
# expect data the reader does not provide. The `include_convection` settings
# field is preserved as a forward-looking flag for the Commit-8 wiring.
has_convection(::GEOSSettings) = false

# ---------------------------------------------------------------------------
# Canonical AbstractMetSettings interface implementations.
# ---------------------------------------------------------------------------

"""
    open_day(settings::GEOSSettings, date::Date; next_day_handle=true) -> GEOSDayHandles

Canonical-contract alias for `open_geos_day`. The orchestrator calls this
once per day and threads the returned handles through every per-window
`read_window!`.
"""
open_day(settings::GEOSSettings, date::Date; next_day_handle::Bool=true) =
    open_geos_day(settings, date; next_day_handle=next_day_handle)

"""Canonical-contract alias for `close_geos_day!`."""
close_day!(handles::GEOSDayHandles) = close_geos_day!(handles)

"""
    source_grid(settings::GEOSSettings) -> CubedSphereMesh

The native source mesh GEOS data is archived on (`Nc × Nc` per panel,
GEOS-native panel convention).
"""
function source_grid(settings::GEOSSettings; FT::Type{<:AbstractFloat}=Float64)
    return CubedSphereMesh(; Nc = settings.Nc, FT = FT,
                            convention = GEOSNativePanelConvention(),
                            radius = FT(R_EARTH))
end

"""
    allocate_raw_window(settings::GEOSSettings; FT, Nz) -> RawWindow

Pre-allocate a per-window workspace for the GEOS reader: 6 zero-filled
panel arrays each for `m`, `m_next`, `qv`, `qv_next`, `am`, `bm` (shape
`(Nc, Nc, Nz)`) and `ps`, `ps_next` (shape `(Nc, Nc)`). `u`, `v`,
`cmfmc`, `dtrain` stay `nothing` until they are wired in by Section C
(convection) and Section D (cross-topology).
"""
function allocate_raw_window(settings::GEOSSettings; FT::Type{<:AbstractFloat}, Nz::Int)
    Nc = settings.Nc
    panels_3d() = ntuple(_ -> zeros(FT, Nc, Nc, Nz),    6)
    panels_2d() = ntuple(_ -> zeros(FT, Nc, Nc),        6)
    m       = panels_3d(); ps      = panels_2d(); qv      = panels_3d()
    m_next  = panels_3d(); ps_next = panels_2d(); qv_next = panels_3d()
    am      = panels_3d(); bm      = panels_3d()
    return RawWindow{FT, typeof(ps), typeof(m)}(
        m,       ps,      qv,
        m_next,  ps_next, qv_next,
        am,      bm,
        nothing, nothing,
        nothing, nothing,
    )
end

"""
    read_window!(raw, settings, handles, date, win_idx) -> raw

Fill `raw` in place with one window of GEOS data on the source CS grid.
Both endpoints (t_n, t_{n+1}) carry dry DELP + dry PS reconstructed from
PS_total via the hybrid coordinate, plus the original QV. Window-
integrated `am`/`bm` are MFXC/MFYC scaled by `1/mass_flux_dt`.

The signature matches the canonical `AbstractMetSettings` contract
declared in `met_sources.jl::read_window!`.
"""
function read_window!(raw::RawWindow{FT}, settings::GEOSSettings,
                      handles::GEOSDayHandles, date::Date, win_idx::Int) where {FT}
    nw = windows_per_day(settings, date)
    1 <= win_idx <= nw || error("window $win_idx out of range 1..$nw")

    or = handles.orientation
    vc = handles.vc

    # ---- Endpoints: PS (units-aware → Pa), QV ----
    ps_factor_today = _ps_pa_factor(handles.ctm_i1["PS"]; FT=FT)
    ps_n_total = ntuple(p -> _read_panels_2d(handles.ctm_i1, "PS", win_idx; FT=FT)[p] .* ps_factor_today, 6)
    qv_n_panels = _read_panels_3d(handles.ctm_i1, "QV", win_idx, or; FT=FT)

    if win_idx < nw
        ps_np1_total = ntuple(p -> _read_panels_2d(handles.ctm_i1, "PS", win_idx + 1; FT=FT)[p] .* ps_factor_today, 6)
        qv_np1_panels = _read_panels_3d(handles.ctm_i1, "QV", win_idx + 1, or; FT=FT)
    elseif handles.next_ctm_i1 !== nothing
        ps_factor_next = _ps_pa_factor(handles.next_ctm_i1["PS"]; FT=FT)
        ps_np1_total = ntuple(p -> _read_panels_2d(handles.next_ctm_i1, "PS", 1; FT=FT)[p] .* ps_factor_next, 6)
        qv_np1_panels = _read_panels_3d(handles.next_ctm_i1, "QV", 1, or; FT=FT)
    else
        error("last window ($win_idx of $nw) has no next-day CTM_I1 endpoint; " *
              "open the day with `next_day_handle=true` and ensure the next-day file is on disk")
    end

    # ---- Window-integrated horizontal mass flux (dry, /mass_flux_dt) ----
    inv_dt = inv(FT(settings.mass_flux_dt))
    mfxc_raw = _read_panels_3d(handles.ctm_a1, "MFXC", win_idx, or; FT=FT)
    mfyc_raw = _read_panels_3d(handles.ctm_a1, "MFYC", win_idx, or; FT=FT)

    # ---- Fill `raw` in place. Endpoint dry-mass derivation lives in
    #      `endpoint_dry_mass!` and writes into the raw buffer directly. ----
    @assert raw.qv      !== nothing "GEOS RawWindow must carry qv"
    @assert raw.qv_next !== nothing "GEOS RawWindow must carry qv_next"
    for p in 1:6
        copyto!(raw.qv[p],      qv_n_panels[p])
        copyto!(raw.qv_next[p], qv_np1_panels[p])
        endpoint_dry_mass!(raw.m[p],      raw.ps[p],      ps_n_total[p],   raw.qv[p],      vc)
        endpoint_dry_mass!(raw.m_next[p], raw.ps_next[p], ps_np1_total[p], raw.qv_next[p], vc)
        _scale_flux!(raw.am[p], mfxc_raw[p], inv_dt)
        _scale_flux!(raw.bm[p], mfyc_raw[p], inv_dt)
    end

    return raw
end

# Per-window 2D variable handle by name (NCDataset[name]).
_read_panels_2d(ds::NCDataset, name, win_idx; FT) =
    _read_panels_2d(ds[name], win_idx; FT=FT)

_read_panels_3d(ds::NCDataset, name, win_idx, orientation; FT) =
    _read_panels_3d(ds[name], win_idx, orientation; FT=FT)
