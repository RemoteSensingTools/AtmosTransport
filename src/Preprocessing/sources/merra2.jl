# ===========================================================================
# Native MERRA-2 NetCDF reader for the wind-derived → CS preprocessor.
#
# MERRA-2 reproduces the validated GEOS-Chem CO₂ transport input path: derive
# horizontal mass fluxes from MERRA-2 WINDS (U/V) plus a Cameron-Smith
# column pressure-fix (the Poisson balance), instead of GEOS native
# cubed-sphere MFXC. This is purely additive — the GEOS-native and ERA5 paths
# are untouched.
#
# Data lives on a regular 0.5° × 0.625° latitude-longitude grid (576 × 361),
# 72 hybrid sigma-pressure levels (the GEOS-5 L72 coordinate, SAME as GEOS-FP),
# 3-hourly (8 windows/day), split by collection:
#
#   M2I3NVASM  inst3_3d_asm_Nv  3-hr INSTANTANEOUS  PS, QV (mass endpoints)
#   M2T3NVASM  tavg3_3d_asm_Nv  3-hr TIME-AVERAGE   U, V (advecting winds)
#
# Critical conventions baked into this reader:
#
#   * MERRA-2 stores levels TOP-DOWN (lev=1 ≈ TOA), SAME as the GEOS L72
#     coefficient table — so NO level flip is applied here (unlike the
#     GEOS-IT bottom-up reader). DELP[k=1] is the small TOA-side thickness.
#
#   * PS is in Pa, U/V in m/s, QV in kg/kg. lat is ascending (S→N), lon is
#     ascending (W→E, -180..179.375°) — both already match the project's
#     `LatLonMesh(longitude=(-180,180), latitude=(-90,90))` convention, so
#     no spatial reorientation is applied.
#
#   * NCDatasets returns dimensions REVERSED from the CDL header, so a CDL
#     `QV(time, lev, lat, lon)` is read as `ds["QV"][lon, lat, lev, time]`.
#     The window readers slice the trailing `time` axis at the 1-based window
#     index and return plain `(lon, lat[, lev])` arrays.
#
#   * Windowing contract (mirrors the GEOS-IT sliding window): window `win`
#     (1..8) endpoint dry mass uses inst3 slice `win` PS/QV; the advecting
#     winds use tavg3 slice `win` (the time AVERAGE over [3(win-1), 3win]Z).
#     The final window's right endpoint is next-day inst3 slice 1.
# ===========================================================================

"""
    MERRA2Settings <: AbstractMetSettings

Settings for the MERRA-2 wind-derived cubed-sphere preprocessor.

- `root_dir` — directory holding the MERRA-2 collection trees
  (`{root_dir}/M2{I,T}3NVASM/{YYYY}/{MM}/...nc4`).
- `coefficients_file` — hybrid σ-pressure coefficient TOML (the GEOS L72
  table; MERRA-2 shares the GEOS-5 L72 coordinate).
- `winds_collection` — `:tavg3` (time-averaged U/V, the default and the
  GEOS-Chem-faithful choice) or `:inst3` (instantaneous U/V from the inst3
  collection, no separate tavg3 file needed).
- `include_surface` / `include_convection` / `include_vdiff_fields` /
  `include_tm5_diffusion` — optional payload toggles. The first build wires
  only the core mass/flux payload; these are reserved for follow-on work and
  default to `false`.
"""
Base.@kwdef struct MERRA2Settings <: AbstractMetSettings
    root_dir              :: String
    coefficients_file     :: String = "config/geos_L72_coefficients.toml"
    winds_collection      :: Symbol = :tavg3
    include_surface       :: Bool   = false
    include_convection    :: Bool   = false
    include_vdiff_fields  :: Bool   = false
    include_tm5_diffusion :: Bool   = false
end

const MERRA2_NATIVE_LEVEL_COUNT = 72
const MERRA2_NX = 576
const MERRA2_NY = 361
const MERRA2_VALID_WINDS_COLLECTIONS = (:tavg3, :inst3)

# ---------------------------------------------------------------------------
# On-disk layout.
#
# inst3 (PS/QV endpoints): collection `inst3_3d_asm_Nv`, dataset dir M2I3NVASM.
# tavg3 (U/V winds):       collection `tavg3_3d_asm_Nv`, dataset dir M2T3NVASM.
# Stream code by year: 100 (1980-91), 200 (92-00), 300 (01-10), 400 (11-99).
# ---------------------------------------------------------------------------

const _MERRA2_COLLECTIONS = (
    inst3 = (dir = "M2I3NVASM", name = "inst3_3d_asm_Nv"),
    tavg3 = (dir = "M2T3NVASM", name = "tavg3_3d_asm_Nv"),
)

"""
    merra2_stream_code(date) -> String

MERRA-2 production stream code for `date` (same year→stream map the download
path uses): 100 (1980-91), 200 (92-00), 300 (01-10), 400 (2011 onward).
"""
function merra2_stream_code(date::Date)
    yr = year(date)
    yr <= 1991 && return "100"
    yr <= 2000 && return "200"
    yr <= 2010 && return "300"
    return "400"
end

"""
    merra2_path(settings, date, collection) -> String

Resolve the on-disk path for one MERRA-2 `collection` (`:inst3` or `:tavg3`)
on `date`:
`{root_dir}/{dir}/{YYYY}/{MM}/MERRA2_{stream}.{name}.{YYYYMMDD}.nc4`.
Existence is not checked here — `open_day` decides whether a missing file is
fatal or merely "no next-day endpoint available".
"""
function merra2_path(settings::MERRA2Settings, date::Date, collection::Symbol)
    hasproperty(_MERRA2_COLLECTIONS, collection) ||
        throw(ArgumentError("unknown MERRA-2 collection $(collection); expected one of " *
                            string(propertynames(_MERRA2_COLLECTIONS))))
    layout  = getproperty(_MERRA2_COLLECTIONS, collection)
    stream  = merra2_stream_code(date)
    yyyy    = Dates.format(date, "yyyy")
    mm      = Dates.format(date, "mm")
    datestr = Dates.format(date, "yyyymmdd")
    fname   = "MERRA2_$(stream).$(layout.name).$(datestr).nc4"
    return joinpath(settings.root_dir, layout.dir, yyyy, mm, fname)
end

# ---------------------------------------------------------------------------
# Day-handle container.
# ---------------------------------------------------------------------------

"""
    MERRA2DayHandles

Open `NCDataset` handles for one UTC day:

- `inst3` — PS/QV instantaneous endpoints (always open).
- `tavg3` — U/V time-averaged winds; `nothing` when
  `settings.winds_collection == :inst3` (winds then come from `inst3`).
- `next_inst3` — next day's inst3 dataset, for the final window's right
  endpoint look-ahead; `nothing` on the archive boundary or when
  `next_day_handle = false`.
"""
mutable struct MERRA2DayHandles
    settings   :: MERRA2Settings
    date       :: Date
    inst3      :: NCDataset
    tavg3      :: Union{Nothing, NCDataset}
    next_inst3 :: Union{Nothing, NCDataset}
end

"""
    open_merra2_day(settings, date; next_day_handle=true) -> MERRA2DayHandles

Open the day's inst3 (and tavg3, when `winds_collection==:tavg3`) datasets,
asserting both are on disk. When `next_day_handle=true` and the next-day
inst3 file exists, opens it for the final-window endpoint look-ahead.
"""
function open_merra2_day(settings::MERRA2Settings, date::Date;
                         next_day_handle::Bool = true)
    settings.winds_collection in MERRA2_VALID_WINDS_COLLECTIONS ||
        throw(ArgumentError("MERRA-2 winds_collection must be one of " *
                            join(MERRA2_VALID_WINDS_COLLECTIONS, ", ") *
                            "; got :$(settings.winds_collection)"))

    inst3_path = merra2_path(settings, date, :inst3)
    isfile(inst3_path) || error("MERRA-2 inst3 file not found: $inst3_path")
    inst3 = NCDataset(inst3_path, "r")

    tavg3 = nothing
    if settings.winds_collection === :tavg3
        tavg3_path = merra2_path(settings, date, :tavg3)
        isfile(tavg3_path) ||
            error("MERRA-2 tavg3 file not found: $tavg3_path (winds_collection=:tavg3)")
        tavg3 = NCDataset(tavg3_path, "r")
    end

    next_inst3 = nothing
    if next_day_handle
        candidate = merra2_path(settings, date + Day(1), :inst3)
        if isfile(candidate)
            try
                next_inst3 = NCDataset(candidate, "r")
            catch
                # Treat an unreadable next-day file as an archive boundary.
                next_inst3 = nothing
            end
        end
    end

    return MERRA2DayHandles(settings, date, inst3, tavg3, next_inst3)
end

"""
    close_merra2_day!(handles)

Close every open `NCDataset`. Idempotent — safe to call from a `finally`.
"""
function close_merra2_day!(handles::MERRA2DayHandles)
    close(handles.inst3)
    handles.tavg3      === nothing || close(handles.tavg3)
    handles.next_inst3 === nothing || close(handles.next_inst3)
    return nothing
end

# ---------------------------------------------------------------------------
# AbstractMetSettings trait hooks.
# ---------------------------------------------------------------------------

open_day(settings::MERRA2Settings, date::Date; next_day_handle::Bool = true) =
    open_merra2_day(settings, date; next_day_handle = next_day_handle)

close_day!(handles::MERRA2DayHandles) = close_merra2_day!(handles)

windows_per_day(::MERRA2Settings, ::Date) = 8

"""
    source_grid(settings::MERRA2Settings; FT=Float64) -> LatLonMesh

The native MERRA-2 source mesh (576 × 361, -180..180 lon, -90..90 lat). The
preprocessor builds its own regridder against the *target* mesh radius; this
descriptor uses `R_EARTH` and is provided for the canonical trait surface.
"""
source_grid(::MERRA2Settings; FT::Type{<:AbstractFloat} = Float64) =
    LatLonMesh(; FT = FT, Nx = MERRA2_NX, Ny = MERRA2_NY,
                longitude = (-180, 180), latitude = (-90, 90),
                radius = FT(R_EARTH))

# On this branch the optional RawWindow physics payloads are not populated by
# the MERRA-2 reader (the wind-derived writer only emits the core mass/flux
# sections). Report the conservative trait so generic downstream code does not
# try to allocate/write physics sections that this path does not produce.
has_surface(::MERRA2Settings)      = false
has_convection(::MERRA2Settings)   = false
has_vdiff_fields(::MERRA2Settings) = false

# ---------------------------------------------------------------------------
# Per-window field readers.
#
# NCDatasets reverses the CDL dim order, so the variables are indexed as:
#   PS(lon, lat, time)        → ds["PS"][:, :, win]      → (Nx, Ny)
#   QV/U/V(lon, lat, lev, time) → ds["..."][:, :, :, win] → (Nx, Ny, Nz)
# No level flip (MERRA-2 is top-down, matching the L72 coefficients) and no
# lat/lon reorientation (both already ascending, matching LatLonMesh).
# ---------------------------------------------------------------------------

"""
    read_merra2_ps_slice(ds, win; FT) -> Matrix{FT}  (Nx, Ny)

Read the surface pressure (Pa) at 1-based time-slice `win` from an inst3
NCDataset, as an `(Nx, Ny) = (lon, lat)` matrix.
"""
function read_merra2_ps_slice(ds::NCDataset, win::Integer; FT::Type{<:AbstractFloat})
    ps = Array{FT}(ds["PS"][:, :, Int(win)])
    size(ps) == (MERRA2_NX, MERRA2_NY) ||
        throw(DimensionMismatch("MERRA-2 PS slice $(size(ps)) ≠ ($(MERRA2_NX), $(MERRA2_NY))"))
    return ps
end

"""
    read_merra2_3d_slice(ds, name, win, Nz; FT) -> Array{FT,3}  (Nx, Ny, Nz)

Read a 3D field (`name` ∈ {"QV", "U", "V"}) at 1-based time-slice `win`, as an
`(Nx, Ny, Nz) = (lon, lat, lev)` array. `lev` is top-down (k=1 = TOA); no flip.
"""
function read_merra2_3d_slice(ds::NCDataset, name::AbstractString,
                              win::Integer, Nz::Integer; FT::Type{<:AbstractFloat})
    f = Array{FT}(ds[name][:, :, :, Int(win)])
    size(f) == (MERRA2_NX, MERRA2_NY, Int(Nz)) ||
        throw(DimensionMismatch("MERRA-2 $(name) slice $(size(f)) ≠ " *
                                "($(MERRA2_NX), $(MERRA2_NY), $(Nz))"))
    return f
end

"""
    read_merra2_window_fields(handles, win, Nz; FT)
        -> (; ps, qv, u, v)

Read one window's worth of native MERRA-2 LL fields for 1-based window index
`win` (1..8):

  - `ps` (Pa) and `qv` (kg/kg) from the **inst3** handle slice `win`
    (instantaneous mass endpoint),
  - `u`, `v` (m/s) from the **tavg3** handle slice `win` (3-hr time-average
    advecting winds), or from inst3 when `winds_collection==:inst3`.

All arrays are `(lon, lat[, lev])`, top-down, no reorientation.
"""
function read_merra2_window_fields(handles::MERRA2DayHandles, win::Integer,
                                   Nz::Integer; FT::Type{<:AbstractFloat})
    nw = windows_per_day(handles.settings, handles.date)
    1 <= win <= nw || throw(ArgumentError("window $win out of range 1..$nw"))

    ps = read_merra2_ps_slice(handles.inst3, win; FT = FT)
    qv = read_merra2_3d_slice(handles.inst3, "QV", win, Nz; FT = FT)

    winds_ds = handles.settings.winds_collection === :tavg3 ?
        (handles.tavg3 === nothing ?
            error("MERRA-2 winds_collection=:tavg3 but tavg3 handle is nothing") :
            handles.tavg3) :
        handles.inst3
    u = read_merra2_3d_slice(winds_ds, "U", win, Nz; FT = FT)
    v = read_merra2_3d_slice(winds_ds, "V", win, Nz; FT = FT)

    return (; ps = ps, qv = qv, u = u, v = v)
end

"""
    read_merra2_next_day_endpoint(handles, Nz; FT) -> (; ps, qv)

Read the next day's inst3 slice-1 PS/QV — the right endpoint for the final
window of `handles.date`. Returns `nothing` when no next-day inst3 handle is
open (archive boundary).
"""
function read_merra2_next_day_endpoint(handles::MERRA2DayHandles, Nz::Integer;
                                       FT::Type{<:AbstractFloat})
    handles.next_inst3 === nothing && return nothing
    ps = read_merra2_ps_slice(handles.next_inst3, 1; FT = FT)
    qv = read_merra2_3d_slice(handles.next_inst3, "QV", 1, Nz; FT = FT)
    return (; ps = ps, qv = qv)
end
