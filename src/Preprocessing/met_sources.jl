# ===========================================================================
# Met source abstraction for the preprocessor pipeline.
#
# A met source describes how raw NetCDF / GRIB data on its native grid is
# read into the canonical per-window intermediate `RawWindow`. The
# orchestrator dispatches on `<: AbstractMetSettings` (source axis) to choose
# the reader, and on `<: AbstractTargetGeometry` (target axis) to choose the
# regrid / flux-reconstruction / write path. The two axes never branch on
# each other — they meet in `RawWindow`.
# ===========================================================================

"""
    AbstractMetSettings

Top-level supertype for typed met-data source descriptors used by the
preprocessor. Concrete subtypes (e.g. `GEOSITSettings`, `MERRA2Settings`,
`SpectralERA5Settings`) carry source-specific paths and parameters and
implement the `read_window!` / `source_grid` / `windows_per_day` interface.

`process_day(date, grid::AbstractTargetGeometry, settings::AbstractMetSettings,
vertical; ...)` dispatches on `settings` to pick the reader.
"""
abstract type AbstractMetSettings end

"""
    RawWindow{FT, A2, A3}

Per-window source-grid intermediate carrying **both window endpoints**
(t_n and t_{n+1}) and the **window-integrated horizontal mass fluxes**
between them.

The right endpoint of window n is the left endpoint of window n+1, so the
reader caches it across calls. For the last window of the day, the right
endpoint comes from the next day's first instantaneous file (the existing
`next_day_hour0` plumbing in the orchestrator).

Cell-center winds `u`, `v` (geographic frame) are filled only when the
target grid differs from the source grid and fluxes must be reconstructed
downstream. For native passthrough (source mesh == target mesh) `u`/`v`
stay `nothing` and `am`/`bm` are written through directly after vertical
merging.

`cmfmc`/`dtrain` are filled only when the source supports convection and
the user has enabled it via `settings.include_convection`.
"""
struct RawWindow{FT <: AbstractFloat, P, V}
    # left endpoint (t_n)
    m       :: V
    ps      :: P
    qv      :: Union{Nothing, V}
    # right endpoint (t_{n+1})
    m_next  :: V
    ps_next :: P
    qv_next :: Union{Nothing, V}
    # window-mean horizontal mass flux on the source grid (units source-defined:
    # for GEOS, Pa·m²/s = MFXC / mass_flux_dt; the orchestrator handles any
    # unit conversion at write time).
    am      :: V
    bm      :: V
    # cell-center winds (geographic frame); set only when regrid happens downstream
    u       :: Union{Nothing, V}
    v       :: Union{Nothing, V}
    # optional convection sources
    cmfmc   :: Union{Nothing, V}   # at interfaces
    dtrain  :: Union{Nothing, V}   # at centers
end

# `P` is the per-window 2D field type (PS-shaped); `V` is the per-window 3D
# field type (DELP / mass-flux shaped). For structured (LL/RG) sources these
# are `Matrix{FT}` and `Array{FT, 3}`. For CS native sources they are
# `NTuple{6, Matrix{FT}}` and `NTuple{6, Array{FT, 3}}` so that downstream
# code can dispatch panel-aware operations naturally.

# ---------------------------------------------------------------------------
# Interface methods (concrete subtypes implement these)
# ---------------------------------------------------------------------------

"""
    read_window!(raw::RawWindow, settings::AbstractMetSettings, date::Date, win_idx::Int) -> raw

Fill `raw` in place with one window of source data on the source's native
grid. Subtypes of `AbstractMetSettings` must implement this method.
Implementations should not allocate per call (`raw` is a pre-allocated
workspace owned by the orchestrator) and should be idempotent in
`(date, win_idx)`.
"""
function read_window! end

"""
    source_grid(settings::AbstractMetSettings)

Return the source grid mesh that `read_window!` produces data on. Used by
the orchestrator to build the appropriate regridder for the target grid.
"""
function source_grid end

"""
    windows_per_day(settings::AbstractMetSettings, date::Date) -> Int

Number of preprocessing windows per UTC day for this source on `date`.
For most sources this is constant (24 for hourly), but date-dependent
sources (e.g. a leap-second day) may override.
"""
function windows_per_day end

"""
    has_convection(settings::AbstractMetSettings) -> Bool

Whether this source can populate `RawWindow.cmfmc` and `RawWindow.dtrain`.
Defaults to `false`; sources that support convection override and gate
the actual output behind a user flag (e.g. `settings.include_convection`).
"""
has_convection(::AbstractMetSettings) = false
