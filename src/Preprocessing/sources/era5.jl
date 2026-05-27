# ===========================================================================
# Native-GRIB ERA5 reader for the preprocessor.
#
# Consumes the canonical ECMWF download layout produced by
# `config/downloads/era5_native_daily.toml`:
#
#   <root_dir>/ml_an_native_core/era5_core_YYYYMMDD.grib
#   <root_dir>/ml_fc_convection/era5_convection_YYYYMMDD.grib   # gated by include_convection
#   <root_dir>/sfc_an_native/era5_surface_YYYYMMDD.grib          # gated by include_surface
#
# The `core` stream is the model-level analysis bundle (T, Q, VO, LNSP, D).
# T/VO/D/LNSP are stored as spherical-harmonic coefficients (`gridType=sh`),
# while Q is already on the reduced linear-Gaussian mesh (`gridType=reduced_gg`).
# The reader emits source-grid fields on the N-th reduced linear-Gaussian mesh
# selected by the `flavor` type parameter.
#
# Breakpoint A in this file: settings type, file-path resolution, per-day
# handle, and `AbstractMetSettings` trait hooks. The window-level GRIB decoder
# (`read_window!`) and spectral-synthesis plumbing land in subsequent commits.
# ===========================================================================

"""
    AbstractERA5GRIBSettings <: AbstractMetSettings

Abstract supertype for ERA5 native-GRIB sources. Concrete subtypes pick the
source mesh via the `flavor` parameter on [`ERA5GRIBSettings`](@ref).
"""
abstract type AbstractERA5GRIBSettings <: AbstractMetSettings end

"""
    ERA5GRIBSettings{flavor} <: AbstractERA5GRIBSettings

Typed settings for one ERA5 native-GRIB flavor:

- `flavor = :n320` — reduced linear-Gaussian N320, the default MARS native grid
  for ERA5 analyses (640 longitudes at the equator, 137 hybrid levels).

ERA5 archives store hybrid model levels top-down (k = 1 is the TOA-side cap).
`level_orientation = :top_down` reflects that, matching `era5.toml`. The reader
flips to the project's runtime convention downstream — same path used by the
GEOS-IT bottom-up source.
"""
Base.@kwdef struct ERA5GRIBSettings{flavor} <: AbstractERA5GRIBSettings
    root_dir             :: String
    include_surface      :: Bool   = false
    include_convection   :: Bool   = false
    include_vdiff_fields :: Bool   = false
    coefficients_file    :: String = "config/era5_L137_coefficients.toml"
    level_orientation    :: Symbol = :top_down
end

const ERA5N320Settings = ERA5GRIBSettings{:n320}

# ---------------------------------------------------------------------------
# Stream layout.
#
# `ERA5_GRIB_STREAMS` keeps the on-disk subdirectory and filename stem for each
# GRIB stream in one place so adding a new flavor (e.g. `:o320`) does not need
# to touch any of the call sites.
# ---------------------------------------------------------------------------

const ERA5_GRIB_STREAMS = (
    core       = (subdir = "ml_an_native_core", stem = "era5_core"),
    convection = (subdir = "ml_fc_convection",  stem = "era5_convection"),
    surface    = (subdir = "sfc_an_native",     stem = "era5_surface"),
)

"""
    era5_grib_path(settings, date, stream) -> String

Resolve the on-disk GRIB path for `stream` on `date`. `stream` must be one of
`:core`, `:convection`, or `:surface`. Existence is *not* checked here — the
caller (typically [`open_era5_day`](@ref)) decides whether a missing file is
fatal or merely "no next-day endpoint available".
"""
function era5_grib_path(settings::AbstractERA5GRIBSettings, date::Date,
                        stream::Symbol)
    hasproperty(ERA5_GRIB_STREAMS, stream) ||
        throw(ArgumentError("unknown ERA5 GRIB stream $(stream); expected one of " *
                            string(propertynames(ERA5_GRIB_STREAMS))))
    layout   = getproperty(ERA5_GRIB_STREAMS, stream)
    datestr  = Dates.format(date, "yyyymmdd")
    filename = "$(layout.stem)_$(datestr).grib"
    return joinpath(settings.root_dir, layout.subdir, filename)
end

# ---------------------------------------------------------------------------
# Day-handle container.
#
# GRIB.jl iterators are forward-only and cheap to reopen, so the handle holds
# resolved paths rather than pinned `GribFile` objects. This makes
# `close_day!` a no-op (idempotent) and avoids leaking file descriptors when
# the reader is restarted mid-day.
# ---------------------------------------------------------------------------

"""
    ERA5GRIBDayHandles{S<:AbstractERA5GRIBSettings}

Per-day source-file context. Carries the resolved on-disk paths to the day's
GRIB streams plus the optional next-day `core` path that supplies the right
endpoint of the last window.

`convection_path` is `nothing` unless `settings.include_convection` is set;
likewise `surface_path` is `nothing` unless `settings.include_surface` is set.
`next_core_path` is `nothing` either when the caller passed
`next_day_handle=false` or when the next-day file is not on disk (last day of
the available archive).
"""
struct ERA5GRIBDayHandles{S <: AbstractERA5GRIBSettings}
    settings             :: S
    date                 :: Date
    core_path            :: String
    convection_path      :: Union{Nothing, String}
    surface_path         :: Union{Nothing, String}
    next_core_path       :: Union{Nothing, String}
    # Previous-day convection file. ERA5 convection forecasts run from 06 UTC
    # and 18 UTC bases with 1-12 h steps; hours 0..5 of `date` are covered
    # by the previous day's 18 UTC base (steps 6..11). For dates at the
    # start of an archive (where date-1 isn't downloaded) this stays
    # `nothing` and the convection reader rejects requests for hours 0..5.
    prev_convection_path :: Union{Nothing, String}
end

const ERA5_VALID_LEVEL_ORIENTATIONS = (:top_down, :bottom_up)

"""
    open_era5_day(settings, date; next_day_handle=true) -> ERA5GRIBDayHandles

Resolve the GRIB stream paths for `date` and assert that today's required
files are on disk. When `next_day_handle=true` and `<date+1>` has a `core`
GRIB available, records its path so the last window's right endpoint can be
read from the next day's hour-0 fields.

Errors are explicit about which file is missing so a misconfigured `root_dir`
or an incomplete download is immediately visible.
"""
function open_era5_day(settings::AbstractERA5GRIBSettings, date::Date;
                       next_day_handle::Bool = true)
    settings.level_orientation in ERA5_VALID_LEVEL_ORIENTATIONS ||
        throw(ArgumentError("ERA5 level_orientation must be one of " *
                            join(ERA5_VALID_LEVEL_ORIENTATIONS, ", ") *
                            "; got :$(settings.level_orientation)"))

    core_path = era5_grib_path(settings, date, :core)
    isfile(core_path) ||
        error("ERA5 core GRIB not found: $core_path")

    convection_path = nothing
    if settings.include_convection
        candidate = era5_grib_path(settings, date, :convection)
        isfile(candidate) ||
            error("ERA5 convection GRIB not found: $candidate " *
                  "(include_convection=true)")
        convection_path = candidate
    end

    surface_path = nothing
    if settings.include_surface
        candidate = era5_grib_path(settings, date, :surface)
        isfile(candidate) ||
            error("ERA5 surface GRIB not found: $candidate " *
                  "(include_surface=true)")
        surface_path = candidate
    end

    next_core_path = nothing
    if next_day_handle
        candidate = era5_grib_path(settings, date + Day(1), :core)
        next_core_path = isfile(candidate) ? candidate : nothing
    end

    prev_convection_path = nothing
    if settings.include_convection
        candidate = era5_grib_path(settings, date - Day(1), :convection)
        prev_convection_path = isfile(candidate) ? candidate : nothing
    end

    return ERA5GRIBDayHandles{typeof(settings)}(
        settings, date,
        core_path, convection_path, surface_path, next_core_path,
        prev_convection_path)
end

"""
    close_era5_day!(handles::ERA5GRIBDayHandles)

Release any resources held by the day handle. Idempotent — safe to call from
a `finally` block. This is a no-op today because the handle only stores paths,
but the symbol exists so future flavors that pin file descriptors don't change
the call surface.
"""
close_era5_day!(::ERA5GRIBDayHandles) = nothing

# ---------------------------------------------------------------------------
# AbstractMetSettings trait hooks.
# ---------------------------------------------------------------------------

open_day(settings::AbstractERA5GRIBSettings, date::Date;
         next_day_handle::Bool = true) =
    open_era5_day(settings, date; next_day_handle = next_day_handle)

close_day!(handles::ERA5GRIBDayHandles) = close_era5_day!(handles)

windows_per_day(::AbstractERA5GRIBSettings, ::Date) = 24

has_surface(s::AbstractERA5GRIBSettings)      = s.include_surface
has_convection(s::AbstractERA5GRIBSettings)   = s.include_convection
has_vdiff_fields(s::AbstractERA5GRIBSettings) = s.include_vdiff_fields

# ===========================================================================
# Breakpoint B — per-window field synthesis on the N320 source mesh.
#
# Each daily N320 `core` GRIB carries spectral T/VO/D/LNSP and reduced-Gaussian
# Q across 24 hours × 137 hybrid levels in a single file. For one window the
# reader produces:
#
#   - U, V on N320 cell centers (vod2uv on spectral coefficients, then
#     ring-by-ring synthesis via `spectral_to_reduced_scalar!`)
#   - T on N320 cell centers (direct spectral synthesis)
#   - PS on N320 cell centers (synthesise LNSP, exponentiate)
#   - Q on N320 cell centers (direct reduced_gg gridpoint reorder; the GRIB
#     stores rings north→south but the mesh stores them south→north)
#
# Hybrid pressure → layer mass, regrid to C180, convection conversion all
# land in subsequent breakpoints on this branch.
# ===========================================================================

const ERA5_NATIVE_LEVEL_COUNT = 137

# ECMWF parameter ids for the ERA5 model-level analyses we consume.
const _ERA5_PARAM_T    = 130
const _ERA5_PARAM_Q    = 133
const _ERA5_PARAM_VO   = 138
const _ERA5_PARAM_LNSP = 152
const _ERA5_PARAM_D    = 155

"""
    discover_era5_n320_source_grid(core_path; FT=Float64) -> ReducedGaussianTargetGeometry

Build the N320 source-grid descriptor from the first `gridType=reduced_gg`
message in `core_path` (the `q` field is always present). The resulting mesh
ring order is south→north — the GRIB stores rings north→south and the
`read_era5_reduced_gaussian_*` helpers flip into the project convention.
"""
function discover_era5_n320_source_grid(core_path::AbstractString;
                                         FT::Type{<:AbstractFloat} = Float64)
    geom = read_era5_reduced_gaussian_geometry(core_path; FT = FT)
    mesh = ReducedGaussianMesh(geom.latitudes, geom.nlon_per_ring;
                                FT = FT, radius = FT(R_EARTH))
    lons_by_ring = [FT.(ring_longitudes(mesh, j)) for j in 1:nrings(mesh)]
    return ReducedGaussianTargetGeometry{FT, typeof(mesh)}(
        mesh,
        String(core_path),
        geom.gaussian_number,
        copy(geom.nlon_per_ring),
        copy(geom.latitudes),
        lons_by_ring,
    )
end

"""
    discover_era5_spectral_truncation(core_path) -> Int

Read the first spectral (`gridType=sh`) message in `core_path` and return its
triangular truncation `J = K = M`. ERA5 native model-level analyses are T639
in the current archive; the helper avoids hard-coding that value in case a
future archive convention shifts.
"""
function discover_era5_spectral_truncation(core_path::AbstractString)
    truncation = 0
    GribFile(core_path) do gf
        for msg in gf
            String(msg["gridType"]) == "sh" || continue
            truncation = Int(msg["J"])
            break
        end
    end
    truncation > 0 ||
        error("No spectral (gridType=sh) message found in $core_path")
    return truncation
end

# ---------------------------------------------------------------------------
# Workspace + output field structs.
#
# The workspace owns the level-indexed spectral cubes for one hour and the
# Legendre / FFT synthesis caches. The output fields struct owns the
# gridpoint result arrays. The two are decoupled so callers can keep the
# workspace alive across windows (cheap) and reuse a single output buffer
# across multiple consumers.
# ---------------------------------------------------------------------------

"""
    ERA5N320SpectralWorkspace{FT, G}

Per-window workspace for ERA5 N320 spectral synthesis. Owns:

  - the spectral coefficient cubes for T, VO, D, LNSP for the current hour
    (sized `(T+1) × (T+1) × Nz` in `ComplexF64`),
  - per-level scratch matrices `u_spec` / `v_spec` reused inside the
    synthesis loop,
  - a `ReducedSpectralThreadCache` with Legendre column buffer plus FFT and
    real ring buffers sized to every unique ring length in the source mesh,
  - a `read_buf` scratch used by `read_spectral_coeffs!` for the raw ecCodes
    `codes_get_double_array` payload,
  - completion bookkeeping (`have_t` / `have_vo` / `have_d` / `have_lnsp`).

The cubes dominate memory: at T = 639 and Nz = 137 each cube is ≈ 0.9 GB.
Workspaces are intended to be allocated once per day-handle and reused across
the 24 hours.
"""
struct ERA5N320SpectralWorkspace{FT <: AbstractFloat,
                                  G <: ReducedGaussianTargetGeometry{FT}}
    source_grid  :: G
    T            :: Int
    Nz           :: Int
    vo_spec      :: Array{ComplexF64, 3}
    d_spec       :: Array{ComplexF64, 3}
    t_spec       :: Array{ComplexF64, 3}
    lnsp_spec    :: Matrix{ComplexF64}
    u_spec       :: Matrix{ComplexF64}
    v_spec       :: Matrix{ComplexF64}
    synth_cache  :: ReducedSpectralThreadCache
    read_buf     :: Vector{Float64}
    # Per-spectral-field decode scratch. `read_spectral_coeffs!` wants a
    # concrete `Matrix{ComplexF64}` (not a view), so we keep one matrix per
    # spectral field and copy the result into the level slot of the cube.
    # Avoids ~70 GB/day of allocator churn at T639/Nz137.
    t_scratch    :: Matrix{ComplexF64}
    vo_scratch   :: Matrix{ComplexF64}
    d_scratch    :: Matrix{ComplexF64}
    # Gridpoint synthesis scratch — `spectral_to_reduced_scalar!` writes
    # Float64. Reused for U/V/T/LNSP synthesis when the output buffer is
    # narrower than Float64 (e.g. Float32 production runs).
    grid_scratch :: Vector{Float64}
    have_t       :: BitVector
    have_vo      :: BitVector
    have_d       :: BitVector
    have_lnsp    :: Base.RefValue{Bool}
    lnsp_grid    :: Vector{Float64}
end

"""
    allocate_era5_n320_spectral_workspace(source_grid, T, Nz)

Allocate a fresh workspace sized to `source_grid` (build via
[`discover_era5_n320_source_grid`](@ref)), spectral truncation `T`
(from [`discover_era5_spectral_truncation`](@ref)) and vertical level count
`Nz` (defaults to 137 in callers, but the workspace itself stays generic).
"""
function allocate_era5_n320_spectral_workspace(source_grid::ReducedGaussianTargetGeometry{FT},
                                                T::Integer, Nz::Integer) where FT
    T  >= 1 || throw(ArgumentError("T must be ≥ 1, got $T"))
    Nz >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))
    T_int  = Int(T)
    Nz_int = Int(Nz)

    mesh = source_grid.mesh
    nc = T_int + 1

    buffer_lengths = sort!(unique(vcat(collect(mesh.nlon_per_ring),
                                       collect(mesh.boundary_counts))))
    cache = ReducedSpectralThreadCache(
        zeros(Float64, nc, nc),
        Dict(n => zeros(ComplexF64, n) for n in buffer_lengths),
        Dict(n => zeros(Float64, n)    for n in buffer_lengths),
        zeros(ComplexF64, nc, nc),
        zeros(ComplexF64, nc, nc),
    )

    n_cells = ncells(mesh)
    return ERA5N320SpectralWorkspace{FT, typeof(source_grid)}(
        source_grid, T_int, Nz_int,
        zeros(ComplexF64, nc, nc, Nz_int),     # vo_spec
        zeros(ComplexF64, nc, nc, Nz_int),     # d_spec
        zeros(ComplexF64, nc, nc, Nz_int),     # t_spec
        zeros(ComplexF64, nc, nc),             # lnsp_spec
        zeros(ComplexF64, nc, nc),             # u_spec  — per-level scratch
        zeros(ComplexF64, nc, nc),             # v_spec  — per-level scratch
        cache,
        Float64[],                             # read_buf — grows in read_spectral_coeffs!
        zeros(ComplexF64, nc, nc),             # t_scratch
        zeros(ComplexF64, nc, nc),             # vo_scratch
        zeros(ComplexF64, nc, nc),             # d_scratch
        zeros(Float64, n_cells),               # grid_scratch — Float64 synthesis target
        falses(Nz_int), falses(Nz_int), falses(Nz_int),
        Ref(false),
        zeros(Float64, n_cells),               # lnsp_grid — scratch for PS synthesis
    )
end

"""
    ERA5N320WindowFields{FT}

Per-window output container. `u` / `v` (m/s, geographic frame), `t` (K),
`qv` (kg/kg specific humidity) are `(n_cells, Nz)`; `ps` (Pa) is `(n_cells,)`.
Dry-basis layer mass derivation, regridding, and convection conversion are
downstream of this struct.
"""
struct ERA5N320WindowFields{FT <: AbstractFloat}
    u  :: Matrix{FT}
    v  :: Matrix{FT}
    t  :: Matrix{FT}
    qv :: Matrix{FT}
    ps :: Vector{FT}
end

function allocate_era5_n320_window_fields(source_grid::ReducedGaussianTargetGeometry{FT},
                                            Nz::Integer) where FT
    Nz >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))
    nc = ncells(source_grid.mesh)
    Nz_int = Int(Nz)
    return ERA5N320WindowFields{FT}(
        zeros(FT, nc, Nz_int),
        zeros(FT, nc, Nz_int),
        zeros(FT, nc, Nz_int),
        zeros(FT, nc, Nz_int),
        zeros(FT, nc),
    )
end

# ---------------------------------------------------------------------------
# reduced_gg → mesh ring reorder.
#
# ERA5 stores rings north→south (jScansPositively=0). The
# `read_era5_reduced_gaussian_*` helpers flip ring order so the mesh is
# south→north. Cells within each ring are already west→east; only the ring
# axis needs reversing. This helper is small enough to bake here without
# pulling in MetDrivers, and it asserts the per-ring count match so a
# miscounted `pl` array fails loudly instead of silently aliasing rings.
# ---------------------------------------------------------------------------

"""
    _reorder_grib_reduced_gg_to_mesh!(mesh_vals, native_vals, native_nlon, mesh) -> mesh_vals

Copy `native_vals` (GRIB native ring order, north→south) into `mesh_vals`
(`ReducedGaussianMesh` order, south→north). Asserts the per-ring counts in
`native_nlon` match `mesh.nlon_per_ring` (after reversal). Cells within a
ring are not permuted — only the ring axis flips.
"""
function _reorder_grib_reduced_gg_to_mesh!(mesh_vals::AbstractVector,
                                            native_vals::AbstractVector,
                                            native_nlon::AbstractVector{<:Integer},
                                            mesh::ReducedGaussianMesh)
    nrings_native = length(native_nlon)
    nrings_native == nrings(mesh) ||
        throw(DimensionMismatch("native_nlon has $(nrings_native) rings; mesh has $(nrings(mesh))"))
    length(mesh_vals) == length(native_vals) == sum(native_nlon) ||
        throw(DimensionMismatch("native_vals/mesh_vals/native_nlon length mismatch"))

    native_offset = 1
    @inbounds for j_native in 1:nrings_native
        j_mesh = nrings_native - j_native + 1
        n = Int(native_nlon[j_native])
        n == mesh.nlon_per_ring[j_mesh] ||
            throw(DimensionMismatch("native ring $j_native nlon=$n vs mesh ring $j_mesh nlon=$(mesh.nlon_per_ring[j_mesh])"))
        mesh_start = mesh.ring_offsets[j_mesh]
        mesh_end   = mesh.ring_offsets[j_mesh + 1] - 1
        @views mesh_vals[mesh_start:mesh_end] .= native_vals[native_offset:(native_offset + n - 1)]
        native_offset += n
    end
    return mesh_vals
end

# ---------------------------------------------------------------------------
# Window read + synthesis.
# ---------------------------------------------------------------------------

"""
    read_era5_n320_window_fields!(fields, workspace, handles, date, hour) -> fields

Fill `fields` with one window's worth of N320 source-grid fields for
`(date, hour)`. Performs one forward pass over `handles.core_path`, decoding
every message whose `(dataDate, dataTime)` matches into the workspace's
level-indexed spectral cubes (for `gridType=sh`) or directly into the
output `qv` array (for `gridType=reduced_gg`). After the pass the function
synthesizes spectral T per level, applies `vod2uv!` per level and synthesizes
U/V, and synthesizes LNSP → PS. Errors loudly if any required level/field is
absent so a partial download is immediately visible.

The function does not allocate beyond `read_buf` resizing inside
`read_spectral_coeffs!`. Reusing `(fields, workspace)` across the 24 windows
of a day is the intended call pattern.
"""
function read_era5_n320_window_fields!(fields::ERA5N320WindowFields{FT},
                                        workspace::ERA5N320SpectralWorkspace{FT},
                                        handles::ERA5GRIBDayHandles,
                                        date::Date,
                                        hour::Integer) where FT
    0 <= hour <= 23 || throw(ArgumentError("hour must be in 0..23, got $hour"))

    expected_data_date = parse(Int, Dates.format(date, "yyyymmdd"))
    expected_data_time = Int(hour) * 100

    fill!(workspace.have_t,  false)
    fill!(workspace.have_vo, false)
    fill!(workspace.have_d,  false)
    workspace.have_lnsp[] = false

    mesh = workspace.source_grid.mesh
    T  = workspace.T
    Nz = workspace.Nz
    nc = ncells(mesh)
    length(fields.ps) == nc ||
        throw(DimensionMismatch("fields.ps length $(length(fields.ps)) != n_cells $nc"))
    size(fields.qv) == (nc, Nz) ||
        throw(DimensionMismatch("fields.qv size $(size(fields.qv)) != ($nc, $Nz)"))

    GribFile(handles.core_path) do gf
        for msg in gf
            Int(msg["dataDate"]) == expected_data_date || continue
            Int(msg["dataTime"]) == expected_data_time || continue

            grid_type  = String(msg["gridType"])
            short_name = String(msg["shortName"])
            level      = Int(msg["level"])

            if grid_type == "sh"
                if short_name == "t"
                    _read_into_level_slot!(workspace.t_spec, workspace.t_scratch,
                                            msg, workspace.read_buf, level, Nz)
                    workspace.have_t[level] = true
                elseif short_name == "vo"
                    _read_into_level_slot!(workspace.vo_spec, workspace.vo_scratch,
                                            msg, workspace.read_buf, level, Nz)
                    workspace.have_vo[level] = true
                elseif short_name == "d"
                    _read_into_level_slot!(workspace.d_spec, workspace.d_scratch,
                                            msg, workspace.read_buf, level, Nz)
                    workspace.have_d[level] = true
                elseif short_name == "lnsp"
                    read_spectral_coeffs!(workspace.lnsp_spec, msg, workspace.read_buf)
                    workspace.have_lnsp[] = true
                end
            elseif grid_type == "reduced_gg" && short_name == "q"
                1 <= level <= Nz ||
                    error("Q level $level outside [1, $Nz] for date=$date hour=$hour")
                vals = msg["values"]
                pl   = msg["pl"]
                _reorder_grib_reduced_gg_to_mesh!(
                    view(fields.qv, :, level), vals, pl, mesh)
            end
        end
    end

    # Completeness gates — fail with the missing fields named so logs are debuggable.
    workspace.have_lnsp[] ||
        error("ERA5 N320 read: LNSP missing for $(date) hour $(hour)")
    all(workspace.have_t) ||
        error("ERA5 N320 read: T missing for $(date) hour $(hour) at levels $(findall(!, workspace.have_t))")
    all(workspace.have_vo) ||
        error("ERA5 N320 read: VO missing for $(date) hour $(hour) at levels $(findall(!, workspace.have_vo))")
    all(workspace.have_d) ||
        error("ERA5 N320 read: D missing for $(date) hour $(hour) at levels $(findall(!, workspace.have_d))")

    # Spectral → gridpoint synthesis per level.
    grid = workspace.source_grid
    cache = workspace.synth_cache

    @inbounds for k in 1:Nz
        vo_lvl = view(workspace.vo_spec, :, :, k)
        d_lvl  = view(workspace.d_spec,  :, :, k)
        t_lvl  = view(workspace.t_spec,  :, :, k)

        # vod2uv! produces ECMWF's `U·cos(φ)` / `V·cos(φ)` "pseudo-winds";
        # the per-ring division below recovers physical `U`, `V` in m/s.
        vod2uv!(workspace.u_spec, workspace.v_spec, vo_lvl, d_lvl, T)

        _synthesize_into_column!(view(fields.u, :, k), workspace.u_spec, T,
                                  grid, cache, workspace.grid_scratch)
        _synthesize_into_column!(view(fields.v, :, k), workspace.v_spec, T,
                                  grid, cache, workspace.grid_scratch)
        _synthesize_into_column!(view(fields.t, :, k), t_lvl,            T,
                                  grid, cache, workspace.grid_scratch)

        _divide_by_cos_lat_per_ring!(view(fields.u, :, k), mesh)
        _divide_by_cos_lat_per_ring!(view(fields.v, :, k), mesh)
    end

    # LNSP → PS = exp(LNSP). The `lnsp_grid` buffer is Float64 already, so it
    # serves as both column and scratch (the in-method copy becomes a self-copy
    # of ~500 KB and is amortised across the synthesis kernel cost).
    _synthesize_into_column!(workspace.lnsp_grid, workspace.lnsp_spec, T,
                              grid, cache, workspace.lnsp_grid)
    @inbounds for c in 1:nc
        fields.ps[c] = exp(workspace.lnsp_grid[c])
    end

    return fields
end

# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------

"""Decode `msg` spectral coefficients into level slot `level` of `cube`, using
the workspace-owned `scratch` matrix as the `read_spectral_coeffs!` target.
The cube slice is updated via `copyto!` to avoid any per-call allocation.
Asserts `1 ≤ level ≤ Nz` so a stray off-archive level fails loudly instead
of silently aliasing a neighbouring slot."""
function _read_into_level_slot!(cube::Array{ComplexF64, 3},
                                 scratch::Matrix{ComplexF64},
                                 msg,
                                 read_buf::Vector{Float64},
                                 level::Int, Nz::Int)
    1 <= level <= Nz ||
        error("spectral level $level outside [1, $Nz]")
    read_spectral_coeffs!(scratch, msg, read_buf)
    @inbounds @views cube[:, :, level] .= scratch
    return cube
end

"""Divide an `n_cells`-laid-out field by `cos(latitude)` of its ring. Used to
recover physical `U`, `V` from the ECMWF `U·cos(φ)` / `V·cos(φ)` form produced
by `vod2uv!`. The polar rings of N320 are at ±89.78° (`cos(φ) ≈ 0.004`); the
guard clamps any non-positive value to `eps(Float64)` so floating-point noise
near `cos(90°)` never crashes a 13-hour preprocess run. Synthesis noise at
the poles is bounded by the spectral truncation and tolerated downstream
(Poisson balance, VDIFF payload regrid)."""
function _divide_by_cos_lat_per_ring!(field::AbstractVector,
                                       mesh::ReducedGaussianMesh)
    @inbounds for j in 1:nrings(mesh)
        cos_lat = max(cosd(Float64(mesh.latitudes[j])), eps(Float64))
        inv_cos = 1.0 / cos_lat
        ring_start = mesh.ring_offsets[j]
        ring_end   = mesh.ring_offsets[j + 1] - 1
        @views field[ring_start:ring_end] .*= inv_cos
    end
    return field
end

# ===========================================================================
# Breakpoint C — hybrid pressure → dry-air mass on the N320 source grid.
#
# Mirrors the GEOS endpoint dry-mass derivation (`endpoint_dry_mass!`) so the
# dry-basis runtime contract is nominally bit-identical across source paths:
#
#   ΔA[k] = A[k+1] − A[k]   (Pa)
#   ΔB[k] = B[k+1] − B[k]   (dimensionless)
#   DELP_full[k] = ΔA[k] + ΔB[k] · PS_total
#   DELP_dry[k]  = (1 − Q[k]) · DELP_full[k]
#   PS_dry       = Σ_k DELP_dry[k]
#   m_dry[k]     = DELP_dry[k] · cell_area / g
#
# All arithmetic runs in Float64 internally and is downcast to FT only on
# write. Vertical merge (e.g. `MergeAbovePressure` for parity with the GEOS
# L72 cap) is delegated to the existing `apply_vertical!` plumbing in the
# breakpoint-F glue.
# ===========================================================================

"""
    ERA5N320DryMassFields{FT}

Per-window dry-basis output container. `m_dry` is dry-air mass per cell per
layer (kg), `delp_dry` is dry pressure thickness per cell per layer (Pa),
`ps_dry` is the column-integrated dry surface pressure (Pa). `ps_dry_acc` is
a Float64 accumulator that backs `ps_dry` so Σ_k DELP_dry stays accurate
even when FT = Float32 (137-layer summation with ~10 hPa values per layer
would otherwise lose ~100 Pa to single-precision rounding).
"""
struct ERA5N320DryMassFields{FT <: AbstractFloat}
    m_dry      :: Matrix{FT}     # (n_cells, Nz)
    delp_dry   :: Matrix{FT}     # (n_cells, Nz)
    ps_dry     :: Vector{FT}     # (n_cells,)
    ps_dry_acc :: Vector{Float64} # Float64 accumulator scratch
end

function allocate_era5_n320_dry_mass_fields(source_grid::ReducedGaussianTargetGeometry{FT},
                                              Nz::Integer) where FT
    Nz >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))
    nc = ncells(source_grid.mesh)
    Nz_int = Int(Nz)
    return ERA5N320DryMassFields{FT}(
        zeros(FT, nc, Nz_int),
        zeros(FT, nc, Nz_int),
        zeros(FT, nc),
        zeros(Float64, nc),
    )
end

"""
    n320_cell_areas(source_grid) -> Vector{Float64}

Materialise per-cell areas (m²) for the N320 source mesh. Always returns
`Vector{Float64}` regardless of `source_grid`'s element type — downstream
dry-mass arithmetic runs in Float64 for precision and the per-cell mesh
quadrature is itself Float64. Cached by callers that derive dry mass for
many windows of the same day.
"""
function n320_cell_areas(source_grid::ReducedGaussianTargetGeometry)
    mesh = source_grid.mesh
    return [Float64(cell_area(mesh, c)) for c in 1:ncells(mesh)]
end

"""
    derive_n320_dry_mass!(dry, window, vc, cell_areas; grav=GRAV) -> dry

Reconstruct dry-air mass per layer, dry pressure thickness, and dry surface
pressure from a populated `window::ERA5N320WindowFields` (moist PS, Q) using
the hybrid coordinate `vc` (length `Nz+1` A and B arrays in Pa and 1
respectively, top-down) and the per-cell areas (m²). Matches the GEOS
`endpoint_dry_mass!` formula so the runtime sees a nominally bit-identical
dry-basis contract regardless of source.

Asserts shapes and `length(vc.A) == length(vc.B) == Nz + 1` so a coefficient
table that does not match the workspace `Nz` fails immediately with a clear
DimensionMismatch.
"""
function derive_n320_dry_mass!(dry::ERA5N320DryMassFields{FT},
                                window::ERA5N320WindowFields,
                                vc::HybridSigmaPressure,
                                cell_areas::AbstractVector{<:Real};
                                grav::Real = GRAV) where FT
    nc, Nz = size(dry.m_dry)
    size(window.qv) == (nc, Nz) ||
        throw(DimensionMismatch("window.qv $(size(window.qv)) ≠ ($nc, $Nz)"))
    length(window.ps) == nc ||
        throw(DimensionMismatch("window.ps length $(length(window.ps)) ≠ $nc"))
    length(cell_areas) == nc ||
        throw(DimensionMismatch("cell_areas length $(length(cell_areas)) ≠ $nc"))
    length(dry.ps_dry_acc) == nc ||
        throw(DimensionMismatch("ps_dry_acc length $(length(dry.ps_dry_acc)) ≠ $nc"))
    length(vc.A) == length(vc.B) == Nz + 1 ||
        throw(DimensionMismatch("hybrid A/B length $(length(vc.A))/$(length(vc.B)) ≠ Nz+1 = $(Nz + 1)"))

    A = vc.A
    B = vc.B
    inv_g = 1.0 / Float64(grav)
    fill!(dry.ps_dry_acc, 0.0)

    # k-outer / c-inner traversal: the `(n_cells, Nz)` Float arrays are
    # column-major, so the contiguous axis is `c`. This keeps every write
    # to `delp_dry[c, k]`, `m_dry[c, k]` and every read from `qv[c, k]` on
    # a unit stride. dA, dB are level-only and hoist out of the cell loop.
    # `ps_dry_acc` accumulates in Float64 to keep precision when FT=Float32.
    @inbounds for k in 1:Nz
        dA = Float64(A[k + 1]) - Float64(A[k])
        dB = Float64(B[k + 1]) - Float64(B[k])
        for c in 1:nc
            ps_total   = Float64(window.ps[c])
            area       = Float64(cell_areas[c])
            delp_full  = dA + dB * ps_total
            delp_dry_k = (1.0 - Float64(window.qv[c, k])) * delp_full
            dry.delp_dry[c, k] = FT(delp_dry_k)
            dry.m_dry[c, k]    = FT(delp_dry_k * area * inv_g)
            dry.ps_dry_acc[c] += delp_dry_k
        end
    end
    @inbounds for c in 1:nc
        dry.ps_dry[c] = FT(dry.ps_dry_acc[c])
    end
    return dry
end

# ===========================================================================
# Breakpoint E — convection forecast fields on the N320 source grid.
#
# The ERA5 convection product is a forecast bundle: model-level UDMF, DDMF,
# UDRF, DDRF (param ids 235009-235012, GRIB shortNames `avg_umf`, `avg_dmf`,
# `avg_udr`, `avg_ddr`) archived twice daily from 06 UTC and 18 UTC bases,
# each carrying hourly time-mean values for steps 1..12. One UTC day of
# windowed transport-binary output is covered by:
#
#   hour 0..5  → previous-day `era5_convection_(D-1).grib`, 18 UTC base,
#                stepRange "$(h+6)-$(h+7)"
#   hour 6..17 → today's `era5_convection_D.grib`,           06 UTC base,
#                stepRange "$(h-6)-$(h-5)"
#   hour 18..23 → today's `era5_convection_D.grib`,          18 UTC base,
#                 stepRange "$(h-18)-$(h-17)"
#
# All four fields live on the N320 reduced_gg mesh — same layout as Q. The
# reader reuses `_reorder_grib_reduced_gg_to_mesh!` to flip the ring axis
# from the GRIB's native N→S to the mesh's S→N convention.
# ===========================================================================

"""
    ERA5N320ConvectionFields{FT}

Per-window convection forecast fields on the N320 source mesh. All four
fields are `(n_cells, Nz)`:

  - `udmf` — updraft convective mass flux (kg m⁻² s⁻¹)
  - `ddmf` — downdraft convective mass flux (kg m⁻² s⁻¹)
  - `udrf` — updraft detrainment rate (kg m⁻³ s⁻¹)
  - `ddrf` — downdraft detrainment rate (kg m⁻³ s⁻¹)

Downstream conversion to GEOS-style CMFMC + DTRAIN (or TM5-style entu/entd)
happens in the breakpoint-F glue once both source and target geometries
are known.
"""
struct ERA5N320ConvectionFields{FT <: AbstractFloat}
    udmf :: Matrix{FT}
    ddmf :: Matrix{FT}
    udrf :: Matrix{FT}
    ddrf :: Matrix{FT}
end

function allocate_era5_n320_convection_fields(source_grid::ReducedGaussianTargetGeometry{FT},
                                                Nz::Integer) where FT
    Nz >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))
    nc = ncells(source_grid.mesh)
    return ERA5N320ConvectionFields{FT}(
        zeros(FT, nc, Int(Nz)),
        zeros(FT, nc, Int(Nz)),
        zeros(FT, nc, Int(Nz)),
        zeros(FT, nc, Int(Nz)),
    )
end

"""
    era5_convection_hour_address(hour) -> (use_prev_day, data_time_hhmm, step_range)

Map a UTC hour `h ∈ 0..23` to the GRIB header tuple that addresses the
matching ECMWF convection forecast sample (see `read_era5_n320_convection_window!`
docstring for the cycle layout). Returns `(::Bool, ::Int, ::String)`.
"""
function era5_convection_hour_address(hour::Integer)
    0 <= hour <= 23 || throw(ArgumentError("hour must be in 0..23, got $hour"))
    h = Int(hour)
    if h < 6
        return (true,  1800, "$(h + 6)-$(h + 7)")
    elseif h < 18
        return (false,  600, "$(h - 6)-$(h - 5)")
    else
        return (false, 1800, "$(h - 18)-$(h - 17)")
    end
end

# Maps ERA5 GRIB shortName → field slot in `ERA5N320ConvectionFields`.
const _ERA5_CONVECTION_SHORT_NAMES = (
    avg_umf = :udmf,
    avg_dmf = :ddmf,
    avg_udr = :udrf,
    avg_ddr = :ddrf,
)

"""
    read_era5_n320_convection_window!(fields, handles, mesh, date, hour) -> fields

Fill `fields` with one hour's worth of N320 convection forecast fields for
`(date, hour)`. The reader picks `handles.convection_path` or
`handles.prev_convection_path` based on the hour-address mapping and
forward-iterates that file, matching messages whose `(dataTime, stepRange,
shortName)` triple falls within the requested sample. Each matching message
is decoded directly into the appropriate `(n_cells, Nz)` output slot via
`_reorder_grib_reduced_gg_to_mesh!`.

Completeness gates name the missing field in any error so a corrupt or
partial download is immediately visible. All four fields × 137 levels must
be present for the call to succeed.
"""
function read_era5_n320_convection_window!(fields::ERA5N320ConvectionFields{FT},
                                            handles::ERA5GRIBDayHandles,
                                            mesh::ReducedGaussianMesh,
                                            date::Date,
                                            hour::Integer) where FT
    handles.convection_path !== nothing ||
        error("ERA5 convection read requested but settings.include_convection=false")

    use_prev, data_time, step_range = era5_convection_hour_address(hour)
    path = if use_prev
        handles.prev_convection_path !== nothing ||
            error("ERA5 convection hour $hour of $(date) needs the previous-day file " *
                  "($(date - Day(1)) era5_convection*.grib) which is not on disk")
        handles.prev_convection_path
    else
        handles.convection_path
    end

    nc, Nz = size(fields.udmf)
    nc == ncells(mesh) ||
        throw(DimensionMismatch("fields.udmf rows $nc ≠ ncells(mesh) $(ncells(mesh))"))

    # NamedTuple of (slot_symbol → output Matrix) and (slot_symbol → completion
    # BitVector). The shared `field_slot` symbol indexes both, so adding a
    # fifth convection field is a one-liner here and in
    # `_ERA5_CONVECTION_SHORT_NAMES`.
    field_matrices = (udmf = fields.udmf, ddmf = fields.ddmf,
                       udrf = fields.udrf, ddrf = fields.ddrf)
    have = (udmf = falses(Nz), ddmf = falses(Nz),
            udrf = falses(Nz), ddrf = falses(Nz))

    GribFile(path) do gf
        for msg in gf
            Int(msg["dataTime"]) == data_time || continue
            String(msg["stepRange"]) == step_range || continue
            short_name = String(msg["shortName"])
            field_slot = get(_ERA5_CONVECTION_SHORT_NAMES, Symbol(short_name), nothing)
            field_slot === nothing && continue

            level = Int(msg["level"])
            1 <= level <= Nz || continue

            vals = msg["values"]
            pl   = msg["pl"]
            _reorder_grib_reduced_gg_to_mesh!(
                view(getproperty(field_matrices, field_slot), :, level),
                vals, pl, mesh)
            getproperty(have, field_slot)[level] = true
        end
    end

    for name in propertynames(have)
        bits = getproperty(have, name)
        all(bits) || error("ERA5 convection read: $(uppercase(string(name))) missing for " *
                            "$(date) hour $hour at levels $(findall(!, bits))")
    end

    return fields
end

# ===========================================================================
# Breakpoint D — conservative regrid from N320 source mesh to a C180
# cubed-sphere target. Intensive scalars (PS, T, Q, U, V) use the
# `ConservativeRegridding` weights cached on disk; dry-mass derivation on
# the C180 target stays a downstream concern (re-derived from regridded
# PS + Q in the breakpoint-F glue) so the regridder operates on a small
# fixed set of fields.
# ===========================================================================

"""
    ERA5C180RegridFields{FT}

Per-window output container for fields regridded onto the C180 cubed-sphere
target. Holds:

  - `ps` — 2D `(Nc, Nc)` matrix per panel (Pa, moist surface pressure),
  - `u`, `v`, `t`, `qv` — 3D `(Nc, Nc, Nz)` arrays per panel (U/V in
    geographic east/north frame; rotation to panel-local axes happens in
    the breakpoint-F glue where the panel basis is known).

Mass fields (m_dry, delp_dry, ps_dry) are *not* regridded directly; they
are re-derived on the C180 mesh from the regridded PS + Q so that
`Σ_k DELP_dry == PS_dry` to roundoff on the target side as well.
"""
struct ERA5C180RegridFields{FT <: AbstractFloat}
    ps :: NTuple{6, Matrix{FT}}
    u  :: NTuple{6, Array{FT, 3}}
    v  :: NTuple{6, Array{FT, 3}}
    t  :: NTuple{6, Array{FT, 3}}
    qv :: NTuple{6, Array{FT, 3}}
end

function allocate_era5_c180_regrid_fields(target_grid::CubedSphereTargetGeometry{FT},
                                            Nz::Integer) where FT
    Nz >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))
    Nc = target_grid.mesh.Nc
    Nz_int = Int(Nz)
    return ERA5C180RegridFields{FT}(
        ntuple(_ -> zeros(FT, Nc, Nc), 6),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
    )
end

"""
    ERA5C180RegridWorkspace{FT, R}

Owns the conservative regridder + flat scratch buffers used by
[`regrid_n320_to_c180!`](@ref). Allocated once per (source_grid,
target_grid) pair and reused across every window.
"""
struct ERA5C180RegridWorkspace{FT <: AbstractFloat, R}
    regridder    :: R
    src_flat_2d  :: Vector{Float64}
    src_flat_3d  :: Matrix{Float64}
    dst_flat_2d  :: Vector{Float64}
    dst_flat_3d  :: Matrix{Float64}
end

"""
    allocate_era5_c180_regrid_workspace(source_grid, target_grid, Nz; cache_dir=nothing)

Build (or load from `cache_dir`) the N320 → C180 conservative regridder and
allocate the flat scratch buffers. The regridder's `intersections` matrix is
the expensive piece — on first run it is built and serialised to JLD2 under
`cache_dir`, and subsequent runs load it in milliseconds.
"""
function allocate_era5_c180_regrid_workspace(source_grid::ReducedGaussianTargetGeometry{FT},
                                                target_grid::CubedSphereTargetGeometry{FT},
                                                Nz::Integer;
                                                cache_dir::Union{Nothing, AbstractString} = nothing) where FT
    Nz >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))
    Nz_int = Int(Nz)
    regridder = build_regridder(source_grid.mesh, target_grid.mesh;
                                 normalize = false,
                                 cache_dir = cache_dir)
    n_src = length(regridder.src_areas)
    n_dst = length(regridder.dst_areas)
    n_src == ncells(source_grid.mesh) ||
        throw(DimensionMismatch("regridder src_areas length $n_src ≠ N320 cells $(ncells(source_grid.mesh))"))
    n_dst == ncells(target_grid.mesh) ||
        throw(DimensionMismatch("regridder dst_areas length $n_dst ≠ C180 cells $(ncells(target_grid.mesh))"))
    return ERA5C180RegridWorkspace{FT, typeof(regridder)}(
        regridder,
        zeros(Float64, n_src),               # src_flat_2d
        zeros(Float64, n_src, Nz_int),       # src_flat_3d
        zeros(Float64, n_dst),               # dst_flat_2d
        zeros(Float64, n_dst, Nz_int),       # dst_flat_3d
    )
end

"""
    regrid_n320_to_c180!(c180_fields, n320_window, workspace, target_grid) -> c180_fields

Conservatively regrid PS (2D) and U, V, T, Q (3D) from the N320 source mesh
to the C180 cubed-sphere target. All five fields are intensive scalars; the
regridder weights apply directly. The flat scratch buffers in `workspace`
hold the intermediate Float64 arrays so the call is allocation-free after
warm-up.
"""
function regrid_n320_to_c180!(c180_fields::ERA5C180RegridFields{FT},
                                n320_window::ERA5N320WindowFields,
                                workspace::ERA5C180RegridWorkspace{FT},
                                target_grid::CubedSphereTargetGeometry{FT}) where FT
    Nc = target_grid.mesh.Nc
    Nz = size(n320_window.u, 2)
    size(workspace.src_flat_3d, 2) == Nz ||
        throw(DimensionMismatch("workspace Nz $(size(workspace.src_flat_3d, 2)) ≠ window Nz $Nz"))
    size(workspace.dst_flat_3d, 2) == Nz ||
        throw(DimensionMismatch("workspace dst Nz $(size(workspace.dst_flat_3d, 2)) ≠ $Nz"))

    # PS — 2D intensive field.
    _regrid_2d_intensive!(workspace.dst_flat_2d, workspace.src_flat_2d,
                           workspace.regridder, n320_window.ps)
    _unpack_flat_to_cs_panels_2d!(c180_fields.ps, workspace.dst_flat_2d, Nc)

    # 3D intensive fields — all share the same workspace scratch.
    for (src_field, dst_panels) in (
            (n320_window.u,  c180_fields.u),
            (n320_window.v,  c180_fields.v),
            (n320_window.t,  c180_fields.t),
            (n320_window.qv, c180_fields.qv))
        _regrid_3d_intensive!(workspace.dst_flat_3d, workspace.src_flat_3d,
                               workspace.regridder, src_field)
        _unpack_flat_to_cs_panels_3d!(dst_panels, workspace.dst_flat_3d, Nc, Nz)
    end

    return c180_fields
end

# ---------------------------------------------------------------------------
# Internal regrid helpers.
# ---------------------------------------------------------------------------

"""Conservative regrid of a 2D intensive field into the flat `n_dst` output
buffer via the workspace-owned Float64 `src_scratch`. The single `copyto!`
handles any Float32 → Float64 promotion that `apply_regridder!`'s sparse-
matmul requires."""
function _regrid_2d_intensive!(dst_flat::AbstractVector{Float64},
                                src_scratch::AbstractVector{Float64},
                                regridder,
                                src::AbstractVector)
    length(src_scratch) == length(src) ||
        throw(DimensionMismatch("src_scratch length $(length(src_scratch)) ≠ source $(length(src))"))
    copyto!(src_scratch, src)
    apply_regridder!(dst_flat, regridder, src_scratch)
    return dst_flat
end

"""Conservative regrid of a 3D intensive field stored as `(n_src, Nz)` into
the flat `(n_dst, Nz)` output. `apply_regridder!` accepts `AbstractMatrix`
sources and iterates the column dimension internally — the workspace
`src_scratch` only exists to materialise a Float64 view of a Float32
source (one `copyto!`) without per-window allocation."""
function _regrid_3d_intensive!(dst_flat::AbstractMatrix{Float64},
                                src_scratch::AbstractMatrix{Float64},
                                regridder,
                                src::AbstractMatrix)
    size(src_scratch) == size(src) ||
        throw(DimensionMismatch("src_scratch size $(size(src_scratch)) ≠ source $(size(src))"))
    copyto!(src_scratch, src)
    apply_regridder!(dst_flat, regridder, src_scratch)
    return dst_flat
end

"""Unpack a flat `n_dst = 6 × Nc²` vector into 6 panels of `(Nc, Nc)`. The
CS mesh enumerates cells panel-major in column-major Julia order within each
panel: `flat_index = (p-1)·Nc² + (j-1)·Nc + i`, and
`panel[i, j] = flat[offset + (j-1)·Nc + i]`."""
function _unpack_flat_to_cs_panels_2d!(panels::NTuple{6, Matrix{FT}},
                                        flat::AbstractVector{Float64},
                                        Nc::Int) where FT
    length(flat) == 6 * Nc * Nc ||
        throw(DimensionMismatch("flat length $(length(flat)) ≠ 6×$Nc² = $(6*Nc*Nc)"))
    @inbounds for p in 1:6
        panel = panels[p]
        offset = (p - 1) * Nc * Nc
        for j in 1:Nc, i in 1:Nc
            panel[i, j] = FT(flat[offset + (j - 1) * Nc + i])
        end
    end
    return panels
end

"""Unpack a flat `(n_dst, Nz)` matrix into 6 panels of `(Nc, Nc, Nz)`."""
function _unpack_flat_to_cs_panels_3d!(panels::NTuple{6, Array{FT, 3}},
                                        flat::AbstractMatrix{Float64},
                                        Nc::Int, Nz::Int) where FT
    size(flat, 1) == 6 * Nc * Nc ||
        throw(DimensionMismatch("flat rows $(size(flat, 1)) ≠ 6×$Nc² = $(6*Nc*Nc)"))
    size(flat, 2) == Nz ||
        throw(DimensionMismatch("flat cols $(size(flat, 2)) ≠ Nz=$Nz"))
    @inbounds for p in 1:6
        panel = panels[p]
        offset = (p - 1) * Nc * Nc
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            panel[i, j, k] = FT(flat[offset + (j - 1) * Nc + i, k])
        end
    end
    return panels
end

# ---------------------------------------------------------------------------
# Internal helpers carried over from breakpoint B.
# ---------------------------------------------------------------------------

"""Synthesize one 2D spectral field onto an `n_cells`-laid-out output column.
`spectral_to_reduced_scalar!` always writes `Float64`, so the caller supplies
a Float64 `scratch` of the same length as `column` and the result is then
copied into `column` (which may be Float32 or Float64). The copy is one pass
of `n_cells` and is <<1% of the synthesis cost. Callers reuse a single
workspace-owned `scratch` across every level to keep the hot path
allocation-free."""
function _synthesize_into_column!(column::AbstractVector,
                                   spec::AbstractMatrix{ComplexF64},
                                   T::Int,
                                   grid::ReducedGaussianTargetGeometry,
                                   cache::ReducedSpectralThreadCache,
                                   scratch::AbstractVector{Float64})
    length(column) == length(scratch) == ncells(grid.mesh) ||
        throw(DimensionMismatch("column/scratch length mismatch with mesh ($(ncells(grid.mesh)))"))
    spectral_to_reduced_scalar!(scratch, spec, T, grid, cache; centered = true)
    @inbounds for c in eachindex(column)
        column[c] = scratch[c]
    end
    return column
end
