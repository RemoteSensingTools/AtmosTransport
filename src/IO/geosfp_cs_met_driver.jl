# ---------------------------------------------------------------------------
# GEOSFPCubedSphereMetDriver — GEOS-FP mass fluxes on cubed-sphere panels
#
# Two ingestion modes:
#   :binary — preprocessed flat binary via CSBinaryReader (fast, mmap)
#   :netcdf — raw GEOS-FP NetCDF files via read_geosfp_cs_timestep
#
# Extracted from: scripts/run_forward_geosfp_cs_gpu.jl
# ---------------------------------------------------------------------------

using Dates
using Printf
using Statistics: mean
using NCDatasets: NCDataset

# ===========================================================================
# Bilinear interpolation: regular lat-lon → cubed-sphere panels
# ===========================================================================

"""
    _precompute_bilinear_weights(cs_lons, cs_lats, src_lons, src_lats, Nc)

Precompute bilinear interpolation weights for mapping a regular lat-lon grid
to cubed-sphere panel cell centers.

`cs_lons`/`cs_lats` are `(Xdim, Ydim, nf)` arrays from `read_geosfp_cs_grid_info`.
`src_lons`/`src_lats` are 1D vectors of the regular grid coordinates.

Returns `(i0, j0, wx, wy)` — each an NTuple{6} of (Nc, Nc) arrays.
For each CS cell (ii, jj, panel), the interpolated value is:

    (1-wx)*(1-wy)*src[i0,j0] + wx*(1-wy)*src[i0+1,j0] +
    (1-wx)*wy*src[i0,j0+1] + wx*wy*src[i0+1,j0+1]
"""
function _precompute_bilinear_weights(cs_lons, cs_lats, src_lons, src_lats, Nc)
    Nlon = length(src_lons)
    Nlat = length(src_lats)
    dlon = Float64(src_lons[2] - src_lons[1])
    dlat = Float64(src_lats[2] - src_lats[1])
    lon0 = Float64(src_lons[1])
    lat0 = Float64(src_lats[1])

    i0_panels = ntuple(_ -> zeros(Int32, Nc, Nc), 6)
    j0_panels = ntuple(_ -> zeros(Int32, Nc, Nc), 6)
    wx_panels = ntuple(_ -> zeros(Float32, Nc, Nc), 6)
    wy_panels = ntuple(_ -> zeros(Float32, Nc, Nc), 6)

    for p in 1:6
        for jj in 1:Nc, ii in 1:Nc
            # cs_lons/cs_lats from read_geosfp_cs_grid_info: (Xdim, Ydim, nf)
            lon = Float64(cs_lons[ii, jj, p])
            lat = Float64(cs_lats[ii, jj, p])

            # Wrap longitude to source grid range
            while lon < lon0;        lon += 360.0; end
            while lon >= lon0 + 360; lon -= 360.0; end

            # Fractional index in source grid
            fi = (lon - lon0) / dlon + 1.0
            fj = (lat - lat0) / dlat + 1.0

            i_lo = clamp(floor(Int32, fi), Int32(1), Int32(Nlon - 1))
            j_lo = clamp(floor(Int32, fj), Int32(1), Int32(Nlat - 1))

            i0_panels[p][ii, jj] = i_lo
            j0_panels[p][ii, jj] = j_lo
            wx_panels[p][ii, jj] = Float32(clamp(fi - i_lo, 0.0, 1.0))
            wy_panels[p][ii, jj] = Float32(clamp(fj - j_lo, 0.0, 1.0))
        end
    end

    return (i0_panels, j0_panels, wx_panels, wy_panels)
end

"""
Apply precomputed bilinear weights to interpolate a 2D field from lat-lon to
one CS panel. `out` is (Nc, Nc), `src` is (Nlon, Nlat).
"""
function _apply_bilinear_2d!(out::AbstractMatrix, src::AbstractMatrix,
                              i0::Matrix{Int32}, j0::Matrix{Int32},
                              wx::Matrix{Float32}, wy::Matrix{Float32})
    Nc = size(out, 1)
    @inbounds for jj in 1:Nc, ii in 1:Nc
        il = i0[ii, jj];  jl = j0[ii, jj]
        w_x = wx[ii, jj]; w_y = wy[ii, jj]
        out[ii, jj] = (1 - w_x) * (1 - w_y) * src[il,     jl]     +
                            w_x  * (1 - w_y) * src[il + 1, jl]     +
                       (1 - w_x) *      w_y  * src[il,     jl + 1] +
                            w_x  *      w_y  * src[il + 1, jl + 1]
    end
end

"""
Apply bilinear weights to a 3D field (lon, lat, lev) → one CS panel (Nc, Nc, Nz).
"""
function _apply_bilinear_3d!(out::AbstractArray{<:Any, 3}, src::AbstractArray{<:Any, 3},
                              i0::Matrix{Int32}, j0::Matrix{Int32},
                              wx::Matrix{Float32}, wy::Matrix{Float32})
    Nc = size(out, 1)
    Nz = size(out, 3)
    @inbounds for k in 1:Nz, jj in 1:Nc, ii in 1:Nc
        il = i0[ii, jj];  jl = j0[ii, jj]
        w_x = wx[ii, jj]; w_y = wy[ii, jj]
        out[ii, jj, k] = (1 - w_x) * (1 - w_y) * src[il,     jl,     k] +
                               w_x  * (1 - w_y) * src[il + 1, jl,     k] +
                          (1 - w_x) *      w_y  * src[il,     jl + 1, k] +
                               w_x  *      w_y  * src[il + 1, jl + 1, k]
    end
end

"""
Read source (lat-lon) grid coordinates from the first available A1 file
in the given directory for the date range.
"""
function _read_ll_grid_coords(ll_dir::String, start_date::Date, end_date::Date)
    for date in start_date:Day(1):end_date
        datestr = Dates.format(date, "yyyymmdd")
        f = joinpath(ll_dir, "GEOSFP.$(datestr).A1.025x03125.nc")
        if isfile(f)
            return NCDataset(f, "r") do ds
                Array{Float64}(ds["lon"][:]), Array{Float64}(ds["lat"][:])
            end
        end
    end
    error("No A1 lat-lon files found in $ll_dir for $start_date to $end_date")
end

# ===========================================================================

"""
$(TYPEDEF)

Met driver for GEOS-FP cubed-sphere mass fluxes.

$(FIELDS)
"""
struct GEOSFPCubedSphereMetDriver{FT} <: AbstractMassFluxMetDriver{FT}
    "ordered list of data file paths (.bin or .nc4)"
    files           :: Vector{String}
    "ingestion mode: :binary or :netcdf"
    mode            :: Symbol
    "windows per file (for binary multi-file mode)"
    windows_per_file :: Vector{Int}
    "total number of met windows"
    n_windows       :: Int
    "cells per panel edge"
    Nc              :: Int
    "number of vertical levels"
    Nz              :: Int
    "halo width"
    Hp              :: Int
    "time between met updates [s]"
    met_interval    :: FT
    "advection sub-step size [s]"
    dt              :: FT
    "number of advection sub-steps per met window"
    steps_per_win   :: Int
    "NetCDF file for reading panel coordinates (used by binary mode for regridding)"
    coord_file      :: String
    "merge map for vertical level merging (native level → merged level index), or nothing"
    merge_map       :: Union{Nothing, Vector{Int}}
    "accumulation time for mass fluxes [s] (dynamics timestep; defaults to met_interval)"
    mass_flux_dt    :: FT
    "simulation start date (for output timestamps)"
    _start_date     :: Date
    "directory with regridded CS surface fields (A1, A3mstE NetCDF); empty = use CTM co-location"
    surface_data_dir :: String
    "directory with flat binary CS surface fields (A1, A3mstE .bin); empty = use NetCDF fallback"
    surface_data_bin_dir :: String
    "directory with raw 0.25° lat-lon surface fields for on-the-fly regrid to CS"
    surface_data_ll_dir :: String
    "precomputed bilinear weights (i0, j0, wx, wy) for LL→CS regrid; nothing if disabled"
    _bilinear_weights :: Any
    "print field statistics (wind speed, DELP ordering, NaN) on first window and every 24th"
    verbose         :: Bool
end

"""
    GEOSFPCubedSphereMetDriver(; FT, preprocessed_dir="", netcdf_files=[],
                                  start_date, end_date, dt=900, met_interval=3600, Hp=3)

Construct a GEOS-FP cubed-sphere met driver.

If `preprocessed_dir` is given and contains `.bin` files, uses binary mode.
Otherwise falls back to NetCDF files (either from `netcdf_files` or
discovered in a data directory).
"""
function GEOSFPCubedSphereMetDriver(;
        FT::Type{<:AbstractFloat} = Float32,
        preprocessed_dir::String = "",
        netcdf_files::Vector{String} = String[],
        coord_file::String = "",
        start_date::Date = Date("2024-06-01"),
        end_date::Date = Date("2024-06-05"),
        dt::Real = 900,
        met_interval::Real = 3600,
        Hp::Int = 3,
        merge_map::Union{Nothing, Vector{Int}} = nothing,
        mass_flux_dt::Real = met_interval,
        surface_data_dir::String = "",
        surface_data_bin_dir::String = "",
        surface_data_ll_dir::String = "",
        verbose::Bool = false)

    ft_tag = FT == Float32 ? "float32" : "float64"
    steps_per_win = max(1, round(Int, met_interval / dt))
    actual_window_dt = dt * steps_per_win
    if abs(actual_window_dt - met_interval) > 0.01 * met_interval
        error("dt=$dt does not evenly divide met_interval=$met_interval " *
              "(steps_per_win=$steps_per_win gives window of $(actual_window_dt)s). " *
              "Choose dt so that met_interval/dt is an integer.")
    end

    # Precompute bilinear weights for on-the-fly LL→CS regridding
    bilinear_weights = nothing
    if !isempty(surface_data_ll_dir) && isdir(surface_data_ll_dir)
        # Need a coord_file to read CS grid coordinates
        cfile_for_weights = !isempty(coord_file) ? coord_file :
                            !isempty(netcdf_files) ? netcdf_files[1] : ""
        if !isempty(cfile_for_weights) && isfile(cfile_for_weights)
            @info "Precomputing bilinear weights for on-the-fly LL→CS surface regridding..."
            cs_lons, cs_lats, _, _ = read_geosfp_cs_grid_info(cfile_for_weights)
            Nc_grid = size(cs_lons, 1)

            # Find first A1 file to get source grid coordinates
            src_lons, src_lats = _read_ll_grid_coords(surface_data_ll_dir, start_date, end_date)
            bilinear_weights = _precompute_bilinear_weights(
                cs_lons, cs_lats, src_lons, src_lats, Nc_grid)
            @info "  Bilinear weights ready ($(Nc_grid)×$(Nc_grid)×6 cells, " *
                  "$(length(src_lons))×$(length(src_lats)) source grid)"
        else
            @warn "surface_data_ll_dir set but no coord_file available — " *
                  "on-the-fly regridding disabled"
        end
    end

    if !isempty(preprocessed_dir) && isdir(preprocessed_dir)
        # Binary mode
        bin_files = find_preprocessed_cs_files(preprocessed_dir, start_date, end_date, ft_tag)
        isempty(bin_files) && error("No preprocessed .bin files found in $preprocessed_dir")

        wins_per = Int[]
        Nc = 0; Nz_file = 0; Hp_file = 0
        for f in bin_files
            r = CSBinaryReader(f, FT)
            push!(wins_per, r.Nt)
            Nc = r.Nc; Nz_file = r.Nz; Hp_file = r.Hp
            close(r)
        end
        total = sum(wins_per)

        # Use explicit coord_file kwarg, else first NetCDF file, for panel coordinate reading
        cfile = !isempty(coord_file) ? coord_file :
                !isempty(netcdf_files) ? netcdf_files[1] : ""

        return GEOSFPCubedSphereMetDriver{FT}(
            bin_files, :binary, wins_per, total,
            Nc, Nz_file, Hp_file,
            FT(met_interval), FT(dt), steps_per_win,
            cfile, merge_map, FT(mass_flux_dt), start_date,
            surface_data_dir, surface_data_bin_dir, surface_data_ll_dir,
            bilinear_weights, verbose)
    else
        # NetCDF mode
        files = isempty(netcdf_files) ? String[] : netcdf_files
        isempty(files) && error(
            "GEOSFPCubedSphereMetDriver: no preprocessed_dir or netcdf_files provided")

        # Read grid info from first file
        ts0 = read_geosfp_cs_timestep(files[1]; FT)
        Nc = ts0.Nc
        Nz_file = ts0.Nz

        # Probe time dimension: daily files have 24 timesteps, hourly files have 1
        wins_per = Int[]
        for f in files
            nt = NCDataset(f, "r") do ds
                length(ds["time"])
            end
            push!(wins_per, nt)
        end
        n_windows = sum(wins_per)

        return GEOSFPCubedSphereMetDriver{FT}(
            files, :netcdf, wins_per, n_windows,
            Nc, Nz_file, Hp,
            FT(met_interval), FT(dt), steps_per_win,
            files[1],  # coord_file = first NetCDF file
            merge_map, FT(mass_flux_dt), start_date,
            surface_data_dir, surface_data_bin_dir, surface_data_ll_dir,
            bilinear_weights, verbose)
    end
end

# --- Interface implementations ---

total_windows(d::GEOSFPCubedSphereMetDriver)    = d.n_windows
window_dt(d::GEOSFPCubedSphereMetDriver)        = d.dt * d.steps_per_win
steps_per_window(d::GEOSFPCubedSphereMetDriver) = d.steps_per_win
met_interval(d::GEOSFPCubedSphereMetDriver)     = d.met_interval
start_date(d::GEOSFPCubedSphereMetDriver)      = d._start_date

"""
    window_to_file_local(driver::GEOSFPCubedSphereMetDriver, win) → (file_idx, local_win)

Map a global window index to (file index, within-file window index).
"""
function window_to_file_local(d::GEOSFPCubedSphereMetDriver, win::Int)
    cumul = 0
    for (fi, nw) in enumerate(d.windows_per_file)
        if win <= cumul + nw
            return fi, win - cumul
        end
        cumul += nw
    end
    error("Window index $win out of range (total: $(d.n_windows))")
end

"""
    _surface_data_path(driver, file_idx, collection) → String

Look up a regridded surface data file in `surface_data_dir` by date.
The date is derived from the file index (one file per day, day = file_idx offset
from start_date). Returns empty string if the file doesn't exist.
"""
function _surface_data_path(driver::GEOSFPCubedSphereMetDriver, file_idx::Int,
                             collection::String)
    isempty(driver.surface_data_dir) && return ""
    # Compute date: each file covers one day
    date = driver._start_date + Day(file_idx - 1)
    datestr = Dates.format(date, "yyyymmdd")
    fname = "GEOSFP_CS$(driver.Nc).$(datestr).$(collection).nc"
    path = joinpath(driver.surface_data_dir, fname)
    return isfile(path) ? path : ""
end

"""
Derive the A3mstE file path from a CTM_A1 file path by replacing the collection
name in the filename. Returns empty string if the file doesn't exist.
"""
function _a3mste_path_from_ctm(ctm_path::String)
    dir = dirname(ctm_path)
    fname = basename(ctm_path)
    a3_fname = replace(fname, "CTM_A1" => "A3mstE")
    a3_path = joinpath(dir, a3_fname)
    return isfile(a3_path) ? a3_path : ""
end

"""
    _surface_ll_path(driver, file_idx, collection) → String

Look up a raw 0.25° lat-lon surface data file in `surface_data_ll_dir` by date.
Returns empty string if the file doesn't exist or LL dir is not set.
"""
function _surface_ll_path(driver::GEOSFPCubedSphereMetDriver, file_idx::Int,
                           collection::String)
    isempty(driver.surface_data_ll_dir) && return ""
    date = driver._start_date + Day(file_idx - 1)
    datestr = Dates.format(date, "yyyymmdd")
    fname = "GEOSFP.$(datestr).$(collection).025x03125.nc"
    path = joinpath(driver.surface_data_ll_dir, fname)
    return isfile(path) ? path : ""
end

# ===========================================================================
# Flat binary surface readers (A1 + A3mstE .bin files)
#
# Binary layout: [8192 JSON header | timestep₁ | timestep₂ | … | timestepₙ]
# A1 per-timestep:     PBLH_p1..p6  USTAR_p1..p6  HFLUX_p1..p6  T2M_p1..p6
# A3mstE per-timestep: CMFMC_p1..p6   (each panel Nc×Nc×Nz_edge)
# ===========================================================================

const _SFC_BIN_HEADER_SIZE = 8192

"""
Look up a flat binary surface data file in `surface_data_bin_dir` by date.
Returns empty string if the file doesn't exist or dir is not set.
Results are cached to avoid repeated stat syscalls.
"""
const _PATH_CACHE = Dict{Tuple{String,Int,String}, String}()

function _surface_bin_path(driver::GEOSFPCubedSphereMetDriver, file_idx::Int,
                            collection::String)
    isempty(driver.surface_data_bin_dir) && return ""
    key = (driver.surface_data_bin_dir, file_idx, collection)
    return get!(_PATH_CACHE, key) do
        date = driver._start_date + Day(file_idx - 1)
        datestr = Dates.format(date, "yyyymmdd")
        fname = "GEOSFP_CS$(driver.Nc).$(datestr).$(collection).bin"
        path = joinpath(driver.surface_data_bin_dir, fname)
        isfile(path) ? path : ""
    end
end

"""
Load A1 surface fields directly into pre-allocated `sfc_panels`, avoiding
intermediate allocations. File handle, header, and read buffer are cached across calls.

Supports v1 binaries (4 vars: PBLH USTAR HFLUX T2M) and v2 binaries
(adds TROPPT and PS from CTM_A1). The header's `var_names` field determines layout.
"""
# Cached IO for A1 surface binary: (path, IOStream, n_vars, var_names) or nothing
const _SFC_BIN_IO_CACHE = Ref{Any}(nothing)
# Pre-allocated read buffer for A1 binary
const _SFC_BIN_BUF = Ref{Any}(nothing)

function _get_sfc_bin_io(filepath::String)
    cache = _SFC_BIN_IO_CACHE[]
    if cache !== nothing
        old_path, old_io, old_nvars, old_varnames = cache
        if old_path == filepath && isopen(old_io)
            return old_io, old_nvars, old_varnames
        end
        isopen(old_io) && close(old_io)
    end
    io = open(filepath, "r")
    # Parse header for var count and names
    hdr_bytes = Vector{UInt8}(undef, _SFC_BIN_HEADER_SIZE)
    read!(io, hdr_bytes)
    json_end = something(findfirst(==(0x00), hdr_bytes), _SFC_BIN_HEADER_SIZE + 1) - 1
    hdr = JSON3.read(String(hdr_bytes[1:json_end]))
    n_vars = Int(hdr.n_vars)
    var_names = String.(hdr.var_names)
    _SFC_BIN_IO_CACHE[] = (filepath, io, n_vars, var_names)
    return io, n_vars, var_names
end

function _load_sfc_from_bin!(sfc_panels::NamedTuple,
                              filepath::String, time_index::Int,
                              Nc::Int, Hp::Int, ::Type{FT};
                              ps_panels=nothing, troph_panels=nothing) where FT
    io, n_vars, var_names = _get_sfc_bin_io(filepath)

    panel_elems = Nc * Nc
    elems_per_ts = n_vars * 6 * panel_elems
    offset = _SFC_BIN_HEADER_SIZE + (time_index - 1) * elems_per_ts * sizeof(FT)

    # Reuse pre-allocated buffer
    buf = _SFC_BIN_BUF[]
    if buf === nothing || length(buf) != elems_per_ts || eltype(buf) != FT
        buf = Vector{FT}(undef, elems_per_ts)
        _SFC_BIN_BUF[] = buf
    end
    data = buf::Vector{FT}

    seek(io, offset)
    read!(io, data)

    # Unpack each variable by name from the header-declared order
    for (var_idx, varname) in enumerate(var_names)
        target = if varname == "PBLH"
            sfc_panels.pblh
        elseif varname == "USTAR"
            sfc_panels.ustar
        elseif varname == "HFLUX"
            sfc_panels.hflux
        elseif varname == "T2M"
            sfc_panels.t2m
        elseif varname == "PS" && ps_panels !== nothing
            ps_panels
        elseif varname == "TROPPT" && troph_panels !== nothing
            troph_panels
        else
            nothing  # unknown or unneeded variable — skip
        end
        target === nothing && continue

        @inbounds for p in 1:6
            start = ((var_idx - 1) * 6 + (p - 1)) * panel_elems + 1
            fill!(target[p], zero(FT))
            for jj in 1:Nc, ii in 1:Nc
                target[p][ii + Hp, jj + Hp] = data[start + (jj - 1) * Nc + (ii - 1)]
            end
        end
    end
    return true
end

"""
Read CMFMC from binary and write directly into pre-allocated `cmfmc_panels`,
applying vertical merge_map inline. Allocates only a single panel buffer (~151 MB)
instead of the ~2.7 GB of intermediates that the old path created per call.

Reverses vertical dimension (binary stores bottom-to-top from CS NetCDF source)
to match model convention (k=1=TOA), consistent with `read_geosfp_cs_cmfmc`.

File handle, parsed header, and panel buffer are cached across calls.
"""
# Cached IO + header for CMFMC binary: (path, IOStream, Nz_edge) or nothing
const _CMFMC_BIN_IO_CACHE = Ref{Any}(nothing)
# Pre-allocated panel buffer for CMFMC binary
const _CMFMC_PANEL_BUF = Ref{Any}(nothing)

function _get_cmfmc_bin_io(filepath::String)
    cache = _CMFMC_BIN_IO_CACHE[]
    if cache !== nothing
        old_path, old_io, old_nz = cache
        if old_path == filepath && isopen(old_io)
            return old_io, old_nz
        end
        isopen(old_io) && close(old_io)
    end
    io = open(filepath, "r")
    hdr_bytes = Vector{UInt8}(undef, _SFC_BIN_HEADER_SIZE)
    read!(io, hdr_bytes)
    json_end = something(findfirst(==(0x00), hdr_bytes), _SFC_BIN_HEADER_SIZE + 1) - 1
    hdr = JSON3.read(String(hdr_bytes[1:json_end]))
    Nz_edge = Int(hdr.Nz_edge)
    _CMFMC_BIN_IO_CACHE[] = (filepath, io, Nz_edge)
    return io, Nz_edge
end

function _load_cmfmc_from_bin!(cmfmc_panels::NTuple{6},
                                filepath::String, time_index::Int,
                                Nc::Int, Hp::Int,
                                merge_map::Union{Nothing, Vector{Int}},
                                ::Type{FT}) where FT
    io, Nz_edge = _get_cmfmc_bin_io(filepath)

    panel_elems = Nc * Nc * Nz_edge
    elems_per_ts = 6 * panel_elems
    base_offset = _SFC_BIN_HEADER_SIZE + (time_index - 1) * elems_per_ts * sizeof(FT)

    # Reuse pre-allocated panel buffer
    pbuf = _CMFMC_PANEL_BUF[]
    if pbuf === nothing || size(pbuf) != (Nc, Nc, Nz_edge) || eltype(pbuf) != FT
        pbuf = Array{FT}(undef, Nc, Nc, Nz_edge)
        _CMFMC_PANEL_BUF[] = pbuf
    end
    panel_buf = pbuf::Array{FT, 3}

    Nz_edge_merged = merge_map !== nothing ? maximum(merge_map) + 1 : Nz_edge

    for p in 1:6
        seek(io, base_offset + (p - 1) * panel_elems * sizeof(FT))
        read!(io, panel_buf)

        fill!(cmfmc_panels[p], zero(FT))

        if merge_map !== nothing
            @inbounds for k_file in 1:Nz_edge
                # Reverse: file k=1 → model k=Nz_edge (bottom-to-top → TOA-first)
                k_model = Nz_edge - k_file + 1
                # Map model edge to merged edge
                k_merged = if k_model == 1
                    1
                elseif k_model == Nz_edge
                    Nz_edge_merged
                else
                    merge_map[k_model - 1] + 1
                end
                for jj in 1:Nc, ii in 1:Nc
                    cmfmc_panels[p][ii + Hp, jj + Hp, k_merged] =
                        max(cmfmc_panels[p][ii + Hp, jj + Hp, k_merged],
                            panel_buf[ii, jj, k_file])
                end
            end
        else
            @inbounds for k_file in 1:Nz_edge
                k_model = Nz_edge - k_file + 1
                for jj in 1:Nc, ii in 1:Nc
                    cmfmc_panels[p][ii + Hp, jj + Hp, k_model] = panel_buf[ii, jj, k_file]
                end
            end
        end
    end
    return true
end

"""
Read surface fields from a raw 0.25° A1 file and bilinearly interpolate to
CS panels. Returns a NamedTuple `(pblh, ustar, hflux, t2m)` where each is
an NTuple{6} of haloed 2D arrays `(Nc+2Hp, Nc+2Hp)`.
"""
function _read_surface_fields_ll(filepath::String, time_index::Int,
                                  weights, Nc::Int, Hp::Int,
                                  ::Type{FT}) where FT
    i0_p, j0_p, wx_p, wy_p = weights
    panel_buf = zeros(FT, Nc, Nc)

    function _regrid_2d(src_2d)
        ntuple(6) do p
            _apply_bilinear_2d!(panel_buf, src_2d, i0_p[p], j0_p[p], wx_p[p], wy_p[p])
            arr = zeros(FT, Nc + 2Hp, Nc + 2Hp)
            arr[Hp+1:Hp+Nc, Hp+1:Hp+Nc] .= panel_buf
            arr
        end
    end

    ds = NCDataset(filepath, "r")
    try
        pblh  = _regrid_2d(Array{FT}(ds["PBLH"][:, :, time_index]))
        ustar = _regrid_2d(Array{FT}(ds["USTAR"][:, :, time_index]))
        hflux = _regrid_2d(Array{FT}(ds["HFLUX"][:, :, time_index]))
        t2m   = _regrid_2d(Array{FT}(ds["T2M"][:, :, time_index]))
        return (; pblh, ustar, hflux, t2m)
    finally
        close(ds)
    end
end

"""
Read CMFMC from a raw 0.25° A3mstE file and bilinearly interpolate to
CS panels. Returns NTuple{6} of haloed 3D arrays `(Nc+2Hp, Nc+2Hp, Nz_edge)`.

GEOS-FP 0.25° is already top-to-bottom (k=1=TOA), matching model convention.
"""
function _read_cmfmc_ll(filepath::String, time_index::Int,
                         weights, Nc::Int, Hp::Int,
                         ::Type{FT}) where FT
    i0_p, j0_p, wx_p, wy_p = weights

    ds = NCDataset(filepath, "r")
    try
        # CMFMC: (lon=1152, lat=721, lev=73, time=8)
        cmfmc_ll = Array{FT}(ds["CMFMC"][:, :, :, time_index])  # (Nlon, Nlat, Nz_edge)
        Nz_edge = size(cmfmc_ll, 3)

        panel_buf = zeros(FT, Nc, Nc, Nz_edge)

        return ntuple(6) do p
            _apply_bilinear_3d!(panel_buf, cmfmc_ll,
                                 i0_p[p], j0_p[p], wx_p[p], wy_p[p])
            arr = zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz_edge)
            arr[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :] .= panel_buf
            arr
        end
    finally
        close(ds)
    end
end

"""
    load_cmfmc_window!(cmfmc_panels, driver::GEOSFPCubedSphereMetDriver, grid, win)

Load convective mass flux (CMFMC) from A3mstE files into pre-allocated panels.

CMFMC data is 3-hourly (8 timesteps/day) while CTM_A1 is hourly (24/day).
Each A3mstE timestep covers 3 hourly windows.

Caches the last (file_idx, time_index) to skip redundant re-reads: when
3 consecutive hourly windows share the same A3mstE timestep, the data in
`cmfmc_panels` is already correct and the read is skipped entirely.

Returns `true` if CMFMC was loaded, `false` if A3mstE data is not available.
"""
# Module-level cache for CMFMC timestep deduplication
const _CMFMC_CACHE = Ref((file_idx=0, time_idx=0))

function load_cmfmc_window!(cmfmc_panels::NTuple{6},
                             driver::GEOSFPCubedSphereMetDriver{FT},
                             grid, win::Int) where FT
    file_idx, local_win = window_to_file_local(driver, win)

    # Map hourly CTM_A1 window → 3-hourly A3mstE timestep
    # CTM_A1 local_win 1-3 → A3mstE t=1, 4-6 → t=2, ..., 22-24 → t=8
    a3_time_index = cld(local_win, 3)  # ceiling division

    # Skip re-read when the A3mstE timestep hasn't changed — the data in
    # cmfmc_panels is already correct from the previous window's load.
    # Returns :cached so callers can skip redundant GPU uploads.
    if _CMFMC_CACHE[].file_idx == file_idx && _CMFMC_CACHE[].time_idx == a3_time_index
        return :cached
    end

    # Priority 0: flat binary — combined read + merge, minimal allocation
    bin_path = _surface_bin_path(driver, file_idx, "A3mstE")
    if !isempty(bin_path)
        local_win == 1 && @info "  CMFMC: binary reader ($(basename(bin_path)), t=$a3_time_index)"
        result = _load_cmfmc_from_bin!(cmfmc_panels, bin_path, a3_time_index,
                                        driver.Nc, driver.Hp, driver.merge_map, FT)
        result && (_CMFMC_CACHE[] = (file_idx=file_idx, time_idx=a3_time_index))
        return result
    end

    # Priority 1: raw 0.25° lat-lon with on-the-fly bilinear regrid
    ll_path = _surface_ll_path(driver, file_idx, "A3mstE")
    cmfmc_haloed = if !isempty(ll_path) && driver._bilinear_weights !== nothing
        _read_cmfmc_ll(ll_path, a3_time_index, driver._bilinear_weights,
                        driver.Nc, driver.Hp, FT)
    else
        # Priority 2: pre-regridded CS surface_data_dir (NetCDF)
        a3_path = _surface_data_path(driver, file_idx, "A3mstE")
        # Priority 3: co-located A3mstE alongside CTM_A1 (GEOS-IT naming)
        if isempty(a3_path) && driver.mode === :netcdf
            a3_path = _a3mste_path_from_ctm(driver.files[file_idx])
        end
        isempty(a3_path) && return false
        read_geosfp_cs_cmfmc(a3_path; FT, time_index=a3_time_index, Hp=driver.Hp)
    end

    # Handle vertical level merging if active (NetCDF/LL paths only)
    mm = driver.merge_map
    if mm !== nothing
        Nz_merged = maximum(mm)
        Nz_edge_merged = Nz_merged + 1
        for p in 1:6
            fill!(cmfmc_panels[p], zero(FT))
            Nz_edge_native = size(cmfmc_haloed[p], 3)
            Nc_h = size(cmfmc_panels[p], 1)
            for k_native in 1:Nz_edge_native
                if k_native == 1
                    k_merged = 1
                elseif k_native == Nz_edge_native
                    k_merged = Nz_edge_merged
                else
                    k_merged = mm[k_native - 1] + 1
                end
                for jj in 1:Nc_h, ii in 1:Nc_h
                    cmfmc_panels[p][ii, jj, k_merged] =
                        max(cmfmc_panels[p][ii, jj, k_merged],
                            cmfmc_haloed[p][ii, jj, k_native])
                end
            end
        end
    else
        for p in 1:6
            copyto!(cmfmc_panels[p], cmfmc_haloed[p])
        end
    end

    _CMFMC_CACHE[] = (file_idx=file_idx, time_idx=a3_time_index)
    return true
end

"""
Derive the A3dyn file path from a CTM_A1 file path by replacing the collection
name in the filename. Returns empty string if the file doesn't exist.

A3dyn contains 3-hourly dynamics diagnostics including DTRAIN (detraining mass
flux), OMEGA, RH, U, V at layer centers. Used by the RAS convection scheme.
"""
function _a3dyn_path_from_ctm(ctm_path::String)
    dir = dirname(ctm_path)
    fname = basename(ctm_path)
    a3_fname = replace(fname, "CTM_A1" => "A3dyn")
    a3_path = joinpath(dir, a3_fname)
    return isfile(a3_path) ? a3_path : ""
end

"""
$(SIGNATURES)

Load DTRAIN (detraining mass flux) from A3dyn files into pre-allocated panels.

DTRAIN is the convective detrainment rate [kg/m²/s] at layer centers (Nz levels),
required by the RAS convection scheme. The data is 3-hourly (8 timesteps/day),
same cadence as CMFMC in A3mstE.

Priority system (same as CMFMC):
1. Flat binary in `surface_data_bin_dir` (if available)
2. Co-located A3dyn alongside CTM_A1 files (GEOS-IT naming convention)

Returns `true` if DTRAIN was loaded, `false` if A3dyn data is not available.
"""
# Module-level cache for DTRAIN timestep deduplication (same pattern as CMFMC)
const _DTRAIN_CACHE = Ref((file_idx=0, time_idx=0))

function load_dtrain_window!(dtrain_panels::NTuple{6},
                              driver::GEOSFPCubedSphereMetDriver{FT},
                              grid, win::Int) where FT
    file_idx, local_win = window_to_file_local(driver, win)

    # Map hourly CTM_A1 window → 3-hourly A3dyn timestep (same as A3mstE)
    a3_time_index = cld(local_win, 3)

    # Skip re-read when the A3dyn timestep hasn't changed
    # Returns :cached so callers can skip redundant GPU uploads.
    if _DTRAIN_CACHE[].file_idx == file_idx && _DTRAIN_CACHE[].time_idx == a3_time_index
        return :cached
    end

    # Priority 0: flat binary
    bin_path = _surface_bin_path(driver, file_idx, "A3dyn")
    if !isempty(bin_path)
        local_win == 1 && @info "  DTRAIN: binary reader ($(basename(bin_path)), t=$a3_time_index)"
        result = _load_dtrain_from_bin!(dtrain_panels, bin_path, a3_time_index,
                                         driver.Nc, driver.Hp, driver.merge_map, FT)
        result && (_DTRAIN_CACHE[] = (file_idx=file_idx, time_idx=a3_time_index))
        return result
    end

    # Priority 1: co-located A3dyn alongside CTM_A1 (GEOS-IT naming)
    a3_path = if driver.mode === :netcdf
        _a3dyn_path_from_ctm(driver.files[file_idx])
    else
        ""
    end
    isempty(a3_path) && return false

    local_win == 1 && @info "  DTRAIN: NetCDF reader ($(basename(a3_path)), t=$a3_time_index)"
    dtrain_haloed = read_geosfp_cs_dtrain(a3_path; FT, time_index=a3_time_index, Hp=driver.Hp)

    # Handle vertical level merging if active
    mm = driver.merge_map
    if mm !== nothing
        Nz_merged = maximum(mm)
        for p in 1:6
            fill!(dtrain_panels[p], zero(FT))
            Nz_native = size(dtrain_haloed[p], 3)
            Nc_h = size(dtrain_panels[p], 1)
            # DTRAIN at layer centers: sum detrainment from native levels
            # that map to the same merged level
            for k_native in 1:Nz_native
                k_merged = mm[k_native]
                for jj in 1:Nc_h, ii in 1:Nc_h
                    dtrain_panels[p][ii, jj, k_merged] += dtrain_haloed[p][ii, jj, k_native]
                end
            end
        end
    else
        for p in 1:6
            copyto!(dtrain_panels[p], dtrain_haloed[p])
        end
    end

    _DTRAIN_CACHE[] = (file_idx=file_idx, time_idx=a3_time_index)
    return true
end

# Cached IO + header for DTRAIN binary: (path, IOStream, Nz) or nothing
const _DTRAIN_BIN_IO_CACHE = Ref{Any}(nothing)
const _DTRAIN_PANEL_BUF = Ref{Any}(nothing)

function _get_dtrain_bin_io(filepath::String)
    cache = _DTRAIN_BIN_IO_CACHE[]
    if cache !== nothing
        old_path, old_io, old_nz = cache
        if old_path == filepath && isopen(old_io)
            return old_io, old_nz
        end
        isopen(old_io) && close(old_io)
    end
    io = open(filepath, "r")
    hdr_bytes = Vector{UInt8}(undef, _SFC_BIN_HEADER_SIZE)
    read!(io, hdr_bytes)
    json_end = something(findfirst(==(0x00), hdr_bytes), _SFC_BIN_HEADER_SIZE + 1) - 1
    hdr = JSON3.read(String(hdr_bytes[1:json_end]))
    Nz = Int(hdr.Nz)
    _DTRAIN_BIN_IO_CACHE[] = (filepath, io, Nz)
    return io, Nz
end

"""
Load DTRAIN from flat binary A3dyn file into pre-allocated panels.

Uses JSON-header binary format (same as A3mstE). Binary layout:
  [8192-byte JSON header] [timestep 1: DTRAIN p1..p6] [timestep 2] ...
Each panel is Nc×Nc×Nz Float32 at layer centers, bottom-to-top (GEOS-IT native).
"""
function _load_dtrain_from_bin!(dtrain_panels::NTuple{6}, bin_path::String,
                                 time_index::Int, Nc::Int, Hp::Int,
                                 merge_map, ::Type{FT}) where FT
    io, Nz_native = _get_dtrain_bin_io(bin_path)

    panel_elems = Nc * Nc * Nz_native
    elems_per_ts = 6 * panel_elems
    base_offset = _SFC_BIN_HEADER_SIZE + (time_index - 1) * elems_per_ts * sizeof(FT)

    # Reuse pre-allocated panel buffer
    pbuf = _DTRAIN_PANEL_BUF[]
    if pbuf === nothing || size(pbuf) != (Nc, Nc, Nz_native) || eltype(pbuf) != FT
        pbuf = Array{FT}(undef, Nc, Nc, Nz_native)
        _DTRAIN_PANEL_BUF[] = pbuf
    end
    panel_buf = pbuf::Array{FT, 3}

    Nz_out = size(dtrain_panels[1], 3)

    for p in 1:6
        seek(io, base_offset + (p - 1) * panel_elems * sizeof(FT))
        read!(io, panel_buf)

        fill!(dtrain_panels[p], zero(FT))

        if merge_map !== nothing && Nz_native != Nz_out
            @inbounds for k_file in 1:Nz_native
                k_model = Nz_native - k_file + 1  # flip bottom-to-top → TOA-first
                k_merged = merge_map[k_model]
                for jj in 1:Nc, ii in 1:Nc
                    dtrain_panels[p][ii + Hp, jj + Hp, k_merged] +=
                        FT(panel_buf[ii, jj, k_file])
                end
            end
        else
            @inbounds for k_file in 1:Nz_native
                k_model = Nz_native - k_file + 1  # flip bottom-to-top → TOA-first
                for jj in 1:Nc, ii in 1:Nc
                    dtrain_panels[p][ii + Hp, jj + Hp, k_model] = panel_buf[ii, jj, k_file]
                end
            end
        end
    end
    return true
end

"""
Derive the A1 file path from a CTM_A1 file path by replacing the collection
name in the filename. Returns empty string if the file doesn't exist.
"""
function _a1_path_from_ctm(ctm_path::String)
    dir = dirname(ctm_path)
    fname = basename(ctm_path)
    a1_fname = replace(fname, "CTM_A1" => "A1")
    a1_path = joinpath(dir, a1_fname)
    return isfile(a1_path) ? a1_path : ""
end

"""
    load_surface_fields_window!(sfc_panels, driver::GEOSFPCubedSphereMetDriver, grid, win;
                                 troph_panels=nothing, ps_panels=nothing)

Load surface fields (PBLH, USTAR, HFLUX, T2M) from A1 files into pre-allocated panels.

A1 data is hourly (24 timesteps/day), same as CTM_A1, so the time index maps directly.

`sfc_panels` is a NamedTuple `(pblh, ustar, hflux, t2m)` where each is an
NTuple{6} of 2D arrays (Nc+2Hp, Nc+2Hp).

If `troph_panels` is provided, also loads TROPPT (tropopause pressure, Pa).
If `ps_panels` is provided, also loads PS (surface pressure, Pa) from v2+ binaries.

Returns `true` if surface fields were loaded, `false` if A1 data is not available.
"""
function load_surface_fields_window!(sfc_panels::NamedTuple,
                                      driver::GEOSFPCubedSphereMetDriver{FT},
                                      grid, win::Int;
                                      troph_panels=nothing,
                                      ps_panels=nothing) where FT
    file_idx, local_win = window_to_file_local(driver, win)

    # Priority 0: flat binary — direct write into pre-allocated panels
    bin_path = _surface_bin_path(driver, file_idx, "A1")
    if !isempty(bin_path)
        local_win == 1 && @info "  A1: binary reader ($(basename(bin_path)), t=$local_win)"
        return _load_sfc_from_bin!(sfc_panels, bin_path, local_win,
                                    driver.Nc, driver.Hp, FT;
                                    ps_panels=ps_panels, troph_panels=troph_panels)
    end

    # Priority 1: raw 0.25° lat-lon with on-the-fly bilinear regrid
    ll_path = _surface_ll_path(driver, file_idx, "A1")
    fields = if !isempty(ll_path) && driver._bilinear_weights !== nothing
        _read_surface_fields_ll(ll_path, local_win, driver._bilinear_weights,
                                 driver.Nc, driver.Hp, FT)
    else
        # Priority 2: pre-regridded CS surface_data_dir (NetCDF)
        a1_path = _surface_data_path(driver, file_idx, "A1")
        # Priority 3: co-located A1 alongside CTM_A1 (GEOS-IT naming)
        if isempty(a1_path) && driver.mode === :netcdf
            a1_path = _a1_path_from_ctm(driver.files[file_idx])
        end
        isempty(a1_path) && return false
        read_geosfp_cs_surface_fields(a1_path; FT, time_index=local_win, Hp=driver.Hp)
    end

    for p in 1:6
        copyto!(sfc_panels.pblh[p],  fields.pblh[p])
        copyto!(sfc_panels.ustar[p], fields.ustar[p])
        copyto!(sfc_panels.hflux[p], fields.hflux[p])
        copyto!(sfc_panels.t2m[p],   fields.t2m[p])
    end

    # Copy tropopause pressure if both source data and target buffer exist
    if troph_panels !== nothing && hasproperty(fields, :troph)
        for p in 1:6
            copyto!(troph_panels[p], fields.troph[p])
        end
    end

    return true
end

# ---------------------------------------------------------------------------
# Verbose sanity checks
#
# Called on window 1 and every 24th window when driver.verbose = true.
# Checks for:
#   - NaN / Inf contamination
#   - Column pressure sum (~1e5 Pa expected)
#   - Inverted vertical ordering (DELP[k=1] should be thin, stratospheric)
#   - Plausible wind speed from mass flux (catches wrong mass_flux_dt)
# ---------------------------------------------------------------------------

const _GRAV_CS = 9.80616f0
const _R_EARTH_CS = 6.371e6

function _sanity_check_cs_buf(delp::NTuple{6}, am::NTuple{6}, bm::NTuple{6},
                                win::Int, mass_flux_dt::Real)
    p_delp = delp[1]   # (Nc+2Hp, Nc+2Hp, Nz)
    p_am   = am[1]     # (Nc+1, Nc, Nz)

    Nc = size(p_am, 2)
    dy = 2π * _R_EARTH_CS / (4 * Nc)   # approximate cell edge length [m]

    Hp = (size(p_delp, 1) - size(p_am, 1) + 1)   # halo width
    inner = p_delp[Hp:end-Hp+1, Hp:end-Hp+1, :]  # strip halo

    n_nan = count(isnan, inner) + count(isnan, p_am) + count(isnan, bm[1])
    n_inf = count(isinf, inner) + count(isinf, p_am) + count(isinf, bm[1])
    if n_nan > 0 || n_inf > 0
        @warn "[met sanity win=$win] NaN=$n_nan, Inf=$n_inf — data corruption!"
    end

    # Column pressure sum: sum(DELP) over levels ≈ surface pressure ~1e5 Pa
    ps_mean = mean(sum(inner, dims=3))
    if !(8e4 < ps_mean < 1.1e5)
        msg = @sprintf("[met sanity win=%d] Column DELP sum = %.0f Pa (expected ~1e5 Pa). Check units or vertical ordering.", win, ps_mean)
        @warn msg
    end

    # Vertical ordering: DELP at level 1 should be thin (stratosphere ≈ few Pa)
    delp_top = mean(inner[:, :, 1])
    delp_bot = mean(inner[:, :, end])
    if delp_top > delp_bot
        msg = @sprintf("[met sanity win=%d] DELP[k=1]=%.2f > DELP[k=end]=%.1f Pa — levels appear inverted (bottom-to-top). Check auto-detection.", win, delp_top, delp_bot)
        @warn msg
    end

    # Estimated surface wind: am is total mass flux [kg/s] through the cell face.
    # u = am * g / (DELP * dy)  where dy is the cell edge length.
    am_rms   = sqrt(mean(x -> x^2, p_am[:, :, end]))
    u_est    = am_rms * _GRAV_CS / (delp_bot * dy)
    if u_est > 80.0 && win == 1
        msg = @sprintf("[met sanity win=%d] Estimated |u_sfc| ≈ %.1f m/s — suspiciously high! Check mass_flux_dt (currently %.0fs).", win, u_est, mass_flux_dt)
        @warn msg
    elseif u_est < 0.5 && win == 1
        msg = @sprintf("[met sanity win=%d] Estimated |u_sfc| ≈ %.3f m/s — too low! Check mass_flux_dt (currently %.0fs).", win, u_est, mass_flux_dt)
        @warn msg
    end

    msg = @sprintf("[met sanity win=%d] ps≈%.0fPa | DELP top=%.2f bot=%.1f Pa | est |u_sfc|=%.1f m/s (dy=%.0fm) | mass_flux_dt=%.0fs", win, ps_mean, delp_top, delp_bot, u_est, dy, mass_flux_dt)
    @info msg
end

"""
    load_met_window!(cpu_buf::CubedSphereCPUBuffer, driver::GEOSFPCubedSphereMetDriver,
                      grid, win)

Read cubed-sphere met fields for window `win` into CPU staging buffer.

In binary mode, reads directly from mmap'd binary via CSBinaryReader.
The reader is cached across windows in the same file to avoid re-opening
and re-mmapping the ~4 GB binary every window (keeps page cache warm).

In NetCDF mode, reads raw GEOS-FP data and converts to staggered format.
"""
# Module-level cache for CSBinaryReader — keeps mmap alive across windows
const _MET_READER_CACHE = Ref{Any}(nothing)   # (path, reader) or nothing

function _get_cached_reader(filepath::String, ::Type{FT}) where FT
    cached_path = ensure_local_cache(filepath)
    cache = _MET_READER_CACHE[]
    if cache !== nothing
        old_path, old_reader = cache
        if old_path == cached_path
            return old_reader
        end
        close(old_reader)
    end
    reader = CSBinaryReader(cached_path, FT)
    _MET_READER_CACHE[] = (cached_path, reader)
    return reader
end

function load_met_window!(cpu_buf::CubedSphereCPUBuffer,
                           driver::GEOSFPCubedSphereMetDriver{FT},
                           grid, win::Int) where FT
    file_idx, local_win = window_to_file_local(driver, win)
    filepath = driver.files[file_idx]

    if driver.mode === :binary
        reader = _get_cached_reader(filepath, FT)
        mm = driver.merge_map
        Nz_target = size(cpu_buf.delp[1], 3)
        if mm !== nothing && reader.Nz != Nz_target
            # Read native-Nz data into temp buffers, then merge into merged-Nz cpu_buf
            Nz_native = reader.Nz
            Nc = driver.Nc;  Hp = driver.Hp
            delp_tmp = ntuple(_ -> Array{FT}(undef, Nc + 2Hp, Nc + 2Hp, Nz_native), 6)
            am_tmp   = ntuple(_ -> Array{FT}(undef, Nc + 1,   Nc,       Nz_native), 6)
            bm_tmp   = ntuple(_ -> Array{FT}(undef, Nc,       Nc + 1,   Nz_native), 6)
            load_cs_window!(delp_tmp, am_tmp, bm_tmp, reader, local_win)
            for p in 1:6
                fill!(cpu_buf.delp[p], zero(FT))
                fill!(cpu_buf.am[p],   zero(FT))
                fill!(cpu_buf.bm[p],   zero(FT))
                for k in 1:length(mm)
                    km = mm[k]
                    cpu_buf.delp[p][:, :, km] .+= delp_tmp[p][:, :, k]
                    cpu_buf.am[p][:, :, km]   .+= am_tmp[p][:, :, k]
                    cpu_buf.bm[p][:, :, km]   .+= bm_tmp[p][:, :, k]
                end
            end
        else
            # Binary already at target Nz (pre-merged) or no merge needed
            load_cs_window!(cpu_buf.delp, cpu_buf.am, cpu_buf.bm, reader, local_win)
        end
    else  # :netcdf
        # NetCDF mode — read, halo, and stagger
        ts = read_geosfp_cs_timestep(filepath; FT, time_index=local_win,
                                      dt_met=driver.mass_flux_dt,
                                      convert_to_kgs=true)
        delp_haloed, mfxc, mfyc = to_haloed_panels(ts; Hp=driver.Hp)
        am_stag, bm_stag = cgrid_to_staggered_panels(mfxc, mfyc)

        mm = driver.merge_map
        if mm !== nothing
            # Regrid native levels → merged levels by summing within groups
            for p in 1:6
                cpu_buf.delp[p] .= zero(FT)
                cpu_buf.am[p]   .= zero(FT)
                cpu_buf.bm[p]   .= zero(FT)
                for k in 1:length(mm)
                    km = mm[k]
                    cpu_buf.delp[p][:, :, km] .+= delp_haloed[p][:, :, k]
                    cpu_buf.am[p][:, :, km]   .+= am_stag[p][:, :, k]
                    cpu_buf.bm[p][:, :, km]   .+= bm_stag[p][:, :, k]
                end
            end
        else
            for p in 1:6
                copyto!(cpu_buf.delp[p], delp_haloed[p])
                copyto!(cpu_buf.am[p], am_stag[p])
                copyto!(cpu_buf.bm[p], bm_stag[p])
            end
        end
    end

    if driver.verbose && (win == 1 || win % 24 == 0)
        _sanity_check_cs_buf(cpu_buf.delp, cpu_buf.am, cpu_buf.bm,
                              win, driver.mass_flux_dt)
    end
    return nothing
end

"""
    load_all_window!(cpu_buf, cmfmc_cpu, dtrain_cpu, sfc_cpu, troph_cpu,
                      driver, grid, win; needs_cmfmc, needs_dtrain, needs_sfc,
                      ps_panels=nothing)

Load all met data for a single window: mass fluxes + CMFMC + surface fields + DTRAIN.
Designed to be called from the async background thread in the double-buffer run loop
so that all disk reads overlap with GPU compute.

Returns a NamedTuple `(cmfmc, dtrain, sfc)` with the status of each load
(`false`, `true`, or `:cached`).
"""
function load_all_window!(cpu_buf::CubedSphereCPUBuffer,
                           cmfmc_cpu, dtrain_cpu, sfc_cpu, troph_cpu,
                           driver::GEOSFPCubedSphereMetDriver,
                           grid, win::Int;
                           needs_cmfmc::Bool=false,
                           needs_dtrain::Bool=false,
                           needs_sfc::Bool=false,
                           ps_panels=nothing)
    # 1. Mass fluxes (fast via cached mmap reader)
    load_met_window!(cpu_buf, driver, grid, win)

    # 2. CMFMC
    cmfmc_status = if needs_cmfmc && cmfmc_cpu !== nothing
        load_cmfmc_window!(cmfmc_cpu, driver, grid, win)
    else
        false
    end

    # 3. DTRAIN (only if CMFMC loaded)
    dtrain_status = if needs_dtrain && dtrain_cpu !== nothing && cmfmc_status !== false
        load_dtrain_window!(dtrain_cpu, driver, grid, win)
    else
        false
    end

    # 4. Surface fields (+ PS and TROPPT if available in binary)
    sfc_status = if needs_sfc && sfc_cpu !== nothing
        load_surface_fields_window!(sfc_cpu, driver, grid, win;
                                     troph_panels=troph_cpu,
                                     ps_panels=ps_panels)
    else
        false
    end

    return (; cmfmc=cmfmc_status, dtrain=dtrain_status, sfc=sfc_status)
end
