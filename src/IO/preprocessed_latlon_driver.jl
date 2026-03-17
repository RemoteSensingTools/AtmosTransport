# ---------------------------------------------------------------------------
# PreprocessedLatLonMetDriver — reads pre-computed mass fluxes from binary/NetCDF
#
# Mass fluxes (m, am, bm, cm, ps) were pre-computed by
# preprocess_mass_fluxes.jl. This driver reads them directly — no wind
# staggering or pressure computation needed.
#
# Supports two file formats:
#   .bin — mmap'd flat binary via MassFluxBinaryReader (fast, preferred)
#   .nc  — NetCDF4 random-access
#
# Supports monthly-sharded files: set `directory` and file discovery
# chains through shards in chronological order.
#
# Extracted from: scripts/run_forward_preprocessed.jl
# ---------------------------------------------------------------------------

using NCDatasets
using Dates

"""
$(TYPEDEF)

Met driver for pre-computed lat-lon mass fluxes (binary or NetCDF).

$(FIELDS)
"""
struct PreprocessedLatLonMetDriver{FT} <: AbstractMassFluxMetDriver{FT}
    "ordered list of mass-flux file paths (.bin or .nc)"
    files           :: Vector{String}
    "windows per file (indexed by file position)"
    windows_per_file :: Vector{Int}
    "total number of met windows across all files"
    n_windows       :: Int
    "advection sub-step size [s]"
    dt              :: FT
    "number of advection sub-steps per met window"
    steps_per_win   :: Int
    "longitude vector"
    lons            :: Vector{FT}
    "latitude vector"
    lats            :: Vector{FT}
    Nx              :: Int
    Ny              :: Int
    Nz              :: Int
    "topmost model level index"
    level_top       :: Int
    "bottommost model level index"
    level_bot       :: Int
    "merge map for vertical level merging (native level → merged level index), or nothing"
    merge_map       :: Union{Nothing, Vector{Int}}
    "simulation start date (auto-detected from file time variable)"
    _start_date     :: Date
end

"""
    PreprocessedLatLonMetDriver(; FT, files, dt=nothing, merge_map=nothing)

Construct a preprocessed lat-lon met driver from file list.
Grid metadata and dt are read from the first file's header.
If `dt` is provided, it overrides the file's embedded value.
If `merge_map` is provided, levels are merged on load (see `merge_upper_levels`).
"""
function PreprocessedLatLonMetDriver(; FT::Type{<:AbstractFloat} = Float64,
                                       files::Vector{String},
                                       dt::Union{Nothing, Real} = nothing,
                                       merge_map::Union{Nothing, Vector{Int}} = nothing,
                                       max_windows::Union{Nothing, Int} = nothing)
    isempty(files) && error("PreprocessedLatLonMetDriver: no files provided")

    merge_map !== nothing && !isempty(files) && endswith(files[1], ".bin") &&
        error("Binary mode with layer merging not yet supported. Use NetCDF files instead.")

    # Read metadata from first file — dispatch on extension
    if endswith(files[1], ".bin")
        r = MassFluxBinaryReader(files[1], FT)
        Nx, Ny, Nz = r.Nx, r.Ny, r.Nz
        file_dt = r.dt_seconds
        steps_per = r.steps_per_met
        lons = copy(r.lons)
        lats = copy(r.lats)
        level_top = r.level_top
        level_bot = r.level_bot
        close(r)
    else
        # NetCDF mass-flux shard
        ds = NCDataset(files[1], "r")
        try
            lons = FT.(ds["lon"][:])
            lats = FT.(ds["lat"][:])
            Nx = length(lons)
            Ny = length(lats)
            m_var = ds["m"]
            Nz = size(m_var, 3)
            level_top = get(ds.attrib, "level_top", 50)
            level_bot = get(ds.attrib, "level_bot", 137)
            file_dt = FT(get(ds.attrib, "dt_seconds", 900))
            steps_per = get(ds.attrib, "steps_per_met_window", 4)
        finally
            close(ds)
        end
    end

    actual_dt = dt === nothing ? file_dt : FT(dt)
    file_met_interval = file_dt * steps_per
    steps_per_win = dt === nothing ? steps_per : max(1, round(Int, file_met_interval / actual_dt))
    actual_window_dt = actual_dt * steps_per_win
    if abs(actual_window_dt - file_met_interval) > 0.01 * file_met_interval
        error("dt=$actual_dt does not evenly divide met window duration=$file_met_interval " *
              "(steps_per_win=$steps_per_win gives window of $(actual_window_dt)s). " *
              "Choose dt so that met_interval/dt is an integer.")
    end

    # Count windows per file
    wins_per = Int[]
    total = 0
    for f in files
        if endswith(f, ".bin")
            r2 = MassFluxBinaryReader(f, FT)
            push!(wins_per, r2.Nt)
            close(r2)
        else
            ds = NCDataset(f, "r")
            push!(wins_per, size(ds["m"], 4))
            close(ds)
        end
        total += wins_per[end]
    end

    # Apply max_windows limit if specified
    if max_windows !== nothing && max_windows > 0
        total = min(total, max_windows)
    end

    # Auto-detect start date from first file
    _start = _detect_start_date(files[1])

    PreprocessedLatLonMetDriver{FT}(
        files, wins_per, total,
        FT(actual_dt), steps_per_win,
        lons, lats, Nx, Ny, Nz,
        level_top, level_bot, merge_map, _start)
end

"""Parse start date from a preprocessed file's time variable units attribute."""
function _detect_start_date(filepath::String)
    if endswith(filepath, ".bin")
        return Date(2000, 1, 1)  # binary files have no time metadata
    end
    try
        NCDataset(filepath, "r") do ds
            units = ds["time"].attrib["units"]  # e.g. "hours since 2023-06-01 00:00:00"
            m = match(r"since\s+(\d{4})-(\d{2})-(\d{2})", units)
            m !== nothing && return Date(parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3]))
            return Date(2000, 1, 1)
        end
    catch
        @warn "Could not auto-detect start_date from preprocessed file; defaulting to 2000-01-01"
        return Date(2000, 1, 1)
    end
end

# --- Interface implementations ---

total_windows(d::PreprocessedLatLonMetDriver)    = d.n_windows
window_dt(d::PreprocessedLatLonMetDriver)        = d.dt * d.steps_per_win
steps_per_window(d::PreprocessedLatLonMetDriver) = d.steps_per_win
start_date(d::PreprocessedLatLonMetDriver)       = d._start_date

"""
    window_to_file_local(driver, win) → (file_idx, local_win)

Map a global window index to (file index, within-file window index).
"""
function window_to_file_local(d::PreprocessedLatLonMetDriver, win::Int)
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
    load_met_window!(cpu_buf::LatLonCPUBuffer, driver::PreprocessedLatLonMetDriver, grid, win)

Read pre-computed mass fluxes for window `win` into `cpu_buf`.
Opens the appropriate file, reads the window, and closes.
"""
function load_met_window!(cpu_buf::LatLonCPUBuffer,
                           driver::PreprocessedLatLonMetDriver{FT},
                           grid, win::Int) where FT
    file_idx, local_win = window_to_file_local(driver, win)
    filepath = driver.files[file_idx]

    mm = driver.merge_map

    if endswith(filepath, ".bin")
        reader = MassFluxBinaryReader(filepath, FT)
        load_window!(cpu_buf.m, cpu_buf.am, cpu_buf.bm, cpu_buf.cm, cpu_buf.ps,
                     reader, local_win)
        close(reader)
    elseif mm === nothing
        # Direct read — no level merging
        ds = NCDataset(filepath, "r")
        try
            cpu_buf.m  .= FT.(ds["m"][:, :, :, local_win])
            cpu_buf.am .= FT.(ds["am"][:, :, :, local_win])
            cpu_buf.bm .= FT.(ds["bm"][:, :, :, local_win])
            cpu_buf.cm .= FT.(ds["cm"][:, :, :, local_win])
            cpu_buf.ps .= FT.(ds["ps"][:, :, local_win])
        finally
            close(ds)
        end
    else
        # Read native levels into temporaries, then merge
        ds = NCDataset(filepath, "r")
        try
            cpu_buf.ps .= FT.(ds["ps"][:, :, local_win])
            _merge_levels_latlon!(cpu_buf,
                FT.(ds["m"][:, :, :, local_win]),
                FT.(ds["am"][:, :, :, local_win]),
                FT.(ds["bm"][:, :, :, local_win]),
                FT.(ds["cm"][:, :, :, local_win]), mm)
        finally
            close(ds)
        end
    end

    # Distribute the continuity residual across all cm interfaces.
    # The preprocessor computes cm from continuity (top→bottom accumulation),
    # so the surface value cm[:,:,Nz+1] accumulates the column mass-budget
    # residual. In relative terms this is tiny (< 0.001% of column mass),
    # but it can exceed thin-cell mass near the surface, causing CFL_z > 300
    # and thousands of pointless Z subcycles.
    # Fix: distribute the residual proportionally to the cumulative mass
    # fraction above each interface, so cm[1] = cm[Nz+1] = 0.
    _correct_cm_residual!(cpu_buf.cm, cpu_buf.m)

    return nothing
end

"""
Merge native-level mass fluxes into coarser merged levels using `merge_map`.

For m/am/bm: sum native levels within each merged group.
For cm: pick native cm at merged interface boundaries (consistent with continuity).
"""
function _merge_levels_latlon!(cpu_buf::LatLonCPUBuffer{FT},
                                m_native, am_native, bm_native, cm_native,
                                mm::Vector{Int}) where FT
    Nz_native = length(mm)
    Nz_merged = maximum(mm)

    # Zero out merged buffers
    fill!(cpu_buf.m, zero(FT))
    fill!(cpu_buf.am, zero(FT))
    fill!(cpu_buf.bm, zero(FT))
    fill!(cpu_buf.cm, zero(FT))

    # Sum m/am/bm over native levels within each merged group
    for k in 1:Nz_native
        km = mm[k]
        @views cpu_buf.m[:, :, km]  .+= m_native[:, :, k]
        @views cpu_buf.am[:, :, km] .+= am_native[:, :, k]
        @views cpu_buf.bm[:, :, km] .+= bm_native[:, :, k]
    end

    # cm at merged interfaces: pick from native cm at group boundaries
    # cm_merged[:,:,1] = cm_native[:,:,1] = 0 (TOA)
    @views cpu_buf.cm[:, :, 1] .= cm_native[:, :, 1]
    for km in 1:Nz_merged
        # Bottom interface of merged level km = first native interface
        # after the last native level in group km
        k_last = findlast(==(km), mm)
        @views cpu_buf.cm[:, :, km + 1] .= cm_native[:, :, k_last + 1]
    end
    return nothing
end

"""
Distribute the continuity residual in cm so that cm[:,;,1]=0 (TOA) and
cm[:,:,Nz+1]=0 (surface). The correction at each interface is proportional
to the cumulative mass fraction above it:

    cm_corrected[k] = cm_raw[k] - residual × Σm[1:k-1] / Σm[1:Nz]

This preserves the divergence between adjacent interfaces to first order
while removing the accumulated numerical error from spectral wind closure.
"""
function _correct_cm_residual!(cm::Array{FT,3}, m::Array{FT,3}) where FT
    Nx, Ny, Nz_plus1 = size(cm)
    Nz = Nz_plus1 - 1

    @inbounds for j in 1:Ny, i in 1:Nx
        residual = cm[i, j, Nz + 1]
        abs(residual) < eps(FT) && continue

        # Total column mass
        col_mass = zero(FT)
        for k in 1:Nz
            col_mass += m[i, j, k]
        end
        col_mass < eps(FT) && continue

        # Distribute: cm[1] stays 0, cm[Nz+1] becomes 0
        cum_mass = zero(FT)
        cm[i, j, 1] = zero(FT)
        for k in 1:Nz
            cum_mass += m[i, j, k]
            cm[i, j, k + 1] -= residual * cum_mass / col_mass
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Optional physics fields: convective mass flux + surface fields
#
# These are read from NetCDF if the variables exist in the file.
# Binary format does not support physics fields (returns false).
# ---------------------------------------------------------------------------

"""
    load_cmfmc_window!(cmfmc, driver::PreprocessedLatLonMetDriver, grid, win) → Bool

Read convective mass flux for window `win` into `cmfmc` (Nx, Ny, Nz+1).
Returns `true` if data was loaded, `false` if the file doesn't contain
`conv_mass_flux` (or is binary format).
"""
function load_cmfmc_window!(cmfmc::Array{FT, 3},
                             driver::PreprocessedLatLonMetDriver{FT},
                             grid, win::Int) where FT
    file_idx, local_win = window_to_file_local(driver, win)
    filepath = driver.files[file_idx]

    # Binary format doesn't support physics fields
    endswith(filepath, ".bin") && return false

    ds = NCDataset(filepath, "r")
    try
        haskey(ds, "conv_mass_flux") || return false
        cmfmc .= FT.(ds["conv_mass_flux"][:, :, :, local_win])
    finally
        close(ds)
    end
    return true
end

"""
    load_tm5conv_window!(tm5conv, driver::PreprocessedLatLonMetDriver, grid, win) → Bool

Read TM5 convection fields (entu, detu, entd, detd) for window `win`.
`tm5conv` is a NamedTuple with fields `entu`, `detu`, `entd`, `detd`,
each of size (Nx, Ny, Nz).
Returns `true` if all four fields were loaded, `false` if any are missing
(or binary format).
"""
function load_tm5conv_window!(tm5conv::NamedTuple,
                               driver::PreprocessedLatLonMetDriver{FT},
                               grid, win::Int) where FT
    file_idx, local_win = window_to_file_local(driver, win)
    filepath = driver.files[file_idx]

    endswith(filepath, ".bin") && return false

    ds = NCDataset(filepath, "r")
    try
        for (name, arr) in pairs(tm5conv)
            sname = String(name)
            haskey(ds, sname) || return false
            arr .= FT.(ds[sname][:, :, :, local_win])
        end
    finally
        close(ds)
    end
    return true
end

"""
    load_surface_fields_window!(sfc, driver::PreprocessedLatLonMetDriver, grid, win) → Bool

Read PBL surface fields for window `win` into `sfc` NamedTuple with
`pblh`, `ustar`, `hflux`, `t2m` arrays (each Nx × Ny).
Returns `true` if all four fields were loaded, `false` if any are missing
(or binary format).
"""
function load_surface_fields_window!(sfc::NamedTuple,
                                      driver::PreprocessedLatLonMetDriver{FT},
                                      grid, win::Int) where FT
    file_idx, local_win = window_to_file_local(driver, win)
    filepath = driver.files[file_idx]

    endswith(filepath, ".bin") && return false

    ds = NCDataset(filepath, "r")
    try
        for name in (:pblh, :ustar, :hflux, :t2m)
            haskey(ds, String(name)) || return false
        end
        sfc.pblh  .= FT.(ds["pblh"][:, :, local_win])
        sfc.ustar .= FT.(ds["ustar"][:, :, local_win])
        sfc.hflux .= FT.(ds["hflux"][:, :, local_win])
        sfc.t2m   .= FT.(ds["t2m"][:, :, local_win])
    finally
        close(ds)
    end
    return true
end

"""
    load_qv_window!(qv, driver::PreprocessedLatLonMetDriver, grid, win) → Bool

Read specific humidity (QV) for window `win` into `qv` (Nx, Ny, Nz).
Returns `true` if data was loaded, `false` if the file doesn't contain
a QV variable (or is binary format).
"""
function load_qv_window!(qv::Array{FT, 3},
                          driver::PreprocessedLatLonMetDriver{FT},
                          grid, win::Int) where FT
    file_idx, local_win = window_to_file_local(driver, win)
    filepath = driver.files[file_idx]

    endswith(filepath, ".bin") && return false

    ds = NCDataset(filepath, "r")
    try
        for varname in ("qv", "QV", "q", "specific_humidity")
            if haskey(ds, varname)
                qv .= FT.(ds[varname][:, :, :, local_win])
                return true
            end
        end
        return false
    finally
        close(ds)
    end
end
