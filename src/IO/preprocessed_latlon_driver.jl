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
    "optional directory with external QV/thermo files for binaries that omit embedded QV"
    qv_dir          :: String
    "if true, suppress all LL QV loading even when files provide it"
    disable_qv      :: Bool
    "native-grid hybrid A coefficients used to merge external QV with ps weights"
    native_A_ifc    :: Vector{Float64}
    "native-grid hybrid B coefficients used to merge external QV with ps weights"
    native_B_ifc    :: Vector{Float64}
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
                                       max_windows::Union{Nothing, Int} = nothing,
                                       qv_dir::String = "",
                                       disable_qv::Bool = false,
                                       native_A_ifc::Vector{Float64} = Float64[],
                                       native_B_ifc::Vector{Float64} = Float64[])
    isempty(files) && error("PreprocessedLatLonMetDriver: no files provided")

    # Note: v2 binary files have pre-merged levels — merge_map should be nothing.
    # Runtime merging from binary is not supported; use the preprocessor instead.

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
        level_top, level_bot, merge_map,
        expanduser(qv_dir), disable_qv, native_A_ifc, native_B_ifc, _start)
end

"""
    embedded_vertical_coordinate(driver) → (A_ifc, B_ifc) or nothing

Return the A/B interface coefficients embedded in a v2 binary header.
Returns `nothing` for v1 binary or NetCDF files.
"""
function embedded_vertical_coordinate(driver::PreprocessedLatLonMetDriver{FT}) where FT
    isempty(driver.files) && return nothing
    filepath = driver.files[1]
    endswith(filepath, ".bin") || return nothing
    r = MassFluxBinaryReader(filepath, FT)
    A = copy(r.A_ifc)
    B = copy(r.B_ifc)
    close(r)
    isempty(A) && return nothing
    return (A_ifc=A, B_ifc=B)
end

function _extract_date_from_filename(filepath::String)
    bn = basename(filepath)
    m = match(r"(20\d{6})", bn)
    if m !== nothing
        s = m[1]
        return Date(parse(Int, s[1:4]), parse(Int, s[5:6]), parse(Int, s[7:8]))
    end
    m = match(r"(20\d{4})", bn)
    if m !== nothing
        s = m[1]
        return Date(parse(Int, s[1:4]), parse(Int, s[5:6]), 1)
    end
    return nothing
end

"""Parse start date from a preprocessed file's time variable units attribute."""
function _detect_start_date(filepath::String)
    if endswith(filepath, ".bin")
        guessed = _extract_date_from_filename(filepath)
        return something(guessed, Date(2000, 1, 1))
    end
    try
        NCDataset(filepath, "r") do ds
            units = ds["time"].attrib["units"]  # e.g. "hours since 2023-06-01 00:00:00"
            m = match(r"since\s+(\d{4})-(\d{2})-(\d{2})", units)
            m !== nothing && return Date(parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3]))
            guessed = _extract_date_from_filename(filepath)
            return something(guessed, Date(2000, 1, 1))
        end
    catch
        @warn "Could not auto-detect start_date from preprocessed file; defaulting to 2000-01-01"
        return Date(2000, 1, 1)
    end
end

function _window_datetime(driver::PreprocessedLatLonMetDriver, win::Int)
    seconds_per_window = round(Int, Float64(window_dt(driver)))
    return DateTime(start_date(driver)) + Dates.Second((win - 1) * seconds_per_window)
end

function _lookup_external_qv_file(driver::PreprocessedLatLonMetDriver, date::Date)
    isempty(driver.qv_dir) && return ""
    isdir(driver.qv_dir) || return ""

    datestr = Dates.format(date, "yyyymmdd")
    direct_candidates = (
        joinpath(driver.qv_dir, "era5_thermo_ml_$(datestr).nc"),
        joinpath(driver.qv_dir, "era5_qv_$(datestr).nc"),
    )
    for path in direct_candidates
        isfile(path) && return path
    end

    for f in readdir(driver.qv_dir; join=true)
        bn = lowercase(basename(f))
        endswith(bn, ".nc") || continue
        contains(bn, datestr) || continue
        (contains(bn, "thermo") || contains(bn, "qv")) || continue
        return f
    end
    return ""
end

function _parse_cf_time_units(units::AbstractString)
    m = match(r"^\s*([A-Za-z]+)\s+since\s+" *
              raw"(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2})(?::(\d{2})(?::(\d{2}))?)?)?",
              units)
    m === nothing && return nothing

    unit = lowercase(m[1])
    year = parse(Int, m[2])
    month = parse(Int, m[3])
    day = parse(Int, m[4])
    hour = m[5] === nothing ? 0 : parse(Int, m[5])
    minute = m[6] === nothing ? 0 : parse(Int, m[6])
    second = m[7] === nothing ? 0 : parse(Int, m[7])
    return (; unit, origin=DateTime(year, month, day, hour, minute, second))
end

function _cf_time_to_datetime(value, units::AbstractString)
    value isa DateTime && return value
    value isa Date && return DateTime(value)
    value isa Real || return nothing

    parsed = _parse_cf_time_units(units)
    parsed === nothing && return nothing

    seconds_per_unit = if startswith(parsed.unit, "second")
        1.0
    elseif startswith(parsed.unit, "minute")
        60.0
    elseif startswith(parsed.unit, "hour")
        3600.0
    elseif startswith(parsed.unit, "day")
        86400.0
    else
        return nothing
    end

    delta_ms = round(Int, 1000 * Float64(value) * seconds_per_unit)
    return parsed.origin + Dates.Millisecond(delta_ms)
end

function _find_qv_time_index(ds, target_dt::DateTime, fallback_idx::Int)
    for tkey in ("valid_time", "time")
        haskey(ds, tkey) || continue
        tvar = ds[tkey]
        times = tvar[:]
        units = try
            String(tvar.attrib["units"])
        catch
            ""
        end

        for (idx, raw_time) in pairs(times)
            decoded = _cf_time_to_datetime(raw_time, units)
            decoded === nothing && continue
            abs(Dates.value(decoded - target_dt)) <= 1000 && return idx
        end

        !isempty(times) && return clamp(fallback_idx, 1, length(times))
    end
    return fallback_idx
end

function _needs_lat_flip(ds)
    for lname in ("latitude", "lat")
        haskey(ds, lname) || continue
        lats = ds[lname][:]
        return length(lats) > 1 && lats[1] > lats[end]
    end
    return false
end

_is_time_dim(name::AbstractString) = name in ("time", "valid_time")
_is_lon_dim(name::AbstractString) = name in ("lon", "longitude")
_is_lat_dim(name::AbstractString) = name in ("lat", "latitude")
_is_level_dim(name::AbstractString) = name in ("lev", "level", "levels", "hybrid", "model_level_number")

function _extract_qv_native(var, tidx::Int, ::Type{FT}) where FT
    dims = lowercase.(String.(collect(dimnames(var))))
    nd = ndims(var)

    lon_axis = findfirst(_is_lon_dim, dims)
    lat_axis = findfirst(_is_lat_dim, dims)
    lev_axis = findfirst(_is_level_dim, dims)
    (lon_axis === nothing || lat_axis === nothing || lev_axis === nothing) && return nothing

    time_axis = findfirst(_is_time_dim, dims)
    idx = if time_axis === nothing
        ntuple(_ -> Colon(), nd)
    else
        ntuple(i -> i == time_axis ? tidx : Colon(), nd)
    end

    raw = FT.(var[idx...])
    kept_dims = time_axis === nothing ? dims : [dims[i] for i in 1:nd if i != time_axis]
    perm = (
        findfirst(_is_lon_dim, kept_dims),
        findfirst(_is_lat_dim, kept_dims),
        findfirst(_is_level_dim, kept_dims),
    )
    any(p -> p === nothing, perm) && return nothing

    perm_tuple = (perm[1]::Int, perm[2]::Int, perm[3]::Int)
    return perm_tuple == (1, 2, 3) ? raw : permutedims(raw, perm_tuple)
end

function _load_ps_window!(ps::Array{FT,2},
                          driver::PreprocessedLatLonMetDriver{FT},
                          win::Int) where FT
    file_idx, local_win = window_to_file_local(driver, win)
    filepath = driver.files[file_idx]

    if endswith(filepath, ".bin")
        reader = MassFluxBinaryReader(filepath, FT)
        off = (local_win - 1) * reader.elems_per_window +
              reader.n_m + reader.n_am + reader.n_bm + reader.n_cm
        copyto!(ps, 1, reader.data, off + 1, reader.n_ps)
        close(reader)
        return true
    end

    ds = NCDataset(filepath, "r")
    try
        haskey(ds, "ps") || return false
        ps .= FT.(ds["ps"][:, :, local_win])
        return true
    finally
        close(ds)
    end
end

function _merge_qv_from_ps!(qv_merged::Array{FT,3}, qv_native::Array{FT,3},
                            ps::Array{FT,2}, A_ifc::Vector{Float64},
                            B_ifc::Vector{Float64}, mm::Vector{Int}) where FT
    Nx, Ny = size(qv_merged, 1), size(qv_merged, 2)
    fill!(qv_merged, zero(FT))
    m_sum = zeros(FT, Nx, Ny, size(qv_merged, 3))

    @inbounds for k in 1:length(mm)
        km = mm[k]
        dA = FT(A_ifc[k + 1] - A_ifc[k])
        dB = FT(B_ifc[k + 1] - B_ifc[k])
        @views for j in 1:Ny, i in 1:Nx
            w = dA + dB * ps[i, j]
            qv_merged[i, j, km] += qv_native[i, j, k] * w
            m_sum[i, j, km] += w
        end
    end

    @inbounds for km in 1:size(qv_merged, 3), j in 1:Ny, i in 1:Nx
        ms = m_sum[i, j, km]
        qv_merged[i, j, km] = ms > zero(FT) ? qv_merged[i, j, km] / ms : zero(FT)
    end
    return nothing
end

function _derive_merge_map_from_interfaces(native_A::Vector{Float64},
                                           native_B::Vector{Float64},
                                           merged_A::Vector{Float64},
                                           merged_B::Vector{Float64})
    length(native_A) == length(native_B) || return nothing
    length(merged_A) == length(merged_B) || return nothing
    length(native_A) >= length(merged_A) || return nothing

    same_ifc(i, j) = isapprox(native_A[i], merged_A[j]; rtol=0, atol=1e-8) &&
                     isapprox(native_B[i], merged_B[j]; rtol=0, atol=1e-10)

    mm = Vector{Int}(undef, length(native_A) - 1)
    native_ifc = 1
    for km in 1:length(merged_A)-1
        same_ifc(native_ifc, km) || return nothing
        while native_ifc < length(native_A)
            mm[native_ifc] = km
            native_ifc += 1
            same_ifc(native_ifc, km + 1) && break
        end
        same_ifc(native_ifc, km + 1) || return nothing
    end

    native_ifc == length(native_A) || return nothing
    return mm
end

function _load_qv_window_external!(qv::Array{FT,3},
                                   driver::PreprocessedLatLonMetDriver{FT},
                                   win::Int) where FT
    qv_file = _lookup_external_qv_file(driver, Date(_window_datetime(driver, win)))
    isempty(qv_file) && return false

    ds = NCDataset(qv_file, "r")
    try
        varname = nothing
        for candidate in ("qv", "QV", "q", "specific_humidity")
            if haskey(ds, candidate)
                varname = candidate
                break
            end
        end
        varname === nothing && return false

        target_dt = _window_datetime(driver, win)
        file_idx, local_win = window_to_file_local(driver, win)
        tidx = _find_qv_time_index(ds, target_dt, local_win)
        qv_native = _extract_qv_native(ds[varname], tidx, FT)
        qv_native === nothing && return false
        if _needs_lat_flip(ds)
            qv_native = qv_native[:, end:-1:1, :]
        end

        size(qv_native, 1) == size(qv, 1) || return false
        size(qv_native, 2) == size(qv, 2) || return false

        if size(qv_native, 3) == size(qv, 3)
            qv .= qv_native
            return true
        end

        mm = driver.merge_map
        if mm === nothing
            embedded_vc = embedded_vertical_coordinate(driver)
            if embedded_vc !== nothing &&
               length(driver.native_A_ifc) == size(qv_native, 3) + 1
                mm = _derive_merge_map_from_interfaces(driver.native_A_ifc,
                                                       driver.native_B_ifc,
                                                       embedded_vc.A_ifc,
                                                       embedded_vc.B_ifc)
            end
        end
        if mm !== nothing &&
           size(qv_native, 3) == length(mm) &&
           length(driver.native_A_ifc) == length(mm) + 1 &&
           length(driver.native_B_ifc) == length(mm) + 1
            ps = Array{FT}(undef, size(qv, 1), size(qv, 2))
            _load_ps_window!(ps, driver, win) || return false
            _merge_qv_from_ps!(qv, qv_native, ps,
                               driver.native_A_ifc, driver.native_B_ifc, mm)
            return true
        end

        return false
    finally
        close(ds)
    end
end

# --- Interface implementations ---

total_windows(d::PreprocessedLatLonMetDriver)    = d.n_windows

"""Check if the first binary file has v4 flux deltas."""
function has_flux_delta(d::PreprocessedLatLonMetDriver{FT}) where FT
    isempty(d.files) && return false
    f = d.files[1]
    endswith(f, ".bin") || return false
    r = MassFluxBinaryReader(f, FT)
    result = r.has_flux_delta
    close(r)
    return result
end
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
        # v4: load flux deltas if available and buffer is allocated
        if length(cpu_buf.dam) > 0
            load_flux_delta_window!(cpu_buf.dam, cpu_buf.dbm, cpu_buf.dm, reader, local_win)
        end
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

    # Enforce cm boundary: cm[:,:,1]=0 (TOA) and cm[:,:,Nz+1]=0 (surface).
    # Binary files have residual correction applied during preprocessing.
    # Raw NetCDF files may have non-zero surface cm from Float32 accumulation
    # or missing B-correction. TM5 explicitly enforces these (advectz.F90:320).
    _enforce_cm_boundaries!(cpu_buf.cm)

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

    # Recompute cm from merged horizontal divergence (continuity equation).
    # Picking native interface values was incorrect — it breaks continuity
    # when combined with the residual correction.
    Nx = size(cpu_buf.m, 1)
    Ny = size(cpu_buf.m, 2)
    @inbounds for k in 1:Nz_merged, j in 1:Ny, i in 1:Nx
        div_h = (cpu_buf.am[i+1, j, k] - cpu_buf.am[i, j, k]) +
                (cpu_buf.bm[i, j+1, k] - cpu_buf.bm[i, j, k])
        cpu_buf.cm[i, j, k+1] = cpu_buf.cm[i, j, k] - div_h
    end
    # Do NOT apply _correct_cm_residual! — it breaks per-level continuity.
    # The recomputed cm satisfies continuity exactly (Float32 precision).
    return nothing
end

"""
Enforce cm[:,:,1]=0 and cm[:,:,Nz+1]=0 (TM5 advectz.F90:320).
"""
function _enforce_cm_boundaries!(cm::Array{FT,3}) where FT
    @views cm[:, :, 1]   .= zero(FT)
    @views cm[:, :, end] .= zero(FT)
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

    if endswith(filepath, ".bin")
        reader = MassFluxBinaryReader(filepath, FT)
        ok = load_cmfmc_window!(cmfmc, reader, local_win)
        close(reader)
        return ok
    end

    ds = NCDataset(filepath, "r")
    try
        haskey(ds, "conv_mass_flux") || return false
        mm = driver.merge_map
        if mm === nothing
            cmfmc .= FT.(ds["conv_mass_flux"][:, :, :, local_win])
        else
            # Read native interfaces, merge: pick interface at merged level boundaries
            cmfmc_native = FT.(ds["conv_mass_flux"][:, :, :, local_win])
            Nz_merged = maximum(mm)
            cmfmc[:, :, 1] .= cmfmc_native[:, :, 1]  # TOA
            for km in 1:Nz_merged
                k_last = findlast(==(km), mm)
                cmfmc[:, :, km + 1] .= cmfmc_native[:, :, k_last + 1]
            end
        end
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

    if endswith(filepath, ".bin")
        reader = MassFluxBinaryReader(filepath, FT)
        ok = load_tm5conv_window!(tm5conv.entu, tm5conv.detu,
                                   tm5conv.entd, tm5conv.detd,
                                   reader, local_win)
        close(reader)
        return ok
    end

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

    if endswith(filepath, ".bin")
        reader = MassFluxBinaryReader(filepath, FT)
        ok = load_surface_window!(sfc.pblh, sfc.t2m, sfc.ustar, sfc.hflux,
                                   reader, local_win)
        close(reader)
        return ok
    end

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
If level merging is active, reads native levels and merges with mass-weighting.
Returns `true` if data was loaded, `false` if not available.
"""
function load_qv_window!(qv::Array{FT, 3},
                          driver::PreprocessedLatLonMetDriver{FT},
                          grid, win::Int) where FT
    driver.disable_qv && return false
    file_idx, local_win = window_to_file_local(driver, win)
    filepath = driver.files[file_idx]

    if endswith(filepath, ".bin")
        reader = MassFluxBinaryReader(filepath, FT)
        ok = load_qv_window!(qv, reader, local_win)
        close(reader)
        ok && return ok
        return _load_qv_window_external!(qv, driver, win)
    end

    ds = NCDataset(filepath, "r")
    try
        for varname in ("qv", "QV", "q", "specific_humidity")
            haskey(ds, varname) || continue
            mm = driver.merge_map
            if mm === nothing
                qv .= FT.(ds[varname][:, :, :, local_win])
            else
                # Read native levels, merge QV mass-weighted by air mass
                qv_native = FT.(ds[varname][:, :, :, local_win])
                m_native  = FT.(ds["m"][:, :, :, local_win])
                _merge_qv!(qv, qv_native, m_native, mm)
            end
            return true
        end
        return false
    finally
        close(ds)
    end
end

"""
    load_temperature_window!(t, driver::PreprocessedLatLonMetDriver, grid, win) → Bool

Read model-level temperature for window `win` into `t` (Nx, Ny, Nz).
Returns `true` if data was loaded, `false` if not available.
"""
function load_temperature_window!(t::Array{FT, 3},
                                   driver::PreprocessedLatLonMetDriver{FT},
                                   grid, win::Int) where FT
    file_idx, local_win = window_to_file_local(driver, win)
    filepath = driver.files[file_idx]

    if endswith(filepath, ".bin")
        reader = MassFluxBinaryReader(filepath, FT)
        ok = load_temperature_window!(t, reader, local_win)
        close(reader)
        return ok
    end

    ds = NCDataset(filepath, "r")
    try
        for varname in ("temperature", "t", "T")
            haskey(ds, varname) || continue
            mm = driver.merge_map
            if mm === nothing
                t .= FT.(ds[varname][:, :, :, local_win])
            else
                t_native = FT.(ds[varname][:, :, :, local_win])
                m_native = FT.(ds["m"][:, :, :, local_win])
                _merge_qv!(t, t_native, m_native, mm)  # mass-weighted avg same as QV
            end
            return true
        end
        return false
    finally
        close(ds)
    end
end

"""Mass-weighted merge of QV: qv_merged[km] = Σ(qv[k] × m[k]) / Σ(m[k]) over native levels in group km."""
function _merge_qv!(qv_merged::Array{FT,3}, qv_native::Array{FT,3},
                     m_native::Array{FT,3}, mm::Vector{Int}) where FT
    Nx, Ny = size(qv_merged, 1), size(qv_merged, 2)
    Nz_native = length(mm)
    fill!(qv_merged, zero(FT))
    m_sum = zeros(FT, Nx, Ny, size(qv_merged, 3))
    for k in 1:Nz_native
        km = mm[k]
        @views begin
            qv_merged[:, :, km] .+= qv_native[:, :, k] .* m_native[:, :, k]
            m_sum[:, :, km]     .+= m_native[:, :, k]
        end
    end
    for km in 1:size(qv_merged, 3)
        @views begin
            mask = m_sum[:, :, km] .> zero(FT)
            qv_merged[:, :, km] .= ifelse.(mask, qv_merged[:, :, km] ./ m_sum[:, :, km], zero(FT))
        end
    end
end
