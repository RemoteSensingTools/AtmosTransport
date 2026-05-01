# ERA5 single-level surface reader for raw PBL diffusion fields.
#
# Supports NetCDF files that are either plain `.nc` or CDS ZIP payloads
# saved with an `.nc` suffix. Fields are normalized to the lat-lon
# transport-binary convention: longitude centered on [-180, 180) and
# latitude south-to-north.

const ERA5_SURFACE_VAR_CANDIDATES = Dict(
    :pblh  => ("pblh", "blh", "boundary_layer_height"),
    :ustar => ("ustar", "zust", "friction_velocity"),
    :hflux => ("hflux", "sshf", "surface_sensible_heat_flux"),
    :t2m   => ("t2m", "2t", "2m_temperature"),
    :u10   => ("u10", "10u", "10m_u_component_of_wind"),
    :v10   => ("v10", "10v", "10m_v_component_of_wind"),
)

const ERA5_SURFACE_TIME_DIMS = ("time", "valid_time")
const ERA5_USTAR_DRAG_COEFFICIENT = 1.2e-3

mutable struct ERA5SurfaceReader
    datasets :: Vector{NCDataset}
    path     :: String
    tmp_dir  :: Union{Nothing, String}
    date     :: Date
    Nx       :: Int
    Ny       :: Int
    time_len :: Int
end

function _resolve_era5_surface_path(surface_dir::AbstractString, date::Date)
    root = expand_data_path(surface_dir)
    ym = Dates.format(date, "yyyymm")
    ymd = Dates.format(date, "yyyymmdd")
    candidates = String[
        joinpath(root, "era5_surface_$(ymd).nc"),
        joinpath(root, "era5_surface_$(ym).nc"),
        joinpath(root, "surface_$(ymd).nc"),
        joinpath(root, "surface_$(ym).nc"),
        joinpath(root, "sfc_an_native", "era5_surface_$(ymd).nc"),
        joinpath(root, "sfc_an_native", "era5_surface_$(ym).nc"),
    ]
    for path in candidates
        isfile(path) && return path
    end
    throw(ArgumentError(
        "ERA5 surface fields not found for $(date). Looked for: " *
        join(candidates, ", ")))
end

function _is_zip_payload(path::String)
    open(path, "r") do io
        sig = read(io, min(4, filesize(path)))
        return length(sig) >= 2 && sig[1] == 0x50 && sig[2] == 0x4b
    end
end

function _surface_extract_dir(path::String)
    parent = dirname(abspath(path))
    try
        return mktempdir(parent; prefix = ".surface_extract_")
    catch err
        @warn "Could not create ERA5 surface extraction directory next to source; using scratch volume" path exception=(err, catch_backtrace())
        return _surface_scratch_extract_dir()
    end
end

function _surface_scratch_extract_dir()
    candidates = String[]
    for key in ("ATMOSTR_SCRATCH_DIR", "ATMOSTR_TMPDIR", "SCRATCH", "TMPDIR", "TEMP", "TMP")
        value = strip(get(ENV, key, ""))
        isempty(value) && continue
        append!(candidates, split(value, Sys.iswindows() ? ';' : ':'))
    end
    push!(candidates, tempdir())

    seen = Set{String}()
    for parent in candidates
        parent = isempty(parent) ? parent : abspath(expanduser(parent))
        isempty(parent) && continue
        parent in seen && continue
        push!(seen, parent)
        isdir(parent) || continue
        try
            return mktempdir(parent; prefix = "era5_surface_extract_")
        catch
            continue
        end
    end
    return mktempdir(; prefix = "era5_surface_extract_")
end

function _surface_nc_paths(path::String)
    _is_zip_payload(path) || return String[path], nothing

    entries = strip.(readlines(`unzip -Z1 $path`))
    nc_entries = [entry for entry in entries
                  if !isempty(entry) &&
                     !endswith(entry, "/") &&
                     endswith(lowercase(entry), ".nc")]
    isempty(nc_entries) && throw(ArgumentError(
        "ERA5 surface ZIP $(path) contains no NetCDF entries"))

    tmp_dir = _surface_extract_dir(path)
    nc_paths = String[]
    try
        for (idx, entry) in enumerate(nc_entries)
            out_path = joinpath(tmp_dir, @sprintf("%02d_%s", idx, basename(entry)))
            open(out_path, "w") do out
                run(pipeline(`unzip -p $path $entry`; stdout = out))
            end
            push!(nc_paths, out_path)
        end
    catch
        rm(tmp_dir; recursive = true, force = true)
        rethrow()
    end
    return nc_paths, tmp_dir
end

function open_era5_surface_reader(surface_dir::AbstractString, date::Date,
                                  Nx::Int, Ny::Int)
    path = _resolve_era5_surface_path(surface_dir, date)
    nc_paths, tmp_dir = _surface_nc_paths(path)
    datasets = NCDataset[]
    try
        for nc_path in nc_paths
            push!(datasets, NCDataset(nc_path, "r"))
        end
    catch
        for ds in datasets
            close(ds)
        end
        tmp_dir === nothing || rm(tmp_dir; recursive = true, force = true)
        rethrow()
    end
    time_len = isempty(datasets) ? 0 : maximum(_surface_time_len, datasets)
    return ERA5SurfaceReader(datasets, path, tmp_dir, date, Nx, Ny, time_len)
end

function close_era5_surface_reader(reader::ERA5SurfaceReader)
    for ds in reader.datasets
        close(ds)
    end
    reader.tmp_dir === nothing || rm(reader.tmp_dir; recursive = true, force = true)
    return nothing
end

function _surface_time_len(ds::NCDataset)
    for dim_name in ERA5_SURFACE_TIME_DIMS
        if haskey(ds.dim, dim_name)
            return Int(ds.dim[dim_name])
        elseif haskey(ds, dim_name)
            return length(ds[dim_name])
        end
    end
    return 1
end

function _surface_time_index(reader::ERA5SurfaceReader, ds::NCDataset, win_idx::Int)
    time_len = _surface_time_len(ds)
    time_len <= 1 && return 1
    if time_len == 24
        return win_idx
    end
    idx = (day(reader.date) - 1) * 24 + win_idx
    idx <= time_len || throw(ArgumentError(
        "ERA5 surface file $(reader.path) has $(time_len) time slots; " *
        "need index $(idx) for $(reader.date) window $(win_idx)."))
    return idx
end

function _surface_var_name(ds::NCDataset, key::Symbol)
    for candidate in ERA5_SURFACE_VAR_CANDIDATES[key]
        for name in keys(ds)
            lowercase(String(name)) == lowercase(candidate) && return String(name)
        end
    end
    return nothing
end

function _surface_var_binding(reader::ERA5SurfaceReader, key::Symbol; required::Bool = true)
    for ds in reader.datasets
        name = _surface_var_name(ds, key)
        name === nothing || return ds, name
    end
    required || return nothing
    throw(ArgumentError(
        "ERA5 surface file $(reader.path) is missing $(key); tried " *
        join(ERA5_SURFACE_VAR_CANDIDATES[key], ", ")))
end

function _read_surface_slice(ds::NCDataset, var_name::String, time_idx::Int)
    v = ds[var_name]
    dims = dimnames(v)
    lon_dim = findfirst(==("longitude"), dims)
    lat_dim = findfirst(==("latitude"), dims)
    time_dim = findfirst(d -> d in ERA5_SURFACE_TIME_DIMS, dims)
    lon_dim === nothing && throw(ArgumentError("surface variable $(var_name) lacks longitude dimension"))
    lat_dim === nothing && throw(ArgumentError("surface variable $(var_name) lacks latitude dimension"))

    idx = ntuple(d -> d == lon_dim || d == lat_dim ? Colon() :
                      d == time_dim ? time_idx : 1,
                 length(dims))
    raw = Array(v[idx...])
    kept = [dims[d] for d in eachindex(dims) if d == lon_dim || d == lat_dim]
    field = kept == ["longitude", "latitude"] ? raw :
            kept == ["latitude", "longitude"] ? permutedims(raw, (2, 1)) :
            throw(ArgumentError("unsupported surface variable dimension order for $(var_name): $(dims)"))
    return field
end

function _normalize_surface_ll!(field::AbstractMatrix{FT},
                                ds::NCDataset) where FT
    out = field
    if haskey(ds, "latitude")
        lats = Float64.(ds["latitude"][:])
        if length(lats) == size(out, 2) && lats[1] > lats[end]
            out = out[:, end:-1:1]
        end
    end
    if haskey(ds, "longitude")
        tmp3 = reshape(out, size(out, 1), size(out, 2), 1)
        out = @view _normalize_lon_to_centered(tmp3, Float64.(ds["longitude"][:]))[:, :, 1]
    end
    return Array{FT}(out)
end

function _surface_hflux_to_upward_wm2(raw, ds::NCDataset, var_name::String, ::Type{FT}) where FT
    units = lowercase(String(get(ds[var_name].attrib, "units", "")))
    lname = lowercase(var_name)
    if occursin("j", units) || lname in ("sshf", "surface_sensible_heat_flux")
        return FT.(-raw ./ 3600)
    end
    return FT.(raw)
end

function _load_surface_field(reader::ERA5SurfaceReader,
                             key::Symbol,
                             win_idx::Int,
                             ::Type{FT}) where FT
    ds, var_name = _surface_var_binding(reader, key)
    time_idx = _surface_time_index(reader, ds, win_idx)
    field = _normalize_surface_ll!(FT.(_read_surface_slice(ds, var_name, time_idx)), ds)
    return field, ds, var_name
end

function _load_surface_raw_field(reader::ERA5SurfaceReader,
                                 key::Symbol,
                                 win_idx::Int,
                                 ::Type{FT}) where FT
    ds, var_name = _surface_var_binding(reader, key)
    time_idx = _surface_time_index(reader, ds, win_idx)
    field = _normalize_surface_ll!(FT.(_read_surface_slice(ds, var_name, time_idx)), ds)
    return field, ds, var_name
end

function _load_surface_ustar(reader::ERA5SurfaceReader,
                             win_idx::Int,
                             ::Type{FT}) where FT
    direct = _surface_var_binding(reader, :ustar; required = false)
    if direct !== nothing
        ds, var_name = direct
        time_idx = _surface_time_index(reader, ds, win_idx)
        return _normalize_surface_ll!(FT.(_read_surface_slice(ds, var_name, time_idx)), ds)
    end

    u10, _, _ = _load_surface_field(reader, :u10, win_idx, FT)
    v10, _, _ = _load_surface_field(reader, :v10, win_idx, FT)
    size(u10) == size(v10) || throw(DimensionMismatch(
        "ERA5 surface u10 has size $(size(u10)); v10 has size $(size(v10))"))

    ustar = similar(u10, FT)
    drag = FT(sqrt(ERA5_USTAR_DRAG_COEFFICIENT))
    @. ustar = drag * hypot(u10, v10)
    return ustar
end

function _validate_era5_surface!(surface, path::String, win_idx::Int)
    all(isfinite, surface.pblh) || error("ERA5 surface PBLH contains non-finite values in $(path) window $(win_idx)")
    all(isfinite, surface.ustar) || error("ERA5 surface USTAR contains non-finite values in $(path) window $(win_idx)")
    all(isfinite, surface.hflux) || error("ERA5 surface HFLUX contains non-finite values in $(path) window $(win_idx)")
    all(isfinite, surface.t2m) || error("ERA5 surface T2M contains non-finite values in $(path) window $(win_idx)")
    minimum(surface.pblh) > 0 || error("ERA5 surface PBLH must be positive in $(path) window $(win_idx)")
    minimum(surface.ustar) >= 0 || error("ERA5 surface USTAR must be nonnegative in $(path) window $(win_idx)")
    150 < minimum(surface.t2m) < 350 || error("ERA5 surface T2M minimum is out of range in $(path) window $(win_idx)")
    150 < maximum(surface.t2m) < 350 || error("ERA5 surface T2M maximum is out of range in $(path) window $(win_idx)")
    maximum(abs, surface.hflux) < 5000 || error("ERA5 surface HFLUX magnitude is out of range in $(path) window $(win_idx)")
    return nothing
end

function _validate_surface_shapes!(surface, expected::Tuple{Int, Int})
    for (name, field) in pairs(surface)
        size(field) == expected ||
            throw(DimensionMismatch("ERA5 surface $(name) has size $(size(field)), expected $(expected)"))
    end
    return nothing
end

function load_era5_surface_window(reader::ERA5SurfaceReader,
                                  win_idx::Int,
                                  ::Type{FT}) where FT
    pblh, _, _ = _load_surface_field(reader, :pblh, win_idx, FT)
    ustar = _load_surface_ustar(reader, win_idx, FT)
    hflux_raw, hflux_ds, hflux_name = _load_surface_raw_field(reader, :hflux, win_idx, FT)
    hflux = _surface_hflux_to_upward_wm2(hflux_raw, hflux_ds, hflux_name, FT)
    t2m, _, _ = _load_surface_field(reader, :t2m, win_idx, FT)

    surface = (pblh = pblh, t2m = t2m, ustar = ustar, hflux = hflux)
    _validate_surface_shapes!(surface, (reader.Nx, reader.Ny))
    _validate_era5_surface!(surface, reader.path, win_idx)
    return surface
end
