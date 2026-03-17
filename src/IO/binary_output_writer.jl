# ---------------------------------------------------------------------------
# Binary output writer — fast sequential writes with persistent file handle
#
# Writes raw Float32 data to a flat binary file during simulation.
# Follows the same JSON-header + contiguous-data pattern as CSBinaryReader.
# Post-simulation conversion to NetCDF via convert_binary_to_netcdf().
#
# Supports daily file splitting: when split=:daily, the writer rolls over
# to a new file at each day boundary. Filenames are date-stamped:
#   {stem}_{YYYYMMDD}.bin
# ---------------------------------------------------------------------------

using JSON3
using Dates
using NCDatasets
using ..Grids: grid_size, floattype, LatitudeLongitudeGrid, CubedSphereGrid, znode, Center
using ..Fields: interior, AbstractField
using ..Architectures: array_type
using ..Diagnostics: AbstractDiagnostic, ColumnMeanDiagnostic, ColumnMassDiagnostic,
                     SurfaceSliceDiagnostic, SigmaLevelDiagnostic,
                     RegridDiagnostic, Full3DDiagnostic, MetField2DDiagnostic,
                     EmissionFluxDiagnostic, ColumnFluxDiagnostic,
                     column_mean!, surface_slice!, regrid_cs_to_latlon,
                     RegridMapping, build_regrid_mapping, regrid_cs_to_latlon!

const BINARY_OUTPUT_HEADER_SIZE = 16384

# =====================================================================
# BinaryOutputWriter struct
# =====================================================================

"""
$(TYPEDEF)

Write diagnostic fields to a flat binary file for fast sequential I/O.

File layout: `[8192-byte JSON header | timestep₁ | timestep₂ | ...]`

Each timestep: `Float64 time_seconds | Float32 field₁ | Float32 field₂ | ...`

The file handle stays open for the duration of the simulation (or until the
next daily rollover when `split=:daily`).

Call `finalize_output!` at the end of a run to update the header with the
actual number of timesteps and optionally convert to NetCDF.

$(FIELDS)
"""
struct BinaryOutputWriter{S <: AbstractOutputSchedule, OG} <: AbstractOutputWriter
    "output file path (.bin) or stem for daily splitting"
    filepath      :: String
    "name → field, function, or AbstractDiagnostic"
    fields        :: Dict{Symbol, Any}
    "output schedule"
    schedule      :: S
    "optional output grid for regridding (nothing = native grid)"
    output_grid   :: OG
    "number of writes so far (for current file when split=:daily)"
    _write_count  :: Ref{Int}
    "lazily-built RegridMapping for GPU CS→lat-lon regridding"
    _regrid_cache :: Ref{Any}
    "persistent file handle (nothing until first write)"
    _io           :: Ref{Union{Nothing, IOStream}}
    "metadata dict written as JSON header"
    _header       :: Dict{String, Any}
    "if true, convert to NetCDF after finalize"
    auto_convert  :: Bool
    "simulation start date for CF-convention time units"
    start_date    :: Date
    "file splitting mode: :none or :daily"
    split         :: Symbol
    "current date of open file (for daily splitting)"
    _current_date :: Ref{Date}
    "path of currently open file (may differ from filepath for daily split)"
    _current_path :: Ref{String}
    "writes to current file (reset on daily rollover, used for per-file Nt)"
    _file_write_count :: Ref{Int}
end

function BinaryOutputWriter(filepath::String, fields::Dict, schedule::S;
                             output_grid=nothing,
                             auto_convert::Bool=false,
                             start_date::Date=Date(2000,1,1),
                             split::Symbol=:none) where S <: AbstractOutputSchedule
    return BinaryOutputWriter{S, typeof(output_grid)}(
        filepath, fields, schedule, output_grid,
        Ref(0), Ref{Any}(nothing),
        Ref{Union{Nothing, IOStream}}(nothing),
        Dict{String, Any}(),
        auto_convert, start_date, split,
        Ref(Date(1, 1, 1)),
        Ref(""),
        Ref(0))
end

# =====================================================================
# Daily file naming
# =====================================================================

"""Generate date-stamped filepath for daily splitting."""
function _daily_filepath(writer::BinaryOutputWriter, date::Date)
    stem = writer.filepath
    # Strip .bin extension if present
    if endswith(stem, ".bin")
        stem = stem[1:end-4]
    end
    return stem * "_" * Dates.format(date, "yyyymmdd") * ".bin"
end

# =====================================================================
# Initialize — populate header metadata
# =====================================================================

function initialize_output!(writer::BinaryOutputWriter, model)
    grid = model.grid
    hdr = writer._header
    hdr["format"]      = "atmos_transport_output"
    hdr["version"]     = 2
    hdr["FT"]          = "Float32"
    hdr["Nt"]          = 0
    hdr["header_size"] = BINARY_OUTPUT_HEADER_SIZE
    hdr["start_date"]  = string(writer.start_date)

    # Run provenance metadata
    if hasproperty(model, :metadata) && !isempty(model.metadata)
        meta = model.metadata
        hdr["user"]          = get(meta, "user", "unknown")
        hdr["hostname"]      = get(meta, "hostname", "unknown")
        hdr["julia_version"] = get(meta, "julia_version", string(VERSION))
        hdr["run_started"]   = get(meta, "created", string(Dates.now()))
        if haskey(meta, "config")
            hdr["config"] = meta["config"]
        end
    end

    # Field names in deterministic sorted order
    field_names = sort(collect(keys(writer.fields)))
    hdr["fields"] = [string(f) for f in field_names]
    hdr["field_dims"] = Dict{String, Any}()
    hdr["field_units"] = Dict{String, String}()
    hdr["field_long_names"] = Dict{String, String}()

    if grid isa LatitudeLongitudeGrid
        gs  = grid_size(grid)
        FT  = floattype(grid)
        hdr["grid_type"] = "latlon"
        hdr["Nx"] = gs.Nx
        hdr["Ny"] = gs.Ny
        hdr["Nz"] = gs.Nz
        hdr["lons"] = Float32.(grid.λᶜ_cpu)
        hdr["lats"] = Float32.(grid.φᶜ_cpu)
        hdr["levs"] = Float32[znode(k, grid, Center()) for k in 1:gs.Nz]
        for fname in field_names
            fe = writer.fields[fname]
            dims = _output_dims(fe, grid, writer.output_grid)
            hdr["field_dims"][string(fname)] = collect(dims)
            attribs = _field_attribs(fe)
            hdr["field_units"][string(fname)] = get(attribs, "units", "")
            hdr["field_long_names"][string(fname)] = get(attribs, "long_name", "")
        end

    elseif grid isa CubedSphereGrid
        FT = floattype(grid)
        hdr["grid_type"] = "cubed_sphere"
        hdr["Nc"] = grid.Nc
        hdr["Nz"] = grid.Nz
        # Store hybrid sigma-pressure A/B coefficients for lev coordinate
        vc = grid.vertical
        if hasproperty(vc, :A) && hasproperty(vc, :B)
            hdr["hyai"] = collect(Float64, vc.A)  # interface A (Nz+1)
            hdr["hybi"] = collect(Float64, vc.B)  # interface B (Nz+1)
            hdr["reference_pressure"] = Float64(grid.reference_pressure)
        end
        # Store coordinate file path for NetCDF converter
        if hasproperty(model.met_data, :coord_file) && !isempty(model.met_data.coord_file)
            hdr["coordinate_file"] = model.met_data.coord_file
        elseif hasproperty(model.met_data, :mode) && model.met_data.mode == :netcdf &&
               hasproperty(model.met_data, :files) && !isempty(model.met_data.files)
            hdr["coordinate_file"] = model.met_data.files[1]
        end
        if writer.output_grid isa LatLonOutputGrid
            og = writer.output_grid
            hdr["output_grid"] = "latlon"
            hdr["Nlon"] = og.Nlon
            hdr["Nlat"] = og.Nlat
            hdr["lon0"] = og.lon0
            hdr["lon1"] = og.lon1
            hdr["lat0"] = og.lat0
            hdr["lat1"] = og.lat1
        else
            hdr["output_grid"] = "native"
        end
        for fname in field_names
            fe = writer.fields[fname]
            dims = _output_dims(fe, grid, writer.output_grid)
            hdr["field_dims"][string(fname)] = collect(dims)
            attribs = _field_attribs(fe)
            hdr["field_units"][string(fname)] = get(attribs, "units", "")
            hdr["field_long_names"][string(fname)] = get(attribs, "long_name", "")
        end
    end

    # Compute elements per timestep (Float32 count, excluding the Float64 time stamp)
    elems = 0
    for fname in field_names
        dims = hdr["field_dims"][string(fname)]
        n = 1
        for d in dims
            d == "time" && continue
            if d == "lon";  n *= get(hdr, "Nlon", get(hdr, "Nx", 0)); end
            if d == "lat";  n *= get(hdr, "Nlat", get(hdr, "Ny", 0)); end
            if d == "lev";  n *= get(hdr, "Nz", 0); end
            if d == "x";    n *= get(hdr, "Nc", 0); end
            if d == "y";    n *= get(hdr, "Nc", 0); end
            if d == "panel"; n *= 6; end
        end
        elems += n
    end
    hdr["elems_per_timestep"] = elems

    # Store emission source metadata (propagates to NC global attributes)
    if hasproperty(model, :sources) && !isempty(model.sources)
        emis_meta = Dict{String, Any}()
        for s in model.sources
            # Flatten CombinedFlux components
            components = hasproperty(s, :components) ? collect(s.components) : [s]
            for c in components
                sp = hasproperty(c, :species) ? string(c.species) : "unknown"
                label = hasproperty(c, :label) ? c.label : ""
                nt = if hasproperty(c, :flux_data) && c.flux_data isa AbstractArray && ndims(c.flux_data) == 3
                    size(c.flux_data, 3)
                elseif hasproperty(c, :flux_data) && c.flux_data isa AbstractVector
                    length(c.flux_data)
                elseif hasproperty(c, :time_hours)
                    length(c.time_hours)
                else
                    1
                end
                idx = hasproperty(c, :current_idx) ? c.current_idx : 1
                key = sp * (haskey(emis_meta, sp) ? "_$(label)" : "")
                emis_meta[key] = Dict("label" => label, "species" => sp,
                                      "n_snapshots" => nt, "initial_idx" => idx)
            end
        end
        hdr["emission_sources"] = emis_meta
    end

    return nothing
end

# =====================================================================
# Open binary file and write header
# =====================================================================

function _open_binary_file!(writer::BinaryOutputWriter, filepath::String)
    mkpath(dirname(filepath))
    io = open(filepath, "w")
    json_str   = JSON3.write(writer._header)
    json_bytes = Vector{UInt8}(json_str)
    if length(json_bytes) >= BINARY_OUTPUT_HEADER_SIZE - 1
        error("Binary output header too large ($(length(json_bytes)) bytes). " *
              "Reduce field count or coordinate precision.")
    end
    write(io, json_bytes)
    write(io, UInt8(0))  # null terminator
    padding = BINARY_OUTPUT_HEADER_SIZE - length(json_bytes) - 1
    write(io, zeros(UInt8, padding))
    writer._io[] = io
    writer._current_path[] = filepath
    return nothing
end

# =====================================================================
# _finalize_current_file! — close current file, update header
# =====================================================================

function _finalize_current_file!(writer::BinaryOutputWriter)
    io = writer._io[]
    io === nothing && return nothing

    flush(io)
    writer._header["Nt"] = writer._file_write_count[]
    writer._header["run_finished"] = string(Dates.now())

    # Rewrite header in-place with updated Nt
    json_str   = JSON3.write(writer._header)
    json_bytes = Vector{UInt8}(json_str)
    seek(io, 0)
    write(io, json_bytes)
    write(io, UInt8(0))
    padding = BINARY_OUTPUT_HEADER_SIZE - length(json_bytes) - 1
    write(io, zeros(UInt8, padding))

    close(io)
    writer._io[] = nothing

    path = writer._current_path[]
    @info "Binary output finalized: $(writer._file_write_count[]) timesteps → $path"
    return path
end

# =====================================================================
# write_output! — fast binary path with daily rollover
# =====================================================================

function write_output!(writer::BinaryOutputWriter, model, time;
                       air_mass=nothing, tracers=nothing, met_fields=nothing)
    model_time = time isa Number ? Float64(time) : Float64(Dates.value(time) / 1000.0)
    iteration  = hasproperty(model, :clock) ? model.clock.iteration : 0

    if !should_write(writer, model_time, iteration)
        return nothing
    end

    # Lazy init: populate header on first write
    if isempty(writer._header)
        initialize_output!(writer, model)
    end

    # Determine target date and filepath
    current_date = writer.start_date + Day(floor(Int, model_time / 86400))

    if writer.split === :daily
        if writer._io[] === nothing
            # First write: open file for this date
            writer._current_date[] = current_date
            writer._header["file_date"] = Dates.format(current_date, "yyyy-mm-dd")
            _open_binary_file!(writer, _daily_filepath(writer, current_date))
        elseif current_date != writer._current_date[]
            # Date changed: finalize current file and open new one
            prev_path = _finalize_current_file!(writer)
            # Auto-convert each daily file in background thread
            if writer.auto_convert && prev_path !== nothing
                let p = prev_path
                    Threads.@spawn begin
                        nc_path = replace(p, ".bin" => ".nc")
                        convert_binary_to_netcdf(p; nc_path)
                    end
                end
            end
            # NOTE: do NOT reset _write_count — it must stay in sync with
            # absolute sim_time for should_write threshold to work correctly.
            writer._file_write_count[] = 0
            writer._current_date[] = current_date
            writer._header["Nt"] = 0
            writer._header["file_date"] = Dates.format(current_date, "yyyy-mm-dd")
            _open_binary_file!(writer, _daily_filepath(writer, current_date))
        end
    else
        # No splitting: single file
        if writer._io[] === nothing
            _open_binary_file!(writer, writer.filepath)
        end
    end

    io   = writer._io[]
    grid = model.grid

    # Write time stamp (Float64)
    write(io, model_time)

    # Write fields in sorted order (matches header["fields"])
    field_names = sort(collect(keys(writer.fields)))
    for fname in field_names
        arr = _extract_field_data(writer.fields[fname], model; air_mass, tracers,
                                  regrid_cache=writer._regrid_cache,
                                  output_grid=writer.output_grid, met_fields)
        _write_binary_field!(io, arr, grid, writer.output_grid)
    end

    writer._write_count[] += 1
    writer._file_write_count[] += 1
    return nothing
end

# =====================================================================
# _write_binary_field! — dispatch on grid type
# =====================================================================

function _write_binary_field!(io::IOStream, arr::AbstractArray,
                               grid::LatitudeLongitudeGrid, output_grid)
    arr_cpu = arr isa Array ? arr : Array(arr)
    write(io, Float32.(arr_cpu))
end

function _write_binary_field!(io::IOStream, arr, grid::CubedSphereGrid,
                               output_grid::LatLonOutputGrid)
    if arr isa NTuple
        # Fallback: regrid panels without file coordinates (gnomonic coords only)
        @warn "CS→lat-lon regridding without file coordinates in binary writer. " *
              "Output coordinates may be incorrect." maxlog=1
        FT = floattype(grid)
        panels_cpu = ntuple(p -> arr[p] isa Array ? arr[p] : Array(arr[p]), 6)
        out_ll, _, _ = regrid_cs_to_latlon(panels_cpu, grid;
            Nlon=output_grid.Nlon, Nlat=output_grid.Nlat,
            lon0=FT(output_grid.lon0), lon1=FT(output_grid.lon1),
            lat0=FT(output_grid.lat0), lat1=FT(output_grid.lat1))
        write(io, Float32.(out_ll))
    else
        write(io, Float32.(arr))
    end
end

function _write_binary_field!(io::IOStream, arr, grid::CubedSphereGrid, output_grid)
    # Native CS output — panels concatenated
    if arr isa NTuple
        for p in 1:6
            panel_data = arr[p] isa Array ? arr[p] : Array(arr[p])
            write(io, Float32.(panel_data))
        end
    else
        write(io, Float32.(arr))
    end
end

# =====================================================================
# finalize_output! — close file, update header, optionally convert
# =====================================================================

"""
$(SIGNATURES)

Close the binary output file and update the header with the actual number
of timesteps written. If `auto_convert=true`, converts to NetCDF.
"""
function finalize_output!(writer::BinaryOutputWriter)
    path = _finalize_current_file!(writer)
    path === nothing && return nothing

    if writer.auto_convert
        nc_path = replace(path, ".bin" => ".nc")
        convert_binary_to_netcdf(path; nc_path)
    end
    return nothing
end

# =====================================================================
# convert_binary_to_netcdf — standalone post-processing utility
# =====================================================================

"""
    convert_binary_to_netcdf(bin_path; nc_path=nothing, deflate_level=0) → nc_path

Read a binary output file and convert it to CF-compliant NetCDF.
Streams timesteps one at a time for memory efficiency.

Supports both lat-lon and native cubed-sphere grids.

Can be called standalone from the Julia REPL:
```julia
using AtmosTransport
convert_binary_to_netcdf("output.bin"; deflate_level=4)
```
"""
function convert_binary_to_netcdf(bin_path::String;
                                   nc_path::Union{String, Nothing}=nothing,
                                   deflate_level::Int=0)
    nc_path = nc_path === nothing ? replace(bin_path, ".bin" => ".nc") : nc_path

    io = open(bin_path, "r")
    hdr_bytes = read(io, BINARY_OUTPUT_HEADER_SIZE)
    json_end  = something(findfirst(==(0x00), hdr_bytes), BINARY_OUTPUT_HEADER_SIZE + 1) - 1
    hdr = JSON3.read(String(hdr_bytes[1:json_end]))

    Nt = Int(hdr.Nt)
    field_names = [Symbol(f) for f in hdr.fields]
    field_dims  = hdr.field_dims

    grid_type   = get(hdr, :grid_type, "latlon")
    output_grid = get(hdr, :output_grid, "")
    Nz = haskey(hdr, :Nz) ? Int(hdr.Nz) : 0
    _sd = haskey(hdr, :start_date) ? string(hdr.start_date) : "2000-01-01"

    # Dispatch on grid type
    if grid_type == "cubed_sphere" && output_grid == "native"
        _convert_cs_native(io, nc_path, hdr, Nt, field_names, field_dims, Nz, _sd;
                           deflate_level)
    else
        _convert_latlon(io, nc_path, hdr, Nt, field_names, field_dims, Nz, _sd;
                        deflate_level)
    end

    close(io)
    @info "Converted $bin_path → $nc_path ($Nt timesteps)"
    return nc_path
end

# =====================================================================
# Gnomonic cubed-sphere coordinate computation (for NetCDF conversion)
# =====================================================================

"""Gnomonic projection: tangent-plane (ξ, η) → Cartesian (x, y, z) on unit sphere."""
@inline function _gnomonic_xyz_conv(ξ, η, panel::Int)
    d = 1 / sqrt(1 + ξ^2 + η^2)
    if     panel == 1;  return ( d,  ξ*d,  η*d)
    elseif panel == 2;  return (-ξ*d,  d,  η*d)
    elseif panel == 3;  return (-d, -ξ*d,  η*d)
    elseif panel == 4;  return ( ξ*d, -d,  η*d)
    elseif panel == 5;  return (-η*d,  ξ*d,  d)
    else;               return ( η*d,  ξ*d, -d)
    end
end

"""Compute standard gnomonic cubed-sphere coordinates for Nc cells per edge.

Returns `(lons, lats, corner_lons, corner_lats)` where:
- `lons, lats`: `(Nc, Nc, 6)` cell-center coordinates [degrees, 0–360]
- `corner_lons, corner_lats`: `(Nc+1, Nc+1, 6)` cell-vertex coordinates [degrees]
"""
function _compute_cs_coordinates(Nc::Int)
    dα = π / (2 * Nc)
    α_faces   = [-π/4 + (i-1)*dα for i in 1:(Nc+1)]
    α_centers = [(α_faces[i] + α_faces[i+1])/2 for i in 1:Nc]

    lons = zeros(Float64, Nc, Nc, 6)
    lats = zeros(Float64, Nc, Nc, 6)
    corner_lons = zeros(Float64, Nc+1, Nc+1, 6)
    corner_lats = zeros(Float64, Nc+1, Nc+1, 6)

    for p in 1:6
        for j in 1:Nc, i in 1:Nc
            ξ = tan(α_centers[i])
            η = tan(α_centers[j])
            x, y, z = _gnomonic_xyz_conv(ξ, η, p)
            lon = atand(y, x)
            lat = asind(z / sqrt(x^2 + y^2 + z^2))
            lons[i, j, p] = mod(lon, 360.0)
            lats[i, j, p] = lat
        end
        for j in 1:(Nc+1), i in 1:(Nc+1)
            ξ = tan(α_faces[i])
            η = tan(α_faces[j])
            x, y, z = _gnomonic_xyz_conv(ξ, η, p)
            lon = atand(y, x)
            lat = asind(z / sqrt(x^2 + y^2 + z^2))
            corner_lons[i, j, p] = mod(lon, 360.0)
            corner_lats[i, j, p] = lat
        end
    end

    return lons, lats, corner_lons, corner_lats
end

"""Convert lat-lon or CS-regridded-to-latlon binary to NetCDF."""
function _convert_latlon(io, nc_path, hdr, Nt, field_names, field_dims, Nz, start_date;
                          deflate_level=0)
    Nlon = haskey(hdr, :Nlon) ? Int(hdr.Nlon) : (haskey(hdr, :Nx) ? Int(hdr.Nx) : 0)
    Nlat = haskey(hdr, :Nlat) ? Int(hdr.Nlat) : (haskey(hdr, :Ny) ? Int(hdr.Ny) : 0)

    lons = if haskey(hdr, :lons)
        Float32.(collect(hdr.lons))
    else
        lon0 = Float32(get(hdr, :lon0, -180))
        lon1 = Float32(get(hdr, :lon1, 180))
        dlon = (lon1 - lon0) / Nlon
        collect(Float32, range(lon0 + dlon/2, lon1 - dlon/2, length=Nlon))
    end
    lats = if haskey(hdr, :lats)
        Float32.(collect(hdr.lats))
    else
        lat0 = Float32(get(hdr, :lat0, -90))
        lat1 = Float32(get(hdr, :lat1, 90))
        dlat = (lat1 - lat0) / Nlat
        collect(Float32, range(lat0 + dlat/2, lat1 - dlat/2, length=Nlat))
    end
    Nlon = max(Nlon, length(lons))
    Nlat = max(Nlat, length(lats))

    NCDataset(nc_path, "c") do ds
        defDim(ds, "lon", Nlon)
        defDim(ds, "lat", Nlat)
        if Nz > 0
            defDim(ds, "lev", Nz)
            if haskey(hdr, :levs)
                defVar(ds, "lev", Float32, ("lev",);
                       attrib=Dict("units" => "Pa"))[:] = Float32.(collect(hdr.levs))
            end
        end
        defDim(ds, "time", 0)

        ds.attrib["source"] = "AtmosTransport.jl"
        ds.attrib["Conventions"] = "CF-1.8"

        # Provenance metadata from binary header
        for pkey in (:user, :hostname, :julia_version, :run_started, :run_finished)
            haskey(hdr, pkey) && (ds.attrib[string(pkey)] = String(hdr[pkey]))
        end

        # Write emission source metadata as global attributes
        if haskey(hdr, :emission_sources)
            for (key, meta) in pairs(hdr.emission_sources)
                prefix = "emission_$(key)"
                ds.attrib["$(prefix)_label"] = String(get(meta, :label, ""))
                ds.attrib["$(prefix)_n_snapshots"] = Int(get(meta, :n_snapshots, 0))
                ds.attrib["$(prefix)_initial_idx"] = Int(get(meta, :initial_idx, 0))
            end
        end

        defVar(ds, "lon", Float32, ("lon",);
               attrib=Dict("units" => "degrees_east"))[:] = lons
        defVar(ds, "lat", Float32, ("lat",);
               attrib=Dict("units" => "degrees_north"))[:] = lats
        defVar(ds, "time", Float64, ("time",);
               attrib=Dict("units" => "seconds since $(start_date) 00:00:00"))

        for fname in field_names
            dims = Tuple(String.(field_dims[string(fname)]))
            attribs = Dict{String,String}()
            if haskey(hdr, :field_units)
                attribs["units"] = get(hdr.field_units, string(fname), "")
                attribs["long_name"] = get(hdr.field_long_names, string(fname), "")
            end
            defVar(ds, string(fname), Float32, dims;
                   deflatelevel=deflate_level,
                   attrib=attribs)
        end

        seek(io, BINARY_OUTPUT_HEADER_SIZE)
        for t in 1:Nt
            time_val = read(io, Float64)
            ds["time"][t] = time_val
            for fname in field_names
                dims  = field_dims[string(fname)]
                shape = _dims_to_shape(dims; Nlon, Nlat, Nz)
                n_elems = prod(shape)
                data = Vector{Float32}(undef, n_elems)
                read!(io, data)
                arr = reshape(data, shape...)
                if length(shape) == 2
                    ds[string(fname)][:, :, t] = arr
                elseif length(shape) == 3
                    ds[string(fname)][:, :, :, t] = arr
                end
            end
        end
    end
end

"""Convert native cubed-sphere binary to GEOS-Chem-compatible NetCDF."""
function _convert_cs_native(io, nc_path, hdr, Nt, field_names, field_dims, Nz, start_date;
                             deflate_level=0)
    Nc = Int(hdr.Nc)

    # Load GMAO coordinates from the reference file stored in the header,
    # falling back to gnomonic computation if unavailable
    cs_lons, cs_lats, cs_clons, cs_clats = if haskey(hdr, :coordinate_file) &&
                                               isfile(String(hdr.coordinate_file))
        coord_file = String(hdr.coordinate_file)
        @info "  Loading GMAO coordinates from $(basename(coord_file))"
        read_geosfp_cs_grid_info(coord_file)
    else
        @warn "  No coordinate file in header — using gnomonic coordinates (may be wrong)"
        _compute_cs_coordinates(Nc)
    end

    # If GMAO file didn't have corner coords, compute gnomonic corners as fallback
    if cs_clons === nothing || cs_clats === nothing
        _, _, cs_clons, cs_clats = _compute_cs_coordinates(Nc)
    end

    NCDataset(nc_path, "c") do ds
        # Global attributes (GEOS-Chem / CF compatible)
        ds.attrib["Conventions"] = "COARDS"
        ds.attrib["grid_mapping_name"] = "gnomonic cubed-sphere"
        ds.attrib["source"] = "AtmosTransport.jl"

        # Provenance metadata from binary header
        for pkey in (:user, :hostname, :julia_version, :run_started, :run_finished)
            haskey(hdr, pkey) && (ds.attrib[string(pkey)] = String(hdr[pkey]))
        end

        # Write emission source metadata as global attributes
        if haskey(hdr, :emission_sources)
            for (key, meta) in pairs(hdr.emission_sources)
                prefix = "emission_$(key)"
                ds.attrib["$(prefix)_label"] = String(get(meta, :label, ""))
                ds.attrib["$(prefix)_n_snapshots"] = Int(get(meta, :n_snapshots, 0))
                ds.attrib["$(prefix)_initial_idx"] = Int(get(meta, :initial_idx, 0))
            end
        end

        defDim(ds, "Xdim", Nc)
        defDim(ds, "Ydim", Nc)
        defDim(ds, "nf", 6)
        defDim(ds, "XCdim", Nc + 1)
        defDim(ds, "YCdim", Nc + 1)
        if Nz > 0
            defDim(ds, "lev", Nz)
            defDim(ds, "ilev", Nz + 1)
        end
        defDim(ds, "time", 0)

        # Fake coordinate variables for GrADS compatibility (matches GEOS-Chem)
        defVar(ds, "Xdim", Float32, ("Xdim",);
               attrib=Dict("units" => "degrees_east",
                           "long_name" => "Fake Longitude for GrADS Compatibility")
               )[:] = collect(Float32, 1:Nc)
        defVar(ds, "Ydim", Float32, ("Ydim",);
               attrib=Dict("units" => "degrees_north",
                           "long_name" => "Fake Latitude for GrADS Compatibility")
               )[:] = collect(Float32, 1:Nc)
        defVar(ds, "nf", Int32, ("nf",);
               attrib=Dict("long_name" => "cubed-sphere face",
                           "axis" => "e",
                           "grads_dim" => "e"))[:] = Int32.(1:6)
        if Nz > 0
            # Compute eta (sigma) levels from hybrid A/B coefficients
            if haskey(hdr, :hyai) && haskey(hdr, :hybi)
                hyai = Float64.(collect(hdr.hyai))
                hybi = Float64.(collect(hdr.hybi))
                p_ref = haskey(hdr, :reference_pressure) ? Float64(hdr.reference_pressure) : 101325.0
                # Mid-level eta: eta = A/p_ref + B
                eta_mid = Float32[(hyai[k] + hyai[k+1]) / (2 * p_ref) +
                                  (hybi[k] + hybi[k+1]) / 2 for k in 1:Nz]
                eta_iface = Float32[hyai[k] / p_ref + hybi[k] for k in 1:(Nz+1)]
                # Reverse to surface-first: lev[1]=surface (eta≈1), lev[Nz]=TOA (eta≈0)
                reverse!(eta_mid)
                reverse!(eta_iface)
                reverse!(hyai)
                reverse!(hybi)
                defVar(ds, "lev", Float32, ("lev",);
                       attrib=Dict("units" => "1",
                                   "long_name" => "hybrid sigma-pressure level",
                                   "positive" => "up",
                                   "coordinate" => "eta",
                                   "standard_name" => "atmosphere_hybrid_sigma_pressure_coordinate",
                                   "formula_terms" => "ap: hyai b: hybi ps: surface_pressure p0: p_ref"))[:] = eta_mid
                defVar(ds, "ilev", Float32, ("ilev",);
                       attrib=Dict("units" => "1",
                                   "long_name" => "hybrid sigma-pressure level at interfaces",
                                   "positive" => "up"))[:] = eta_iface
                defVar(ds, "hyai", Float64, ("ilev",);
                       attrib=Dict("units" => "Pa",
                                   "long_name" => "hybrid A coefficient at interfaces"))[:] = hyai
                defVar(ds, "hybi", Float64, ("ilev",);
                       attrib=Dict("units" => "1",
                                   "long_name" => "hybrid B coefficient at interfaces"))[:] = hybi
                ds.attrib["reference_pressure_Pa"] = p_ref
            else
                # Fallback: integer level indices (surface-first)
                defVar(ds, "lev", Float32, ("lev",);
                       attrib=Dict("units" => "1",
                                   "long_name" => "model level",
                                   "positive" => "up"))[:] = collect(Float32, Nz:-1:1)
            end
        end
        defVar(ds, "time", Float64, ("time",);
               attrib=Dict("units" => "minutes since $(start_date)T00:00:00",
                           "calendar" => "standard",
                           "long_name" => "time"))

        # Real coordinates matching GEOS-Chem format
        defVar(ds, "lons", Float64, ("Xdim", "Ydim", "nf");
               attrib=Dict("units" => "degrees_east",
                           "long_name" => "longitude"))[:] = cs_lons
        defVar(ds, "lats", Float64, ("Xdim", "Ydim", "nf");
               attrib=Dict("units" => "degrees_north",
                           "long_name" => "latitude"))[:] = cs_lats
        defVar(ds, "corner_lons", Float64, ("XCdim", "YCdim", "nf");
               attrib=Dict("units" => "degrees_east",
                           "long_name" => "longitude of cell corners"))[:] = cs_clons
        defVar(ds, "corner_lats", Float64, ("XCdim", "YCdim", "nf");
               attrib=Dict("units" => "degrees_north",
                           "long_name" => "latitude of cell corners"))[:] = cs_clats

        # Define field variables with CS dimensions + GEOS-Chem-compatible attributes
        for fname in field_names
            raw_dims = field_dims[string(fname)]
            nc_dims = _cs_dims_to_nc_dims(raw_dims)
            attribs = Dict("coordinates" => "lons lats",
                           "grid_mapping" => "cubed_sphere")
            if haskey(hdr, :field_units)
                attribs["units"] = get(hdr.field_units, string(fname), "")
                attribs["long_name"] = get(hdr.field_long_names, string(fname), "")
            end
            defVar(ds, string(fname), Float32, nc_dims;
                   deflatelevel=deflate_level,
                   attrib=attribs)
        end

        # Stream timesteps (binary stores seconds; NC uses minutes)
        seek(io, BINARY_OUTPUT_HEADER_SIZE)
        for t in 1:Nt
            time_seconds = read(io, Float64)
            ds["time"][t] = time_seconds / 60.0  # convert to minutes

            for fname in field_names
                dims  = field_dims[string(fname)]
                shape = _dims_to_shape(dims; Nc, Nz)
                n_elems = prod(shape)
                data = Vector{Float32}(undef, n_elems)
                read!(io, data)
                if length(shape) == 3
                    # CS 2D: binary layout (Nc, Nc, 6) matches directly
                    arr = reshape(data, shape...)
                    ds[string(fname)][:, :, :, t] = arr
                elseif length(shape) == 4
                    # CS 3D: binary stores panels sequentially P1(Nc×Nc×Nz)..P6(Nc×Nc×Nz)
                    # so flat order is (x, y, lev, panel). Reshape accordingly,
                    # then permute to NC dim order (Xdim, Ydim, nf, lev).
                    arr = reshape(data, Nc, Nc, Nz, 6)
                    arr = permutedims(arr, (1, 2, 4, 3))
                    # Reverse lev axis: surface first (matches reversed lev coordinate)
                    arr = reverse(arr; dims=4)
                    ds[string(fname)][:, :, :, :, t] = arr
                end
            end
        end
    end
end

"""Map binary field dim names to NetCDF dimension names for CS native output.
Handles both internal names (x, y, panel) and direct NetCDF names (Xdim, Ydim, nf)."""
function _cs_dims_to_nc_dims(dims)
    nc_dims = String[]
    for d in dims
        d == "time"  && (push!(nc_dims, "time");  continue)
        d == "lev"   && (push!(nc_dims, "lev");   continue)
        (d == "x"     || d == "Xdim")  && (push!(nc_dims, "Xdim"); continue)
        (d == "y"     || d == "Ydim")  && (push!(nc_dims, "Ydim"); continue)
        (d == "panel" || d == "nf")    && (push!(nc_dims, "nf");   continue)
    end
    return Tuple(nc_dims)
end

"""Map dimension names to their sizes (excluding "time").
Handles both internal names (x, y, panel) and NetCDF names (Xdim, Ydim, nf)."""
function _dims_to_shape(dims; Nc=0, Nlon=0, Nlat=0, Nz=0)
    shape = Int[]
    for d in dims
        d == "time"  && continue
        d == "lon"   && (push!(shape, Nlon); continue)
        d == "lat"   && (push!(shape, Nlat); continue)
        d == "lev"   && (push!(shape, Nz);   continue)
        (d == "x"     || d == "Xdim")  && (push!(shape, Nc > 0 ? Nc : Nlon); continue)
        (d == "y"     || d == "Ydim")  && (push!(shape, Nc > 0 ? Nc : Nlat); continue)
        (d == "panel" || d == "nf")    && (push!(shape, 6); continue)
    end
    return tuple(shape...)
end

export BinaryOutputWriter
export convert_binary_to_netcdf
