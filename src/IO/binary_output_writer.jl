# ---------------------------------------------------------------------------
# Binary output writer — fast sequential writes with persistent file handle
#
# Writes raw Float32 data to a flat binary file during simulation.
# Follows the same JSON-header + contiguous-data pattern as CSBinaryReader.
# Post-simulation conversion to NetCDF via convert_binary_to_netcdf().
# ---------------------------------------------------------------------------

using JSON3
using Dates
using NCDatasets
using ..Grids: grid_size, floattype, LatitudeLongitudeGrid, CubedSphereGrid, znode, Center
using ..Fields: interior, AbstractField
using ..Architectures: array_type
using ..Diagnostics: AbstractDiagnostic, ColumnMeanDiagnostic, SurfaceSliceDiagnostic,
                     RegridDiagnostic, Full3DDiagnostic, MetField2DDiagnostic,
                     column_mean!, surface_slice!, regrid_cs_to_latlon,
                     RegridMapping, build_regrid_mapping, regrid_cs_to_latlon!

const BINARY_OUTPUT_HEADER_SIZE = 8192

# =====================================================================
# BinaryOutputWriter struct
# =====================================================================

"""
$(TYPEDEF)

Write diagnostic fields to a flat binary file for fast sequential I/O.

File layout: `[8192-byte JSON header | timestep₁ | timestep₂ | ...]`

Each timestep: `Float64 time_seconds | Float32 field₁ | Float32 field₂ | ...`

The file handle stays open for the duration of the simulation.
Call `finalize_output!` at the end of a run to update the header with the
actual number of timesteps and optionally convert to NetCDF.

$(FIELDS)
"""
struct BinaryOutputWriter{S <: AbstractOutputSchedule, OG} <: AbstractOutputWriter
    "output file path (.bin)"
    filepath      :: String
    "name → field, function, or AbstractDiagnostic"
    fields        :: Dict{Symbol, Any}
    "output schedule"
    schedule      :: S
    "optional output grid for regridding (nothing = native grid)"
    output_grid   :: OG
    "number of writes so far"
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
end

function BinaryOutputWriter(filepath::String, fields::Dict, schedule::S;
                             output_grid=nothing,
                             auto_convert::Bool=false,
                             start_date::Date=Date(2000,1,1)) where S <: AbstractOutputSchedule
    return BinaryOutputWriter{S, typeof(output_grid)}(
        filepath, fields, schedule, output_grid,
        Ref(0), Ref{Any}(nothing),
        Ref{Union{Nothing, IOStream}}(nothing),
        Dict{String, Any}(),
        auto_convert, start_date)
end

# =====================================================================
# Initialize — populate header metadata
# =====================================================================

function initialize_output!(writer::BinaryOutputWriter, model)
    grid = model.grid
    hdr = writer._header
    hdr["format"]      = "atmos_transport_output"
    hdr["version"]     = 1
    hdr["FT"]          = "Float32"
    hdr["Nt"]          = 0
    hdr["header_size"] = BINARY_OUTPUT_HEADER_SIZE
    hdr["start_date"]  = string(writer.start_date)

    # Field names in deterministic sorted order
    field_names = sort(collect(keys(writer.fields)))
    hdr["fields"] = [string(f) for f in field_names]
    hdr["field_dims"] = Dict{String, Any}()

    if grid isa LatitudeLongitudeGrid
        gs  = grid_size(grid)
        FT  = floattype(grid)
        hdr["grid_type"] = "latlon"
        hdr["Nx"] = gs.Nx
        hdr["Ny"] = gs.Ny
        hdr["Nz"] = gs.Nz
        hdr["lons"] = collect(Float32, grid.λᶜ)
        hdr["lats"] = collect(Float32, grid.φᶜ)
        hdr["levs"] = Float32[znode(k, grid, Center()) for k in 1:gs.Nz]
        for fname in field_names
            dims = _output_dims(writer.fields[fname], grid, writer.output_grid)
            hdr["field_dims"][string(fname)] = collect(dims)
        end

    elseif grid isa CubedSphereGrid
        FT = floattype(grid)
        hdr["grid_type"] = "cubed_sphere"
        hdr["Nc"] = grid.Nc
        hdr["Nz"] = grid.Nz
        if writer.output_grid isa LatLonOutputGrid
            og = writer.output_grid
            hdr["output_grid"] = "latlon"
            hdr["Nlon"] = og.Nlon
            hdr["Nlat"] = og.Nlat
            hdr["lon0"] = og.lon0
            hdr["lon1"] = og.lon1
            hdr["lat0"] = og.lat0
            hdr["lat1"] = og.lat1
            # Coordinate arrays reconstructed from grid params in convert_binary_to_netcdf
        else
            hdr["output_grid"] = "native"
        end
        for fname in field_names
            dims = _output_dims(writer.fields[fname], grid, writer.output_grid)
            hdr["field_dims"][string(fname)] = collect(dims)
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

    return nothing
end

# =====================================================================
# Open binary file and write header
# =====================================================================

function _open_binary_file!(writer::BinaryOutputWriter)
    io = open(writer.filepath, "w")
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
    return nothing
end

# =====================================================================
# write_output! — fast binary path
# =====================================================================

function write_output!(writer::BinaryOutputWriter, model, time;
                       air_mass=nothing, tracers=nothing)
    model_time = time isa Number ? Float64(time) : Float64(Dates.value(time) / 1000.0)
    iteration  = hasproperty(model, :clock) ? model.clock.iteration : 0

    if !should_write(writer, model_time, iteration)
        return nothing
    end

    # Lazy init: populate header and open file on first write
    if writer._io[] === nothing
        initialize_output!(writer, model)
        _open_binary_file!(writer)
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
                                  output_grid=writer.output_grid)
        _write_binary_field!(io, arr, grid, writer.output_grid)
    end

    writer._write_count[] += 1
    return nothing
end

# =====================================================================
# _write_binary_field! — dispatch on grid type
# =====================================================================

function _write_binary_field!(io::IOStream, arr::AbstractArray,
                               grid::LatitudeLongitudeGrid, output_grid)
    write(io, Float32.(arr))
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
    io = writer._io[]
    io === nothing && return nothing

    # Flush pending writes
    flush(io)
    writer._header["Nt"] = writer._write_count[]

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

    @info "Binary output finalized: $(writer._write_count[]) timesteps → $(writer.filepath)"

    if writer.auto_convert
        nc_path = replace(writer.filepath, ".bin" => ".nc")
        convert_binary_to_netcdf(writer.filepath; nc_path)
    end
    return nothing
end

# =====================================================================
# convert_binary_to_netcdf — standalone post-processing utility
# =====================================================================

"""
    convert_binary_to_netcdf(bin_path; nc_path=nothing) → nc_path

Read a binary output file and convert it to CF-compliant NetCDF.
Streams timesteps one at a time for memory efficiency.

Can be called standalone from the Julia REPL:
```julia
using AtmosTransport
convert_binary_to_netcdf("output.bin")
```
"""
function convert_binary_to_netcdf(bin_path::String;
                                   nc_path::Union{String, Nothing}=nothing)
    nc_path = nc_path === nothing ? replace(bin_path, ".bin" => ".nc") : nc_path

    io = open(bin_path, "r")
    hdr_bytes = read(io, BINARY_OUTPUT_HEADER_SIZE)
    json_end  = something(findfirst(==(0x00), hdr_bytes), BINARY_OUTPUT_HEADER_SIZE + 1) - 1
    hdr = JSON3.read(String(hdr_bytes[1:json_end]))

    Nt = Int(hdr.Nt)
    field_names = [Symbol(f) for f in hdr.fields]
    field_dims  = hdr.field_dims

    Nlon = haskey(hdr, :Nlon) ? Int(hdr.Nlon) : (haskey(hdr, :Nx) ? Int(hdr.Nx) : 0)
    Nlat = haskey(hdr, :Nlat) ? Int(hdr.Nlat) : (haskey(hdr, :Ny) ? Int(hdr.Ny) : 0)
    Nz   = haskey(hdr, :Nz) ? Int(hdr.Nz) : 0

    # Reconstruct coordinate arrays from grid params if not stored in header
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
        defDim(ds, "time", 0)  # unlimited

        defVar(ds, "lon", Float32, ("lon",);
               attrib=Dict("units" => "degrees_east"))[:] = lons
        defVar(ds, "lat", Float32, ("lat",);
               attrib=Dict("units" => "degrees_north"))[:] = lats
        # Use start_date from header if available, else default epoch
        _sd = haskey(hdr, :start_date) ? string(hdr.start_date) : "2000-01-01"
        defVar(ds, "time", Float64, ("time",);
               attrib=Dict("units" => "seconds since $(_sd) 00:00:00"))

        for fname in field_names
            dims = Tuple(String.(field_dims[string(fname)]))
            defVar(ds, string(fname), Float32, dims;
                   attrib=Dict("units" => "ppm"))
        end

        # Stream timesteps from binary
        seek(io, BINARY_OUTPUT_HEADER_SIZE)
        for t in 1:Nt
            time_val = read(io, Float64)
            ds["time"][t] = time_val

            for fname in field_names
                dims  = field_dims[string(fname)]
                shape = _dims_to_shape(dims, Nlon, Nlat, Nz)
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
    close(io)

    @info "Converted $bin_path → $nc_path ($Nt timesteps)"
    return nc_path
end

"""Map dimension names to their sizes (excluding "time")."""
function _dims_to_shape(dims, Nlon, Nlat, Nz)
    shape = Int[]
    for d in dims
        d == "time"  && continue
        d == "lon"   && push!(shape, Nlon)
        d == "lat"   && push!(shape, Nlat)
        d == "lev"   && push!(shape, Nz)
        d == "x"     && push!(shape, Nlon)
        d == "y"     && push!(shape, Nlat)
    end
    return tuple(shape...)
end

export BinaryOutputWriter
export convert_binary_to_netcdf
