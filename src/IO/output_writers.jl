# ---------------------------------------------------------------------------
# Output writers — schedule-based diagnostic output
# ---------------------------------------------------------------------------

using Dates
using NCDatasets
using ..Grids: grid_size, floattype, LatitudeLongitudeGrid, znode, Center
using ..Fields: interior, AbstractField

"""
    AbstractOutputWriter

Supertype for output writers. Subtype and implement `write_output!`.
"""
abstract type AbstractOutputWriter end

"""
    AbstractOutputSchedule

Supertype for output scheduling.
"""
abstract type AbstractOutputSchedule end

"""Output every `interval` seconds of simulation time."""
struct TimeIntervalSchedule <: AbstractOutputSchedule
    interval :: Float64
end

"""Output every `interval` iterations."""
struct IterationIntervalSchedule <: AbstractOutputSchedule
    interval :: Int
end

"""
    NetCDFOutputWriter{S} <: AbstractOutputWriter

Write selected fields to NetCDF files on a schedule.

# Fields
- `filename :: String`
- `fields :: Dict{Symbol, Any}` — name → field or function
- `schedule :: S`
- `_write_count :: Ref{Int}` — number of writes so far
"""
struct NetCDFOutputWriter{S <: AbstractOutputSchedule} <: AbstractOutputWriter
    filename     :: String
    fields       :: Dict{Symbol, Any}
    schedule     :: S
    _write_count :: Ref{Int}

    function NetCDFOutputWriter(filename::String, fields::Dict, schedule::S) where S <: AbstractOutputSchedule
        return new{S}(filename, fields, schedule, Ref(0))
    end
end

"""
    should_write(writer::NetCDFOutputWriter, model_time, iteration) -> Bool

Return true if the schedule says we should write at this time/iteration.
"""
function should_write(writer::NetCDFOutputWriter, model_time, iteration)
    s = writer.schedule
    if s isa TimeIntervalSchedule
        # Write when model_time >= (_write_count + 1) * interval
        threshold = (writer._write_count[] + 1) * s.interval
        return model_time >= threshold
    elseif s isa IterationIntervalSchedule
        return iteration % s.interval == 0
    else
        return false
    end
end

"""
    _extract_field_data(field_or_func, model)

Extract 3D array from a field entry (AbstractField, function, or raw array).
Ensures CPU array for NetCDF writing.
"""
function _extract_field_data(field_or_func, model)
    if field_or_func isa AbstractField
        arr = interior(field_or_func)
        return Array(arr)  # ensure CPU for NCDatasets
    elseif field_or_func isa Function
        arr = field_or_func(model)
        return Array(arr)
    elseif field_or_func isa AbstractArray
        return Array(field_or_func)
    else
        error("Field must be AbstractField, Function, or Array, got $(typeof(field_or_func))")
    end
end

"""
    _create_netcdf_file(writer::NetCDFOutputWriter, model, grid::LatitudeLongitudeGrid)

Create a new NetCDF file with dimensions (lon, lat, lev, time) and coordinate variables.
Defines all output variables. Does not write any time slices.
"""
function _create_netcdf_file(writer::NetCDFOutputWriter, model, grid::LatitudeLongitudeGrid)
    gs = grid_size(grid)
    FT = floattype(grid)
    Nx, Ny, Nz = gs.Nx, gs.Ny, gs.Nz

    lon = collect(FT, grid.λᶜ)
    lat = collect(FT, grid.φᶜ)
    lev = [znode(k, grid, Center()) for k in 1:Nz]

    NCDataset(writer.filename, "c") do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        defDim(ds, "lev", Nz)
        defDim(ds, "time", 0)  # unlimited

        defVar(ds, "lon", FT, ("lon",))
        defVar(ds, "lat", FT, ("lat",))
        defVar(ds, "lev", FT, ("lev",))
        defVar(ds, "time", FT, ("time",))

        ds["lon"][:] = lon
        ds["lat"][:] = lat
        ds["lev"][:] = lev

        for (name, _) in writer.fields
            defVar(ds, string(name), FT, ("lon", "lat", "lev", "time"))
        end
    end
    return nothing
end

"""
    initialize_output!(writer::NetCDFOutputWriter, model)

Create the NetCDF file with proper dimensions from `model.grid`, define all variables
from `writer.fields`, and write coordinate variables (lon, lat, lev).
"""
function initialize_output!(writer::NetCDFOutputWriter, model)
    grid = model.grid
    if !(grid isa LatitudeLongitudeGrid)
        error("NetCDFOutputWriter currently only supports LatitudeLongitudeGrid, got $(typeof(grid))")
    end
    _create_netcdf_file(writer, model, grid)
    return nothing
end

"""
    write_output!(writer::NetCDFOutputWriter, model, time)

Write output if the schedule condition is met. Opens or creates the NetCDF file,
appends a new time slice for each field, and increments `_write_count`.
"""
function write_output!(writer::NetCDFOutputWriter, model, time)
    model_time = time isa Number ? Float64(time) : Float64(Dates.value(time) / 1000.0)
    iteration = hasproperty(model, :clock) ? model.clock.iteration : 0

    if !should_write(writer, model_time, iteration)
        return nothing
    end

    grid = model.grid
    if !(grid isa LatitudeLongitudeGrid)
        error("NetCDFOutputWriter currently only supports LatitudeLongitudeGrid, got $(typeof(grid))")
    end

    # Create file if it doesn't exist
    if !isfile(writer.filename)
        _create_netcdf_file(writer, model, grid)
    end

    # Append time slice
    NCDataset(writer.filename, "a") do ds
        n_time = size(ds["time"], 1)
        ds["time"][n_time + 1] = model_time

        for (name, field_or_func) in writer.fields
            arr = _extract_field_data(field_or_func, model)
            ds[string(name)][:, :, :, n_time + 1] = arr
        end
    end

    writer._write_count[] += 1
    return nothing
end

"""
    write_output!(writer::AbstractOutputWriter, model, time)

Write output if the schedule condition is met. Implementation stub for other writers.
"""
function write_output!(writer::AbstractOutputWriter, model, time)
    error("write_output! not implemented for $(typeof(writer))")
end

export AbstractOutputWriter, AbstractOutputSchedule
export NetCDFOutputWriter, TimeIntervalSchedule, IterationIntervalSchedule
export write_output!, initialize_output!
