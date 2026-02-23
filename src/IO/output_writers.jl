# ---------------------------------------------------------------------------
# Output writers — schedule-based diagnostic output
#
# Supports both LatitudeLongitudeGrid and CubedSphereGrid via dispatch.
# For cubed-sphere, output can be regridded to lat-lon via LatLonOutputGrid.
# AbstractDiagnostic entries in the fields dict are auto-computed on write.
# ---------------------------------------------------------------------------

using Dates
using NCDatasets
using ..Grids: grid_size, floattype, LatitudeLongitudeGrid, CubedSphereGrid, znode, Center
using ..Fields: interior, AbstractField
using ..Architectures: array_type
using ..Diagnostics: AbstractDiagnostic, ColumnMeanDiagnostic, SurfaceSliceDiagnostic,
                     RegridDiagnostic, Full3DDiagnostic, MetField2DDiagnostic,
                     column_mean!, surface_slice!, regrid_cs_to_latlon,
                     RegridMapping, build_regrid_mapping, regrid_cs_to_latlon!

"""
$(TYPEDEF)

Supertype for output writers. Subtype and implement `write_output!`.
"""
abstract type AbstractOutputWriter end

"""
$(TYPEDEF)

Supertype for output scheduling.
"""
abstract type AbstractOutputSchedule end

"""
$(TYPEDEF)

Output every `interval` seconds of simulation time.

$(FIELDS)
"""
struct TimeIntervalSchedule <: AbstractOutputSchedule
    "output interval in seconds of simulation time"
    interval :: Float64
end

"""
$(TYPEDEF)

Output every `interval` iterations.

$(FIELDS)
"""
struct IterationIntervalSchedule <: AbstractOutputSchedule
    "output interval in number of iterations"
    interval :: Int
end

# =====================================================================
# Output grid abstraction (for CS → lat-lon regridding)
# =====================================================================

"""
$(TYPEDEF)

Supertype for output grid specifications. Used when the output grid
differs from the model grid (e.g. cubed-sphere → lat-lon regridding).
"""
abstract type AbstractOutputGrid end

"""
$(TYPEDEF)

Lat-lon output grid for regridding cubed-sphere data before writing.

$(FIELDS)
"""
struct LatLonOutputGrid <: AbstractOutputGrid
    "number of longitude points"
    Nlon :: Int
    "number of latitude points"
    Nlat :: Int
end

LatLonOutputGrid(; Nlon=720, Nlat=361) = LatLonOutputGrid(Nlon, Nlat)

# =====================================================================
# NetCDFOutputWriter
# =====================================================================

"""
$(TYPEDEF)

Write selected fields to NetCDF files on a schedule.

Fields can be:
- `AbstractField` — extracted via `interior()`
- `Function` — called with `(model)` argument
- `AbstractArray` — written directly
- `AbstractDiagnostic` — auto-computed from model state (column mean, surface slice, etc.)

For cubed-sphere grids, set `output_grid` to a `LatLonOutputGrid` to
regrid panel data to lat-lon before writing.

$(FIELDS)
"""
struct NetCDFOutputWriter{S <: AbstractOutputSchedule, OG} <: AbstractOutputWriter
    "output file path"
    filename      :: String
    "name → field, function, or AbstractDiagnostic"
    fields        :: Dict{Symbol, Any}
    "output schedule"
    schedule      :: S
    "optional output grid for regridding (nothing = native grid)"
    output_grid   :: OG
    "number of writes so far"
    _write_count  :: Ref{Int}
    "lazily-built RegridMapping for GPU CS→lat-lon regridding (nothing until first write)"
    _regrid_cache :: Ref{Any}
end

function NetCDFOutputWriter(filename::String, fields::Dict, schedule::S;
                            output_grid=nothing) where S <: AbstractOutputSchedule
    return NetCDFOutputWriter{S, typeof(output_grid)}(
        filename, fields, schedule, output_grid, Ref(0), Ref{Any}(nothing))
end

# Backward compat: 3-arg constructor without output_grid
function NetCDFOutputWriter(filename::String, fields::Dict, schedule::S,
                            ::Nothing) where S <: AbstractOutputSchedule
    return NetCDFOutputWriter(filename, fields, schedule; output_grid=nothing)
end

"""
$(SIGNATURES)

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

# =====================================================================
# Field extraction — dispatch on entry type
# =====================================================================

"""
$(SIGNATURES)

Extract array data from a field entry for NetCDF writing.
Ensures result is a CPU array.
"""
function _extract_field_data(field_or_func, model;
                             air_mass=nothing, tracers=nothing, regrid_cache=nothing)
    if field_or_func isa AbstractField
        arr = interior(field_or_func)
        return Array(arr)  # ensure CPU for NCDatasets
    elseif field_or_func isa Function
        arr = field_or_func(model)
        return Array(arr)
    elseif field_or_func isa AbstractArray
        return Array(field_or_func)
    elseif field_or_func isa AbstractDiagnostic
        return _compute_diagnostic(field_or_func, model; air_mass, tracers, regrid_cache)
    else
        error("Field must be AbstractField, Function, Array, or AbstractDiagnostic, " *
              "got $(typeof(field_or_func))")
    end
end

"""
Compute a diagnostic from model state. Returns a CPU array.
"""
function _compute_diagnostic(diag::ColumnMeanDiagnostic, model;
                             air_mass=nothing, tracers=nothing, regrid_cache=nothing)
    grid = model.grid
    if grid isa LatitudeLongitudeGrid
        c = _get_tracer(model, diag.species; tracers)
        m = air_mass !== nothing ? air_mass : _compute_uniform_mass(c)
        c_col = similar(c, size(c, 1), size(c, 2))
        column_mean!(c_col, c, m)
        return Array(c_col)
    elseif grid isa CubedSphereGrid
        rm = _get_tracer_panels(model, diag.species; tracers)
        Nc = grid.Nc
        Nz = size(rm[1], 3)
        Hp = div(size(rm[1], 1) - Nc, 2)
        m = air_mass !== nothing ? air_mass : ntuple(_ -> ones(eltype(rm[1]), size(rm[1])), 6)
        c_col_panels = ntuple(_ -> similar(rm[1], Nc, Nc), 6)
        column_mean!(c_col_panels, rm, m, Nc, Nz, Hp)
        panels_cpu = ntuple(p -> Array(c_col_panels[p]), 6)
        return panels_cpu
    end
end

function _compute_diagnostic(diag::SurfaceSliceDiagnostic, model;
                             air_mass=nothing, tracers=nothing, regrid_cache=nothing)
    grid = model.grid
    if grid isa LatitudeLongitudeGrid
        c = _get_tracer(model, diag.species; tracers)
        c_sfc = similar(c, size(c, 1), size(c, 2))
        surface_slice!(c_sfc, c)
        return Array(c_sfc)
    elseif grid isa CubedSphereGrid
        c_panels = _get_tracer_panels(model, diag.species; tracers)
        Nc = grid.Nc
        Nz = size(c_panels[1], 3)
        Hp = div(size(c_panels[1], 1) - Nc, 2)
        c_sfc_panels = ntuple(_ -> similar(c_panels[1], Nc, Nc), 6)
        surface_slice!(c_sfc_panels, c_panels, Nc, Nz, Hp)
        panels_cpu = ntuple(p -> Array(c_sfc_panels[p]), 6)
        return panels_cpu
    end
end

function _compute_diagnostic(diag::RegridDiagnostic, model;
                             air_mass=nothing, tracers=nothing, regrid_cache=nothing)
    grid = model.grid
    grid isa CubedSphereGrid || error("RegridDiagnostic only applies to CubedSphereGrid")
    rm = _get_tracer_panels(model, diag.species; tracers)
    Nc = grid.Nc
    Nz = size(rm[1], 3)
    Hp = div(size(rm[1], 1) - Nc, 2)
    m = air_mass !== nothing ? air_mass : ntuple(_ -> ones(eltype(rm[1]), size(rm[1])), 6)
    c_col_panels = ntuple(_ -> similar(rm[1], Nc, Nc), 6)
    column_mean!(c_col_panels, rm, m, Nc, Nz, Hp)

    # GPU scatter path: build RegridMapping once then reuse every write
    if regrid_cache !== nothing
        if regrid_cache[] === nothing
            AT = array_type(model.architecture)
            regrid_cache[] = build_regrid_mapping(grid, AT, diag.Nlon, diag.Nlat)
        end
        mapping = regrid_cache[]::RegridMapping
        out_ll = regrid_cs_to_latlon!(mapping, c_col_panels)
        return Array(out_ll)
    end

    # CPU fallback (no cache provided)
    panels_cpu = ntuple(p -> Array(c_col_panels[p]), 6)
    out, _, _ = regrid_cs_to_latlon(panels_cpu, grid; Nlon=diag.Nlon, Nlat=diag.Nlat)
    return out
end

function _compute_diagnostic(diag::Full3DDiagnostic, model;
                             air_mass=nothing, tracers=nothing, regrid_cache=nothing)
    grid = model.grid
    if grid isa LatitudeLongitudeGrid
        c = _get_tracer(model, diag.species; tracers)
        return Array(c)
    elseif grid isa CubedSphereGrid
        # Return panels — will be regridded or written natively by _write_field_slice!
        c_panels = _get_tracer_panels(model, diag.species; tracers)
        return ntuple(p -> Array(c_panels[p]), 6)
    end
end

function _compute_diagnostic(diag::MetField2DDiagnostic, model;
                             air_mass=nothing, tracers=nothing, regrid_cache=nothing)
    met = model.met_data
    fn = diag.field_name

    # Try to extract the field from met_data (NamedTuple or struct with fields)
    if met isa NamedTuple && haskey(met, fn)
        return Array(met[fn])
    elseif hasproperty(met, fn)
        return Array(getproperty(met, fn))
    elseif met isa NamedTuple && fn === :surface_pressure && haskey(met, :ps)
        return Array(met[:ps])
    elseif met isa NamedTuple && fn === :pbl_height && haskey(met, :pblh)
        return Array(met[:pblh])
    elseif met isa NamedTuple && fn === :tropopause_height && haskey(met, :troph)
        return Array(met[:troph])
    else
        @warn "MetField2DDiagnostic: field '$fn' not found in met_data — returning zeros"
        grid = model.grid
        if grid isa LatitudeLongitudeGrid
            return zeros(Float32, grid.Nx, grid.Ny)
        else
            return zeros(Float32, grid.Nc, grid.Nc)
        end
    end
end

# Model accessor helpers — use `tracers` override if provided
function _get_tracer(model, species::Symbol; tracers=nothing)
    tr = tracers !== nothing ? tracers : model.tracers
    return getfield(tr, species)
end

function _get_tracer_panels(model, species::Symbol; tracers=nothing)
    tr = tracers !== nothing ? tracers : model.tracers
    return getfield(tr, species)
end

# Fallback: uniform mass weighting when air mass not provided
function _compute_uniform_mass(c::AbstractArray{FT}) where FT
    return ones(FT, size(c))
end

# =====================================================================
# File creation — dispatch on grid type
# =====================================================================

"""
$(SIGNATURES)

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

        defVar(ds, "lon", FT, ("lon",); attrib=Dict("units" => "degrees_east"))
        defVar(ds, "lat", FT, ("lat",); attrib=Dict("units" => "degrees_north"))
        defVar(ds, "lev", FT, ("lev",); attrib=Dict("units" => "Pa"))
        defVar(ds, "time", Float64, ("time",);
               attrib=Dict("units" => "seconds since 2000-01-01 00:00:00"))

        ds["lon"][:] = lon
        ds["lat"][:] = lat
        ds["lev"][:] = lev

        for (name, field_entry) in writer.fields
            dims = _output_dims(field_entry, grid, writer.output_grid)
            defVar(ds, string(name), FT, dims)
        end
    end
    return nothing
end

"""
Create NetCDF file for cubed-sphere grid output.

If `output_grid` is a `LatLonOutputGrid`, dimensions are (lon, lat, time).
Otherwise dimensions are (panel, x, y, time) for native CS output.
"""
function _create_netcdf_file(writer::NetCDFOutputWriter, model, grid::CubedSphereGrid)
    FT = floattype(grid)
    Nc = grid.Nc

    NCDataset(writer.filename, "c") do ds
        if writer.output_grid isa LatLonOutputGrid
            og = writer.output_grid
            Nlon, Nlat = og.Nlon, og.Nlat
            lons = collect(range(FT(-180), FT(180) - FT(360)/Nlon, length=Nlon))
            lats = collect(range(FT(-90), FT(90), length=Nlat))

            defDim(ds, "lon", Nlon)
            defDim(ds, "lat", Nlat)
            defDim(ds, "time", 0)

            defVar(ds, "lon", Float32, ("lon",);
                   attrib=Dict("units" => "degrees_east"))[:] = Float32.(lons)
            defVar(ds, "lat", Float32, ("lat",);
                   attrib=Dict("units" => "degrees_north"))[:] = Float32.(lats)
            defVar(ds, "time", Float64, ("time",);
                   attrib=Dict("units" => "seconds since 2000-01-01 00:00:00"))

            for (name, field_entry) in writer.fields
                dims = _output_dims(field_entry, grid, writer.output_grid)
                defVar(ds, string(name), Float32, dims;
                       attrib=Dict("units" => "ppm"))
            end
        else
            # Native cubed-sphere output
            defDim(ds, "panel", 6)
            defDim(ds, "x", Nc)
            defDim(ds, "y", Nc)
            defDim(ds, "time", 0)

            defVar(ds, "time", Float64, ("time",))

            for (name, _) in writer.fields
                defVar(ds, string(name), Float32, ("x", "y", "panel", "time"))
            end
        end
    end
    return nothing
end

"""Determine NetCDF dimension tuple for a field entry."""
function _output_dims(field_entry, grid::LatitudeLongitudeGrid, output_grid)
    if field_entry isa ColumnMeanDiagnostic || field_entry isa SurfaceSliceDiagnostic
        return ("lon", "lat", "time")
    elseif field_entry isa RegridDiagnostic
        return ("lon", "lat", "time")
    elseif field_entry isa MetField2DDiagnostic
        return ("lon", "lat", "time")
    elseif field_entry isa Full3DDiagnostic
        return ("lon", "lat", "lev", "time")
    else
        return ("lon", "lat", "lev", "time")
    end
end

function _output_dims(field_entry, grid::CubedSphereGrid, output_grid::LatLonOutputGrid)
    if field_entry isa Full3DDiagnostic
        return ("lon", "lat", "lev", "time")
    else
        return ("lon", "lat", "time")
    end
end

function _output_dims(field_entry, grid::CubedSphereGrid, output_grid)
    return ("x", "y", "panel", "time")
end

# =====================================================================
# Initialize + Write — dispatch on grid type
# =====================================================================

"""
$(SIGNATURES)

Create the NetCDF file with proper dimensions from `model.grid`, define all variables
from `writer.fields`, and write coordinate variables (lon, lat, lev).
"""
function initialize_output!(writer::NetCDFOutputWriter, model)
    _create_netcdf_file(writer, model, model.grid)
    return nothing
end

"""
$(SIGNATURES)

Write output if the schedule condition is met. Opens or creates the NetCDF file,
appends a new time slice for each field, and increments `_write_count`.
"""
function write_output!(writer::NetCDFOutputWriter, model, time;
                       air_mass=nothing, tracers=nothing)
    model_time = time isa Number ? Float64(time) : Float64(Dates.value(time) / 1000.0)
    iteration = hasproperty(model, :clock) ? model.clock.iteration : 0

    if !should_write(writer, model_time, iteration)
        return nothing
    end

    # Create file if it doesn't exist
    if !isfile(writer.filename)
        _create_netcdf_file(writer, model, model.grid)
    end

    grid = model.grid

    # Append time slice
    NCDataset(writer.filename, "a") do ds
        n_time = size(ds["time"], 1)
        ds["time"][n_time + 1] = model_time

        for (name, field_entry) in writer.fields
            arr = _extract_field_data(field_entry, model; air_mass, tracers,
                                      regrid_cache=writer._regrid_cache)
            _write_field_slice!(ds, string(name), arr, grid, writer.output_grid, n_time + 1)
        end
    end

    writer._write_count[] += 1
    return nothing
end

"""Write a single field slice to the NetCDF dataset."""
function _write_field_slice!(ds, name::String, arr::AbstractArray, grid::LatitudeLongitudeGrid,
                              output_grid, tidx::Int)
    if ndims(arr) == 2
        ds[name][:, :, tidx] = Float32.(arr)
    else
        ds[name][:, :, :, tidx] = Float32.(arr)
    end
end

function _write_field_slice!(ds, name::String, arr, grid::CubedSphereGrid,
                              output_grid::LatLonOutputGrid, tidx::Int)
    # arr is either a regridded lat-lon matrix or NTuple{6} of panels
    if arr isa NTuple
        # Regrid panels to lat-lon
        panels_cpu = ntuple(p -> arr[p] isa Array ? arr[p] : Array(arr[p]), 6)
        out_ll, _, _ = regrid_cs_to_latlon(panels_cpu, grid;
                                            Nlon=output_grid.Nlon, Nlat=output_grid.Nlat)
        ds[name][:, :, tidx] = Float32.(out_ll)
    else
        ds[name][:, :, tidx] = Float32.(arr)
    end
end

function _write_field_slice!(ds, name::String, arr, grid::CubedSphereGrid,
                              output_grid, tidx::Int)
    # Native CS output — write panels
    if arr isa NTuple
        for p in 1:6
            panel_data = arr[p] isa Array ? arr[p] : Array(arr[p])
            ds[name][:, :, p, tidx] = Float32.(panel_data)
        end
    else
        ds[name][:, :, :, tidx] = Float32.(arr)
    end
end

"""
$(SIGNATURES)

Write output if the schedule condition is met. Implementation stub for other writers.
"""
function write_output!(writer::AbstractOutputWriter, model, time)
    error("write_output! not implemented for $(typeof(writer))")
end

export AbstractOutputWriter, AbstractOutputSchedule
export NetCDFOutputWriter, TimeIntervalSchedule, IterationIntervalSchedule
export AbstractOutputGrid, LatLonOutputGrid
export write_output!, initialize_output!
