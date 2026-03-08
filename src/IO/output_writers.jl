# ---------------------------------------------------------------------------
# Output writers — schedule-based diagnostic output
#
# Supports both LatitudeLongitudeGrid and CubedSphereGrid via dispatch.
# For cubed-sphere, output can be regridded to lat-lon via LatLonOutputGrid.
# AbstractDiagnostic entries in the fields dict are auto-computed on write.
# ---------------------------------------------------------------------------

using Dates
using NCDatasets
using ..Grids: grid_size, floattype, LatitudeLongitudeGrid, CubedSphereGrid, znode, Center,
               cell_area
using ..Fields: interior, AbstractField
using ..Architectures: array_type
using ..Diagnostics: AbstractDiagnostic, ColumnMeanDiagnostic, ColumnMassDiagnostic,
                     SurfaceSliceDiagnostic,
                     RegridDiagnostic, Full3DDiagnostic, MetField2DDiagnostic,
                     SigmaLevelDiagnostic, ColumnFluxDiagnostic,
                     EmissionFluxDiagnostic,
                     column_mean!, column_mass!, surface_slice!, sigma_level_slice!,
                     column_tracer_flux!,
                     regrid_cs_to_latlon,
                     RegridMapping, build_regrid_mapping, regrid_cs_to_latlon!
using ..Sources: molar_mass_for_species, M_AIR

# ---- Specific humidity helper for wet → dry mole fraction conversion ----

"""Extract QV panels from met_fields if available, or return nothing."""
_get_qv(met_fields) = met_fields isa NamedTuple && hasproperty(met_fields, :qv) ? met_fields.qv : nothing

const _QV_WARNED = Ref(false)
function _warn_no_qv()
    if !_QV_WARNED[]
        @warn "QV not available — reporting wet mole fractions (no dry correction)"
        _QV_WARNED[] = true
    end
end

# =====================================================================
# Field metadata — map diagnostic types to CF-convention units/long_name
# =====================================================================

"""
    _field_attribs(field_entry) → Dict{String,String}

Return a `Dict` of NetCDF attributes (`units`, `long_name`) for a diagnostic
or field entry. Used by both direct NetCDF writers and the binary→NC converter.
"""
function _field_attribs(field_entry)
    if field_entry isa ColumnMeanDiagnostic
        return Dict("units" => "mol mol-1",
                     "long_name" => "column-mean mole fraction of $(field_entry.species)")
    elseif field_entry isa ColumnMassDiagnostic
        return Dict("units" => "kg m-2",
                     "long_name" => "column mass of $(field_entry.species)")
    elseif field_entry isa SurfaceSliceDiagnostic
        return Dict("units" => "mol mol-1",
                     "long_name" => "surface mole fraction of $(field_entry.species)")
    elseif field_entry isa SigmaLevelDiagnostic
        sigma_str = string(round(field_entry.sigma; digits=3))
        return Dict("units" => "mol mol-1",
                     "long_name" => "mole fraction of $(field_entry.species) at sigma=$(sigma_str)")
    elseif field_entry isa Full3DDiagnostic
        return Dict("units" => "mol mol-1",
                     "long_name" => "mole fraction of $(field_entry.species)")
    elseif field_entry isa EmissionFluxDiagnostic
        return Dict("units" => "kg m-2 s-1",
                     "long_name" => "surface emission flux of $(field_entry.species)")
    elseif field_entry isa ColumnFluxDiagnostic
        dir = field_entry.direction == :east ? "eastward" : "northward"
        return Dict("units" => "kg m-1",
                     "long_name" => "column-integrated $(dir) tracer flux of $(field_entry.species)")
    elseif field_entry isa MetField2DDiagnostic
        return _met_field_attribs(field_entry.field_name)
    elseif field_entry isa RegridDiagnostic
        return Dict("units" => "mol mol-1",
                     "long_name" => "regridded mole fraction of $(field_entry.species)")
    else
        return Dict{String,String}()
    end
end

"""Map known met field names to units/long_name."""
function _met_field_attribs(fname::Symbol)
    if fname == :surface_pressure
        return Dict("units" => "Pa", "long_name" => "surface pressure")
    elseif fname == :pbl_height
        return Dict("units" => "m", "long_name" => "planetary boundary layer height")
    elseif fname == :tropopause_height || fname == :tropopause_pressure
        return Dict("units" => "Pa", "long_name" => "tropopause pressure")
    elseif fname == :surface_temperature || fname == :t2m
        return Dict("units" => "K", "long_name" => "2-metre temperature")
    else
        return Dict("units" => "", "long_name" => string(fname))
    end
end

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
Supports regional subsetting via bounding box (default: global).

$(FIELDS)
"""
struct LatLonOutputGrid{FT} <: AbstractOutputGrid
    "number of longitude points"
    Nlon :: Int
    "number of latitude points"
    Nlat :: Int
    "western boundary [degrees]"
    lon0 :: FT
    "eastern boundary [degrees]"
    lon1 :: FT
    "southern boundary [degrees]"
    lat0 :: FT
    "northern boundary [degrees]"
    lat1 :: FT
end

function LatLonOutputGrid(; Nlon=720, Nlat=361,
                            lon_range=(-180.0, 180.0),
                            lat_range=(-90.0, 90.0))
    FT = typeof(float(lon_range[1]))
    LatLonOutputGrid{FT}(Nlon, Nlat,
                          FT(lon_range[1]), FT(lon_range[2]),
                          FT(lat_range[1]), FT(lat_range[2]))
end

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
    "NetCDF deflate compression level (0 = off, 1–9 = increasing compression)"
    deflate_level :: Int
    "decimal places for rounding before write (nothing = no rounding)"
    digits        :: Union{Nothing, Int}
    "simulation start date for CF-convention time units"
    start_date    :: Date
    "lazily-built flux accumulator state for ColumnFluxDiagnostic (nothing until first use)"
    _flux_accumulators :: Ref{Any}
end

function NetCDFOutputWriter(filename::String, fields::Dict, schedule::S;
                            output_grid=nothing, deflate_level::Int=0,
                            digits::Union{Nothing,Int}=nothing,
                            start_date::Date=Date(2000,1,1)) where S <: AbstractOutputSchedule
    return NetCDFOutputWriter{S, typeof(output_grid)}(
        filename, fields, schedule, output_grid, Ref(0), Ref{Any}(nothing),
        deflate_level, digits, start_date, Ref{Any}(nothing))
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
function should_write(writer::AbstractOutputWriter, model_time, iteration)
    # Always write the IC snapshot at t=0
    if writer._write_count[] == 0 && model_time == 0.0
        return true
    end
    s = writer.schedule
    if s isa TimeIntervalSchedule
        # IC write at t=0 doesn't count toward the regular schedule:
        # subtract 1 from write_count when an IC was written (write_count >= 1 and t>0)
        effective_count = writer._write_count[] > 0 ? writer._write_count[] - 1 : writer._write_count[]
        threshold = (effective_count + 1) * s.interval
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
                             air_mass=nothing, tracers=nothing, regrid_cache=nothing,
                             output_grid=nothing, met_fields=nothing)
    if field_or_func isa AbstractField
        arr = interior(field_or_func)
        return Array(arr)  # ensure CPU for NCDatasets
    elseif field_or_func isa Function
        arr = field_or_func(model)
        return Array(arr)
    elseif field_or_func isa AbstractArray
        return Array(field_or_func)
    elseif field_or_func isa AbstractDiagnostic
        return _compute_diagnostic(field_or_func, model; air_mass, tracers, regrid_cache,
                                    output_grid, met_fields)
    else
        error("Field must be AbstractField, Function, Array, or AbstractDiagnostic, " *
              "got $(typeof(field_or_func))")
    end
end

"""
Compute a diagnostic from model state. Returns a CPU array.
"""
function _compute_diagnostic(diag::ColumnMeanDiagnostic, model;
                             air_mass=nothing, tracers=nothing, regrid_cache=nothing,
                             output_grid=nothing, met_fields=nothing)
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
        # Use dry air mass if QV available: m_dry = m * (1 - qv)
        qv_panels = _get_qv(met_fields)
        m_for_mean = if qv_panels !== nothing && air_mass !== nothing
            ntuple(6) do p
                m_p = Array(m[p])
                qv_p = qv_panels[p]
                @inbounds for k in 1:Nz, jj in Hp+1:Hp+Nc, ii in Hp+1:Hp+Nc
                    m_p[ii, jj, k] *= (1 - qv_p[ii, jj, k])
                end
                typeof(m[p])(m_p)  # back to device if needed
            end
        else
            m
        end
        c_col_panels = ntuple(_ -> similar(rm[1], Nc, Nc), 6)
        column_mean!(c_col_panels, rm, m_for_mean, Nc, Nz, Hp)
        # Regrid to lat-lon using GEOS-FP file coordinates
        if output_grid isa LatLonOutputGrid
            return _regrid_panels_to_latlon(c_col_panels, model, grid, output_grid, regrid_cache)
        end
        panels_cpu = ntuple(p -> Array(c_col_panels[p]), 6)
        return panels_cpu
    end
end

function _compute_diagnostic(diag::ColumnMassDiagnostic, model;
                             air_mass=nothing, tracers=nothing, regrid_cache=nothing,
                             output_grid=nothing, met_fields=nothing)
    grid = model.grid
    # mol_ratio converts rm (mol/mol × kg_air) to tracer mass (kg)
    mol_ratio = M_AIR / molar_mass_for_species(diag.species)
    if grid isa LatitudeLongitudeGrid
        c = _get_tracer(model, diag.species; tracers)
        m = air_mass !== nothing ? air_mass : _compute_uniform_mass(c)
        cm_col = similar(c, size(c, 1), size(c, 2))
        column_mass!(cm_col, c, m)
        result = Array(cm_col)
        Nx, Ny = size(result)
        for j in 1:Ny, i in 1:Nx
            result[i, j] /= (mol_ratio * cell_area(i, j, grid))
        end
        return result
    elseif grid isa CubedSphereGrid
        rm = _get_tracer_panels(model, diag.species; tracers)
        Nc = grid.Nc
        Nz = size(rm[1], 3)
        Hp = div(size(rm[1], 1) - Nc, 2)
        cm_panels = ntuple(_ -> similar(rm[1], Nc, Nc), 6)
        column_mass!(cm_panels, rm, Nc, Nz, Hp)
        # Divide by mol_ratio (mol/mol × kg → kg) and cell area to get kg/m²
        cm_cpu = ntuple(6) do p
            panel = Array(cm_panels[p])
            panel ./= (mol_ratio .* grid.Aᶜ[p])
            panel
        end
        if output_grid isa LatLonOutputGrid
            AT = array_type(model.architecture)
            cm_device = ntuple(p -> AT(cm_cpu[p]), 6)
            return _regrid_panels_to_latlon(cm_device, model, grid, output_grid, regrid_cache)
        end
        return cm_cpu
    end
end

function _compute_diagnostic(diag::SurfaceSliceDiagnostic, model;
                             air_mass=nothing, tracers=nothing, regrid_cache=nothing,
                             output_grid=nothing, met_fields=nothing)
    grid = model.grid
    if grid isa LatitudeLongitudeGrid
        c = _get_tracer(model, diag.species; tracers)
        c_sfc = similar(c, size(c, 1), size(c, 2))
        surface_slice!(c_sfc, c)
        return Array(c_sfc)
    elseif grid isa CubedSphereGrid
        rm_panels = _get_tracer_panels(model, diag.species; tracers)
        Nc = grid.Nc
        Nz = size(rm_panels[1], 3)
        Hp = div(size(rm_panels[1], 1) - Nc, 2)
        rm_sfc_panels = ntuple(_ -> similar(rm_panels[1], Nc, Nc), 6)
        surface_slice!(rm_sfc_panels, rm_panels, Nc, Nz, Hp)
        # CS tracers are rm = air_mass × mixing_ratio; divide by surface air mass
        qv_panels = _get_qv(met_fields)
        if air_mass !== nothing
            m_sfc_panels = ntuple(_ -> similar(air_mass[1], Nc, Nc), 6)
            surface_slice!(m_sfc_panels, air_mass, Nc, Nz, Hp)
            sfc_panels = ntuple(6) do p
                rm_sfc_panels[p] ./= m_sfc_panels[p]  # wet mole fraction
                # Apply dry correction using surface-level QV
                if qv_panels !== nothing
                    qv_sfc = qv_panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz]  # surface = last level
                    rm_sfc_panels[p] ./= (1 .- qv_sfc)
                end
                rm_sfc_panels[p]
            end
        else
            sfc_panels = rm_sfc_panels
        end
        # Regrid to lat-lon using GEOS-FP file coordinates
        if output_grid isa LatLonOutputGrid
            return _regrid_panels_to_latlon(sfc_panels, model, grid, output_grid, regrid_cache)
        end
        panels_cpu = ntuple(p -> Array(sfc_panels[p]), 6)
        return panels_cpu
    end
end

function _compute_diagnostic(diag::SigmaLevelDiagnostic, model;
                             air_mass=nothing, tracers=nothing, regrid_cache=nothing,
                             output_grid=nothing, met_fields=nothing)
    grid = model.grid
    if grid isa LatitudeLongitudeGrid
        c = _get_tracer(model, diag.species; tracers)
        m = air_mass !== nothing ? air_mass : _compute_uniform_mass(c)
        c_sig = similar(c, size(c, 1), size(c, 2))
        sigma_level_slice!(c_sig, c, m, diag.sigma)
        return Array(c_sig)
    elseif grid isa CubedSphereGrid
        rm_panels = _get_tracer_panels(model, diag.species; tracers)
        Nc = grid.Nc
        Nz = size(rm_panels[1], 3)
        Hp = div(size(rm_panels[1], 1) - Nc, 2)
        m = air_mass !== nothing ? air_mass : ntuple(_ -> ones(eltype(rm_panels[1]), size(rm_panels[1])), 6)
        sig_panels = ntuple(_ -> similar(rm_panels[1], Nc, Nc), 6)
        sigma_level_slice!(sig_panels, rm_panels, m, Nc, Nz, Hp, diag.sigma)
        if output_grid isa LatLonOutputGrid
            return _regrid_panels_to_latlon(sig_panels, model, grid, output_grid, regrid_cache)
        end
        return ntuple(p -> Array(sig_panels[p]), 6)
    end
end

function _compute_diagnostic(diag::RegridDiagnostic, model;
                             air_mass=nothing, tracers=nothing, regrid_cache=nothing,
                             output_grid=nothing, met_fields=nothing)
    grid = model.grid
    grid isa CubedSphereGrid || error("RegridDiagnostic only applies to CubedSphereGrid")
    FT = floattype(grid)
    rm = _get_tracer_panels(model, diag.species; tracers)
    Nc = grid.Nc
    Nz = size(rm[1], 3)
    Hp = div(size(rm[1], 1) - Nc, 2)
    m = air_mass !== nothing ? air_mass : ntuple(_ -> ones(eltype(rm[1]), size(rm[1])), 6)
    c_col_panels = ntuple(_ -> similar(rm[1], Nc, Nc), 6)
    column_mean!(c_col_panels, rm, m, Nc, Nz, Hp)

    # Build a LatLonOutputGrid from the diagnostic's own Nlon/Nlat + bounding box
    og = output_grid
    _lon0 = og isa LatLonOutputGrid ? FT(og.lon0) : FT(-180)
    _lon1 = og isa LatLonOutputGrid ? FT(og.lon1) : FT(180)
    _lat0 = og isa LatLonOutputGrid ? FT(og.lat0) : FT(-90)
    _lat1 = og isa LatLonOutputGrid ? FT(og.lat1) : FT(90)
    diag_og = LatLonOutputGrid{FT}(diag.Nlon, diag.Nlat, _lon0, _lon1, _lat0, _lat1)

    return _regrid_panels_to_latlon(c_col_panels, model, grid, diag_og, regrid_cache)
end

"""
    _get_cs_file_coords(model)

Load cubed-sphere cell-center coordinates from the GEOS-FP file if available.
Returns `(lons, lats)` as `(Nc×Nc×6)` arrays, or `(nothing, nothing)` if
no coordinate file is available.
"""
function _get_cs_file_coords(model)
    met = model.met_data
    # Use coord_file if available (works for both binary and netcdf modes)
    if hasproperty(met, :coord_file) && !isempty(met.coord_file) &&
       isfile(met.coord_file)
        lons, lats, _, _ = read_geosfp_cs_grid_info(met.coord_file)
        return lons, lats
    end
    # Fallback: try first data file in netcdf mode
    if hasproperty(met, :mode) && met.mode == :netcdf &&
       hasproperty(met, :files) && !isempty(met.files)
        lons, lats, _, _ = read_geosfp_cs_grid_info(met.files[1])
        return lons, lats
    end
    return nothing, nothing
end

"""
    _regrid_panels_to_latlon(panels, model, grid, output_grid, regrid_cache)

Regrid NTuple{6} of 2D panel data to a lat-lon array using GEOS-FP file
coordinates (NOT gnomonic grid coordinates). Builds the RegridMapping lazily
via `regrid_cache` (shared across all diagnostics).

Returns a CPU `Array{FT,2}` of size `(Nlon, Nlat)`.
"""
function _regrid_panels_to_latlon(panels::NTuple{6}, model, grid::CubedSphereGrid{FT},
                                   output_grid::LatLonOutputGrid,
                                   regrid_cache) where FT
    og = output_grid
    if regrid_cache !== nothing
        if regrid_cache[] === nothing
            file_lons, file_lats = _get_cs_file_coords(model)
            AT = array_type(model.architecture)
            regrid_cache[] = build_regrid_mapping(grid, AT, og.Nlon, og.Nlat;
                                                   lon0=FT(og.lon0), lon1=FT(og.lon1),
                                                   lat0=FT(og.lat0), lat1=FT(og.lat1),
                                                   file_lons, file_lats)
        end
        mapping = regrid_cache[]::RegridMapping
        # Ensure panels are on the same device as the mapping (surface fields live on CPU)
        AT = array_type(model.architecture)
        gpu_panels = ntuple(p -> panels[p] isa AT ? panels[p] : AT(panels[p]), 6)
        return Array(regrid_cs_to_latlon!(mapping, gpu_panels))
    end
    # CPU fallback
    file_lons, file_lats = _get_cs_file_coords(model)
    panels_cpu = ntuple(p -> Array(panels[p]), 6)
    out, _, _ = regrid_cs_to_latlon(panels_cpu, grid;
                                     Nlon=og.Nlon, Nlat=og.Nlat,
                                     lon0=FT(og.lon0), lon1=FT(og.lon1),
                                     lat0=FT(og.lat0), lat1=FT(og.lat1),
                                     file_lons, file_lats)
    return out
end

function _compute_diagnostic(diag::Full3DDiagnostic, model;
                             air_mass=nothing, tracers=nothing, regrid_cache=nothing,
                             output_grid=nothing, met_fields=nothing)
    grid = model.grid
    if grid isa LatitudeLongitudeGrid
        c = _get_tracer(model, diag.species; tracers)
        return Array(c)
    elseif grid isa CubedSphereGrid
        rm_panels = _get_tracer_panels(model, diag.species; tracers)
        if output_grid isa LatLonOutputGrid
            # Level-by-level CS → lat-lon regridding of dry mole fraction
            Nc = grid.Nc
            Hp = div(size(rm_panels[1], 1) - Nc, 2)
            Nz = size(rm_panels[1], 3)
            FT = floattype(grid)
            og = output_grid
            qv_panels = _get_qv(met_fields)
            result = zeros(FT, og.Nlon, og.Nlat, Nz)
            for k in 1:Nz
                level_panels = ntuple(6) do p
                    rm_slice = rm_panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
                    if air_mass !== nothing
                        m_slice = air_mass[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
                        q_wet = rm_slice ./ m_slice
                        if qv_panels !== nothing
                            qv_slice = qv_panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, k]
                            q_wet ./ (1 .- qv_slice)  # dry mole fraction
                        else
                            q_wet
                        end
                    else
                        rm_slice
                    end
                end
                result[:, :, k] = _regrid_panels_to_latlon(
                    level_panels, model, grid, og, regrid_cache)
            end
            return result
        end
        # Native CS output — strip halos and convert to dry mole fraction
        Nc = grid.Nc
        Hp = div(size(rm_panels[1], 1) - Nc, 2)
        Nz = size(rm_panels[1], 3)
        qv_panels = _get_qv(met_fields)
        qv_panels === nothing && _warn_no_qv()
        return ntuple(6) do p
            rm_inner = Array(rm_panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, 1:Nz])
            if air_mass !== nothing
                m_inner = Array(air_mass[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, 1:Nz])
                rm_inner ./= m_inner  # wet mole fraction
                if qv_panels !== nothing
                    qv_inner = qv_panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, 1:Nz]
                    rm_inner ./= (1 .- qv_inner)  # wet → dry
                end
            end
            rm_inner
        end
    end
end

function _compute_diagnostic(diag::MetField2DDiagnostic, model;
                             air_mass=nothing, tracers=nothing, regrid_cache=nothing,
                             output_grid=nothing, met_fields=nothing)
    fn = diag.field_name
    grid = model.grid

    # Canonical field name mapping (config name → NamedTuple key)
    key = fn === :surface_pressure ? :ps :
          fn === :pbl_height       ? :pblh :
          fn === :tropopause_height ? :troph : fn

    # Priority 1: met_fields kwarg (explicitly passed from run loop)
    if met_fields isa NamedTuple && haskey(met_fields, key)
        field_data = met_fields[key]
        # CS panels (NTuple{6}) — strip halos and regrid to lat-lon
        if field_data isa NTuple && grid isa CubedSphereGrid
            Nc = grid.Nc
            Hp = div(size(field_data[1], 1) - Nc, 2)
            inner = ntuple(p -> Array(field_data[p])[Hp+1:Hp+Nc, Hp+1:Hp+Nc], 6)
            if output_grid isa LatLonOutputGrid
                return _regrid_panels_to_latlon(inner, model, grid, output_grid,
                                                 regrid_cache)
            end
            return inner
        end
        return Array(field_data)
    end

    # Priority 2: model.met_data (NamedTuple or struct with fields)
    met = model.met_data
    if met isa NamedTuple && haskey(met, fn)
        return Array(met[fn])
    elseif hasproperty(met, fn)
        return Array(getproperty(met, fn))
    elseif met isa NamedTuple && haskey(met, key)
        return Array(met[key])
    end

    @warn "MetField2DDiagnostic: field '$fn' not found in met_data — returning zeros"
    if grid isa LatitudeLongitudeGrid
        return zeros(Float32, grid.Nx, grid.Ny)
    elseif output_grid isa LatLonOutputGrid
        return zeros(Float32, output_grid.Nlon, output_grid.Nlat)
    elseif grid isa CubedSphereGrid
        return ntuple(_ -> zeros(Float32, grid.Nc, grid.Nc), 6)
    else
        return zeros(Float32, grid.Nc, grid.Nc)
    end
end

function _compute_diagnostic(diag::EmissionFluxDiagnostic, model;
                             air_mass=nothing, tracers=nothing, regrid_cache=nothing,
                             output_grid=nothing, met_fields=nothing)
    grid = model.grid
    species = diag.species
    # Search model.sources for a source matching this species
    for src in model.sources
        src_species = hasproperty(src, :species) ? src.species : nothing
        src_species === species || continue
        # Time-varying: get active snapshot via flux_data accessor or field
        if hasproperty(src, :flux_data) && hasproperty(src, :current_idx)
            fd = src.flux_data
            panels = fd isa AbstractVector ? fd[src.current_idx] : @view fd[:, :, src.current_idx]
        elseif hasproperty(src, :flux)
            fd = src.flux
            fd isa AbstractMatrix && return Array(fd)
            panels = fd  # NTuple{6, Matrix} for CS
        else
            continue
        end
        # CS panels → NTuple{6}
        if output_grid isa LatLonOutputGrid
            return _regrid_panels_to_latlon(panels, model, grid, output_grid, regrid_cache)
        end
        return panels   # native CS: NTuple{6, Matrix}
    end
    @warn "EmissionFluxDiagnostic: no source found for species '$species' — returning zeros"
    if grid isa LatitudeLongitudeGrid
        return zeros(Float32, grid.Nx, grid.Ny)
    elseif output_grid isa LatLonOutputGrid
        return zeros(Float32, output_grid.Nlon, output_grid.Nlat)
    else
        Nc = grid.Nc
        return ntuple(_ -> zeros(Float32, Nc, Nc), 6)
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
# Flux accumulation for ColumnFluxDiagnostic
#
# Called every met window (before should_write check). Computes the
# vertically-integrated column tracer flux on GPU, downloads the 2D
# result, and accumulates on CPU. Reset after each output write.
# =====================================================================

"""
    _accumulate_flux_diagnostics!(writer, model; air_mass, tracers, met_fields)

Accumulate column tracer flux diagnostics for all `ColumnFluxDiagnostic` fields
in the writer. Called every met window. Requires `met_fields` to contain:
- `:mass_flux_x` — NTuple{6} GPU panels (Nc+1, Nc, Nz), scaled by `mf_scale`
- `:mass_flux_y` — NTuple{6} GPU panels (Nc, Nc+1, Nz), scaled by `mf_scale`
- `:mf_scale`    — scale factor (half_dt) applied to mass fluxes
- `:dt_window`   — window duration in seconds
"""
function _accumulate_flux_diagnostics!(writer::NetCDFOutputWriter, model;
                                        air_mass=nothing, tracers=nothing, met_fields=nothing)
    # Early exit: any ColumnFluxDiagnostic fields?
    has_flux = any(f -> f isa ColumnFluxDiagnostic, values(writer.fields))
    !has_flux && return

    # Need mass fluxes from met_fields
    met_fields isa NamedTuple || return
    haskey(met_fields, :mass_flux_x) || return

    am = met_fields[:mass_flux_x]
    bm = met_fields[:mass_flux_y]
    mf_scale  = met_fields[:mf_scale]    # half_dt (mass fluxes already multiplied by this)
    dt_window = met_fields[:dt_window]

    grid = model.grid
    grid isa CubedSphereGrid || return  # CS only for now

    Nc = grid.Nc
    Nz = grid.Nz
    Hp = div(size(air_mass[1], 1) - Nc, 2)
    FT = eltype(air_mass[1])

    # Lazy init: accumulators (CPU) + GPU workspace
    if writer._flux_accumulators[] === nothing
        gpu_ws = ntuple(_ -> similar(air_mass[1], Nc, Nc), 6)  # GPU (Nc×Nc) per panel
        accum  = Dict{Symbol, NTuple{6, Matrix{FT}}}()
        writer._flux_accumulators[] = (; accum, gpu_ws)
    end
    state = writer._flux_accumulators[]
    accum  = state.accum
    gpu_ws = state.gpu_ws

    # Scale factor: am_gpu = am_raw × mf_scale, want am_raw × dt_window
    scale = FT(dt_window / mf_scale)

    for (name, field_entry) in writer.fields
        field_entry isa ColumnFluxDiagnostic || continue

        # Get tracer data (rm and air mass on GPU)
        rm = _get_tracer_panels(model, field_entry.species; tracers)
        m  = air_mass

        # Select mass flux panels based on direction
        mf = field_entry.direction === :east ? am : bm

        # Compute Σ_k mf[face_k] × (rm[k]/m[k]) on GPU → gpu_ws
        column_tracer_flux!(gpu_ws, mf, rm, m, Nc, Nz, Hp, field_entry.direction)

        # Init accumulator for this field if first call
        if !haskey(accum, name)
            accum[name] = ntuple(_ -> zeros(FT, Nc, Nc), 6)
        end

        # Download GPU result and accumulate (scale: convert from mf_scale to dt_window)
        for p in 1:6
            cpu_flux = Array(gpu_ws[p])
            accum[name][p] .+= cpu_flux .* scale
        end
    end
end

"""
    _extract_accumulated_flux(writer, name, model; regrid_cache, output_grid)

Extract accumulated column tracer flux for output. Returns a CPU array
(regridded to lat-lon if `output_grid` is a `LatLonOutputGrid`).
"""
function _extract_accumulated_flux(writer::NetCDFOutputWriter, name::Symbol, model;
                                    regrid_cache=nothing, output_grid=nothing)
    state = writer._flux_accumulators[]
    grid  = model.grid

    if state === nothing || !haskey(state.accum, name)
        @warn "No accumulated flux data for $name — returning zeros"
        if output_grid isa LatLonOutputGrid
            return zeros(Float32, output_grid.Nlon, output_grid.Nlat)
        elseif grid isa CubedSphereGrid
            Nc = grid.Nc
            return ntuple(_ -> zeros(Float32, Nc, Nc), 6)
        else
            return zeros(Float32, grid.Nx, grid.Ny)
        end
    end

    panels = state.accum[name]  # NTuple{6} of CPU (Nc×Nc) arrays

    if output_grid isa LatLonOutputGrid && grid isa CubedSphereGrid
        # Upload to GPU for regridding (small: 6 × Nc² × 4 bytes)
        AT = array_type(model.architecture)
        panels_gpu = ntuple(p -> AT(panels[p]), 6)
        return _regrid_panels_to_latlon(panels_gpu, model, grid, output_grid, regrid_cache)
    end
    return panels
end

"""Reset all flux accumulators to zero (called after output write)."""
function _reset_flux_accumulators!(writer::NetCDFOutputWriter)
    state = writer._flux_accumulators[]
    state === nothing && return
    for (_, panels) in state.accum
        for p in 1:6
            fill!(panels[p], 0)
        end
    end
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
               attrib=Dict("units" => "seconds since $(writer.start_date) 00:00:00"))

        ds["lon"][:] = lon
        ds["lat"][:] = lat
        ds["lev"][:] = lev

        dl = writer.deflate_level
        for (name, field_entry) in writer.fields
            dims = _output_dims(field_entry, grid, writer.output_grid)
            defVar(ds, string(name), FT, dims;
                   deflatelevel=dl, attrib=_field_attribs(field_entry))
        end
    end
    return nothing
end

"""
Create NetCDF file for cubed-sphere grid output.

If `output_grid` is a `LatLonOutputGrid`, dimensions are (lon, lat, time).
Otherwise native CS output uses GEOSCHEM-compatible dimensions:
(Xdim, Ydim, nf, time) for 2D fields, (Xdim, Ydim, nf, lev, time) for 3D.
Includes `lons(Xdim, Ydim, nf)` and `lats(Xdim, Ydim, nf)` coordinate arrays.
"""
function _create_netcdf_file(writer::NetCDFOutputWriter, model, grid::CubedSphereGrid)
    FT = floattype(grid)
    Nc = grid.Nc

    NCDataset(writer.filename, "c") do ds
        if writer.output_grid isa LatLonOutputGrid
            og = writer.output_grid
            Nlon, Nlat = og.Nlon, og.Nlat
            dlon = (FT(og.lon1) - FT(og.lon0)) / Nlon
            dlat = (FT(og.lat1) - FT(og.lat0)) / Nlat
            lons = collect(range(FT(og.lon0) + dlon/2, FT(og.lon1) - dlon/2, length=Nlon))
            lats = collect(range(FT(og.lat0) + dlat/2, FT(og.lat1) - dlat/2, length=Nlat))

            defDim(ds, "lon", Nlon)
            defDim(ds, "lat", Nlat)
            defDim(ds, "time", 0)

            # Add "lev" dimension if any Full3D fields are present
            has_3d = any(f -> f isa Full3DDiagnostic, values(writer.fields))
            if has_3d
                Nz = grid.Nz
                lev = [znode(k, grid, Center()) for k in 1:Nz]
                defDim(ds, "lev", Nz)
                defVar(ds, "lev", Float32, ("lev",);
                       attrib=Dict("units" => "Pa"))[:] = Float32.(lev)
            end

            defVar(ds, "lon", Float32, ("lon",);
                   attrib=Dict("units" => "degrees_east"))[:] = Float32.(lons)
            defVar(ds, "lat", Float32, ("lat",);
                   attrib=Dict("units" => "degrees_north"))[:] = Float32.(lats)
            defVar(ds, "time", Float64, ("time",);
                   attrib=Dict("units" => "seconds since $(writer.start_date) 00:00:00"))

            dl = writer.deflate_level
            for (name, field_entry) in writer.fields
                dims = _output_dims(field_entry, grid, writer.output_grid)
                defVar(ds, string(name), Float32, dims;
                       deflatelevel=dl, attrib=_field_attribs(field_entry))
            end
        else
            # Native cubed-sphere output — GEOSCHEM-compatible format
            # Dimensions: (Xdim, Ydim, nf, [lev,] time)
            defDim(ds, "Xdim", Nc)
            defDim(ds, "Ydim", Nc)
            defDim(ds, "nf", 6)
            defDim(ds, "time", 0)

            # Add "lev" dimension if any Full3D fields are present
            has_3d = any(f -> f isa Full3DDiagnostic, values(writer.fields))
            if has_3d
                Nz = grid.Nz
                lev = [znode(k, grid, Center()) for k in 1:Nz]
                defDim(ds, "lev", Nz)
                defVar(ds, "lev", Float32, ("lev",);
                       attrib=Dict("units" => "Pa"))[:] = Float32.(lev)
            end

            defVar(ds, "time", Float64, ("time",);
                   attrib=Dict("units" => "seconds since $(writer.start_date) 00:00:00"))

            # Coordinate arrays: lons(Xdim, Ydim, nf) and lats(Xdim, Ydim, nf)
            file_lons, file_lats = _get_cs_file_coords(model)
            if file_lons !== nothing
                defVar(ds, "lons", Float64, ("Xdim", "Ydim", "nf");
                       attrib=Dict("units" => "degrees_east",
                                   "long_name" => "longitude"))[:] = file_lons
                defVar(ds, "lats", Float64, ("Xdim", "Ydim", "nf");
                       attrib=Dict("units" => "degrees_north",
                                   "long_name" => "latitude"))[:] = file_lats
            end

            dl = writer.deflate_level
            for (name, field_entry) in writer.fields
                dims = _output_dims(field_entry, grid, writer.output_grid)
                attribs = merge(Dict("coordinates" => "lons lats",
                                     "grid_mapping" => "cubed_sphere"),
                                _field_attribs(field_entry))
                defVar(ds, string(name), Float32, dims;
                       deflatelevel=dl, attrib=attribs)
            end
        end
    end
    return nothing
end

"""Determine NetCDF dimension tuple for a field entry."""
function _output_dims(field_entry, grid::LatitudeLongitudeGrid, output_grid)
    if field_entry isa ColumnMeanDiagnostic || field_entry isa ColumnMassDiagnostic ||
       field_entry isa SurfaceSliceDiagnostic || field_entry isa SigmaLevelDiagnostic ||
       field_entry isa ColumnFluxDiagnostic || field_entry isa EmissionFluxDiagnostic
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
    # Use internal names (x, y, panel) — _cs_dims_to_nc_dims maps to NetCDF names,
    # _dims_to_shape maps to sizes
    if field_entry isa Full3DDiagnostic
        return ("x", "y", "panel", "lev", "time")
    else
        return ("x", "y", "panel", "time")
    end
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

# Round array values to `d` decimal places (improves deflate compression).
_quantize(arr, ::Nothing) = arr
_quantize(arr::AbstractArray, d::Int) = round.(Float32.(arr); digits=d)
_quantize(arr::NTuple, d::Int) = ntuple(p -> round.(Float32.(arr[p]); digits=d), length(arr))

"""
$(SIGNATURES)

Write output if the schedule condition is met. Opens or creates the NetCDF file,
appends a new time slice for each field, and increments `_write_count`.
"""
function write_output!(writer::NetCDFOutputWriter, model, time;
                       air_mass=nothing, tracers=nothing, met_fields=nothing)
    model_time = time isa Number ? Float64(time) : Float64(Dates.value(time) / 1000.0)
    iteration = hasproperty(model, :clock) ? model.clock.iteration : 0

    # Always accumulate flux diagnostics (every window, before schedule check)
    _accumulate_flux_diagnostics!(writer, model; air_mass, tracers, met_fields)

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
            if field_entry isa ColumnFluxDiagnostic
                # Read from accumulator (already computed every window)
                arr = _extract_accumulated_flux(writer, name, model;
                                                 regrid_cache=writer._regrid_cache,
                                                 output_grid=writer.output_grid)
            else
                arr = _extract_field_data(field_entry, model; air_mass, tracers,
                                          regrid_cache=writer._regrid_cache,
                                          output_grid=writer.output_grid,
                                          met_fields)
            end
            arr = _quantize(arr, writer.digits)
            _write_field_slice!(ds, string(name), arr, grid, writer.output_grid, n_time + 1)
        end
    end

    # Reset flux accumulators after writing
    _reset_flux_accumulators!(writer)

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
    # arr is either a regridded lat-lon array (2D or 3D) or NTuple{6} of panels
    if arr isa NTuple
        # Fallback: regrid panels without file coordinates (gnomonic coords only)
        @warn "CS→lat-lon regridding without file coordinates for field '$name'. " *
              "Output coordinates may be incorrect." maxlog=1
        FT = floattype(grid)
        panels_cpu = ntuple(p -> arr[p] isa Array ? arr[p] : Array(arr[p]), 6)
        out_ll, _, _ = regrid_cs_to_latlon(panels_cpu, grid;
                                            Nlon=output_grid.Nlon, Nlat=output_grid.Nlat,
                                            lon0=FT(output_grid.lon0), lon1=FT(output_grid.lon1),
                                            lat0=FT(output_grid.lat0), lat1=FT(output_grid.lat1))
        ds[name][:, :, tidx] = Float32.(out_ll)
    elseif arr isa AbstractArray && ndims(arr) == 3
        ds[name][:, :, :, tidx] = Float32.(arr)
    else
        ds[name][:, :, tidx] = Float32.(arr)
    end
end

function _write_field_slice!(ds, name::String, arr, grid::CubedSphereGrid,
                              output_grid, tidx::Int)
    # Native CS output — GEOSCHEM-compatible (Xdim, Ydim, nf, [lev,] time)
    if arr isa NTuple
        if ndims(arr[1]) == 2
            # 2D panels → (Xdim, Ydim, nf, time)
            for p in 1:6
                panel_data = arr[p] isa Array ? arr[p] : Array(arr[p])
                ds[name][:, :, p, tidx] = Float32.(panel_data)
            end
        else
            # 3D panels → (Xdim, Ydim, nf, lev, time)
            for p in 1:6
                panel_data = arr[p] isa Array ? arr[p] : Array(arr[p])
                ds[name][:, :, p, :, tidx] = Float32.(panel_data)
            end
        end
    elseif arr isa AbstractArray && ndims(arr) == 4
        # Pre-assembled (Xdim, Ydim, nf, lev) array
        ds[name][:, :, :, :, tidx] = Float32.(arr)
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

"""
    finalize_output!(writer)

Finalize an output writer at the end of a simulation. No-op for most writers.
For `BinaryOutputWriter`, updates the header with final `Nt` and optionally
converts to NetCDF.
"""
finalize_output!(::AbstractOutputWriter) = nothing

export AbstractOutputWriter, AbstractOutputSchedule
export NetCDFOutputWriter, TimeIntervalSchedule, IterationIntervalSchedule
export AbstractOutputGrid, LatLonOutputGrid
export write_output!, initialize_output!, finalize_output!
