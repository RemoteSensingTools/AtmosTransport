# ---------------------------------------------------------------------------
# TOML-based run configuration
#
# build_model_from_config(path) reads a TOML file and constructs a fully
# configured TransportModel ready for run!(). Maps TOML sections to
# concrete types via multiple dispatch.
# ---------------------------------------------------------------------------

using TOML
using Dates
using ..Architectures: CPU, GPU, array_type
using ..Grids: LatitudeLongitudeGrid, CubedSphereGrid
using ..Parameters: load_parameters
using ..Sources: AbstractSource, EdgarSource, CarbonTrackerSource, GFASSource,
                 JenaCarboScopeSource, CATRINESource, load_inventory,
                 load_cams_co2, load_gridfed_fossil_co2, load_edgar_sf6, load_zhang_rn222
using ..Diagnostics: ColumnMeanDiagnostic, SurfaceSliceDiagnostic, RegridDiagnostic,
                     Full3DDiagnostic, MetField2DDiagnostic
using ..Chemistry: NoChemistry, RadioactiveDecay, CompositeChemistry

"""
$(SIGNATURES)

Load a TOML configuration file and return a Dict.

Standard sections:
- `[architecture]` — CPU/GPU, float type
- `[grid]`         — grid type, resolution, vertical levels
- `[met_data]`     — met driver type and paths
- `[tracers]`      — tracer definitions + emission sources
- `[output]`       — output fields, schedule, filename
- `[buffering]`    — single or double buffer
"""
function load_configuration(path::String)
    return TOML.parsefile(path)
end

"""
    build_model_from_config(path::String) → TransportModel
    build_model_from_config(config::Dict) → TransportModel

Build a fully configured `TransportModel` from a TOML file or parsed Dict.

# Example TOML

```toml
[architecture]
use_gpu = true
float_type = "Float32"

[grid]
type = "cubed_sphere"       # or "latlon"
Nc = 720                    # for cubed_sphere
met_source = "geosfp"       # or "era5"

[met_data]
driver = "geosfp_cs"        # or "era5", "preprocessed_latlon"
preprocessed_dir = "~/data/geosfp_cs/preprocessed"
start_date = "2024-06-01"
end_date = "2024-06-05"
dt = 900
met_interval = 3600

[tracers.co2]
emission = "edgar"
edgar_version = "v8.0"
year = 2022

[output]
filename = "output.nc"
interval = 3600
output_grid = "latlon"
Nlon = 720
Nlat = 361

[buffering]
strategy = "single"          # or "double"
```
"""
function build_model_from_config(path::String)
    config = load_configuration(path)
    return build_model_from_config(config)
end

function build_model_from_config(config::Dict)
    # --- Architecture ---
    arch_cfg = get(config, "architecture", Dict())
    use_gpu = get(arch_cfg, "use_gpu", false)
    ft_str  = get(arch_cfg, "float_type", "Float64")
    FT = ft_str == "Float32" ? Float32 : Float64
    arch = use_gpu ? GPU() : CPU()

    # --- Parameters ---
    params = load_parameters(FT)
    pp = params.planet

    # --- Grid ---
    grid_cfg = get(config, "grid", Dict())
    grid_type = get(grid_cfg, "type", "latlon")
    met_source_name = get(grid_cfg, "met_source", "era5")

    met_cfg_default = default_met_config(met_source_name)
    vc = build_vertical_coordinate(met_cfg_default; FT)

    grid = if grid_type == "cubed_sphere" || grid_type == "cs"
        Nc = get(grid_cfg, "Nc", 720)
        CubedSphereGrid(arch; FT, Nc,
            vertical=vc,
            radius=pp.radius, gravity=pp.gravity,
            reference_pressure=pp.reference_surface_pressure)
    else
        sz = get(grid_cfg, "size", [360, 181, 72])
        Nx, Ny, Nz = sz[1], sz[2], sz[3]
        lons = get(grid_cfg, "longitude", [0.0, 360.0])
        lats = get(grid_cfg, "latitude", [-90.0, 90.0])
        LatitudeLongitudeGrid(arch;
            FT, size=(Nx, Ny, Nz),
            longitude=(FT(lons[1]), FT(lons[2])),
            latitude=(FT(lats[1]), FT(lats[2])),
            vertical=vc,
            radius=pp.radius, gravity=pp.gravity,
            reference_pressure=pp.reference_surface_pressure)
    end

    # --- Met driver ---
    met_data_cfg = get(config, "met_data", Dict())
    driver_type = get(met_data_cfg, "driver", "preprocessed_latlon")
    met = _build_met_driver(driver_type, met_data_cfg, met_cfg_default, FT)

    # --- Tracers + sources ---
    tracers_cfg = get(config, "tracers", Dict())
    tracer_names = Symbol.(collect(keys(tracers_cfg)))
    isempty(tracer_names) && (tracer_names = [:co2])

    sources = AbstractSource[]
    for (_, tcfg) in tracers_cfg
        src = _build_emission_source(tcfg, grid, FT)
        src !== nothing && push!(sources, src)
    end

    # Build tracer NamedTuple
    tracers = _build_tracers(tracer_names, grid, arch, FT)

    # --- Output ---
    output_cfg = get(config, "output", Dict())
    writers = _build_output_writers(output_cfg)

    # --- Buffering ---
    buf_cfg = get(config, "buffering", Dict())
    strategy = get(buf_cfg, "strategy", "single")
    buffering = strategy == "double" ? DoubleBuffer() : SingleBuffer()

    # --- Chemistry ---
    chemistry = _build_chemistry(config, tracer_names, FT)

    # --- Initial conditions ---
    ic_cfg = get(config, "initial_conditions", Dict())
    if !isempty(ic_cfg) && haskey(ic_cfg, "file")
        ic_file = expanduser(ic_cfg["file"])
        var_map = Dict{Symbol, String}()
        for (k, v) in ic_cfg
            k == "file" && continue
            k == "time_index" && continue
            var_map[Symbol(k)] = String(v)
        end
        ti = get(ic_cfg, "time_index", 1)
        if isfile(ic_file)
            load_initial_conditions!(tracers, ic_file, grid;
                                      variable_map=var_map, time_index=ti)
        else
            @warn "Initial conditions file not found: $ic_file"
        end
    end

    # --- Build model ---
    Δt = get(met_data_cfg, "dt", 900.0)

    return TransportModel(;
        grid, tracers, met_data=met, Δt,
        sources, output_writers=writers, buffering, chemistry)
end

# =====================================================================
# Internal builder helpers
# =====================================================================

function _build_met_driver(driver_type::String, cfg::Dict, met_cfg_default, ::Type{FT}) where FT
    dt = FT(get(cfg, "dt", 900))
    met_interval = FT(get(cfg, "met_interval", 3600))

    if driver_type == "era5"
        datadirs = get(cfg, "datadirs", String[])
        datadirs = String[expanduser(d) for d in datadirs]
        files = find_era5_files(datadirs)

        level_top = get(cfg, "level_top", 50)
        level_bot = get(cfg, "level_bot", 137)
        A_full, B_full = load_vertical_coefficients(met_cfg_default; FT)
        A_coeff = A_full[level_top:level_bot+1]
        B_coeff = B_full[level_top:level_bot+1]

        return ERA5MetDriver(; FT, files, A_coeff, B_coeff,
                               met_interval, dt, level_top, level_bot)

    elseif driver_type == "preprocessed_latlon"
        dir = expanduser(get(cfg, "directory", ""))
        ft_tag = FT == Float32 ? "float32" : "float64"
        files = if !isempty(dir) && isdir(dir)
            find_massflux_shards(dir, ft_tag)
        else
            f = expanduser(get(cfg, "file", ""))
            isempty(f) ? String[] : [f]
        end
        return PreprocessedLatLonMetDriver(; FT, files, dt)

    elseif driver_type == "geosfp_cs"
        preproc_dir = expanduser(get(cfg, "preprocessed_dir", ""))
        start_date = Date(get(cfg, "start_date", "2024-06-01"))
        end_date   = Date(get(cfg, "end_date", "2024-06-05"))
        Hp = get(cfg, "Hp", 3)

        nc_dir = expanduser(get(cfg, "netcdf_dir", ""))
        nc_files = if !isempty(nc_dir) && isdir(nc_dir)
            find_geosfp_cs_files(nc_dir, start_date, end_date)
        else
            String[]
        end

        return GEOSFPCubedSphereMetDriver(; FT,
            preprocessed_dir=preproc_dir,
            netcdf_files=nc_files,
            start_date, end_date, dt, met_interval, Hp)
    else
        error("Unknown met driver type: $driver_type. " *
              "Use 'era5', 'preprocessed_latlon', or 'geosfp_cs'.")
    end
end

function _build_emission_source(tcfg::Dict, grid, ::Type{FT}) where FT
    emission = get(tcfg, "emission", "none")
    emission == "none" && return nothing

    year = get(tcfg, "year", 2022)

    if emission == "edgar"
        version  = get(tcfg, "edgar_version", "v8.0")
        filepath = expanduser(get(tcfg, "edgar_file", ""))
        src = EdgarSource(; version, filepath)
        return load_inventory(src, grid; year, file=filepath)
    elseif emission == "carbontracker"
        filepath = expanduser(get(tcfg, "file", ""))
        src = CarbonTrackerSource(; filepath)
        return load_inventory(src, grid; year, file=filepath)
    elseif emission == "gfas"
        filepath = expanduser(get(tcfg, "file", ""))
        src = GFASSource(; filepath)
        return load_inventory(src, grid; year, file=filepath)
    elseif emission == "jena"
        filepath = expanduser(get(tcfg, "file", ""))
        src = JenaCarboScopeSource(; filepath)
        return load_inventory(src, grid; year, file=filepath)

    # CATRINE-specific emission types
    elseif emission == "cams_co2" || emission == "catrine_co2"
        filepath = expanduser(get(tcfg, "file", ""))
        species = Symbol(get(tcfg, "species", "co2"))
        return load_cams_co2(filepath, grid; year, species)
    elseif emission == "gridfed" || emission == "gridfed_fossil_co2"
        filepath = expanduser(get(tcfg, "file", ""))
        species = Symbol(get(tcfg, "species", "fossil_co2"))
        return load_gridfed_fossil_co2(filepath, grid; year, species)
    elseif emission == "edgar_sf6"
        filepath = expanduser(get(tcfg, "file", ""))
        noaa_file = expanduser(get(tcfg, "noaa_growth_file", ""))
        scale_year = get(tcfg, "scale_year", year)
        return load_edgar_sf6(filepath, grid; year, noaa_growth_file=noaa_file,
                               scale_year)
    elseif emission == "zhang_rn222"
        dirpath = expanduser(get(tcfg, "dir", get(tcfg, "file", "")))
        return load_zhang_rn222(dirpath, grid)
    elseif emission == "catrine"
        dataset = get(tcfg, "dataset", "")
        filepath = expanduser(get(tcfg, "file", get(tcfg, "dir", "")))
        src = CATRINESource(; dataset, filepath)
        return load_inventory(src, grid; year)
    else
        @warn "Unknown emission type: $emission"
        return nothing
    end
end

function _build_tracers(names::Vector{Symbol}, grid::LatitudeLongitudeGrid{FT},
                        arch, ::Type{FT}) where FT
    AT = array_type(arch)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    pairs = [n => AT(zeros(FT, Nx, Ny, Nz)) for n in names]
    return NamedTuple(pairs)
end

function _build_tracers(names::Vector{Symbol}, grid::CubedSphereGrid{FT},
                        arch, ::Type{FT}) where FT
    # CS tracers are panel tuples, allocated in _run_loop!
    # Return placeholder NamedTuple
    AT = array_type(arch)
    pairs = [n => AT(zeros(FT, 1)) for n in names]
    return NamedTuple(pairs)
end

function _build_output_writers(cfg::Dict)
    filename = get(cfg, "filename", "")
    isempty(filename) && return AbstractOutputWriter[]

    interval = get(cfg, "interval", 3600.0)
    schedule = TimeIntervalSchedule(Float64(interval))

    og = if get(cfg, "output_grid", "") == "latlon"
        Nlon = get(cfg, "Nlon", 720)
        Nlat = get(cfg, "Nlat", 361)
        LatLonOutputGrid(Nlon, Nlat)
    else
        nothing
    end

    fields_cfg = get(cfg, "fields", Dict())
    fields = Dict{Symbol, Any}()
    for (name, ftype) in fields_cfg
        sym = Symbol(name)
        if ftype == "column_mean"
            fields[sym] = ColumnMeanDiagnostic(sym)
        elseif ftype == "surface_slice"
            fields[sym] = SurfaceSliceDiagnostic(sym)
        elseif ftype == "regrid"
            Nlon_f = get(cfg, "Nlon", 720)
            Nlat_f = get(cfg, "Nlat", 361)
            fields[sym] = RegridDiagnostic(sym; Nlon=Nlon_f, Nlat=Nlat_f)
        elseif ftype == "full_3d"
            fields[sym] = Full3DDiagnostic(sym)
        elseif ftype == "surface_pressure"
            fields[sym] = MetField2DDiagnostic(:surface_pressure)
        elseif ftype == "pbl_height"
            fields[sym] = MetField2DDiagnostic(:pbl_height)
        elseif ftype == "tropopause_height"
            fields[sym] = MetField2DDiagnostic(:tropopause_height)
        elseif startswith(string(ftype), "met_")
            fields[sym] = MetField2DDiagnostic(Symbol(ftype[5:end]))
        end
    end

    writer = NetCDFOutputWriter(expanduser(filename), fields, schedule;
                                output_grid=og)
    return AbstractOutputWriter[writer]
end

"""Build chemistry scheme from config. Supports per-tracer decay and composite."""
function _build_chemistry(config::Dict, tracer_names::Vector{Symbol}, ::Type{FT}) where FT
    chem_cfg = get(config, "chemistry", Dict())
    isempty(chem_cfg) && return NoChemistry()

    # Global chemistry type
    global_type = get(chem_cfg, "type", "none")
    global_type == "none" && return NoChemistry()

    if global_type == "decay"
        species = Symbol(get(chem_cfg, "species", "rn222"))
        half_life = FT(get(chem_cfg, "half_life", 330350.4))
        return RadioactiveDecay(species, half_life)
    elseif global_type == "composite"
        # Build from per-tracer chemistry sections
        schemes = _build_per_tracer_chemistry(config, tracer_names, FT)
        return isempty(schemes) ? NoChemistry() : CompositeChemistry(tuple(schemes...))
    end

    # Check per-tracer chemistry in [tracers.xxx] sections
    schemes = _build_per_tracer_chemistry(config, tracer_names, FT)
    if length(schemes) == 1
        return schemes[1]
    elseif length(schemes) > 1
        return CompositeChemistry(tuple(schemes...))
    end

    return NoChemistry()
end

function _build_per_tracer_chemistry(config::Dict, tracer_names::Vector{Symbol}, ::Type{FT}) where FT
    tracers_cfg = get(config, "tracers", Dict())
    schemes = []
    for name in tracer_names
        tcfg = get(tracers_cfg, string(name), Dict())
        chem = get(tcfg, "chemistry", "none")
        if chem == "decay"
            half_life = FT(get(tcfg, "half_life", 330350.4))
            push!(schemes, RadioactiveDecay(name, half_life))
        end
    end
    return schemes
end

export load_configuration, build_model_from_config
