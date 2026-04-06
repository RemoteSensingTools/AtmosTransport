# ---------------------------------------------------------------------------
# TOML-based run configuration
#
# build_model_from_config(path) reads a TOML file and constructs a fully
# configured TransportModel ready for run!(). Maps TOML sections to
# concrete types via multiple dispatch.
# ---------------------------------------------------------------------------

using TOML
using Dates
using ..Architectures: CPU, GPU, array_type, build_panel_map, is_multi_gpu
using ..Grids: LatitudeLongitudeGrid, CubedSphereGrid, HybridSigmaPressure, grid_size,
               merge_upper_levels, merge_thin_levels, n_levels, set_coord_status!
using ..Parameters: load_parameters
using ..Sources: AbstractSurfaceFlux, compute_areas_from_corners,
                 EdgarSource, CarbonTrackerSource, GFASSource,
                 JenaCarboScopeSource, CATRINESource, load_inventory,
                 load_cams_co2, load_lmdz_co2, load_gridfed_fossil_co2, load_edgar_sf6, load_zhang_rn222,
                 SurfaceFlux, TimeVaryingSurfaceFlux, CombinedFlux, NoFlux
using ..Diagnostics: ColumnMeanDiagnostic, ColumnMassDiagnostic, SurfaceSliceDiagnostic,
                     RegridDiagnostic, ColumnFluxDiagnostic, EmissionFluxDiagnostic,
                     Full3DDiagnostic, MetField2DDiagnostic, SigmaLevelDiagnostic
using ..Chemistry: NoChemistry, RadioactiveDecay, CompositeChemistry
using ..Advection: SlopesAdvection, PPMAdvection, PratherAdvection
using ..Diffusion: BoundaryLayerDiffusion, NoDiffusion, PBLDiffusion, NonLocalPBLDiffusion
using ..Convection: TiedtkeConvection, NoConvection, RASConvection, TM5MatrixConvection

# Module-level storage for deferred CS initial conditions
const _PENDING_IC = Ref(PendingInitialConditions())

"""
    get_pending_ic() → PendingInitialConditions

Retrieve and clear any pending initial conditions (for CS run loops).
"""
function get_pending_ic()
    pic = _PENDING_IC[]
    _PENDING_IC[] = PendingInitialConditions()
    return pic
end

"""
    _parse_initial_conditions(ic_cfg::Dict) → PendingInitialConditions

Parse IC config into entries (file + variable_map + time_index).
Also stores uniform IC entries (uniform_value) for deferred application.
"""
function _parse_initial_conditions(ic_cfg::Dict)
    pic = PendingInitialConditions()
    isempty(ic_cfg) && return pic

    if haskey(ic_cfg, "file")
        # Legacy format: single file with variable_map
        ic_file = expanduser(ic_cfg["file"])
        var_map = Dict{Symbol, String}()
        for (k, v) in ic_cfg
            k == "file" && continue
            k == "time_index" && continue
            var_map[Symbol(k)] = String(v)
        end
        ti = get(ic_cfg, "time_index", 1)
        push!(pic.entries, (file=ic_file, variable_map=var_map, time_index=ti))
    else
        # Per-tracer format
        for (tracer_key, tracer_ic) in ic_cfg
            tracer_ic isa Dict || continue

            # Uniform IC: constant mixing ratio everywhere
            if haskey(tracer_ic, "uniform_value")
                val = Float64(tracer_ic["uniform_value"])
                _store_uniform_ic(UniformICData(Symbol(tracer_key), val))
                @info "IC for $(tracer_key): uniform mixing ratio = $val"
                continue
            end

            haskey(tracer_ic, "file") || continue
            ic_file = expanduser(tracer_ic["file"])
            nc_var  = get(tracer_ic, "variable", uppercase(String(tracer_key)))
            ti      = get(tracer_ic, "time_index", 1)
            var_map = Dict{Symbol, String}(Symbol(tracer_key) => nc_var)
            push!(pic.entries, (file=ic_file, variable_map=var_map, time_index=ti))
        end
    end
    return pic
end

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

    # --- Multi-GPU panel map ---
    panel_map = build_panel_map(config)
    is_multi_gpu(panel_map) &&
        @info "Multi-GPU: $(panel_map.n_gpus) GPUs, panels=$(panel_map.panel_to_gpu)"

    # --- Parameters ---
    params = load_parameters(FT)
    pp = params.planet

    # --- Grid ---
    grid_cfg = get(config, "grid", Dict())
    grid_type = get(grid_cfg, "type", "latlon")
    met_source_name = get(grid_cfg, "met_source", "era5")

    met_cfg_default = default_met_config(met_source_name)
    vc = build_vertical_coordinate(met_cfg_default; FT)

    # Optional: merge thin levels
    merge_map = nothing
    merge_Pa = get(grid_cfg, "merge_levels_above_Pa", nothing)
    merge_min_dp = get(grid_cfg, "merge_min_thickness_Pa", nothing)
    if merge_min_dp !== nothing
        # General merge: thin levels from both top and bottom
        vc, merge_map = merge_thin_levels(vc; min_thickness_Pa=FT(merge_min_dp))
    elseif merge_Pa !== nothing
        # Legacy: merge upper-atmosphere levels only
        vc, merge_map = merge_upper_levels(vc, FT(merge_Pa))
    end

    grid = if grid_type == "cubed_sphere" || grid_type == "cs"
        Nc = get(grid_cfg, "Nc", 720)
        CubedSphereGrid(arch; FT, Nc,
            vertical=vc,
            radius=pp.radius, gravity=pp.gravity,
            reference_pressure=pp.reference_surface_pressure,
            panel_map)
    else
        sz = get(grid_cfg, "size", [360, 181, 72])
        Nx, Ny = sz[1], sz[2]
        # Use merged level count when level merging is active;
        # otherwise fall back to config's size[3]
        Nz = merge_map !== nothing ? n_levels(vc) : sz[3]
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
    met = _build_met_driver(driver_type, met_data_cfg, met_cfg_default, FT; merge_map)

    # --- Auto-detect grid from v2 binary header ---
    # If the met driver has embedded A/B coefficients (v2 binary), override the
    # vertical coordinate and rebuild the grid. This makes pre-merged binaries
    # fully self-describing — no manual size or merge config needed in TOML.
    # Also auto-detect Nx/Ny from the driver (binary header is authoritative).
    if grid isa LatitudeLongitudeGrid && met isa PreprocessedLatLonMetDriver && merge_map === nothing
        _need_rebuild = false
        gs = grid_size(grid)
        Nx_new, Ny_new, Nz_new = gs.Nx, gs.Ny, gs.Nz
        vc_new = vc

        # Check if Nx/Ny from driver differ from config defaults
        if met.Nx != gs.Nx || met.Ny != gs.Ny
            Nx_new, Ny_new = met.Nx, met.Ny
            _need_rebuild = true
        end

        # Check for embedded vertical coordinate (v2 binary)
        _vc_embedded = embedded_vertical_coordinate(met)
        if _vc_embedded !== nothing
            vc_new = HybridSigmaPressure(FT.(_vc_embedded.A_ifc), FT.(_vc_embedded.B_ifc))
            Nz_new = n_levels(vc_new)
            _need_rebuild = _need_rebuild || (Nz_new != gs.Nz)
        end

        if _need_rebuild
            @info "Auto-detected grid from binary: $(Nx_new)×$(Ny_new)×$(Nz_new) (was $(gs.Nx)×$(gs.Ny)×$(gs.Nz))"
            lons = get(grid_cfg, "longitude", [0.0, 360.0])
            lats = get(grid_cfg, "latitude", [-90.0, 90.0])
            grid = LatitudeLongitudeGrid(arch;
                FT, size=(Nx_new, Ny_new, Nz_new),
                longitude=(FT(lons[1]), FT(lons[2])),
                latitude=(FT(lats[1]), FT(lats[2])),
                vertical=vc_new,
                radius=pp.radius, gravity=pp.gravity,
                reference_pressure=pp.reference_surface_pressure)
        end
    end

    # --- Load authoritative GMAO coordinates for CS grids ---
    if grid isa CubedSphereGrid
        coord_file = _find_coord_file(config)
        if coord_file !== nothing
            _load_gmao_coordinates!(grid, coord_file)
        else
            @warn "No coordinate file found — using gnomonic coordinates. " *
                  "Set [met_data].coord_file or netcdf_dir for correct GMAO coordinates."
        end
    end

    # --- Tracers + sources ---
    tracers_cfg = get(config, "tracers", Dict())
    tracer_names = Symbol.(collect(keys(tracers_cfg)))
    isempty(tracer_names) && (tracer_names = [:co2])

    sources = AbstractSurfaceFlux[]
    for (_, tcfg) in tracers_cfg
        src = _build_emission_source(tcfg, grid, FT; met_driver=met,
                                     sim_start_date=start_date(met))
        src !== nothing && push!(sources, src)
    end

    # Print structured summary of all emission sources
    _log_emission_summary(sources)

    # Build tracer NamedTuple
    tracers = _build_tracers(tracer_names, grid, arch, FT)

    # --- Output ---
    output_cfg = get(config, "output", Dict())
    sim_start_date = start_date(met)
    writers = _build_output_writers(output_cfg; start_date=sim_start_date)

    # --- Buffering ---
    # Reference via parent module (Models is loaded after IO)
    _Models = parentmodule(@__MODULE__).Models
    buf_cfg = get(config, "buffering", Dict())
    strategy = get(buf_cfg, "strategy", "single")
    buffering = strategy == "double" ? _Models.DoubleBuffer() : _Models.SingleBuffer()

    # --- Chemistry ---
    chemistry = _build_chemistry(config, tracer_names, FT)

    # --- Advection ---
    advection = _build_advection(config, FT)

    # --- Diffusion ---
    diffusion = _build_diffusion(config, FT)

    # --- Convection ---
    convection = _build_convection(config, FT)

    # --- Initial conditions ---
    ic_cfg = get(config, "initial_conditions", Dict())
    pending_ic = _parse_initial_conditions(ic_cfg)

    if grid isa LatitudeLongitudeGrid
        # Apply immediately for LL grids (tracers are properly allocated)
        apply_pending_ic!(tracers, pending_ic, grid)
    else
        # CS grids: defer to run loop (tracers are placeholders here)
        _PENDING_IC[] = pending_ic
    end

    # --- Build model ---
    Δt = get(met_data_cfg, "dt", 900.0)

    # --- Run metadata for provenance tracking ---
    adv_cfg = get(config, "advection", Dict())
    mass_fixer = get(adv_cfg, "mass_fixer", true)
    mass_fixer_tracers = get(adv_cfg, "mass_fixer_tracers", String[])  # empty = all tracers
    dry_correction = get(adv_cfg, "dry_correction", true)
    remap_pressure_fix = get(adv_cfg, "remap_pressure_fix", true)
    # Transport basis: "dry" (default when dry_correction=true) or "moist" (GCHP-faithful)
    pressure_basis = get(adv_cfg, "pressure_basis", dry_correction ? "dry" : "moist")
    # Git provenance (best-effort, non-fatal)
    _repo_dir = dirname(dirname(@__DIR__))  # src/IO/ → src/ → repo root
    git_commit = try
        strip(read(`git -C $_repo_dir rev-parse HEAD`, String))
    catch
        "unknown"
    end
    git_dirty = try
        !isempty(strip(read(`git -C $_repo_dir status --porcelain`, String)))
    catch
        false
    end

    metadata = Dict{String, Any}(
        "config"    => config,
        "user"      => get(ENV, "USER", "unknown"),
        "hostname"  => gethostname(),
        "julia_version" => string(VERSION),
        "created"   => string(Dates.now()),
        "git_commit" => git_dirty ? git_commit * "-dirty" : git_commit,
        "mass_fixer" => mass_fixer,
        "mass_fixer_tracers" => mass_fixer_tracers,
        "dry_correction" => dry_correction,
        "pressure_basis" => pressure_basis,
        "remap_pressure_fix" => remap_pressure_fix,
        "panel_map" => panel_map,
    )

    @info "Git: $(metadata["git_commit"])"

    return _Models.TransportModel(;
        grid, tracers, met_data=met, Δt,
        sources, output_writers=writers, buffering, advection, chemistry, diffusion, convection,
        metadata)
end

# =====================================================================
# Internal builder helpers
# =====================================================================

"""
    _resolve_data_path(raw_path) → String

Resolve a configuration path with `~` home directory expansion.
"""
function _resolve_data_path(raw_path::String)
    isempty(raw_path) && return ""
    return expanduser(raw_path)
end

function _build_met_driver(driver_type::String, cfg::Dict, met_cfg_default, ::Type{FT};
                           merge_map::Union{Nothing, Vector{Int}} = nothing) where FT
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
        dir = _resolve_data_path(get(cfg, "directory", ""))
        ft_tag = FT == Float32 ? "float32" : "float64"
        files = if !isempty(dir) && isdir(dir)
            find_massflux_shards(dir, ft_tag)
        else
            f = _resolve_data_path(get(cfg, "file", ""))
            isempty(f) ? String[] : [f]
        end
        max_win_cfg = get(cfg, "max_windows", nothing)
        max_windows = max_win_cfg !== nothing ? Int(max_win_cfg) : nothing
        qv_dir_cfg = expanduser(get(cfg, "qv_directory", ""))
        disable_qv_cfg = Bool(get(cfg, "disable_qv", false))
        # Load native-grid A/B coefficients for QV merging (137→merged levels)
        native_A = Float64[]
        native_B = Float64[]
        try
            A_full, B_full = load_vertical_coefficients(met_cfg_default; FT=Float64)
            native_A = A_full
            native_B = B_full
        catch; end
        return PreprocessedLatLonMetDriver(; FT, files, dt, merge_map, max_windows,
                                            qv_dir=qv_dir_cfg,
                                            disable_qv=disable_qv_cfg,
                                            native_A_ifc=native_A,
                                            native_B_ifc=native_B)

    elseif driver_type == "geosfp_cs"
        preproc_dir = _resolve_data_path(get(cfg, "preprocessed_dir", ""))
        start_date = Date(get(cfg, "start_date", "2024-06-01"))
        end_date   = Date(get(cfg, "end_date", "2024-06-05"))
        Hp = get(cfg, "Hp", 3)
        product = get(cfg, "product", "geosfp_c720")

        nc_dir = expanduser(get(cfg, "netcdf_dir", ""))
        nc_files = if !isempty(nc_dir) && isdir(nc_dir)
            find_geosfp_cs_files(nc_dir, start_date, end_date; product)
        else
            String[]
        end

        coord_file_cfg      = expanduser(get(cfg, "coord_file", ""))
        mass_flux_dt        = FT(get(cfg, "mass_flux_dt", met_interval))
        surface_data_dir     = expanduser(get(cfg, "surface_data_dir", ""))
        surface_data_bin_dir = expanduser(get(cfg, "surface_data_bin_dir", ""))
        surface_data_ll_dir  = expanduser(get(cfg, "surface_data_ll_dir", ""))
        verbose              = get(cfg, "verbose", false)

        return GEOSFPCubedSphereMetDriver(; FT,
            preprocessed_dir=preproc_dir,
            netcdf_files=nc_files,
            coord_file=coord_file_cfg,
            start_date, end_date, dt, met_interval, Hp, merge_map,
            mass_flux_dt, surface_data_dir, surface_data_bin_dir,
            surface_data_ll_dir, verbose)
    else
        error("Unknown met driver type: $driver_type. " *
              "Use 'era5', 'preprocessed_latlon', or 'geosfp_cs'.")
    end
end

function _build_emission_source(tcfg::Dict, grid, ::Type{FT};
                                 met_driver=nothing,
                                 sim_start_date::Date=Date(2022,1,1)) where FT
    emission = get(tcfg, "emission", "none")
    emission == "none" && return nothing

    year = get(tcfg, "year", 2022)

    # Load GEOS-FP file coords for CS emission regridding (on-the-fly fallback)
    file_lons, file_lats = _get_cs_file_coords_from_driver(met_driver)

    if emission == "edgar"
        version  = get(tcfg, "edgar_version", "v8.0")
        filepath = expanduser(get(tcfg, "edgar_file", ""))
        species  = Symbol(get(tcfg, "species", "co2"))
        src = EdgarSource(; version, filepath, species)
        if grid isa CubedSphereGrid
            return load_inventory(src, grid; year, file=filepath, binary_file=filepath,
                                  file_lons, file_lats)
        else
            return load_inventory(src, grid; year, file=filepath)
        end
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
    elseif emission == "lmdz_co2"
        dirpath = expanduser(get(tcfg, "file", get(tcfg, "dir", "")))
        species = Symbol(get(tcfg, "species", "co2"))
        flux_var = get(tcfg, "flux_var", "flux_apos")
        sd = Date(get(tcfg, "start_date", "$(year)-01-01"))
        ed = Date(get(tcfg, "end_date",   "$(year)-12-31"))
        return load_lmdz_co2(dirpath, grid; start_date=sd, end_date=ed,
                              species, flux_var)
    elseif emission == "gridfed" || emission == "gridfed_fossil_co2"
        filepath = expanduser(get(tcfg, "file", get(tcfg, "dir", "")))
        species = Symbol(get(tcfg, "species", "fossil_co2"))
        return load_gridfed_fossil_co2(filepath, grid; year, species,
                                       start_date=sim_start_date)
    elseif emission == "edgar_sf6"
        filepath = expanduser(get(tcfg, "file", ""))
        noaa_file = expanduser(get(tcfg, "noaa_growth_file", ""))
        scale_year = get(tcfg, "scale_year", year)
        return load_edgar_sf6(filepath, grid; year, noaa_growth_file=noaa_file,
                               scale_year)
    elseif emission == "zhang_rn222"
        dirpath = expanduser(get(tcfg, "file", get(tcfg, "dir", "")))
        return load_zhang_rn222(dirpath, grid; start_date=sim_start_date)
    elseif emission == "catrine"
        dataset = get(tcfg, "dataset", "")
        filepath = expanduser(get(tcfg, "file", get(tcfg, "dir", "")))
        src = CATRINESource(; dataset, filepath)
        return load_inventory(src, grid; year)
    elseif emission == "prebuild" || emission == "prebuilt"
        filepath = expanduser(get(tcfg, "file", ""))
        isfile(filepath) || error("Prebuilt emission file not found: $filepath")
        return _load_prebuilt_emission(filepath, FT)
    elseif emission == "uniform_surface"
        rate    = FT(get(tcfg, "rate", 1.0e-8))    # kg/m²/s
        species = Symbol(get(tcfg, "species", "co2"))
        label   = "Uniform surface $(uppercase(string(species))) $(rate) kg/m²/s"
        if grid isa CubedSphereGrid
            flux_panels = ntuple(_ -> fill(rate, grid.Nc, grid.Nc), 6)
            return SurfaceFlux(flux_panels, species, label)
        else
            flux = fill(rate, grid.Nx, grid.Ny)
            return SurfaceFlux(flux, species, label)
        end
    else
        @warn "Unknown emission type: $emission"
        return nothing
    end
end

"""Load a prebuilt emission binary (4096-byte TOML header + Float64 time + Float32 flux)."""
function _load_prebuilt_emission(filepath::String, ::Type{FT}) where FT
    open(filepath, "r") do io
        header_block = read(io, 4096)
        null_pos = findfirst(==(0x00), header_block)
        header_str = String(header_block[1:null_pos-1])
        header = TOML.parse(header_str)

        Nx = Int(header["Nx"])
        Ny = Int(header["Ny"])
        Nt = Int(header["Nt"])

        time_hours = Vector{Float64}(undef, Nt)
        read!(io, time_hours)

        flux_f32 = Array{Float32}(undef, Nx, Ny, Nt)
        read!(io, flux_f32)

        species = Symbol(header["species"])
        molar_mass = FT(header["molar_mass"])
        label = get(header, "label", "prebuilt")
        cyclic = get(header, "cyclic", false)

        flux = FT.(flux_f32)
        @info "Loaded prebuilt emission: $species ($Nx×$Ny×$Nt, $(label))"
        return TimeVaryingSurfaceFlux(flux, time_hours, species;
                                       label=label, molar_mass=molar_mass, cyclic=cyclic)
    end
end

"""Print a structured summary of all loaded emission sources."""
function _log_emission_summary(sources::Vector{AbstractSurfaceFlux})
    isempty(sources) && return

    lines = String[]
    push!(lines, "")
    push!(lines, "  Emission Sources")
    push!(lines, "  " * "─"^62)

    for src in sources
        if src isa TimeVaryingSurfaceFlux
            Nt = length(src.time_hours)
            t0 = src.time_hours[1]
            t1 = src.time_hours[end]
            push!(lines, "  $(rpad(src.species, 14)) │ $(rpad(src.label, 30)) │ $(Nt) snapshots")
            push!(lines, "  $(rpad("", 14)) │ time_hours: [$(round(t0, digits=1)) … $(round(t1, digits=1))] h")
        elseif src isa SurfaceFlux
            push!(lines, "  $(rpad(src.species, 14)) │ $(rpad(src.label, 30)) │ static")
        elseif src isa CombinedFlux
            push!(lines, "  $(rpad("combined", 14)) │ $(rpad(src.label, 30)) │ $(length(src.components)) components")
        elseif src isa NoFlux
            push!(lines, "  $(rpad("—", 14)) │ $(rpad("NoFlux", 30)) │ —")
        end
    end

    push!(lines, "  " * "─"^62)
    @info join(lines, "\n")
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

function _build_output_writers(cfg::Dict; start_date::Date=Date(2000,1,1))
    filename = get(cfg, "filename", "")
    isempty(filename) && return AbstractOutputWriter[]

    interval = get(cfg, "interval", 3600.0)
    schedule = TimeIntervalSchedule(Float64(interval))

    og = if get(cfg, "output_grid", "") == "latlon"
        Nlon = get(cfg, "Nlon", 720)
        Nlat = get(cfg, "Nlat", 361)
        lon_range_arr = get(cfg, "lon_range", [-180.0, 180.0])
        lat_range_arr = get(cfg, "lat_range", [-90.0, 90.0])
        LatLonOutputGrid(; Nlon, Nlat,
            lon_range=(lon_range_arr[1], lon_range_arr[2]),
            lat_range=(lat_range_arr[1], lat_range_arr[2]))
    else
        nothing
    end

    fields_cfg = get(cfg, "fields", Dict())
    fields = Dict{Symbol, Any}()
    for (name, fval) in fields_cfg
        sym = Symbol(name)
        # Support both string ("column_mean") and dict ({type="surface_slice", species="co2"})
        if fval isa Dict
            ftype   = get(fval, "type", "column_mean")
            species = Symbol(get(fval, "species", name))
        else
            ftype   = string(fval)
            species = sym
        end
        if ftype == "column_mean"
            fields[sym] = ColumnMeanDiagnostic(species)
        elseif ftype == "column_mass"
            fields[sym] = ColumnMassDiagnostic(species)
        elseif ftype == "surface_slice"
            fields[sym] = SurfaceSliceDiagnostic(species)
        elseif ftype == "regrid"
            Nlon_f = get(cfg, "Nlon", 720)
            Nlat_f = get(cfg, "Nlat", 361)
            fields[sym] = RegridDiagnostic(species; Nlon=Nlon_f, Nlat=Nlat_f)
        elseif ftype == "sigma_level"
            sigma = get(fval, "sigma", 0.8)
            fields[sym] = SigmaLevelDiagnostic(species, Float64(sigma))
        elseif ftype == "full_3d"
            fields[sym] = Full3DDiagnostic(species)
        elseif ftype == "column_flux"
            direction = Symbol(get(fval, "direction", "east"))
            direction ∈ (:east, :north) || @warn "Unknown flux direction '$direction', expected :east or :north"
            fields[sym] = ColumnFluxDiagnostic(species, direction)
        elseif ftype == "surface_pressure"
            fields[sym] = MetField2DDiagnostic(:surface_pressure)
        elseif ftype == "pbl_height"
            fields[sym] = MetField2DDiagnostic(:pbl_height)
        elseif ftype == "tropopause_height"
            fields[sym] = MetField2DDiagnostic(:tropopause_height)
        elseif ftype == "emission_flux"
            fields[sym] = EmissionFluxDiagnostic(species)
        elseif startswith(string(ftype), "met_")
            fields[sym] = MetField2DDiagnostic(Symbol(ftype[5:end]))
        end
    end

    format = get(cfg, "format", "netcdf")
    expanded_filename = expanduser(filename)

    deflate_level = get(cfg, "deflate_level", 0)
    digits_cfg    = get(cfg, "digits", nothing)
    digits = digits_cfg === nothing ? nothing : Int(digits_cfg)

    writer = if format == "binary"
        auto_convert = get(cfg, "auto_convert", false)
        bin_path = if endswith(expanded_filename, ".bin")
            expanded_filename
        else
            replace(expanded_filename, r"\.\w+$" => ".bin")
        end
        # Ensure we got a .bin extension even if no extension was present
        if !endswith(bin_path, ".bin")
            bin_path = expanded_filename * ".bin"
        end
        split_mode = Symbol(get(cfg, "split", "none"))
        BinaryOutputWriter(bin_path, fields, schedule;
                           output_grid=og, auto_convert, start_date,
                           split=split_mode)
    else
        NetCDFOutputWriter(expanded_filename, fields, schedule;
                           output_grid=og, deflate_level, digits, start_date)
    end

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
        return RadioactiveDecay(; species, half_life, FT)
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
            push!(schemes, RadioactiveDecay(; species=name, half_life, FT))
        end
    end
    return schemes
end

"""
    _get_cs_file_coords_from_driver(met_driver)

Extract GEOS-FP file coordinates from a met driver (if available).
Returns `(lons, lats)` as `(Nc×Nc×6)` arrays, or `(nothing, nothing)`.
"""
function _get_cs_file_coords_from_driver(met_driver)
    met_driver === nothing && return nothing, nothing
    # Use coord_file if available (works for both binary and netcdf modes)
    if hasproperty(met_driver, :coord_file) && !isempty(met_driver.coord_file) &&
       isfile(met_driver.coord_file)
        lons, lats, _, _ = read_geosfp_cs_grid_info(met_driver.coord_file)
        return lons, lats
    end
    # Fallback: try first data file in netcdf mode
    if hasproperty(met_driver, :mode) && met_driver.mode == :netcdf &&
       hasproperty(met_driver, :files) && !isempty(met_driver.files)
        lons, lats, _, _ = read_geosfp_cs_grid_info(met_driver.files[1])
        return lons, lats
    end
    return nothing, nothing
end

"""
    _find_coord_file(config::Dict) → Union{String, Nothing}

Find a GEOS cubed-sphere coordinate reference file from the configuration.
Checks `[met_data].coord_file` first, then falls back to the first `.nc` file
in `[met_data].netcdf_dir`.
"""
function _find_coord_file(config::Dict)
    met_cfg = get(config, "met_data", Dict())

    # Explicit coord_file
    cf = expanduser(get(met_cfg, "coord_file", ""))
    !isempty(cf) && isfile(cf) && return cf

    # Fallback: first .nc file in netcdf_dir
    nc_dir = expanduser(get(met_cfg, "netcdf_dir", ""))
    if !isempty(nc_dir) && isdir(nc_dir)
        for d in sort(readdir(nc_dir))
            sub = joinpath(nc_dir, d)
            isdir(sub) || continue
            for f in readdir(sub)
                if endswith(f, ".nc")
                    return joinpath(sub, f)
                end
            end
        end
        # Maybe .nc files directly in netcdf_dir
        for f in readdir(nc_dir)
            endswith(f, ".nc") && return joinpath(nc_dir, f)
        end
    end

    return nothing
end

"""Spherical quadrilateral area from 4 corner lon/lat pairs (degrees), R in meters."""
function _gmao_cell_area(lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4, R)
    _ll2xyz(lo, la) = (cospi(la/180)*cospi(lo/180), cospi(la/180)*sinpi(lo/180), sinpi(la/180))
    function _tri_area(v1, v2, v3)
        a = acos(clamp(v2[1]*v3[1]+v2[2]*v3[2]+v2[3]*v3[3], -1.0, 1.0))
        b = acos(clamp(v1[1]*v3[1]+v1[2]*v3[2]+v1[3]*v3[3], -1.0, 1.0))
        c = acos(clamp(v1[1]*v2[1]+v1[2]*v2[2]+v1[3]*v2[3], -1.0, 1.0))
        s = (a + b + c) / 2
        4 * atan(sqrt(max(tan(s/2)*tan((s-a)/2)*tan((s-b)/2)*tan((s-c)/2), 0.0)))
    end
    v1, v2 = _ll2xyz(lon1, lat1), _ll2xyz(lon2, lat2)
    v3, v4 = _ll2xyz(lon3, lat3), _ll2xyz(lon4, lat4)
    R^2 * (_tri_area(v1, v2, v3) + _tri_area(v1, v3, v4))
end

"""
    _load_gmao_coordinates!(grid::CubedSphereGrid, filepath::String)

Overwrite the gnomonic-computed cell-center coordinates in `grid` with
the authoritative GMAO coordinates from a GEOS-FP/GEOS-IT NetCDF file
or gridspec file. When corner coordinates are available, also overwrites
cell areas — this eliminates up to 44% per-cell area errors that cause
spurious oscillations in the pressure-fixer cm computation.
"""
function _load_gmao_coordinates!(grid::CubedSphereGrid, filepath::String)
    lons, lats, clons, clats = read_geosfp_cs_grid_info(filepath)
    Nc = grid.Nc
    @assert size(lons) == (Nc, Nc, 6) "Coordinate file Nc=$(size(lons,1)) doesn't match grid Nc=$Nc"
    for p in 1:6
        for j in 1:Nc, i in 1:Nc
            grid.λᶜ[p][i, j] = lons[i, j, p]
            grid.φᶜ[p][i, j] = lats[i, j, p]
        end
    end

    # Overwrite analytical gnomonic areas with GMAO corner-based areas
    # when corner coordinates are available. This eliminates up to 44%
    # per-cell area errors from the gnomonic vs GMAO grid mismatch,
    # which cause spurious cm residuals in the pressure fixer → oscillations.
    if clons !== nothing && clats !== nothing
        R = Float64(grid.radius)
        FTA = eltype(grid.Aᶜ[1])
        for p in 1:6
            for j in 1:Nc, i in 1:Nc
                grid.Aᶜ[p][i, j] = FTA(_gmao_cell_area(
                    clons[i,j,p], clats[i,j,p], clons[i+1,j,p], clats[i+1,j,p],
                    clons[i+1,j+1,p], clats[i+1,j+1,p], clons[i,j+1,p], clats[i,j+1,p], R))
            end
        end
        @info "Loaded GMAO coordinates + cell areas from $(basename(filepath))"
    else
        @info "Loaded GMAO coordinates from $(basename(filepath)) (no corner coords — using gnomonic areas)"
    end

    # Track that this grid has GMAO coordinates
    set_coord_status!(grid, :gmao, filepath)

    return nothing
end

"""
Build diffusion scheme from `[diffusion]` TOML section.

```toml
[diffusion]
type = "boundary_layer"    # or "none" (default)
Kz_max = 100.0             # Pa²/s  maximum diffusivity
H_scale = 8.0              # e-folding depth in levels from surface
```
"""
function _build_diffusion(config::Dict, ::Type{FT}) where FT
    diff_cfg = get(config, "diffusion", Dict())
    isempty(diff_cfg) && return nothing

    dtype = get(diff_cfg, "type", "none")
    dtype == "none" && return nothing

    if dtype == "boundary_layer"
        Kz_max  = FT(get(diff_cfg, "Kz_max", 100.0))
        H_scale = FT(get(diff_cfg, "H_scale", 8.0))
        return BoundaryLayerDiffusion(Kz_max, H_scale)
    elseif dtype == "pbl"
        β_h    = FT(get(diff_cfg, "beta_h", 15.0))
        Kz_bg  = FT(get(diff_cfg, "Kz_bg", 0.1))
        Kz_min = FT(get(diff_cfg, "Kz_min", 0.01))
        Kz_max = FT(get(diff_cfg, "Kz_max", 500.0))
        return PBLDiffusion(β_h, Kz_bg, Kz_min, Kz_max)
    elseif dtype == "nonlocal_pbl"
        β_h    = FT(get(diff_cfg, "beta_h", 15.0))
        Kz_bg  = FT(get(diff_cfg, "Kz_bg", 0.1))
        Kz_min = FT(get(diff_cfg, "Kz_min", 0.01))
        Kz_max = FT(get(diff_cfg, "Kz_max", 500.0))
        fak    = FT(get(diff_cfg, "fak", 8.5))
        sffrac = FT(get(diff_cfg, "sffrac", 0.1))
        return NonLocalPBLDiffusion(β_h, Kz_bg, Kz_min, Kz_max, fak, sffrac)
    else
        @warn "Unknown diffusion type: $dtype — using no diffusion"
        return nothing
    end
end

"""
Build convection scheme from `[convection]` TOML section.

```toml
[convection]
type = "tiedtke"    # or "none" (default)
```
"""
function _build_convection(config::Dict, ::Type{FT}) where FT
    conv_cfg = get(config, "convection", Dict())
    isempty(conv_cfg) && return nothing

    ctype = get(conv_cfg, "type", "none")
    ctype == "none" && return nothing

    if ctype == "tiedtke"
        return TiedtkeConvection()
    elseif ctype == "ras"
        return RASConvection()
    elseif ctype == "tm5"
        lmax_conv = get(conv_cfg, "lmax_conv", 0)
        return TM5MatrixConvection(lmax_conv=lmax_conv)
    else
        @warn "Unknown convection type: $ctype — using no convection"
        return nothing
    end
end

"""
Build advection scheme from `[advection]` TOML section.

```toml
[advection]
scheme = "slopes"       # or "ppm" (default "slopes")
ppm_order = 7           # ORD ∈ {4, 5, 6, 7}, only if scheme="ppm"
linrood = false         # Lin-Rood cross-term splitting (CS grids, ppm only)
vertical_remap = false  # remap path for CS PPM
remap_pressure_fix = true  # scale target dp column to source mass (remap path)
```
"""
function _build_advection(config::Dict, ::Type{FT}) where FT
    adv_cfg = get(config, "advection", Dict())
    isempty(adv_cfg) && return SlopesAdvection()

    scheme = get(adv_cfg, "scheme", "slopes")
    if scheme == "slopes"
        prog_slopes = get(adv_cfg, "prognostic_slopes", false)
        return SlopesAdvection(; prognostic_slopes=prog_slopes)
    end

    if scheme == "ppm"
        ppm_order = get(adv_cfg, "ppm_order", 7)
        ppm_order ∈ (4, 5, 6, 7) || @error "ppm_order must be in {4, 5, 6, 7}, got $ppm_order"
        damp_coeff = get(adv_cfg, "damp_coeff", 0.0)
        use_linrood = get(adv_cfg, "linrood", false)
        use_vertical_remap = get(adv_cfg, "vertical_remap", false)
        use_gchp = get(adv_cfg, "gchp", false)
        per_step_remap = get(adv_cfg, "per_step_remap", false)
        if use_vertical_remap && !use_linrood && !use_gchp
            @info "vertical_remap requires linrood — enabling automatically"
            use_linrood = true
        end
        if use_gchp
            @info "GCHP-faithful transport enabled — area-based pre-advection + Courant PPM"
        end
        if per_step_remap
            @info "per_step_remap enabled — vertical remap after every horizontal substep (matches GCHP)"
        end
        return PPMAdvection{ppm_order}(; damp_coeff, use_linrood, use_vertical_remap,
                                        use_gchp, per_step_remap)
    elseif scheme == "prather" || scheme == "som"
        use_limiter = get(adv_cfg, "use_limiter", true)
        return PratherAdvection(use_limiter)
    else
        @warn "Unknown advection scheme: $scheme — using slopes advection"
        return SlopesAdvection()
    end
end

export load_configuration, build_model_from_config
