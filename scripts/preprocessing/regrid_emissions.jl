#!/usr/bin/env julia
# ===========================================================================
# Generic TOML-driven emission preprocessor
#
# Converts any NetCDF lat-lon emission source to flat binary on cubed-sphere.
# Each emission source is defined by a TOML config file (config/emissions/*.toml).
#
# Usage:
#   julia --project=. scripts/preprocessing/regrid_emissions.jl config/emissions/gridfed_fossil_co2.toml
#   julia --project=. scripts/preprocessing/regrid_emissions.jl config/emissions/*.toml
#
# Binary layout:
#   [Header]  JSON metadata (Nc, Nt, species, time_hours, ...)
#   For each snapshot t = 1..Nt:
#     panel_1(Nc×Nc) | panel_2 | ... | panel_6
# ===========================================================================

using AtmosTransport
using AtmosTransport.Architectures: CPU
using AtmosTransport.Grids
using AtmosTransport.Grids: HybridSigmaPressure
using AtmosTransport.Parameters
using AtmosTransport.Sources
using AtmosTransport.IO: read_geosfp_cs_grid_info
using NCDatasets
using Dates
using Printf
using JSON3
using TOML

# =====================================================================
# Configuration parsing
# =====================================================================

struct EmissionConfig
    # Source
    source_type   :: String
    source_path   :: String
    variable      :: String
    lon_coord     :: String
    lat_coord     :: String
    time_dim      :: String        # "" if no time dimension
    date_range    :: Tuple{Date, Date}  # (Date(1), Date(9999,12,31)) if unset

    # Units
    conversions   :: Vector{String}

    # Target
    grid_type     :: String
    Nc            :: Int
    coord_file    :: String
    gridspec_file :: String        # "" if not set; provides exact CS corners + areas

    # Output
    output_path   :: String
    species       :: String
    float_type    :: Type

    # Validation (optional)
    expected_total :: Float64
    expected_unit  :: String
    tolerance      :: Float64
end

function parse_emission_config(toml_path::String)
    cfg = TOML.parsefile(toml_path)

    src = cfg["source"]
    source_type = get(src, "type", "netcdf_latlon")
    source_path = expanduser(src["path"])
    variable    = get(src, "variable", "auto")
    lon_coord   = get(src, "lon_coord", "auto")
    lat_coord   = get(src, "lat_coord", "auto")
    time_dim    = get(src, "time_dim", "")

    date_range = if haskey(src, "date_range")
        dr = src["date_range"]
        (Date(dr[1]), Date(dr[2]))
    else
        (Date(1), Date(9999, 12, 31))
    end

    units = get(cfg, "units", Dict())
    conversions = get(units, "conversions", String[])

    tgt = cfg["target"]
    grid_type  = get(tgt, "grid", "cubed_sphere")
    Nc         = get(tgt, "Nc", 180)
    coord_file = expanduser(tgt["coord_file"])
    gridspec_file = haskey(tgt, "gridspec_file") ? expanduser(tgt["gridspec_file"]) : ""

    out = cfg["output"]
    output_path = expanduser(out["path"])
    species     = get(out, "species", "unknown")
    ft_str      = get(out, "float_type", "Float32")
    float_type  = ft_str == "Float64" ? Float64 : Float32

    val = get(cfg, "validation", Dict())
    expected_total = get(val, "expected_total", NaN)
    expected_unit  = get(val, "expected_unit", "")
    tolerance      = get(val, "tolerance", 0.05)

    return EmissionConfig(
        source_type, source_path, variable, lon_coord, lat_coord,
        time_dim, date_range, conversions, grid_type, Nc, coord_file,
        gridspec_file, output_path, species, float_type,
        expected_total, expected_unit, tolerance
    )
end

# =====================================================================
# Grid builder (cached across configs with same coord_file + Nc)
# =====================================================================

const GRID_CACHE = Dict{Tuple{String, Int}, Any}()
const CS_AREAS_CACHE = Dict{String, Any}()

function get_or_build_grid(coord_file::String, Nc::Int, FT::Type;
                            gridspec_file::String="")
    key = (coord_file, Nc)
    haskey(GRID_CACHE, key) && return GRID_CACHE[key]

    @info "Building CubedSphereGrid C$Nc..."
    params = load_parameters(FT)
    pp = params.planet
    vc = HybridSigmaPressure(FT[0, 0], FT[0, 1])
    grid = CubedSphereGrid(CPU(); FT, Nc, vertical=vc,
                            radius=pp.radius, gravity=pp.gravity,
                            reference_pressure=pp.reference_surface_pressure)

    if isfile(coord_file)
        lons, lats, clons, clats = read_geosfp_cs_grid_info(coord_file)
        for p in 1:6, j in 1:Nc, i in 1:Nc
            grid.λᶜ[p][i, j] = lons[i, j, p]
            grid.φᶜ[p][i, j] = lats[i, j, p]
        end
        if clons !== nothing && clats !== nothing
            R = Float64(grid.radius)
            gmao_areas = Sources.compute_areas_from_corners(
                Float64.(clons), Float64.(clats), R, Nc)
            for p in 1:6, j in 1:Nc, i in 1:Nc
                grid.Aᶜ[p][i, j] = FT(gmao_areas[p][i, j])
            end
            @info "  GMAO coordinates + corner-based areas from $(basename(coord_file))"
        else
            @info "  GMAO coordinates from $(basename(coord_file)) (no corners — gnomonic areas)"
        end
        # Mark grid as having GMAO coordinates
        Grids.set_coord_status!(grid, :gmao, coord_file)
    else
        error("Coordinate file not found: $coord_file")
    end

    # Load exact areas from gridspec file if provided
    if gridspec_file != "" && isfile(gridspec_file)
        gs = NCDataset(gridspec_file)
        gs_areas = Array(gs["areas"])  # (180, 180, 6)
        close(gs)
        cs_areas_exact = ntuple(p -> FT.(gs_areas[:, :, p]), 6)
        CS_AREAS_CACHE[gridspec_file] = cs_areas_exact
        @info "  Loaded exact CS areas from $(basename(gridspec_file))"
    end

    GRID_CACHE[key] = grid
    return grid
end

# =====================================================================
# NetCDF coordinate auto-detection
# =====================================================================

function detect_coord_name(ds, hint::String, candidates::Vector{String})
    hint != "auto" && return hint
    for name in candidates
        haskey(ds, name) && return name
    end
    error("Cannot auto-detect coordinate. Tried: $candidates")
end

function detect_variable(ds, hint::String)
    hint != "auto" && return hint
    skip = Set(["lon", "lat", "longitude", "latitude", "time",
                "lev", "level", "x", "y", "bounds"])
    for name in keys(ds)
        name in skip && continue
        startswith(name, "time") && continue
        ndims(ds[name]) >= 2 && return name
    end
    error("Cannot auto-detect emission variable")
end

# =====================================================================
# Unit conversions
# =====================================================================

function apply_conversions!(flux::Matrix{FT}, conversions::Vector{String},
                             lons, lats, R::FT) where FT
    for conv in conversions
        if conv == "divide_by_seconds_per_month"
            flux ./= FT(SECONDS_PER_MONTH)
        elseif conv == "divide_by_seconds_per_year"
            flux ./= FT(SECONDS_PER_YEAR)
        elseif conv == "multiply_by_KGC_TO_KGCO2"
            flux .*= FT(KGC_TO_KGCO2)
        elseif conv == "tonnes_per_cell_to_kgm2s"
            areas = Sources.latlon_cell_areas(FT.(lons), FT.(lats), R)
            Nlon, Nlat = size(flux)
            @inbounds for j in 1:Nlat, i in 1:Nlon
                flux[i, j] = flux[i, j] * FT(KG_PER_TONNE) / (FT(SECONDS_PER_YEAR) * areas[j])
            end
        else
            error("Unknown conversion: '$conv'")
        end
    end
    return flux
end

# =====================================================================
# Binary writer
# =====================================================================

function write_cs_binary(outpath::String, panels_vec::Vector, Nc::Int, FT::Type;
                          source::String="", species::String="",
                          units::String="kg/m2/s",
                          time_hours::Vector{Float64}=Float64[],
                          extra::Dict{String,Any}=Dict{String,Any}())
    Nt = length(panels_vec)
    n_panel = Nc * Nc
    header_size = Nt > 100 ? 65536 : 4096

    header = Dict{String,Any}(
        "magic"        => "ECSF",
        "version"      => 2,
        "Nc"           => Nc,
        "n_panels"     => 6,
        "Nt"           => Nt,
        "float_type"   => string(FT),
        "float_bytes"  => sizeof(FT),
        "header_bytes" => header_size,
        "n_per_panel"  => n_panel,
        "units"        => units,
        "source"       => source,
        "species"      => species,
        "time_hours"   => time_hours,
    )
    merge!(header, extra)

    header_json = JSON3.write(header)
    length(header_json) < header_size ||
        error("Header JSON too large ($(length(header_json)) > $header_size)")

    mkpath(dirname(outpath))
    open(outpath, "w") do io
        header_buf = zeros(UInt8, header_size)
        copyto!(header_buf, 1, Vector{UInt8}(header_json), 1, length(header_json))
        write(io, header_buf)
        for t in 1:Nt, p in 1:6
            write(io, vec(FT.(panels_vec[t][p])))
        end
    end

    actual = filesize(outpath)
    expected = header_size + Nt * 6 * n_panel * sizeof(FT)
    @info @sprintf("  Written: %s (%.1f MB, Nt=%d)", outpath, actual / 1e6, Nt)
    actual == expected || @warn "Size mismatch: $actual vs expected $expected"
end

# =====================================================================
# Validation
# =====================================================================

"""
    validate_mass_conservation(src_integrals, panels_vec, grid, cfg)

Per-snapshot validation: compare source integral vs regridded target integral.
This is the ground-truth mass conservation check — did regridding preserve mass?

Also reports the time-mean mass rate in physical units if `[validation]` is set.
"""
function validate_mass_conservation(src_integrals::Vector{Float64},
                                     panels_vec, grid, cfg::EmissionConfig)
    Nt = length(panels_vec)
    Nt == 0 && return

    # Per-snapshot mass conservation
    max_err = 0.0
    mean_err = 0.0
    for t in 1:Nt
        tgt = sum(sum(Float64.(panels_vec[t][p]) .* Float64.(grid.Aᶜ[p])) for p in 1:6)
        src = src_integrals[t]
        if abs(src) > 1e-30
            err = abs(tgt - src) / abs(src)
            max_err = max(max_err, err)
            mean_err += err
        end
    end
    mean_err /= Nt

    status = max_err < 0.001 ? "PASS" : (max_err < 0.01 ? "OK" : "WARN")
    @info @sprintf("  Mass conservation [%s]: mean %.4f%%, max %.4f%% (across %d snapshots)",
                    status, mean_err * 100, max_err * 100, Nt)

    # Physical total (informational) if validation section exists
    if !isnan(cfg.expected_total) && cfg.expected_unit != ""
        avg_mass_rate = zero(Float64)
        for t in 1:Nt, p in 1:6
            avg_mass_rate += sum(Float64.(panels_vec[t][p]) .* Float64.(grid.Aᶜ[p]))
        end
        avg_mass_rate /= Nt

        unit = cfg.expected_unit
        computed = if unit in ("PgCO2/yr", "Pg/yr", "GtCO2/yr", "Gt/yr")
            avg_mass_rate * SECONDS_PER_YEAR / 1e12
        elseif unit == "kt/yr"
            avg_mass_rate * SECONDS_PER_YEAR / 1e6
        elseif unit == "kg/yr"
            avg_mass_rate * SECONDS_PER_YEAR
        elseif unit in ("TgCO2/yr", "Tg/yr")
            avg_mass_rate * SECONDS_PER_YEAR / 1e9
        else
            NaN
        end

        if !isnan(computed)
            @info @sprintf("  Physical total: %.4f %s (reference: %.4f, %d-snapshot mean)",
                            computed, unit, cfg.expected_total, Nt)
        end
    end
end

# =====================================================================
# Main processing
# =====================================================================

function process_emission(cfg::EmissionConfig)
    @info "=" ^ 70
    @info "Processing: $(cfg.species) [$(basename(cfg.output_path))]"
    @info "=" ^ 70

    FT = cfg.float_type
    grid = get_or_build_grid(cfg.coord_file, cfg.Nc, FT;
                              gridspec_file=cfg.gridspec_file)
    Nc = grid.Nc
    R = FT(grid.radius)

    # Find source files
    source_dir = dirname(cfg.source_path)
    pattern = basename(cfg.source_path)
    if occursin("*", pattern) || occursin("?", pattern)
        all_files = sort(readdir(source_dir, join=true))
        # Match glob pattern (simple * wildcard)
        regex = Regex("^" * replace(replace(pattern, "." => "\\."), "*" => ".*") * "\$")
        nc_files = filter(f -> occursin(regex, basename(f)), all_files)
    else
        nc_files = isfile(cfg.source_path) ? [cfg.source_path] : String[]
    end

    isempty(nc_files) && error("No files found matching: $(cfg.source_path)")
    @info "  Found $(length(nc_files)) source file(s)"

    # Read coordinates from first file
    ds0 = NCDataset(nc_files[1])
    lon_name = detect_coord_name(ds0, cfg.lon_coord,
                                  ["longitude", "lon", "x", "Longitude"])
    lat_name = detect_coord_name(ds0, cfg.lat_coord,
                                  ["latitude", "lat", "y", "Latitude"])
    lon_src = Float64.(Array(ds0[lon_name]))
    lat_raw = Float64.(Array(ds0[lat_name]))
    close(ds0)

    need_flip = length(lat_raw) > 1 && lat_raw[1] > lat_raw[end]
    lat_src = need_flip ? reverse(lat_raw) : lat_raw

    # Build regrid map once (with exact areas if available)
    @info "  Building regrid map ($(length(lon_src))×$(length(lat_src)) → C$Nc)..."
    cs_areas_kw = haskey(CS_AREAS_CACHE, cfg.gridspec_file) ?
        CS_AREAS_CACHE[cfg.gridspec_file] : nothing
    cs_map = Sources.build_conservative_cs_map(FT.(lon_src), FT.(lat_src), grid;
                                                cs_areas=cs_areas_kw)
    @info "  Regrid map built."

    # Date filtering for time-varying sources
    has_time = cfg.time_dim != ""
    sim_start = DateTime(cfg.date_range[1])

    snapshots = NTuple{6, Matrix{FT}}[]
    time_hours = Float64[]
    src_integrals = Float64[]  # source mass rate [kg/s] per snapshot
    src_areas = Sources.latlon_cell_areas(FT.(lon_src), FT.(lat_src), FT(grid.radius))

    for filepath in nc_files
        # Filter by date if date_range specified
        if has_time && cfg.date_range[1] != Date(1)
            m = match(r"(\d{4,6})", basename(filepath))
            if m !== nothing
                ds_str = m[1]
                file_date = if length(ds_str) == 6
                    Date(parse(Int, ds_str[1:4]), parse(Int, ds_str[5:6]), 1)
                elseif length(ds_str) == 4
                    Date(parse(Int, ds_str), 1, 1)
                else
                    Date(1)  # can't parse
                end
                (file_date < cfg.date_range[1] || file_date > cfg.date_range[2]) && continue
            end
        end

        ds = NCDataset(filepath)
        var_name = detect_variable(ds, cfg.variable)

        data = ds[var_name]
        ndim = ndims(data)

        if has_time && ndim >= 3
            raw = Array(data)
            Nt_file = size(raw, 3)

            time_vals = if haskey(ds, cfg.time_dim)
                try
                    ds[cfg.time_dim][:]
                catch
                    nothing
                end
            else
                nothing
            end

            for ti in 1:Nt_file
                slice = FT.(replace(raw[:, :, ti], missing => zero(FT)))
                if need_flip
                    slice = slice[:, end:-1:1]
                end

                apply_conversions!(slice, cfg.conversions, lon_src, lat_src, R)
                # Source integral before regridding
                src_int = sum(Float64(slice[i,j]) * Float64(src_areas[j])
                              for j in eachindex(lat_src), i in eachindex(lon_src))
                push!(src_integrals, src_int)
                panels = Sources.regrid_latlon_to_cs(slice, FT.(lon_src), FT.(lat_src),
                                                       grid; cs_map)
                push!(snapshots, panels)

                # Compute time offset (hours since sim_start)
                hrs = if time_vals !== nothing && ti <= length(time_vals)
                    tv = time_vals[ti]
                    if tv isa DateTime
                        Dates.value(tv - sim_start) / 3_600_000.0
                    elseif tv isa Date
                        Dates.value(DateTime(tv) - sim_start) / 3_600_000.0
                    else
                        # Numeric time: assume monthly snapshots, compute offset
                        Float64(length(time_hours)) * (SECONDS_PER_MONTH / 3600.0)
                    end
                else
                    # No time coordinate: assume monthly snapshots
                    Float64(length(time_hours)) * (SECONDS_PER_MONTH / 3600.0)
                end
                push!(time_hours, hrs)
            end
        else
            # Static or 2D field
            raw_2d = if ndim == 3
                Array(data)[:, :, 1]
            else
                Array(data)[:, :]
            end
            slice = FT.(replace(raw_2d, missing => zero(FT)))
            if need_flip
                slice = slice[:, end:-1:1]
            end

            apply_conversions!(slice, cfg.conversions, lon_src, lat_src, R)
            src_int = sum(Float64(slice[i,j]) * Float64(src_areas[j])
                          for j in eachindex(lat_src), i in eachindex(lon_src))
            push!(src_integrals, src_int)
            panels = Sources.regrid_latlon_to_cs(slice, FT.(lon_src), FT.(lat_src),
                                                   grid; cs_map)
            push!(snapshots, panels)
            push!(time_hours, 0.0)
        end

        close(ds)
        @info "  $(basename(filepath)): processed"
    end

    Nt = length(snapshots)
    @info "  Total: $Nt snapshot(s)"

    # Validation: per-snapshot mass conservation
    validate_mass_conservation(src_integrals, snapshots, grid, cfg)

    # Write binary
    write_cs_binary(cfg.output_path, snapshots, Nc, FT;
                     source=basename(first(nc_files)),
                     species=cfg.species,
                     time_hours=time_hours)
end

# =====================================================================
# Entry point
# =====================================================================

function main()
    if isempty(ARGS)
        println("Usage: julia --project=. scripts/preprocessing/regrid_emissions.jl config/emissions/*.toml")
        println("\nAvailable configs:")
        for f in sort(readdir(joinpath(@__DIR__, "../../config/emissions"), join=true))
            endswith(f, ".toml") && println("  ", f)
        end
        exit(1)
    end

    toml_files = filter(f -> endswith(f, ".toml"), ARGS)
    @info "Processing $(length(toml_files)) emission config(s)"

    for toml_path in toml_files
        cfg = parse_emission_config(toml_path)
        try
            process_emission(cfg)
        catch e
            @error "Failed processing $toml_path" exception=(e, catch_backtrace())
        end
    end

    @info "\nAll done."
end

main()
