# ---------------------------------------------------------------------------
# CATRINE intercomparison emission readers
#
# Readers for the four CATRINE D7.1 tracer flux types:
#   1. CAMS CO2 (disaggregated with OCO-2)  — from IPSL THREDDS
#   2. GridFED fossil CO2 (v2024.1)          — from Zenodo
#   3. EDGAR v8.0 SF6 (scaled to NOAA)       — from EDGAR JRC
#   4. Zhang et al. 2021 222Rn               — from Harvard FTP
#
# Data download URLs:
#   CO2:        https://thredds-su.ipsl.fr/.../CATRINE/catalog.html
#   Fossil CO2: https://zenodo.org/records/8386803
#   SF6:        https://edgar.jrc.ec.europa.eu/dataset_ghg80
#   SF6 scale:  https://gml.noaa.gov/webdata/ccgg/trends/sf6/sf6_gr_gl.txt
#   Rn222:      http://ftp.as.harvard.edu/gcgrid/data/ExtData/HEMCO/ZHANG_Rn222/v2021-11/
# ---------------------------------------------------------------------------

const N_AVOGADRO = 6.02214076e23   # atoms/mol

# =====================================================================
# 1. CAMS CO2 fluxes (disaggregated with OCO-2)
# =====================================================================

"""
    load_cams_co2(filepath, target_grid; year, species=:co2)

Load CAMS CO2 surface fluxes from CATRINE THREDDS NetCDF and regrid to
`target_grid`.

The CAMS product provides total CO2 flux (biosphere + ocean + fire + fossil
combined) at ~0.1°, in kg CO2/m²/s. Files may contain multiple monthly
time steps.

Returns a `TimeVaryingSurfaceFlux` for multi-step files, or a `SurfaceFlux`
for single-step files.
"""
function load_cams_co2(filepath::String,
                       target_grid::LatitudeLongitudeGrid{FT};
                       year::Int = 2022,
                       species::Symbol = :co2) where FT
    isfile(filepath) || error("CAMS CO2 file not found: $filepath")
    ds = NCDataset(filepath)

    # Discover coordinate variables
    lon_var = _find_coord_var(ds, ["lon", "longitude", "x"])
    lat_var = _find_coord_var(ds, ["lat", "latitude", "y"])
    lon_src = Float64.(ds[lon_var][:])
    lat_src = Float64.(ds[lat_var][:])

    # Discover the flux variable — try common CAMS names
    flux_var = _find_flux_var(ds, ["flux_apos_tot", "flux_tot", "co2_flux",
                                   "flux", "CO2_flux", "fco2",
                                   "flux_apos_bio", "bio_flux"])
    if flux_var === nothing
        varlist = join(keys(ds), ", ")
        error("No CO2 flux variable found in $filepath. Available variables: $varlist")
    end

    # Check units; convert if in mol/m²/s instead of kg/m²/s
    units = lowercase(get(ds[flux_var].attrib, "units", "kg/m2/s"))
    mol_to_kg = contains(units, "mol") && !contains(units, "kg")

    Nt = _get_time_dim(ds)
    Nx_m, Ny_m = target_grid.Nx, target_grid.Ny

    flux_mats = Matrix{FT}[]
    time_hours = Float64[]

    for ti in 1:Nt
        raw = _read_2d_or_3d(ds, flux_var, ti, FT)
        if mol_to_kg
            raw .*= FT(M_CO2)  # mol CO2/m²/s → kg CO2/m²/s
        end

        # Flip latitude if N→S
        raw, lat_use = ensure_south_to_north(raw, lat_src)
        lon_use, raw = remap_lon_neg180_to_0_360(lon_src, raw)

        # Regrid: conservative for high-res (< 0.5°), simple otherwise
        Δlon = abs(lon_use[2] - lon_use[1])
        flux_model = if Δlon < 0.5
            _conservative_regrid(raw, FT.(lon_use), lat_use, target_grid)
        else
            nearest_neighbor_regrid(raw, lon_use, lat_use, target_grid)
        end

        push!(flux_mats, flux_model)
        push!(time_hours, (ti - 1) * _guess_dt_hours(ds, Nt))
    end
    close(ds)

    Nt_out = length(flux_mats)
    @info "CAMS CO2: loaded $Nt_out time steps from $(basename(filepath))"

    if Nt_out == 1
        return SurfaceFlux(flux_mats[1], species, "CAMS CO2 $year")
    else
        stack = _stack_matrices(flux_mats, Nx_m, Ny_m, FT)
        return TimeVaryingSurfaceFlux(stack, time_hours, species;
                                    label="CAMS CO2 $year")
    end
end


# =====================================================================
# 1b. LMDZ/CAMS posterior CO2 fluxes (multi-file monthly, 3-hourly)
# =====================================================================

const KGC_TO_KGCO2 = 44.01 / 12.011   # kgC → kgCO2 conversion

"""
    load_lmdz_co2(dirpath, target_grid; start_date, end_date, species=:co2,
                  flux_var="flux_apos")

Load LMDZ/CAMS posterior CO2 surface fluxes from a directory of monthly
NetCDF files and regrid to `target_grid`.

Files have naming pattern:
    z_cams_l_cams55_YYYYMM_FT24r2_ra_sfc_3h_co2_flux.nc

Each file contains 3-hourly flux fields (time × lat × lon) in kgC/m²/s.
These are converted to kgCO2/m²/s (×3.664) for the emission pipeline.

Returns a `TimeVaryingSurfaceFlux` for lat-lon grids.
"""
function load_lmdz_co2(dirpath::String,
                       target_grid::LatitudeLongitudeGrid{FT};
                       start_date::Date = Date(2021, 12, 1),
                       end_date::Date   = Date(2023, 12, 31),
                       species::Symbol  = :co2,
                       flux_var::String = "flux_apos") where FT
    isdir(dirpath) || error("LMDZ CO2 directory not found: $dirpath")

    # Discover monthly files in date range
    files = _find_lmdz_files(dirpath, start_date, end_date)
    isempty(files) && error("No LMDZ files found in $(dirpath) for $(start_date) to $(end_date)")

    Nx_m, Ny_m = target_grid.Nx, target_grid.Ny
    flux_mats  = Matrix{FT}[]
    time_hours = Float64[]

    sim_start = DateTime(start_date)

    for (_, filepath) in files
        ds = NCDataset(filepath)
        lon_var = _find_coord_var(ds, ["lon", "longitude", "x"])
        lat_var = _find_coord_var(ds, ["lat", "latitude", "y"])
        lon_src = Float64.(ds[lon_var][:])
        lat_src = Float64.(ds[lat_var][:])

        haskey(ds, flux_var) || begin
            @warn "Variable '$flux_var' not found in $(basename(filepath)), skipping"
            close(ds)
            continue
        end

        # Read time axis — NCDatasets auto-converts to DateTime
        time_datetimes = ds["time"][:]

        Nt = length(time_datetimes)
        raw_flux = ds[flux_var]

        for ti in 1:Nt
            raw = FT.(nomissing(raw_flux[:, :, ti], zero(FT)))
            raw .*= FT(KGC_TO_KGCO2)

            raw, lat_use = ensure_south_to_north(raw, lat_src)
            lon_use, raw = remap_lon_neg180_to_0_360(lon_src, raw)

            flux_model = nearest_neighbor_regrid(raw, lon_use, lat_use, target_grid)
            push!(flux_mats, flux_model)

            hrs_since_start = Dates.value(DateTime(time_datetimes[ti]) - sim_start) /
                              3_600_000.0  # ms → hours
            push!(time_hours, hrs_since_start)
        end
        close(ds)
        @info "LMDZ CO2: $(basename(filepath)) — $Nt timesteps loaded"
    end

    @info "LMDZ CO2: total $(length(flux_mats)) timesteps from $(length(files)) files"

    stack = _stack_matrices(flux_mats, Nx_m, Ny_m, FT)
    return TimeVaryingSurfaceFlux(stack, time_hours, species;
                                label="LMDZ CO2 posterior ($flux_var)")
end

"""
    load_lmdz_co2(dirpath, grid::CubedSphereGrid; ...)

Cubed-sphere version: loads LMDZ fluxes at 1° and regrids to CS panels.
Returns a `TimeVaryingSurfaceFlux`.
"""
function load_lmdz_co2(dirpath::String,
                       grid::CubedSphereGrid{FT};
                       start_date::Date = Date(2021, 12, 1),
                       end_date::Date   = Date(2023, 12, 31),
                       species::Symbol  = :co2,
                       flux_var::String = "flux_apos") where FT
    isdir(dirpath) || error("LMDZ CO2 directory not found: $(dirpath)")

    # Check for preprocessed binary
    bin_path = _default_emission_bin_path(dirpath, species, grid.Nc)
    if !isempty(bin_path) && isfile(bin_path)
        @info "Loading preprocessed LMDZ CO2 binary: $bin_path"
        panels_vec, time_hours, _ = load_cs_emission_binary(bin_path, FT)
        @info "LMDZ CO2 from binary: $(length(panels_vec)) timesteps"
        return TimeVaryingSurfaceFlux(panels_vec, time_hours, species;
                                               label="LMDZ CO2 posterior (binary)",
                                               molar_mass=M_CO2)
    end

    files = _find_lmdz_files(dirpath, start_date, end_date)
    isempty(files) && error("No LMDZ files found in $(dirpath) for $(start_date) to $(end_date)")

    Nc = grid.Nc
    snapshots = Vector{NTuple{6, Matrix{FT}}}()
    time_hours = Float64[]
    sim_start = DateTime(start_date)
    cs_map = nothing  # built lazily on first timestep

    for (_, filepath) in files
        ds = NCDataset(filepath)
        lon_var = _find_coord_var(ds, ["lon", "longitude", "x"])
        lat_var = _find_coord_var(ds, ["lat", "latitude", "y"])
        lon_src = Float64.(ds[lon_var][:])
        lat_src = Float64.(ds[lat_var][:])

        haskey(ds, flux_var) || begin
            @warn "Variable '$flux_var' not found in $(basename(filepath)), skipping"
            close(ds)
            continue
        end

        time_datetimes = ds["time"][:]
        Nt = length(time_datetimes)
        raw_flux = ds[flux_var]

        for ti in 1:Nt
            raw = FT.(nomissing(raw_flux[:, :, ti], zero(FT)))
            raw .*= FT(KGC_TO_KGCO2)

            raw, lat_use = ensure_south_to_north(raw, lat_src)
            lon_use, raw = remap_lon_neg180_to_0_360(lon_src, raw)

            lon_typed = FT.(lon_use)
            lat_typed = FT.(lat_use)

            if cs_map === nothing
                @info "Building LMDZ→C$(Nc) regrid map..."
                cs_map = build_latlon_to_cs_map(lon_typed, lat_typed, grid)
                @info "  Regrid map built."
            end

            flux_panels = regrid_latlon_to_cs(raw, lon_typed, lat_typed, grid;
                                                cs_map)
            push!(snapshots, flux_panels)

            hrs = Dates.value(DateTime(time_datetimes[ti]) - sim_start) / 3_600_000.0
            push!(time_hours, hrs)
        end
        close(ds)
        @info "LMDZ CO2: $(basename(filepath)) — $Nt timesteps loaded"
    end

    @info "LMDZ CO2 on C$(Nc): total $(length(snapshots)) timesteps from $(length(files)) files"

    return TimeVaryingSurfaceFlux(snapshots, time_hours, species;
                                           label="LMDZ CO2 posterior ($flux_var)",
                                           molar_mass=M_CO2)
end

"""Find LMDZ monthly files matching the date range."""
function _find_lmdz_files(dirpath::String, start_date::Date, end_date::Date)
    files = Tuple{Date, String}[]
    for f in sort(readdir(dirpath))
        endswith(f, ".nc") || continue
        m = match(r"(\d{6})", f)
        m === nothing && continue
        ym = m.captures[1]
        yr = parse(Int, ym[1:4])
        mo = parse(Int, ym[5:6])
        file_date = Date(yr, mo, 1)
        # Include file if its month overlaps with the run period
        file_end = file_date + Month(1) - Day(1)
        if file_date <= end_date && file_end >= start_date
            push!(files, (file_date, joinpath(dirpath, f)))
        end
    end
    return sort(files; by=first)
end

"""Parse 'hours since YYYY-MM-DD HH:MM:SS' into a DateTime."""
function _parse_time_epoch(units_str::String)
    m = match(r"(\w+)\s+since\s+(\d{4})-(\d{2})-(\d{2})", units_str)
    m === nothing && return DateTime(2021, 12, 1)  # fallback
    return DateTime(parse(Int, m.captures[2]),
                    parse(Int, m.captures[3]),
                    parse(Int, m.captures[4]))
end


# =====================================================================
# 2. GridFED fossil CO2 (GridFEDv2024.x)
# =====================================================================

"""
    _find_gridfed_flux_var(ds) → (dataset_or_group, varname) or nothing

Find the GridFED emission variable, checking both flat variables and
the CO2 group structure used in GridFEDv2024.1+ (group "CO2", variable "TOTAL").
"""
function _find_gridfed_flux_var(ds)
    # Try flat variables first
    flat = _find_flux_var(ds, ["TOTAL", "co2_excl_short_cycle_org_c",
                               "emi_co2", "CO2_em_anthro", "fossil_co2",
                               "emissions", "flux", "CO2_emissions"])
    flat !== nothing && return (ds, flat)

    # Try CO2 group (GridFEDv2024.1 layout: group "CO2", variable "TOTAL")
    if hasproperty(ds, :group) || applicable(keys, ds.group)
        for gname in ["CO2", "co2"]
            try
                g = ds.group[gname]
                gvar = _find_flux_var(g, ["TOTAL", "total", "emissions", "flux"])
                gvar !== nothing && return (g, gvar)
            catch
                continue
            end
        end
    end

    return nothing
end

"""
    _gridfed_convert_units(raw, units_str, lon, lat, R, FT) → Matrix{FT}

Convert GridFED raw data to kg/m²/s based on the `units` attribute.
"""
function _gridfed_convert_units(raw::Matrix{FT}, units_str::String,
                                 lon, lat, R, ::Type{FT}) where FT
    already_per_m2 = contains(units_str, "m2") || contains(units_str, "m²") ||
                     contains(units_str, "/m")

    if contains(units_str, "kg") && (contains(units_str, "/s") || contains(units_str, "sec"))
        return raw  # already kg/m²/s
    elseif contains(units_str, "tonnes") || contains(units_str, "ton")
        return tonnes_per_year_to_kgm2s(raw, lon, lat, R)
    elseif contains(units_str, "kg") && contains(units_str, "month")
        sec_per_month = FT(365.25 / 12 * 86400)
        if already_per_m2
            return raw ./ sec_per_month  # kg/month/m² → kg/s/m²
        else
            return mass_per_cell_to_kgm2s(raw, lon, lat, R, sec_per_month)
        end
    elseif contains(units_str, "kg") && contains(units_str, "year")
        sec_per_year = FT(365.25 * 86400)
        if already_per_m2
            return raw ./ sec_per_year  # kg/year/m² → kg/s/m²
        else
            return mass_per_cell_to_kgm2s(raw, lon, lat, R, sec_per_year)
        end
    else
        @warn "GridFED: unrecognized units '$units_str', assuming kg/m²/s"
        return raw
    end
end

"""
    load_gridfed_fossil_co2(filepath, target_grid::LatitudeLongitudeGrid; year, species)

Load GridFED fossil fuel CO₂ emissions and conservatively regrid to `target_grid`.

`filepath` may be a single NetCDF file or a directory containing per-year
NetCDF files (e.g. from GridFEDv2024.0 yearly ZIPs). For directories, all
`.nc` files are loaded and concatenated into a `TimeVaryingSurfaceFlux` with
monthly time steps.

GridFED data is at 0.1° resolution. Conservative regridding preserves
global mass (CATRINE D7.1 protocol requirement).

Returns a `TimeVaryingSurfaceFlux` (multi-month) or `SurfaceFlux` (single step).
"""
function load_gridfed_fossil_co2(filepath::String,
                                 target_grid::LatitudeLongitudeGrid{FT};
                                 year::Int = 2022,
                                 species::Symbol = :fossil_co2,
                                 start_date::Date = Date(year, 1, 1)) where FT
    if isdir(filepath)
        return _load_gridfed_dir_latlon(filepath, target_grid, year, species, FT;
                                        start_date)
    end
    isfile(filepath) || error("GridFED path not found: $filepath")
    return _load_gridfed_file_latlon(filepath, target_grid, year, species, FT;
                                     start_date)
end

"""Extract year from a GridFED filename like `GCP-GridFEDv2024.0_2021.short.nc`."""
function _gridfed_file_year(filepath::String)
    bn = basename(filepath)
    m = match(r"(\d{4})\.(?:short\.)?nc", bn)
    m === nothing && return nothing
    return parse(Int, m[1])
end

"""Load a single GridFED NetCDF file to lat-lon grid."""
function _load_gridfed_file_latlon(filepath::String,
                                    target_grid::LatitudeLongitudeGrid{FT},
                                    year::Int, species::Symbol,
                                    ::Type{FT};
                                    start_date::Date = Date(year, 1, 1)) where FT
    ds = NCDataset(filepath)

    lon_var = _find_coord_var(ds, ["lon", "longitude", "x"])
    lat_var = _find_coord_var(ds, ["lat", "latitude", "y"])
    lon_src = Float64.(ds[lon_var][:])
    lat_src = Float64.(ds[lat_var][:])

    result = _find_gridfed_flux_var(ds)
    if result === nothing
        varlist = join(keys(ds), ", ")
        close(ds)
        error("No fossil CO2 variable found in $filepath. Variables: $varlist")
    end
    flux_ds, flux_var = result

    units_str = lowercase(get(flux_ds[flux_var].attrib, "units", ""))
    R = FT(target_grid.radius)

    Nt = _get_time_dim(ds)
    Nx_m, Ny_m = target_grid.Nx, target_grid.Ny

    # Infer file year from filename (GridFED has no calendar time variable)
    file_year = _gridfed_file_year(filepath)
    if file_year === nothing
        file_year = year
    end
    sim_start = DateTime(start_date)

    flux_mats = Matrix{FT}[]
    time_hours = Float64[]

    for ti in 1:Nt
        # Calendar-aware: month ti of file_year
        month_date = DateTime(file_year, ti, 1)
        hrs = Float64(Dates.value(month_date - sim_start)) / 3_600_000.0
        hrs < 0 && continue  # skip months before simulation start

        raw = _read_2d_or_3d(flux_ds, flux_var, ti, FT)
        flux_native = _gridfed_convert_units(raw, units_str, lon_src, lat_src, R, FT)

        flux_native_2, lat_use = ensure_south_to_north(flux_native, lat_src)
        lon_use, flux_native_3 = remap_lon_neg180_to_0_360(lon_src, flux_native_2)

        flux_model = _conservative_regrid(flux_native_3, FT.(lon_use), lat_use,
                                           target_grid)
        push!(flux_mats, flux_model)
        push!(time_hours, hrs)
    end
    close(ds)

    Nt_out = length(flux_mats)
    @info "GridFED fossil CO2: loaded $Nt_out time steps from $(basename(filepath)) (file_year=$file_year, start_date=$start_date)"

    if Nt_out == 0
        return SurfaceFlux(zeros(FT, Nx_m, Ny_m), species,
                                "GridFED fossil CO2 (empty)"; molar_mass=M_CO2)
    elseif Nt_out == 1
        return SurfaceFlux(flux_mats[1], species,
                                "GridFED fossil CO2 $year"; molar_mass=M_CO2)
    else
        stack = _stack_matrices(flux_mats, Nx_m, Ny_m, FT)
        return TimeVaryingSurfaceFlux(stack, time_hours, species;
                                    label="GridFED fossil CO2 $year",
                                    molar_mass=M_CO2)
    end
end

"""Load all GridFED NetCDF files from a directory to lat-lon grid."""
function _load_gridfed_dir_latlon(dirpath::String,
                                   target_grid::LatitudeLongitudeGrid{FT},
                                   year::Int, species::Symbol,
                                   ::Type{FT};
                                   start_date::Date = Date(year, 1, 1)) where FT
    nc_files = sort(filter(f -> endswith(f, ".nc"), readdir(dirpath; join=true)))
    isempty(nc_files) && error("No NetCDF files found in $dirpath")

    sim_start = DateTime(start_date)
    Nx_m, Ny_m = target_grid.Nx, target_grid.Ny
    flux_mats = Matrix{FT}[]
    time_hours = Float64[]

    for filepath in nc_files
        ds = NCDataset(filepath)

        lon_var = _find_coord_var(ds, ["lon", "longitude", "x"])
        lat_var = _find_coord_var(ds, ["lat", "latitude", "y"])
        lon_src = Float64.(ds[lon_var][:])
        lat_src = Float64.(ds[lat_var][:])

        result = _find_gridfed_flux_var(ds)
        if result === nothing
            @warn "No flux variable in $(basename(filepath)) — skipping"
            close(ds)
            continue
        end
        flux_ds, flux_var = result

        units_str = lowercase(get(flux_ds[flux_var].attrib, "units", ""))
        R = FT(target_grid.radius)
        Nt = _get_time_dim(ds)

        # Infer file year from filename
        file_year = _gridfed_file_year(filepath)
        if file_year === nothing
            file_year = year
            @warn "Could not infer year from $(basename(filepath)), using default year=$year"
        end

        for ti in 1:Nt
            # Calendar-aware: month ti of file_year
            month_date = DateTime(file_year, ti, 1)
            hrs = Float64(Dates.value(month_date - sim_start)) / 3_600_000.0
            hrs < 0 && continue  # skip months before simulation start

            raw = _read_2d_or_3d(flux_ds, flux_var, ti, FT)
            flux_native = _gridfed_convert_units(raw, units_str, lon_src, lat_src, R, FT)

            flux_native_2, lat_use = ensure_south_to_north(flux_native, lat_src)
            lon_use, flux_native_3 = remap_lon_neg180_to_0_360(lon_src, flux_native_2)

            flux_model = _conservative_regrid(flux_native_3, FT.(lon_use), lat_use,
                                               target_grid)
            push!(flux_mats, flux_model)
            push!(time_hours, hrs)
        end
        close(ds)
    end

    Nt_out = length(flux_mats)
    @info "GridFED fossil CO2: loaded $Nt_out monthly steps from $(length(nc_files)) files (start_date=$start_date)"

    if Nt_out == 0
        return SurfaceFlux(zeros(FT, Nx_m, Ny_m), species,
                                "GridFED fossil CO2 (empty)"; molar_mass=M_CO2)
    elseif Nt_out == 1
        return SurfaceFlux(flux_mats[1], species,
                                "GridFED fossil CO2 $year"; molar_mass=M_CO2)
    else
        stack = _stack_matrices(flux_mats, Nx_m, Ny_m, FT)
        return TimeVaryingSurfaceFlux(stack, time_hours, species;
                                    label="GridFED fossil CO2 (multi-year)",
                                    molar_mass=M_CO2)
    end
end

# =====================================================================
# Binary emission reader (shared by GridFED and EDGAR SF6)
# =====================================================================

const _CATRINE_HEADER_SIZE = 4096

"""
    load_cs_emission_binary(bin_path, FT) → (panels_vec, time_hours, header)

Read a preprocessed cubed-sphere emission binary file.
Header size is auto-detected from the `header_bytes` field (default 4096).
Returns a vector of NTuple{6, Matrix{FT}} snapshots and time axis.
"""
function load_cs_emission_binary(bin_path::String, ::Type{FT}) where FT
    io = open(bin_path, "r")
    # Read initial 4096 bytes; scan for header_bytes before full JSON parse
    initial_bytes = read(io, _CATRINE_HEADER_SIZE)
    initial_str = String(initial_bytes[1:something(findfirst(==(0x00), initial_bytes),
                                                    _CATRINE_HEADER_SIZE + 1) - 1])

    # Extract header_bytes from raw string (works even with truncated JSON)
    header_size = _CATRINE_HEADER_SIZE
    m = match(r"\"header_bytes\"\s*:\s*(\d+)", initial_str)
    if m !== nothing
        header_size = parse(Int, m[1])
    end

    # Read full header
    seek(io, 0)
    hdr_bytes = read(io, header_size)
    json_end = something(findfirst(==(0x00), hdr_bytes), header_size + 1) - 1
    hdr = JSON3.read(String(hdr_bytes[1:json_end]))

    Nc = Int(hdr.Nc)
    Nt = Int(get(hdr, :Nt, 1))
    time_hours = haskey(hdr, :time_hours) ? Float64.(hdr.time_hours) :
                 [(t - 1) * (365.25 / max(Nt, 1) * 24.0) for t in 1:Nt]

    # Seek to data start (after full header)
    seek(io, header_size)

    panels_vec = Vector{NTuple{6, Matrix{FT}}}(undef, Nt)
    for t in 1:Nt
        panels_vec[t] = ntuple(6) do _
            arr = Array{FT}(undef, Nc, Nc)
            read!(io, arr)
            arr
        end
    end
    close(io)
    return panels_vec, time_hours, hdr
end

"""
    _default_emission_bin_path(filepath, species, Nc) → String

Find preprocessed binary for a CATRINE emission source.
Searches: preprocessed_c{Nc}/ directory (sibling to Emissions/), then legacy paths.
"""
function _default_emission_bin_path(filepath::String, species::Symbol, Nc::Int)
    dir = isdir(filepath) ? rstrip(filepath, '/') : dirname(filepath)
    sp_tag = lowercase(string(species))

    # Map species to binary filename prefix
    bin_name = if sp_tag == "fossil_co2"
        "gridfed_fossil_co2"
    elseif sp_tag == "sf6"
        "edgar_sf6"
    elseif sp_tag == "co2"
        "lmdz_co2"
    elseif sp_tag == "rn222"
        "zhang_rn222"
    else
        sp_tag
    end
    bin_file = "$(bin_name)_cs_c$(Nc)_float32.bin"

    # Search: preprocessed_c{Nc}/ (new standard), then parent dir, then input dir
    parent = dirname(dir)
    grandparent = dirname(parent)
    search_dirs = [
        joinpath(grandparent, "preprocessed_c$(Nc)"),  # ~/data/.../catrine/preprocessed_c180/
        joinpath(parent, "preprocessed_c$(Nc)"),
        parent,
        dir,
    ]
    for d in search_dirs
        bin = joinpath(d, bin_file)
        isfile(bin) && return bin
    end
    return ""
end

"""
    load_gridfed_fossil_co2(filepath, grid::CubedSphereGrid; year, species)

Load GridFED fossil CO₂ and regrid to cubed-sphere panels.

If a preprocessed binary file exists (same directory, named
`gridfed_fossil_co2_cs_c{Nc}_float32.bin`), loads from binary directly
(~instant). Otherwise loads monthly flux fields at native 0.1° resolution,
regrids each month to CS panels via conservative area-weighted regridding
with global mass renormalization (~80s for C180).

Returns a `TimeVaryingSurfaceFlux` with monthly snapshots.
"""
function load_gridfed_fossil_co2(filepath::String,
                                 grid::CubedSphereGrid{FT};
                                 year::Int = 2022,
                                 species::Symbol = :fossil_co2,
                                 start_date::Date = Date(year, 1, 1)) where FT
    # Check for preprocessed binary first
    bin_path = _default_emission_bin_path(filepath, species, grid.Nc)
    if !isempty(bin_path) && isfile(bin_path)
        @info "Loading preprocessed GridFED binary: $bin_path"
        panels_vec, time_hours, _ = load_cs_emission_binary(bin_path, FT)
        total_GtCO2_yr = sum(sum(panels_vec[ti][p] .* FT.(grid.Aᶜ[p]))
                              for p in 1:6 for ti in eachindex(panels_vec)) /
                         length(panels_vec) * 365.25 * 86400 / 1e12
        @info "GridFED fossil CO2 from binary: $(round(Float64(total_GtCO2_yr), digits=2)) GtCO2/yr ($(length(panels_vec)) snapshots)"
        return TimeVaryingSurfaceFlux(panels_vec, time_hours, species;
                                               label="GridFED fossil CO2 $year (binary)",
                                               molar_mass=M_CO2)
    end

    # Fall back to on-the-fly regridding
    # Load all monthly flux fields at native resolution
    monthly_fluxes, lon_src, lat_src, monthly_time_hours = _load_gridfed_monthly(filepath, year, FT; start_date)
    Nt = length(monthly_fluxes)

    Nc = grid.Nc
    R = FT(grid.radius)
    _, lat_use = ensure_south_to_north(monthly_fluxes[1], Float64.(lat_src))
    Δlon_s = FT(abs(Float64(lon_src[2]) - Float64(lon_src[1])))
    Δlat_s = FT(abs(Float64(lat_use[2]) - Float64(lat_use[1])))
    Nlon_s = length(lon_src)
    Nlat_s = length(lat_use)

    snapshots = Vector{NTuple{6, Matrix{FT}}}(undef, Nt)
    time_hours = monthly_time_hours

    # Build assignment map once — reused for all months (~80s for 0.1° → C180)
    lat_typed = FT.(lat_use)
    lon_typed = FT.(lon_src)
    @info "Building conservative regrid map (0.1° → C$(Nc))..."
    cs_map = build_latlon_to_cs_map(lon_typed, lat_typed, grid)
    @info "  Regrid map built."

    for ti in 1:Nt
        flux_sn, _ = ensure_south_to_north(monthly_fluxes[ti], Float64.(lat_src))

        # Total mass rate on native grid (for renormalization)
        total_native = zero(FT)
        @inbounds for j in 1:Nlat_s, i in 1:Nlon_s
            φ_s = FT(lat_use[j]) - Δlat_s / 2
            φ_n = FT(lat_use[j]) + Δlat_s / 2
            area = R^2 * deg2rad(Δlon_s) * abs(sind(φ_n) - sind(φ_s))
            total_native += FT(flux_sn[i, j]) * area
        end

        # Conservative area-weighted regrid to CS panels + renormalize
        flux_panels = regrid_latlon_to_cs(FT.(flux_sn), lon_typed, lat_typed, grid;
                                            cs_map)
        total_cs = zero(FT)
        for p in 1:6, j in 1:Nc, i in 1:Nc
            total_cs += flux_panels[p][i, j] * cell_area(i, j, grid; panel=p)
        end
        if abs(total_cs) > zero(FT)
            scale = total_native / total_cs
            for p in 1:6
                flux_panels[p] .*= scale
            end
        end

        snapshots[ti] = flux_panels
    end

    total_GtCO2_yr = sum(sum(snapshots[ti][p] .* FT.(grid.Aᶜ[p]))
                              for p in 1:6 for ti in 1:Nt) / Nt * 365.25 * 86400 / 1e12
    @info "GridFED fossil CO2 on C$Nc: $(round(Float64(total_GtCO2_yr), digits=2)) GtCO2/yr ($Nt monthly steps)"

    return TimeVaryingSurfaceFlux(snapshots, time_hours, species;
                                           label="GridFED fossil CO2 $year",
                                           molar_mass=M_CO2)
end

"""Load GridFED monthly flux fields in kg/m²/s at native resolution.

Returns `(monthly_fluxes, lon_src, lat_src, time_hours)` where `time_hours`
is calendar-aware relative to `start_date`, and months before `start_date`
are filtered out.
"""
function _load_gridfed_monthly(filepath::String, year::Int, ::Type{FT};
                                start_date::Date = Date(year, 1, 1)) where FT
    if isdir(filepath)
        nc_files = sort(filter(f -> endswith(f, ".nc"), readdir(filepath; join=true)))
        isempty(nc_files) && error("No NetCDF files in $filepath")
    else
        isfile(filepath) || error("GridFED path not found: $filepath")
        nc_files = [filepath]
    end

    ds0 = NCDataset(nc_files[1])
    lon_var = _find_coord_var(ds0, ["lon", "longitude", "x"])
    lat_var = _find_coord_var(ds0, ["lat", "latitude", "y"])
    lon_src = FT.(ds0[lon_var][:])
    lat_src = FT.(ds0[lat_var][:])
    close(ds0)

    R = FT(6.371e6)
    monthly = Matrix{FT}[]
    time_hours = Float64[]

    for fp in nc_files
        file_year = _gridfed_file_year(fp)
        yr = file_year !== nothing ? file_year : year

        ds = NCDataset(fp)
        result = _find_gridfed_flux_var(ds)
        if result === nothing
            @warn "No flux variable in $(basename(fp)) — skipping"
            close(ds)
            continue
        end
        flux_ds, flux_var = result
        units_str = lowercase(get(flux_ds[flux_var].attrib, "units", ""))
        Nt = _get_time_dim(ds)
        for ti in 1:Nt
            month_dt = DateTime(yr, ti, 1)
            t_h = Dates.value(month_dt - DateTime(start_date)) / 3.6e6
            t_h < 0 && continue

            raw = _read_2d_or_3d(flux_ds, flux_var, ti, FT)
            flux_native = _gridfed_convert_units(raw, units_str, lon_src, lat_src, R, FT)
            push!(monthly, flux_native)
            push!(time_hours, t_h)
        end
        close(ds)
    end

    length(monthly) > 0 || error("No valid time steps loaded from GridFED files")
    t0_month = start_date + Dates.Day(round(Int, time_hours[1] * 3600 / 86400))
    @info "GridFED: loaded $(length(monthly)) monthly steps (start_date=$start_date, first snapshot=$(Dates.monthname(t0_month)) $(Dates.year(t0_month)))"
    return monthly, lon_src, lat_src, time_hours
end

"""Load GridFED data and compute annual mean flux in kg/m²/s at native resolution."""
function _load_gridfed_annual_mean(filepath::String, year::Int, ::Type{FT}) where FT
    if isdir(filepath)
        nc_files = sort(filter(f -> endswith(f, ".nc"), readdir(filepath; join=true)))
        isempty(nc_files) && error("No NetCDF files in $filepath")
    else
        isfile(filepath) || error("GridFED path not found: $filepath")
        nc_files = [filepath]
    end

    # Get coordinates from first file
    ds0 = NCDataset(nc_files[1])
    lon_var = _find_coord_var(ds0, ["lon", "longitude", "x"])
    lat_var = _find_coord_var(ds0, ["lat", "latitude", "y"])
    lon_src = FT.(ds0[lon_var][:])
    lat_src = FT.(ds0[lat_var][:])
    close(ds0)

    Nlon = length(lon_src)
    Nlat = length(lat_src)
    R = FT(6.371e6)

    flux_sum = zeros(FT, Nlon, Nlat)
    n_steps = 0

    for fp in nc_files
        ds = NCDataset(fp)
        result = _find_gridfed_flux_var(ds)
        if result === nothing
            @warn "No flux variable in $(basename(fp)) — skipping"
            close(ds)
            continue
        end
        flux_ds, flux_var = result
        units_str = lowercase(get(flux_ds[flux_var].attrib, "units", ""))
        Nt = _get_time_dim(ds)

        for ti in 1:Nt
            raw = _read_2d_or_3d(flux_ds, flux_var, ti, FT)
            flux_native = _gridfed_convert_units(raw, units_str, lon_src, lat_src, R, FT)
            flux_sum .+= flux_native
            n_steps += 1
        end
        close(ds)
    end

    n_steps > 0 || error("No valid time steps loaded from GridFED files")
    flux_sum ./= FT(n_steps)

    @info "GridFED: computed annual mean from $n_steps monthly steps"
    return flux_sum, lon_src, lat_src
end


# =====================================================================
# 3. EDGAR v8.0 SF6 (with NOAA growth rate scaling)
# =====================================================================

"""
    load_edgar_sf6(filepath, target_grid; year, noaa_growth_file="", scale_year=year)

Load EDGAR v8.0 annual SF6 total emissions and conservatively regrid to
`target_grid`.

EDGAR SF6 has the same 0.1° format as EDGAR CO2 (Tonnes/year on grid).
If `noaa_growth_file` is provided, the emission field is scaled so that the
global total matches the NOAA-observed atmospheric SF6 growth rate for
`scale_year`.

The CATRINE protocol specifies using 2022 EDGAR data for both 2022 and 2023,
scaled to the NOAA SF6 growth rate each year.

Returns a `SurfaceFlux{FT}` with `species=:sf6`.
"""
function load_edgar_sf6(filepath::String,
                        target_grid::LatitudeLongitudeGrid{FT};
                        year::Int = 2022,
                        noaa_growth_file::String = "",
                        scale_year::Int = year) where FT
    isfile(filepath) || error("EDGAR SF6 file not found: $filepath")
    ds = NCDataset(filepath)

    lon_edgar = ds["lon"][:]
    lat_edgar = ds["lat"][:]

    # Try common EDGAR variable names for SF6
    flux_var = _find_flux_var(ds, ["emissions", "emi_sf6", "SF6_emissions",
                                   "emi", "TOTALS"])
    if flux_var === nothing
        varlist = join(keys(ds), ", ")
        close(ds)
        error("No SF6 emission variable found in $filepath. Available variables: $varlist")
    end

    emi_raw = nomissing(ds[flux_var][:, :], 0.0f0)  # Tonnes/year
    close(ds)

    # Convert Tonnes/year → kg/m²/s (same as EDGAR CO2 reader)
    R = target_grid.radius
    Δlon_e = FT(lon_edgar[2] - lon_edgar[1])
    Δlat_e = FT(lat_edgar[2] - lat_edgar[1])
    seconds_per_year = FT(365.25 * 24 * 3600)
    Nlon_e = length(lon_edgar)
    Nlat_e = length(lat_edgar)

    flux_native = Matrix{FT}(undef, Nlon_e, Nlat_e)
    @inbounds for j in 1:Nlat_e, i in 1:Nlon_e
        φ_s = FT(lat_edgar[j]) - Δlat_e / 2
        φ_n = FT(lat_edgar[j]) + Δlat_e / 2
        cell_area_e = R^2 * deg2rad(Δlon_e) * abs(sind(φ_n) - sind(φ_s))
        tonnes_per_year = FT(emi_raw[i, j])
        flux_native[i, j] = tonnes_per_year * FT(1000) / (seconds_per_year * cell_area_e)
    end

    # Longitude remapping
    lon_edgar_use, flux_native_use = if minimum(lon_edgar) < 0
        remap_lon_neg180_to_0_360(lon_edgar, flux_native)
    else
        FT.(lon_edgar), flux_native
    end

    flux_model = _conservative_regrid(flux_native_use, lon_edgar_use, lat_edgar,
                                       target_grid)

    # Apply NOAA SF6 growth rate scaling if file provided
    if !isempty(noaa_growth_file) && isfile(noaa_growth_file)
        scale = _noaa_sf6_scale_factor(noaa_growth_file, year, scale_year)
        flux_model .*= FT(scale)
        @info "EDGAR SF6: applied NOAA scale factor $(round(scale, digits=4)) " *
              "for year $scale_year (base year $year)"
    end

    total_native = sum(emi_raw) * 1000 / seconds_per_year  # kg/s
    total_model  = sum(flux_model[i, j] * cell_area(i, j, target_grid)
                       for j in 1:target_grid.Ny, i in 1:target_grid.Nx)
    @info "EDGAR SF6 loaded: native=$(round(Float64(total_native), digits=3)) kg/s, " *
          "regridded=$(round(Float64(total_model), digits=3)) kg/s"

    return SurfaceFlux(flux_model, :sf6, "EDGAR v8.0 SF6 $year";
                            molar_mass=M_SF6)
end

"""
Load EDGAR SF6 and regrid to cubed-sphere panels.

If a preprocessed binary file exists (same directory, named
`edgar_sf6_cs_c{Nc}_float32.bin`), loads from binary directly.
Otherwise regrids from the EDGAR NetCDF.
"""
function load_edgar_sf6(filepath::String, grid::CubedSphereGrid{FT};
                        year::Int = 2022,
                        noaa_growth_file::String = "",
                        scale_year::Int = year) where FT
    # Check for preprocessed binary first
    bin_path = _default_emission_bin_path(filepath, :sf6, grid.Nc)
    if !isempty(bin_path) && isfile(bin_path)
        @info "Loading preprocessed EDGAR SF6 binary: $bin_path"
        panels_vec, _, hdr = load_cs_emission_binary(bin_path, FT)
        flux_panels = panels_vec[1]  # SF6 is single-snapshot

        # Apply NOAA scaling at runtime (base scale may be baked into binary)
        if !isempty(noaa_growth_file) && isfile(noaa_growth_file)
            scale = FT(_noaa_sf6_scale_factor(noaa_growth_file, year, scale_year))
            flux_panels = ntuple(p -> flux_panels[p] .* scale, 6)
        end

        total_kgs = sum(sum(flux_panels[p] .* FT.(grid.Aᶜ[p])) for p in 1:6)
        @info "EDGAR SF6 from binary: $(round(Float64(total_kgs), digits=4)) kg/s"
        return SurfaceFlux(flux_panels, :sf6,
                                    "EDGAR v8.0 SF6 $year (binary)";
                                    molar_mass=M_SF6)
    end

    isfile(filepath) || error("EDGAR SF6 file not found: $filepath")
    ds = NCDataset(filepath)

    flux_var = _find_flux_var(ds, ["emissions", "emi_sf6", "SF6_emissions",
                                   "emi", "TOTALS"])
    if flux_var === nothing
        varlist = join(keys(ds), ", ")
        close(ds)
        error("No SF6 emission variable found in $filepath. Available variables: $varlist")
    end

    lon_edgar = FT.(ds["lon"][:])
    lat_edgar = FT.(ds["lat"][:])
    emi_raw   = FT.(replace(ds[flux_var][:, :], missing => zero(FT)))
    close(ds)

    flux_panels = regrid_edgar_to_cs(emi_raw, lon_edgar, lat_edgar, grid)

    # Apply NOAA SF6 growth rate scaling
    if !isempty(noaa_growth_file) && isfile(noaa_growth_file)
        scale = FT(_noaa_sf6_scale_factor(noaa_growth_file, year, scale_year))
        flux_panels = ntuple(p -> flux_panels[p] .* scale, 6)
    end

    return SurfaceFlux(flux_panels, :sf6,
                                "EDGAR v8.0 SF6 $year";
                                molar_mass=M_SF6)
end


"""
    _noaa_sf6_scale_factor(noaa_file, base_year, target_year) → Float64

Compute a scaling factor from the NOAA SF6 global growth rate file to
adjust `base_year` emissions for `target_year`.

The NOAA file (`sf6_gr_gl.txt`) contains columns: year, annual_increase (ppt/yr),
uncertainty. The scale factor is: growth_rate[target_year] / growth_rate[base_year].
If target_year == base_year, returns 1.0.
"""
function _noaa_sf6_scale_factor(noaa_file::String, base_year::Int, target_year::Int)
    base_year == target_year && return 1.0

    growth_rates = Dict{Int, Float64}()
    for line in eachline(noaa_file)
        stripped = strip(line)
        (isempty(stripped) || startswith(stripped, '#') || startswith(stripped, '%')) && continue
        parts = split(stripped)
        length(parts) >= 2 || continue
        yr = tryparse(Int, parts[1])
        gr = tryparse(Float64, parts[2])
        if yr !== nothing && gr !== nothing
            growth_rates[yr] = gr
        end
    end

    haskey(growth_rates, base_year) || @warn "NOAA SF6: no growth rate for base year $base_year"
    haskey(growth_rates, target_year) || @warn "NOAA SF6: no growth rate for target year $target_year"

    gr_base   = get(growth_rates, base_year, 1.0)
    gr_target = get(growth_rates, target_year, gr_base)

    return gr_target / gr_base
end


# =====================================================================
# 4. Zhang et al. 2021 radon-222 emissions
# =====================================================================

"""
    load_zhang_rn222(dirpath, target_grid; species=:rn222)

Load Zhang et al. (2021) 222Rn surface emissions from monthly NetCDF files
in `dirpath` and regrid to `target_grid`.

Expected file naming: `*Rn*<MM>.nc` or `*rn222*<MM>.nc` or a single file
containing all 12 months.

Units in Zhang et al.: atoms/cm²/s
Conversion to kg/m²/s: flux × (M_Rn222 / N_A) × 1e4

Returns a `TimeVaryingSurfaceFlux` for monthly data.
"""
function load_zhang_rn222(dirpath::String,
                          target_grid::LatitudeLongitudeGrid{FT};
                          species::Symbol = :rn222,
                          start_date::Date = Date(2021, 12, 1)) where FT
    flux_mats = Matrix{FT}[]
    time_hours = Float64[]

    if isfile(dirpath)
        # Single file with all months
        _load_rn222_file!(flux_mats, time_hours, dirpath,
                          target_grid, FT; start_date)
    elseif isdir(dirpath)
        # Directory with monthly files
        nc_files = sort(filter(readdir(dirpath; join=true)) do f
            bn = lowercase(basename(f))
            endswith(bn, ".nc") && (contains(bn, "rn") || contains(bn, "radon"))
        end; by=basename)

        isempty(nc_files) && error("No Rn222 NetCDF files found in $dirpath")

        for filepath in nc_files
            _load_rn222_file!(flux_mats, time_hours, filepath,
                              target_grid, FT; start_date)
        end
    else
        error("Zhang Rn222 path not found: $dirpath")
    end

    Nt_out = length(flux_mats)
    @info "Zhang Rn222: loaded $Nt_out time steps (start_date=$start_date)"

    if Nt_out == 0
        return SurfaceFlux(zeros(FT, Nx_m, Ny_m), species,
                                "Zhang Rn222 (empty)"; molar_mass=M_RN222)
    elseif Nt_out == 1
        return SurfaceFlux(flux_mats[1], species,
                                "Zhang Rn222"; molar_mass=M_RN222)
    else
        stack = _stack_matrices(flux_mats, Nx_m, Ny_m, FT)
        return TimeVaryingSurfaceFlux(stack, time_hours, species;
                                    label="Zhang Rn222 monthly",
                                    molar_mass=M_RN222)
    end
end

"""
Load Zhang Rn222 and regrid to cubed-sphere panels.
Returns a `SurfaceFlux` using the annual mean flux.
"""
function load_zhang_rn222(dirpath::String, grid::CubedSphereGrid{FT};
                          species::Symbol = :rn222,
                          start_date::Date = Date(2021, 12, 1)) where FT
    # Check for preprocessed binary
    bin_path = _default_emission_bin_path(dirpath, species, grid.Nc)
    if !isempty(bin_path) && isfile(bin_path)
        @info "Loading preprocessed Zhang Rn222 binary: $bin_path"
        panels_vec, time_hours, _ = load_cs_emission_binary(bin_path, FT)
        @info "Zhang Rn222 from binary: $(length(panels_vec)) monthly snapshots"
        return TimeVaryingSurfaceFlux(panels_vec, time_hours, species;
                                               label="Zhang Rn222 (binary)",
                                               molar_mass=M_RN222)
    end

    # Build lat-lon flux at 0.5° (native Zhang resolution) then regrid to CS
    if isfile(dirpath)
        filepath = dirpath
    elseif isdir(dirpath)
        nc_files = sort(filter(readdir(dirpath; join=true)) do f
            bn = lowercase(basename(f))
            endswith(bn, ".nc") && (contains(bn, "rn") || contains(bn, "radon"))
        end; by=basename)
        isempty(nc_files) && error("No Rn222 files found in $dirpath")
        filepath = nc_files[1]  # Use first file or annual mean
    else
        error("Zhang Rn222 path not found: $dirpath")
    end

    ds = NCDataset(filepath)
    lon_var = _find_coord_var(ds, ["lon", "longitude", "x"])
    lat_var = _find_coord_var(ds, ["lat", "latitude", "y"])
    lon_src = FT.(ds[lon_var][:])
    lat_src = FT.(ds[lat_var][:])

    rn_var = _find_flux_var(ds, ["Rn222", "emiss_Rn222", "Rn_land", "rn222",
                                 "radon", "222Rn", "Rn", "EMISS_Rn222", "rnemis"])
    if rn_var === nothing
        varlist = join(keys(ds), ", ")
        close(ds)
        error("No Rn222 variable in $filepath. Variables: $varlist")
    end

    # Unit-aware loading: check file units attribute
    raw = _read_2d_mean(ds, rn_var, FT)
    units_str = get(ds[rn_var].attrib, "units", "")
    if contains(units_str, "atom") || contains(units_str, "cm")
        # Legacy format: atoms/cm²/s → kg/m²/s
        atoms_to_kgm2s = FT(M_RN222 / N_AVOGADRO * 1e4)
        raw .*= atoms_to_kgm2s
        @info "Zhang Rn222: converted from atoms/cm²/s to kg/m²/s"
    else
        # Already in kg/m²/s (e.g., Zhang_Liu mass file)
        @info "Zhang Rn222: data already in $units_str"
    end
    close(ds)

    # Regrid to cubed-sphere via nearest-neighbor
    raw_sn, lat_use = ensure_south_to_north(raw, Float64.(lat_src))
    flux_panels = regrid_latlon_to_cs(FT.(raw_sn), FT.(lon_src), FT.(lat_use), grid)

    return SurfaceFlux(flux_panels, species,
                                "Zhang Rn222"; molar_mass=M_RN222)
end


# =====================================================================
# Internal helpers
# =====================================================================

"""Load one Rn222 NetCDF file, convert units, regrid, append to vectors.

Zhang Rn222 is climatological (12 monthly steps). Each month index `ti`
is mapped to the calendar month relative to `start_date`:
  month_dt = DateTime(start_year, ti, 1)   (wrapping into next year for ti > start_month)
Months before `start_date` are skipped.
"""
function _load_rn222_file!(flux_mats, time_hours, filepath,
                           target_grid::LatitudeLongitudeGrid{FT},
                           ::Type{FT};
                           start_date::Date = Date(2021, 12, 1)) where FT
    ds = NCDataset(filepath)

    lon_var = _find_coord_var(ds, ["lon", "longitude", "x"])
    lat_var = _find_coord_var(ds, ["lat", "latitude", "y"])
    lon_src = Float64.(ds[lon_var][:])
    lat_src = Float64.(ds[lat_var][:])

    rn_var = _find_flux_var(ds, ["Rn222", "emiss_Rn222", "Rn_land", "rn222",
                                 "radon", "222Rn", "Rn", "EMISS_Rn222", "rnemis"])
    if rn_var === nothing
        @warn "No Rn222 variable in $(basename(filepath)) — skipping"
        close(ds)
        return
    end

    # Determine unit conversion from file metadata
    units_str = get(ds[rn_var].attrib, "units", "")
    if contains(units_str, "atom") || contains(units_str, "cm")
        unit_factor = FT(M_RN222 / N_AVOGADRO * 1e4)  # atoms/cm²/s → kg/m²/s
        @info "Zhang Rn222 (lat-lon): converting from atoms/cm²/s to kg/m²/s"
    else
        unit_factor = one(FT)
        @info "Zhang Rn222 (lat-lon): data already in $units_str — no conversion"
    end

    Nt = _get_time_dim(ds)

    # Rn222 is climatological: ti=1 → Jan, ti=12 → Dec
    # Map each month to calendar time relative to start_date
    start_year = Dates.year(start_date)

    for ti in 1:Nt
        # Climatological month ti → calendar DateTime
        # For months before start_month in the same year, push to next year
        month_dt = DateTime(start_year, ti, 1)
        if month_dt < DateTime(start_date)
            month_dt = DateTime(start_year + 1, ti, 1)
        end
        t_h = Dates.value(month_dt - DateTime(start_date)) / 3.6e6
        t_h < 0 && continue

        raw = _read_2d_or_3d(ds, rn_var, ti, FT)
        raw .*= unit_factor

        raw_sn, lat_use = ensure_south_to_north(raw, lat_src)
        lon_use, raw_final = remap_lon_neg180_to_0_360(lon_src, raw_sn)

        # Zhang is 0.5° → use simple regrid
        flux_model = nearest_neighbor_regrid(raw_final, lon_use, lat_use, target_grid)
        push!(flux_mats, flux_model)
        push!(time_hours, t_h)
    end
    close(ds)
end

"""Read a 2D field averaging over time dimension if present."""
function _read_2d_mean(ds, varname, ::Type{FT}) where FT
    raw_var = ds[varname]
    if ndims(raw_var) == 3
        data = FT.(nomissing(raw_var[:, :, :], 0.0f0))
        Nt = size(data, 3)
        result = data[:, :, 1]
        for t in 2:Nt
            result .+= data[:, :, t]
        end
        result ./= FT(Nt)
        return result
    else
        return FT.(nomissing(raw_var[:, :], 0.0f0))
    end
end

"""Find a coordinate variable by trying common names."""
function _find_coord_var(ds, candidates::Vector{String})
    for name in candidates
        haskey(ds, name) && return name
    end
    tried = join(candidates, ", ")
    avail = join(keys(ds), ", ")
    error("No coordinate variable found. Tried: $tried. Available: $avail")
end

"""Find a flux/emission variable by trying common names."""
function _find_flux_var(ds, candidates::Vector{String})
    for name in candidates
        haskey(ds, name) && return name
    end
    return nothing
end

"""Get the number of time steps in a dataset (1 if no time dimension)."""
function _get_time_dim(ds)
    for dim_name in ["time", "month", "t"]
        haskey(ds.dim, dim_name) && return ds.dim[dim_name]
    end
    return 1
end

"""Read a 2D slice from a 2D or 3D variable."""
function _read_2d_or_3d(ds, varname, ti::Int, ::Type{FT}) where FT
    raw_var = ds[varname]
    if ndims(raw_var) >= 3
        return FT.(nomissing(raw_var[:, :, ti], zero(FT)))
    else
        return FT.(nomissing(raw_var[:, :], zero(FT)))
    end
end

"""Guess time step spacing in hours from dataset metadata."""
function _guess_dt_hours(ds, Nt)
    # Try to read time variable
    for tvar in ["time", "month", "t"]
        if haskey(ds, tvar)
            tvals = ds[tvar][:]
            if length(tvals) >= 2
                dt = tvals[2] - tvals[1]
                # Heuristic: if dt looks like days (1-366), convert to hours
                if dt isa Dates.Period
                    return Dates.value(Dates.Hour(dt))
                elseif 0.5 < Float64(dt) < 400
                    return Float64(dt) * 24.0  # assume days
                end
            end
        end
    end
    # Fallback: assume monthly
    return 365.25 / max(Nt, 1) * 24.0
end


# =====================================================================
# load_inventory dispatch for CATRINESource
# =====================================================================

"""
    load_inventory(src::CATRINESource, grid; kwargs...)

Load CATRINE emission data. Dispatches on `src.dataset`:
- `"cams_co2"`          → `load_cams_co2`
- `"gridfed_fossil_co2"` → `load_gridfed_fossil_co2`
- `"edgar_sf6"`         → `load_edgar_sf6`
- `"zhang_rn222"`       → `load_zhang_rn222`
"""
function load_inventory(src::CATRINESource, grid::LatitudeLongitudeGrid{FT};
                        year::Int = 2022, kwargs...) where FT
    filepath = expanduser(src.filepath)
    ds = src.dataset

    if ds == "cams_co2"
        return load_cams_co2(filepath, grid; year, kwargs...)
    elseif ds == "gridfed_fossil_co2"
        return load_gridfed_fossil_co2(filepath, grid; year, kwargs...)
    elseif ds == "edgar_sf6"
        return load_edgar_sf6(filepath, grid; year, kwargs...)
    elseif ds == "zhang_rn222"
        return load_zhang_rn222(filepath, grid; kwargs...)
    else
        error("Unknown CATRINE dataset: '$ds'. " *
              "Expected one of: cams_co2, gridfed_fossil_co2, edgar_sf6, zhang_rn222")
    end
end

function load_inventory(src::CATRINESource, grid::CubedSphereGrid{FT};
                        year::Int = 2022, kwargs...) where FT
    filepath = expanduser(src.filepath)
    ds = src.dataset

    sf = if ds == "gridfed_fossil_co2"
        load_gridfed_fossil_co2(filepath, grid; year, kwargs...)
    elseif ds == "edgar_sf6"
        load_edgar_sf6(filepath, grid; year, kwargs...)
    elseif ds == "zhang_rn222"
        load_zhang_rn222(filepath, grid; kwargs...)
    else
        error("CATRINE dataset '$ds' not yet supported on cubed-sphere grids. " *
              "Supported: gridfed_fossil_co2, edgar_sf6, zhang_rn222")
    end
    log_flux_integral(sf, grid)
    return sf
end
