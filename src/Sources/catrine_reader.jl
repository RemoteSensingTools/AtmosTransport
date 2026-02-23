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

Returns a `TimeVaryingEmission` for multi-step files, or a `GriddedEmission`
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
        raw, lat_use = _ensure_south_to_north(raw, lat_src)
        lon_use, raw = _ensure_lon_0_360(lon_src, raw, FT)

        # Regrid: conservative for high-res (< 0.5°), simple otherwise
        Δlon = abs(lon_use[2] - lon_use[1])
        flux_model = if Δlon < 0.5
            _conservative_regrid(raw, FT.(lon_use), lat_use, target_grid, FT)
        else
            _simple_regrid(raw, Float64.(lon_use), Float64.(lat_use), target_grid, FT)
        end

        push!(flux_mats, flux_model)
        push!(time_hours, (ti - 1) * _guess_dt_hours(ds, Nt))
    end
    close(ds)

    Nt_out = length(flux_mats)
    @info "CAMS CO2: loaded $Nt_out time steps from $(basename(filepath))"

    if Nt_out == 1
        return GriddedEmission(flux_mats[1], species, "CAMS CO2 $year")
    else
        stack = _stack_matrices(flux_mats, Nx_m, Ny_m, FT)
        return TimeVaryingEmission(stack, time_hours, species;
                                    label="CAMS CO2 $year")
    end
end


# =====================================================================
# 2. GridFED fossil CO2 (GridFEDv2024.1)
# =====================================================================

"""
    load_gridfed_fossil_co2(filepath, target_grid; year, species=:fossil_co2)

Load GridFEDv2024.1 fossil fuel CO2 emissions from NetCDF and regrid to
`target_grid`.

GridFED provides monthly anthropogenic CO2 at 0.1° resolution.
Expected units: kg CO2/m²/s (if stored as flux density) or kg CO2/cell/month
(if stored as mass per cell). The reader auto-detects based on the `units`
attribute.

Returns a `TimeVaryingEmission` for monthly data.
"""
function load_gridfed_fossil_co2(filepath::String,
                                 target_grid::LatitudeLongitudeGrid{FT};
                                 year::Int = 2022,
                                 species::Symbol = :fossil_co2) where FT
    isfile(filepath) || error("GridFED file not found: $filepath")
    ds = NCDataset(filepath)

    lon_var = _find_coord_var(ds, ["lon", "longitude", "x"])
    lat_var = _find_coord_var(ds, ["lat", "latitude", "y"])
    lon_src = Float64.(ds[lon_var][:])
    lat_src = Float64.(ds[lat_var][:])
    Nlon_s = length(lon_src)
    Nlat_s = length(lat_src)

    # Discover emission variable
    flux_var = _find_flux_var(ds, ["emi_co2", "CO2_em_anthro", "fossil_co2",
                                   "emissions", "flux", "CO2_emissions",
                                   "co2_excl_short_cycle_org_c"])
    if flux_var === nothing
        varlist = join(keys(ds), ", ")
        error("No fossil CO2 variable found in $filepath. Available variables: $varlist")
    end

    units_str = lowercase(get(ds[flux_var].attrib, "units", ""))
    R = FT(target_grid.radius)
    Δlon_s = FT(abs(lon_src[2] - lon_src[1]))
    Δlat_s = FT(abs(lat_src[2] - lat_src[1]))
    sec_per_month = FT(365.25 / 12 * 86400)

    Nt = _get_time_dim(ds)
    Nx_m, Ny_m = target_grid.Nx, target_grid.Ny

    flux_mats = Matrix{FT}[]
    time_hours = Float64[]

    for ti in 1:Nt
        raw = _read_2d_or_3d(ds, flux_var, ti, FT)

        # Convert to kg CO2/m²/s depending on stored units
        flux_native = if contains(units_str, "kg") && (contains(units_str, "s") || contains(units_str, "sec"))
            # Already kg/m²/s — use directly
            raw
        elseif contains(units_str, "tonnes") || contains(units_str, "ton")
            # Tonnes/cell/year → kg/m²/s  (same pattern as EDGAR)
            _tonnes_per_year_to_kgm2s(raw, lon_src, lat_src, R, FT)
        elseif contains(units_str, "kg") && contains(units_str, "month")
            # kg/cell/month → kg/m²/s
            _mass_per_cell_to_kgm2s(raw, lon_src, lat_src, R, sec_per_month, FT)
        elseif contains(units_str, "kg") && contains(units_str, "year")
            # kg/cell/year → kg/m²/s
            _mass_per_cell_to_kgm2s(raw, lon_src, lat_src, R, FT(365.25 * 86400), FT)
        else
            # Assume flux density in kg/m²/s
            @warn "GridFED: unrecognized units '$units_str', assuming kg/m²/s"
            raw
        end

        flux_native_2, lat_use = _ensure_south_to_north(flux_native, lat_src)
        lon_use, flux_native_3 = _ensure_lon_0_360(lon_src, flux_native_2, FT)

        flux_model = _conservative_regrid(flux_native_3, FT.(lon_use), lat_use,
                                           target_grid, FT)
        push!(flux_mats, flux_model)
        push!(time_hours, (ti - 1) * (365.25 / Nt * 24.0))  # monthly spacing
    end
    close(ds)

    Nt_out = length(flux_mats)
    @info "GridFED fossil CO2: loaded $Nt_out time steps from $(basename(filepath))"

    if Nt_out == 1
        return GriddedEmission(flux_mats[1], species,
                                "GridFED fossil CO2 $year";
                                molar_mass=M_CO2)
    else
        stack = _stack_matrices(flux_mats, Nx_m, Ny_m, FT)
        return TimeVaryingEmission(stack, time_hours, species;
                                    label="GridFED fossil CO2 $year",
                                    molar_mass=M_CO2)
    end
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

Returns a `GriddedEmission{FT}` with `species=:sf6`.
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
        _remap_lon_0_360(lon_edgar, flux_native, FT)
    else
        FT.(lon_edgar), flux_native
    end

    flux_model = _conservative_regrid(flux_native_use, lon_edgar_use, lat_edgar,
                                       target_grid, FT)

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

    return GriddedEmission(flux_model, :sf6, "EDGAR v8.0 SF6 $year";
                            molar_mass=M_SF6)
end

"""
Load EDGAR SF6 and regrid to cubed-sphere panels.
"""
function load_edgar_sf6(filepath::String, grid::CubedSphereGrid{FT};
                        year::Int = 2022,
                        noaa_growth_file::String = "",
                        scale_year::Int = year) where FT
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

    return CubedSphereEmission(flux_panels, :sf6,
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

Returns a `TimeVaryingEmission` for monthly data.
"""
function load_zhang_rn222(dirpath::String,
                          target_grid::LatitudeLongitudeGrid{FT};
                          species::Symbol = :rn222) where FT
    # Conversion factor: atoms/cm²/s → kg/m²/s
    atoms_to_kgm2s = FT(M_RN222 / N_AVOGADRO * 1e4)

    Nx_m, Ny_m = target_grid.Nx, target_grid.Ny
    flux_mats = Matrix{FT}[]
    time_hours = Float64[]

    if isfile(dirpath)
        # Single file with all months
        _load_rn222_file!(flux_mats, time_hours, dirpath,
                          target_grid, atoms_to_kgm2s, FT)
    elseif isdir(dirpath)
        # Directory with monthly files
        nc_files = sort(filter(readdir(dirpath; join=true)) do f
            bn = lowercase(basename(f))
            endswith(bn, ".nc") && (contains(bn, "rn") || contains(bn, "radon"))
        end; by=basename)

        isempty(nc_files) && error("No Rn222 NetCDF files found in $dirpath")

        for filepath in nc_files
            _load_rn222_file!(flux_mats, time_hours, filepath,
                              target_grid, atoms_to_kgm2s, FT)
        end
    else
        error("Zhang Rn222 path not found: $dirpath")
    end

    Nt_out = length(flux_mats)
    @info "Zhang Rn222: loaded $Nt_out time steps"

    if Nt_out == 0
        return GriddedEmission(zeros(FT, Nx_m, Ny_m), species,
                                "Zhang Rn222 (empty)"; molar_mass=M_RN222)
    elseif Nt_out == 1
        return GriddedEmission(flux_mats[1], species,
                                "Zhang Rn222"; molar_mass=M_RN222)
    else
        stack = _stack_matrices(flux_mats, Nx_m, Ny_m, FT)
        return TimeVaryingEmission(stack, time_hours, species;
                                    label="Zhang Rn222 monthly",
                                    molar_mass=M_RN222)
    end
end

"""
Load Zhang Rn222 and regrid to cubed-sphere panels.
Returns a `CubedSphereEmission` using the annual mean flux.
"""
function load_zhang_rn222(dirpath::String, grid::CubedSphereGrid{FT};
                          species::Symbol = :rn222) where FT
    atoms_to_kgm2s = FT(M_RN222 / N_AVOGADRO * 1e4)

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
                                 "radon", "222Rn", "Rn", "EMISS_Rn222"])
    if rn_var === nothing
        varlist = join(keys(ds), ", ")
        close(ds)
        error("No Rn222 variable in $filepath. Variables: $varlist")
    end

    raw = _read_2d_mean(ds, rn_var, FT) .* atoms_to_kgm2s
    close(ds)

    # Regrid to cubed-sphere via nearest-neighbor
    raw_sn, lat_use = _ensure_south_to_north(raw, Float64.(lat_src))
    flux_panels = _regrid_latlon_to_cs(FT.(raw_sn), FT.(lon_src), FT.(lat_use), grid)

    return CubedSphereEmission(flux_panels, species,
                                "Zhang Rn222"; molar_mass=M_RN222)
end


# =====================================================================
# Internal helpers
# =====================================================================

"""Load one Rn222 NetCDF file, convert units, regrid, append to vectors."""
function _load_rn222_file!(flux_mats, time_hours, filepath,
                           target_grid::LatitudeLongitudeGrid{FT},
                           atoms_to_kgm2s, ::Type{FT}) where FT
    ds = NCDataset(filepath)

    lon_var = _find_coord_var(ds, ["lon", "longitude", "x"])
    lat_var = _find_coord_var(ds, ["lat", "latitude", "y"])
    lon_src = Float64.(ds[lon_var][:])
    lat_src = Float64.(ds[lat_var][:])

    rn_var = _find_flux_var(ds, ["Rn222", "emiss_Rn222", "Rn_land", "rn222",
                                 "radon", "222Rn", "Rn", "EMISS_Rn222"])
    if rn_var === nothing
        @warn "No Rn222 variable in $(basename(filepath)) — skipping"
        close(ds)
        return
    end

    Nt = _get_time_dim(ds)
    Nx_m, Ny_m = target_grid.Nx, target_grid.Ny

    for ti in 1:Nt
        raw = _read_2d_or_3d(ds, rn_var, ti, FT)
        raw .*= atoms_to_kgm2s  # atoms/cm²/s → kg/m²/s

        raw_sn, lat_use = _ensure_south_to_north(raw, lat_src)
        lon_use, raw_final = _ensure_lon_0_360(lon_src, raw_sn, FT)

        # Zhang is 0.5° → use simple regrid
        flux_model = _simple_regrid(raw_final, Float64.(lon_use),
                                     Float64.(lat_use), target_grid, FT)
        push!(flux_mats, flux_model)
        push!(time_hours, length(time_hours) * (365.25 / max(Nt, 12) * 24.0))
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

"""Flip array to ensure South→North latitude ordering."""
function _ensure_south_to_north(flux::Matrix, lat_src)
    if length(lat_src) > 1 && lat_src[1] > lat_src[end]
        return flux[:, end:-1:1], reverse(lat_src)
    end
    return flux, lat_src
end

"""Remap longitudes from -180:180 to 0:360 convention."""
function _ensure_lon_0_360(lon_src, flux::Matrix{FT}, ::Type{FT}) where FT
    if minimum(lon_src) < 0
        return _remap_lon_0_360(lon_src, flux, FT)
    end
    return FT.(lon_src), flux
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

"""Convert Tonnes/year per cell → kg/m²/s using cell areas."""
function _tonnes_per_year_to_kgm2s(raw::Matrix{FT}, lon, lat, R, ::Type{FT}) where FT
    Nlon, Nlat = size(raw)
    Δlon = FT(abs(lon[2] - lon[1]))
    Δlat = FT(abs(lat[2] - lat[1]))
    sec_per_yr = FT(365.25 * 86400)
    flux = similar(raw)
    @inbounds for j in 1:Nlat, i in 1:Nlon
        φ_s = FT(lat[j]) - Δlat / 2
        φ_n = FT(lat[j]) + Δlat / 2
        area = R^2 * deg2rad(Δlon) * abs(sind(φ_n) - sind(φ_s))
        flux[i, j] = raw[i, j] * FT(1000) / (sec_per_yr * area)
    end
    return flux
end

"""Convert mass/cell/period → kg/m²/s."""
function _mass_per_cell_to_kgm2s(raw::Matrix{FT}, lon, lat, R, sec_per_period, ::Type{FT}) where FT
    Nlon, Nlat = size(raw)
    Δlon = FT(abs(lon[2] - lon[1]))
    Δlat = FT(abs(lat[2] - lat[1]))
    flux = similar(raw)
    @inbounds for j in 1:Nlat, i in 1:Nlon
        φ_s = FT(lat[j]) - Δlat / 2
        φ_n = FT(lat[j]) + Δlat / 2
        area = R^2 * deg2rad(Δlon) * abs(sind(φ_n) - sind(φ_s))
        flux[i, j] = raw[i, j] / (sec_per_period * area)
    end
    return flux
end

"""Nearest-neighbor regrid from lat-lon to cubed-sphere panels."""
function _regrid_latlon_to_cs(flux_kgm2s::Matrix{FT}, lons::AbstractVector{FT},
                               lats::AbstractVector{FT},
                               grid::CubedSphereGrid{FT}) where FT
    Nc = grid.Nc
    Δlon = lons[2] - lons[1]
    Δlat = lats[2] - lats[1]
    Nlon = length(lons)
    Nlat = length(lats)

    flux_panels = ntuple(6) do p
        pf = zeros(FT, Nc, Nc)
        for j in 1:Nc, i in 1:Nc
            lon = mod(grid.λᶜ[p][i, j] + 180, 360) - 180
            lat = grid.φᶜ[p][i, j]
            ii = clamp(round(Int, (lon - lons[1]) / Δlon) + 1, 1, Nlon)
            jj = clamp(round(Int, (lat - lats[1]) / Δlat) + 1, 1, Nlat)
            pf[i, j] = flux_kgm2s[ii, jj]
        end
        pf
    end
    return flux_panels
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

    if ds == "edgar_sf6"
        return load_edgar_sf6(filepath, grid; year, kwargs...)
    elseif ds == "zhang_rn222"
        return load_zhang_rn222(filepath, grid; kwargs...)
    else
        error("CATRINE dataset '$ds' not yet supported on cubed-sphere grids. " *
              "Supported: edgar_sf6, zhang_rn222")
    end
end
