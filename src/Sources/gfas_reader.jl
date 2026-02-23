# ---------------------------------------------------------------------------
# CAMS GFAS fire CO2 emission reader
#
# GFAS v1.4 provides daily wildfire CO2 emissions at 0.1° resolution.
# Downloaded from ADS: cams-global-fire-emissions-gfas
#
# Variable: "cofire" or "co2fire" — wildfire flux of CO2 [kg/m²/s]
# The data is already in the correct unit for our model.
#
# Since GFAS is at 0.1° and the model may be at 1°, conservative area-weighted
# regridding is used (same approach as EDGAR).
# ---------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Load CAMS GFAS fire CO2 emissions from NetCDF file(s) in `datadir` and
conservatively regrid to `target_grid`.

GFAS files are daily, 0.1° resolution, with variable `cofire` or `co2fire`
in units of kg/m²/s (positive = emission).

If `datadir` contains multiple daily files, they are concatenated into a
`TimeVaryingEmission`. A single file produces a `GriddedEmission`.

Arguments:
- `datadir`: directory containing GFAS NetCDF files (one per day or month)
- `target_grid`: model grid for conservative regridding
- `year`: year to load (filters files by name)
"""
function load_gfas_fire_flux(datadir::String,
                             target_grid::LatitudeLongitudeGrid{FT};
                             year::Int = 2024) where FT
    files = sort(filter(readdir(datadir; join=true)) do f
        bn = basename(f)
        endswith(bn, ".nc") && (contains(bn, "gfas") || contains(bn, "GFAS")) &&
            contains(bn, string(year))
    end; by=basename)

    # Also try a single file named with the year
    if isempty(files)
        single = filter(readdir(datadir; join=true)) do f
            endswith(basename(f), ".nc") && contains(basename(f), string(year))
        end
        files = sort(single; by=basename)
    end

    isempty(files) && error("No GFAS files for year $year in $datadir")
    @info "GFAS: found $(length(files)) files for $year"

    Nx_m, Ny_m = target_grid.Nx, target_grid.Ny
    flux_mats = Matrix{FT}[]
    time_hours = Float64[]

    for filepath in files
        ds = NCDataset(filepath)

        # Identify the CO2 fire variable
        co2_var = _find_gfas_co2_var(ds)
        if co2_var === nothing
            @warn "No CO2 fire variable found in $filepath — skipping"
            close(ds)
            continue
        end

        lon_gfas = Float64.(ds["longitude"][:])
        lat_gfas = Float64.(ds["latitude"][:])
        Nlon_g = length(lon_gfas)
        Nlat_g = length(lat_gfas)

        # Determine number of time steps
        nt = haskey(ds.dim, "time") ? ds.dim["time"] : 1

        for ti in 1:nt
            raw = if ndims(ds[co2_var]) == 3
                FT.(nomissing(ds[co2_var][:, :, ti], 0.0f0))
            else
                FT.(nomissing(ds[co2_var][:, :], 0.0f0))
            end

            # GFAS latitudes are typically N→S; flip to S→N
            if length(lat_gfas) > 1 && lat_gfas[1] > lat_gfas[end]
                raw = raw[:, end:-1:1]
                lat_use = reverse(lat_gfas)
            else
                lat_use = lat_gfas
            end

            # Longitude: GFAS uses 0:360 typically, but handle -180:180
            lon_use = lon_gfas
            if minimum(lon_gfas) < 0
                n = length(lon_gfas)
                split = findfirst(>=(0), lon_gfas)
                if split !== nothing
                    idx = vcat(split:n, 1:split-1)
                    lon_use = mod.(lon_gfas[idx], 360.0)
                    raw = raw[idx, :]
                end
            end

            # Conservative regridding from 0.1° to model grid
            flux_model = _conservative_regrid(raw, FT.(lon_use), lat_use,
                                              target_grid, FT)

            push!(flux_mats, flux_model)

            # Time in hours from start of year
            hour = (length(time_hours)) * 24.0  # daily
            push!(time_hours, hour)
        end

        close(ds)
    end

    Nt_out = length(flux_mats)
    @info "GFAS: loaded $Nt_out daily time steps"

    if Nt_out == 0
        return GriddedEmission(
            zeros(FT, Nx_m, Ny_m), :co2, "GFAS fire (empty)")
    elseif Nt_out == 1
        return GriddedEmission(
            flux_mats[1], :co2, "GFAS fire CO2 $year")
    else
        stack = _stack_matrices(flux_mats, Nx_m, Ny_m, FT)
        return TimeVaryingEmission(stack, time_hours, :co2;
                                    label="GFAS fire CO2 $year")
    end
end

function _find_gfas_co2_var(ds)
    for name in ["cofire", "co2fire", "wildfire_flux_of_carbon_dioxide",
                 "co2_fire", "CO2fire"]
        haskey(ds, name) && return name
    end
    # Fallback: search for any variable with "co2" and "fire" in attributes
    for name in keys(ds)
        v = ds[name]
        long_name = get(v.attrib, "long_name", "")
        if (contains(lowercase(long_name), "co2") || contains(lowercase(long_name), "carbon dioxide")) &&
           contains(lowercase(long_name), "fire")
            return name
        end
    end
    return nothing
end
