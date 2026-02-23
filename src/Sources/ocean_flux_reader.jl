# ---------------------------------------------------------------------------
# Jena CarboScope ocean CO2 flux reader
#
# The Jena CarboScope (oc_v2024) provides ocean-atmosphere CO2 flux on a
# 1x1 deg grid as NetCDF. Units: PgC/yr per grid cell.
#
# Download: https://www.bgc-jena.mpg.de/CarboScope/oc/
# Reference: Rödenbeck et al. (2013)
#
# Variables:
#   co2flux_ocean — ocean-atmosphere CO2 flux [PgC/yr per cell]
#   dxyp          — grid cell area [m²] (use for per-m² conversion)
#   lon, lat      — coordinates
#   mtime         — time (seconds since 2000-01-01)
# ---------------------------------------------------------------------------

const PGC_TO_KG = 1e12   # 1 PgC = 1e15 g = 1e12 kg

"""
$(TYPEDSIGNATURES)

Load Jena CarboScope ocean CO2 flux from `filepath` and regrid to `target_grid`.

Converts from PgC/yr/cell to kg(CO2)/m²/s:
    flux_kgCO2_m2_s = (flux_PgC_yr_cell × 1e12 × (44.01/12.011)) / (dxyp × 365.25 × 86400)

Returns a `TimeVaryingEmission` if the file contains multiple time steps,
or a `GriddedEmission` for a single-time-step file.
"""
function load_jena_ocean_flux(filepath::String,
                              target_grid::LatitudeLongitudeGrid{FT};
                              year::Union{Int, Nothing} = nothing) where FT
    isfile(filepath) || error("Jena CarboScope file not found: $filepath")
    ds = NCDataset(filepath)

    lon_jena = ds["lon"][:]
    lat_jena = ds["lat"][:]
    Nlon_j = length(lon_jena)
    Nlat_j = length(lat_jena)

    # Cell area from the file (m²), or compute if missing
    dxyp = haskey(ds, "dxyp") ? Float64.(ds["dxyp"][:, :]) : nothing

    # Time axis
    time_var = haskey(ds, "mtime") ? "mtime" : "time"
    if haskey(ds, time_var)
        time_raw = ds[time_var][:]
        Nt = length(time_raw)
    else
        Nt = 1
    end

    # If a specific year is requested, find matching time indices
    time_indices = 1:Nt
    if year !== nothing && haskey(ds, "myear")
        myear = ds["myear"][:]
        time_indices = findall(y -> floor(Int, y) == year, myear)
        if isempty(time_indices)
            @warn "No time steps matching year $year in Jena CarboScope file"
            close(ds)
            Nx_m, Ny_m = target_grid.Nx, target_grid.Ny
            return GriddedEmission(
                zeros(FT, Nx_m, Ny_m), :co2, "Jena CarboScope ocean (empty)")
        end
    end

    co2_var = haskey(ds, "co2flux_ocean") ? "co2flux_ocean" : "fgco2"
    Nx_m, Ny_m = target_grid.Nx, target_grid.Ny
    seconds_per_year = FT(365.25 * 86400)
    co2_per_c = FT(M_CO2 / M_C)  # 44.01/12.011 ≈ 3.664

    flux_mats = Matrix{FT}[]
    time_hours = Float64[]

    for (count, ti) in enumerate(time_indices)
        # Read raw flux: PgC/yr per cell, shape (Nlon, Nlat) or (Nlon, Nlat, Nt)
        raw = if ndims(ds[co2_var]) == 3
            FT.(nomissing(ds[co2_var][:, :, ti], 0.0f0))
        else
            FT.(nomissing(ds[co2_var][:, :], 0.0f0))
        end

        # Convert PgC/yr/cell → kg(CO2)/m²/s
        flux_native = Matrix{FT}(undef, Nlon_j, Nlat_j)
        @inbounds for j in 1:Nlat_j, i in 1:Nlon_j
            pgc = raw[i, j]
            kg_c = pgc * FT(PGC_TO_KG)
            kg_co2 = kg_c * co2_per_c
            area = if dxyp !== nothing
                FT(dxyp[i, j])
            else
                _compute_cell_area(lon_jena, lat_jena, i, j, target_grid.radius)
            end
            flux_native[i, j] = area > 0 ? kg_co2 / (area * seconds_per_year) : zero(FT)
        end

        # Longitude remapping if needed (-180:180 → 0:360)
        lon_use, flux_use = if minimum(lon_jena) < 0
            _remap_lon_0_360(lon_jena, flux_native, FT)
        else
            FT.(lon_jena), flux_native
        end

        # Regrid to model grid
        flux_model = _simple_regrid(flux_use, Float64.(lon_use), Float64.(lat_jena),
                                    target_grid, FT)
        push!(flux_mats, flux_model)

        # Daily time steps → hours
        dt_hours = if haskey(ds, "myear") && length(ds["myear"][:]) >= ti
            (ds["myear"][ti] - floor(ds["myear"][ti])) * 365.25 * 24.0
        else
            (count - 1) * 24.0  # assume daily
        end
        push!(time_hours, dt_hours)
    end

    close(ds)

    Nt_out = length(flux_mats)
    @info "Jena CarboScope: loaded $Nt_out time steps from $filepath"

    if Nt_out == 1
        return GriddedEmission(
            flux_mats[1], :co2, "Jena CarboScope ocean $(year)")
    else
        stack = _stack_matrices(flux_mats, Nx_m, Ny_m, FT)
        return TimeVaryingEmission(stack, time_hours, :co2;
                                    label="Jena CarboScope ocean $(year)")
    end
end

function _compute_cell_area(lon, lat, i, j, R)
    FT = Float64
    Δlon = length(lon) > 1 ? abs(lon[min(i+1, length(lon))] - lon[i]) : 1.0
    φ_s = lat[j] - abs(lat[min(j+1, length(lat))] - lat[j]) / 2
    φ_n = lat[j] + abs(lat[min(j+1, length(lat))] - lat[j]) / 2
    return FT(R)^2 * deg2rad(Δlon) * abs(sind(φ_n) - sind(φ_s))
end
