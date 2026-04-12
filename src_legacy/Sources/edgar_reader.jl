# ---------------------------------------------------------------------------
# EDGAR reader — load and regrid EDGAR v8.0 CO2 emission inventories
# ---------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Read EDGAR v8.0 CO2 total emissions from `filepath` and conservatively
regrid to `target_grid`.

EDGAR v8.0 provides annual total emissions in Tonnes on a 0.1°×0.1° grid.
This function:
1. Reads the native emission field
2. Converts from Tonnes/year to kg/m²/s
3. Conservative area-weighted regridding to the model grid

Returns a `SurfaceFlux{FT}`.
"""
function load_edgar_co2(filepath::String, target_grid::LatitudeLongitudeGrid{FT};
                        year::Int = 2022, species::Symbol = :co2) where FT
    ds = NCDataset(filepath)

    lon_edgar = ds["lon"][:]
    lat_edgar = ds["lat"][:]
    emi_raw   = nomissing(ds["emissions"][:, :], 0.0f0)  # Tonnes/year (3600×1800)
    close(ds)

    Δlon_e = FT(lon_edgar[2] - lon_edgar[1])
    Δlat_e = FT(lat_edgar[2] - lat_edgar[1])

    # Convert Tonnes/year → kg/m²/s
    R = target_grid.radius
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

    # If EDGAR uses -180:180 but the model grid uses 0:360, remap longitudes.
    lon_edgar_use, flux_native_use = if minimum(lon_edgar) < 0
        remap_lon_neg180_to_0_360(lon_edgar, flux_native)
    else
        FT.(lon_edgar), flux_native
    end

    flux_model = _conservative_regrid(flux_native_use, lon_edgar_use, lat_edgar,
                                      target_grid)

    total_native = sum(emi_raw) * 1000 / seconds_per_year  # kg/s
    total_model  = sum(flux_model[i, j] * cell_area(i, j, target_grid)
                       for j in 1:target_grid.Ny, i in 1:target_grid.Nx)
    ratio_raw = total_model / total_native

    # Renormalize to conserve mass exactly (compensates boundary-cell losses)
    if abs(total_model) > zero(FT)
        flux_model .*= FT(total_native / total_model)
    end
    total_corrected = sum(flux_model[i, j] * cell_area(i, j, target_grid)
                          for j in 1:target_grid.Ny, i in 1:target_grid.Nx)
    ratio_corr = total_corrected / total_native
    sp = uppercase(string(species))
    @info "EDGAR $sp loaded: native total=$(round(total_native, digits=1)) kg/s, " *
          "pre-renorm ratio=$(round(ratio_raw, digits=6)), " *
          "post-renorm ratio=$(round(ratio_corr, digits=6))"

    return SurfaceFlux(flux_model, species, "EDGAR v8.0 $sp $year")
end

# _conservative_regrid and helpers now live in regrid_utils.jl
# The alias `_conservative_regrid = conservative_regrid_ll` provides backward compatibility.
