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

Returns a `GriddedEmission{FT}`.
"""
function load_edgar_co2(filepath::String, target_grid::LatitudeLongitudeGrid{FT};
                        year::Int = 2022) where FT
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

    flux_model = _conservative_regrid(flux_native, lon_edgar, lat_edgar,
                                      target_grid, FT)

    total_native = sum(emi_raw) * 1000 / seconds_per_year  # kg/s
    total_model  = sum(flux_model[i, j] * cell_area(i, j, target_grid)
                       for j in 1:target_grid.Ny, i in 1:target_grid.Nx)
    ratio = total_model / total_native
    @info "EDGAR CO2 loaded: native total=$(round(total_native, digits=1)) kg/s, " *
          "regridded total=$(round(total_model, digits=1)) kg/s, " *
          "ratio=$(round(ratio, digits=4))"

    return GriddedEmission{FT, typeof(flux_model)}(flux_model, :co2, "EDGAR v8.0 CO2 $year")
end

"""
$(SIGNATURES)

Conservative area-weighted regridding: accumulates mass (flux × source area) into
model cells, then divides by model cell area to get flux density [kg/m²/s].
"""
function _conservative_regrid(flux_native::Matrix{FT},
                              lon_src::Vector, lat_src::Vector,
                              grid::LatitudeLongitudeGrid{FT},
                              ::Type{FT}) where FT
    Nx_m, Ny_m = grid.Nx, grid.Ny
    Δlon_s = FT(lon_src[2] - lon_src[1])
    Δlat_s = FT(lat_src[2] - lat_src[1])
    R = FT(grid.radius)

    mass_model = zeros(FT, Nx_m, Ny_m)

    λᶠ_cpu = grid.λᶠ_cpu
    φᶠ_cpu = grid.φᶠ_cpu

    for js in eachindex(lat_src), is in eachindex(lon_src)
        f = flux_native[is, js]
        f == zero(FT) && continue

        φ_s_south = FT(lat_src[js]) - Δlat_s / 2
        φ_s_north = FT(lat_src[js]) + Δlat_s / 2
        area_src = R^2 * deg2rad(Δlon_s) * abs(sind(φ_s_north) - sind(φ_s_south))
        emission_rate = f * area_src

        lon_s_west = FT(lon_src[is]) - Δlon_s / 2
        lon_s_east = FT(lon_src[is]) + Δlon_s / 2

        im_start = _find_model_index_lon(lon_s_west, λᶠ_cpu)
        im_end   = _find_model_index_lon(lon_s_east - FT(1e-10), λᶠ_cpu)
        jm_start = _find_model_index_lat(φ_s_south, φᶠ_cpu)
        jm_end   = _find_model_index_lat(φ_s_north - FT(1e-10), φᶠ_cpu)

        (im_start === nothing || jm_start === nothing) && continue
        (im_end === nothing || jm_end === nothing) && continue

        for jm in jm_start:jm_end, im in im_start:im_end
            (im < 1 || im > Nx_m || jm < 1 || jm > Ny_m) && continue
            overlap_lon = max(zero(FT), min(lon_s_east, λᶠ_cpu[im + 1]) -
                                        max(lon_s_west, λᶠ_cpu[im]))
            overlap_lat_s = max(φ_s_south, φᶠ_cpu[jm])
            overlap_lat_n = min(φ_s_north, φᶠ_cpu[jm + 1])
            overlap_lat_n <= overlap_lat_s && continue
            frac_lon = overlap_lon / Δlon_s
            frac_lat = abs(sind(overlap_lat_n) - sind(overlap_lat_s)) /
                       abs(sind(φ_s_north) - sind(φ_s_south))
            mass_model[im, jm] += emission_rate * frac_lon * frac_lat
        end
    end

    # Divide by model cell area to get flux density
    flux_model = zeros(FT, Nx_m, Ny_m)
    for jm in 1:Ny_m, im in 1:Nx_m
        a = cell_area(im, jm, grid)
        flux_model[im, jm] = mass_model[im, jm] / a
    end

    return flux_model
end

function _find_model_index_lon(lon_val, λᶠ)
    n = length(λᶠ) - 1
    for i in 1:n
        if lon_val >= λᶠ[i] && lon_val < λᶠ[i + 1]
            return i
        end
    end
    return nothing
end

function _find_model_index_lat(lat_val, φᶠ)
    n = length(φᶠ) - 1
    for i in 1:n
        if lat_val >= φᶠ[i] && lat_val < φᶠ[i + 1]
            return i
        end
    end
    return nothing
end
