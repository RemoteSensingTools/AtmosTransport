# ---------------------------------------------------------------------------
# Emission regridding: lat-lon → cubed-sphere panels
# ---------------------------------------------------------------------------

using ..Grids: CubedSphereGrid

"""
    regrid_edgar_to_cs(edgar_flux, edgar_lons, edgar_lats, grid; file_lons=nothing, file_lats=nothing)

Regrid a lat-lon emission field (kg/m²/yr or Tonnes/yr) to cubed-sphere panels
via nearest-neighbor lookup. Converts to kg/m²/s.

When `file_lons` and `file_lats` are provided (Nc×Nc×6 arrays from GEOS-FP),
uses those coordinates instead of the gnomonic grid coordinates.

Returns `NTuple{6, Matrix{FT}}` of panel flux fields.
"""
function regrid_edgar_to_cs(edgar_raw::Matrix{FT},
                             edgar_lons::AbstractVector{FT},
                             edgar_lats::AbstractVector{FT},
                             grid::CubedSphereGrid{FT};
                             file_lons=nothing, file_lats=nothing) where FT
    Nc = grid.Nc
    R  = FT(grid.radius)
    Δlon = edgar_lons[2] - edgar_lons[1]
    Δlat = edgar_lats[2] - edgar_lats[1]
    sec_per_yr = FT(365.25 * 24 * 3600)
    Nlon_e = length(edgar_lons)
    Nlat_e = length(edgar_lats)
    use_file = file_lons !== nothing && file_lats !== nothing

    # Convert to kg/m²/s on the native grid
    flux_kgm2s = Matrix{FT}(undef, Nlon_e, Nlat_e)
    @inbounds for j in 1:Nlat_e, i in 1:Nlon_e
        φ_s = FT(edgar_lats[j]) - Δlat / 2
        φ_n = FT(edgar_lats[j]) + Δlat / 2
        cell_area_e = R^2 * deg2rad(Δlon) * abs(sind(φ_n) - sind(φ_s))
        flux_kgm2s[i, j] = FT(edgar_raw[i, j]) * FT(1000) / (sec_per_yr * cell_area_e)
    end

    # Compute total source mass rate (kg/s) on native grid
    total_source = zero(FT)
    @inbounds for j in 1:Nlat_e, i in 1:Nlon_e
        φ_s = FT(edgar_lats[j]) - Δlat / 2
        φ_n = FT(edgar_lats[j]) + Δlat / 2
        cell_area_e = R^2 * deg2rad(Δlon) * abs(sind(φ_n) - sind(φ_s))
        total_source += flux_kgm2s[i, j] * cell_area_e
    end

    # Nearest-neighbor assignment to each panel cell
    flux_panels = ntuple(6) do p
        pf = zeros(FT, Nc, Nc)
        for j in 1:Nc, i in 1:Nc
            lon = if use_file
                mod(FT(file_lons[i, j, p]) + FT(180), FT(360)) - FT(180)
            else
                mod(grid.λᶜ[p][i, j] + 180, 360) - 180
            end
            lat = use_file ? FT(file_lats[i, j, p]) : grid.φᶜ[p][i, j]
            ii = clamp(round(Int, (lon - edgar_lons[1]) / Δlon) + 1, 1, Nlon_e)
            jj = clamp(round(Int, (lat - edgar_lats[1]) / Δlat) + 1, 1, Nlat_e)
            pf[i, j] = flux_kgm2s[ii, jj]
        end
        pf
    end

    # Renormalize to conserve total mass exactly
    total_cs = zero(FT)
    for p in 1:6, j in 1:Nc, i in 1:Nc
        total_cs += flux_panels[p][i, j] * cell_area(i, j, grid; panel=p)
    end
    if abs(total_cs) > zero(FT)
        scale = total_source / total_cs
        for p in 1:6
            flux_panels[p] .*= scale
        end
        @info "CS emission renormalization: scale=$(round(scale, digits=6))"
    end

    return flux_panels
end
