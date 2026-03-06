# ---------------------------------------------------------------------------
# Emission regridding: lat-lon → cubed-sphere panels
# ---------------------------------------------------------------------------

using ..Grids: CubedSphereGrid

"""
    regrid_edgar_to_cs(edgar_flux, edgar_lons, edgar_lats, grid; file_lons=nothing, file_lats=nothing)

Regrid a lat-lon emission field (Tonnes/yr per cell) to cubed-sphere panels.
Converts to kg/m²/s, then applies conservative area-weighted regridding via
`regrid_latlon_to_cs`.

When `file_lons` and `file_lats` are provided (legacy), they are ignored —
the unified regridder uses `grid.λᶜ/φᶜ` (which contain GMAO coordinates
when loaded).

Returns `NTuple{6, Matrix{FT}}` of panel flux fields.
"""
function regrid_edgar_to_cs(edgar_raw::Matrix{FT},
                             edgar_lons::AbstractVector{FT},
                             edgar_lats::AbstractVector{FT},
                             grid::CubedSphereGrid{FT};
                             file_lons=nothing, file_lats=nothing) where FT
    R = FT(grid.radius)

    # Convert Tonnes/year per cell → kg/m²/s
    flux_kgm2s = tonnes_per_year_to_kgm2s(edgar_raw, edgar_lons, edgar_lats, R)

    # Conservative area-weighted regrid to cubed sphere
    return regrid_latlon_to_cs(flux_kgm2s, edgar_lons, edgar_lats, grid)
end
