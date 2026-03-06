# ---------------------------------------------------------------------------
# apply_surface_flux! for SurfaceFlux{LatLonLayout} (lat-lon grids)
# ---------------------------------------------------------------------------

"""
$(SIGNATURES)

Inject surface flux into the lowest atmospheric layer on a lat-lon grid.
Converts mass flux [kg/m²/s] to mixing ratio change [mol/mol] using cell air mass.

    Δc = flux * dt * area * (M_air/M_tracer) / m_air_kg

where m_air_kg = Δp * area / g  (hydrostatic air mass of surface layer).
"""
function apply_surface_flux!(tracers::NamedTuple, source::SurfaceFlux{LatLonLayout, FT},
                             grid::LatitudeLongitudeGrid{FT}, dt) where FT
    name = source.species
    haskey(tracers, name) || return nothing

    c = tracers[name]
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    g = grid.gravity
    mol_ratio = FT(M_AIR / source.molar_mass)

    @inbounds for j in 1:Ny, i in 1:Nx
        flux_ij = source.flux[i, j]
        flux_ij == 0 && continue

        area_ij = cell_area(i, j, grid)
        Δp_sfc  = Δz(Nz, grid)
        m_air   = Δp_sfc * area_ij / g

        ΔM_kg   = flux_ij * FT(dt) * area_ij
        Δc      = ΔM_kg * mol_ratio / m_air
        c[i, j, Nz] += Δc
    end
    return nothing
end
