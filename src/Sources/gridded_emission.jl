# ---------------------------------------------------------------------------
# GriddedEmission — 2D surface emission field regridded to model grid
# ---------------------------------------------------------------------------

const M_AIR  = 28.97e-3   # kg/mol
const M_CO2  = 44.01e-3   # kg/mol

"""
$(TYPEDEF)

Supertype for all gridded (spatially-resolved) emission sources.
Subtypes include `GriddedEmission` (lat-lon) and `CubedSphereEmission` (panel-based).
"""
abstract type AbstractGriddedEmission{FT} <: AbstractSource end

"""
$(TYPEDEF)

Regridded surface emission on a lat-lon model grid.

$(FIELDS)
"""
struct GriddedEmission{FT, A <: AbstractMatrix{FT}} <: AbstractGriddedEmission{FT}
    "emission flux [kg/m²/s] on model grid (Nx × Ny)"
    flux    :: A
    "tracer name (e.g., :co2)"
    species :: Symbol
    "human-readable label"
    label   :: String
end

"""
$(SIGNATURES)

Inject emission into the lowest atmospheric layer. Converts emission flux
[kg/m²/s] to mixing ratio change [ppm] using cell air mass.

    Δc_ppm = flux * dt * area * 1e6 * (M_air/M_CO2) / m_air_kg

where m_air_kg = Δp * area / g  (hydrostatic air mass of surface layer).
"""
function apply_surface_flux!(tracers::NamedTuple, source::GriddedEmission{FT},
                             grid::LatitudeLongitudeGrid{FT}, dt) where FT
    name = source.species
    haskey(tracers, name) || return nothing

    c = tracers[name]
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    g = grid.gravity

    @inbounds for j in 1:Ny, i in 1:Nx
        flux_ij = source.flux[i, j]
        flux_ij == 0 && continue

        area_ij = cell_area(i, j, grid)
        Δp_sfc  = Δz(Nz, grid)
        m_air   = Δp_sfc * area_ij / g

        ΔM_kg   = flux_ij * FT(dt) * area_ij
        Δc_ppm  = ΔM_kg * FT(1e6) * FT(M_AIR / M_CO2) / m_air
        c[i, j, Nz] += Δc_ppm
    end
    return nothing
end
