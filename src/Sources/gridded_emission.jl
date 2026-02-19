# ---------------------------------------------------------------------------
# GriddedEmission — 2D surface emission field regridded to model grid
# ---------------------------------------------------------------------------

const M_AIR  = 28.97e-3   # kg/mol
const M_CO2  = 44.01e-3   # kg/mol

"""
    GriddedEmission{FT, A} <: AbstractSource

Regridded surface emission on the model grid.

# Fields
- `flux`    — emission flux [kg/m²/s] on model grid (Nx × Ny)
- `species` — tracer name (e.g., :co2)
- `label`   — human-readable label
"""
struct GriddedEmission{FT, A <: AbstractMatrix{FT}} <: AbstractSource
    flux    :: A
    species :: Symbol
    label   :: String
end

"""
    apply_surface_flux!(tracers, source::GriddedEmission, grid, dt)

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
