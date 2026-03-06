# ---------------------------------------------------------------------------
# apply_surface_flux! for CombinedFlux and TimeVaryingSurfaceFlux{LatLonLayout}
# ---------------------------------------------------------------------------

function apply_surface_flux!(tracers::NamedTuple, source::CombinedFlux,
                             grid::LatitudeLongitudeGrid, dt)
    for component in source.components
        apply_surface_flux!(tracers, component, grid, dt)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# TimeVaryingSurfaceFlux{LatLonLayout} — time-dependent surface flux (lat-lon)
# ---------------------------------------------------------------------------

function apply_surface_flux!(tracers::NamedTuple, source::TimeVaryingSurfaceFlux{LatLonLayout, FT},
                             grid::LatitudeLongitudeGrid{FT}, dt) where FT
    name = source.species
    haskey(tracers, name) || return nothing

    c = tracers[name]
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    g = grid.gravity
    ti = source.current_idx
    mol_ratio = FT(M_AIR / source.molar_mass)

    @inbounds for j in 1:Ny, i in 1:Nx
        flux_ij = source.flux_data[i, j, ti]
        flux_ij == zero(FT) && continue

        area_ij = cell_area(i, j, grid)
        Δp_sfc  = Δz(Nz, grid)
        m_air   = Δp_sfc * area_ij / g

        ΔM_kg   = flux_ij * FT(dt) * area_ij
        Δc      = ΔM_kg * mol_ratio / m_air
        c[i, j, Nz] += Δc
    end
    return nothing
end
