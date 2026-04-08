# ---------------------------------------------------------------------------
# ERA5 dry flux builder
#
# Two paths:
#   1. PreprocessedERA5Driver — reads preprocessed binary (am, bm, cm, m already
#      in mass-flux form from the v4 spectral preprocessor)
#   2. RawERA5Driver — reads u, v, lnsp, q and computes fluxes on the fly
#
# Path (1) is the production path. Path (2) is for testing / on-the-fly.
# ---------------------------------------------------------------------------

"""
    PreprocessedERA5Driver <: AbstractMassFluxMetDriver

Reads preprocessed ERA5 mass fluxes from binary files (v4 spectral preprocessor).
The binary already contains am, bm, cm, m, ps scaled to half-timestep.
"""
struct PreprocessedERA5Driver <: AbstractMassFluxMetDriver
    n_windows :: Int
    dt        :: Float64
    n_substeps :: Int
end

supports_moisture(::PreprocessedERA5Driver) = true

"""
    build_dry_fluxes!(fluxes, cell_mass, met, grid, driver::PreprocessedERA5Driver, closure)

For preprocessed ERA5, the fluxes are already in the binary. This function
just ensures cm is consistent (for DiagnoseVerticalFromHorizontal closure)
and computes dry air mass from Δp and humidity.
"""
function build_dry_fluxes!(fluxes::AbstractStructuredFaceFluxState,
                           cell_mass::AbstractArray,
                           met::MetState,
                           grid::AtmosGrid{<:LatLonMesh},
                           driver::PreprocessedERA5Driver,
                           closure::DiagnoseVerticalFromHorizontal)
    mesh = grid.horizontal
    vc = grid.vertical
    Nx, Ny = mesh.Nx, mesh.Ny
    Nz = n_levels(vc)
    g = grid.gravity
    areas = cell_areas_by_latitude(mesh)

    ps = met.ps
    q  = met.q

    # Build dry air mass: m_dry = Δp × area / g × (1 - q)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dp_k = level_thickness(vc, k, ps[i, j])
        cell_mass[i, j, k] = dp_k * areas[j] / g * (one(eltype(q)) - q[i, j, k])
    end

    # Diagnose cm from horizontal convergence if needed
    bt = eltype(cell_mass)[b_diff(vc, k) for k in 1:Nz]
    am, bm, cm = fluxes.am, fluxes.bm, fluxes.cm
    @inbounds for j in 1:Ny, i in 1:Nx
        pit = zero(eltype(cm))
        for k in 1:Nz
            pit += am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
        end
        acc = zero(eltype(cm))
        cm[i, j, 1] = acc
        for k in 1:Nz
            conv_k = am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
            acc += conv_k - bt[k] * pit
            cm[i, j, k+1] = acc
        end
    end

    return nothing
end

"""
    build_air_mass!(cell_mass, met, grid, driver::PreprocessedERA5Driver)

Compute dry air mass from surface pressure and humidity.
"""
function build_air_mass!(cell_mass::AbstractArray,
                         met::MetState,
                         grid::AtmosGrid{<:LatLonMesh},
                         driver::PreprocessedERA5Driver)
    mesh = grid.horizontal
    vc = grid.vertical
    Nx, Ny = mesh.Nx, mesh.Ny
    Nz = n_levels(vc)
    g = grid.gravity
    areas = cell_areas_by_latitude(mesh)

    ps = met.ps
    q  = met.q

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dp_k = level_thickness(vc, k, ps[i, j])
        cell_mass[i, j, k] = dp_k * areas[j] / g * (one(eltype(q)) - q[i, j, k])
    end
    return nothing
end

export PreprocessedERA5Driver
