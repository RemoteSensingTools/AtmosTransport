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
The binary already contains prepared transport fields (`am`, `bm`, `cm`, `m`, `ps`) in the stored runtime contract expected by `src_v2`.
"""
struct PreprocessedERA5Driver <: AbstractMassFluxMetDriver
    n_windows :: Int
    dt        :: Float64
    n_substeps :: Int
end

supports_moisture(::PreprocessedERA5Driver) = true

# ---------------------------------------------------------------------------
# Core interface: moist fluxes → dry fluxes + dry air mass
# ---------------------------------------------------------------------------

"""
    build_dry_fluxes!(dry_fluxes, cell_mass, moist_fluxes, m_moist, qv, grid,
                      driver::PreprocessedERA5Driver, closure)

Convert moist-basis fluxes from the binary reader into dry-basis fluxes.

# Arguments
- `dry_fluxes  :: StructuredFaceFluxState{DryMassFluxBasis}` — output
- `cell_mass   :: AbstractArray{FT,3}` — filled with dry air mass [kg]
- `moist_fluxes :: StructuredFaceFluxState{MoistMassFluxBasis}` — from binary reader
- `m_moist     :: AbstractArray{FT,3}` — moist cell mass from binary
- `qv          :: AbstractArray{FT,3}` — specific humidity (cell-centered)
- `grid        :: AtmosGrid{<:LatLonMesh}` — mesh + vertical coordinate
- `driver      :: PreprocessedERA5Driver`
- `closure     :: DiagnoseVerticalFromHorizontal`

# Phase A approximation
Horizontal fluxes: `am_dry ≈ am_moist`, `bm_dry ≈ bm_moist`.

The ERA5 spectral preprocessor builds fluxes from VO/D, so am/bm are
effectively `ρ_total × u × A_face`. The exact conversion would be
`F_dry = F_moist × (1 - q_face)` where `q_face` is interpolated to faces.
The correction is ~0.3% and is deferred to a follow-up with conservation
diagnostics to measure the impact.

Vertical flux `cm_dry` is diagnosed from horizontal convergence of
`am_dry`/`bm_dry` using the B-coefficient splitting (TM5 dynam0 convention).
"""
function build_dry_fluxes!(dry_fluxes::StructuredFaceFluxState{DryMassFluxBasis},
                           cell_mass::AbstractArray{FT, 3},
                           moist_fluxes::StructuredFaceFluxState{MoistMassFluxBasis},
                           m_moist::AbstractArray{FT, 3},
                           qv::AbstractArray{FT, 3},
                           grid::AtmosGrid{<:LatLonMesh},
                           driver::PreprocessedERA5Driver,
                           closure::DiagnoseVerticalFromHorizontal) where FT
    mesh = grid.horizontal
    vc = grid.vertical
    _Nx, _Ny = mesh.Nx, mesh.Ny
    _Nz = n_levels(vc)

    am_in  = moist_fluxes.am
    bm_in  = moist_fluxes.bm
    am_out = dry_fluxes.am
    bm_out = dry_fluxes.bm
    cm_out = dry_fluxes.cm

    # --- Dry cell mass: m_dry = m_moist × (1 - qv) ---
    @inbounds for k in 1:_Nz, j in 1:_Ny, i in 1:_Nx
        cell_mass[i, j, k] = m_moist[i, j, k] * (one(FT) - qv[i, j, k])
    end

    # --- Horizontal fluxes: Phase A approximation (am_dry ≈ am_moist) ---
    copyto!(am_out, am_in)
    copyto!(bm_out, bm_in)

    # --- Diagnose cm from horizontal convergence ---
    diagnose_cm_from_continuity_vc!(cm_out, am_out, bm_out, vc, _Nx, _Ny, _Nz)

    return nothing
end

# ---------------------------------------------------------------------------
# Legacy interface (MetState-based) — backward compatible with existing tests
# ---------------------------------------------------------------------------

"""
    build_dry_fluxes!(fluxes, cell_mass, met, grid, driver::PreprocessedERA5Driver, closure)

Legacy interface using `MetState`. For the preprocessed path, fluxes in `met`
are already populated; this method computes dry air mass and diagnoses cm.

Note: this method accepts `AbstractStructuredFaceFluxState` for backward
compatibility with the MetState-based workflow. Use the typed version with
explicit moist/dry flux states for new code.
"""
function build_dry_fluxes!(fluxes::AbstractStructuredFaceFluxState,
                           cell_mass::AbstractArray,
                           met::MetState,
                           grid::AtmosGrid{<:LatLonMesh},
                           driver::PreprocessedERA5Driver,
                           closure::DiagnoseVerticalFromHorizontal)
    mesh = grid.horizontal
    vc = grid.vertical
    _Nx, _Ny = mesh.Nx, mesh.Ny
    _Nz = n_levels(vc)
    g = grid.gravity
    areas = cell_areas_by_latitude(mesh)

    ps = met.ps
    q  = met.q

    @inbounds for k in 1:_Nz, j in 1:_Ny, i in 1:_Nx
        dp_k = level_thickness(vc, k, ps[i, j])
        cell_mass[i, j, k] = dp_k * areas[j] / g * (one(eltype(q)) - q[i, j, k])
    end

    diagnose_cm_from_continuity_vc!(fluxes.cm, fluxes.am, fluxes.bm, vc, _Nx, _Ny, _Nz)

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
    _Nx, _Ny = mesh.Nx, mesh.Ny
    _Nz = n_levels(vc)
    g = grid.gravity
    areas = cell_areas_by_latitude(mesh)

    ps = met.ps
    q  = met.q

    @inbounds for k in 1:_Nz, j in 1:_Ny, i in 1:_Nx
        dp_k = level_thickness(vc, k, ps[i, j])
        cell_mass[i, j, k] = dp_k * areas[j] / g * (one(eltype(q)) - q[i, j, k])
    end
    return nothing
end

export PreprocessedERA5Driver
