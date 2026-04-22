# ---------------------------------------------------------------------------
# CMFMCConvection — GCHP RAS / Grell-Freitas convective transport.
# Plan 18 Commit 3, v5.1 §2.3 Decision 20 (basis-polymorphic),
# Decision 17 (well-mixed sub-cloud), Decision 21 (mandatory CFL
# sub-cycling), Decision 11 (no positivity clamp; adjoint-structure
# preserving).
# ---------------------------------------------------------------------------

"""
    CMFMCConvection()

GEOS-Chem RAS / Grell-Freitas convective transport operator.

No struct fields — the forcing arrays (`cmfmc`, optionally `dtrain`)
arrive via `TransportModel.convection_forcing` (plan 18 v5.1 §2.17
Decision 23), populated each substep by
`DrivenSimulation._refresh_forcing!` from `sim.window.convection`.

# Basis convention

Plan 18 Decision 20: the operator is basis-polymorphic. `cmfmc` and
`dtrain` must be on the SAME basis as `state.air_mass` (dry in
CATRINE usage, moist if the driver supplies moist forcing). The
driver is responsible for basis correction upstream (Commit 7).

# Fields required on `ConvectionForcing`

- `forcing.cmfmc :: AbstractArray{FT, 3}` at interfaces, shape
  `(Nx, Ny, Nz+1)`. Units kg / m² / s on the state's basis.
- `forcing.dtrain :: Union{AbstractArray{FT, 3}, Nothing}` at centers,
  shape `(Nx, Ny, Nz)`. When `nothing`, the kernel runs
  Tiedtke-style single-flux transport (DTRAIN-missing fallback per
  plan 18 v5.1 Decision 2).

Face-indexed ReducedGaussian uses `(ncell, Nz+1)` / `(ncell, Nz)`.
CubedSphere uses `NTuple{6}` of per-panel `(Nc, Nc, Nz+1)` /
`(Nc, Nc, Nz)` arrays.

# CFL sub-cycling (Decision 21)

The kernel sub-cycles internally based on the CMFMC profile:

    n_sub = max(1, ceil(max_over_grid(cmfmc × dt / bmass) / 0.5))

`n_sub` is cached per window in `workspace.cached_n_sub[]`; the
driver clears the cache on window roll via
`invalidate_cmfmc_cache!(workspace)`. Bit-exact: one call with `dt`
matches `n_sub` manual calls with `sdt = dt / n_sub`.

# Well-mixed sub-cloud layer (Decision 17)

Applies GCHP's pressure-weighted well-mixed treatment below cloud
base (`convection_mod.F90:742-782`). Absent in legacy Julia;
deliberate improvement for surface-source tracers. See the kernel
Pass-0 block in `cmfmc_kernels.jl`.

# Scope

`CMFMCConvection` now supports structured LatLon, face-indexed
ReducedGaussian, and panel-native CubedSphere runtime state. The CS
path keeps forcing panel-native too: `forcing.cmfmc` and
`forcing.dtrain` are `NTuple{6}` payloads loaded by the CS transport
driver and applied column-locally on the halo-free panel interior.

# Adjoint path (not shipped in plan 18)

The forward operator is linear in tracer mixing ratio (verified by
the Tier A adjoint-identity test in `test/test_cmfmc_convection.jl`).
NO positivity clamp is applied inside the kernel (Decision 11 +
adjoint addendum §D): the two-term tendency
`cmfmc · (q_above - q_env) + dtrain · (qc - q_env)` stays linear,
and tiny negativities that arise from inconsistent met data are
absorbed by the global mass fixer. A future `Plan 19: Adjoint
operator suite` kernel reverses the two-pass order (tendency first,
then updraft accumulation) with transposed coefficients; the
four-term scavenging-restoring form is a wet-deposition follow-up
(GCHP `convection_mod.F90:DO_RAS_CLOUD_CONVECTION` §5 in the GCHP
source tree).
"""
struct CMFMCConvection <: AbstractConvectionOperator end

# =========================================================================
# Array-level entry: apply_convection!
# =========================================================================

"""
    apply_convection!(q_raw, air_mass, forcing::ConvectionForcing,
                       op::CMFMCConvection, dt, workspace::CMFMCWorkspace,
                       grid::AtmosGrid) -> nothing

Apply one transport step of CMFMC + optional DTRAIN convection on
the structured lat-lon grid. Operates on `q_raw` and `air_mass` in
place; `workspace.qc_scratch` is used as per-column updraft storage
and `workspace.cached_n_sub` / `workspace.cache_valid` manage the
CFL cache.

`dt` is the block-level step length; the kernel internally
sub-cycles `n_sub` times with `sdt = dt / n_sub`.
"""
function _cmfmc_dtrain_array(cmfmc::AbstractArray{FT, 3},
                             dtrain::Nothing,
                             air_mass::AbstractArray{FT, 3}) where FT
    Nx_c, Ny_c, Nz_c = size(air_mass)
    darr = Array{FT}(undef, Nx_c, Ny_c, Nz_c)
    @inbounds for k in 1:Nz_c, j in 1:Ny_c, i in 1:Nx_c
        darr[i, j, k] = max(zero(FT), cmfmc[i, j, k + 1] - cmfmc[i, j, k])
    end
    return darr
end

_cmfmc_dtrain_array(cmfmc, dtrain, air_mass) = dtrain

function _cmfmc_dtrain_array(cmfmc::NTuple{6, <:AbstractArray{FT, 3}},
                             dtrain::Nothing,
                             air_mass::NTuple{6, <:AbstractArray{FT, 3}}) where FT
    return ntuple(6) do p
        Nx_c, Ny_c, Nzp1 = size(cmfmc[p])
        darr = Array{FT}(undef, Nx_c, Ny_c, Nzp1 - 1)
        @inbounds for k in 1:(Nzp1 - 1), j in 1:Ny_c, i in 1:Nx_c
            darr[i, j, k] = max(zero(FT), cmfmc[p][i, j, k + 1] - cmfmc[p][i, j, k])
        end
        darr
    end
end

function apply_convection!(q_raw::AbstractArray{FT, 4},
                            air_mass::AbstractArray{FT, 3},
                            forcing::ConvectionForcing,
                            op::CMFMCConvection,
                            dt,
                            workspace::CMFMCWorkspace,
                            grid::AtmosGrid{<:LatLonMesh}) where FT
    forcing.cmfmc === nothing && throw(ArgumentError(
        "CMFMCConvection requires `forcing.cmfmc` to be populated; got nothing. " *
        "Install via `TransportModel.convection_forcing` or " *
        "`with_convection_forcing(model, ConvectionForcing(cmfmc, dtrain, nothing))`."))

    cmfmc = forcing.cmfmc

    # DTRAIN-missing Tiedtke-style fallback (plan 18 v5.1 Decision 2):
    # when `forcing.dtrain === nothing`, derive per-layer detrainment
    # from the updraft-flux divergence:
    #
    #     dtrain[k] = max(0, cmfmc[k+1] - cmfmc[k])
    #
    # (cmfmc[k+1] = flux at the BOTTOM of layer k going up;
    #  cmfmc[k]   = flux at the TOP of layer k going up.
    # A decreasing cmfmc as k decreases — i.e. flux goes up through
    # the column and diminishes — implies the updraft is shedding
    # mass into the environment at that layer. Without an explicit
    # DTRAIN field, this diagnosed detrainment closes the mass
    # balance. Matches the legacy lat-lon fallback behavior of the
    # earlier Julia port (git commit ec2d2c0,
    # `src_legacy/Convection/ras_convection.jl:365-370`), which
    # delegated to a single-flux Tiedtke operator that made the same
    # diagnosis implicitly.)
    dtrain_arr = _cmfmc_dtrain_array(cmfmc, forcing.dtrain, air_mass)
    # Always true at the kernel level once we've closed the mass
    # balance via derived or provided dtrain.
    has_dtrain_val = true

    Nx, Ny, Nz, Nt = size(q_raw)
    cell_areas_y = workspace.cell_metrics === nothing ?
                   cell_areas_by_latitude(grid.horizontal) :
                   workspace.cell_metrics

    # Cache the CFL sub-step count per met window (Decision 21).
    n_sub = _get_or_compute_n_sub!(workspace, cmfmc, air_mass,
                                    cell_areas_y, dt)
    sdt = FT(dt) / FT(n_sub)

    backend = get_backend(q_raw)
    kernel = _cmfmc_column_kernel!(backend, (16, 16))

    for _ in 1:n_sub
        kernel(q_raw, air_mass, cmfmc, dtrain_arr, cell_areas_y,
               workspace.qc_scratch,
               Nz, Nt, sdt, Val(has_dtrain_val);
               ndrange = (Nx, Ny))
    end
    synchronize(backend)
    return nothing
end

function apply_convection!(q_raw::AbstractArray{FT, 3},
                            air_mass::AbstractArray{FT, 2},
                            forcing::ConvectionForcing,
                            op::CMFMCConvection,
                            dt,
                            workspace::CMFMCWorkspace,
                            grid::AtmosGrid{<:ReducedGaussianMesh}) where FT
    forcing.cmfmc === nothing && throw(ArgumentError(
        "CMFMCConvection requires `forcing.cmfmc` to be populated; got nothing. " *
        "Install via `TransportModel.convection_forcing` or " *
        "`with_convection_forcing(model, ConvectionForcing(cmfmc, dtrain, nothing))`."))

    cell_areas = workspace.cell_metrics
    cell_areas === nothing && throw(ArgumentError(
        "Face-indexed CMFMCConvection requires `workspace.cell_metrics` " *
        "to carry per-cell areas. Build the model with `with_convection` " *
        "or construct `CMFMCWorkspace(air_mass; cell_metrics=...)`."))

    cmfmc = forcing.cmfmc
    dtrain_arr = _cmfmc_dtrain_array(cmfmc, forcing.dtrain, air_mass)
    has_dtrain_val = true

    ncell, Nz, Nt = size(q_raw)
    n_sub = _get_or_compute_n_sub!(workspace, cmfmc, air_mass,
                                   cell_areas, dt)
    sdt = FT(dt) / FT(n_sub)

    backend = get_backend(q_raw)
    kernel = _cmfmc_faceindexed_column_kernel!(backend, 256)

    for _ in 1:n_sub
        kernel(q_raw, air_mass, cmfmc, dtrain_arr, cell_areas,
               workspace.qc_scratch,
               Nz, Nt, sdt, Val(has_dtrain_val);
               ndrange = ncell)
    end
    synchronize(backend)
    return nothing
end

function apply_convection!(q_raw::AbstractArray{FT, 3},
                            air_mass::AbstractArray{FT, 2},
                            ::ConvectionForcing,
                            ::CMFMCConvection,
                            dt, workspace, grid::AtmosGrid) where FT
    throw(ArgumentError(
        "`CMFMCConvection` supports face-indexed state only on " *
        "`ReducedGaussianMesh`; got $(typeof(grid.horizontal))."))
end

function apply_convection!(q_raw::NTuple{6, <:AbstractArray{FT, 4}},
                           air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                           forcing::ConvectionForcing,
                           op::CMFMCConvection,
                           dt,
                           workspace::CMFMCWorkspace,
                           grid::AtmosGrid{<:CubedSphereMesh}) where FT
    forcing.cmfmc === nothing && throw(ArgumentError(
        "CMFMCConvection requires `forcing.cmfmc` to be populated; got nothing. " *
        "Install via `TransportModel.convection_forcing` or " *
        "`with_convection_forcing(model, ConvectionForcing(cmfmc, dtrain, nothing))`."))

    cell_areas = workspace.cell_metrics
    cell_areas === nothing && throw(ArgumentError(
        "Cubed-sphere CMFMCConvection requires `workspace.cell_metrics` " *
        "to carry per-panel cell-area matrices. Build the model with " *
        "`with_convection` or construct `CMFMCWorkspace(air_mass; cell_metrics=...)`."))

    cmfmc = forcing.cmfmc
    dtrain_arr = _cmfmc_dtrain_array(cmfmc, forcing.dtrain, air_mass)
    has_dtrain_val = true

    mesh = grid.horizontal
    Hp = mesh.Hp
    Nc = mesh.Nc
    Nz = size(q_raw[1], 3)
    Nt = size(q_raw[1], 4)
    n_sub = _get_or_compute_n_sub!(workspace, cmfmc, air_mass,
                                   cell_areas, dt)
    sdt = FT(dt) / FT(n_sub)

    backend = get_backend(q_raw[1])
    kernel = _cmfmc_cs_panel_column_kernel!(backend, (16, 16))

    for _ in 1:n_sub
        for p in 1:6
            kernel(q_raw[p], air_mass[p], cmfmc[p], dtrain_arr[p], cell_areas[p],
                   workspace.qc_scratch[p],
                   Nz, Nt, sdt, Hp, Val(has_dtrain_val);
                   ndrange = (Nc, Nc))
        end
    end
    synchronize(backend)
    return nothing
end

# =========================================================================
# State-level entry: apply!
# =========================================================================

"""
    apply!(state::CellState, forcing::ConvectionForcing, grid::AtmosGrid,
           op::CMFMCConvection, dt::Real; workspace) -> state

State-level delegate. Unpacks `state.tracers_raw` and
`state.air_mass`, then forwards to `apply_convection!`. Requires an
allocated `CMFMCWorkspace` — `DrivenSimulation` construction (plan
18 Commit 8) allocates this per Decision 26.
"""
function apply!(state::CellState{B, A, Raw, Names},
                forcing::ConvectionForcing,
                grid::AtmosGrid{<:LatLonMesh},
                op::CMFMCConvection,
                dt::Real;
                workspace::CMFMCWorkspace) where {B, A, Raw <: AbstractArray{<:Any, 4}, Names}
    apply_convection!(state.tracers_raw, state.air_mass, forcing, op, dt,
                       workspace, grid)
    return state
end

function apply!(state::CellState{B, A, Raw, Names},
                forcing::ConvectionForcing,
                grid::AtmosGrid{<:ReducedGaussianMesh},
                op::CMFMCConvection,
                dt::Real;
                workspace::CMFMCWorkspace) where {B, A, Raw <: AbstractArray{<:Any, 3}, Names}
    apply_convection!(state.tracers_raw, state.air_mass, forcing, op, dt,
                      workspace, grid)
    return state
end

function apply!(state::CellState{B, A, Raw, Names},
                forcing::ConvectionForcing, grid::AtmosGrid,
                op::CMFMCConvection, dt::Real;
                workspace = nothing) where {B, A, Raw <: AbstractArray{<:Any, 3}, Names}
    throw(ArgumentError(
        "`CMFMCConvection` supports face-indexed state only on " *
        "`ReducedGaussianMesh`; got $(typeof(grid.horizontal))."))
end

function apply!(state::CubedSphereState{B},
                forcing::ConvectionForcing,
                grid::AtmosGrid{<:CubedSphereMesh},
                op::CMFMCConvection,
                dt::Real;
                workspace::CMFMCWorkspace) where {B}
    apply_convection!(state.tracers_raw, state.air_mass, forcing, op, dt,
                      workspace, grid)
    return state
end
