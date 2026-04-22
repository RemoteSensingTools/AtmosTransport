# ---------------------------------------------------------------------------
# TM5Convection — TM5-style four-field Tiedtke 1989 mass-flux convection.
#
# Plan 23 Commit 1: struct + dispatch stubs.
# Plan 23 Commit 4: real kernel launches via `_tm5_{…}_kernel!`
#                    in tm5_kernels.jl, which wraps
#                    `_tm5_solve_column!` (tm5_column_solve.jl)
#                    per thread.  All three topologies (LatLon,
#                    ReducedGaussian, CubedSphere) land in one
#                    commit per plan 23 principle "topology rollout
#                    policy" (no structured-first staging).
# ---------------------------------------------------------------------------

"""
    TM5Convection()

TM5-style convective transport operator. Four-field mass-flux scheme
following Tiedtke (1989) as implemented in TM5-4DVAR: two entrainment
and two detrainment fields (updraft + downdraft). The backward-Euler
transport matrix `conv1 = I - dt·D` is dense within the cloud window
and identity above; the solver factorizes the full `[1, Nz]` range
with partial-pivot Gaussian elimination and stores the pivot vector
for adjoint replay in plan 19.

No struct fields — the forcing arrays `(entu, detu, entd, detd)`
arrive via `TransportModel.convection_forcing.tm5_fields`, populated
each substep by `DrivenSimulation._refresh_forcing!` from
`sim.window.convection`.

# Basis convention

Plan 23 Commit 0 decision: `TM5Convection` is **basis-polymorphic**,
identical to [`CMFMCConvection`](@ref). The four forcing fields
must be on the same basis as `state.air_mass` (moist by upstream
Fortran convention and by the ec2tm preprocessor default; dry
requires a sibling preprocessor path, out of plan 23 scope).
See [`artifacts/plan23/basis_decision.md`](../../../artifacts/plan23/basis_decision.md).

# Fields required on `ConvectionForcing`

- `forcing.tm5_fields :: NamedTuple{(:entu, :detu, :entd, :detd)}`
  with all four arrays at layer centers in AtmosTransport
  orientation (k=1=TOA, k=Nz=surface). Units kg / m² / s.
  Shapes per topology:
  - Structured LatLon: `(Nx, Ny, Nz)` per field.
  - Face-indexed ReducedGaussian: `(ncells, Nz)` per field.
  - Panel-native CubedSphere: `NTuple{6, AbstractArray{FT, 3}}`
    per field, with per-panel shape `(Nc, Nc, Nz)`.

Orientation conversion + sign flip on `entd` happen in the
preprocessor ([`src/Preprocessing/tm5_convection_conversion.jl`](../../Preprocessing/tm5_convection_conversion.jl),
plan 23 Commit 3). The operator performs zero runtime orientation
gymnastics (plan 23 principle 1).

# Solver class

Partial-pivot Gaussian elimination on the full `[1, Nz]` range per
column (see [`artifacts/plan23/matrix_structure.md`](../../../artifacts/plan23/matrix_structure.md)
for the structure survey). Identity rows above the cloud window
factorize trivially; `lmc`-limited factorization is a Commit 7
optimization target.

Pivoting is kept even though the matrix is diagonally dominant by
construction (upstream Fortran comment says pivoting "not needed").
Per plan 23 principle 3, the pivot vector is stored in
[`TM5Workspace`](@ref) so plan 19 (adjoint) can replay the same
factorization with `trans='T'`.

# CFL sub-cycling

None. The backward-Euler matrix solve is unconditionally stable
for any `dt`, unlike `CMFMCConvection`'s forward-Euler two-pass
update which requires sub-cycling when the CMFMC profile is
strong. The kernel launches once and calls `synchronize(backend)`
once per `apply!`.
"""
struct TM5Convection <: AbstractConvectionOperator end

# =========================================================================
# Array-level entry: apply_convection!
# =========================================================================

"""
    apply_convection!(q_raw, air_mass, forcing::ConvectionForcing,
                       op::TM5Convection, dt, workspace::TM5Workspace,
                       grid::AtmosGrid) -> nothing

Array-level entry point — parallels the CMFMC contract at
[`operators.jl:70–89`](operators.jl#L70-L89). Dispatches on grid
mesh type and launches the matching KA kernel from
[`tm5_kernels.jl`](tm5_kernels.jl). Single `synchronize(backend)`
at the end (TM5 matrix solve is unconditionally stable; no
sub-cycling).
"""
function apply_convection!(q_raw::AbstractArray{FT, 4},
                            air_mass::AbstractArray{FT, 3},
                            forcing::ConvectionForcing,
                            ::TM5Convection,
                            dt::Real,
                            workspace::TM5Workspace,
                            grid::AtmosGrid{<:LatLonMesh}) where {FT}
    _assert_tm5_forcing(forcing)
    tm5 = forcing.tm5_fields
    Nx, Ny, _, _ = size(q_raw)
    backend = get_backend(q_raw)
    kernel = _tm5_column_kernel!(backend)
    kernel(q_raw, air_mass,
           tm5.entu, tm5.detu, tm5.entd, tm5.detd,
           workspace.conv1, workspace.pivots, workspace.cloud_dims,
           workspace.f_scratch, workspace.fu_scratch,
           workspace.amu_scratch, workspace.amd_scratch,
           FT(dt);
           ndrange = (Nx, Ny))
    synchronize(backend)
    return nothing
end

function apply_convection!(q_raw::AbstractArray{FT, 3},
                            air_mass::AbstractArray{FT, 2},
                            forcing::ConvectionForcing,
                            ::TM5Convection,
                            dt::Real,
                            workspace::TM5Workspace,
                            grid::AtmosGrid{<:ReducedGaussianMesh}) where {FT}
    _assert_tm5_forcing(forcing)
    tm5 = forcing.tm5_fields
    ncells = size(q_raw, 1)
    backend = get_backend(q_raw)
    kernel = _tm5_faceindexed_column_kernel!(backend)
    kernel(q_raw, air_mass,
           tm5.entu, tm5.detu, tm5.entd, tm5.detd,
           workspace.conv1, workspace.pivots, workspace.cloud_dims,
           workspace.f_scratch, workspace.fu_scratch,
           workspace.amu_scratch, workspace.amd_scratch,
           FT(dt);
           ndrange = ncells)
    synchronize(backend)
    return nothing
end

function apply_convection!(q_raw::NTuple{6, <:AbstractArray{FT, 4}},
                            air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                            forcing::ConvectionForcing,
                            ::TM5Convection,
                            dt::Real,
                            workspace::TM5Workspace,
                            grid::AtmosGrid{<:CubedSphereMesh}) where {FT}
    _assert_tm5_forcing(forcing)
    tm5 = forcing.tm5_fields
    mesh = grid.horizontal
    Nc = mesh.Nc
    Hp = mesh.Hp
    backend = get_backend(q_raw[1])
    kernel = _tm5_cs_panel_column_kernel!(backend)
    dt_ft = FT(dt)
    for p in 1:6
        kernel(q_raw[p], air_mass[p],
               tm5.entu[p], tm5.detu[p], tm5.entd[p], tm5.detd[p],
               workspace.conv1[p], workspace.pivots[p],
               workspace.cloud_dims[p],
               workspace.f_scratch[p], workspace.fu_scratch[p],
               workspace.amu_scratch[p], workspace.amd_scratch[p],
               Hp, dt_ft;
               ndrange = (Nc, Nc))
    end
    synchronize(backend)
    return nothing
end

# Clear error message when `forcing.tm5_fields === nothing` — this
# is the only "missing input" case because the Commit-1 validator
# in DrivenSimulation rejects it at window-load time.  Direct
# callers (e.g. tests that build forcing by hand) go through this
# guard.
function _assert_tm5_forcing(forcing::ConvectionForcing)
    forcing.tm5_fields === nothing &&
        throw(ArgumentError(
            "TM5Convection requires `forcing.tm5_fields` " *
            "(NamedTuple with :entu, :detu, :entd, :detd) to be populated. " *
            "Use `with_convection_forcing(model, ConvectionForcing(nothing, nothing, tm5_fields))` " *
            "or ensure the driver populates `window.convection.tm5_fields`."))
    return nothing
end

# =========================================================================
# State-level entry: apply!
# =========================================================================

"""
    apply!(state::CellState, forcing::ConvectionForcing, grid::AtmosGrid,
           op::TM5Convection, dt::Real; workspace) -> state

State-level delegate — matches the CMFMC contract at
[`CMFMCConvection.jl:296–316`](CMFMCConvection.jl#L296-L316).
Dispatches on grid mesh type plus the `Raw` parameter of
`CellState{B, A, Raw}`.
"""
function apply!(state::CellState{B, A, Raw, Names},
                forcing::ConvectionForcing,
                grid::AtmosGrid{<:LatLonMesh},
                op::TM5Convection,
                dt::Real;
                workspace::TM5Workspace) where {B, A, Raw <: AbstractArray{<:Any, 4}, Names}
    apply_convection!(state.tracers_raw, state.air_mass, forcing,
                        op, dt, workspace, grid)
    return state
end

function apply!(state::CellState{B, A, Raw, Names},
                forcing::ConvectionForcing,
                grid::AtmosGrid{<:ReducedGaussianMesh},
                op::TM5Convection,
                dt::Real;
                workspace::TM5Workspace) where {B, A, Raw <: AbstractArray{<:Any, 3}, Names}
    apply_convection!(state.tracers_raw, state.air_mass, forcing,
                        op, dt, workspace, grid)
    return state
end

function apply!(state::CubedSphereState{B},
                forcing::ConvectionForcing,
                grid::AtmosGrid{<:CubedSphereMesh},
                op::TM5Convection,
                dt::Real;
                workspace::TM5Workspace) where {B}
    apply_convection!(state.tracers_raw, state.air_mass, forcing,
                        op, dt, workspace, grid)
    return state
end
