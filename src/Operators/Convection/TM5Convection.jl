# ---------------------------------------------------------------------------
# TM5Convection — TM5-style four-field Tiedtke 1989 mass-flux convection.
# Plan 23 Commit 1 — type + dispatch stubs only. Commit 2 ships the
# column solver `_tm5_solve_column!`. Commit 4 wires real kernels in
# `tm5_kernels.jl` and replaces these stubs.
# ---------------------------------------------------------------------------

"""
    TM5Convection()

TM5-style convective transport operator. Four-field mass-flux scheme
following Tiedtke (1989) as implemented in TM5-4DVAR: two entrainment
and two detrainment fields (updraft + downdraft). The backward-Euler
transport matrix `conv1 = I - dt·D` is dense within the cloud window
and identity above; the solver factorizes the active `lmc × lmc`
sub-block with partial-pivot Gaussian elimination and stores the
pivot vector for adjoint replay in plan 19.

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

Partial-pivot Gaussian elimination on the `lmc × lmc` active
sub-block (see [`artifacts/plan23/matrix_structure.md`](../../../artifacts/plan23/matrix_structure.md)
for the structure survey). `lmc` is diagnosed per column from
`(entu, detu, entd, detd)`; columns with `lmc == 0` (no
convection in that column) skip the factorization and the solve
is identity.

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

# Scope

Plan 23 Commit 4 ships `TM5Convection` on all three topologies
(LatLon, RG, CS) in a single commit — no structured-first staging.
Commit 1 (this file) ships only the type and dispatch scaffolding;
`apply!` errors with "not yet implemented — plan 23 Commit 4" so
the plumbing (validator, workspace, forcing) is exercised without
fake numerics.
"""
struct TM5Convection <: AbstractConvectionOperator end

# =========================================================================
# Commit-1 stub messages — replaced by real kernels in Commit 4.
# =========================================================================

const _TM5_COMMIT1_STUB_MSG =
    "TM5Convection kernel not yet implemented — plan 23 Commit 4. " *
    "The type hierarchy, workspace factory, and runtime plumbing are " *
    "live as of Commit 1; the partial-pivot GE column solver lands in " *
    "Commit 2, and the per-topology KA kernels plus full apply! dispatch " *
    "land in Commit 4. To run a live convection block today, install " *
    "CMFMCConvection() and preprocess with GEOS-FP CMFMC data; TM5Convection " *
    "becomes available once plan 23 Commit 4 ships."

# =========================================================================
# State-level entry: apply!
# =========================================================================

"""
    apply!(state::CellState, forcing::ConvectionForcing, grid::AtmosGrid,
           op::TM5Convection, dt::Real; workspace) -> state

State-level delegate. Commit 1 throws an ArgumentError; Commit 4
will unpack `state.tracers_raw` + `state.air_mass` + forcing and
dispatch to `apply_convection!` exactly like
[`CMFMCConvection.jl:296–316`](CMFMCConvection.jl#L296-L316).
Dispatch is on grid mesh type plus `Raw` parameter of
`CellState{B, A, Raw}`; CS uses `CubedSphereState`.
"""
function apply!(::CellState{B, A, Raw, Names},
                ::ConvectionForcing,
                ::AtmosGrid{<:LatLonMesh},
                ::TM5Convection,
                ::Real;
                workspace::TM5Workspace) where {B, A, Raw <: AbstractArray{<:Any, 4}, Names}
    _tm5_stub_throw(workspace)
end

function apply!(::CellState{B, A, Raw, Names},
                ::ConvectionForcing,
                ::AtmosGrid{<:ReducedGaussianMesh},
                ::TM5Convection,
                ::Real;
                workspace::TM5Workspace) where {B, A, Raw <: AbstractArray{<:Any, 3}, Names}
    _tm5_stub_throw(workspace)
end

function apply!(::CubedSphereState{B},
                ::ConvectionForcing,
                ::AtmosGrid{<:CubedSphereMesh},
                ::TM5Convection,
                ::Real;
                workspace::TM5Workspace) where {B}
    _tm5_stub_throw(workspace)
end

# Validates the workspace type (so users notice wiring bugs early)
# and then throws the Commit-1 stub message.
@noinline function _tm5_stub_throw(ws::TM5Workspace)
    throw(ArgumentError(
        "$_TM5_COMMIT1_STUB_MSG " *
        "(received workspace of type $(typeof(ws)) — plumbing is in place; " *
        "kernel is the missing piece.)"))
end

# =========================================================================
# Array-level entry: apply_convection!
# =========================================================================

"""
    apply_convection!(q_raw, air_mass, forcing::ConvectionForcing,
                       op::TM5Convection, dt, workspace::TM5Workspace,
                       grid::AtmosGrid) -> nothing

Array-level entry point — parallels the CMFMC contract at
[`operators.jl:70–89`](operators.jl#L70-L89). Commit 1 throws the
same ArgumentError as the state-level delegate; Commit 4 launches
the per-topology KA kernel that calls
`_tm5_solve_column!` (Commit 2) per column.
"""
function apply_convection!(::AbstractArray, ::AbstractArray, ::ConvectionForcing,
                            ::TM5Convection, ::Real,
                            ::TM5Workspace,
                            ::AtmosGrid)
    throw(ArgumentError(_TM5_COMMIT1_STUB_MSG))
end

function apply_convection!(::NTuple{6, <:AbstractArray}, ::NTuple{6, <:AbstractArray},
                            ::ConvectionForcing,
                            ::TM5Convection, ::Real,
                            ::TM5Workspace,
                            ::AtmosGrid)
    throw(ArgumentError(_TM5_COMMIT1_STUB_MSG))
end
