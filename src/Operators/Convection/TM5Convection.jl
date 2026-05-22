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
    TM5Convection(; tile_workspace_gib = 1.0)

TM5-style convective transport operator. Four-field mass-flux scheme
following Tiedtke (1989) as implemented in TM5-4DVAR: two entrainment
and two detrainment fields (updraft + downdraft). The backward-Euler
transport matrix `conv1 = I - dt·D` is dense within the cloud window
and identity above; the solver assembles and factorizes only the active
lower-right cloud block and stores the pivot vector for adjoint replay
in plan 19.

The forcing arrays `(entu, detu, entd, detd)` arrive via
`TransportModel.convection_forcing.tm5_fields`, populated each
substep by `DrivenSimulation._refresh_forcing!` from
`sim.window.convection`.

# Memory budget

`tile_workspace_gib` (binary GiB) is the per-topology target for
the TM5 column-tile workspace.
`_convection_workspace_for(::TM5Convection, ...)` reads this field
and derives a tile size `B` via [`derive_tile_columns`](@ref); the
[`TM5Workspace`](@ref) then allocates a single `(Nz, Nz, B)`
matrix slab plus matching pivot / cloud-dim / `amu` / `amd`
slabs. A larger budget means fewer kernel launches per substep at
the cost of larger GPU working set; the default 1.0 GiB fits all
production resolutions through C720/L137 with slack on H100. The
tile machinery is the load-bearing storage change: the workspace no longer scales with
`N_cells × Nz²`.

# Basis convention

`TM5Convection` is **basis-polymorphic**, identical to
[`CMFMCConvection`](@ref). The four forcing fields
must be on the same basis as `state.air_mass` (moist by upstream
Fortran convention and by the ec2tm preprocessor default; dry
requires a sibling preprocessor path).

# Fields required on `ConvectionForcing`

- `forcing.tm5_fields :: NamedTuple{(:entu, :detu, :entd, :detd)}`
  with all four arrays at layer centers in AtmosTransport
  orientation (k=1=TOA, k=Nz=surface). Units kg / m² / s.
  Shapes per topology:
  - Structured LatLon: `(Nx, Ny, Nz)` per field.
  - Face-indexed ReducedGaussian: `(ncells, Nz)` per field.
  - Panel-native CubedSphere: `NTuple{6, AbstractArray{FT, 3}}`
    per field, with per-panel shape `(Nc, Nc, Nz)`.

Orientation conversion + sign flip on `entd` happen in the preprocessor
(`src/Preprocessing/tm5_convection_conversion.jl`). The operator performs
zero runtime orientation gymnastics.

# Solver class

Partial-pivot Gaussian elimination on the active lower-right block per
column (see `test/test_tm5_sparsity_above_icltop.jl` for the structure
survey).
Identity rows above the effective cloud top and the lower-left zero
quadrant are skipped by both the factorization and the tracer solve.

Pivoting is kept even though the matrix is diagonally dominant by
construction (upstream Fortran comment says pivoting "not needed").
Per plan 23 principle 3, the pivot vector is stored in
[`TM5Workspace`](@ref) so plan 19 (adjoint) can replay the same
factorization with `trans='T'`.

# CFL sub-cycling

None. The backward-Euler matrix solve is unconditionally stable
for any `dt`, unlike `CMFMCConvection`'s forward-Euler two-pass
update which requires sub-cycling when the CMFMC profile is
strong. The kernel launches once per tile and calls
`synchronize(backend)` once per `apply!`.
"""
struct TM5Convection{FT} <: AbstractConvection
    tile_workspace_gib :: FT
    # Backend-portable workgroup-collaborative LU+solve path. When
    # `true`, `apply_convection!` dispatches to the collaborative
    # kernels in `tm5_kernels.jl` (`_tm5_*_column_collab_kernel!`)
    # instead of the per-thread serial path. Bit-exact to the legacy
    # path within Float32 rounding (~7e-7 max abs deviation on a C180
    # panel); measured 8–11× faster on NVIDIA L40S. Designed for KA
    # portability — same kernel compiles for CUDA and Metal — but
    # see the deployment note below before flipping the default.
    #
    # Default off so existing runs are bit-identical until the user
    # opts in. Production deployment as default still needs:
    #   1. A Metal-machine sanity run (only CUDA L40S has been
    #      measured to date).
    #   2. Confirmation that the per-workgroup `@localmem` footprint
    #      (~31.6 KB at Nz=85, Nt≤8) fits Metal's threadgroup-memory
    #      ceiling on the target chip.
    #   3. The kernel rejects Nz > 91 and Nt > 8 at the host side;
    #      configs outside that envelope keep the legacy path.
    # See `docs/memos/TM5_CONVECTION_AGENTLOOP_SYNTHESIS.md` for the
    # full empirical record and `_tm5_collab_supports` for the
    # runtime envelope check.
    use_collab_lu :: Bool
end

TM5Convection(; tile_workspace_gib::Real = 1.0, use_collab_lu::Bool = false) =
    TM5Convection{typeof(tile_workspace_gib)}(tile_workspace_gib, use_collab_lu)

# =========================================================================
# Array-level entry: apply_convection!
# =========================================================================

# Decide between the collaborative path (workgroup-shared LU in
# `@localmem`) and the legacy per-thread path. The collab kernel
# only supports a bounded `(Nz, Nt)` envelope — `_tm5_collab_supports`
# returns `false` outside that envelope — AND it hardcodes `Float32`
# throughout the `@localmem` allocations and arithmetic. If the
# caller's `FT` is anything else (the F64 curry/A100 runs are the
# common case), we MUST fall back to the per-thread kernel rather
# than silently truncating to F32 inside the shared-memory copy.
#
# We also exclude the KA CPU backend: the collab kernel uses
# `@uniform g = @index(Group)` which KA's CPU-side lowering does
# not accept (it produces `UndefVarError` at runtime). The CPU
# backend is for unit-test convenience anyway; correctness lives
# on the GPU backends (CUDA, Metal).
#
# The host call sites pass `eltype(q_raw)` as `FT` and
# `get_backend(q_raw)` as `backend`.
@inline _use_collab_path(op::TM5Convection, Nz::Integer, Nt::Integer,
                          ::Type{FT}, backend) where FT =
    op.use_collab_lu && FT === Float32 && _tm5_collab_supports(Nz, Nt) &&
    !(backend isa KernelAbstractions.CPU)

"""
    apply_convection!(q_raw, air_mass, forcing::ConvectionForcing,
                       op::TM5Convection, dt, workspace::TM5Workspace,
                       grid::AtmosGrid) -> nothing

Array-level entry point — parallels the CMFMC contract at
`operators.jl:70-89`. Dispatches on grid
mesh type and launches the matching KA kernel from
`tm5_kernels.jl`. Single `synchronize(backend)`
at the end (TM5 matrix solve is unconditionally stable; no
sub-cycling).
"""
function apply_convection!(q_raw::AbstractArray{FT, 4},
                            air_mass::AbstractArray{FT, 3},
                            forcing::ConvectionForcing,
                            op::TM5Convection,
                            dt::Real,
                            workspace::TM5Workspace,
                            grid::AtmosGrid{<:LatLonMesh}) where {FT}
    _assert_tm5_forcing(forcing)
    tm5 = forcing.tm5_fields
    cell_areas_y = _tm5_require_cell_metrics(workspace, "LatLon")
    Nx, Ny, Nz, Nt = size(q_raw)
    N_total = Nx * Ny
    backend = get_backend(q_raw)
    dt_ft   = FT(dt)
    if _use_collab_path(op, Nz, Nt, FT, backend)
        kernel = _tm5_column_collab_kernel!(backend, _TM5_COLLAB_WG_SIZE)
        kernel(q_raw, air_mass,
               tm5.entu, tm5.detu, tm5.entd, tm5.detd,
               cell_areas_y,
               Int(Nx), Int(Nt), Float32(dt), Val(Int(Nz));
               ndrange = _TM5_COLLAB_WG_SIZE * N_total,
               workgroupsize = _TM5_COLLAB_WG_SIZE)
    else
        kernel = _tm5_column_kernel!(backend)
        # Tile loop — KA stream ordering serializes panels safely
        # because the workspace is shared. `synchronize(backend)`
        # after the loop, not per launch.
        B = size(workspace.conv1, 3)
        for tile_off in 0:B:(N_total - 1)
            n = min(B, N_total - tile_off)
            kernel(q_raw, air_mass,
                   tm5.entu, tm5.detu, tm5.entd, tm5.detd,
                   cell_areas_y,
                   workspace.conv1, workspace.pivots, workspace.cloud_dims,
                   workspace.f_scratch,
                   workspace.amu_scratch, workspace.amd_scratch,
                   Int(tile_off), Int(Nx), dt_ft;
                   ndrange = n)
        end
    end
    synchronize(backend)
    return nothing
end

function apply_convection!(q_raw::AbstractArray{FT, 3},
                            air_mass::AbstractArray{FT, 2},
                            forcing::ConvectionForcing,
                            op::TM5Convection,
                            dt::Real,
                            workspace::TM5Workspace,
                            grid::AtmosGrid{<:ReducedGaussianMesh}) where {FT}
    _assert_tm5_forcing(forcing)
    tm5     = forcing.tm5_fields
    cell_areas = _tm5_require_cell_metrics(workspace, "face-indexed TM5Convection")
    N_total, Nz, Nt = size(q_raw)
    backend = get_backend(q_raw)
    dt_ft   = FT(dt)
    if _use_collab_path(op, Nz, Nt, FT, backend)
        kernel = _tm5_faceindexed_column_collab_kernel!(backend, _TM5_COLLAB_WG_SIZE)
        kernel(q_raw, air_mass,
               tm5.entu, tm5.detu, tm5.entd, tm5.detd,
               cell_areas,
               Int(Nt), Float32(dt), Val(Int(Nz));
               ndrange = _TM5_COLLAB_WG_SIZE * N_total,
               workgroupsize = _TM5_COLLAB_WG_SIZE)
    else
        kernel = _tm5_faceindexed_column_kernel!(backend)
        B = size(workspace.conv1, 3)
        for tile_off in 0:B:(N_total - 1)
            n = min(B, N_total - tile_off)
            kernel(q_raw, air_mass,
                   tm5.entu, tm5.detu, tm5.entd, tm5.detd,
                   cell_areas,
                   workspace.conv1, workspace.pivots, workspace.cloud_dims,
                   workspace.f_scratch,
                   workspace.amu_scratch, workspace.amd_scratch,
                   Int(tile_off), dt_ft;
                   ndrange = n)
        end
    end
    synchronize(backend)
    return nothing
end

function apply_convection!(q_raw::NTuple{6, <:AbstractArray{FT, 4}},
                            air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                            forcing::ConvectionForcing,
                            op::TM5Convection,
                            dt::Real,
                            workspace::TM5Workspace,
                            grid::AtmosGrid{<:CubedSphereMesh}) where {FT}
    _assert_tm5_forcing(forcing)
    tm5     = forcing.tm5_fields
    cell_areas = _tm5_require_cell_metrics(workspace, "cubed-sphere TM5Convection")
    mesh    = grid.horizontal
    Nc      = mesh.Nc
    Hp      = mesh.Hp
    N_total = Nc * Nc
    backend = get_backend(q_raw[1])
    dt_ft   = FT(dt)
    # Per-panel Nz / Nt: panels share a vertical resolution by
    # construction, so we read them off panel 1.
    Nz = size(air_mass[1], 3)
    Nt = size(q_raw[1], 4)
    if _use_collab_path(op, Nz, Nt, FT, backend)
        kernel = _tm5_cs_panel_column_collab_kernel!(backend, _TM5_COLLAB_WG_SIZE)
        # The workspace is no longer needed by the collab kernel
        # (its scratch lives in `@localmem`), but the panel loop
        # remains so each panel's halo offset and forcing arrays
        # are passed individually.
        for p in 1:6
            kernel(q_raw[p], air_mass[p],
                   tm5.entu[p], tm5.detu[p], tm5.entd[p], tm5.detd[p],
                   cell_areas[p],
                   Int(Hp), Int(Nc), Int(Nt), Float32(dt), Val(Int(Nz));
                   ndrange = _TM5_COLLAB_WG_SIZE * N_total,
                   workgroupsize = _TM5_COLLAB_WG_SIZE)
        end
    else
        kernel = _tm5_cs_panel_column_kernel!(backend)
        B = size(workspace.conv1, 3)
        # The workspace is shared across panels; KA stream ordering
        # makes that safe (panel n+1 can't start until panel n's
        # writes are visible). The `for p` loop sits *outside* the
        # tile loop so the workspace is reused per panel rather than
        # cloned six times.
        for p in 1:6
            for tile_off in 0:B:(N_total - 1)
                n = min(B, N_total - tile_off)
                kernel(q_raw[p], air_mass[p],
                       tm5.entu[p], tm5.detu[p], tm5.entd[p], tm5.detd[p],
                       cell_areas[p],
                       workspace.conv1, workspace.pivots,
                       workspace.cloud_dims,
                       workspace.f_scratch,
                       workspace.amu_scratch, workspace.amd_scratch,
                       Int(Hp), Int(tile_off), Int(Nc), dt_ft;
                       ndrange = n)
            end
        end
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

function _tm5_require_cell_metrics(workspace::TM5Workspace, context::AbstractString)
    metrics = workspace.cell_metrics
    metrics === nothing && throw(ArgumentError(
        "$context requires `workspace.cell_metrics` to carry cell areas " *
        "so TM5 can convert state air mass from kg/cell to kg/m². " *
        "Build the model with `with_convection` or construct " *
        "`TM5Workspace(air_mass; cell_metrics=...)`."))
    return metrics
end

# =========================================================================
# State-level entry: apply!
# =========================================================================

"""
    apply!(state::CellState, forcing::ConvectionForcing, grid::AtmosGrid,
           op::TM5Convection, dt::Real; workspace) -> state

State-level delegate — matches the CMFMC contract at
`CMFMCConvection.jl:296-316`.
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
