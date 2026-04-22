# ---------------------------------------------------------------------------
# Convection workspace storage — plan 18 Decision 21 (CFL caching) + §2.20
# Decision 26 (caller-owned pre-allocation).
#
# CMFMCWorkspace — scratch + CFL cache for CMFMCConvection.
# TM5Workspace   — `conv1` matrix slab + pivot vectors + cloud-dim
#                  indices for TM5Convection (plan 23 Commit 1).
#                  Per principle 3 (plan 23), pivots stay a
#                  dedicated field even though the TM5 matrix is
#                  diagonally dominant: the adjoint port in plan 19
#                  replays the same factorization with trans='T'.
# ---------------------------------------------------------------------------

"""
    CMFMCWorkspace{FT, QC, CA}

Per-sim pre-allocated workspace for [`CMFMCConvection`](@ref).

# Fields

- `qc_scratch :: QC` — updraft concentration buffer, one entry per
  cell, shape matching `air_mass` (`(Nx, Ny, Nz)` for structured
  grids, `(ncells, Nz)` for face-indexed grids). Reused across all
  substeps and all tracers.
- `cell_metrics :: CA` — pre-adapted cell-area metric vector used by
  the CFL scan and the kernel:
  `cell_areas_by_latitude(mesh)` for structured grids,
  per-cell `cell_area(mesh, c)` for face-indexed grids.
- `cached_n_sub :: Base.RefValue{Int}` — the CFL-derived sub-step
  count for the current met window. Stays valid as long as
  `cache_valid[] == true`; re-computed from the current CMFMC /
  air-mass state on the next `apply!` call when invalid. Int (not
  Float) because it's the integer sub-step count.
- `cache_valid :: Base.RefValue{Bool}` — sentinel; cleared via
  [`invalidate_cmfmc_cache!`](@ref) when the met window advances
  (`DrivenSimulation._maybe_advance_window!`, Commit 8).

# Usage

Constructed once at `DrivenSimulation` setup time and reused for the
whole run. `Adapt.adapt_structure` preserves the cached scalar state
on the host while adapting `qc_scratch` to the requested backend.
"""
struct CMFMCWorkspace{FT, QC, CA}
    qc_scratch   :: QC
    cell_metrics :: CA
    cached_n_sub :: Base.RefValue{Int}
    cache_valid  :: Base.RefValue{Bool}
end

"""
    CMFMCWorkspace(air_mass::AbstractArray{FT}; cell_metrics = nothing) -> CMFMCWorkspace

Construct a fresh workspace from an air-mass payload. `air_mass` may
be a single array (structured / face-indexed) or a cubed-sphere panel
tuple. Shape of `qc_scratch` matches `air_mass`.
"""
@inline _cmfmc_scratch_like(air_mass::AbstractArray{FT}) where FT = similar(air_mass)
@inline function _cmfmc_scratch_like(air_mass::NTuple{N, <:AbstractArray{FT}}) where {N, FT}
    return ntuple(i -> similar(air_mass[i]), N)
end

@inline _cmfmc_metric_buffer(metric, ::Type{FT}) where FT =
    copyto!(similar(metric, FT), metric)
@inline function _cmfmc_metric_buffer(metrics::NTuple{N}, ::Type{FT}) where {N, FT}
    return ntuple(i -> _cmfmc_metric_buffer(metrics[i], FT), N)
end

function CMFMCWorkspace(air_mass::AbstractArray{FT}; cell_metrics = nothing) where FT
    qc_scratch = _cmfmc_scratch_like(air_mass)
    metrics = if cell_metrics === nothing
        nothing
    else
        _cmfmc_metric_buffer(cell_metrics, FT)
    end
    return CMFMCWorkspace{FT, typeof(qc_scratch), typeof(metrics)}(
        qc_scratch,
        metrics,
        Ref{Int}(1),
        Ref{Bool}(false),
    )
end

function CMFMCWorkspace(air_mass::NTuple{N, <:AbstractArray{FT}}; cell_metrics = nothing) where {N, FT}
    qc_scratch = _cmfmc_scratch_like(air_mass)
    metrics = if cell_metrics === nothing
        nothing
    else
        _cmfmc_metric_buffer(cell_metrics, FT)
    end
    return CMFMCWorkspace{FT, typeof(qc_scratch), typeof(metrics)}(
        qc_scratch,
        metrics,
        Ref{Int}(1),
        Ref{Bool}(false),
    )
end

function Adapt.adapt_structure(to, ws::CMFMCWorkspace{FT}) where FT
    qc_scratch = Adapt.adapt(to, ws.qc_scratch)
    cell_metrics = ws.cell_metrics === nothing ? nothing : Adapt.adapt(to, ws.cell_metrics)
    return CMFMCWorkspace{FT, typeof(qc_scratch), typeof(cell_metrics)}(
        qc_scratch,
        cell_metrics,
        Ref{Int}(ws.cached_n_sub[]),
        Ref{Bool}(ws.cache_valid[]),
    )
end

"""
    invalidate_cmfmc_cache!(ws::CMFMCWorkspace) -> nothing

Mark the cached CFL `n_sub` as stale. Called on met-window
advance by `DrivenSimulation._maybe_advance_window!` (plan 18
Commit 8). The next `apply!` recomputes `n_sub` from the fresh CMFMC
array.
"""
function invalidate_cmfmc_cache!(ws::CMFMCWorkspace)
    ws.cache_valid[] = false
    return nothing
end

# No-op for non-CMFMC workspaces (e.g. TM5Workspace); simplifies the
# call site in DrivenSimulation.
invalidate_cmfmc_cache!(::Any) = nothing

# ===========================================================================
# TM5Workspace — plan 23 Commit 1
# ===========================================================================

"""
    TM5Workspace{FT, M, P, C, F, A}

Per-sim pre-allocated workspace for [`TM5Convection`](@ref).

# Fields

- `conv1 :: M` — `conv1 = I - dt·D` matrix slab, one `(Nz, Nz)`
  block per column. Parametric on array type so
  `Adapt.adapt_structure` can swap CPU ↔ GPU without changing the
  `TM5Workspace` type constructor (plan 23 principle 5).
  Shapes per topology:
  - Structured LatLon: `(Nz, Nz, Nx, Ny)` — 4D.
  - Face-indexed ReducedGaussian: `(Nz, Nz, ncells)` — 3D.
  - Panel-native CubedSphere: `NTuple{6, AbstractArray{FT, 4}}`
    with per-panel shape `(Nz, Nz, Nc, Nc)`.
- `pivots :: P` — permutation vector from partial-pivot LU, one
  Nz-length Int slice per column. Per plan 23 principle 3,
  preserved so plan 19 (adjoint) can replay the same factorization
  with transposed back-substitution. Shapes strip the leading Nz
  from `conv1` shape: `(Nz, Nx, Ny)` / `(Nz, ncells)` /
  `NTuple{6, (Nz, Nc, Nc)}`.
- `cloud_dims :: C` — per-column `(icltop, iclbas, icllfs)` triple
  in AtmosTransport indexing (k=1=TOA, k=Nz=surface; the
  preprocessor delivers forcings in this orientation so the solver
  has zero orientation logic). Shape `(3, Nx, Ny)` /
  `(3, ncells)` / `NTuple{6, (3, Nc, Nc)}`.
- `f_scratch :: F`, `fu_scratch :: F` — `(Nz+1, Nz)` per-column
  intermediate matrices for the matrix build (TM5 `f(0:lmx, 1:lmx)`
  and `fu(0:lmx, 1:lmx)`). Plan 23 Commit 4 pre-allocates these
  so `_tm5_build_conv1!` runs without heap allocation inside KA
  kernels (mandatory on GPU; same contract on CPU for parity).
- `amu_scratch :: A`, `amd_scratch :: A` — length-`(Nz+1)`
  per-column boundary-aware mass-flux vectors (TM5 `amu(0:lmx)` /
  `amd(0:lmx)`). Same allocation policy as the scratch matrices.

# Usage

Constructed once at `DrivenSimulation` setup time via
`_convection_workspace_for(::TM5Convection, state, grid)`.
`Adapt.adapt_structure` adapts each array to the requested backend
without reallocating on the host.
"""
struct TM5Workspace{FT, M, P, C, F, A}
    conv1       :: M
    pivots      :: P
    cloud_dims  :: C
    f_scratch   :: F
    fu_scratch  :: F
    amu_scratch :: A
    amd_scratch :: A
end

# Allocation helpers — single-array (structured / face-indexed) vs
# NTuple (CS panel) dispatch, mirroring the CMFMC pattern.

@inline function _tm5_conv1_like(air_mass::AbstractArray{FT, 3}) where FT
    # Structured LatLon: air_mass is (Nx, Ny, Nz) → conv1 (Nz, Nz, Nx, Ny).
    Nx, Ny, Nz = size(air_mass)
    return similar(air_mass, FT, Nz, Nz, Nx, Ny)
end
@inline function _tm5_conv1_like(air_mass::AbstractArray{FT, 2}) where FT
    # Face-indexed RG: air_mass is (ncells, Nz) → conv1 (Nz, Nz, ncells).
    ncells, Nz = size(air_mass)
    return similar(air_mass, FT, Nz, Nz, ncells)
end
@inline function _tm5_conv1_like(air_mass::NTuple{N, <:AbstractArray{FT, 3}}) where {N, FT}
    # CS panel NTuple: per-panel air_mass (Nc, Nc, Nz) → per-panel
    # conv1 (Nz, Nz, Nc, Nc).
    return ntuple(i -> begin
        Nc1, Nc2, Nz = size(air_mass[i])
        similar(air_mass[i], FT, Nz, Nz, Nc1, Nc2)
    end, N)
end

@inline function _tm5_pivots_like(air_mass::AbstractArray{FT, 3}) where FT
    Nx, Ny, Nz = size(air_mass)
    return similar(air_mass, Int, Nz, Nx, Ny)
end
@inline function _tm5_pivots_like(air_mass::AbstractArray{FT, 2}) where FT
    ncells, Nz = size(air_mass)
    return similar(air_mass, Int, Nz, ncells)
end
@inline function _tm5_pivots_like(air_mass::NTuple{N, <:AbstractArray{FT, 3}}) where {N, FT}
    return ntuple(i -> begin
        Nc1, Nc2, Nz = size(air_mass[i])
        similar(air_mass[i], Int, Nz, Nc1, Nc2)
    end, N)
end

@inline function _tm5_cloud_dims_like(air_mass::AbstractArray{FT, 3}) where FT
    Nx, Ny, _ = size(air_mass)
    return similar(air_mass, Int, 3, Nx, Ny)
end
@inline function _tm5_cloud_dims_like(air_mass::AbstractArray{FT, 2}) where FT
    ncells, _ = size(air_mass)
    return similar(air_mass, Int, 3, ncells)
end
@inline function _tm5_cloud_dims_like(air_mass::NTuple{N, <:AbstractArray{FT, 3}}) where {N, FT}
    return ntuple(i -> begin
        Nc1, Nc2, _ = size(air_mass[i])
        similar(air_mass[i], Int, 3, Nc1, Nc2)
    end, N)
end

# Per-column (Nz+1, Nz) scratch for `f` / `fu` — same leading-dim
# layout as conv1 but with an extra row for the TM5 boundary
# convention (row Nz+1 = "below surface").
@inline function _tm5_scratch_f_like(air_mass::AbstractArray{FT, 3}) where FT
    Nx, Ny, Nz = size(air_mass)
    return similar(air_mass, FT, Nz + 1, Nz, Nx, Ny)
end
@inline function _tm5_scratch_f_like(air_mass::AbstractArray{FT, 2}) where FT
    ncells, Nz = size(air_mass)
    return similar(air_mass, FT, Nz + 1, Nz, ncells)
end
@inline function _tm5_scratch_f_like(air_mass::NTuple{N, <:AbstractArray{FT, 3}}) where {N, FT}
    return ntuple(i -> begin
        Nc1, Nc2, Nz = size(air_mass[i])
        similar(air_mass[i], FT, Nz + 1, Nz, Nc1, Nc2)
    end, N)
end

# Per-column length-(Nz+1) scratch for `amu` / `amd`.
@inline function _tm5_scratch_am_like(air_mass::AbstractArray{FT, 3}) where FT
    Nx, Ny, Nz = size(air_mass)
    return similar(air_mass, FT, Nz + 1, Nx, Ny)
end
@inline function _tm5_scratch_am_like(air_mass::AbstractArray{FT, 2}) where FT
    ncells, Nz = size(air_mass)
    return similar(air_mass, FT, Nz + 1, ncells)
end
@inline function _tm5_scratch_am_like(air_mass::NTuple{N, <:AbstractArray{FT, 3}}) where {N, FT}
    return ntuple(i -> begin
        Nc1, Nc2, Nz = size(air_mass[i])
        similar(air_mass[i], FT, Nz + 1, Nc1, Nc2)
    end, N)
end

"""
    TM5Workspace(air_mass) -> TM5Workspace

Construct a fresh workspace from an air-mass payload. `air_mass`
may be a single array (structured `(Nx, Ny, Nz)` or face-indexed
`(ncells, Nz)`) or a cubed-sphere panel tuple. The per-column
element type of `conv1` matches `eltype(air_mass)`.
"""
function TM5Workspace(air_mass::AbstractArray{FT}) where FT
    conv1       = _tm5_conv1_like(air_mass)
    pivots      = _tm5_pivots_like(air_mass)
    cloud_dims  = _tm5_cloud_dims_like(air_mass)
    f_scratch   = _tm5_scratch_f_like(air_mass)
    fu_scratch  = _tm5_scratch_f_like(air_mass)
    amu_scratch = _tm5_scratch_am_like(air_mass)
    amd_scratch = _tm5_scratch_am_like(air_mass)
    return TM5Workspace{FT,
                        typeof(conv1), typeof(pivots), typeof(cloud_dims),
                        typeof(f_scratch), typeof(amu_scratch)}(
        conv1, pivots, cloud_dims,
        f_scratch, fu_scratch, amu_scratch, amd_scratch,
    )
end

function TM5Workspace(air_mass::NTuple{N, <:AbstractArray{FT, 3}}) where {N, FT}
    conv1       = _tm5_conv1_like(air_mass)
    pivots      = _tm5_pivots_like(air_mass)
    cloud_dims  = _tm5_cloud_dims_like(air_mass)
    f_scratch   = _tm5_scratch_f_like(air_mass)
    fu_scratch  = _tm5_scratch_f_like(air_mass)
    amu_scratch = _tm5_scratch_am_like(air_mass)
    amd_scratch = _tm5_scratch_am_like(air_mass)
    return TM5Workspace{FT,
                        typeof(conv1), typeof(pivots), typeof(cloud_dims),
                        typeof(f_scratch), typeof(amu_scratch)}(
        conv1, pivots, cloud_dims,
        f_scratch, fu_scratch, amu_scratch, amd_scratch,
    )
end

function Adapt.adapt_structure(to, ws::TM5Workspace{FT}) where FT
    conv1       = Adapt.adapt(to, ws.conv1)
    pivots      = Adapt.adapt(to, ws.pivots)
    cloud_dims  = Adapt.adapt(to, ws.cloud_dims)
    f_scratch   = Adapt.adapt(to, ws.f_scratch)
    fu_scratch  = Adapt.adapt(to, ws.fu_scratch)
    amu_scratch = Adapt.adapt(to, ws.amu_scratch)
    amd_scratch = Adapt.adapt(to, ws.amd_scratch)
    return TM5Workspace{FT,
                        typeof(conv1), typeof(pivots), typeof(cloud_dims),
                        typeof(f_scratch), typeof(amu_scratch)}(
        conv1, pivots, cloud_dims,
        f_scratch, fu_scratch, amu_scratch, amd_scratch,
    )
end

export CMFMCWorkspace, invalidate_cmfmc_cache!
export TM5Workspace
