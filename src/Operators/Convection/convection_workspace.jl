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
- `f_scratch :: F` — per-column intermediate matrix for the
  matrix build (TM5 `f(0:lmx, 1:lmx)` after the storage plan's
  Commit 3 merged the updraft into `f`). Plan 23 Commit 4
  pre-allocates this so `_tm5_build_conv1!` runs without heap
  allocation inside KA kernels (mandatory on GPU; same contract on
  CPU for parity). `f_scratch` aliases `conv1` in the production
  workspace because `conv1` is only needed after `f` has been
  converted into `I - dt*D`; this saves one dense `(Nz, Nz)` slab
  per column. The standalone `fu_scratch` field that previously
  carried the updraft contribution was dropped in Commit 3 — the
  updraft and downdraft passes write disjoint index ranges, so the
  builder writes directly into `f`.
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
    amu_scratch :: A
    amd_scratch :: A
end

# Topology-shape introspection helpers used by the tile workspace
# constructor. Single dispatch on the air-mass payload picks Nz,
# the per-launch cell count, and a "template" array (used by
# `similar` to inherit backend / element type) — the workspace
# itself is topology-agnostic after Commit 4 of the storage plan.

@inline _tm5_template(air_mass::AbstractArray) = air_mass
@inline _tm5_template(air_mass::NTuple)        = air_mass[1]

# Number of vertical layers — the leading two `(Nz, Nz)` dims of
# the matrix slab.
@inline _tm5_extract_Nz(air_mass::AbstractArray{<:Any, 3}) = size(air_mass, 3)
@inline _tm5_extract_Nz(air_mass::AbstractArray{<:Any, 2}) = size(air_mass, 2)
@inline _tm5_extract_Nz(air_mass::NTuple{N, <:AbstractArray{<:Any, 3}}) where N =
    size(air_mass[1], 3)

# Per-launch cell count — the largest single kernel ndrange the
# workspace has to cover. CS panels are launched one at a time,
# so the workspace is sized for one panel (`Nc²`), not all six.
@inline _tm5_total_cells_per_launch(air_mass::AbstractArray{<:Any, 3}) =
    size(air_mass, 1) * size(air_mass, 2)
@inline _tm5_total_cells_per_launch(air_mass::AbstractArray{<:Any, 2}) =
    size(air_mass, 1)
@inline _tm5_total_cells_per_launch(air_mass::NTuple{N, <:AbstractArray{<:Any, 3}}) where N =
    size(air_mass[1], 1) * size(air_mass[1], 2)

"""
    derive_tile_columns(::Type{FT}, Nz, budget_gib, total_cells) -> Int

Pick a tile size `B` from a per-cell memory cost and a target
budget. The cost model accounts for every per-cell field that
[`TM5Workspace`](@ref) tiles:

- `conv1` : `Nz²` × `sizeof(FT)`
- `amu_scratch` + `amd_scratch` : `2(Nz+1)` × `sizeof(FT)`
- `pivots` : `Nz` × `sizeof(Int)`
- `cloud_dims` : `3` × `sizeof(Int)`

`f_scratch` is a structural alias for `conv1` (saves one matrix
slab per cell). `B` is clamped between 256 (avoid pathological
launches) and `total_cells` (one tile covers the whole topology
when the budget allows it).
"""
function derive_tile_columns(::Type{FT}, Nz::Integer,
                             budget_gib::Real, total_cells::Integer) where FT
    Nz > 0 || throw(ArgumentError("derive_tile_columns: Nz must be > 0"))
    total_cells > 0 ||
        throw(ArgumentError("derive_tile_columns: total_cells must be > 0"))
    budget_gib > 0 ||
        throw(ArgumentError("derive_tile_columns: budget_gib must be > 0"))
    per_cell_bytes = sizeof(FT) * (Nz * Nz + 2 * (Nz + 1)) +
                     sizeof(Int) * (Nz + 3)
    budget_bytes = round(Int, budget_gib * (1 << 30))
    raw = fld(budget_bytes, per_cell_bytes)
    # Floor of 256 avoids pathologically tiny launches; the
    # `min(total_cells, …)` cap keeps small topologies from
    # over-allocating beyond what they need.
    return Int(min(Int(total_cells), max(256, Int(raw))))
end

"""
    TM5Workspace(air_mass; tile_columns = …,
                            tile_workspace_gib = nothing) -> TM5Workspace

Construct a fresh workspace from an air-mass payload. `air_mass`
may be a single array (structured `(Nx, Ny, Nz)` or face-indexed
`(ncells, Nz)`) or a cubed-sphere panel tuple — the workspace
itself is topology-agnostic. The per-column element type of
`conv1` matches `eltype(air_mass)`.

The per-launch column count `B` is set by exactly one of:

- `tile_columns::Integer` (explicit). Default
  `_tm5_total_cells_per_launch(air_mass)` — one tile covers the
  whole launch and the workspace is bit-equal to the pre-Commit-4
  per-cell allocator. Production paths use this branch when they
  already know `B`.
- `tile_workspace_gib::Real` (budget). Picks `B` via
  [`derive_tile_columns`](@ref) from
  [`TM5Convection`](@ref)'s `tile_workspace_gib` field. Bypassed
  if `tile_columns` is also passed (explicit wins).

Specifying both is an error.
"""
function TM5Workspace(air_mass;
                      tile_columns::Union{Integer, Nothing} = nothing,
                      tile_workspace_gib::Union{Real, Nothing} = nothing)
    Nz          = _tm5_extract_Nz(air_mass)
    template    = _tm5_template(air_mass)
    FT          = eltype(template)
    total_cells = _tm5_total_cells_per_launch(air_mass)
    B = if tile_columns !== nothing && tile_workspace_gib !== nothing
        throw(ArgumentError(
            "TM5Workspace: pass either `tile_columns` or `tile_workspace_gib`, not both"))
    elseif tile_columns !== nothing
        Int(tile_columns)
    elseif tile_workspace_gib !== nothing
        derive_tile_columns(FT, Nz, Float64(tile_workspace_gib), total_cells)
    else
        total_cells
    end
    B > 0 || throw(ArgumentError("TM5Workspace: tile size must be > 0"))
    conv1       = similar(template, FT,  Nz, Nz, B)
    pivots      = similar(template, Int, Nz,     B)
    cloud_dims  = similar(template, Int, 3,      B)
    f_scratch   = conv1                     # alias — see docstring
    amu_scratch = similar(template, FT,  Nz + 1, B)
    amd_scratch = similar(template, FT,  Nz + 1, B)
    return TM5Workspace{FT,
                        typeof(conv1), typeof(pivots), typeof(cloud_dims),
                        typeof(f_scratch), typeof(amu_scratch)}(
        conv1, pivots, cloud_dims,
        f_scratch, amu_scratch, amd_scratch,
    )
end

function Adapt.adapt_structure(to, ws::TM5Workspace{FT}) where FT
    f_aliases_conv1 = ws.conv1 === ws.f_scratch
    conv1       = Adapt.adapt(to, ws.conv1)
    pivots      = Adapt.adapt(to, ws.pivots)
    cloud_dims  = Adapt.adapt(to, ws.cloud_dims)
    f_scratch   = f_aliases_conv1 ? conv1 : Adapt.adapt(to, ws.f_scratch)
    amu_scratch = Adapt.adapt(to, ws.amu_scratch)
    amd_scratch = Adapt.adapt(to, ws.amd_scratch)
    return TM5Workspace{FT,
                        typeof(conv1), typeof(pivots), typeof(cloud_dims),
                        typeof(f_scratch), typeof(amu_scratch)}(
        conv1, pivots, cloud_dims,
        f_scratch, amu_scratch, amd_scratch,
    )
end

export CMFMCWorkspace, invalidate_cmfmc_cache!
export TM5Workspace, derive_tile_columns
