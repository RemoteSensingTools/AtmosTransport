# ---------------------------------------------------------------------------
# Convection workspace storage — plan 18 Decision 21 (CFL caching) + §2.20
# Decision 26 (caller-owned pre-allocation).
#
# Commit 3 ships CMFMCWorkspace for CMFMCConvection. TM5Workspace
# (with the conv1_ws matrix buffer + pivots) lands in Commit 4
# alongside TM5Convection.
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

export CMFMCWorkspace, invalidate_cmfmc_cache!
