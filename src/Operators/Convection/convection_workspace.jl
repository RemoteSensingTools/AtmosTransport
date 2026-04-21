# ---------------------------------------------------------------------------
# Convection workspace storage — plan 18 Decision 21 (CFL caching) + §2.20
# Decision 26 (caller-owned pre-allocation).
#
# Commit 3 ships CMFMCWorkspace for CMFMCConvection. TM5Workspace
# (with the conv1_ws matrix buffer + pivots) lands in Commit 4
# alongside TM5Convection.
# ---------------------------------------------------------------------------

"""
    CMFMCWorkspace{FT, QC}

Per-sim pre-allocated workspace for [`CMFMCConvection`](@ref).

# Fields

- `qc_scratch :: QC` — updraft concentration buffer, one entry per
  cell, shape `(Nx, Ny, Nz)` for structured grids. Reused across
  all substeps and all tracers.
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
whole run. Not Adapt-transferable directly in this commit — plan 18
structured-only scope (Decision 25) means the kernel only sees CPU
or KA-backend arrays depending on the state's backend, so the Ref
fields stay host-side and `qc_scratch` is adapted via
`Adapt.adapt_structure`. GPU transfer will be extended if / when
plan 18b brings in face-indexed convection.
"""
struct CMFMCWorkspace{FT, QC <: AbstractArray{FT, 3}}
    qc_scratch   :: QC
    cached_n_sub :: Base.RefValue{Int}
    cache_valid  :: Base.RefValue{Bool}
end

"""
    CMFMCWorkspace(air_mass::AbstractArray{FT, 3}) -> CMFMCWorkspace

Construct a fresh workspace from a structured-grid air-mass array.
Shape of `qc_scratch` matches `air_mass`.
"""
function CMFMCWorkspace(air_mass::AbstractArray{FT, 3}) where FT
    qc_scratch = similar(air_mass)
    return CMFMCWorkspace{FT, typeof(qc_scratch)}(
        qc_scratch,
        Ref{Int}(1),
        Ref{Bool}(false),
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
