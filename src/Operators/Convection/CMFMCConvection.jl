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

Plan 18 is structured-only (Decision 25). Face-indexed state
(`Raw <: AbstractArray{_, 3}`) raises `ArgumentError` pointing at
"Plan 18b: Face-indexed convection" as the follow-up.

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
(see `18_CONVECTION_UPSTREAM_GCHP_NOTES.md` §5.3).
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
    # balance. Matches the legacy lat-lon fallback behavior at
    # `src_legacy/Convection/ras_convection.jl:365-370` which
    # delegated to a single-flux Tiedtke operator that made the same
    # diagnosis implicitly.)
    Nx_c, Ny_c, Nz_c = size(air_mass)
    dtrain_arr = if forcing.dtrain === nothing
        darr = Array{FT}(undef, Nx_c, Ny_c, Nz_c)
        @inbounds for k in 1:Nz_c, j in 1:Ny_c, i in 1:Nx_c
            darr[i, j, k] = max(zero(FT), cmfmc[i, j, k + 1] - cmfmc[i, j, k])
        end
        darr
    else
        forcing.dtrain
    end
    # Always true at the kernel level once we've closed the mass
    # balance via derived or provided dtrain.
    has_dtrain_val = true

    Nx, Ny, Nz, Nt = size(q_raw)
    cell_areas_y = cell_areas_by_latitude(grid.horizontal)

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

# Face-indexed state rejection (plan 18 v5.1 §2.19 Decision 25).
# Dispatches on `Raw <: AbstractArray{_, 3}` (face-indexed
# `tracers_raw` is 3D: `(ncells, Nz, Nt)`). Structured lat-lon has
# 4D `tracers_raw` and goes through the method above.
function apply_convection!(q_raw::AbstractArray{FT, 3},
                            air_mass::AbstractArray{FT, 2},
                            ::ConvectionForcing,
                            ::CMFMCConvection,
                            dt, workspace, grid) where FT
    throw(ArgumentError(
        "Face-indexed convection is not in plan 18 scope (Decision 25). " *
        "`CMFMCConvection` supports structured lat-lon only. " *
        "Follow-up 'Plan 18b: Face-indexed convection' extends the " *
        "kernel to `(ncells, Nz)` layout."))
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
function apply!(state::CellState, forcing::ConvectionForcing,
                grid::AtmosGrid, op::CMFMCConvection, dt::Real;
                workspace::CMFMCWorkspace)
    apply_convection!(state.tracers_raw, state.air_mass, forcing, op, dt,
                       workspace, grid)
    return state
end

# Face-indexed state-level rejection per Decision 25. Dispatch on
# 3D `tracers_raw` shape.
function apply!(state::CellState{B, A, Raw, Names},
                forcing::ConvectionForcing, grid::AtmosGrid,
                op::CMFMCConvection, dt::Real;
                workspace = nothing) where {B, A, Raw <: AbstractArray{<:Any, 3}, Names}
    throw(ArgumentError(
        "Face-indexed convection is not in plan 18 scope (Decision 25). " *
        "`CMFMCConvection` supports structured lat-lon only. " *
        "Follow-up 'Plan 18b: Face-indexed convection' extends the " *
        "kernel to `(ncells, Nz)` layout."))
end
