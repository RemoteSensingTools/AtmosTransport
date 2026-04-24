"""
    InitialConditionIO

Single owner of initial-condition I/O, vertical remap, and
topology-dispatched VMR builders for the unified runtime.

**Status: scaffold (plan 40 Commit 1a, 2026-04-24).**

Plan 40 Commit 1 splits into:

- **1a (this file) — scaffold**: module exists and is included from
  `Models.jl`; no code moved yet. Establishes the architectural
  decision that the topology-dispatched IC builder + file loaders +
  psurf remap + surface-flux builders live here, not in
  `scripts/run_transport_binary.jl`.
- **1b — LL/RG hoist**: move (verbatim, bit-exact)
  `_sample_bilinear_profile!`, `_sample_bilinear_scalar`,
  `_horizontal_interp_weights`, `_load_file_initial_condition_source`,
  `_interpolate_log_pressure_profile!`, `_copy_profile!` and the
  `build_initial_mixing_ratio(grid, air_mass, cfg, FT)` methods for
  LL/RG from `scripts/run_transport_binary.jl:{139,170,186,353,466,524,570,593,622,653}`.
  Add `pack_initial_tracer_mass(grid, air_mass, vmr_dry; mass_basis, qv=nothing)`
  dispatching on `mass_basis::AbstractMassBasis` per
  `feedback_vmr_to_mass_basis_aware`.
  Replace `scripts/run_transport_binary.jl`'s local copies with
  qualified calls. LL/RG numerics unchanged, verified by
  `test_run_transport_binary_recipe.jl` and a new bit-exact
  regression test.
- **1c — CS IC + CS surface flux**: add
  `build_initial_mixing_ratio(grid::AtmosGrid{<:CubedSphereMesh}, …)`
  (file-based path using `regrid_3d_to_cs_panels!` +
  `regrid_2d_to_cs_panels!` + `_interpolate_log_pressure_profile!`)
  and `build_surface_flux_source(grid::AtmosGrid{<:CubedSphereMesh}, …)`
  (conservative regrid + per-panel `cell_area` multiply to satisfy
  `SurfaceFluxSource`'s kg/s-per-cell contract at
  `src/Operators/SurfaceFlux/sources.jl:12`).

Ownership boundary (plan 40 §"Ownership boundary"):
- Binary header drives grid topology + panel convention + mass basis
  + capability set.
- TOML supplies tracer names, init kinds (`uniform | gaussian_blob |
  file | catrine_co2 | netcdf | file_field`), surface-flux kinds,
  scale factors, paths.
- Every public function in this module dispatches on grid /
  `mass_basis::AbstractMassBasis` types; TOML strings are only
  consumed at the Dict-unpacking boundary.

Correctness rules pinned (plan 40 NOTES §"Correctness rules"):
1. IC file inputs are **dry VMR**. `pack_initial_tracer_mass` is
   basis-aware: DryBasis → `vmr .* air_mass`; MoistBasis → `vmr .*
   air_mass .* (1 .- qv)` with `qv` from the first transport window.
2. Surface-flux file inputs are kg/m²/s; the CS builder (and every
   builder in this module) must area-integrate to kg/s per cell
   before handing to `SurfaceFluxSource`.
"""
module InitialConditionIO

# Populated in Commit 1b — hoist from scripts/run_transport_binary.jl:
# using NCDatasets
# using ..State: AbstractMassBasis, DryBasis, MoistBasis
# using ..Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh
# using ..Preprocessing: regrid_3d_to_cs_panels!, regrid_2d_to_cs_panels!
# using ..Regridding: build_regridder, apply_regridder!

# Populated in Commit 1b (hoisted from scripts/run_transport_binary.jl):
# function _horizontal_interp_weights end    # :139
# function _sample_bilinear_profile! end      # :170
# function _sample_bilinear_scalar end        # :186
# function _load_file_initial_condition_source end  # :353
# function _interpolate_log_pressure_profile! end   # :466
# function _copy_profile! end                 # :524
# function build_initial_mixing_ratio end     # :570,:593,:622,:653 (LL/RG)
# function pack_initial_tracer_mass end       # NEW — basis-aware

# Populated in Commit 1c (CS file-based IC + CS surface flux):
# build_initial_mixing_ratio(grid::AtmosGrid{<:CubedSphereMesh}, …)
# build_surface_flux_source(grid::AtmosGrid{<:CubedSphereMesh}, …)

# Surface-flux builders hoisted in Commit 1b (LL/RG) + 1c (CS):
# function _load_file_surface_flux_field end  # :237
# function build_surface_flux_source end      # :758, :791, +CS in 1c
# function build_surface_flux_sources end     # :822

end # module InitialConditionIO
