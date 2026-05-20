"""
    Preprocessing

Transport binary generation from ERA5 spectral meteorological data.

Provides the complete pipeline from raw spectral GRIB (VO, D, LNSP) to
Poisson-balanced transport binaries ready for the runtime `TransportBinaryDriver`.

## Architecture

The pipeline dispatches on `AbstractTargetGeometry` subtypes:

    AbstractTargetGeometry
    ├── LatLonTargetGeometry              — regular lat-lon (any Nx × Ny)
    ├── ReducedGaussianTargetGeometry     — native ERA5 RG (O90, O160, N320, …)
    └── CubedSphereTargetGeometry         — convention-aware CS (C24, C90, C180, …)

Each target geometry has a dedicated `process_day` method:

    process_day(date, grid::LatLonTargetGeometry, settings, vertical; …)
    process_day(date, grid::ReducedGaussianTargetGeometry, settings, vertical; …)
    process_day(date, grid::CubedSphereTargetGeometry, settings, vertical; …)

## Pipeline (per day)

1. Read ERA5 spectral GRIB (VO, D, LNSP) — `spectral_io.jl`
2. Spectral synthesis (Legendre + FFT → gridpoint) — `spectral_synthesis.jl`
3. Merge native 137L → transport levels — `vertical_coordinates.jl`
4. Pin global mean ps (mass fix) — `transport_binary/latlon_workspaces.jl`
5. Topology-specific transport-binary workflow:
   - LL: FFT on circulant Laplacian — `mass_support.jl`
   - RG: compressed-Laplacian CG — `ring_poisson_balance.jl`
   - CS: global 6-panel graph-Laplacian CG — `cs_poisson_balance.jl`
6. Diagnose `cm` from explicit endpoint mass targets — replay continuity
7. Write transport binary with declared payload semantics and replay checks

The high-level transport-binary workflows are split under
`transport_binary/`. Each topology owns a small `process_day` method while
shared binary metadata, endpoint-delta, and replay helpers stay in common
files.

## Usage

```bash
julia --project=. scripts/preprocessing/preprocess_transport_binary.jl config.toml --day 2021-12-01
```

Advanced users can call `process_day` directly from Julia.
"""
module Preprocessing

using Dates
using Logging
using FFTW
using Printf
using JSON3
using JLD2
using TOML
using LinearAlgebra: mul!, dot
using NCDatasets
using GRIB
using SHA
using FastGaussQuadrature: gausslegendre

# Re-export parent module types we need
import ..expand_data_path
using ..Architectures: CPU
using ..Grids: LatLonMesh, ReducedGaussianMesh, CubedSphereMesh,
               HybridSigmaPressure, PanelConnectivity,
               AbstractCubedSpherePanelConvention,
               AtmosGrid, ncells, nfaces, nrings, nlevels, face_cells, cell_area,
               ring_longitudes, ring_cell_count, cell_areas_by_latitude,
               n_levels, pressure_at_interface, level_thickness, floattype,
               panel_convention, panel_connectivity_for,
               panel_cell_center_lonlat,
               panel_cell_local_tangent_basis,
               cs_definition, coordinate_law, center_law, longitude_offset_deg,
               cs_definition_tag, coordinate_law_tag, center_law_tag,
               EquiangularCubedSphereDefinition, GMAOCubedSphereDefinition,
               default_panel_connectivity, gnomonic_panel_connectivity, reciprocal_edge,
               GnomonicPanelConvention, GEOSNativePanelConvention,
               EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST
using ..State: AbstractMassBasis, DryBasis, MoistBasis
using ..Regridding: build_regridder, apply_regridder!
using ..Quantities: QuantityKind, IntensiveCellField, ExtensiveCellField,
                    HorizontalVectorField, HorizontalFluxField
using ..MetDrivers: TransportBinaryReader, TransportBinaryHeader, write_transport_binary,
                    TRANSPORT_BINARY_FORMAT_VERSION,
                    StreamingTransportBinaryWriter,
                    open_streaming_transport_binary, write_streaming_window!,
                    close_streaming_transport_binary!,
                    set_streaming_steps_per_window_schedule!,
                    set_transport_header_steps_per_window_schedule!,
                    open_streaming_cs_transport_binary, write_streaming_cs_window!,
                    load_window!, load_flux_delta_window!,
                    has_tm5_convection, load_tm5_convection_window!,
                    load_surface_window!,
                    TransportBinaryContract, canonical_window_constant_contract,
                    validate_transport_contract!,
                    recompute_cm_from_dm_target!, recompute_faceindexed_cm_from_dm_target!,
                    verify_window_continuity, verify_window_continuity_ll,
                    verify_window_continuity_rg, verify_window_continuity_cs,
                    delta_semantics,
                    replay_tolerance, run_replay_gate,
                    structured_replay_layout, faceindexed_replay_layout

# Re-export the CS preprocessor contract surface. `cubed_sphere_contracts.jl`
# is the single source of truth for the per-window write-time gates that every
# CS-producing preprocessor (spectral, regrid, GEOS-native) must call; making
# these public lets external auditors (the binary inspector, focused tests,
# `scripts/diagnostics/*`) gate a binary against the same contract without
# duplicating logic.
export verify_write_replay_cs!,
       verify_substep_positivity_cs!,
       verify_cs_window_contract!,
       init_cs_positivity_accumulator,
       update_cs_positivity_accumulator,
       summarize_cs_positivity_status

# Typed window-contract surface. The abstract types
# (`AbstractWindowContract` / `AbstractWindowWorkspace` /
# `AbstractBinaryWriter`) plus the concrete LL / CS / RG contract
# structs and per-topology positivity kernels. Mass-basis tags
# (`AbstractMassBasis` / `DryBasis` / `MoistBasis`) are reused from
# `State.Basis` — `AbstractBinaryWriter{G, FT, Basis}` parametrizes
# directly on those existing nominals so a writer↔reader pairing
# mismatch is a compile-time `MethodError`. Exported so external
# auditors (focused tests, the binary inspector,
# `scripts/diagnostics/*`) can gate a binary against the same contract
# without duplicating logic.
export mass_basis_symbol, mass_basis_from_symbol
export AbstractWindowContract, AbstractWindowWorkspace, AbstractBinaryWriter
export SubstepSchedulePolicy, clamp_substeps, initial_substeps,
       required_substeps, next_substeps, rescale_substep_amounts!,
       contract_steps_for_window, set_contract_steps_schedule!
export ReadyWindow, PreverifiedWindow, PreprocessorRunCache
export verify_window!, update_accumulator!, summarize_status!
export contract_replay_tolerance, contract_cfl_limit, contract_require_positivity
export allocate_window_workspace, reset_workspace!,
       ingest_window!, drain_ready_windows!, flush_final_windows!
export LatLonBinaryWriter, ReducedGaussianBinaryWriter, CubedSphereBinaryWriter
export write_window!, close_streaming_binary!, promote_streaming_binary!,
       quarantine_streaming_binary!, writer_staging_path, writer_final_path
export UnifiedPreprocessorDay, run_unified_preprocessor_day!
export driver_windows_per_day, driver_ingest_window!,
       driver_drain_ready_windows!, driver_flush_final_windows!,
       driver_after_write_window!
export CubedSphereContract, LatLonContract, ReducedGaussianContract
export verify_substep_positivity_ll!, verify_ll_window_contract!,
       init_ll_positivity_accumulator, update_ll_positivity_accumulator,
       summarize_ll_positivity_status
export verify_substep_positivity_rg!, verify_rg_window_contract!,
       verify_boundary_stub_flux_rg,
       init_rg_positivity_accumulator, update_rg_positivity_accumulator,
       summarize_rg_positivity_status

# Met source abstraction (AbstractMetSettings + RawWindow)
include("met_sources.jl")

# Physical constants
include("constants.jl")

# Statistics dependency for GEOS reader (level-orientation auto-detect)
using Statistics: mean

# Logging utilities
include("logging.jl")

# Vertical coordinate handling and level merging
include("vertical_coordinates.jl")

# Typed vertical-transform surface (vertical axis of the unified
# preprocessor). Wraps `merge_thin_levels` and
# `select_levels_echlevs` into first-class transform types so layer
# merging is available to every preprocessor pathway (LL/RG/CS ×
# spectral/native), not just the ERA5 spectral path.
include("vertical_transforms.jl")

# Global 6-panel Poisson balance for cubed-sphere grids
# (must precede target_geometry.jl which uses CSGlobalFaceTable)
include("cs_poisson_balance.jl")

# Target grid geometry (LL, RG, and CS)
include("target_geometry.jl")

# GRIB spectral IO
include("spectral_io.jl")

# Spectral synthesis (Legendre + FFT → gridpoint)
include("spectral_synthesis.jl")

# Poisson balance (LL FFT + RG conjugate gradient)
include("mass_support.jl")

# Compressed-Laplacian Poisson balance for RG (replaces slow CG on LCM faces)
include("ring_poisson_balance.jl")

# Reduced Gaussian helpers (RG synthesis, RG balance, RG cm diagnosis)
include("reduced_transport_helpers.jl")

# Cubed-sphere transport helpers (regrid, wind recovery, flux reconstruction)
include("cs_transport_helpers.jl")

# Configuration parsing
include("configuration.jl")

# TM5 convection ec2tm conversion (plan 23 Commit 3)
# (Loaded before binary_pipeline so the LL process_day hook can
# reference TM5PreprocessingWorkspace / TM5CleanupStats by type.)
include("tm5_convection_conversion.jl")

# ERA5 physics NC → BIN converter + mmap reader (plan 24 Commit 2)
include("era5_physics_binary.jl")

# ERA5 single-level surface reader for raw PBL diffusion fields.
include("era5_surface_reader.jl")

# TM5 convection preprocessor pipeline wiring (plan 24 Commit 4)
include("tm5_convection_pipeline.jl")

# Transport-binary workflows and shared preprocessing contracts.
include("binary_pipeline.jl")

# Native GEOS NetCDF reader (Commit 3 of plan indexed-baking-valiant)
include("sources/geos.jl")

# TOML-driven met-source factory (Commit 4)
include("sources/loader.jl")

# Typed met-reader surface (source axis of the unified preprocessor;
# additive facade over `AbstractMetSettings` machinery above). Includes
# here so the abstract types are visible to
# `transport_binary/cubed_sphere_geos.jl` even though the existing
# `process_day` orchestrators are not yet ported.
include("met_readers.jl")

# GEOS → CS passthrough orchestrator (Commit 5)
include("transport_binary/cubed_sphere_geos.jl")

# Met source abstraction (Commit 1 of plan indexed-baking-valiant)
export AbstractMetSettings, RawWindow
export read_window!, source_grid, windows_per_day, has_convection
export open_day, close_day!, allocate_raw_window

# GEOS native NetCDF reader (Commit 3)
export AbstractGEOSSettings, GEOSSettings, GEOSITSettings, GEOSFPSettings
export GEOSDayHandles, open_geos_day, close_geos_day!
export GEOSFPNativeDayHandles, geosfp_native_hourly_ctm_path
export geos_collection_path, detect_level_orientation
export endpoint_dry_mass, endpoint_dry_mass!

# Met-source TOML factory (Commit 4) + vertical-coordinate helper used by GEOS CLI
export load_met_settings, load_hybrid_coefficients

# Typed met-reader surface (source axis)
export AbstractMetReader, AbstractChainPolicy, NoChain, ChainedMass
export GEOSNativeReader, ERA5SpectralReader, ERA5SpectralSettings
export open_reader, close_reader!, end_of_day_seed, set_end_of_day_seed!
export native_vertical, window_metadata
# Note: `windows_per_day` and `read_window!` are already exported above
# (the existing `AbstractMetSettings` trait surface). The reader surface
# adds typed methods on the same generic functions.

# Typed vertical-transform surface (vertical axis)
export AbstractVerticalTransform, VerticalPlan
export IdentityVertical, MergeByIndex, MergeLayersThinnerThan,
       MergeAbovePressure, LevelSelection, PressureOverlap
export AbstractFieldKind
export MassField, TracerMassField, MassFluxField, PressureFluxField,
       ConvectionInterfaceFlux, ConvectionTendencyField,
       IntensiveCenterField, SurfaceField
export plan_vertical, apply_vertical!

# Exports for the CLI script and advanced users
export build_target_geometry, target_summary
export process_day, regrid_transport_binary, regrid_ll_binary_to_cs
export ec2tm!
export ec2tm_from_rates!, TM5CleanupStats
export dz_hydrostatic_virtual!, dz_hydrostatic_constT!
export convert_era5_physics_nc_to_bin
export ERA5PhysicsBinaryReader, ERA5PhysicsBinaryHeader
export open_era5_physics_binary, close_era5_physics_binary, get_era5_physics_field
export tm5_native_fields_for_hour!, merge_tm5_field_3d!
export TM5PreprocessingWorkspace, allocate_tm5_workspace
export compute_tm5_merged_hour_on_source!, log_tm5_cleanup_stats
export tm5_copy_or_regrid_ll!
export resolve_tm5_convection_settings

end # module Preprocessing
