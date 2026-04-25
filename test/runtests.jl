#!/usr/bin/env julia
#
# Main test suite entrypoint for AtmosTransport.
#
# Usage:
#   julia --project=. test/runtests.jl              # core tests (no external data)
#   julia --project=. test/runtests.jl --all         # include real-data tests
#
# Core tests run without any data files and validate types, kernels,
# dispatch, and runtime logic. Real-data tests require preprocessed
# ERA5 binaries and Catrine ICs in ~/data/AtmosTransport/.

const RUN_ALL = "--all" in ARGS

function run_test_file_isolated(test_file::AbstractString)
    mod_name = Symbol("Test_", replace(basename(test_file), "." => "_"))
    mod = Module(mod_name)
    Core.eval(mod, :(include(path::AbstractString) = Base.include($mod, path)))
    Core.eval(mod, :(include(mapexpr::Function, path::AbstractString) = Base.include(mapexpr, $mod, path)))
    Core.eval(mod, :(eval(expr) = Core.eval($mod, expr)))
    return Base.include(mod, joinpath(@__DIR__, test_file))
end

# ── Core tests (no external data) ──────────────────────────────────

core_tests = [
    "test_met_sources_trait.jl",
    "test_identity_regrid.jl",
    "test_geos_reader.jl",
    "test_met_source_loader.jl",
    "test_basis_explicit_core.jl",
    "test_advection_kernels.jl",
    "test_structured_mesh_metadata.jl",
    "test_reduced_gaussian_mesh.jl",
    "test_driven_simulation.jl",
    "test_transport_model_convection.jl",
    "test_tm5_convection.jl",
    "test_tm5_preprocessing.jl",
    "test_tm5_preprocessing_rates.jl",
    "test_era5_physics_binary.jl",
    "test_tm5_vertical_remap.jl",
    "test_tm5_process_day.jl",
    "test_tm5_vs_cmfmc_parity.jl",
    "test_tm5_driven_simulation.jl",
    "test_cubed_sphere_advection.jl",
    "test_cubed_sphere_runtime.jl",
    "test_cs_chemistry.jl",
    "test_poisson_balance.jl",
    "test_replay_consistency.jl",
    "test_run_transport_binary_recipe.jl",
    "test_initial_condition_io.jl",
    "test_binary_path_expander.jl",
    "test_binary_inspector.jl",
    "test_cs_driven_builders.jl",
    "test_ll_to_cs_regrid_script.jl",
    "test_preprocessing_cache_io.jl",
    "test_output_snapshots.jl",
    "test_visualization_snapshots.jl",
    "test_aqua.jl",
    "test_jet.jl",
    "test_readme_current.jl",
]

for test_file in core_tests
    @info "Running $test_file"
    run_test_file_isolated(test_file)
end

# ── Real-data tests (require preprocessed binaries) ────────────────

if RUN_ALL
    real_data_tests = [
        "test_dry_flux_interface.jl",
        "test_transport_binary_reader.jl",
        "test_era5_latlon_e2e.jl",
        "test_run_transport_binary_v2.jl",
        "test_tm5_catrine_1day.jl",
    ]

    for test_file in real_data_tests
        @info "Running $test_file"
        run_test_file_isolated(test_file)
    end
else
    @info "Skipping real-data tests (pass --all to include them)"
end

@info "Test suite complete."
