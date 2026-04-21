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

# ── Core tests (no external data) ──────────────────────────────────

core_tests = [
    "test_basis_explicit_core.jl",
    "test_advection_kernels.jl",
    "test_structured_mesh_metadata.jl",
    "test_reduced_gaussian_mesh.jl",
    "test_driven_simulation.jl",
    "test_transport_model_convection.jl",
    "test_cubed_sphere_advection.jl",
    "test_cubed_sphere_runtime.jl",
    "test_poisson_balance.jl",
]

for test_file in core_tests
    @info "Running $test_file"
    include(test_file)
end

# ── Real-data tests (require preprocessed binaries) ────────────────

if RUN_ALL
    real_data_tests = [
        "test_dry_flux_interface.jl",
        "test_transport_binary_reader.jl",
        "test_transport_binary_v2_dispatch.jl",
        "test_preprocess_transport_binary_v2.jl",
        "test_era5_latlon_e2e.jl",
        "test_run_transport_binary_v2.jl",
    ]

    for test_file in real_data_tests
        @info "Running $test_file"
        include(test_file)
    end
else
    @info "Skipping real-data tests (pass --all to include them)"
end

@info "Test suite complete."
