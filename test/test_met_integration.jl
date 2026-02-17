# ---------------------------------------------------------------------------
# Integration tests for the met data IO pipeline
#
# Tests the full flow: generate synthetic data → read via MetDataSource →
# prepare_met_for_physics → verify staggered arrays and physical correctness.
# ---------------------------------------------------------------------------

using Test
using AtmosTransportModel
using AtmosTransportModel.IO
using AtmosTransportModel.Grids
using AtmosTransportModel.Architectures: CPU

@testset "Met IO integration" begin
    # Ensure test data exists (generates if not present)
    include("generate_test_met_data.jl")

    test_data_dir = joinpath(@__DIR__, "data", "geosfp_test")
    config_path = joinpath(test_data_dir, "test_geosfp.toml")

    @testset "Read synthetic met data" begin
        met = MetDataSource(Float64, config_path; local_path = test_data_dir)
        @test met isa MetDataSource{Float64}
        @test protocol(met) == "local"
        @test source_name(met) == "GEOS-FP-Test"
        @test has_variable(met, :u_wind)
        @test has_variable(met, :w_wind)
        @test has_variable(met, :diffusivity)

        # Read at time 0.0 (matches our synthetic time coordinate)
        read_met!(met, 0.0)
        @test met.current_time[] == 0.0

        # Verify u_wind ≈ 5.0 + sinusoidal pattern
        u = get_field(met, :u_wind)
        @test size(u) == (8, 4, 5)
        @test all(u .>= 3.0) && all(u .<= 7.0)
        @test 4.5 < sum(u) / length(u) < 5.5

        # Verify v_wind ≈ -2.0 + perturbation
        v = get_field(met, :v_wind)
        @test size(v) == (8, 4, 5)
        @test all(v .>= -3.0) && all(v .<= -1.0)

        # Verify temperature decreases with level (280 - 12*k)
        t = get_field(met, :temperature)
        @test size(t) == (8, 4, 5)
        @test t[1, 1, 1] ≈ 268.0  # 280 - 12
        @test t[1, 1, 5] ≈ 220.0  # 280 - 60

        # Verify surface pressure
        ps = get_field(met, :surface_pressure)
        @test size(ps) == (8, 4)
        @test all(ps .≈ 101325.0)

        # Verify pressure thickness
        delp = get_field(met, :pressure_thickness)
        @test size(delp) == (8, 4, 5)
        @test all(delp .> 0)

        # Verify omega (w_wind) - small negative values
        omega = get_field(met, :w_wind)
        @test size(omega) == (8, 4, 5)
        @test all(omega .<= 0)
    end

    @testset "prepare_met_for_physics staggered arrays" begin
        met = MetDataSource(Float64, config_path; local_path = test_data_dir)
        read_met!(met, 0.0)

        vc = HybridSigmaPressure(
            [0.0, 5000.0, 10000.0, 20000.0, 50000.0, 101325.0],
            [0.0, 0.0, 0.1, 0.3, 0.7, 1.0],
        )
        grid = LatitudeLongitudeGrid(CPU();
            size = (8, 4, 5),
            longitude = (-180, 180),
            latitude = (-90, 90),
            vertical = vc,
        )

        result = prepare_met_for_physics(met, grid)

        # Staggered array sizes: u (Nx+1,Ny,Nz), v (Nx,Ny+1,Nz), w (Nx,Ny,Nz+1)
        @test size(result.u) == (9, 4, 5)
        @test size(result.v) == (8, 5, 5)
        @test size(result.w) == (8, 4, 6)

        # Diffusivity from trb_Ne has 6 edge levels
        @test haskey(result, :diffusivity)
        @test size(result.diffusivity) == (8, 4, 6)
    end

    @testset "Physical correctness: omega sign flip for w" begin
        met = MetDataSource(Float64, config_path; local_path = test_data_dir)
        read_met!(met, 0.0)

        vc = HybridSigmaPressure(
            [0.0, 5000.0, 10000.0, 20000.0, 50000.0, 101325.0],
            [0.0, 0.0, 0.1, 0.3, 0.7, 1.0],
        )
        grid = LatitudeLongitudeGrid(CPU();
            size = (8, 4, 5),
            longitude = (-180, 180),
            latitude = (-90, 90),
            vertical = vc,
        )

        result = prepare_met_for_physics(met, grid)

        # omega > 0 is downward (GEOS convention); our w > 0 is upward
        # Bridge negates: w = -(omega[k-1] + omega[k])/2
        # Our synthetic omega is negative (-0.01 * ...), so w should be positive
        omega = get_field(met, :w_wind)
        @test all(omega .<= 0)
        # w at interior interfaces (k=2:Nz) should be positive when omega is negative
        @test all(result.w[:, :, 2:5] .>= 0)
        # Top and bottom boundaries are zero
        @test all(result.w[:, :, 1] .== 0)
        @test all(result.w[:, :, 6] .== 0)
    end
end
