using Test
using AtmosTransportModel.Grids

@testset "Grids" begin
    @testset "LatitudeLongitudeGrid" begin
        vc = HybridSigmaPressure(
            [0.0, 5000.0, 10000.0, 20000.0, 50000.0, 101325.0],
            [0.0, 0.0,    0.1,     0.3,      0.7,     1.0])

        grid = LatitudeLongitudeGrid(CPU();
            size = (36, 18, 5),
            longitude = (-180, 180),
            latitude = (-90, 90),
            vertical = vc)

        gs = grid_size(grid)
        @test gs.Nx == 36
        @test gs.Ny == 18
        @test gs.Nz == 5

        @test topology(grid) == (Periodic(), Bounded())

        # Cell area should be positive and vary with latitude
        a_eq = cell_area(1, 9, grid)   # near equator
        a_pole = cell_area(1, 1, grid)  # near south pole
        @test a_eq > 0
        @test a_pole > 0
        @test a_eq > a_pole  # cells are larger at equator
    end

    @testset "Float32 grid" begin
        vc32 = HybridSigmaPressure(Float32[0, 5000, 10000, 101325], Float32[0, 0, 0.5, 1])
        grid32 = LatitudeLongitudeGrid(CPU(); FT=Float32, size=(36, 18, 3), vertical=vc32)
        @test cell_area(1, 9, grid32) isa Float32
        @test Δx(1, 9, grid32) isa Float32
    end

    @testset "HybridSigmaPressure" begin
        A = [0.0, 5000.0, 10000.0, 101325.0]
        B = [0.0, 0.0,    0.5,     1.0]
        vc = HybridSigmaPressure(A, B)

        @test n_levels(vc) == 3
        @test pressure_at_interface(vc, 1, 101325.0) == 0.0       # top
        @test pressure_at_interface(vc, 4, 101325.0) == 202650.0   # surface
        @test level_thickness(vc, 1, 101325.0) > 0
    end
end
