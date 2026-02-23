using Test
using AtmosTransport.Grids
using AtmosTransport.Fields

@testset "Fields" begin
    @test Center() isa AbstractLocationType
    @test Face() isa AbstractLocationType

    @testset "Field on LatitudeLongitudeGrid" begin
        vc = HybridSigmaPressure(
            [0.0, 5000.0, 10000.0, 20000.0, 50000.0, 101325.0],
            [0.0, 0.0,    0.1,     0.3,      0.7,     1.0])

        grid = LatitudeLongitudeGrid(CPU();
            size = (36, 18, 5),
            longitude = (-180, 180),
            latitude = (-90, 90),
            vertical = vc)

        field = Field(Center(), Center(), Center(), grid)

        # interior has right size (Nx, Ny, Nz)
        int = interior(field)
        gs = grid_size(grid)
        @test size(int) == (gs.Nx, gs.Ny, gs.Nz)
        @test size(int) == (36, 18, 5)

        # set!(field, 1.0) fills all interior values with 1.0
        set!(field, 1.0)
        @test all(int .== 1.0)

        # data(field) returns the full array including halos
        d = data(field)
        hs = halo_size(grid)
        expected_size = (gs.Nx + 2 * hs.Hx, gs.Ny + 2 * hs.Hy, gs.Nz + 2 * hs.Hz)
        @test size(d) == expected_size
    end

    @testset "Float32 field" begin
        vc32 = HybridSigmaPressure(Float32[0, 5000, 10000, 101325], Float32[0, 0, 0.5, 1])
        grid32 = LatitudeLongitudeGrid(CPU(); FT=Float32, size=(36, 18, 3), vertical=vc32)
        field32 = Field(Center(), Center(), Center(), grid32)
        int = interior(field32)
        @test eltype(int) == Float32
    end
end
