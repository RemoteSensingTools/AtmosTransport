using Test
using AtmosTransportModel
using AtmosTransportModel.Architectures
using AtmosTransportModel.Grids
using AtmosTransportModel.IO: default_met_config, build_vertical_coordinate

# Build vertical coordinate from GEOS-FP config
config = default_met_config("geosfp")
vc = build_vertical_coordinate(config; FT = Float64)

# Construct C48 grid
grid = CubedSphereGrid(CPU(); Nc = 48, vertical = vc)

const R_EARTH = 6.371e6  # m
const AREA_SPHERE = 4 * π * R_EARTH^2

@testset "CubedSphereGrid" begin
    Nc = grid.Nc
    Nz = grid.Nz

    @testset "Area tests" begin
        total_area = 0.0
        for p in 1:6
            for j in 1:Nc, i in 1:Nc
                total_area += cell_area(i, j, grid; panel = p)
            end
        end
        @test total_area ≈ AREA_SPHERE rtol = 1e-10

        panel_areas = [sum(cell_area(i, j, grid; panel = p) for i in 1:Nc, j in 1:Nc) for p in 1:6]
        expected_per_panel = total_area / 6
        for p in 1:6
            @test panel_areas[p] ≈ expected_per_panel rtol = 1e-12
        end

        for p in 1:6
            for j in 1:Nc, i in 1:Nc
                @test cell_area(i, j, grid; panel = p) > 0
            end
        end
    end

    @testset "Coordinate tests" begin
        # Center cell indices (closest to panel center for even Nc)
        ic, jc = Nc ÷ 2 + 1, Nc ÷ 2 + 1

        # Panel 1: front (0°E equator)
        lon1 = xnode(ic, jc, grid, Center(); panel = 1)
        lat1 = ynode(ic, jc, grid, Center(); panel = 1)
        @test abs(lon1) < 2
        @test abs(lat1) < 2

        # Panel 2: east (90°E equator)
        lon2 = xnode(ic, jc, grid, Center(); panel = 2)
        lat2 = ynode(ic, jc, grid, Center(); panel = 2)
        @test abs(lon2 - 90) < 2 || abs(lon2 + 270) < 2
        @test abs(lat2) < 2

        # Panel 3: back (180°E equator)
        lon3 = xnode(ic, jc, grid, Center(); panel = 3)
        lat3 = ynode(ic, jc, grid, Center(); panel = 3)
        @test abs(abs(lon3) - 180) < 2
        @test abs(lat3) < 2

        # Panel 4: west (90°W equator)
        lon4 = xnode(ic, jc, grid, Center(); panel = 4)
        lat4 = ynode(ic, jc, grid, Center(); panel = 4)
        @test abs(lon4 + 90) < 2 || abs(lon4 - 270) < 2
        @test abs(lat4) < 2

        # Panel 5: north pole
        lat5 = ynode(ic, jc, grid, Center(); panel = 5)
        @test lat5 > 88

        # Panel 6: south pole
        lat6 = ynode(ic, jc, grid, Center(); panel = 6)
        @test lat6 < -88
    end

    @testset "Metric term tests" begin
        all_dx = Float64[]
        all_dy = Float64[]
        for p in 1:6
            for j in 1:Nc, i in 1:Nc
                push!(all_dx, Δx(i, j, grid; panel = p))
                push!(all_dy, Δy(i, j, grid; panel = p))
            end
        end

        for dx in all_dx
            @test dx > 0
        end
        for dy in all_dy
            @test dy > 0
        end

        @test minimum(all_dx) ≈ minimum(all_dy) rtol = 1e-10
        @test maximum(all_dx) ≈ maximum(all_dy) rtol = 1e-10

        # C48: Δx between 100 km and 300 km
        dx_min_km = minimum(all_dx) / 1e3
        dx_max_km = maximum(all_dx) / 1e3
        @test 100 <= dx_min_km <= 300
        @test 100 <= dx_max_km <= 300
    end

    @testset "Accessor tests" begin
        gs = grid_size(grid)
        @test gs.Nc == 48
        @test gs.Nz == 72
        @test gs.Npanels == 6

        top = topology(grid)
        @test top[1] isa CubedPanel
        @test top[2] isa CubedPanel

        for p in 1:6
            for j in 1:Nc, i in 1:Nc
                @test cell_area(i, j, grid; panel = p) > 0
                @test Δx(i, j, grid; panel = p) > 0
                @test Δy(i, j, grid; panel = p) > 0
            end
        end

        # xnode and ynode for Center and Face
        for p in 1:6
            x_c = xnode(1, 1, grid, Center(); panel = p)
            y_c = ynode(1, 1, grid, Center(); panel = p)
            @test isfinite(x_c) && isfinite(y_c)

            x_f = xnode(1, 1, grid, Face(); panel = p)
            y_f = ynode(1, 1, grid, Face(); panel = p)
            @test isfinite(x_f) && isfinite(y_f)
        end

        # znode, Δz, cell_volume with vertical coordinate
        p_center = znode(1, grid, Center())
        p_face = znode(1, grid, Face())
        @test p_center > 0 && p_face > 0

        dz1 = Δz(1, grid)
        @test dz1 > 0

        vol = cell_volume(1, 1, 1, grid; panel = 1)
        @test vol > 0
    end

    @testset "Symmetry tests" begin
        # Panel 1 and Panel 3: opposite faces, identical cell area distributions
        areas_p1 = [cell_area(i, j, grid; panel = 1) for i in 1:Nc, j in 1:Nc]
        areas_p3 = [cell_area(i, j, grid; panel = 3) for i in 1:Nc, j in 1:Nc]
        @test areas_p1 ≈ areas_p3 rtol = 1e-12

        # Panel 5 and Panel 6: polar faces, identical cell area distributions
        areas_p5 = [cell_area(i, j, grid; panel = 5) for i in 1:Nc, j in 1:Nc]
        areas_p6 = [cell_area(i, j, grid; panel = 6) for i in 1:Nc, j in 1:Nc]
        @test areas_p5 ≈ areas_p6 rtol = 1e-12
    end
end
