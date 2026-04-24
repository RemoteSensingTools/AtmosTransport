# ---------------------------------------------------------------------------
# test_cubed_sphere_corners.jl
#
# Golden tests for `cubed_sphere_face_corners(mesh)`:
# - Returns an NTuple of 6 matrices of shape (Nc+1, Nc+1)
# - All vertices are unit-length (within roundoff)
# - Panel centers land at the expected physical locations on the sphere
# - GEOSNativePanelConvention matches the GEOS-FP/GEOS-IT panel ordering,
#   global -10° longitude offset, and native panel orientations.
# ---------------------------------------------------------------------------

using LinearAlgebra: norm
import GeometryOps as GO
using .AtmosTransport.Regridding: cubed_sphere_face_corners

function _lonlat(pt)
    x, y, z = pt[1], pt[2], pt[3]
    lon = rad2deg(atan(y, x))
    lon < 0 && (lon += 360)
    lat = rad2deg(asin(z / norm((x, y, z))))
    return lon, lat
end

_lon_close(a, b; atol=1e-10) = abs(mod(a - b + 180, 360) - 180) <= atol

@testset "cubed_sphere_face_corners" begin
    @testset "shape + unit norm (Gnomonic)" begin
        Nc = 8
        mesh = CubedSphereMesh(Nc = Nc, convention = GnomonicPanelConvention())
        corners = cubed_sphere_face_corners(mesh)
        @test length(corners) == 6
        for p in 1:6
            @test size(corners[p]) == (Nc + 1, Nc + 1)
            for pt in corners[p]
                @test isapprox(norm((pt[1], pt[2], pt[3])), 1.0; atol = 1e-12)
            end
        end
    end

    @testset "panel centers (Gnomonic)" begin
        # For even Nc the exact panel center lives at corner index
        # (Nc÷2 + 1, Nc÷2 + 1) — the cube-face pole (ξ=0, η=0). We
        # can read that corner directly and compare to the known
        # Cartesian direction of each gnomonic panel.
        Nc = 4
        mesh = CubedSphereMesh(Nc = Nc, convention = GnomonicPanelConvention())
        corners = cubed_sphere_face_corners(mesh)
        mid = Nc ÷ 2 + 1
        panel_center(p) = let pt = corners[p][mid, mid]
            [pt[1], pt[2], pt[3]]
        end

        @test isapprox(panel_center(1), [ 1.0,  0.0,  0.0]; atol = 1e-12)  # X+
        @test isapprox(panel_center(2), [ 0.0,  1.0,  0.0]; atol = 1e-12)  # Y+
        @test isapprox(panel_center(3), [-1.0,  0.0,  0.0]; atol = 1e-12)  # X−
        @test isapprox(panel_center(4), [ 0.0, -1.0,  0.0]; atol = 1e-12)  # Y−
        @test isapprox(panel_center(5), [ 0.0,  0.0,  1.0]; atol = 1e-12)  # N pole
        @test isapprox(panel_center(6), [ 0.0,  0.0, -1.0]; atol = 1e-12)  # S pole
    end

    @testset "panel centers (GEOS native)" begin
        Nc = 4
        mesh = CubedSphereMesh(Nc = Nc, convention = GEOSNativePanelConvention())
        corners = cubed_sphere_face_corners(mesh)
        mid = Nc ÷ 2 + 1
        pc(p) = _lonlat(corners[p][mid, mid])

        # GEOS native equatorial panel centers are offset by -10° from the
        # classical cube. Polar panel center corners are the poles.
        @test _lon_close(pc(1)[1], 350.0)
        @test isapprox(pc(1)[2], 0.0; atol = 1e-12)
        @test _lon_close(pc(2)[1], 80.0)
        @test isapprox(pc(2)[2], 0.0; atol = 1e-12)
        @test isapprox(pc(3)[2], 90.0; atol = 1e-12)
        @test _lon_close(pc(4)[1], 170.0)
        @test isapprox(pc(4)[2], 0.0; atol = 1e-12)
        @test _lon_close(pc(5)[1], 260.0)
        @test isapprox(pc(5)[2], 0.0; atol = 1e-12)
        @test isapprox(pc(6)[2], -90.0; atol = 1e-12)
    end

    @testset "outer corners (GEOS native NetCDF orientation)" begin
        Nc = 4
        mesh = CubedSphereMesh(Nc = Nc, convention = GEOSNativePanelConvention())
        corners = cubed_sphere_face_corners(mesh)
        edge_lat = rad2deg(asin(1 / sqrt(3)))

        expected = Dict(
            1 => ((305.0, -edge_lat), (35.0, -edge_lat), (35.0, edge_lat), (305.0, edge_lat)),
            2 => ((35.0, -edge_lat), (125.0, -edge_lat), (125.0, edge_lat), (35.0, edge_lat)),
            3 => ((35.0, edge_lat), (125.0, edge_lat), (215.0, edge_lat), (305.0, edge_lat)),
            4 => ((125.0, edge_lat), (215.0, edge_lat), (215.0, -edge_lat), (125.0, -edge_lat)),
            5 => ((215.0, edge_lat), (305.0, edge_lat), (305.0, -edge_lat), (215.0, -edge_lat)),
            6 => ((215.0, -edge_lat), (125.0, -edge_lat), (35.0, -edge_lat), (305.0, -edge_lat)),
        )
        for p in 1:6
            actual = (_lonlat(corners[p][1, 1]),
                      _lonlat(corners[p][end, 1]),
                      _lonlat(corners[p][end, end]),
                      _lonlat(corners[p][1, end]))
            for q in 1:4
                @test _lon_close(actual[q][1], expected[p][q][1])
                @test isapprox(actual[q][2], expected[p][q][2]; atol = 1e-10)
            end
        end
    end
end
