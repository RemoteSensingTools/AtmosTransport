# ---------------------------------------------------------------------------
# test_cubed_sphere_corners.jl
#
# Golden tests for `cubed_sphere_face_corners(mesh)`:
# - Returns an NTuple of 6 matrices of shape (Nc+1, Nc+1)
# - All vertices are unit-length (within roundoff)
# - Panel centers land at the expected physical locations on the sphere
# - GnomonicPanelConvention and GEOSNativePanelConvention agree on panel
#   count and shape; GEOS remap swaps panels 3/4/5 so panel 3 of GEOS is
#   the north pole (not X−).
# ---------------------------------------------------------------------------

using LinearAlgebra: norm
import GeometryOps as GO
using .AtmosTransportV2.Regridding: cubed_sphere_face_corners

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

    @testset "panel centers (GEOS native remap)" begin
        Nc = 4
        mesh = CubedSphereMesh(Nc = Nc, convention = GEOSNativePanelConvention())
        corners = cubed_sphere_face_corners(mesh)
        mid = Nc ÷ 2 + 1
        pc(p) = let pt = corners[p][mid, mid]; [pt[1], pt[2], pt[3]] end

        # GEOS ordering per _gnomonic_panel_id:
        #   user 1 → gnomonic 1 (X+)
        #   user 2 → gnomonic 2 (Y+)
        #   user 3 → gnomonic 5 (north pole)
        #   user 4 → gnomonic 3 (X−)
        #   user 5 → gnomonic 4 (Y−)
        #   user 6 → gnomonic 6 (south pole)
        @test isapprox(pc(1), [ 1.0,  0.0,  0.0]; atol = 1e-12)
        @test isapprox(pc(2), [ 0.0,  1.0,  0.0]; atol = 1e-12)
        @test isapprox(pc(3), [ 0.0,  0.0,  1.0]; atol = 1e-12)
        @test isapprox(pc(4), [-1.0,  0.0,  0.0]; atol = 1e-12)
        @test isapprox(pc(5), [ 0.0, -1.0,  0.0]; atol = 1e-12)
        @test isapprox(pc(6), [ 0.0,  0.0, -1.0]; atol = 1e-12)
    end
end
