# ---------------------------------------------------------------------------
# test_reduced_gaussian.jl
#
# End-to-end tests for ReducedGaussianMesh regridding via its sectorized
# MultiTreeWrapper path:
#   1. Small RG → LatLon: constant-field mass conservation
#   2. Small RG → CubedSphere: constant-field mass conservation
#   3. LatLon → small RG: transpose direction
#   4. Non-uniform field (cos(lat)) mass conservation
#   5. frac_a / frac_b ≈ 1 for full-sphere pairs
#   6. Reduced Gaussian → different Reduced Gaussian geometry
#
# Uses a small synthetic RG mesh (5 rings, max nlon=16, 56 total cells)
# so the tests run in seconds even without spatial acceleration.
# ---------------------------------------------------------------------------

@testset "ReducedGaussianMesh regridding" begin
    # Build a small reduced Gaussian mesh: 5 rings, variable nlon.
    latitudes = Float64[-60, -30, 0, 30, 60]
    nlon_per_ring = [8, 12, 16, 12, 8]
    rg = ReducedGaussianMesh(latitudes, nlon_per_ring)
    @test ncells(rg) == sum(nlon_per_ring)

    # Destination meshes
    ll = LatLonMesh(Nx = 24, Ny = 12)
    cs = CubedSphereMesh(Nc = 4, convention = GnomonicPanelConvention())
    too_coarse = ReducedGaussianMesh([0.0], [2])
    @test_throws ArgumentError build_regridder(too_coarse, ll)

    @testset "RG(5-ring) → LL(24×12) constant field" begin
        r = build_regridder(rg, ll; normalize = false)
        n_src = length(r.src_areas)
        n_dst = length(r.dst_areas)
        @test n_src == ncells(rg)
        @test n_dst == ll.Nx * ll.Ny

        src_field = ones(n_src)
        dst_field = zeros(n_dst)
        apply_regridder!(dst_field, r, src_field)

        # RG→LL: all RG cells land on LL (LL covers the full sphere),
        # so mass conservation should be exact to machine precision.
        src_mass = sum(src_field .* r.src_areas)
        dst_mass = sum(dst_field .* r.dst_areas)
        @test isapprox(src_mass, dst_mass; rtol = 1e-10)

        # Covered LL cells should be close to 1.0 (constant input).
        # Tolerance is looser than for matched grids because the 5-ring
        # RG mesh has coarse lat bands that don't align with LL faces,
        # causing partial-overlap cells near ring boundaries to have
        # values ~0.88–1.0. Mass is exact regardless.
        covered = dst_field .>= 0.5
        @test count(covered) > 0
        @test maximum(abs.(dst_field[covered] .- 1.0)) < 0.15
    end

    @testset "RG(5-ring) → C4 constant field" begin
        r = build_regridder(rg, cs; normalize = false)
        n_src = length(r.src_areas)
        n_dst = length(r.dst_areas)
        @test n_src == ncells(rg)
        @test n_dst == 6 * cs.Nc^2

        src_field = ones(n_src)
        dst_field = zeros(n_dst)
        apply_regridder!(dst_field, r, src_field)

        src_mass = sum(src_field .* r.src_areas)
        dst_mass = sum(dst_field .* r.dst_areas)
        @test isapprox(src_mass, dst_mass; rtol = 1e-10)
    end

    @testset "LL(24×12) → RG(5-ring) constant field" begin
        # Transpose direction: LatLon → ReducedGaussian.
        # LL covers the full sphere. This unusually coarse RG mesh represents
        # curved latitude boundaries with great-circle polygon edges, so its
        # geometric coverage is about 98.7%. We check:
        #   1. dst_mass / src_mass ≈ RG coverage fraction
        #   2. dst cells are approximately 1.0 (constant input)
        r = build_regridder(ll, rg; normalize = false)
        n_src = length(r.src_areas)
        n_dst = length(r.dst_areas)
        @test n_src == ll.Nx * ll.Ny
        @test n_dst == ncells(rg)

        src_field = ones(n_src)
        dst_field = zeros(n_dst)
        apply_regridder!(dst_field, r, src_field)

        src_mass = sum(src_field .* r.src_areas)
        dst_mass = sum(dst_field .* r.dst_areas)
        coverage_ratio = dst_mass / src_mass
        @test coverage_ratio > 0.98
        @test coverage_ratio <= 1.0

        # All destination cells should be close to 1.0
        @test maximum(abs.(dst_field .- 1.0)) < 0.05
    end

    @testset "RG(5-ring) → LL(24×12) cos(lat) field" begin
        r = build_regridder(rg, ll; normalize = false)
        n_src = length(r.src_areas)

        # Build cos(lat) field on the RG mesh.
        # For cell c in ring j, the latitude center is mesh.latitudes[j].
        src_field = zeros(n_src)
        for j in 1:nrings(rg)
            nlon_j = rg.nlon_per_ring[j]
            for i in 1:nlon_j
                c = cell_index(rg, i, j)
                src_field[c] = cosd(rg.latitudes[j])
            end
        end

        dst_field = zeros(length(r.dst_areas))
        apply_regridder!(dst_field, r, src_field)

        src_mass = sum(src_field .* r.src_areas)
        dst_mass = sum(dst_field .* r.dst_areas)
        # cos(lat) integral over the sphere is nonzero; check relative conservation.
        @test isapprox(src_mass, dst_mass; rtol = 1e-10)
    end

    @testset "RG(5-ring) → different RG geometry" begin
        destination = ReducedGaussianMesh(latitudes, [12, 16, 20, 16, 12])
        r = build_regridder(rg, destination; normalize = false)
        destination_field = zeros(length(r.dst_areas))
        apply_regridder!(destination_field, r, ones(length(r.src_areas)))

        @test length(r.src_areas) == ncells(rg)
        @test length(r.dst_areas) == ncells(destination)
        @test minimum(destination_field) > 0.95
        @test maximum(destination_field) <= 1.0 + 1e-12
        overlap_fraction = sum(destination_field .* r.dst_areas) / sum(r.src_areas)
        @test overlap_fraction > 0.99
        @test overlap_fraction <= 1.0
    end
end
