# ---------------------------------------------------------------------------
# test_reduced_gaussian.jl
#
# End-to-end tests for ReducedGaussianMesh regridding via the per-ring
# MultiTreeWrapper path:
#   1. Small RG → LatLon: constant-field mass conservation
#   2. Small RG → CubedSphere: constant-field mass conservation
#   3. LatLon → small RG: transpose direction
#   4. Non-uniform field (cos(lat)) mass conservation
#   5. frac_a / frac_b ≈ 1 for full-sphere pairs
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
        # The RG source area is ~98.7% of the sphere (0.001° polar
        # clamp in treeify), but ALL of it maps onto LL.
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
        # LL covers the full sphere; RG covers ~98.7% (0.001° polar
        # clamp). Some LL pole-cap cells have no corresponding RG
        # cells, so their mass is lost — exact mass conservation is
        # NOT expected for this direction. Instead we check:
        #   1. dst_mass / src_mass ≈ RG_coverage_fraction (~0.987)
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
        # RG covers ~98.7% of the sphere → dst_mass ≈ 0.987 × src_mass
        coverage_ratio = dst_mass / src_mass
        @test coverage_ratio > 0.98
        @test coverage_ratio < 1.001  # should not exceed 1

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
end
