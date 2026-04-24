# ---------------------------------------------------------------------------
# test_conservation.jl
#
# Global mass conservation for LatLon → CubedSphere conservative
# regridding, across three field families:
#   - constant (should be exact)
#   - linear in sin(lat) (smooth, zonally uniform)
#   - zonal + meridional mode (Fourier-style mix)
#
# Two directions (LL→CS and CS→LL), two grid sizes. Each test checks
# the global mass integral: sum(f .* src_areas) ≈ sum(dst .* dst_areas)
# to rtol 1e-12.
# ---------------------------------------------------------------------------

using LinearAlgebra

@testset "conservation LatLon ↔ CubedSphere" begin
    # Use two grid pairs: a tiny one for fast signal and a slightly
    # larger one to flex the multithreaded dual DFS code paths.
    for (Nx, Ny, Nc) in ((36, 18, 4), (72, 36, 12))
        src = LatLonMesh(Nx = Nx, Ny = Ny)
        dst = CubedSphereMesh(Nc = Nc, convention = GnomonicPanelConvention())

        @testset "LL($Nx×$Ny) → C$Nc" begin
            r = build_regridder(src, dst; normalize = false)
            n_src = length(r.src_areas)
            n_dst = length(r.dst_areas)

            # Field 1: constant — should be exact to machine precision.
            src_const = ones(n_src)
            dst_const = zeros(n_dst)
            apply_regridder!(dst_const, r, src_const)
            @test maximum(abs.(dst_const .- 1.0)) < 1e-12

            src_mass = sum(src_const .* r.src_areas)
            dst_mass = sum(dst_const .* r.dst_areas)
            @test isapprox(src_mass, dst_mass; rtol = 1e-12)

            # Field 2: sin(lat) — smooth, latitude-only.
            src_linlat = Float64[sind(src.φᶜ[j]) for i in 1:Nx, j in 1:Ny]
            src_linlat_flat = reshape(src_linlat, n_src)
            dst_linlat = zeros(n_dst)
            apply_regridder!(dst_linlat, r, src_linlat_flat)
            src_mass = sum(src_linlat_flat .* r.src_areas)
            dst_mass = sum(dst_linlat .* r.dst_areas)
            # Both integrals are ~ zero (antisymmetric about equator),
            # so check absolute difference against the magnitude of
            # src_areas.
            area_scale = sum(r.src_areas)
            @test abs(src_mass - dst_mass) < 1e-12 * area_scale

            # Field 3: cos(lat)·sin(2·lon) — Fourier mode, nonzero mass.
            src_wave = Float64[cosd(src.φᶜ[j]) * sind(2 * src.λᶜ[i])
                               for i in 1:Nx, j in 1:Ny]
            src_wave_flat = reshape(src_wave, n_src)
            dst_wave = zeros(n_dst)
            apply_regridder!(dst_wave, r, src_wave_flat)
            src_mass = sum(src_wave_flat .* r.src_areas)
            dst_mass = sum(dst_wave .* r.dst_areas)
            @test abs(src_mass - dst_mass) < 1e-12 * area_scale
        end

        @testset "C$Nc → LL($Nx×$Ny)" begin
            rt = build_regridder(dst, src; normalize = false)
            n_src_t = length(rt.src_areas)  # CS cells
            n_dst_t = length(rt.dst_areas)  # LL cells

            # Constant
            src_const = ones(n_src_t)
            dst_const = zeros(n_dst_t)
            apply_regridder!(dst_const, rt, src_const)
            @test maximum(abs.(dst_const .- 1.0)) < 1e-12
            @test isapprox(sum(src_const .* rt.src_areas),
                           sum(dst_const .* rt.dst_areas); rtol = 1e-12)
        end
    end

    @testset "GEOS-native CubedSphere treeify" begin
        Nx, Ny, Nc = 36, 18, 4
        ll = LatLonMesh(Nx = Nx, Ny = Ny)
        geos = CubedSphereMesh(Nc = Nc, convention = GEOSNativePanelConvention())

        @testset "LL → GEOS-native C$Nc" begin
            r = build_regridder(ll, geos; normalize = false)
            src = ones(length(r.src_areas))
            dst = zeros(length(r.dst_areas))
            apply_regridder!(dst, r, src)
            @test maximum(abs.(dst .- 1.0)) < 1e-12
            @test isapprox(sum(src .* r.src_areas), sum(dst .* r.dst_areas); rtol = 1e-12)
        end

        @testset "GEOS-native C$Nc → LL" begin
            r = build_regridder(geos, ll; normalize = false)
            src = ones(length(r.src_areas))
            dst = zeros(length(r.dst_areas))
            apply_regridder!(dst, r, src)
            @test maximum(abs.(dst .- 1.0)) < 1e-12
            @test isapprox(sum(src .* r.src_areas), sum(dst .* r.dst_areas); rtol = 1e-12)
        end
    end
end
