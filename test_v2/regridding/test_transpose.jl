# ---------------------------------------------------------------------------
# test_transpose.jl
#
# The LatLon→CS regridder and the independently-built CS→LatLon
# regridder must both be globally mass-conservative. Additionally,
# regridding LL→CS→LL on a constant field should recover the input
# to machine precision on every destination cell (the "round-trip
# identity").
#
# Note: LL→CS→LL does NOT preserve arbitrary smooth fields (that would
# require reciprocal intersection matrices, which neither we nor xESMF
# provide). Only constant fields are exact under round-trip; smooth
# fields get smoothed by the second mapping.
# ---------------------------------------------------------------------------

@testset "transpose direction" begin
    Nx, Ny, Nc = 36, 18, 4
    src = LatLonMesh(Nx = Nx, Ny = Ny)
    dst = CubedSphereMesh(Nc = Nc, convention = GnomonicPanelConvention())

    r_fwd = build_regridder(src, dst; normalize = false)
    r_bwd = build_regridder(dst, src; normalize = false)

    n_src = length(r_fwd.src_areas)
    n_dst = length(r_fwd.dst_areas)

    @testset "constant round-trip LL → CS → LL" begin
        ll_in = ones(n_src)
        cs_mid = zeros(n_dst)
        apply_regridder!(cs_mid, r_fwd, ll_in)
        # CS should be uniformly 1
        @test maximum(abs.(cs_mid .- 1.0)) < 1e-12

        ll_out = zeros(n_src)
        apply_regridder!(ll_out, r_bwd, cs_mid)
        @test maximum(abs.(ll_out .- 1.0)) < 1e-12
    end

    @testset "mass conservation in both directions" begin
        # Forward: LL → CS
        src_field = [sind(2 * src.φᶜ[j]) * cosd(src.λᶜ[i])
                     for i in 1:Nx, j in 1:Ny]
        ll_flat = reshape(Float64.(src_field), n_src)
        cs_flat = zeros(n_dst)
        apply_regridder!(cs_flat, r_fwd, ll_flat)

        m_src = sum(ll_flat .* r_fwd.src_areas)
        m_dst = sum(cs_flat .* r_fwd.dst_areas)
        area_scale = sum(r_fwd.src_areas)
        @test abs(m_src - m_dst) < 1e-12 * area_scale

        # Backward: CS (constant) → LL
        cs_in = ones(n_dst)
        ll_out = zeros(n_src)
        apply_regridder!(ll_out, r_bwd, cs_in)
        m_src_b = sum(cs_in .* r_bwd.src_areas)
        m_dst_b = sum(ll_out .* r_bwd.dst_areas)
        @test isapprox(m_src_b, m_dst_b; rtol = 1e-12)
    end

    @testset "transpose regridder via LinearAlgebra.transpose" begin
        # CR.jl's `transpose(regridder)` creates a shared-data reverse
        # direction without building a new sparse matrix. Confirm it
        # gives the same result as an independently-built reverse
        # regridder on a constant field.
        import LinearAlgebra: transpose
        r_bwd_via_transpose = transpose(r_fwd)
        ll_out_a = zeros(n_src)
        ll_out_b = zeros(n_src)
        cs_in = ones(n_dst)
        apply_regridder!(ll_out_a, r_bwd, cs_in)
        apply_regridder!(ll_out_b, r_bwd_via_transpose, cs_in)
        @test maximum(abs.(ll_out_a .- ll_out_b)) < 1e-12
    end
end
