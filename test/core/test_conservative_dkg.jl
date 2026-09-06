include(joinpath(@__DIR__, "..", "helpers", "conservative_dkg.jl"))

@testset "Conservative Dkg mass solve and transpose" begin
    for FT in (Float32, Float64), Nz in (1, 2, 66), Nt in (1, 7), strength in (0, 0.1, 40, 1e4)
        @testset "$FT Nz=$Nz Nt=$Nt exchange=$strength" begin
            check_conservative_dkg(FT, 2, Nz, Nt, strength)
        end
    end
end

@testset "Dkg resolves weak transfers into empty layers" begin
    for FT in (Float32,Float64)
        check_dkg_weak_exchange(FT)
    end
end

@testset "Dkg leaves isolated layers bit-exact" begin
    for FT in (Float32,Float64)
        check_dkg_isolated_layers(FT)
    end
end

@testset "Dkg validates all workspace panels before mutation" begin
    m, d, rm = dkg_mass_fixture(Float64, 2, 3, 7, 1)
    op = DkgDiff.ImplicitVerticalDiffusion(; kz_field=DkgState.PrecomputedCSDkgField(d))
    good = DkgDiff.DiffusionWorkspace(m, 1, 7)
    factors = ntuple(p -> p == 6 ? zeros(1,1,1) : good.factors[p], 6)
    bad = DkgDiff.DiffusionWorkspace(factors, good.layer_thickness, good.references)
    original = map(copy, rm)
    @test_throws DimensionMismatch DkgDiff.apply_vertical_diffusion_vmr!(rm,m,op,bad,1;halo_width=1)
    @test rm == original
    @test_throws ArgumentError DkgDiff.apply_vertical_diffusion_vmr!(rm,m,op,good,1;halo_width=-1)
    @test rm == original
end

@testset "Dkg zero-carrier convention matches the VMR solve" begin
    m, d, rm = dkg_mass_fixture(Float64, 2, 4, 2, 0.7)
    for p in 1:6
        m[p][2,2,2] = 0
    end
    op = DkgDiff.ImplicitVerticalDiffusion(; kz_field=DkgState.PrecomputedCSDkgField(d))
    ws = DkgDiff.DiffusionWorkspace(m,1,2)
    expected = map(copy, rm)
    DkgDiff._cs_scale_tracer_mass_to_vmr!(expected,m,1)
    DkgDiff.apply_vertical_diffusion!(expected,m,op,ws,1;halo_width=1)
    DkgDiff._cs_scale_vmr_to_tracer_mass!(expected,m,1)
    DkgDiff.apply_vertical_diffusion_vmr!(rm,m,op,ws,1;halo_width=1)
    @test all(p -> isapprox(rm[p],expected[p];rtol=3e-15),1:6)
end

@testset "Dkg closed interfaces do not pass rounding residuals" begin
    for FT in (Float32, Float64)
        m, d, rm = dkg_mass_fixture(FT,2,4,7,40)
        for p in 1:6
            d[p][:,:,2] .= 0
            rm[p][2:3,2:3,3:4,:] .= 0
        end
        op=DkgDiff.ImplicitVerticalDiffusion(;kz_field=DkgState.PrecomputedCSDkgField(d))
        ws=DkgDiff.DiffusionWorkspace(m,1,7)
        for _ in 1:20
            DkgDiff.apply_vertical_diffusion_vmr!(rm,m,op,ws,one(FT);halo_width=1)
        end
        @test all(p -> all(iszero,rm[p][2:3,2:3,3:4,:]),1:6)
    end
end
