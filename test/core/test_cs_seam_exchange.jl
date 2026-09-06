using Test, LinearAlgebra
include(joinpath(@__DIR__, "..", "helpers", "cs_seam_exchange.jl"))

@testset "CS seam exchange agrees with independent scalar loops" begin
    for FT in (Float32, Float64), Nt in (1, 7), convention in
            (CSSeamGrids.GnomonicPanelConvention(), CSSeamGrids.GEOSNativePanelConvention()),
            scheme in (UpwindScheme(), SlopesScheme(MonotoneLimiter()), PPMScheme())
        mesh, m, packed, am, bm = cs_seam_fixture(FT, 5, 3, Nt, convention)
        ws = CSSeamAdv.CSAdvectionWorkspace(mesh, m[1]; n_tracers=Nt)
        original = map(copy, packed)
        for direction in (1, 2)
            reference = [reference_cs_group(map(a -> a[:, :, :, t], original),
                m, am, bm, mesh, scheme, direction, FT(1)) for t in 1:Nt]
            actual, air = map(copy, original), map(copy, m)
            CSSeamAdv._sweep_cs_horizontal!(actual, air, direction == 1 ? am : bm,
                mesh, scheme, ws, Val(direction))
            tol = FT == Float32 ? 5e-6 : 2e-14
            @test cs_seam_maxdiff(air, reference[1][2], mesh) < tol
            for t in 1:Nt
                @test cs_seam_maxdiff(map(a -> a[:, :, :, t], actual), reference[t][1], mesh) < tol
            end
        end
    end
end

@testset "CS seam adjoint differentiates the coupled production sweep" begin
    Adj = AtmosTransport.Adjoints
    for convention in (CSSeamGrids.GnomonicPanelConvention(), CSSeamGrids.GEOSNativePanelConvention()),
            scheme in (UpwindScheme(), SlopesScheme(CSSeamAdv.NoLimiter()),
                       PPMScheme(CSSeamAdv.NoLimiter()), PPMScheme()), axis in (1, 2)
        mesh, m, packed, am, bm = cs_seam_fixture(Float64, 5, 3, 2, convention)
        rm = map(a -> a[:, :, :, 2], packed)
        rng = MersenneTwister(2917)
        direction = map(a -> randn(rng, size(a)), rm)
        seed = map(a -> zero(a), rm)
        for p in 1:6
            cs_seam_interior(seed[p], mesh) .= randn(rng, mesh.Nc, mesh.Nc, 3)
            # Only physical cells are independent inputs; halo values are copied.
            interior = copy(cs_seam_interior(direction[p], mesh))
            fill!(direction[p], 0)
            cs_seam_interior(direction[p], mesh) .= interior
        end
        flux = axis == 1 ? am : bm
        symbol = axis == 1 ? :x : :y
        ws = CSSeamAdv.CSAdvectionWorkspace(mesh, m[1])
        aws = Adj.CSAdjointWorkspace(mesh, m[1])
        function objective(h)
            r = map((a, d) -> a .+ h .* d, rm, direction)
            mass = map(copy, m)
            CSSeamAdv.fill_panel_halos!(r, mesh)
            CSSeamAdv._sweep_cs_horizontal!(r, mass, flux, mesh, scheme, ws, Val(axis))
            return sum(dot(seed[p], r[p]) for p in 1:6)
        end
        gradient = map(copy, seed)
        if scheme isa PPMScheme{MonotoneLimiter}
            Adj._adjoint_scheme_sweep!(gradient, m, rm, flux, symbol, scheme, mesh, aws, 1.0)
        else
            Adj._adjoint_scheme_sweep!(gradient, m, flux, symbol, scheme, mesh, aws, 1.0)
        end
        Adj._adjoint_fill_panel_halos!(gradient, mesh)
        expected = sum(dot(gradient[p], direction[p]) for p in 1:6)
        h = 1e-5
        finite_difference = (objective(h) - objective(-h)) / (2h)
        @test expected ≈ finite_difference rtol=2e-7 atol=2e-8
    end
end

@testset "CS seam cache follows its scheme and rejects incompatible geometry" begin
    mesh, m, packed, am, bm = cs_seam_fixture(Float64, 5, 3, 7, CSSeamGrids.GEOSNativePanelConvention())
    lr = CSSeamAdv.CSLinRoodAdvectionWorkspace(mesh, m[1]; n_tracers=7)
    @test isempty(lr.cs.seam_flux)
    original = map(copy, packed)
    @test_throws DimensionMismatch CSSeamAdv._sweep_cs_horizontal!(
        packed, m, am, mesh, PPMScheme(), lr.cs, Val(1))
    narrow = CSSeamAdv.CSAdvectionWorkspace(mesh, m[1])
    @test_throws DimensionMismatch CSSeamAdv._sweep_cs_horizontal!(
        packed, m, am, mesh, PPMScheme(), narrow, Val(1))
    @test packed == original
end
