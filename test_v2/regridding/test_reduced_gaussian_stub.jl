# ---------------------------------------------------------------------------
# test_reduced_gaussian_stub.jl
#
# Tier 1 ships with ReducedGaussianMesh → anything regridding stubbed out.
# Verify the stub errors with the expected message so that a future
# implementation can flip this test to the positive case.
# ---------------------------------------------------------------------------

@testset "ReducedGaussianMesh stub" begin
    rg = ReducedGaussianMesh([-60.0, -30.0, 0.0, 30.0, 60.0], [8, 12, 16, 12, 8])
    dst = CubedSphereMesh(Nc = 4, convention = GnomonicPanelConvention())

    err = try
        build_regridder(rg, dst; normalize = false)
        nothing
    catch e
        e
    end
    @test err !== nothing
    @test occursin("not yet implemented", sprint(showerror, err))
end
