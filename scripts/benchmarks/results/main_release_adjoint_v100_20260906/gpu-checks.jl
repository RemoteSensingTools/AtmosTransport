using CUDA, Test, TOML
CUDA.functional(true) || error("CUDA is unavailable")
length(collect(CUDA.devices())) == 1 || error("Expose only the authorized V100")
occursin("V100", CUDA.name(CUDA.device())) || error("Expected V100")
CUDA.allowscalar(false)
using AtmosTransport
println("SOURCE_COMMIT=1a89772e970d7df04394832669b32c28f9f0c283")
println("JULIA=", VERSION, " CUDA=", pkgversion(CUDA), " RUNTIME=", CUDA.runtime_version())
println("DEVICE=", CUDA.name(CUDA.device()))
println("PACKAGE=", pathof(AtmosTransport))
root = @__DIR__
@testset "Release adjoint checks on V100" begin
    for file in (
        "core/test_cs_ppm_adjoint_footprint.jl",
        "core/test_cmfmc_adjoint_identity.jl",
        "core/test_adjoint_identity_model_space.jl",
        "core/test_adjoint_identity_preconditioned.jl",
        "core/test_cs_stride_checkpoint.jl",
        "core/test_cs_tape_mmap_roundtrip.jl",
        "core/test_linrood_kernel_adjoints.jl",
        "diagnostic/test_linrood_adjoint_integration.jl",
    )
        println("RUNNING ", file)
        flush(stdout)
        mod = Module(gensym(:ReleaseTest))
        Core.eval(mod, :(include(path::AbstractString) = Base.include($mod, path)))
        Base.include(mod, joinpath(root, "test", file))
        if endswith(file, "test_cs_ppm_adjoint_footprint.jl")
            @test getfield(mod, :HAS_GPU)
        end
        CUDA.synchronize()
        println("COMPLETED ", file)
        flush(stdout)
    end
end
println("RELEASE_ADJOINT_CHECKS_PASSED")
