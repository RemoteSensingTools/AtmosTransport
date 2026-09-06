# Set CUDA_VISIBLE_DEVICES explicitly and opt in before running.
using Test
if get(ENV, "ATMOSTR_RUN_DKG_GPU_TESTS", "0") != "1"
    @info "Skipping opt-in conservative Dkg GPU tests"
else
    using CUDA
    CUDA.allowscalar(false)
    expected = get(ENV,"ATMOSTR_DKG_GPU_NAME","A100")
    @assert !isempty(expected) && occursin(expected,CUDA.name(CUDA.device()))
    include(joinpath(@__DIR__,"..","helpers","conservative_dkg.jl"))
    @testset "CUDA conservative Dkg through 65 tracers" begin
        for FT in (Float32,Float64), (Nc,Nz) in ((3,66),(35,3)), Nt in (1,7,32,65), strength in (0,40)
            check_conservative_dkg(FT,Nc,Nz,Nt,strength;array_type=CuArray)
        end
        for FT in (Float32,Float64)
            check_dkg_isolated_layers(FT;array_type=CuArray)
            check_dkg_weak_exchange(FT;array_type=CuArray)
        end
    end
end
