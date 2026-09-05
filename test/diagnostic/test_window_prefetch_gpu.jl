# CUDA_VISIBLE_DEVICES=0 ATMOSTR_RUN_PREFETCH_GPU_TESTS=1 \
# ATMOSTR_PREFETCH_GPU_NAME=V100 julia --threads=4 --project=. test/diagnostic/test_window_prefetch_gpu.jl
using Test
if get(ENV,"ATMOSTR_RUN_PREFETCH_GPU_TESTS","0") != "1"
    @info "Skipping opt-in window prefetch GPU tests"
else
    using CUDA
    expected_device = get(ENV,"ATMOSTR_PREFETCH_GPU_NAME","A100")
    isempty(expected_device) && error("ATMOSTR_PREFETCH_GPU_NAME must name the authorized GPU")
    @assert occursin(expected_device,CUDA.name(CUDA.device())) "Wrong device for prefetch GPU tests"
    @assert Threads.nthreads() > 1 "Prefetch requires multiple Julia threads"
    CUDA.allowscalar(false)
    include(joinpath(@__DIR__,"..","fixtures","window_prefetch.jl"))
    using .WindowPrefetchFixtures
    @testset "GPU startup reads once and owns independent prefetch storage" begin
        check_prefetch_startup(CuArray)
        check_prefetch_startup(CuArray;device_windows=true)
        check_prefetch_startup(CuArray;enabled=false)
        check_prefetch_startup(CuArray;stop_window=1)
    end
end
