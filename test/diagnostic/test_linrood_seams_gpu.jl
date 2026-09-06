# Opt-in check on an explicitly selected CUDA device.
# ATMOSTR_RUN_LR_SEAMS_GPU_TESTS=1 ATMOSTR_LR_GPU_NAME=V100 \
# CUDA_VISIBLE_DEVICES=<authorized UUID> julia --project=. test/diagnostic/test_linrood_seams_gpu.jl
using Test
if get(ENV, "ATMOSTR_RUN_LR_SEAMS_GPU_TESTS", "0") != "1"
    @info "Skipping opt-in Lin-Rood seam GPU tests"
else
    using CUDA, AtmosTransport, Random
    expected_device = get(ENV, "ATMOSTR_LR_GPU_NAME", "A100")
    isempty(expected_device) && error("ATMOSTR_LR_GPU_NAME must identify the selected device")
    @assert occursin(expected_device, CUDA.name(CUDA.device())) "Wrong device for seam tests"
    CUDA.allowscalar(false)
    include(joinpath(@__DIR__, "..", "core", "test_linrood_seams.jl"))

    @testset "CUDA Lin-Rood shared faces match the independent contact map" begin
        for FT in (Float32, Float64), Nc in (5, 35), convention in
                (SeamGrids.GnomonicPanelConvention(), SeamGrids.GEOSNativePanelConvention())
            mesh = CubedSphereMesh(; FT, Nc, Hp=3, convention)
            original = seam_face_fixture(FT, Nc, 7, MersenneTwister(4917))
            device_faces = map(panels -> map(CuArray, panels), original)
            SeamAdvection._share_lr_seam_faces!(device_faces..., mesh)
            actual = map(panels -> map(Array, panels), device_faces)
            @test actual == independent_seam_mean(original, mesh)
            SeamAdvection._share_lr_seam_faces!(device_faces..., mesh)
            @test map(panels -> map(Array, panels), device_faces) == actual
        end
    end
end
