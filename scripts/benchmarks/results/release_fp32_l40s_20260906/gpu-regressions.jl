using CUDA, AtmosTransport, Test, TOML
CUDA.functional(true) || error("CUDA unavailable")
@assert length(collect(CUDA.devices())) == 1
@assert occursin("L40S",CUDA.name(CUDA.device()))
CUDA.allowscalar(false)
CUDA.versioninfo()
const root=joinpath(@__DIR__,"current")
const cases=[
("test_cs_ppm_launch_gpu.jl", "CS_PPM_LAUNCH", "PPM"),
("test_conservative_dkg_gpu.jl", "DKG", "DKG"),
("test_tm5_tracer_batching_gpu.jl", "MATRIX_BATCH", "MATRIX"),
("test_cs_seam_exchange_gpu.jl", "CS_SEAMS", "CS_SEAMS"),
("test_linrood_seams_gpu.jl", "LR_SEAMS", "LR"),
("test_snapshot_totals_gpu.jl", "SNAPSHOT", "SNAPSHOT"),
("test_cs_transport_adjoint_gpu.jl", "TRANSPORT_ADJOINT", "ADJOINT"),
("test_window_prefetch_gpu.jl", "PREFETCH", "PREFETCH"),
("test_window_prefetch_failure_gpu.jl", "PREFETCH", "PREFETCH"),
("test_cs_multifile_gpu.jl", "CS_MULTIFILE", "MULTIFILE")]
for (index, (file, flag, device)) in enumerate(cases)
    index < parse(Int,get(ENV,"ATMOSTR_GPU_START_CASE","1")) && continue
    ENV["ATMOSTR_RUN_"*flag*"_GPU_TESTS"]="1"
    ENV["ATMOSTR_"*device*"_GPU_NAME"]="L40S"
    println("RUNNING ",file);flush(stdout);flush(stderr)
    mod=Module(gensym(:L40Check))
    Core.eval(mod, :(include(path::AbstractString)=Base.include($mod,path)))
    Base.include(mod,joinpath(root,"test","diagnostic",file))
    CUDA.synchronize()
    GC.gc(true);CUDA.reclaim()
    println("PASSED ",file);flush(stdout);flush(stderr)
end
println("L40_GPU_REGRESSIONS_PASSED")
