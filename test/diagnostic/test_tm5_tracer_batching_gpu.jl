# Explicit opt-in; normal test discovery must not initialize a GPU.
# On the selected GPU (A100 by default):
# ATMOSTR_RUN_MATRIX_BATCH_GPU_TESTS=1 julia --project=. test/diagnostic/test_tm5_tracer_batching_gpu.jl
# For the authorized V100 run, also set ATMOSTR_MATRIX_GPU_NAME=V100.
using Test, Random, Adapt
using AtmosTransport
import KernelAbstractions as KA
const C = AtmosTransport.Operators.Convection

function batch_fixture(nz, depth, nt, downdrafts; signed=false)
    rng = MersenneTwister(9042026)
    nc = 4
    m = 50f0 .+ 150f0 .* rand(rng, Float32, nc, nz)
    q = (rand(rng, Float32, nc, nz, nt) .- (signed ? 0.5f0 : 0f0)) .* m
    entu, detu, entd, detd = (zeros(Float32, nc, nz) for _ in 1:4)
    shift = nz-depth
    # Column 1 is cloud-free. The others have shallow/deep closed clouds.
    for c in 2:nc
        top = shift + (c == 2 ? 1 : c)
        entu[c, nz-1] = 0.04f0
        detu[c, top] = 0.04f0
        if downdrafts && c != 2
            # Mix column structures within one launch, including a tiny
            # positive downdraft that must still use the general LU bound.
            flux = c == 4 ? eps(Float32)^2 : 0.01f0
            entd[c, top+1] = flux
            detd[c, nz] = flux
        end
    end
    (; m, q, rates=(entu, detu, entd, detd), nz, depth, nt, signed)
end

function batch_reference(f)
    result = copy(f.q)
    for c in axes(result, 1)
        rates = map(r -> view(r,c,:), f.rates)
        top, _, lfs = C._tm5_diagnose_cloud_dims(rates[2],rates[3],f.nz)
        top > f.nz && continue
        lo = min(lfs,max(top,2)-1)
        A = zeros(Float32,f.nz,f.nz)
        pivots = zeros(Int,f.nz)
        C._tm5_build_conv1!(A, rates..., view(f.m,c,:), top, lfs, 1800f0, f.nz;
            f=A, amu=zeros(Float32,f.nz+1), amd=zeros(Float32,f.nz+1))
        # Explicitly dense reference, independent of automatic structure selection.
        C._tm5_lu!(A, pivots, f.nz; icltop_eff=lo)
        C._tm5_solve!(view(result,c,:,:), A, pivots, f.nz, f.nt; icltop_eff=lo)
    end
    result
end

function batch_device_arrays(f, topology)
    if topology == :rg
        return CUDA.CuArray(f.q), CUDA.CuArray(f.m),
               map(CUDA.CuArray, f.rates), CUDA.ones(Float32, 4)
    end
    rates = map(r -> CUDA.CuArray(reshape(r, 2, 2, f.nz)), f.rates)
    if topology == :ll
        return CUDA.CuArray(reshape(f.q, 2, 2, f.nz, f.nt)),
               CUDA.CuArray(reshape(f.m, 2, 2, f.nz)), rates, CUDA.ones(Float32, 2)
    end
    q = fill(-99f0, 4, 4, f.nz, f.nt)
    m = fill(-99f0, 4, 4, f.nz)
    q[2:3, 2:3, :, :] .= reshape(f.q, 2, 2, f.nz, f.nt)
    m[2:3, 2:3, :] .= reshape(f.m, 2, 2, f.nz)
    CUDA.CuArray(q), CUDA.CuArray(m), rates, CUDA.ones(Float32, 2, 2)
end

function launch_batch!(q, m, rates, areas, f, topology)
    backend = KA.get_backend(q)
    nt = size(q, ndims(q))
    wg = C._TM5_COLLAB_WG_SIZE
    if topology == :rg
        C._tm5_faceindexed_column_collab_kernel!(backend, wg)(
            q, m, rates..., areas, f.nz, nt, 1800f0, Val(f.depth); ndrange=wg*4)
    elseif topology == :ll
        C._tm5_column_collab_kernel!(backend, wg)(
            q, m, rates..., areas, 2, f.nz, nt, 1800f0, Val(f.depth); ndrange=wg*4)
    else
        C._tm5_cs_panel_column_collab_kernel!(backend, wg)(
            q, m, rates..., areas, 1, 2, f.nz, nt, 1800f0, Val(f.depth); ndrange=wg*4)
    end
    KA.synchronize(backend)
end

function check_batches(f, topology)
    q, m, rates, areas = batch_device_arrays(f, topology)
    initial = Array(q)
    launch_batch!(q, m, rates, areas, f, topology)
    result = Array(q)
    interior = topology == :cs ? result[2:3, 2:3, :, :] : result
    columns = reshape(interior, 4, f.nz, f.nt)
    reference = batch_reference(f)
    @test isapprox(columns, reference; rtol=100eps(Float32))
    @test f.signed ? minimum(columns) < 0 < maximum(columns) : minimum(columns) >= 0
    @test columns[1, :, :] == f.q[1, :, :] # No-convection column, all batches.
    @test columns[:, 1:(f.nz-f.depth), :] == f.q[:, 1:(f.nz-f.depth), :]
    before = sum(Float64.(f.q); dims=2)
    after = sum(Float64.(columns); dims=2)
    scale = sum(abs.(Float64.(f.q)); dims=2)
    @test maximum(abs.(after-before) ./ scale) < 100eps(Float32)
    if topology == :cs
        @test result[[1,4], :, :, :] == initial[[1,4], :, :, :]
        @test result[:, [1,4], :, :] == initial[:, [1,4], :, :]
    end
    # Independent <=6-tracer launches must reproduce each slice exactly.
    # This catches off-by-one indexing and stale shared slots in tail batches.
    split = similar(initial)
    leading = ntuple(_ -> Colon(), ndims(initial)-1)
    for first in 1:6:f.nt
        selected = first:min(first+5, f.nt)
        part = CUDA.CuArray(initial[leading..., selected])
        launch_batch!(part, m, rates, areas, f, topology)
        split[leading..., selected] .= Array(part)
    end
    @test reinterpret(UInt32, vec(result)) == reinterpret(UInt32, vec(split))
end

if get(ENV, "ATMOSTR_RUN_MATRIX_BATCH_GPU_TESTS", "0") == "1"
    using CUDA
    expected_device = get(ENV, "ATMOSTR_MATRIX_GPU_NAME", "A100")
    isempty(expected_device) && error("ATMOSTR_MATRIX_GPU_NAME must name the selected GPU")
    CUDA.functional() || error("This opt-in test requires a functional CUDA GPU")
    occursin(expected_device, CUDA.name(CUDA.device())) ||
        error("Select the requested $expected_device before running this test")
    CUDA.allowscalar(false)
    @testset "Deferred scratch adaptation on $expected_device" begin
        cpu_ws = C.TM5Workspace(zeros(Float32,2,2,8); tile_columns=4,
                                cell_metrics=ones(Float32,2), defer_scratch=true)
        # Simulate a CPU fallback before adapting the model to the GPU.
        C._ensure_tm5_scratch!(cpu_ws)
        gpu_ws = Adapt.adapt(CUDA.CuArray, cpu_ws)
        @test isempty(gpu_ws.conv1)
        @test gpu_ws.scratch_columns == 4
        @test gpu_ws.cell_metrics isa CUDA.CuArray
        @test gpu_ws.f_scratch === gpu_ws.conv1
        C._ensure_tm5_scratch!(gpu_ws)
        @test size(gpu_ws.conv1) == (8,8,4)
        @test gpu_ws.f_scratch === gpu_ws.conv1
        @test isempty(Adapt.adapt(Array, gpu_ws).conv1)
    end
    @testset "Collaborative tracer batching on $expected_device" begin
        for topology in (:ll, :rg, :cs), (nz, depth) in ((8,8), (91,85)),
            nt in (1,6,7,12,32,65), downdrafts in (false,true)
            @testset "$topology Nz=$nz Nt=$nt downdrafts=$downdrafts" begin
                check_batches(batch_fixture(nz, depth, nt, downdrafts), topology)
            end
        end
    end
    @testset "Signed tracer batching on $expected_device" begin
        for topology in (:ll, :rg, :cs), (nz, depth) in ((8,8), (91,85)),
            nt in (7,32,65), downdrafts in (false,true)
            @testset "$topology Nz=$nz Nt=$nt downdrafts=$downdrafts" begin
                check_batches(batch_fixture(nz, depth, nt, downdrafts; signed=true), topology)
            end
        end
    end
else
    @testset "Collaborative tracer batching on CUDA (opt-in)" begin
        @test_skip "Set ATMOSTR_RUN_MATRIX_BATCH_GPU_TESTS=1 and select the requested GPU"
    end
end
