# Run with CUDA_VISIBLE_DEVICES=0 julia --project=benchmarking this_file.jl.
using Test, CUDA, Adapt, AtmosTransport
using AtmosTransportBenchmarks
using AtmosTransport.Output
const B = AtmosTransportBenchmarks
CUDA.allowscalar(false)
@assert occursin("A100", CUDA.name(CUDA.device())) "This regression must run on the A100"

@testset "A100 transport and selected snapshot parity" begin
    for FT in (Float32,Float64), scheme in (:linrood5,:linrood7,:upwind,:ppm)
        cpu_case = B.BenchmarkCase(:cpu,FT,12,8,4,:advection,scheme,3,1,3,"review")
        gpu_case = B.BenchmarkCase(:cuda,FT,12,8,4,:advection,scheme,3,1,3,"review")
        cpu, gpu = B._build_model(cpu_case), B._build_model(gpu_case)
        for _ in 1:3
            step!(cpu, FT(600))
            step!(gpu, FT(600))
        end
        CUDA.synchronize()
        tol = FT === Float32 ? 3e-5 : 2e-12
        for p in 1:6
            @test Array(gpu.state.air_mass[p]) ≈ cpu.state.air_mass[p] rtol=tol
            @test Array(gpu.state.tracers_raw[p]) ≈ cpu.state.tracers_raw[p] rtol=tol
        end
        for mode in ("none","selected")
            fields = output_field_spec(Dict{String,Any}("tracers"=>["tr1","tr3"],
                "layers"=>mode,"levels"=>[1,8],"air_mass_layers"=>mode,
                "column_mean"=>true,"column_mass_per_area"=>true,
                "column_air_mass_per_area"=>true))
            frame = capture_snapshot(gpu;halo_width=3,fields)
            full = capture_snapshot(gpu;halo_width=3)
            @test Set(keys(frame.tracers)) == Set([:tr1,:tr3])
            for name in (:tr1,:tr3), p in 1:6
                @test AtmosTransport.Output._frame_column_mean(frame,name)[p] ≈
                      column_mean_mixing_ratio(full.air_mass,full.tracers[name])[p] rtol=1e-12
            end
            @test Base.summarysize(frame) < Base.summarysize(full)
        end
    end
end

@testset "A100 diffusion workspace ownership and parity" begin
    for FT in (Float32,Float64), operator in (:diffusion,:full)
        cpu = B._build_model(B.BenchmarkCase(:cpu,FT,4,8,2,operator,:upwind,1,1,1,"review"))
        gpu = B._build_model(B.BenchmarkCase(:cuda,FT,4,8,2,operator,:upwind,1,1,1,"review"))
        if operator === :diffusion
            @test gpu.workspace.advection_ws === nothing
        else
            @test gpu.workspace.diffusion_ws.w_scratch === gpu.workspace.advection_ws.w_scratch
        end
        @test all(a -> a isa CuArray, gpu.workspace.diffusion_ws.w_scratch)
        step!(cpu,FT(600)); step!(gpu,FT(600)); CUDA.synchronize()
        for p in 1:6
            @test Array(gpu.state.tracers_raw[p]) ≈ cpu.state.tracers_raw[p] rtol=(FT===Float32 ? 3e-5 : 2e-12)
        end
    end
end

@testset "A100 LL/RG selected capture" begin
    for FT in (Float32,Float64), topology in (:ll,:rg)
        mesh = topology === :ll ? LatLonMesh(;Nx=6,Ny=4,FT) :
               ReducedGaussianMesh(FT[-45,45],[4,8];FT)
        shape = topology === :ll ? (6,4,8) : (12,8)
        air = reshape(FT.(1:prod(shape)),shape)
        state = CellState(DryBasis,air;co2=air .* FT(400e-6))
        vc = HybridSigmaPressure(zeros(FT,9),FT.(range(0,1;length=9)))
        model = (;state=Adapt.adapt(CuArray,state),grid=AtmosGrid(mesh,vc,CPU();FT))
        full = capture_snapshot(model)
        fields = output_field_spec(Dict("layers"=>"selected","levels"=>[2,7],
            "air_mass_layers"=>"none","column_mean"=>true))
        frame = capture_snapshot(model;fields)
        @test AtmosTransport.Output._frame_column_mean(frame,:co2) ≈
              column_mean_mixing_ratio(full.air_mass,full.tracers[:co2]) rtol=1e-12
        @test AtmosTransport.Output._frame_vmr(frame,:co2,[2,7]) ==
              AtmosTransport.Output._frame_vmr(full,:co2,[2,7])
    end
end
