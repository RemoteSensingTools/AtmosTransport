#!/usr/bin/env julia
# Synthetic, warm-cache binary reader → driven transport → NetCDF benchmark.
# julia --project=benchmarking benchmarking/run_pipeline_benchmarks.jl cpu output.json
# CUDA_VISIBLE_DEVICES=0 julia --project=benchmarking benchmarking/run_pipeline_benchmarks.jl cuda output.json
using AtmosTransport, JSON3, Statistics, Logging
using AtmosTransport.Grids: ncells, nfaces
const MD = AtmosTransport.MetDrivers
const backend = isempty(ARGS) ? "cpu" : ARGS[1]
backend in ("cpu","cuda") || error("backend must be cpu or cuda")
if backend == "cuda"
    @eval using CUDA
    @assert occursin("A100",CUDA.name(CUDA.device())) "Use the A100 for CUDA benchmarks"
    CUDA.allowscalar(false)
end
const destination = length(ARGS)>1 ? ARGS[2] : "pipeline_results.json"
const Nc = parse(Int,get(ENV,"ATMOSTR_BENCH_NC","12"))
const Nz = parse(Int,get(ENV,"ATMOSTR_BENCH_NZ","16"))
const repeats = parse(Int,get(ENV,"ATMOSTR_BENCH_REPEATS","3"))

function fixture(path, topology, ::Type{FT}) where FT
    vc = HybridSigmaPressure(zeros(FT,Nz+1),FT.(range(0,1;length=Nz+1)))
    mesh = topology === :cs ? CubedSphereMesh(;Nc,Hp=3,FT) :
           topology === :ll ? LatLonMesh(;Nx=2Nc,Ny=Nc,FT) :
           ReducedGaussianMesh(FT.(range(-75,75;length=Nc)),fill(2Nc,Nc);FT)
    grid = AtmosGrid(mesh,vc,CPU();FT)
    window = if topology === :cs
        (;m=ntuple(_->ones(FT,Nc,Nc,Nz),6),am=ntuple(_->zeros(FT,Nc+1,Nc,Nz),6),
          bm=ntuple(_->zeros(FT,Nc,Nc+1,Nz),6),cm=ntuple(_->zeros(FT,Nc,Nc,Nz+1),6),
          ps=ntuple(_->fill(FT(100000),Nc,Nc),6))
    elseif topology === :ll
        (;m=ones(FT,2Nc,Nc,Nz),am=fill(FT(0.001),2Nc+1,Nc,Nz),
          bm=zeros(FT,2Nc,Nc+1,Nz),cm=zeros(FT,2Nc,Nc,Nz+1),ps=fill(FT(100000),2Nc,Nc))
    else
        (;m=ones(FT,ncells(mesh),Nz),hflux=zeros(FT,nfaces(mesh),Nz),
          cm=zeros(FT,ncells(mesh),Nz+1),ps=fill(FT(100000),ncells(mesh)))
    end
    if topology === :cs
        writer = MD.open_streaming_cs_transport_binary(path,Nc,6,Nz,2,vc;
            FT,dt_met_seconds=3600.0,steps_per_window=2,mass_basis=:dry)
        try
            for _ in 1:2
                MD.write_streaming_cs_window!(writer,window,Nc,6)
            end
        finally
            MD.close_streaming_transport_binary!(writer)
        end
    else
        write_transport_binary(path,grid,[window,window];FT,dt_met_seconds=3600.0,
            half_dt_seconds=900.0,steps_per_window=2,mass_basis=:dry,
            source_flux_sampling=:window_start_endpoint,flux_sampling=:window_constant,
            extra_header=Dict("poisson_balance_target_scale"=>0.25,
                "poisson_balance_target_semantics"=>"forward_window_mass_difference / (2 * steps_per_window)"))
    end
    return grid
end

records = Any[]
mktempdir() do dir
    for topology in (:ll,:rg,:cs)
        binary = joinpath(dir,"$(topology).bin")
        grid = fixture(binary,topology,Float32)
        for nt in (1,4), layers in ("full","none")
            output = joinpath(dir,"$(topology)_$(nt)_$(layers).nc")
            cfg = Dict{String,Any}(
                "input"=>Dict("binary_paths"=>[binary]),
                "architecture"=>Dict("use_gpu"=>backend=="cuda"),
                "numerics"=>Dict("float_type"=>"Float32"),
                "advection"=>Dict("scheme"=>"upwind"),
                "tracers"=>Dict("tr$t"=>Dict("init"=>Dict("kind"=>"uniform","background"=>400e-6)) for t in 1:nt),
                "output"=>Dict("path"=>output,"hours"=>[0.0,1.0,2.0],
                    "fields"=>Dict("layers"=>layers,"air_mass_layers"=>layers,
                                   "column_mean"=>true)))
            samples = []
            for r in 0:repeats
                result = with_logger(NullLogger()) do
                    redirect_stdout(devnull) do
                        redirect_stderr(devnull) do
                            @timed AtmosTransport.Models.run_driven_simulation(cfg)
                        end
                    end
                end
                r>0 && push!(samples,result)
            end
            record = Dict("topology"=>String(topology),"backend"=>backend,"Nc"=>Nc,
                "levels"=>Nz,"tracers"=>nt,"layers"=>layers,"windows"=>2,
                "median_seconds"=>median(x.time for x in samples),
                "host_allocated_bytes"=>median(x.bytes for x in samples),
                "input_bytes"=>filesize(binary),"output_bytes"=>filesize(output),
                "cache"=>"warm OS page cache; includes setup, capture and NetCDF write",
                "device"=>backend=="cuda" ? CUDA.name(CUDA.device()) : Sys.cpu_info()[1].model)
            push!(records,record)
            println(JSON3.write(record))
        end
    end
end
open(destination,"w") do io
    JSON3.pretty(io,records)
end
