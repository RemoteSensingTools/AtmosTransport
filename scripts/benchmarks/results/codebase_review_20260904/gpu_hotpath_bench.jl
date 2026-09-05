using CUDA, Adapt, Test, Statistics, JSON3, AtmosTransport, AtmosTransportBenchmarks
using AtmosTransport.Output
const B=AtmosTransportBenchmarks
const A=AtmosTransport.Operators.Advection
CUDA.allowscalar(false)
@assert occursin("A100",CUDA.name(CUDA.device()))
Base.include(A,joinpath(@__DIR__,"linrood_reference.jl"))
records=[]
for FT in (Float32,Float64), Nc in (12,90), order in (5,7)
    payload=B._build_cs_payload(FT,Nc,32,1,:cuda)
    mesh=payload.mesh
    old_m=map(copy,payload.panels_m);old_rm=map(copy,payload.panels_rm)
    new_m=map(copy,payload.panels_m);new_rm=map(copy,payload.panels_rm)
    old_ws=A.CSLinRoodAdvectionWorkspace(mesh,old_m[1]);new_ws=A.CSLinRoodAdvectionWorkspace(mesh,new_m[1])
    old!()=A._review_reference_fv_tp_2d_cs!(old_rm,old_m,payload.panels_am,payload.panels_bm,mesh,Val(order),old_ws.cs,old_ws.linrood)
    new!()=A.fv_tp_2d_cs!(new_rm,new_m,payload.panels_am,payload.panels_bm,mesh,Val(order),new_ws.cs,new_ws.linrood)
    old!();new!();CUDA.synchronize()
    @test all(p->Array(old_m[p])==Array(new_m[p]),1:6)
    @test all(p->Array(old_rm[p])==Array(new_rm[p]),1:6)
    old_times=Float64[];new_times=Float64[]
    for _ in 1:20
        for (f,times) in ((old!,old_times),(new!,new_times))
            CUDA.synchronize();t=time_ns();f();CUDA.synchronize();push!(times,(time_ns()-t)/1e9)
        end
    end
    old=median(old_times);new=median(new_times)
    record=(;FT=string(FT),Nc,Nz=32,order,old_seconds=old,new_seconds=new,speedup=old/new,bitwise_equal=true,device=CUDA.name(CUDA.device()))
    push!(records,record);println(JSON3.write(record))
end
open(joinpath(@__DIR__,"gpu_hotpath_results.json"),"w") do io;JSON3.pretty(io,records);end
