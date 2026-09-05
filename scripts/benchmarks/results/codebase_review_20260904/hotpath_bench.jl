using AtmosTransport, Statistics, JSON3
using AtmosTransport.Output
function old_column(air,rm)
    Nx,Ny,Nz=size(air);out=zeros(Float64,Nx,Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        num=0.0;den=0.0
        for k in 1:Nz
            num+=Float64(rm[i,j,k]);den+=Float64(air[i,j,k])
        end
        out[i,j]=den>0 ? num/den : NaN
    end
    out
end
records=[]
for FT in (Float32,Float64), Nc in (90,180)
    Nz=72; air=reshape(FT.(range(1,2;length=Nc*Nc*Nz)),Nc,Nc,Nz);rm=air .* FT(400e-6)
    @assert old_column(air,rm)==column_mean_mixing_ratio(air,rm)
    old_column(air,rm);column_mean_mixing_ratio(air,rm)
    old=median([@elapsed old_column(air,rm) for _ in 1:20])
    new=median([@elapsed column_mean_mixing_ratio(air,rm) for _ in 1:20])
    mesh=CubedSphereMesh(;Nc,Hp=0,FT); grid=AtmosGrid(mesh,HybridSigmaPressure(zeros(FT,Nz+1),FT.(range(0,1;length=Nz+1))),CPU();FT)
    state=CubedSphereState(DryBasis,ntuple(_->copy(air),6);halo_width=0,co2=ntuple(_->copy(rm),6),tr2=ntuple(_->copy(rm),6),tr3=ntuple(_->copy(rm),6),tr4=ntuple(_->copy(rm),6))
    model=(;state,grid)
    fields=output_field_spec(Dict("layers"=>"none","air_mass_layers"=>"none"))
    full=capture_snapshot(model);compact=capture_snapshot(model;fields)
    record=(;FT=string(FT),Nc,Nz,tracers=4,old_seconds=old,new_seconds=new,speedup=old/new,full_host_bytes=Base.summarysize(full),column_host_bytes=Base.summarysize(compact))
    push!(records,record);println(JSON3.write(record))
end
open(ARGS[1],"w") do io; JSON3.pretty(io,records); end
