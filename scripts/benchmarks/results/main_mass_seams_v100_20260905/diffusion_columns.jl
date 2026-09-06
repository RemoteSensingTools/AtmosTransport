using AtmosTransport, LinearAlgebra, TOML
const path="/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/era5_n320_transport_20181201_float32.bin"
driver=TransportBinaryDriver(path;FT=Float32,Hp=3,validate_windows=false)
window=load_transport_window(driver,1); grid=driver_grid(driver)
air=map(copy,window.air_mass); dkg=map(copy,window.dkg); close(driver)
println("PROBE mass ",size(air[1])," ",eltype(air[1])," mass33=",air[1][4,4,33]," kg; dkg ",size(dkg[1])," max=",maximum(maximum,dkg)," kg/s")
function old_solve(rm,m,d,dt; variant=:old)
    FT=eltype(rm);n=length(rm);q=rm./m;ref=minimum(q);w=similar(q);prevw=zero(FT);prevg=zero(FT)
    for k in 1:n
        da=k>1 ? d[k-1] : zero(FT);db=k<n ? d[k] : zero(FT)
        invm=one(FT)/m[k];a=-dt*da*invm;c=-dt*db*invm
        b=variant==:rowsum ? one(FT)-a-c : one(FT)+dt*(da+db)*invm
        den=k==1 ? b : b-a*prevw
        g=(q[k]-ref-a*prevg)/den
        w[k]=c/den;q[k]=g;prevw=w[k];prevg=g
    end
    for k in n-1:-1:1;q[k]-=w[k]*q[k+1];end
    return (q.+ref).*m
end
function reference(rm,m,d,dt)
    n=length(m);f=Float64(dt).*Float64.(d[1:n-1])./Float64.(m[1:n-1]);b=Float64(dt).*Float64.(d[1:n-1])./Float64.(m[2:n])
    diag=ones(n);diag[1:n-1].+=f;diag[2:n].+=b
    return Tridiagonal(-f,diag,-b)\Float64.(rm)
end
rows=Dict[]
for p in 1:6,j in 1:15:90,i in 1:15:90,kseed in (16,33,50,64)
    m=air[p][i+3,j+3,:];d=dkg[p][i,j,:];rm=zeros(Float32,length(m));rm[kseed]=m[kseed]*4f-4
    r=reference(rm,m,d,360f0);base=sum(Float64,rm)
    for variant in (:old,:rowsum,:float64)
        out=variant==:float64 ? old_solve(Float64.(rm),Float64.(m),Float64.(d),360.;variant=:old) : old_solve(rm,m,d,360f0;variant)
        push!(rows,Dict("variant"=>String(variant),"panel"=>p,"i"=>i,"j"=>j,"seed"=>kseed,
            "drift"=>(sum(Float64,out)-base)/base,"relative_error"=>norm(Float64.(out)-r)/norm(r),"reference_drift"=>(sum(r)-base)/base,
            "max_exchange_fraction"=>maximum(360. .*Float64.(d[1:end-1])./min.(Float64.(m[1:end-1]),Float64.(m[2:end])))))
    end
end
for v in ("old","rowsum","float64")
    rs=filter(r->r["variant"]==v,rows)
    println(v," drift=",maximum(abs(r["drift"]) for r in rs)," reference_error=",maximum(r["relative_error"] for r in rs)," mean_drift=",sum(r["drift"] for r in rs)/length(rs))
end
println("REFERENCE drift=",maximum(abs(r["reference_drift"]) for r in rows)," max exchange=",maximum(r["max_exchange_fraction"] for r in rows))
open("/tmp/atmos-diffusion-column-probe.toml","w") do io;TOML.print(io,Dict("rows"=>rows));end
