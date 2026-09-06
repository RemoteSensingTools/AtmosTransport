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
# Conservative LU: scale the two bidiagonal factors to make each column sum
# one. Each inverse becomes a directed retention/transfer pass; no total fix.
function conservative_solve(rm, m, d, dt)
    FT=eltype(rm); out=copy(rm); n=length(rm); beta=similar(rm)
    vprev=zero(FT); incoming=zero(FT)
    for k in 1:n
        e=one(FT)+vprev
        u=k<n ? dt*d[k]/m[k] : zero(FT)
        retain=e/(e+u)
        beta[k]=one(FT)/e
        accum=out[k]+incoming
        out[k]=retain*accum
        incoming=accum-out[k]
        vprev=k<n ? (dt*d[k]/m[k+1])*retain : zero(FT)
    end
    incoming=zero(FT)
    for k in n:-1:1
        accum=out[k]+incoming
        out[k]=beta[k]*accum
        incoming=accum-out[k]
    end
    return out
end

using Random, Test
rng=MersenneTwister(1947)
@testset "Conservative LU matches backward Euler on signed and stiff columns" begin
    for FT in (Float32,Float64), n in (1,2,3,66), strength in (0.,1e-4,1.,40.,1e4), profile in (:positive,:signed,:uniform,:offset)
        m=FT.(10 .^ (2 .*rand(rng,n)))
        d=FT.(strength .* min.(m[1:end-1],m[2:end])); push!(d,zero(FT))
        q=profile==:positive ? rand(rng,FT,n) : profile==:signed ? randn(rng,FT,n) : profile==:uniform ? fill(FT(4e-4),n) : FT(4e-4) .+ FT(1e-9).*randn(rng,FT,n)
        rm=q.*m
        out=conservative_solve(rm,m,d,one(FT)); r=reference(rm,m,d,one(FT))
        scale=sum(abs,Float64.(rm)); tol=FT==Float32 ? 6e-7 : 3e-12
        @test abs(sum(Float64,out)-sum(Float64,rm))/scale < (FT==Float32 ? 2e-7 : 2e-15)
        @test norm(Float64.(out)-r)/norm(r)<tol
        profile in (:positive,:uniform,:offset) && @test all(>=(0),out)
    end
end
