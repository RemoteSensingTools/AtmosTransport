using CUDA, AtmosTransport, KernelAbstractions, TOML, Test
using AtmosTransport.Operators.Advection: MonotoneLimiter
include(joinpath(@__DIR__, "trial_kernels.jl"))
const Adv=AtmosTransport.Operators.Advection
CUDA.allowscalar(false)
occursin("V100",CUDA.name(CUDA.device())) || error("V100 required")
const source_path="/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/era5_n320_transport_20181201_float32.bin"
driver=TransportBinaryDriver(source_path;FT=Float32,Hp=3,validate_windows=false)
mesh=driver_grid(driver).horizontal
window=load_transport_window(driver,1)
all_mass=map(copy,window.air_mass)
fluxes=(copy(window.fluxes.am[1]),copy(window.fluxes.bm[1]),copy(window.fluxes.cm[1]))
close(driver)
Adv.fill_panel_halos!(all_mass,mesh)
mass=all_mass[1]
nc,hp,nz=mesh.Nc,mesh.Hp,size(mass,3)
println("PROBE ",summary(mesh)," mass shape=",size(mass)," dtype=",eltype(mass)," interior mass=",mass[hp+1,hp+1,33])
rows=Dict[]
@testset "Parallel-tracer sweep trial" begin
    for nt in (1,6,32)
        host=Array{Float32}(undef,size(mass)...,nt)
        for t in 1:nt,k in axes(mass,3),j in axes(mass,2),i in axes(mass,1)
            host[i,j,k,t]=mass[i,j,k] * Float32((isodd(t) ? -1 : 1)*(4e-4+1e-5*sin(i/8)+2e-5*cos(j/7)+1e-5*sin(k/4+t)))
        end
        rm=CuArray(host);m=CuArray(mass)
        old_rm=CUDA.fill(-999f0,size(rm));new_rm=similar(old_rm)
        old_m=CUDA.fill(-999f0,size(m));new_m=similar(old_m)
        for (direction,flux) in zip((:x,:y,:z),fluxes)
            f=CuArray(flux)
            old_kernel=getproperty(Adv,Symbol("_cs_",direction,"sweep_mt_kernel!"))
            new_kernel=getproperty(SweepTrial,Symbol("parallel_",direction,"!"))
            # Each launch reads fixed initial data and writes an independent result.
            for threads in (128,256)
                old=old_kernel(get_backend(rm),threads);new=new_kernel(get_backend(rm),threads)
                for sample in 0:5
                    order=iseven(sample) ? (:serial,:parallel) : (:parallel,:serial)
                    for method in order
                        CUDA.synchronize()
                        elapsed = CUDA.@elapsed begin
                            if method==:serial
                                old(old_rm,rm,old_m,m,f,PPMScheme(MonotoneLimiter()),
                                    Int32(direction==:z ? nz : nc),Int32(hp),Int32(nt),1f0;ndrange=(nc,nc,nz))
                            else
                                new(new_rm,rm,new_m,m,f,PPMScheme(MonotoneLimiter()),
                                    Int32(direction==:z ? nz : nc),Int32(hp),Int32(nt),1f0;ndrange=(nc,nc,nz,nt))
                            end
                        end
                        push!(rows,Dict("tracers"=>nt,"direction"=>String(direction),"threads"=>threads,
                            "sample"=>sample,"method"=>String(method),"seconds"=>elapsed))
                    end
                end
                a=Array(old_rm)[hp+1:hp+nc,hp+1:hp+nc,:,:]
                b=Array(new_rm)[hp+1:hp+nc,hp+1:hp+nc,:,:]
                @test a==b
                @test all(isfinite,b)
                @test Array(old_m)[hp+1:hp+nc,hp+1:hp+nc,:]==Array(new_m)[hp+1:hp+nc,hp+1:hp+nc,:]
            end
        end
        rm=nothing;m=nothing;old_rm=nothing;new_rm=nothing;old_m=nothing;new_m=nothing
        GC.gc(true);CUDA.reclaim()
    end
end
open("/tmp/atmos-sweep-trial.toml","w") do io
    TOML.print(io,Dict("measurements"=>rows))
end
