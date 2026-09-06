using CUDA, AtmosTransport, KernelAbstractions, TOML, Test, Statistics
using AtmosTransport.Operators.Advection: MonotoneLimiter
const Adv = AtmosTransport.Operators.Advection
CUDA.allowscalar(false)
occursin("V100",CUDA.name(CUDA.device())) || error("V100 required")
const path = "/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/era5_n320_transport_20181201_float32.bin"
driver = TransportBinaryDriver(path; FT=Float64, Hp=3, validate_windows=false)
mesh = driver_grid(driver).horizontal
window = load_transport_window(driver,1)
all_mass = map(copy,window.air_mass)
fluxes = (copy(window.fluxes.am[1]),copy(window.fluxes.bm[1]),copy(window.fluxes.cm[1]))
close(driver)
Adv.fill_panel_halos!(all_mass,mesh)
mass = all_mass[1]
nc,hp,nz = mesh.Nc,mesh.Hp,size(mass,3)
println("PROBE mass=",size(mass)," ",eltype(mass)," level33_kg=",mass[hp+1,hp+1,33],
        " flux_shapes=",map(size,fluxes))
rows = Dict[]
@testset "Float64 packed sweep tile identity on real C90 L66 inputs" begin
    for nt in (1,6,32,65)
        host = Array{Float64}(undef,size(mass)...,nt)
        for t in 1:nt,k in axes(mass,3),j in axes(mass,2),i in axes(mass,1)
            host[i,j,k,t] = mass[i,j,k] * (isodd(t) ? -1 : 1) *
                           (4e-4+1e-5*sin(i/8)+2e-5*cos(j/7)+1e-5*sin(k/4+t))
        end
        rm,m = CuArray(host),CuArray(mass)
        out_rm,out_m = CUDA.fill(-999.0,size(rm)),CUDA.fill(-999.0,size(m))
        for (direction,flux) in zip((:x,:y,:z),fluxes)
            f = CuArray(flux)
            kernel = getproperty(Adv,Symbol("_cs_",direction,"sweep_mt_kernel!"))
            reference = nothing
            for tile in (256,(32,2),(32,4),(16,8),(16,4),(8,8),32)
                launch! = kernel(get_backend(rm),tile)
                samples = Float64[]
                for sample in 0:5
                    CUDA.synchronize()
                    elapsed = CUDA.@elapsed launch!(out_rm,rm,out_m,m,f,
                        PPMScheme(MonotoneLimiter()),Int32(direction==:z ? nz : nc),
                        Int32(hp),Int32(nt),1.0; ndrange=(nc,nc,nz))
                    sample > 0 && push!(samples,elapsed)
                end
                result = (Array(out_rm),Array(out_m))
                if reference === nothing
                    reference = result
                else
                    @test isequal(result,reference)
                end
                @test all(isfinite,result[1])
                @test all(isfinite,result[2])
                row = Dict("tracers"=>nt,"direction"=>String(direction),"tile"=>string(tile),
                           "median_seconds"=>median(samples),"samples"=>samples)
                push!(rows,row)
                println(row); flush(stdout)
            end
        end
        rm=nothing;m=nothing;out_rm=nothing;out_m=nothing;host=nothing
        GC.gc(true);CUDA.reclaim()
    end
end
open("/tmp/atmos-f64-sweep-tiles.toml","w") do io
    TOML.print(io,Dict("measurements"=>rows))
end
