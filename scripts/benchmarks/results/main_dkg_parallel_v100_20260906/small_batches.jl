using CUDA, AtmosTransport, Statistics, TOML, Test
using KernelAbstractions: get_backend, @kernel, @index, @Const
CUDA.allowscalar(false)
occursin("V100", CUDA.name(CUDA.device())) || error("V100 required")
const Diff = AtmosTransport.Operators.Diffusion
const input_path = "/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/era5_n320_transport_20181201_float32.bin"
driver = TransportBinaryDriver(input_path; FT=Float32, Hp=3, validate_windows=false)
window = load_transport_window(driver, 1)
air, dkg = copy(window.air_mass[1]), copy(window.dkg[1])
close(driver)
println("PROBE air=",size(air)," ",eltype(air)," mass33=",air[4,4,33]," kg; Dkg=",size(dkg)," max=",maximum(dkg)," kg/s")
@kernel function factor_columns!(factors, @Const(air), field, dt, Nz, Hp)
    ii,jj = @index(Global,NTuple)
    Diff._dkg_factor_column!(factors,air,field,dt,ii,jj,Nz,Hp)
end
@kernel function solve_tracers!(rm, @Const(air), field, @Const(factors), dt, Nz, Hp)
    ii,jj,t = @index(Global,NTuple)
    Diff._dkg_diffuse_mass_column!(rm,air,field,factors,dt,ii,jj,Nz,Hp,t)
end
const rows = Dict[]
function trial(FT,Nt)
    Nc,Nz,Hp = 90,66,3
    m = CuArray(FT.(air))
    field = AtmosTransport.State.PreComputedKzField(CuArray(FT.(dkg)))
    factors = CUDA.zeros(FT,Nc,Nc,Nz)
    host = zeros(FT,96,96,Nz,Nt)
    for t in 1:Nt
        k = clamp(round(Int, 16+48*(t-1)/max(Nt-1,1)),1,Nz)
        host[4:93,4:93,k,t] .= FT(4e-4) .* air[4:93,4:93,k]
    end
    initial = CuArray(host)
    rm = similar(initial)
    baseline = nothing
    factor! = factor_columns!(get_backend(rm),(32,2))
    for (method,tile) in ((:serial,(8,8)),(:serial,(32,2)),(:parallel,(32,2,1)),(:parallel,(32,4,1)),(:parallel,(32,1,2)),(:parallel,(16,2,2)))
        kernel! = method == :serial ? Diff._vertical_diffusion_cs_mass_dkg_packed_kernel!(get_backend(rm),tile) : solve_tracers!(get_backend(rm),tile)
        function apply!()
            if method == :serial
                kernel!(rm,m,field,factors,FT(360),Nz,Nt,Hp;ndrange=(Nc,Nc))
            else
                factor!(factors,m,field,FT(360),Nz,Hp;ndrange=(Nc,Nc))
                kernel!(rm,m,field,factors,FT(360),Nz,Hp;ndrange=(Nc,Nc,Nt))
            end
        end
        copyto!(rm,initial)
        CUDA.@sync apply!()
        output = Array(rm)
        if baseline === nothing
            baseline = output
        else
            @test output == baseline
        end
        samples = Float64[]
        for sample in 1:7
            copyto!(rm,initial)
            CUDA.synchronize()
            push!(samples, CUDA.@elapsed apply!())
        end
        row=Dict("float_type"=>string(FT),"tracers"=>Nt,"method"=>String(method),"tile"=>collect(tile),
                 "median_seconds"=>median(samples),"samples_seconds"=>samples)
        push!(rows,row)
        println(row);flush(stdout)
    end
    nothing
end
for FT in (Float32,Float64), Nt in (2,3,4)
    trial(FT,Nt)
    GC.gc(true);CUDA.reclaim()
end
open("/tmp/atmos-dkg-small-batch-probe.toml","w") do io
    TOML.print(io,Dict("measurements"=>rows))
end
