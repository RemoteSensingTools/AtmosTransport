using CUDA, AtmosTransport, Statistics, TOML, Test
using KernelAbstractions: get_backend
CUDA.allowscalar(false)
occursin("V100", CUDA.name(CUDA.device())) || error("V100 required")
const Diff = AtmosTransport.Operators.Diffusion
const input_path = "/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/era5_n320_transport_20181201_float32.bin"
driver = TransportBinaryDriver(input_path; FT=Float32, Hp=3, validate_windows=false)
window = load_transport_window(driver, 1)
air, dkg = copy(window.air_mass[1]), copy(window.dkg[1])
close(driver)
println("PROBE air=",size(air)," ",eltype(air)," mass33=",air[4,4,33]," kg; Dkg=",size(dkg)," max=",maximum(dkg)," kg/s")
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
    tiles = ((8,8),(16,4),(32,2),(32,4),(64,1),(32,1),(16,8))
    baseline = nothing
    for tile in tiles
        kernel! = Diff._vertical_diffusion_cs_mass_dkg_packed_kernel!(get_backend(rm),tile)
        copyto!(rm,initial)
        CUDA.@sync kernel!(rm,m,field,factors,FT(360),Nz,Nt,Hp;ndrange=(Nc,Nc))
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
            elapsed = CUDA.@elapsed kernel!(rm,m,field,factors,FT(360),Nz,Nt,Hp;ndrange=(Nc,Nc))
            push!(samples,elapsed)
        end
        row=Dict("float_type"=>string(FT),"tracers"=>Nt,"tile"=>collect(tile),
                 "median_seconds"=>median(samples),"samples_seconds"=>samples)
        push!(rows,row)
        println(row);flush(stdout)
    end
    nothing
end
for FT in (Float32,Float64), Nt in (6,32,65)
    trial(FT,Nt)
    GC.gc(true);CUDA.reclaim()
end
open("/tmp/atmos-dkg-launch-probe.toml","w") do io
    TOML.print(io,Dict("measurements"=>rows))
end
