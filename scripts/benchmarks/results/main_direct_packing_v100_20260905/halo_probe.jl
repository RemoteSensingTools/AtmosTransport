using CUDA, AtmosTransport, TOML, Test
using AtmosTransport.Operators.Advection: fill_panel_halos!
CUDA.allowscalar(false)
occursin("V100",CUDA.name(CUDA.device())) || error("V100 required")
const input_path = "/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/era5_n320_transport_20181201_float32.bin"
const driver = TransportBinaryDriver(input_path; FT=Float32, Hp=3, validate_windows=false)
const mesh = driver_grid(driver).horizontal
close(driver)
println("Halo probe: ", summary(mesh), "; L66 Float32 panel-marker fields")
rows=Dict[]
@testset "Isolated halo GPU/CPU equivalence" begin
    for nt in (6,32), direction in (1,2)
        FT=Float32; nc=90; hp=3; nz=66
        @assert mesh.Nc == nc && mesh.Hp == hp
        host=ntuple(p -> fill(FT(p), nc+2hp,nc+2hp,nz,nt),6)
        device=map(CuArray,host)
        # Include directional corners in both paths; start/end each timed call
        # with an idle stream to exclude previously queued advection work.
        fill_panel_halos!(host,mesh;dir=direction)
        for sample in 0:10
            CUDA.synchronize()
            t=@elapsed begin
                fill_panel_halos!(device,mesh;dir=direction)
                CUDA.synchronize()
            end
            push!(rows,Dict("tracers"=>nt,"direction"=>direction,"sample"=>sample,"seconds"=>t))
        end
        for p in 1:6
            @test Array(device[p]) == host[p]
        end
        device=nothing;host=nothing
        GC.gc(true);CUDA.reclaim()
    end
end
open("/tmp/atmos-v100-halo-probe.toml","w") do io
    TOML.print(io, Dict("measurements"=>rows))
end
println("Halo measurements complete");flush(stdout)
