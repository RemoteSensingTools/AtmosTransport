using AtmosTransport, TOML
const Models = AtmosTransport.Models.InitialConditionIO
const input_path = "/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/era5_n320_transport_20181201_float32.bin"
driver = TransportBinaryDriver(input_path; FT=Float32, Hp=3, validate_windows=false)
grid = driver_grid(driver)
window = load_transport_window(driver, 1)
air = map(copy, window.air_mass)
ps = map(copy, window.surface_pressure)
close(driver)
println("PROBE air=", size(air[1]), " ", eltype(air[1]),
        " interior level-33 mass=", air[1][4,4,33], " kg; ps=", ps[1][1,1], " Pa")
function measure_phases(grid, air, ps, nt, sample)
    raw = ntuple(p -> similar(air[p], size(air[p])..., nt), 6)
    configs = [Dict{String,Any}("kind"=>"pressure_layer", "psurf_fraction"=>0.2+0.7*(i-1)/max(nt-1,1), "total_molecules"=>1e35) for i in 1:nt]
    GC.gc(true)
    build_seconds = 0.0; build_bytes = 0
    pack_seconds = 0.0; pack_bytes = 0
    for (index, cfg) in enumerate(configs)
        build = @timed build_initial_mixing_ratio(air, grid, cfg; surface_pressure=ps)
        build_seconds += build.time; build_bytes += build.bytes
        destination = ntuple(p -> selectdim(raw[p], 4, index), 6)
        pack = @timed Models._cs_pack_interior_into_halo!(destination, grid, air, build.value, nothing)
        pack_seconds += pack.time; pack_bytes += pack.bytes
    end
    hp=grid.horizontal.Hp; nc=grid.horizontal.Nc
    mass = sum(sum(Float64, @view a[hp+1:hp+nc,hp+1:hp+nc,:,nt]) for a in raw)
    result = Dict("tracers"=>nt,"sample"=>sample,"build_seconds"=>build_seconds,
        "build_bytes"=>build_bytes,"pack_seconds"=>pack_seconds,"pack_bytes"=>pack_bytes,
        "last_tracer_molecules"=>mass*6.02214076e23/0.0289644)
    println(result);flush(stdout)
    return result
end
results = [measure_phases(grid, air, ps, nt, sample) for nt in (6,32) for sample in 0:5]
open("/tmp/atmos-initial-phases.toml","w") do io
    TOML.print(io,Dict("results"=>results))
end
