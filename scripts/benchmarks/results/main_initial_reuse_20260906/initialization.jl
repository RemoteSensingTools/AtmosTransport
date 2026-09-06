using AtmosTransport, TOML, Test, Statistics
const IC = AtmosTransport.Models.InitialConditionIO
const Runner = AtmosTransport.Models.DrivenRunner
const input_path = "/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/era5_n320_transport_20181201_float32.bin"
driver = TransportBinaryDriver(input_path; FT=Float32, Hp=3, validate_windows=false)
grid = driver_grid(driver)
window = load_transport_window(driver, 1)
air, ps = map(copy, window.air_mass), map(copy, window.surface_pressure)
close(driver)
println("PROBE air=",size(air[1])," ",eltype(air[1])," mass33=",air[1][4,4,33]," kg; ps=",ps[1][1,1]," Pa")
# Baseline runner from 5db7c780: one fresh interior VMR tuple per tracer.
function allocating_initial_state(grid, air, configs; surface_pressure)
    indices = Dict{Symbol,Int}()
    for (name,_) in configs
        indices[name] = 0
    end
    names = Tuple(keys(indices))
    for (index,name) in enumerate(names)
        indices[name] = index
    end
    raw = ntuple(p -> similar(air[p],size(air[p])...,length(names)),6)
    for (name,cfg) in configs
        vmr = build_initial_mixing_ratio(air,grid,cfg;surface_pressure)
        destination = ntuple(p -> selectdim(raw[p],4,indices[name]),6)
        IC._cs_pack_interior_into_halo!(destination,grid,air,vmr,nothing)
    end
    return CubedSphereState(DryBasis,air,raw,names;halo_width=grid.horizontal.Hp)
end
rows = Dict[]
for nt in (6,32)
    configs = Dict(Symbol("tracer",lpad(i,2,'0'))=>Dict{String,Any}("kind"=>"pressure_layer",
        "psurf_fraction"=>0.2+0.7*(i-1)/max(nt-1,1),"total_molecules"=>1e35) for i in 1:nt)
    a = allocating_initial_state(grid,air,configs;surface_pressure=ps)
    b = Runner._initialize_cs_dry_state(grid,air,configs;surface_pressure=ps)
    @test tracer_names(a) == tracer_names(b)
    @test a.tracers_raw == b.tracers_raw
    for sample in 1:5, method in (isodd(sample) ? (:allocating,:reusing) : (:reusing,:allocating))
        GC.gc(true)
        f = method == :allocating ? allocating_initial_state : Runner._initialize_cs_dry_state
        stats = @timed f(grid,air,configs;surface_pressure=ps)
        row = Dict("tracers"=>nt,"method"=>String(method),"sample"=>sample,
                   "seconds"=>stats.time,"bytes"=>stats.bytes)
        push!(rows,row)
        println(row);flush(stdout)
    end
end
open("/tmp/atmos-initial-reuse-probe.toml","w") do io
    TOML.print(io,Dict("measurements"=>rows))
end
