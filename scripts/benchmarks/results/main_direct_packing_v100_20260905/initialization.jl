using AtmosTransport, TOML, Test
const DR = AtmosTransport.Models.DrivenRunner
function previous_state(grid, air, configs, ps)
    temporary = Dict{Symbol,typeof(air)}()
    for (name, cfg) in configs
        vmr = build_initial_mixing_ratio(air,grid,cfg;surface_pressure=ps)
        temporary[name] = pack_initial_tracer_mass(grid,air,vmr;mass_basis=DryBasis())
    end
    return CubedSphereState(DryBasis,grid.horizontal,air;temporary...)
end
direct_state(grid,air,configs,ps) = DR._initialize_cs_dry_state(grid,air,configs;surface_pressure=ps)
function measure(method, grid, air, configs, ps, nt, sample, label)
    GC.gc(true)
    stats = @timed method(grid,air,configs,ps)
    # Accurate host reduction of the final tracer's storage, outside timing.
    panels = get_tracer(stats.value,Symbol("tracer",lpad(nt,2,'0')))
    hp = grid.horizontal.Hp; nc = grid.horizontal.Nc
    total = sum(sum(Float64, @view a[hp+1:hp+nc,hp+1:hp+nc,:]) for a in panels)
    result = Dict("tracers"=>nt,"sample"=>sample,"method"=>label,
        "seconds"=>stats.time,"host_allocated_bytes"=>stats.bytes,
        "molecules"=>total*6.02214076e23/0.0289644,
        "panel_shape"=>collect(size(stats.value.tracers_raw[1])))
    println("PACKING ", result); flush(stdout)
    return result
end
results = Dict[]
for nt in (6,32)
    FT=Float32
    mesh=CubedSphereMesh(;Nc=90,Hp=3,FT)
    grid=AtmosGrid(mesh,HybridSigmaPressure(zeros(FT,67),FT.(range(0,1;length=67))),CPU();FT)
    air=ntuple(_->fill(FT(1e16),96,96,66),6)
    ps=ntuple(_->fill(FT(100000),90,90),6)
    configs=Dict(Symbol("tracer",lpad(i,2,'0'))=>Dict{String,Any}("kind"=>"pressure_layer",
                 "psurf_fraction"=>0.2+0.7*(i-1)/max(nt-1,1),"total_molecules"=>1e35) for i in 1:nt)
    for sample in 0:5
        methods = iseven(sample) ? ((previous_state,"previous"),(direct_state,"direct")) :
                                  ((direct_state,"direct"),(previous_state,"previous"))
        for (method,label) in methods
            push!(results,measure(method,grid,air,configs,ps,nt,sample,label))
        end
    end
end
open(get(ENV,"ATMOSTR_PACKING_PROBE_RESULT","/tmp/atmos-state-packing-probe.toml"),"w") do io
    TOML.print(io, Dict("results"=>results))
end
