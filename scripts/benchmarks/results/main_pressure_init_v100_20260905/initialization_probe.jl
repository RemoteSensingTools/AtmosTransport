using AtmosTransport, TOML
const IC = AtmosTransport.Models.InitialConditionIO
function probe(nt)
    FT = Float32
    mesh = CubedSphereMesh(; Nc=90, Hp=3, FT)
    grid = AtmosGrid(mesh, HybridSigmaPressure(zeros(FT,67),FT.(range(0,1;length=67))), CPU();FT)
    air = ntuple(_ -> fill(FT(1e16),96,96,66),6)
    ps = ntuple(_ -> fill(FT(100000),90,90),6)
    tracers = Dict{Symbol,typeof(air)}()
    vmr_bytes = pack_bytes = 0
    vmr_time = pack_time = 0.0
    for i in 1:nt
        cfg = Dict{String,Any}("kind"=>"pressure_layer", "psurf_fraction"=>0.2+0.7*(i-1)/max(nt-1,1), "total_molecules"=>1e35)
        a = @timed IC.build_initial_mixing_ratio(air, grid, cfg; surface_pressure=ps)
        b = @timed IC.pack_initial_tracer_mass(grid, air, a.value; mass_basis=DryBasis())
        vmr_bytes += a.bytes; pack_bytes += b.bytes
        vmr_time += a.time; pack_time += b.time
        tracers[Symbol("tracer",i)] = b.value
    end
    c = @timed CubedSphereState(DryBasis,mesh,air;tracers...)
    println("PROBE ", (;nt, vmr_bytes, pack_bytes, state_bytes=c.bytes, vmr_time, pack_time, state_time=c.time,
        shape=size(c.value.tracers_raw[1]), total=Float64(AtmosTransport.State.total_mass(c.value,Symbol("tracer",nt)))))
    flush(stdout)
end
probe(6)
GC.gc(true)
probe(6)
GC.gc(true)
probe(32)
GC.gc(true)
probe(32)
