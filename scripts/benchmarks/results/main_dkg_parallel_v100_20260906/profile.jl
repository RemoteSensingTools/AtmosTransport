get(ENV, "ATMOSTR_TIMERS", "") == "1" || error("Set ATMOSTR_TIMERS=1 to match the baseline profile")
using CUDA, AtmosTransport, TOML, NCDatasets
CUDA.allowscalar(false)
occursin("V100",CUDA.name(CUDA.device())) || error("V100 required")
const input_path = "/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/era5_n320_transport_20181201_float32.bin"
length(ARGS) == 1 && ARGS[1] in ("before", "after") || error("Pass before or after")
const output_dir = "/tmp/atmos-dkg-parallel-day-" * ARGS[1]
mkpath(output_dir)
function run_case(nt, sample)
    path = joinpath(output_dir,"tracers$(nt)_sample$(sample).nc")
    cfg = Dict{String,Any}(
        "architecture"=>Dict("backend"=>"cuda"),
        "numerics"=>Dict("float_type"=>"Float32"),
        "input"=>Dict("binary_paths"=>[input_path]),
        "advection"=>Dict("scheme"=>"ppm"),
        "diffusion"=>Dict("kind"=>"tm5_dkg"),
        "convection"=>Dict("kind"=>"tm5","use_collab_lu"=>true,"lmax_conv"=>0,"n_merge"=>1),
        "run"=>Dict("stop_window"=>24,"air_mass_reset_mode"=>"preserve_tracer_mass"),
        "output"=>Dict("path"=>path,"snapshot_hours"=>[0,24],
            "fields"=>Dict("layers"=>"none","column_mean"=>true,"column_mass"=>false,
                "air_mass"=>false,"air_mass_per_area"=>false,"column_air_mass_per_area"=>false)),
        "tracers"=>Dict("tracer$(lpad(i,2,'0'))"=>Dict("init"=>Dict("kind"=>"pressure_layer",
            "psurf_fraction"=>0.2+0.7*(i-1)/max(nt-1,1),"total_molecules"=>1e35)) for i in 1:nt))
    open(joinpath(output_dir,"tracers$(nt).toml"),"w") do io
        TOML.print(io,cfg)
    end
    GC.gc(true); CUDA.reclaim()
    stats = @timed AtmosTransport.Models.DrivenRunner.run_driven_simulation(cfg)
    CUDA.synchronize()
    model=stats.value
    totals=Dict(String(name)=>Float64(AtmosTransport.State.total_mass(model.state,name))
                for name in AtmosTransport.State.tracer_names(model.state))
    result=Dict("tracers"=>nt,"sample"=>sample,"wall_seconds"=>stats.time,
        "host_allocated_bytes"=>stats.bytes,"gc_seconds"=>stats.gctime,
        "output_bytes"=>filesize(path),"final_totals"=>totals)
    open(joinpath(output_dir,"result_$(nt)_$(sample).toml"),"w") do io
        TOML.print(io,result)
    end
    println("PROFILE ",result);flush(stdout);flush(stderr)
    nothing
end
for nt in (32,), sample in 0:2
    run_case(nt,sample)
end
