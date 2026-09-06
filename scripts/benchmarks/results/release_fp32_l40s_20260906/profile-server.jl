get(ENV,"ATMOSTR_TIMERS","")=="1" || error("Set ATMOSTR_TIMERS=1")
using CUDA, AtmosTransport, TOML, NCDatasets
CUDA.allowscalar(false)
@assert length(collect(CUDA.devices()))==1
expected=get(ENV,"ATMOSTR_PROFILE_GPU_NAME","")
@assert !isempty(expected) && occursin(expected,CUDA.name(CUDA.device()))
length(ARGS)==1 || error("Pass output directory")
const output_dir=abspath(ARGS[1])
const input_path="/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/era5_n320_transport_20181201_float32.bin"
mkpath(output_dir)
CUDA.versioninfo()
function run_case(nt, windows, sample)
    stem="w$(windows)_tracers$(nt)_sample$(sample)"
    path=joinpath(output_dir,stem*".nc")
    collab=!(get(ENV,"ATMOSTR_PROFILE_LEGACY32","0")=="1" && nt>6)
    cfg=Dict{String,Any}(
        "architecture"=>Dict("backend"=>"cuda"),
        "numerics"=>Dict("float_type"=>"Float32"),
        "input"=>Dict("binary_paths"=>[input_path]),
        "advection"=>Dict("scheme"=>"ppm"),
        "diffusion"=>Dict("kind"=>"tm5_dkg"),
        "convection"=>Dict("kind"=>"tm5","use_collab_lu"=>collab,"lmax_conv"=>0,"n_merge"=>1),
        "run"=>Dict("stop_window"=>windows,"air_mass_reset_mode"=>"preserve_tracer_mass"),
        "output"=>Dict("path"=>path,"snapshot_hours"=>[0,windows],
            "fields"=>Dict("layers"=>"none","column_mean"=>true,"column_mass"=>false,
                "air_mass"=>false,"air_mass_per_area"=>false,"column_air_mass_per_area"=>false)),
        "tracers"=>Dict("tracer$(lpad(i,2,'0'))"=>Dict("init"=>Dict("kind"=>"pressure_layer",
            "psurf_fraction"=>0.2+0.7*(i-1)/max(nt-1,1),"total_molecules"=>1e35)) for i in 1:nt))
    open(joinpath(output_dir,"w$(windows)_tracers$(nt).toml"),"w") do io
        TOML.print(io,cfg)
    end
    GC.gc(true); CUDA.reclaim(); CUDA.synchronize()
    stats=@timed begin
        model=AtmosTransport.Models.DrivenRunner.run_driven_simulation(cfg)
        CUDA.synchronize()
        model
    end
    model=stats.value
    names=AtmosTransport.State.tracer_names(model.state)
    # Stable Float64 sums in NetCDF are validated separately; these native
    # state reductions retain the original profile's diagnostic only.
    totals=Dict(String(name)=>Float64(AtmosTransport.State.total_mass(model.state,name)) for name in names)
    result=Dict("tracers"=>nt,"windows"=>windows,"sample"=>sample,
        "wall_seconds"=>stats.time,"host_allocated_bytes"=>stats.bytes,
        "gc_seconds"=>stats.gctime,"output_bytes"=>filesize(path),
        "precision"=>"Float32","collab"=>collab,"device"=>CUDA.name(CUDA.device()),"final_totals"=>totals)
    target=joinpath(output_dir,stem*".toml")
    open(target*".tmp","w") do io
        TOML.print(io,result)
    end
    mv(target*".tmp",target)
    println("PROFILE_COMPLETED ",stem," seconds=",stats.time);flush(stdout);flush(stderr)
    nothing
end
println("PROFILE_SERVER_READY");flush(stdout);flush(stderr)
for line in eachline(stdin)
    line=="quit" && break
    args=parse.(Int,split(strip(line),','))
    length(args)==3 || error("Expected tracers,windows,sample")
    run_case(args...)
end
println("PROFILE_SERVER_PASSED");flush(stdout);flush(stderr)
