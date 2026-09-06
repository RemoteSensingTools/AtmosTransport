get(ENV, "ATMOSTR_TIMERS", "") == "1" || error("Set ATMOSTR_TIMERS=1 to match the baseline profile")
using CUDA, AtmosTransport, TOML, NCDatasets
CUDA.allowscalar(false)
occursin("V100",CUDA.name(CUDA.device())) || error("V100 required")
const input_path = "/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/era5_n320_transport_20181201_float32.bin"
length(ARGS) == 1 && ARGS[1] in ("256", "32", "32x2") || error("Pass 256, 32 or 32x2")
const precision = "Float64"
const tile = ARGS[1] == "32x2" ? (32,2) : parse(Int, ARGS[1])
const Ext = Base.get_extension(AtmosTransport, :AtmosTransportCUDAExt)
Core.eval(Ext, :(AtmosTransport.Operators.Advection._cs_packed_sweep_workgroupsize(
    ::CUDABackend, ::AtmosTransport.Operators.Advection.PPMScheme,
    ::Type{Float64}) = $tile))
const output_dir = joinpath(get(ENV, "ATMOSTR_BENCHMARK_OUTPUT_ROOT", "/tmp"), "atmos-f64-tiles-" * ARGS[1])
mkpath(output_dir)
function run_case(nt, sample, collab)
    mode = collab ? "collab" : "legacy"
    path = joinpath(output_dir,"$(mode)_tracers$(nt)_sample$(sample).nc")
    cfg = Dict{String,Any}(
        "architecture"=>Dict("backend"=>"cuda"),
        "numerics"=>Dict("float_type"=>precision),
        "input"=>Dict("binary_paths"=>[input_path]),
        "advection"=>Dict("scheme"=>"ppm"),
        "diffusion"=>Dict("kind"=>"tm5_dkg"),
        "convection"=>Dict("kind"=>"tm5","use_collab_lu"=>collab,"lmax_conv"=>0,"n_merge"=>1),
        "run"=>Dict("stop_window"=>24,"air_mass_reset_mode"=>"preserve_tracer_mass"),
        "output"=>Dict("path"=>path,"snapshot_hours"=>[0,24],
            "fields"=>Dict("layers"=>"none","column_mean"=>true,"column_mass"=>false,
                "air_mass"=>false,"air_mass_per_area"=>false,"column_air_mass_per_area"=>false)),
        "tracers"=>Dict("tracer$(lpad(i,2,'0'))"=>Dict("init"=>Dict("kind"=>"pressure_layer",
            "psurf_fraction"=>0.2+0.7*(i-1)/max(nt-1,1),"total_molecules"=>1e35)) for i in 1:nt))
    open(joinpath(output_dir,"$(mode)_tracers$(nt).toml"),"w") do io
        TOML.print(io,cfg)
    end
    GC.gc(true); CUDA.reclaim()
    stats = @timed AtmosTransport.Models.DrivenRunner.run_driven_simulation(cfg)
    CUDA.synchronize()
    model=stats.value
    totals=Dict(String(name)=>Float64(AtmosTransport.State.total_mass(model.state,name))
                for name in AtmosTransport.State.tracer_names(model.state))
    result=Dict("tracers"=>nt,"sample"=>sample,"collab"=>collab,"precision"=>precision,"wall_seconds"=>stats.time,
        "host_allocated_bytes"=>stats.bytes,"gc_seconds"=>stats.gctime,
        "output_bytes"=>filesize(path),"final_totals"=>totals)
    open(joinpath(output_dir,"$(mode)_result_$(nt)_$(sample).toml"),"w") do io
        TOML.print(io,result)
    end
    # Preserve one final full-precision state outside the timed region.
    # Snapshot fields are Float32 on disk even for a Float64 simulation.
    if sample == 1 && precision == "Float64"
        open(joinpath(output_dir,"$(mode)_tracers$(nt).state"),"w") do io
            for panel in model.state.tracers_raw
                write(io, Array(panel))
            end
        end
    end
    println("PROFILE ",result);flush(stdout);flush(stderr)
    nothing
end
for nt in (6,32), sample in 0:2
    run_case(nt,sample,true)
end
