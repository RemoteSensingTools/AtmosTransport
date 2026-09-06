using CUDA, AtmosTransport, TOML, NCDatasets
CUDA.allowscalar(false)
occursin("V100", CUDA.name(CUDA.device())) || error("V100 required")
const input_path=get(ENV,"ATMOSTR_MASS_INPUT","/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/era5_n320_transport_20181201_float32.bin")
const out="/tmp/atmos-drift-ablation"
mkpath(out)
for case in ("all", "no_diffusion", "no_convection", "advection_only", "all_float64")
    cfg=Dict{String,Any}(
        "architecture"=>Dict("backend"=>"cuda"),
        "numerics"=>Dict("float_type"=>case=="all_float64" ? "Float64" : "Float32"),
        "input"=>Dict("binary_paths"=>[input_path]),
        "advection"=>Dict("scheme"=>"ppm"),
        "diffusion"=>Dict("kind"=>case in ("no_diffusion","advection_only") ? "none" : "tm5_dkg"),
        "convection"=>case in ("no_convection","advection_only") ? Dict("kind"=>"none") : Dict("kind"=>"tm5","use_collab_lu"=>case!="all_float64"),
        "run"=>Dict("stop_window"=>24,"air_mass_reset_mode"=>"preserve_tracer_mass"),
        "output"=>Dict("path"=>joinpath(out,case*".nc"),"snapshot_hours"=>collect(0:24),
            "fields"=>Dict("layers"=>"none","column_mean"=>false,"column_mass"=>false,"air_mass"=>false,"air_mass_per_area"=>false,"column_air_mass_per_area"=>false)),
        "tracers"=>Dict("tracer$(i)"=>Dict("init"=>Dict("kind"=>"pressure_layer","psurf_fraction"=>0.2+0.7*(i-1)/5,"total_molecules"=>1e35)) for i in 1:6))
    println("CASE ",case); flush(stdout)
    AtmosTransport.Models.DrivenRunner.run_driven_simulation(cfg)
    NCDataset(joinpath(out,case*".nc")) do ds
        rows=Dict(k=>Float64.(ds[k][:]) for k in keys(ds) if endswith(k,"_total_mass"))
        for (k,a) in sort(collect(rows))
            println("DRIFT ",case," ",k," ",(a[end]-a[1])/a[1])
        end
        open(joinpath(out,case*".toml"),"w") do io
            TOML.print(io,rows)
        end
    end
    GC.gc(true); CUDA.reclaim(); flush(stdout);flush(stderr)
end
