using CUDA, AtmosTransport, TOML, NCDatasets, Test, LinearAlgebra
CUDA.allowscalar(false)
occursin("V100", CUDA.name(CUDA.device())) || error("V100 required")
length(ARGS) == 3 || error("Usage: weekly_drift.jl label Float32|Float64 tracer_count")
label, precision = ARGS[1:2]
nt = parse(Int, ARGS[3])
const folder = "/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour"
paths = [joinpath(folder,"era5_n320_transport_2018120$(d)_float32.bin") for d in 1:7]
for path in paths
    println("INPUT ",basename(path)," bytes=",filesize(path))
end
out = "/tmp/atmos-weekly-drift-$(label)-$(precision)-$(nt)"
mkpath(out)
cfg = Dict{String,Any}(
    "architecture"=>Dict("backend"=>"cuda"),
    "numerics"=>Dict("float_type"=>precision),
    "input"=>Dict("binary_paths"=>paths),
    "advection"=>Dict("scheme"=>"ppm"),
    "diffusion"=>Dict("kind"=>"tm5_dkg"),
    "convection"=>Dict("kind"=>"tm5","use_collab_lu"=>true),
    "run"=>Dict("air_mass_reset_mode"=>"preserve_tracer_mass"),
    "output"=>Dict("path"=>joinpath(out,"weekly.nc"),"snapshot_hours"=>collect(0:24:168),
        "fields"=>Dict("layers"=>"none","column_mean"=>true,"column_mass"=>false,
            "air_mass"=>false,"air_mass_per_area"=>false,"column_air_mass_per_area"=>false)),
    "tracers"=>Dict("tracer$(lpad(i,2,'0'))"=>Dict("init"=>Dict("kind"=>"pressure_layer",
        "psurf_fraction"=>0.2+0.7*(i-1)/max(nt-1,1),"total_molecules"=>1e35)) for i in 1:nt))
open(joinpath(out,"config.toml"),"w") do io
    TOML.print(io,cfg)
end
stats = @timed AtmosTransport.Models.DrivenRunner.run_driven_simulation(cfg)
CUDA.synchronize()
rows = Dict{String,Any}("wall_seconds_including_first_compilation"=>stats.time,
    "cumulative_host_bytes"=>stats.bytes,"gc_seconds"=>stats.gctime,
    "process_max_rss_bytes"=>Sys.maxrss(),"tracers"=>nt,"precision"=>precision)
@testset "Seven complete daily files, finite output and complete snapshots" begin
    NCDataset(joinpath(out,"weekly.nc")) do ds
        @test ds.attrib["completed_snapshots"] == 8
        for key in keys(ds)
            value = Array(ds[key])
            @test all(isfinite,value)
            if endswith(key,"_total_mass")
                @test length(value) == 8
                totals = vec(Float64.(value))
                drift = (totals .- first(totals)) ./ first(totals)
                rows[key] = totals
                rows[key*"_relative_drift"] = drift
                println(key," daily relative drift=",drift)
            elseif endswith(key,"_column_mean")
                println(key," column mean extrema=",extrema(value))
            end
        end
    end
end
open(joinpath(out,"result.toml"),"w") do io
    TOML.print(io,rows)
end
println("WEEKLY RESULT ",out," wall=",stats.time," cumulative_host_bytes=",stats.bytes,
        " process_max_rss_bytes=",Sys.maxrss())
