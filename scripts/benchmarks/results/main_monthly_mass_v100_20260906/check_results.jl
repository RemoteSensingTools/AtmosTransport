using Test, TOML, NCDatasets, LinearAlgebra
length(ARGS) == 3 || error("Usage: check_results.jl float32_folder float64_folder output_folder")
f32_dir, f64_dir, output_dir = ARGS
mkpath(output_dir)
results = [TOML.parsefile(joinpath(folder, "result.toml")) for folder in (f32_dir, f64_dir)]
metrics = Dict{String,Any}()
@testset "Complete monthly mass measurements" begin
    for result in results
        @test result["daily_files"] == 31
        @test result["snapshot_count"] == 32
        @test result["tracers"] == 6
        @test result["input_bytes"] == 96_542_170_112
    end
    open(joinpath(output_dir, "daily_drift.csv"), "w") do io
        println(io, "day,float32,float64")
        for day in 0:31
            maxima = map(results) do result
                maximum(1:6) do t
                    totals = result["tracer$(lpad(t,2,'0'))_total_mass"]
                    abs((totals[day+1] - first(totals))/first(totals))
                end
            end
            @test all(isfinite, maxima)
            # Float64 conservation tolerance tests the complete composed run.
            @test maxima[2] < 1e-13
            println(io, day, ',', join(maxima, ','))
            metrics["day$(day)_max_relative_drift"] = maxima
        end
    end
end
@testset "Monthly full-precision fields and signed storage" begin
    shapes = [Tuple(result["native_state_shape"]) for result in results]
    @test shapes[1] == shapes[2] == (96,96,66,6)
    @test results[1]["native_state_tracers"] == results[2]["native_state_tracers"]
    for (folder, FT, shape) in zip((f32_dir,f64_dir), (Float32,Float64), shapes)
        @test filesize(joinpath(folder,"final.state")) == 6 * prod(shape) * sizeof(FT)
    end
    # Read one panel per precision at a time; exclude the three-cell halos.
    a, b = Array{Float32}(undef,shapes[1]), Array{Float64}(undef,shapes[2])
    open(joinpath(f32_dir,"final.state")) do stream_a
        open(joinpath(f64_dir,"final.state")) do stream_b
            for panel in 1:6
                read!(stream_a,a); read!(stream_b,b)
                for t in 1:6
                    x = Float64.(view(a,4:93,4:93,:,t))
                    y = view(b,4:93,4:93,:,t)
                    @test all(isfinite,x)
                    @test all(isfinite,y)
                    relative_l2 = norm(x .- y)/norm(y)
                    @test isfinite(relative_l2)
                    prefix = "panel$(panel)_" * results[1]["native_state_tracers"][t]
                    metrics[prefix*"_relative_L2"] = relative_l2
                    metrics[prefix*"_minimum_storage"] = [minimum(x),minimum(y)]
                    metrics[prefix*"_negative_mass_fraction"] =
                        [sum(v -> -min(v,0.0),z)/sum(abs,z) for z in (x,y)]
                end
            end
        end
    end
end
open(joinpath(output_dir,"comparison.toml"),"w") do io
    TOML.print(io,metrics)
end
println("MONTHLY CHECKS COMPLETE: ",output_dir)
