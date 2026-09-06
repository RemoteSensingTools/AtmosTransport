using Test, TOML

root = @__DIR__
@testset "Full-day seam investigation: hourly compensated totals" begin
    open(joinpath(root, "drift_summary.csv"), "w") do io
        println(io, "experiment,case,tracer,initial_total_kg,final_total_kg,signed_final_relative_drift,max_absolute_relative_drift")
        for experiment in ("ablation", "before", "after")
            for path in sort(readdir(joinpath(root, experiment); join=true))
                endswith(path, ".toml") || continue
                case = splitext(basename(path))[1]
                rows = TOML.parsefile(path)
                @test length(rows) == 6
                for (tracer, values) in sort(collect(rows))
                    @test length(values) == 25
                    @test all(isfinite, values)
                    drift = (values .- first(values)) ./ first(values)
                    largest = maximum(abs, drift)
                    println(io, join((experiment, case, tracer, first(values), last(values), last(drift), largest), ','))
                    if experiment == "after"
                        @test largest < (case == "all_float64" ? 3e-14 : 1e-6)
                        before = TOML.parsefile(joinpath(root, "before", basename(path)))[tracer]
                        @test first(values) == first(before)
                        @test abs(last(drift)) < abs((last(before)-first(before))/first(before))
                    end
                end
            end
        end
    end
end
