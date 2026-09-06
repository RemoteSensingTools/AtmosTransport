using Test, TOML

root = @__DIR__
baseline = joinpath(root, "..", "main_mass_seams_v100_20260905", "ablation")
@testset "Split PPM full-day compensated totals" begin
    open(joinpath(root, "drift_summary.csv"), "w") do io
        println(io, "case,tracer,initial_total_kg,before_final_relative_drift,after_final_relative_drift,after_max_absolute_relative_drift")
        for case in ("all", "all_float64")
            before = TOML.parsefile(joinpath(baseline, case * ".toml"))
            after = TOML.parsefile(joinpath(root, "after", case * ".toml"))
            @test length(after) == 6
            @test keys(before) == keys(after)
            for (tracer, values) in sort(collect(after))
                old = before[tracer]
                @test length(values) == length(old) == 25
                @test all(isfinite, values)
                @test first(values) == first(old)
                drift = (values .- first(values)) ./ first(values)
                old_drift = (last(old) - first(old)) / first(old)
                @test maximum(abs, drift) < (case == "all_float64" ? 3e-14 : 1e-6)
                println(io, join((case, tracer, first(values), old_drift, last(drift), maximum(abs, drift)), ','))
            end
        end
    end
end

@testset "Analytic solid-body rotation fields and conservation" begin
    rows = TOML.parsefile(joinpath(root, "solid_rotation.toml"))["rows"]
    @test length(rows) == 24
    for a in filter(r -> r["method"] == "after", rows)
        b = only(filter(r -> r["method"] == "before" && all(
            r[k] == a[k] for k in ("Nc", "convention", "rotation_fraction")), rows))
        @test abs(a["relative_mass_drift"]) < 7e-15
        @test a["relative_l2"] <= b["relative_l2"]
        @test 0 < a["min_q"] <= a["max_q"]
    end
end
