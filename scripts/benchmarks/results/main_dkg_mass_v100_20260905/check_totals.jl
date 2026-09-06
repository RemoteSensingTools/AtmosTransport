# Recheck the archived small TOMLs without the meteorological archive or GPU.
using Test, TOML, Statistics
const here = @__DIR__
const baseline = joinpath(here,"..","main_split_mass_seams_v100_20260905","after")
@testset "Archived conservative Dkg daily totals" begin
    for case in ("all", "all_float64")
        a = TOML.parsefile(joinpath(baseline,case*".toml"))
        b = TOML.parsefile(joinpath(here,"after",case*".toml"))
        @test Set(keys(a)) == Set(keys(b))
        old_max, new_max = 0.0, 0.0
        for key in sort(collect(keys(b)))
            x,y = a[key],b[key]
            @test length(x) == length(y) == 25
            @test first(x) == first(y)
            @test all(isfinite,y)
            old_max = max(old_max, maximum(abs, (x .- first(x)) ./ first(x)))
            new_max = max(new_max, maximum(abs, (y .- first(y)) ./ first(y)))
        end
        @test new_max < old_max
        println(case," worst hourly relative drift: ",old_max," -> ",new_max,
                "; exploratory 1e-7 target met: ",new_max < 1e-7)
    end
end
for stage in (baseline, joinpath(here,"after"))
    samples = [TOML.parsefile(joinpath(stage,"result_32_$(i).toml")) for i in (1,2)]
    println(stage," median wall seconds=",median(s["wall_seconds"] for s in samples),
            "; median cumulative host bytes=",median(s["host_allocated_bytes"] for s in samples))
end
