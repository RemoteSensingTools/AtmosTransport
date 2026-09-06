using Test, TOML
const here = @__DIR__
const labels = ("seam_only-Float32-6", "conservative-Float32-6", "conservative-Float64-6")
const results = map(label -> TOML.parsefile(joinpath(here,label,"result.toml")),labels)
const many = TOML.parsefile(joinpath(here,"conservative-Float32-32","result.toml"))
@testset "Archived seven-day mass comparison" begin
    for key in ("tracer01_total_mass", "tracer06_total_mass")
        counterpart = startswith(key,"tracer06") ? "tracer32_total_mass" : key
        @test results[2][key] == many[counterpart]
    end
    open(joinpath(here,"daily_drift.csv"),"w") do io
        println(io,"day,seam_only_float32,conservative_float32,conservative_float64")
        for day in 0:7
            # Compute differences before division to retain Float64 total ulps.
            maxima = map(results) do result
                maximum(begin
                    a = result["tracer$(lpad(t,2,'0'))_total_mass"]
                    abs((a[day+1]-first(a))/first(a))
                end for t in 1:6)
            end
            @test all(isfinite,maxima)
            @test maxima[3] < 1e-14
            day > 0 && (@test maxima[2] < maxima[1])
            println(io,day,',',join(maxima,','))
        end
    end
end
