using NCDatasets, Test, TOML, LinearAlgebra
const old = "/tmp/atmos-weekly-drift-conservative-Float64-6"
const new = "/tmp/atmos-weekly-drift-f64collab-Float64-6"
metrics = Dict{String,Float64}()
@testset "Float64 collaborative and legacy seven-day output" begin
    NCDataset(joinpath(old,"weekly.nc")) do a
        NCDataset(joinpath(new,"weekly.nc")) do b
            @test a.attrib["completed_snapshots"] == b.attrib["completed_snapshots"] == 8
            @test Set(keys(a)) == Set(keys(b))
            for key in keys(a)
                x,y = Array(a[key]),Array(b[key])
                @test size(x) == size(y)
                @test all(isfinite,y)
                if endswith(key,"_total_mass")
                    @test first(x) == first(y)
                    drift = maximum(abs.(y .- first(y)))/abs(first(y))
                    @test drift < 1e-14
                    @test isapprox(x,y;rtol=1e-14)
                    metrics[key*"_max_daily_drift"] = drift
                elseif endswith(key,"_column_mean")
                    err = norm(y-x)/norm(x)
                    @test err < 2e-13
                    metrics[key*"_relative_L2"] = err
                else
                    @test x == y
                end
            end
        end
    end
end
open("/tmp/atmos-f64-weekly-comparison.toml","w") do io
    TOML.print(io,metrics)
end
println("MAX_DAILY_DRIFT ",maximum(v for (k,v) in metrics if endswith(k,"_max_daily_drift")))
println("MAX_COLUMN_RELATIVE_L2 ",maximum(v for (k,v) in metrics if endswith(k,"_relative_L2")))
