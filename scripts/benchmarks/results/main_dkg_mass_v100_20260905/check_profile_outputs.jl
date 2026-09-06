using NCDatasets, Test, LinearAlgebra
const before = "/tmp/atmos-split-day-profile-after"
const after = "/tmp/atmos-dkg-day-profile-after"
largest = 0.0
baseline_largest = 0.0
@testset "32-tracer conservative Dkg full-day outputs" begin
    for sample in (1,2)
        name="tracers32_sample$(sample).nc"
        NCDataset(joinpath(before,name)) do a
            NCDataset(joinpath(after,name)) do b
                @test Set(keys(a)) == Set(keys(b))
                @test b.attrib["completed_snapshots"] == 2
                for key in keys(b)
                    x,y = Array(a[key]),Array(b[key])
                    @test size(x) == size(y)
                    @test all(isfinite,y)
                    if endswith(key,"_total_mass")
                        @test first(x) == first(y)
                        drift=abs(last(y)-first(y))/abs(first(y))
                        global baseline_largest=max(baseline_largest,abs(last(x)-first(x))/abs(first(x)))
                        global largest=max(largest,drift)
                        println(sample,' ',key," before=",abs(last(x)-first(x))/abs(first(x))," after=",drift)
                    elseif endswith(key,"_column_mean")
                        println(sample,' ',key," relative_L2_change=",norm(Float64.(x)-Float64.(y))/norm(Float64.(x)))
                    end
                end
            end
        end
    end
    @test largest < baseline_largest
end
println("Exploratory 1e-7 target met: ", largest < 1e-7)
println("Baseline maximum drift: ", baseline_largest)
println("Maximum final absolute relative drift: ",largest)
