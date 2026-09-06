using NCDatasets, Test

const baseline = "/tmp/atmos-main-ppm-day-after"
const candidate = "/tmp/atmos-split-day-profile-after"
max_before = 0.0
max_after = 0.0
@testset "32-tracer full-day split PPM conservation and finite fields" begin
    for sample in (1, 2)
        name = "tracers32_sample$(sample).nc"
        NCDataset(joinpath(baseline, name)) do a
            NCDataset(joinpath(candidate, name)) do b
                @test Set(keys(a)) == Set(keys(b))
                @test b.attrib["completed_snapshots"] == 2
                for key in keys(b)
                    x, y = a[key][:], b[key][:]
                    @test size(x) == size(y)
                    @test all(isfinite, y)
                    if endswith(key, "_total_mass")
                        @test first(x) == first(y)
                        before_drift = abs(last(x) - first(x)) / abs(first(x))
                        after_drift = abs(last(y) - first(y)) / abs(first(y))
                        global max_before = max(max_before, before_drift)
                        global max_after = max(max_after, after_drift)
                        @test after_drift < 1e-6
                        println(sample, ' ', key, " before=", before_drift, " after=", after_drift)
                    end
                end
            end
        end
    end
end
println("Maximum absolute final relative drift: before=", max_before, " after=", max_after)
