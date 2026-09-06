using NCDatasets, Test, TOML, Statistics
const before = "/tmp/atmos-dkg-parallel-day-after"
const after = "/tmp/atmos-initial-reuse-day-after"
@testset "Full-day reusable initialization output identity" begin
    for sample in (1, 2)
        name = "tracers32_sample$(sample).nc"
        NCDataset(joinpath(before, name)) do a
            NCDataset(joinpath(after, name)) do b
                @test Set(keys(a)) == Set(keys(b))
                @test b.attrib["completed_snapshots"] == 2
                for key in keys(b)
                    x, y = Array(a[key]), Array(b[key])
                    @test size(x) == size(y)
                    @test all(isfinite, y)
                    @test isequal(x, y)
                end
            end
        end
    end
end
for stage in (before, after)
    samples = [TOML.parsefile(joinpath(stage, "result_32_$(i).toml")) for i in (1, 2)]
    println(stage, " median whole-run seconds=", median(s["wall_seconds"] for s in samples),
            "; median cumulative host bytes=", median(s["host_allocated_bytes"] for s in samples))
end
