using NCDatasets, Test, SHA, TOML, Statistics
const baseline = get(ENV, "ATMOSTR_BENCHMARK_REFERENCE", "/tmp/atmos-f64-profile-Float64")
const output_root = get(ENV, "ATMOSTR_BENCHMARK_OUTPUT_ROOT", "/tmp")
folder_for(tile) = joinpath(output_root, "atmos-f64-tiles-" * tile)
@testset "Float64 PPM launch layouts preserve every saved value" begin
    for tile in ("256","32","32x2"), nt in (6,32), sample in 1:2
        folder = folder_for(tile)
        name = "collab_tracers$(nt)_sample$(sample).nc"
        NCDataset(joinpath(baseline,name)) do a
            NCDataset(joinpath(folder,name)) do b
                @test b.attrib["completed_snapshots"] == 2
                @test Set(keys(a)) == Set(keys(b))
                for key in keys(a)
                    x,y = Array(a[key]),Array(b[key])
                    @test size(x) == size(y)
                    @test all(isfinite,y)
                    @test isequal(x,y)
                end
            end
        end
    end
end
@testset "Full-precision packed states are byte-identical after layout changes" begin
    for tile in ("256","32","32x2"), nt in (6,32)
        name = "collab_tracers$(nt).state"
        before = joinpath(baseline,name)
        after = joinpath(folder_for(tile),name)
        @test filesize(before) == filesize(after)
        @test open(sha256,before) == open(sha256,after)
    end
end
for nt in (6,32), tile in ("baseline","256","32","32x2")
    folder = tile == "baseline" ? baseline : folder_for(tile)
    samples = [TOML.parsefile(joinpath(folder,"collab_result_$(nt)_$(i).toml")) for i in 1:2]
    println("Nt=",nt," tile=",tile," median_seconds=",median(r["wall_seconds"] for r in samples),
            " host_bytes=",median(r["host_allocated_bytes"] for r in samples))
end
