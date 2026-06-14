# Rolling NVMe input stager: rolling/eviction/disk-bound, disabled passthrough,
# copy-failure fallback, and cross-run reuse. CPU-only, no transport binaries.
using Test

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Models: InputStager, staged_path_for!, cleanup_staging!
const _IS = AtmosTransport.Models.InputStaging   # for the internal name helper

# fake NAS files with distinct content/size per day
function _fake_nas(n)
    dir = mktempdir()
    paths = [joinpath(dir, "geos_transport_2021120$(i)_float32.bin") for i in 1:n]
    for (i, p) in enumerate(paths)
        write(p, rand(UInt8, 400_000 + 7 * i))
    end
    return paths
end

@testset "InputStaging — rolling NVMe stager" begin
    paths = _fake_nas(5)

    @testset "enabled: rolling stage, eviction, disk bound, content" begin
        stage = mktempdir()
        mgr = InputStager(paths, Dict("enabled" => true, "dir" => stage,
                                      "lookahead_days" => 2, "keep_behind_days" => 0))
        for idx in 1:5
            sp = staged_path_for!(mgr, idx)
            @test dirname(sp) == stage
            @test endswith(sp, basename(paths[idx]))
            @test isfile(sp) && read(sp) == read(paths[idx])          # bit-identical copy
            @test count(f -> endswith(f, ".bin"), readdir(stage)) <= 3 # ≤ lookahead+1
            for j in 1:idx-1                                            # keep_behind=0 ⇒ evicted
                @test !isfile(joinpath(stage, _IS._staged_basename(paths[j])))
            end
        end
        cleanup_staging!(mgr)
        @test isempty(filter(f -> endswith(f, ".bin") || endswith(f, ".part"), readdir(stage)))
    end

    @testset "disabled: returns NAS paths, no staging" begin
        stage = mktempdir()
        mgr = InputStager(paths, Dict("enabled" => false))
        for idx in 1:5
            @test staged_path_for!(mgr, idx) == paths[idx]
        end
        @test isempty(readdir(stage))
    end

    @testset "fallback: missing source ⇒ NAS path, no crash" begin
        stage = mktempdir()
        bad = copy(paths); bad[3] = joinpath(dirname(paths[1]), "MISSING.bin")
        mgr = InputStager(bad, Dict("enabled" => true, "dir" => stage, "lookahead_days" => 1))
        @test (@test_logs (:warn,) match_mode = :any staged_path_for!(mgr, 3)) == bad[3]
        cleanup_staging!(mgr)
    end

    @testset "reuse correct + replace stale across runs" begin
        stage = mktempdir()
        cp(paths[1], joinpath(stage, _IS._staged_basename(paths[1])))             # correct: reuse
        write(joinpath(stage, _IS._staged_basename(paths[2])), rand(UInt8, 9))    # wrong size: re-copy
        mgr = InputStager(paths, Dict("enabled" => true, "dir" => stage,
                                      "lookahead_days" => 0, "cleanup_on_exit" => false))
        @test read(staged_path_for!(mgr, 1)) == read(paths[1])
        @test read(staged_path_for!(mgr, 2)) == read(paths[2])
    end

    @testset "config validation" begin
        @test_throws ArgumentError InputStager(paths, Dict("enabled" => true))  # missing dir
    end
end
