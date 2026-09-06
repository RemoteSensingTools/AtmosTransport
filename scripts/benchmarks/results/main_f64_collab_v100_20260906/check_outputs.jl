using NCDatasets, Test, LinearAlgebra, TOML, Statistics
const folder = "/tmp/atmos-f64-profile-Float64"
rows = Dict[]
@testset "Float64 full-day collaborative versus legacy output" begin
    for nt in (6,32), sample in 1:2
        a = joinpath(folder,"legacy_tracers$(nt)_sample$(sample).nc")
        b = joinpath(folder,"collab_tracers$(nt)_sample$(sample).nc")
        NCDataset(a) do before
            NCDataset(b) do after
                @test before.attrib["completed_snapshots"] == after.attrib["completed_snapshots"] == 2
                @test Set(keys(before)) == Set(keys(after))
                for key in keys(before)
                    x,y = Array(before[key]),Array(after[key])
                    @test size(x) == size(y)
                    @test all(isfinite,y)
                    if endswith(key,"_column_mean")
                        err = norm(y-x)/norm(x)
                        push!(rows,Dict("tracers"=>nt,"sample"=>sample,"variable"=>key,"relative_L2"=>err))
                        @test err < 2e-13
                    elseif endswith(key,"_total_mass")
                        @test first(x) == first(y)
                        drift = abs(last(y)-first(y))/abs(first(y))
                        @test drift < 1e-14
                        @test isapprox(x,y;rtol=2e-14)
                        push!(rows,Dict("tracers"=>nt,"sample"=>sample,"variable"=>key,"final_drift"=>drift))
                    else
                        @test x == y
                    end
                end
            end
        end
    end
end
@testset "Float64 final three-dimensional state, every panel and tracer" begin
    for nt in (6,32)
        shape = (96,96,66,nt)
        a = joinpath(folder,"legacy_tracers$(nt).state")
        b = joinpath(folder,"collab_tracers$(nt).state")
        @test filesize(a) == filesize(b) == 6*prod(shape)*sizeof(Float64)
        open(a) do before
            open(b) do after
                x,y = Array{Float64}(undef,shape), Array{Float64}(undef,shape)
                for panel in 1:6
                    read!(before,x); read!(after,y)
                    @test all(isfinite,y)
                    for tracer in 1:nt
                        xp,yp = view(x,4:93,4:93,:,tracer),view(y,4:93,4:93,:,tracer)
                        err = norm(yp-xp)/norm(xp)
                        @test err < 2e-13
                        push!(rows,Dict("tracers"=>nt,"panel"=>panel,"tracer"=>tracer,
                                       "full_state_relative_L2"=>err))
                    end
                end
            end
        end
    end
end
@testset "Float32 full-day output remains bit-exact" begin
    for sample in 1:2
        a = "/tmp/atmos-initial-reuse-day-after/tracers32_sample$(sample).nc"
        b = "/tmp/atmos-f64-profile-Float32/collab_tracers32_sample$(sample).nc"
        NCDataset(a) do before
            NCDataset(b) do after
                @test after.attrib["completed_snapshots"] == 2
                @test Set(keys(before)) == Set(keys(after))
                for key in keys(before)
                    x,y = Array(before[key]),Array(after[key])
                    @test size(x) == size(y)
                    @test all(isfinite,y)
                    @test isequal(x,y)
                end
            end
        end
    end
end
for nt in (6,32), mode in ("legacy","collab")
    samples = [TOML.parsefile(joinpath(folder,"$(mode)_result_$(nt)_$(i).toml")) for i in 1:2]
    println("PERFORMANCE Nt=",nt," ",mode," median_seconds=",median(r["wall_seconds"] for r in samples),
            " median_host_bytes=",median(r["host_allocated_bytes"] for r in samples))
end
open("/tmp/atmos-f64-profile-comparison.toml","w") do io
    TOML.print(io,Dict("measurements"=>rows))
end
println("MAX_FIELD_RELATIVE_L2 ",maximum(get(r,"relative_L2",0.0) for r in rows))
println("MAX_FINAL_RELATIVE_DRIFT ",maximum(get(r,"final_drift",0.0) for r in rows))

println("MAX_FULL_STATE_RELATIVE_L2 ",maximum(get(r,"full_state_relative_L2",0.0) for r in rows))
