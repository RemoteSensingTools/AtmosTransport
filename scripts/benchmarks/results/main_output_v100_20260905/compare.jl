using NCDatasets, Test
const before="/tmp/atmos-main-real-profile"
const after="/tmp/atmos-main-real-output-after"
comparisons=0
identical=0
max_relative=0.0
mass_drift=0.0
@testset "Real ERA5 output before/after selected streaming port" begin
    for nt in (6,32), sample in (1,2)
        name="tracers$(nt)_sample$(sample).nc"
        NCDataset(joinpath(before,name)) do a
            NCDataset(joinpath(after,name)) do b
                @test Set(keys(a))==Set(keys(b))
                @test b.attrib["completed_snapshots"]==2
                for key in keys(a)
                    x=a[key][:];y=b[key][:]
                    global comparisons+=1
                    if isequal(x,y)
                        global identical+=1
                        @test true
                    else
                        @test all(isfinite,x) && all(isfinite,y)
                        delta=maximum(abs.(Float64.(x).-Float64.(y)))
                        scale=max(maximum(abs.(Float64.(x))),eps(Float64))
                        global max_relative=max(max_relative,delta/scale)
                        @test delta/scale < (endswith(key,"_total_mass") ? 1e-12 : 4eps(Float32))
                    end
                    if endswith(key,"_total_mass")
                        drift=abs(y[2]-y[1])/abs(y[1])
                        global mass_drift=max(mass_drift,drift)
                        @test drift < 1e-5
                    end
                end
            end
        end
    end
end
println("Arrays compared: ",comparisons,"; exactly equal: ",identical)
println("Maximum scaled array difference: ",max_relative)
println("Maximum relative mass drift: ",mass_drift)
