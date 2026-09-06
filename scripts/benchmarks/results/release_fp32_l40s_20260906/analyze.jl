using TOML, Statistics, JSON3, NCDatasets, Test, LinearAlgebra
base=@__DIR__
counts=parse.(Int,split(get(ENV,"ATMOSTR_ANALYZE_TRACERS","6,32"),','))
groups=[("l40_baseline",joinpath(base,"baseline-day")),
        ("l40_current",joinpath(base,"current-day")),
        ("v100_current",joinpath(base,"v100","current-day"))]
rows=Dict[]
for (label,folder) in groups, nt in counts
    samples=[TOML.parsefile(joinpath(folder,"w24_tracers$(nt)_sample$(i).toml")) for i in 1:5]
    times=[r["wall_seconds"] for r in samples]
    phases=Dict{String,Any}()
    for phase in ("advection","diffusion","convection","snapshot_capture","snapshot_write",
                  "window_backend_copy","window_load_host","prefetch_fetch_wait")
        values=Float64[]
        for i in 1:5
            file=joinpath(folder,"w24_tracers$(nt)_sample$(i).timings.csv")
            lines=readlines(file)
            header=split(first(lines),',')
            for line in lines[2:end]
                d=Dict(zip(header,split(line,',')))
                get(d,"section","")==phase && push!(values,parse(Float64,d["total_s"]))
            end
        end
        isempty(values) || (phases[phase]=median(values))
    end
    push!(rows,Dict("label"=>label,"tracers"=>nt,"median_seconds"=>median(times),
        "min_seconds"=>minimum(times),"max_seconds"=>maximum(times),"samples_seconds"=>times,
        "median_host_allocated_bytes"=>median(r["host_allocated_bytes"] for r in samples),
        "median_gc_seconds"=>median(r["gc_seconds"] for r in samples),"phases"=>phases))
end
metrics=Dict[]
@testset "Full-day files are complete, finite, repeatable and conservative" begin
    for (label,folder) in groups, nt in counts
        maxdrift=0.0
        reference=Dict{String,Any}()
        for i in 0:5
            NCDataset(joinpath(folder,"w24_tracers$(nt)_sample$(i).nc")) do ds
                @test ds["time"][:]==[0,24]
                if label!="l40_baseline"
                    @test ds.attrib["completed_snapshots"]==2
                end
                for key in keys(ds)
                    x=Array(ds[key])
                    @test all(isfinite,x)
                    if i==0
                        reference[key]=x
                    else
                        @test x==reference[key]
                    end
                    if endswith(key,"_total_mass")
                        drift=abs(last(x)-first(x))/abs(first(x))
                        maxdrift=max(maxdrift,drift)
                        label=="l40_baseline" || @test drift<1e-6
                    end
                end
            end
        end
        push!(metrics,Dict("label"=>label,"tracers"=>nt,"max_final_mass_drift"=>maxdrift))
    end
end
@testset "Current L40S and V100 outputs agree" begin
    for nt in counts
        a=joinpath(base,"current-day","w24_tracers$(nt)_sample1.nc")
        b=joinpath(base,"v100","current-day","w24_tracers$(nt)_sample1.nc")
        maxrel=0.0;maxabs=0.0
        NCDataset(a) do xds
            NCDataset(b) do yds
                @test Set(keys(xds))==Set(keys(yds))
                for key in keys(xds)
                    x,y=Array(xds[key]),Array(yds[key])
                    @test size(x)==size(y)
                    @test isapprox(x,y;rtol=2e-6,atol=0)
                    if endswith(key,"_column_mean")
                        maxrel=max(maxrel,norm(Float64.(x).-y)/max(norm(Float64.(y)),eps()))
                        maxabs=max(maxabs,maximum(abs.(Float64.(x).-y)))
                    end
                end
            end
        end
        push!(metrics,Dict("comparison"=>"current_L40S_vs_V100","tracers"=>nt,"max_column_relative_L2"=>maxrel,"max_column_absolute"=>maxabs))
    end
end
# Main predates conservation fixes: quantify its output difference without
# presenting it as an exact numerical reference for the repaired release.
for nt in counts
    maxrel=0.0
    NCDataset(joinpath(base,"baseline-day","w24_tracers$(nt)_sample1.nc")) do before
        NCDataset(joinpath(base,"current-day","w24_tracers$(nt)_sample1.nc")) do after
            for key in keys(after)
                endswith(key,"_column_mean") || continue
                x,y=Array(before[key]),Array(after[key])
                @test size(x)==size(y)
                maxrel=max(maxrel,norm(Float64.(y).-x)/max(norm(Float64.(x)),eps()))
            end
        end
    end
    push!(metrics,Dict("comparison"=>"current_vs_main","tracers"=>nt,"max_column_relative_L2"=>maxrel))
end
open(joinpath(base,"analysis"*get(ENV,"ATMOSTR_ANALYZE_SUFFIX","")*".json"),"w") do io
    JSON3.pretty(io,Dict("timings"=>rows,"correctness"=>metrics));println(io)
end
for r in rows
    println(r["label"]," tracers=",r["tracers"]," median=",r["median_seconds"]," range=",(r["min_seconds"],r["max_seconds"]))
end
println("DAY_COMPARISON_PASSED tracers=",counts)
