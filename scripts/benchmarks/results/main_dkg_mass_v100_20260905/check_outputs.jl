using NCDatasets, Test, LinearAlgebra, TOML
const before = "/tmp/atmos-split-drift-after"
const after = "/tmp/atmos-dkg-drift-after"

@testset "Dkg full-day totals, complete snapshots, and finite fields" begin
    for case in ("all", "all_float64")
        totals = TOML.parsefile(joinpath(after,case*".toml"))
        before_max, after_max = Float64[], Float64[]
        NCDataset(joinpath(before,case*".nc")) do a
            NCDataset(joinpath(after,case*".nc")) do b
                @test Set(keys(a)) == Set(keys(b))
                @test b.attrib["completed_snapshots"] == 25
                for key in keys(b)
                    x,y = Array(a[key]),Array(b[key])
                    @test size(x) == size(y)
                    @test all(isfinite,y)
                    if endswith(key,"_total_mass")
                        @test first(x) == first(y)
                        @test vec(y) == totals[key]
                        drift = (y .- first(y)) ./ first(y)
                        push!(before_max, maximum(abs, (x .- first(x)) ./ first(x)))
                        push!(after_max, maximum(abs, drift))
                        case == "all_float64" && (@test maximum(abs, drift) < 3e-14)
                        println(case,' ',key," before=",(last(x)-first(x))/first(x)," after=",last(drift)," max_hourly=",maximum(abs,drift))
                    elseif startswith(key,"tracer")
                        println(case,' ',key," extrema=",extrema(y))
                    end
                end
            end
        end
        # Compare worst drift across the same six tracers and all hourly totals.
        # The exploratory 1e-7 Float32 target is reported, not hidden: the
        # version preserving weak transfers misses it (see README and log).
        @test maximum(after_max) < maximum(before_max)
        println(case, " worst_hourly_before=", maximum(before_max),
                " worst_hourly_after=", maximum(after_max),
                " exploratory_1e-7_target_met=", maximum(after_max) < 1e-7)
    end
end

# Cross-precision field comparison supplements the independent column solves.
# The F64 workload uses legacy convection; F32 uses collaborative LU, so this
# is not an isolated diffusion-kernel error estimate or forcing validation.
@testset "Dkg cross-precision final column means" begin
    NCDataset(joinpath(after,"all_float64.nc")) do reference
        for stage in (before,after)
            NCDataset(joinpath(stage,"all.nc")) do ds
                for key in filter(k -> endswith(k,"_column_mean"),collect(keys(ds)))
                    x,y = Array(ds[key]),Array(reference[key])
                    a,b = Float64.(selectdim(x,ndims(x),25)),selectdim(y,ndims(y),25)
                    error = norm(a-b)/norm(b)
                    @test isfinite(error)
                    println(basename(stage),' ',key," relative_L2_to_F64=",error)
                end
            end
        end
    end
end
