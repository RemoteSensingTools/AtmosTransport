using TOML, NCDatasets, LinearAlgebra, Test
const labels = ("seam_only-Float32-6", "conservative-Float32-6", "conservative-Float64-6")
const dirs = map(label -> "/tmp/atmos-weekly-drift-"*label, labels)
const results = map(d -> TOML.parsefile(joinpath(d,"result.toml")),dirs)
function mass_drift(result, key)
    total = result[key]
    return (total .- first(total)) ./ first(total)
end
@testset "Weekly conserved totals and transported fields" begin
    for tracer in 1:6
        key = "tracer$(lpad(tracer,2,'0'))_total_mass"
        @test first(results[1][key]) == first(results[2][key])
        for result in results
            @test length(result[key]) == 8
            @test all(isfinite,result[key])
        end
        @test maximum(abs,mass_drift(results[3],key)) < 1e-14
    end
    for day in 2:8
        maxima = map(results) do result
            maximum(abs(mass_drift(result,"tracer$(lpad(t,2,'0'))_total_mass")[day]) for t in 1:6)
        end
        @test maxima[2] < maxima[1]
        println("DAY ",day-1," max relative drift: seam_only=",maxima[1],
                " conservative_F32=",maxima[2]," conservative_F64=",maxima[3])
    end
    NCDataset(joinpath(dirs[1],"weekly.nc")) do before
        NCDataset(joinpath(dirs[2],"weekly.nc")) do after
            NCDataset(joinpath(dirs[3],"weekly.nc")) do reference
                @test Set(keys(before)) == Set(keys(after)) == Set(keys(reference))
                for tracer in 1:6
                    key = "tracer$(lpad(tracer,2,'0'))_column_mean"
                    a,b,r = map(ds -> Float64.(Array(ds[key])),(before,after,reference))
                    @test size(a) == size(b) == size(r)
                    @test all(isfinite,b)
                    final = selectdim(r,ndims(r),8)
                    e_before = norm(selectdim(a,ndims(a),8)-final)/norm(final)
                    e_after = norm(selectdim(b,ndims(b),8)-final)/norm(final)
                    @test e_after < e_before
                    println(key," final relative L2 to F64: before=",e_before," after=",e_after,
                            "; extrema_after=",extrema(b))
                end
            end
        end
    end
end
