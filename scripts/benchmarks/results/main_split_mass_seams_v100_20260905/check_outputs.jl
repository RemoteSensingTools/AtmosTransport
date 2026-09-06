using Test, NCDatasets, TOML

@testset "Split PPM hourly output completeness and finite fields" begin
    root = "/tmp/atmos-split-drift-after"
    for case in ("all", "all_float64")
        totals = TOML.parsefile(joinpath(root, case * ".toml"))
        NCDataset(joinpath(root, case * ".nc")) do ds
            @test ds.attrib["completed_snapshots"] == 25
            fields = filter(k -> startswith(k, "tracer"), collect(keys(ds)))
            @test length(filter(k -> endswith(k, "_total_mass"), fields)) == 6
            for field in fields
                a = Array(ds[field])
                @test all(isfinite, a)
                @test size(a, ndims(a)) == 25
                if haskey(totals, field)
                    @test vec(a) == totals[field]
                else
                    println(case, ' ', field, " extrema=", extrema(a))
                end
            end
        end
    end
end
