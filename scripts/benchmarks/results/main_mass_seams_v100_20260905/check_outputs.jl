using Test, NCDatasets, TOML

@testset "Full-day Lin-Rood output completeness and finite values" begin
    for stage in ("before", "after"), case in ("all", "all_float64")
        root = "/tmp/atmos-lr-drift-" * stage
        totals = TOML.parsefile(joinpath(root, case * ".toml"))
        NCDataset(joinpath(root, case * ".nc")) do ds
            fields = filter(k -> startswith(k, "tracer"), collect(keys(ds)))
            @test length(filter(k -> endswith(k, "_total_mass"), fields)) == 6
            for field in fields
                a = Array(ds[field])
                @test all(isfinite, a)
                @test size(a, ndims(a)) == 25
                if haskey(totals, field)
                    @test vec(a) == totals[field]
                else
                    println(stage, ' ', case, ' ', field, " extrema=", extrema(a))
                end
            end
        end
    end
end
