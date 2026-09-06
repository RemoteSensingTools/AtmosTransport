using Test, NCDatasets, TOML, LinearAlgebra

@testset "Lin-Rood 6/32-tracer V100 profile outputs" begin
    for nt in (6, 32), sample in 1:2
        before = "/tmp/atmos-lr-seams-profile-before/tracers$(nt)_sample$(sample).nc"
        after = "/tmp/atmos-lr-seams-profile-after/tracers$(nt)_sample$(sample).nc"
        NCDataset(before) do a
            NCDataset(after) do b
                fields = filter(k -> startswith(k, "tracer"), collect(keys(a)))
                @test Set(keys(a)) == Set(keys(b))
                @test count(k -> endswith(k, "_total_mass"), fields) == nt
                for field in fields
                    x, y = Array(a[field]), Array(b[field])
                    @test size(x) == size(y)
                    @test all(isfinite, x) && all(isfinite, y)
                    @test size(y, ndims(y)) == 2
                    @test selectdim(x, ndims(x), 1) == selectdim(y, ndims(y), 1)
                    if endswith(field, "_total_mass")
                        @test abs((y[end] - y[1]) / y[1]) < 1e-6
                    end
                end
            end
        end
    end
end
