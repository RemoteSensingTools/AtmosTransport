# ---------------------------------------------------------------------------
# test_serialization.jl
#
# Serialization round-trip tests for:
#   1. JLD2 save/load (strict identity of intersections + areas)
#   2. On-disk cache (build_regridder hits the cached path on 2nd call)
#   3. ESMF offline-weights NetCDF export (schema presence, attribute
#      tags, frac_a / frac_b ≈ 1 for a full-sphere pair)
# ---------------------------------------------------------------------------

using NCDatasets

@testset "serialization round-trips" begin
    Nx, Ny, Nc = 36, 18, 4
    src = LatLonMesh(Nx = Nx, Ny = Ny)
    dst = CubedSphereMesh(Nc = Nc, convention = GnomonicPanelConvention())

    r = build_regridder(src, dst; normalize = false)

    @testset "JLD2 explicit save/load" begin
        mktempdir() do tmp
            path = joinpath(tmp, "explicit.jld2")
            save_regridder(path, r)
            @test isfile(path)
            r2 = load_regridder(path)
            @test r2.intersections == r.intersections
            @test r2.dst_areas == r.dst_areas
            @test r2.src_areas == r.src_areas
            # Temp arrays are rebuilt fresh, not persisted.
            @test length(r2.dst_temp) == length(r.dst_areas)
            @test length(r2.src_temp) == length(r.src_areas)
        end
    end

    @testset "build_regridder cache_dir hit" begin
        mktempdir() do tmp
            # First call builds
            r1 = build_regridder(src, dst; normalize = false, cache_dir = tmp)
            # Second call should return an equivalent regridder from cache
            r2 = build_regridder(src, dst; normalize = false, cache_dir = tmp)
            @test r1.intersections == r2.intersections
            @test r1.dst_areas == r2.dst_areas
            @test r1.src_areas == r2.src_areas
        end
    end

    @testset "ESMF NetCDF export" begin
        mktempdir() do tmp
            path = joinpath(tmp, "esmf.nc")
            save_esmf_weights(path, r;
                              src_shape = (src.Nx, src.Ny),
                              dst_shape = (dst.Nc, dst.Nc, 6))
            @test isfile(path)
            NCDataset(path) do ds
                # All the canonical ESMF variables must be present.
                for v in ("S", "row", "col", "frac_a", "frac_b", "area_a", "area_b")
                    @test haskey(ds, v)
                end
                # Shapes
                n_s = length(ds["S"][:])
                n_a = length(ds["area_a"][:])
                n_b = length(ds["area_b"][:])
                @test n_a == Nx * Ny
                @test n_b == 6 * Nc * Nc
                @test n_s == length(r.intersections.nzval)

                # Full-sphere pair ⇒ frac_a, frac_b ≈ 1 everywhere
                frac_a = ds["frac_a"][:]
                frac_b = ds["frac_b"][:]
                @test maximum(abs.(frac_a .- 1.0)) < 1e-10
                @test maximum(abs.(frac_b .- 1.0)) < 1e-10

                # Index ranges
                row = ds["row"][:]
                col = ds["col"][:]
                @test minimum(row) >= 1 && maximum(row) <= n_b
                @test minimum(col) >= 1 && maximum(col) <= n_a

                # Provenance attributes
                @test haskey(ds.attrib, "title")
                @test haskey(ds.attrib, "normalization")
                @test ds.attrib["normalization"] == "destarea"
            end
        end
    end
end
