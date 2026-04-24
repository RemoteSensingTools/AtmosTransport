using Test
using NCDatasets

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

@testset "Output NetCDF snapshots" begin
    @testset "write options validate compression settings" begin
        @test SnapshotWriteOptions(float_type=Float32, deflate_level=1).deflate_level == 1
        @test_throws ArgumentError SnapshotWriteOptions(float_type=Float32, deflate_level=10)
        @test_throws ArgumentError SnapshotWriteOptions(float_type=Int32)
        @test_throws ArgumentError SnapshotWriteOptions(float_type=Float16)
    end

    @testset "writer rejects inconsistent snapshot contracts" begin
        mktempdir() do dir
            mesh = LatLonMesh(; FT=Float64, Nx=4, Ny=3)
            grid = AtmosGrid(mesh, HybridSigmaPressure(Float64[0, 1, 2], Float64[0, 0.5, 1]),
                             CPU(); FT=Float64)
            air = fill(2.0, 4, 3, 2)
            frame = SnapshotFrame(0.0, air, Dict(:co2 => copy(air)), :dry)
            bad_basis = SnapshotFrame(1.0, air, Dict(:co2 => copy(air)), :moist)
            bad_shape = SnapshotFrame(1.0, air, Dict(:co2 => fill(1.0, 4, 3, 1)), :dry)

            @test_throws ArgumentError write_snapshot_netcdf(joinpath(dir, "empty.nc"),
                                                             SnapshotFrame[], grid)
            @test_throws ArgumentError write_snapshot_netcdf(joinpath(dir, "basis.nc"),
                                                             [frame, bad_basis], grid;
                                                             mass_basis=:dry)
            @test_throws DimensionMismatch write_snapshot_netcdf(joinpath(dir, "shape.nc"),
                                                                  [frame, bad_shape], grid;
                                                                  mass_basis=:dry)
        end
    end

    @testset "LL writer carries full fields and derived columns" begin
        mktempdir() do dir
            mesh = LatLonMesh(; FT=Float64, Nx=4, Ny=3)
            grid = AtmosGrid(mesh, HybridSigmaPressure(Float64[0, 1, 2], Float64[0, 0.5, 1]),
                             CPU(); FT=Float64)
            air = fill(2.0, 4, 3, 2)
            rm = similar(air)
            rm[:, :, 1] .= 300e-6 .* air[:, :, 1]
            rm[:, :, 2] .= 600e-6 .* air[:, :, 2]
            frame = SnapshotFrame(0.0, air, Dict(:co2 => rm), :dry)
            path = write_snapshot_netcdf(joinpath(dir, "ll.nc"), [frame], grid;
                                         mass_basis=:dry,
                                         options=SnapshotWriteOptions(float_type=Float64,
                                                                      deflate_level=1))

            NCDataset(path, "r") do ds
                @test ds.attrib["output_contract"] == "AtmosTransport snapshot v2"
                @test ds.attrib["grid_type"] == "latlon"
                @test haskey(ds, "air_mass")
                @test haskey(ds, "air_mass_per_area")
                @test haskey(ds, "column_air_mass_per_area")
                @test haskey(ds, "co2")
                @test haskey(ds, "co2_column_mean")
                @test haskey(ds, "co2_column_mass_per_area")
                @test NCDatasets.deflate(ds["co2"].var) == (true, true, 1)
                @test ds["co2_column_mean"][1, 1, 1] ≈ 450e-6
                @test ds["co2"][1, 1, 2, 1] ≈ 600e-6
            end
        end
    end

    @testset "RG writer preserves native cells plus legacy raster" begin
        mktempdir() do dir
            mesh = ReducedGaussianMesh([-45.0, 45.0], [4, 8]; FT=Float64)
            grid = AtmosGrid(mesh, HybridSigmaPressure(Float64[0, 1, 2], Float64[0, 0.5, 1]),
                             CPU(); FT=Float64)
            air = fill(2.0, ncells(mesh), 2)
            rm = fill(400e-6, ncells(mesh), 2) .* air
            frame = SnapshotFrame(3.0, air, Dict(:co2 => rm), :dry)
            path = write_snapshot_netcdf(joinpath(dir, "rg.nc"), [frame], grid;
                                         mass_basis=:dry,
                                         options=SnapshotWriteOptions(float_type=Float64))

            NCDataset(path, "r") do ds
                @test ds.attrib["grid_type"] == "reduced_gaussian"
                @test haskey(ds, "cell_lon")
                @test haskey(ds, "cell_lat")
                @test haskey(ds, "co2_column_mean_native")
                @test haskey(ds, "co2_column_mean")
                @test size(ds["co2_column_mean_native"]) == (ncells(mesh), 1)
                @test size(ds["co2_column_mean"]) == (maximum(mesh.nlon_per_ring), nrings(mesh), 1)
                @test all(isapprox.(ds["co2_column_mean_native"][:, 1], 400e-6; atol=0))
            end
        end
    end

    @testset "CS writer emits GEOS-native coordinates and panel diagnostics" begin
        mktempdir() do dir
            mesh = CubedSphereMesh(; FT=Float64, Nc=4, convention=GEOSNativePanelConvention())
            grid = AtmosGrid(mesh, HybridSigmaPressure(Float64[0, 1, 2], Float64[0, 0.5, 1]),
                             CPU(); FT=Float64)
            air = ntuple(_ -> fill(2.0, mesh.Nc, mesh.Nc, 2), 6)
            rm = ntuple(_ -> begin
                a = Array{Float64}(undef, mesh.Nc, mesh.Nc, 2)
                a[:, :, 1] .= 300e-6 .* 2.0
                a[:, :, 2] .= 600e-6 .* 2.0
                a
            end, 6)
            frame = SnapshotFrame(6.0, air, Dict(:co2 => rm), :dry)
            path = write_snapshot_netcdf(joinpath(dir, "cs.nc"), [frame], grid;
                                         mass_basis=:dry,
                                         options=SnapshotWriteOptions(float_type=Float64))

            NCDataset(path, "r") do ds
                @test ds.attrib["grid_type"] == "cubed_sphere"
                @test ds.attrib["panel_convention"] == "geos_native"
                @test ds.attrib["longitude_of_central_meridian"] == -10.0
                @test haskey(ds, "lons")
                @test haskey(ds, "corner_lons")
                @test haskey(ds, "cubed_sphere")
                @test ds["cubed_sphere"].attrib["panel_convention"] == "geos_native"
                @test ds["corner_lons"][1, 1, 1] ≈ 305.0 atol=1e-10
                @test ds["corner_lons"][1, 1, 3] ≈ 35.0 atol=1e-10
                @test ds["corner_lats"][1, 1, 3] ≈ rad2deg(asin(1 / sqrt(3))) atol=1e-10
                @test ds["co2_column_mean"][1, 1, 1, 1] ≈ 450e-6
                @test ds["co2"][1, 1, 1, 2, 1] ≈ 600e-6
                @test ds["co2"].attrib["grid_mapping"] == "cubed_sphere"
                @test ds["co2"].attrib["coordinates"] == "lons lats"
            end
        end
    end

    @testset "driven runner delegates snapshot layout to Output" begin
        mktempdir() do dir
            mesh = LatLonMesh(; FT=Float64, Nx=2, Ny=2)
            grid = AtmosGrid(mesh, HybridSigmaPressure(Float64[0, 1], Float64[0, 1]),
                             CPU(); FT=Float64)
            m = fill(2.0, 2, 2, 1)
            windows = [(
                m = m,
                am = zeros(Float64, 3, 2, 1),
                bm = zeros(Float64, 2, 3, 1),
                cm = zeros(Float64, 2, 2, 2),
                ps = fill(95000.0, 2, 2),
            )]
            bin = joinpath(dir, "transport.bin")
            write_transport_binary(bin, grid, windows;
                                   FT=Float64,
                                   dt_met_seconds=3600.0,
                                   half_dt_seconds=1800.0,
                                   steps_per_window=1,
                                   mass_basis=:dry,
                                   source_flux_sampling=:window_start_endpoint,
                                   flux_sampling=:window_constant)
            out = joinpath(dir, "snapshot.nc")
            cfg = Dict{String, Any}(
                "input" => Dict("binary_paths" => [bin]),
                "numerics" => Dict("float_type" => "Float64"),
                "run" => Dict("tracer_name" => "co2"),
                "init" => Dict("kind" => "uniform", "background" => 400e-6),
                "output" => Dict("snapshot_file" => out,
                                  "snapshot_hours" => [0.0, 1.0],
                                  "deflate_level" => 1),
            )
            run_driven_simulation(cfg)
            NCDataset(out, "r") do ds
                @test ds.attrib["output_contract"] == "AtmosTransport snapshot v2"
                @test haskey(ds, "co2")
                @test haskey(ds, "co2_column_mean")
                @test NCDatasets.deflate(ds["co2"].var) == (true, true, 1)
                @test size(ds["co2_column_mean"]) == (2, 2, 2)
                @test all(isapprox.(ds["co2_column_mean"][:, :, :], 400e-6; atol=1e-14))
            end
        end
    end
end
