#!/usr/bin/env julia

using Test
using Dates
using NCDatasets

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))

@testset "ERA5 surface reader normalizes raw PBL fields" begin
    mktempdir() do dir
        path = joinpath(dir, "era5_surface_20211201.nc")
        NCDataset(path, "c") do ds
            ds.dim["longitude"] = 4
            ds.dim["latitude"] = 3
            ds.dim["time"] = 24
            defVar(ds, "longitude", [0.0, 90.0, 180.0, 270.0], ("longitude",))
            defVar(ds, "latitude", [90.0, 0.0, -90.0], ("latitude",))
            defVar(ds, "time", collect(0:23), ("time",))

            base = zeros(Float64, 4, 3, 24)
            for t in 1:24, j in 1:3, i in 1:4
                base[i, j, t] = 100i + 10j + t
            end
            defVar(ds, "blh", base .+ 1000, ("longitude", "latitude", "time"))
            defVar(ds, "zust", fill(0.25, 4, 3, 24), ("longitude", "latitude", "time"))
            defVar(ds, "sshf", fill(-360.0, 4, 3, 24), ("longitude", "latitude", "time");
                   attrib = ["units" => "J m**-2"])
            defVar(ds, "t2m", fill(290.0, 4, 3, 24), ("longitude", "latitude", "time"))
        end

        reader = AtmosTransport.Preprocessing.open_era5_surface_reader(dir, Date(2021, 12, 1), 4, 3)
        try
            surface = AtmosTransport.Preprocessing.load_era5_surface_window(reader, 2, Float64)
            @test size(surface.pblh) == (4, 3)
            # Latitude flips S->N, then longitude rolls 0..360 to centered [-180,180).
            raw = [100i + 10j + 2 + 1000 for i in 1:4, j in 1:3]
            expected = circshift(raw[:, end:-1:1], (2, 0))
            @test surface.pblh == expected
            @test all(surface.ustar .== 0.25)
            @test all(surface.hflux .== 0.1)
            @test all(surface.t2m .== 290.0)
        finally
            AtmosTransport.Preprocessing.close_era5_surface_reader(reader)
        end
    end
end

@testset "ERA5 surface reader opens split CDS ZIP payloads" begin
    mktempdir() do dir
        inst_path = joinpath(dir, "data_stream-oper_stepType-instant.nc")
        accum_path = joinpath(dir, "data_stream-oper_stepType-accum.nc")
        zip_path = joinpath(dir, "era5_surface_20211202.nc")

        NCDataset(inst_path, "c") do ds
            ds.dim["longitude"] = 4
            ds.dim["latitude"] = 3
            ds.dim["valid_time"] = 24
            defVar(ds, "longitude", [0.0, 90.0, 180.0, 270.0], ("longitude",))
            defVar(ds, "latitude", [90.0, 0.0, -90.0], ("latitude",))
            defVar(ds, "valid_time", collect(0:23), ("valid_time",))

            base = zeros(Float64, 24, 3, 4)
            for t in 1:24, j in 1:3, i in 1:4
                base[t, j, i] = 100i + 10j + t
            end
            defVar(ds, "blh", base .+ 1000, ("valid_time", "latitude", "longitude"))
            defVar(ds, "u10", fill(3.0, 24, 3, 4), ("valid_time", "latitude", "longitude"))
            defVar(ds, "v10", fill(4.0, 24, 3, 4), ("valid_time", "latitude", "longitude"))
            defVar(ds, "t2m", fill(291.0, 24, 3, 4), ("valid_time", "latitude", "longitude"))
        end

        NCDataset(accum_path, "c") do ds
            ds.dim["longitude"] = 4
            ds.dim["latitude"] = 3
            ds.dim["valid_time"] = 24
            defVar(ds, "longitude", [0.0, 90.0, 180.0, 270.0], ("longitude",))
            defVar(ds, "latitude", [90.0, 0.0, -90.0], ("latitude",))
            defVar(ds, "valid_time", collect(0:23), ("valid_time",))
            defVar(ds, "sshf", fill(-720.0, 24, 3, 4),
                   ("valid_time", "latitude", "longitude");
                   attrib = ["units" => "J m**-2"])
        end

        run(`zip -j -q $zip_path $inst_path $accum_path`)

        reader = AtmosTransport.Preprocessing.open_era5_surface_reader(dir, Date(2021, 12, 2), 4, 3)
        try
            surface = AtmosTransport.Preprocessing.load_era5_surface_window(reader, 3, Float64)
            raw = [100i + 10j + 3 + 1000 for i in 1:4, j in 1:3]
            expected = circshift(raw[:, end:-1:1], (2, 0))
            @test surface.pblh == expected
            @test all(isapprox.(surface.ustar, sqrt(1.2e-3) * 5.0; rtol = 1e-12))
            @test all(surface.hflux .== 0.2)
            @test all(surface.t2m .== 291.0)
        finally
            AtmosTransport.Preprocessing.close_era5_surface_reader(reader)
        end
    end
end
