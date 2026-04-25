using Test
using NCDatasets
using AtmosTransport

const PP = AtmosTransport.Preprocessing

function write_time_first_qv(path)
    NCDataset(path, "c") do ds
        defDim(ds, "time", 4)
        defDim(ds, "hybrid", 2)
        defDim(ds, "latitude", 3)
        defDim(ds, "longitude", 4)
        lat = defVar(ds, "latitude", Float64, ("latitude",))
        lon = defVar(ds, "longitude", Float64, ("longitude",))
        lat[:] = [90.0, 0.0, -90.0]
        lon[:] = [-180.0, -90.0, 0.0, 90.0]
        q = defVar(ds, "q", Float32, ("time", "hybrid", "latitude", "longitude"))
        for t in 1:4, k in 1:2, j in 1:3, i in 1:4
            q[t, k, j, i] = Float32(1000t + 100k + 10j + i)
        end
    end
end

function write_lon_first_qv(path)
    NCDataset(path, "c") do ds
        defDim(ds, "longitude", 4)
        defDim(ds, "latitude", 3)
        defDim(ds, "hybrid", 2)
        defDim(ds, "time", 4)
        lat = defVar(ds, "latitude", Float64, ("latitude",))
        lon = defVar(ds, "longitude", Float64, ("longitude",))
        lat[:] = [90.0, 0.0, -90.0]
        lon[:] = [-180.0, -90.0, 0.0, 90.0]
        q = defVar(ds, "q", Float32, ("longitude", "latitude", "hybrid", "time"))
        for i in 1:4, j in 1:3, k in 1:2, t in 1:4
            q[i, j, k, t] = Float32(1000t + 100k + 10j + i)
        end
    end
end

@testset "daily qv preload matches hourly reader" begin
    mktempdir() do dir
        for (name, writer) in (("time_first.nc", write_time_first_qv),
                               ("lon_first.nc", write_lon_first_qv))
            path = joinpath(dir, name)
            writer(path)

            daily = PP.read_daily_qv_from_thermo(path, 4, 3, 2; FT=Float64, time_block=2)
            @test size(daily) == (4, 3, 2, 4)
            for hour in 1:4
                hourly = PP.read_qv_from_thermo(path, hour, 4, 3, 2; FT=Float64)
                @test @views daily[:, :, :, hour] == hourly
            end
        end
    end
end

@testset "spectral coefficient cache round trip" begin
    mktempdir() do dir
        spec = (
            hours = [0, 1],
            lnsp_all = Dict(0 => ComplexF64[1 0; 2 3], 1 => ComplexF64[4 0; 5 6]),
            vo_by_hour = Dict(0 => reshape(ComplexF64[1, 2, 3, 4], 2, 2, 1),
                              1 => reshape(ComplexF64[5, 6, 7, 8], 2, 2, 1)),
            d_by_hour = Dict(0 => reshape(ComplexF64[9, 10, 11, 12], 2, 2, 1),
                             1 => reshape(ComplexF64[13, 14, 15, 16], 2, 2, 1)),
            T = 1,
            n_times = 2,
        )

        path = joinpath(dir, "spectral_cache.jld2")
        PP._write_spectral_day_cache(path, spec)
        loaded = PP._load_spectral_day_cache(path)

        @test loaded.hours == spec.hours
        @test loaded.T == spec.T
        @test loaded.n_times == spec.n_times
        @test loaded.lnsp_all[1] == spec.lnsp_all[1]
        @test loaded.vo_by_hour[0] == spec.vo_by_hour[0]
        @test loaded.d_by_hour[1] == spec.d_by_hour[1]
    end
end
