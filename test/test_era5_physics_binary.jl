#!/usr/bin/env julia
"""
Plan 24 Commit 2 — ERA5 physics BIN round-trip tests.

Builds a pair of minimal synthetic NCs (convection in the
forecast-stream `(lon, lat, hybrid, step, time)` layout; thermo in
the analysis `(lon, lat, hybrid, time)` layout), converts to BIN,
reads via mmap, verifies:

  (A) byte-exact roundtrip of the data.
  (B) latitude flip works (ERA5 N→S → AtmosTransport S→N).
  (C) calendar-day hour splicing (prev-day hours 00–06 + today's
      hours 07–23) is correct.
  (D) missing-file errors are clear and actionable.
  (E) header round-trip preserves metadata (date, provenance).

Real-data path (behind `--all`): convert one day from
`~/data/AtmosTransport/met/era5/0.5x0.5/physics/` and spot-check
value ranges against known ERA5 magnitudes.
"""

using Test
using Dates
using NCDatasets

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

using .AtmosTransport.Preprocessing: convert_era5_physics_nc_to_bin,
                                       open_era5_physics_binary,
                                       close_era5_physics_binary,
                                       get_era5_physics_field,
                                       ERA5PhysicsBinaryReader,
                                       ERA5_PHYSICS_BINARY_VARS

const RUN_ALL = "--all" in ARGS

# ─────────────────────────────────────────────────────────────────────────
# Helpers to build synthetic NCs in the exact ERA5 format
# ─────────────────────────────────────────────────────────────────────────

"""
    _make_convection_nc(path, date; Nlon, Nlat, Nlev, lat_N_to_S=true,
                                    base_values=(udmf=0.01, ddmf=-0.005,
                                                 udrf=1e-5, ddrf=5e-6))

Write a minimal ERA5-convection-shaped NC to `path` for `date`.
Values are constant per variable across all hours, levels, and
grid cells (easy to roundtrip-check). `lat_N_to_S=true` means the
latitude dim runs 90 → -90 (ERA5 default).
"""
function _make_convection_nc(path::String, date::Date;
                              Nlon::Int = 4, Nlat::Int = 3, Nlev::Int = 5,
                              lat_N_to_S::Bool = true,
                              base_values = (udmf = 0.01f0, ddmf = -0.005f0,
                                              udrf = 1.0f-5, ddrf = 5.0f-6))
    ds = NCDataset(path, "c")
    try
        defDim(ds, "longitude", Nlon)
        defDim(ds, "latitude",  Nlat)
        defDim(ds, "hybrid",    Nlev)
        defDim(ds, "step",      12)
        defDim(ds, "time",       2)

        lons = collect(range(0.0, step = 360.0 / Nlon, length = Nlon))
        lats = lat_N_to_S ?
                collect(range(90.0, stop = -90.0, length = Nlat)) :
                collect(range(-90.0, stop = 90.0, length = Nlat))
        hyb  = collect(1.0:Nlev)

        defVar(ds, "longitude", lons, ("longitude",))
        defVar(ds, "latitude",  lats, ("latitude",))
        defVar(ds, "hybrid",    hyb,  ("hybrid",))
        # Base times: 06:00, 18:00 UTC of `date`.
        defVar(ds, "time", Float64[0.0, 12.0], ("time",))
        # `valid_time[step, time]`: step s + (t-1)*12 = 1..24 hours
        # from the base time. For base 06 t=1, valid_time[s, 1] =
        # date + hour(7..18). Convert to days-since-epoch Float64.
        vt = Array{Float64, 2}(undef, 12, 2)
        for t in 1:2, s in 1:12
            # Base 06 (t=1) + step s → hour 6+s. Base 18 (t=2) + step s → hour 18+s.
            base_hour = t == 1 ? 6 : 18
            hour_offset = base_hour + s
            vt[s, t] = Dates.datetime2rata(DateTime(date) + Hour(hour_offset)) - Dates.datetime2rata(DateTime(date))
        end
        defVar(ds, "valid_time", vt, ("step", "time"))

        Nstep, Ntime = 12, 2
        for (key, val) in pairs(base_values)
            arr = fill(Float32(val), Nlon, Nlat, Nlev, Nstep, Ntime)
            defVar(ds, String(key), arr,
                   ("longitude", "latitude", "hybrid", "step", "time"))
        end
    finally
        close(ds)
    end
    return path
end

"""
    _make_thermo_nc(path, date; Nlon, Nlat, Nlev, lat_N_to_S=true,
                                T_value=240, Q_value=0.002)

Write a minimal ERA5-thermo-shaped NC. Calendar-day aligned (time=24).
"""
function _make_thermo_nc(path::String, ::Date;
                          Nlon::Int = 4, Nlat::Int = 3, Nlev::Int = 5,
                          lat_N_to_S::Bool = true,
                          T_value::Real = 240.0,
                          Q_value::Real = 0.002)
    ds = NCDataset(path, "c")
    try
        defDim(ds, "longitude", Nlon)
        defDim(ds, "latitude",  Nlat)
        defDim(ds, "hybrid",    Nlev)
        defDim(ds, "time",      24)

        lons = collect(range(0.0, step = 360.0 / Nlon, length = Nlon))
        lats = lat_N_to_S ?
                collect(range(90.0, stop = -90.0, length = Nlat)) :
                collect(range(-90.0, stop = 90.0, length = Nlat))
        hyb  = collect(1.0:Nlev)

        defVar(ds, "longitude", lons, ("longitude",))
        defVar(ds, "latitude",  lats, ("latitude",))
        defVar(ds, "hybrid",    hyb,  ("hybrid",))
        defVar(ds, "time", collect(0.0:23.0), ("time",))
        defVar(ds, "valid_time", collect(0.0:23.0), ("time",))

        t_arr = fill(Float32(T_value), Nlon, Nlat, Nlev, 24)
        q_arr = fill(Float32(Q_value), Nlon, Nlat, Nlev, 24)
        defVar(ds, "t", t_arr, ("longitude", "latitude", "hybrid", "time"))
        defVar(ds, "q", q_arr, ("longitude", "latitude", "hybrid", "time"))
    finally
        close(ds)
    end
    return path
end

# ─────────────────────────────────────────────────────────────────────────
# Roundtrip tests (core, no real data)
# ─────────────────────────────────────────────────────────────────────────

@testset "plan 24 Commit 2: BIN roundtrip on synthetic NCs" begin
    mktempdir() do tmp
        nc_dir  = joinpath(tmp, "nc")
        bin_dir = joinpath(tmp, "bin")
        mkpath(nc_dir)

        target_date = Date(2021, 12, 1)
        prev_date   = Date(2021, 11, 30)

        # Distinct values so we can tell which NC contributes which hour.
        prev_conv = (udmf = 0.011f0, ddmf = -0.006f0,
                      udrf = 2.0f-5, ddrf = 6.0f-6)
        today_conv = (udmf = 0.022f0, ddmf = -0.012f0,
                       udrf = 4.0f-5, ddrf = 1.2f-5)

        _make_convection_nc(
            joinpath(nc_dir, "era5_convection_$(Dates.format(prev_date, "yyyymmdd")).nc"),
            prev_date; base_values = prev_conv)
        _make_convection_nc(
            joinpath(nc_dir, "era5_convection_$(Dates.format(target_date, "yyyymmdd")).nc"),
            target_date; base_values = today_conv)
        _make_thermo_nc(
            joinpath(nc_dir, "era5_thermo_ml_$(Dates.format(target_date, "yyyymmdd")).nc"),
            target_date; T_value = 250.0, Q_value = 0.003)

        bin_path = convert_era5_physics_nc_to_bin(
            nc_dir, bin_dir, target_date; verbose = false)

        @test isfile(bin_path)
        @test occursin("/2021/era5_physics_20211201.bin", bin_path)

        reader = open_era5_physics_binary(bin_dir, target_date)
        try
            @test reader.header.format_version == 1
            @test reader.header.date           == target_date
            @test reader.header.Nlon           == 4
            @test reader.header.Nlat           == 3
            @test reader.header.Nlev           == 5
            @test reader.header.Nt             == 24
            @test reader.header.latitude_convention == :S_to_N

            # Provenance present.
            @test haskey(reader.header.provenance, "source_convection_today")
            @test haskey(reader.header.provenance, "source_convection_prev")
            @test haskey(reader.header.provenance, "source_thermo")

            udmf = get_era5_physics_field(reader, :udmf)
            ddmf = get_era5_physics_field(reader, :ddmf)
            t_fld = get_era5_physics_field(reader, :t)
            q_fld = get_era5_physics_field(reader, :q)

            @test size(udmf) == (4, 3, 5, 24)
            @test size(t_fld) == (4, 3, 5, 24)

            # Hours 00-06 come from prev-day NC (udmf=0.011).
            # Hours 07-23 come from today's NC (udmf=0.022).
            @test all(udmf[:, :, :, 1:7]   .≈ prev_conv.udmf)
            @test all(udmf[:, :, :, 8:24]  .≈ today_conv.udmf)
            @test all(ddmf[:, :, :, 1:7]   .≈ prev_conv.ddmf)
            @test all(ddmf[:, :, :, 8:24]  .≈ today_conv.ddmf)

            # Thermo constant across all hours (calendar-day aligned).
            @test all(t_fld .≈ 250.0f0)
            @test all(q_fld .≈ 0.003f0)
        finally
            close_era5_physics_binary(reader)
        end
    end
end

@testset "plan 24 Commit 2: latitude flip N→S to S→N" begin
    mktempdir() do tmp
        nc_dir  = joinpath(tmp, "nc")
        bin_dir = joinpath(tmp, "bin")
        mkpath(nc_dir)

        target_date = Date(2021, 12, 1)
        prev_date   = Date(2021, 11, 30)

        _make_convection_nc(
            joinpath(nc_dir, "era5_convection_$(Dates.format(prev_date, "yyyymmdd")).nc"),
            prev_date; Nlat = 5, lat_N_to_S = true)
        _make_convection_nc(
            joinpath(nc_dir, "era5_convection_$(Dates.format(target_date, "yyyymmdd")).nc"),
            target_date; Nlat = 5, lat_N_to_S = true)
        _make_thermo_nc(
            joinpath(nc_dir, "era5_thermo_ml_$(Dates.format(target_date, "yyyymmdd")).nc"),
            target_date; Nlat = 5, lat_N_to_S = true)

        bin_path = convert_era5_physics_nc_to_bin(
            nc_dir, bin_dir, target_date; verbose = false)

        reader = open_era5_physics_binary(bin_dir, target_date)
        try
            # After flip, latitude range should run S→N (-90 → 90).
            lat_range = reader.header.latitude_range
            @test lat_range[1] < lat_range[2]   # first < last → S→N
            @test lat_range[1] ≈ -90.0
            @test lat_range[2] ≈  90.0
        finally
            close_era5_physics_binary(reader)
        end
    end
end

@testset "plan 24 Commit 2: missing-file errors name the fix" begin
    mktempdir() do tmp
        nc_dir  = joinpath(tmp, "nc")
        bin_dir = joinpath(tmp, "bin")
        mkpath(nc_dir)

        # No NCs present → missing convection (today) error.
        err = try
            convert_era5_physics_nc_to_bin(nc_dir, bin_dir, Date(2021, 12, 1);
                                            verbose = false)
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("missing convection NC", err.msg)
        @test occursin("download_era5_physics.py", err.msg)

        # Today but no prev-day NC → prev-day error.
        _make_convection_nc(joinpath(nc_dir, "era5_convection_20211201.nc"),
                             Date(2021, 12, 1))
        _make_thermo_nc(joinpath(nc_dir, "era5_thermo_ml_20211201.nc"),
                         Date(2021, 12, 1))
        err = try
            convert_era5_physics_nc_to_bin(nc_dir, bin_dir, Date(2021, 12, 1);
                                            verbose = false)
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("missing previous-day", err.msg)
    end
end

@testset "plan 24 Commit 2: open on missing BIN errors clearly" begin
    mktempdir() do tmp
        err = try
            open_era5_physics_binary(tmp, Date(2021, 12, 1))
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("missing", err.msg)
        @test occursin("convert_era5_physics_nc_to_bin.jl", err.msg)
    end
end

@testset "plan 24 Commit 2: idempotent write (skip if exists)" begin
    mktempdir() do tmp
        nc_dir  = joinpath(tmp, "nc")
        bin_dir = joinpath(tmp, "bin")
        mkpath(nc_dir)

        target_date = Date(2021, 12, 1)
        prev_date   = Date(2021, 11, 30)
        _make_convection_nc(joinpath(nc_dir, "era5_convection_20211130.nc"),
                             prev_date)
        _make_convection_nc(joinpath(nc_dir, "era5_convection_20211201.nc"),
                             target_date)
        _make_thermo_nc(joinpath(nc_dir, "era5_thermo_ml_20211201.nc"),
                         target_date)

        bin_path = convert_era5_physics_nc_to_bin(
            nc_dir, bin_dir, target_date; verbose = false)
        @test isfile(bin_path)

        mtime_before = mtime(bin_path)
        sleep(0.05)   # mtime resolution floor

        # Running again with force_rewrite=false skips writing.
        bin_path2 = convert_era5_physics_nc_to_bin(
            nc_dir, bin_dir, target_date; verbose = false, force_rewrite = false)
        @test bin_path2 == bin_path
        @test mtime(bin_path) == mtime_before   # not rewritten

        # force_rewrite=true rewrites.
        bin_path3 = convert_era5_physics_nc_to_bin(
            nc_dir, bin_dir, target_date; verbose = false, force_rewrite = true)
        @test mtime(bin_path3) > mtime_before
    end
end

@testset "plan 24 Commit 2: get_era5_physics_field returns views (no alloc)" begin
    mktempdir() do tmp
        nc_dir  = joinpath(tmp, "nc")
        bin_dir = joinpath(tmp, "bin")
        mkpath(nc_dir)

        target_date = Date(2021, 12, 1)
        prev_date   = Date(2021, 11, 30)
        _make_convection_nc(joinpath(nc_dir, "era5_convection_20211130.nc"),
                             prev_date)
        _make_convection_nc(joinpath(nc_dir, "era5_convection_20211201.nc"),
                             target_date)
        _make_thermo_nc(joinpath(nc_dir, "era5_thermo_ml_20211201.nc"),
                         target_date)

        convert_era5_physics_nc_to_bin(nc_dir, bin_dir, target_date;
                                        verbose = false)
        reader = open_era5_physics_binary(bin_dir, target_date)
        try
            # Warm up JIT.
            for v in ERA5_PHYSICS_BINARY_VARS
                get_era5_physics_field(reader, v)
            end
            # Second call: zero allocation.
            alloc = @allocated get_era5_physics_field(reader, :udmf)
            @test alloc == 0
        finally
            close_era5_physics_binary(reader)
        end
    end
end

@testset "plan 24 Commit 2: unknown var symbol errors cleanly" begin
    mktempdir() do tmp
        nc_dir  = joinpath(tmp, "nc")
        bin_dir = joinpath(tmp, "bin")
        mkpath(nc_dir)

        target_date = Date(2021, 12, 1)
        prev_date   = Date(2021, 11, 30)
        _make_convection_nc(joinpath(nc_dir, "era5_convection_20211130.nc"),
                             prev_date)
        _make_convection_nc(joinpath(nc_dir, "era5_convection_20211201.nc"),
                             target_date)
        _make_thermo_nc(joinpath(nc_dir, "era5_thermo_ml_20211201.nc"),
                         target_date)

        convert_era5_physics_nc_to_bin(nc_dir, bin_dir, target_date;
                                        verbose = false)
        reader = open_era5_physics_binary(bin_dir, target_date)
        try
            @test_throws ArgumentError get_era5_physics_field(reader, :nonexistent)
        finally
            close_era5_physics_binary(reader)
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────
# Real-data test (behind --all)
# ─────────────────────────────────────────────────────────────────────────

if RUN_ALL
    @testset "plan 24 Commit 2: real ERA5 physics roundtrip (--all)" begin
        nc_dir = expanduser("~/data/AtmosTransport/met/era5/0.5x0.5/physics")
        bin_dir = mktempdir()
        target_date = Date(2021, 12, 2)   # Dec 2 has both day-1 and day-2 NCs

        if !isfile(joinpath(nc_dir, "era5_convection_20211201.nc"))
            @warn "ERA5 convection NC missing; skipping --all test"
            return
        end

        bin_path = convert_era5_physics_nc_to_bin(
            nc_dir, bin_dir, target_date; verbose = true)
        @test isfile(bin_path)
        bin_size_gb = filesize(bin_path) / 1e9
        @test 10.0 < bin_size_gb < 25.0   # sanity: ~18 GB expected for 0.5° × 137L

        reader = open_era5_physics_binary(bin_dir, target_date)
        try
            @test reader.header.Nlon == 720
            @test reader.header.Nlat == 361
            @test reader.header.Nlev == 137
            @test reader.header.Nt   == 24

            udmf = get_era5_physics_field(reader, :udmf)
            t_fld = get_era5_physics_field(reader, :t)

            # ERA5 UDMF is typically in [0, 0.5] kg/m²/s for deep
            # convection; zero in most cells.
            @test 0.0f0 <= minimum(udmf)
            @test maximum(udmf) < 1.0f0
            # T should be in the 180 K - 310 K range across the atmosphere.
            @test 150.0f0 < minimum(t_fld) < 220.0f0
            @test 250.0f0 < maximum(t_fld) < 320.0f0
        finally
            close_era5_physics_binary(reader)
        end
    end
end
