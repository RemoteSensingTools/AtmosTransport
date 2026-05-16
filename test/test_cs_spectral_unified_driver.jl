#!/usr/bin/env julia
# Plan 41 P3f - CS spectral unified-driver opt-in parity.

using Test
using Dates

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Preprocessing: build_target_geometry, process_day

const Pre = AtmosTransport.Preprocessing

function _write_fake_cs_spectral_cache!(spectral_dir::String,
                                        cache_dir::String,
                                        date::Date)
    mkpath(spectral_dir)
    mkpath(cache_dir)
    ds = Dates.format(date, "yyyymmdd")
    vo_d_path = joinpath(spectral_dir, "era5_spectral_$(ds)_vo_d.gb")
    lnsp_path = joinpath(spectral_dir, "era5_spectral_$(ds)_lnsp.gb")
    touch(vo_d_path)
    touch(lnsp_path)

    lnsp = fill(complex(log(100000.0), 0.0), 1, 1)
    zero_level = zeros(ComplexF64, 1, 1, 1)
    spec = (
        hours = [0, 1],
        lnsp_all = Dict(0 => copy(lnsp), 1 => copy(lnsp)),
        vo_by_hour = Dict(0 => copy(zero_level), 1 => copy(zero_level)),
        d_by_hour = Dict(0 => copy(zero_level), 1 => copy(zero_level)),
        T = 0,
        n_times = 2,
    )
    path = Pre.spectral_day_cache_path(cache_dir, vo_d_path, lnsp_path;
                                       T_target = 0)
    Pre._write_spectral_day_cache(path, spec)
    return path
end

function _cs_test_vertical(::Type{FT}) where FT
    vc = AtmosTransport.HybridSigmaPressure(FT[0, 0], FT[0, 1])
    return (
        Nz_native = 1,
        Nz = 1,
        level_range = 1:1,
        ab = (dA = FT[0], dB = FT[1], b_ifc = FT[0, 1]),
        merge_map = [1],
        merged_vc = vc,
    )
end

function _cs_test_settings(::Type{FT}, spectral_dir, cache_dir, out_dir) where FT
    return (
        output_float_type = FT,
        spectral_dir = spectral_dir,
        spectral_cache_dir = cache_dir,
        T_target = 0,
        min_dp = 0.0,
        include_qv = false,
        mass_basis = :moist,
        mass_fix_enable = false,
        target_ps_dry_pa = 98726.0,
        qv_global_climatology = 0.0,
        thermo_dir = dirname(out_dir),
        half_dt = 450.0,
        met_interval = 3600.0,
        dt = 900.0,
        out_dir = out_dir,
    )
end

@testset "CS spectral unified driver emits stable bytes" begin
    mktempdir() do tmp
        FT = Float64
        date = Date(2021, 12, 1)
        spectral_dir = joinpath(tmp, "spectral")
        cache_dir = joinpath(tmp, "cache")
        regrid_cache_dir = joinpath(tmp, "regrid_cache")
        _write_fake_cs_spectral_cache!(spectral_dir, cache_dir, date)

        grid = build_target_geometry(
            Val(:cubed_sphere),
            Dict{String, Any}(
                "Nc" => 2,
                "staging_nlon" => 4,
                "staging_nlat" => 3,
                "regridder_cache_dir" => regrid_cache_dir,
            ),
            FT)
        vertical = _cs_test_vertical(FT)
        default_settings = _cs_test_settings(FT, spectral_dir, cache_dir,
                                             joinpath(tmp, "default"))
        unified_settings = _cs_test_settings(FT, spectral_dir, cache_dir,
                                             joinpath(tmp, "unified"))

        default_path = process_day(date, grid, default_settings, vertical;
                                   positivity_cfl_limit = 0.95)
        unified_path = process_day(date, grid, unified_settings, vertical;
                                   positivity_cfl_limit = 0.95)

        @test isfile(default_path)
        @test isfile(unified_path)
        @test filesize(default_path) == filesize(unified_path)
        @test read(unified_path) == read(default_path)
    end
end
