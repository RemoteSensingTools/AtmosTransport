#!/usr/bin/env julia
# Thin CLI wrapper plus target implementation for the config-driven ERA5
# spectral -> reduced-Gaussian transport-binary v2 path.
#
# Users normally run this script directly. Contributors can reuse the stable
# target API via `build_spectral_transport_binary_v2_target` and
# `run_spectral_transport_binary_v2_preprocessor`.

using Dates
using Logging
using Printf
using TOML

include(joinpath(@__DIR__, "preprocess_spectral_v4_binary.jl"))
include(joinpath(@__DIR__, "spectral_transport_binary_v2_dispatch.jl"))
include(joinpath(PREPROCESS_SPECTRAL_V4_DIR, "reduced_transport_helpers.jl"))
using .AtmosTransportV2: AtmosGrid, CPU, write_transport_binary,
    ncells, nfaces, nrings, cell_area, face_cells

function output_reduced_transport_binary_v2_path(date::Date,
                                                 out_dir::String,
                                                 gaussian_number::Int,
                                                 min_dp::Float64,
                                                 ::Type{FT}) where FT
    dp_tag = @sprintf("merged%dPa", round(Int, min_dp))
    ft_tag = FT == Float64 ? "float64" : "float32"
    date_str = Dates.format(date, "yyyymmdd")
    return joinpath(out_dir, "era5_transport_v2_rgN$(gaussian_number)_$(date_str)_$(dp_tag)_$(ft_tag).bin")
end

function build_reduced_transport_binary_grid(grid::ReducedGaussianTargetGeometry, vertical, ::Type{FT}) where FT
    vc = AtmosTransportV2.HybridSigmaPressure(FT.(vertical.merged_vc.A), FT.(vertical.merged_vc.B))
    return AtmosGrid(grid.mesh, vc, CPU(); FT=FT)
end

function build_reduced_transport_windows(storage::ReducedWindowStorage{FT}) where FT
    Nt = length(storage.all_m)
    return [
        (; m = storage.all_m[win_idx],
           hflux = storage.all_hflux[win_idx],
           cm = storage.all_cm[win_idx],
           ps = storage.all_ps[win_idx])
        for win_idx in 1:Nt
    ]
end

function write_reduced_transport_binary_v2_from_storage!(bin_path::AbstractString,
                                                          grid::ReducedGaussianTargetGeometry,
                                                          vertical,
                                                          storage::ReducedWindowStorage{FT},
                                                          settings) where FT
    windows = build_reduced_transport_windows(storage)
    transport_grid = build_reduced_transport_binary_grid(grid, vertical, FT)
    steps_per_met = exact_steps_per_window(settings.met_interval, settings.dt)
    write_transport_binary(bin_path, transport_grid, windows;
                           FT=FT,
                           dt_met_seconds=settings.met_interval,
                           half_dt_seconds=settings.half_dt,
                           steps_per_window=steps_per_met,
                           source_flux_sampling=:window_start_endpoint,
                           air_mass_sampling=:window_start_endpoint,
                           flux_sampling=:window_constant,
                           flux_kind=:substep_mass_amount,
                           humidity_sampling=:none,
                           delta_semantics=:none,
                           mass_basis=settings.mass_basis,
                           extra_header=Dict(
                               "gaussian_number" => grid.gaussian_number,
                               "geometry_source_grib" => grid.geometry_source_grib,
                           ),
                           threaded=true)
    return nothing
end

function process_day_reduced_transport_binary_v2(date::Date,
                                                 grid::ReducedGaussianTargetGeometry,
                                                 settings,
                                                 vertical)
    settings.mass_basis == :moist || error("ReducedGaussian transport-binary v2 currently supports moist basis only")
    settings.include_qv && error("ReducedGaussian transport-binary v2 does not write qv endpoints yet")

    FT = settings.output_float_type
    date_str = Dates.format(date, "yyyymmdd")
    vo_d_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_vo_d.gb")
    lnsp_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_lnsp.gb")
    if !isfile(vo_d_path) || !isfile(lnsp_path)
        @warn "Missing GRIB files for $date_str, skipping"
        return nothing
    end

    mkpath(settings.out_dir)
    bin_path = output_reduced_transport_binary_v2_path(date, settings.out_dir, grid.gaussian_number, settings.min_dp, FT)
    if isfile(bin_path)
        @info "  SKIP (exists): $(basename(bin_path))"
        return (path = bin_path, report = nothing)
    end

    t_day = time()
    @info "  Reading spectral data for $date_str..."
    spec = read_day_spectral_streaming(vo_d_path, lnsp_path; T_target=settings.T_target)
    @info @sprintf("  Spectral data read: T=%d, %d hours", spec.T, length(spec.lnsp_all))

    transform = allocate_reduced_transform_workspace(grid, settings.T_target, vertical.Nz_native)
    merged = allocate_reduced_merge_workspace(grid, vertical.Nz_native, vertical.Nz, FT)
    storage = allocate_reduced_window_storage(length(spec.lnsp_all), FT, grid, vertical.Nz)

    hours = spec.hours
    @info "  Computing spectral -> native reduced Gaussian for $(length(hours)) windows..."
    for (win_idx, hour) in enumerate(hours)
        spectral_to_native_fields!(transform,
                                   spec.lnsp_all[hour],
                                   spec.vo_by_hour[hour],
                                   spec.d_by_hour[hour],
                                   settings.T_target,
                                   vertical.level_range,
                                   vertical.ab,
                                   grid,
                                   settings.half_dt)
        merge_reduced_window!(merged, transform, vertical)
        store_reduced_window!(storage, merged, transform.sp, win_idx)
        should_log_window(win_idx, length(hours)) &&
            @info(@sprintf("    Window %d/%d (hour %02d)", win_idx, length(hours), hour))
    end

    steps_per_met = exact_steps_per_window(settings.met_interval, settings.dt)
    apply_reduced_poisson_balance!(storage, transform, vertical, steps_per_met)

    @info "  Writing reduced Gaussian transport binary..."
    write_reduced_transport_binary_v2_from_storage!(bin_path, grid, vertical, storage, settings)
    @info @sprintf("  Done: %s (%.1fs)", basename(bin_path), time() - t_day)
    return (path = bin_path, report = nothing)
end

"""
    ReducedGaussianSpectralTransportBinaryV2Target

Config-driven ERA5 spectral -> reduced-Gaussian transport-binary v2 target.
"""
struct ReducedGaussianSpectralTransportBinaryV2Target{G, S, V} <: AbstractSpectralTransportBinaryV2Target
    config_path :: String
    grid :: G
    settings :: S
    vertical :: V
    dates :: Vector{Date}
end

target_dates(target::ReducedGaussianSpectralTransportBinaryV2Target) = target.dates
target_summary(target::ReducedGaussianSpectralTransportBinaryV2Target) =
    "ERA5 spectral -> transport binary v2 reduced Gaussian ($(target_summary(target.grid)))"

function log_spectral_transport_binary_v2_configuration(target::ReducedGaussianSpectralTransportBinaryV2Target)
    @info "ERA5 reduced-Gaussian transport-binary v2 preprocessor"
    @info "  Spectral dir: $(target.settings.spectral_dir)"
    @info "  Output dir:   $(target.settings.out_dir)"
    @info "  Target grid:  $(target_summary(target.grid))"
    @info "  DT:           $(target.settings.dt) s (half_dt=$(target.settings.half_dt) s)"
    @info "  Merged levels: $(target.vertical.Nz)"
    @info "  Float type:   $(target.settings.output_float_type)"
    return nothing
end

function build_spectral_transport_binary_v2_target(::Val{:era5_native_reduced_gaussian},
                                                   config_path::AbstractString,
                                                   cfg,
                                                   argv::AbstractVector{<:AbstractString},
                                                   grid::ReducedGaussianTargetGeometry,
                                                   settings,
                                                   vertical,
                                                   dates;
                                                   FT::Type{T} = Float64) where T <: AbstractFloat
    return ReducedGaussianSpectralTransportBinaryV2Target(
        String(config_path),
        grid,
        settings,
        vertical,
        collect(dates),
    )
end

function process_spectral_transport_binary_v2_day(target::ReducedGaussianSpectralTransportBinaryV2Target,
                                                  date::Date)
    return process_day_reduced_transport_binary_v2(date, target.grid, target.settings, target.vertical)
end

function main(argv=ARGS)
    base_logger = ConsoleLogger(stderr, Logging.Info; show_limited=false)
    global_logger(_FlushingLogger(base_logger))
    println(stderr, "[transport-v2-rg] Logger installed, starting…")
    flush(stderr)

    isempty(argv) && error("Usage: julia --project=. scripts/preprocessing/preprocess_era5_reduced_gaussian_transport_binary_v2.jl config.toml [--day YYYY-MM-DD]")

    target = build_spectral_transport_binary_v2_target(argv[1], argv[2:end]; FT=Float64)
    target isa ReducedGaussianSpectralTransportBinaryV2Target ||
        error("grid.type must be a reduced-Gaussian target for this script")
    run_spectral_transport_binary_v2_preprocessor(target)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
