#!/usr/bin/env julia
# Thin CLI wrapper plus target implementation for the config-driven ERA5
# spectral -> structured transport-binary v2 path.
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
using .AtmosTransport: AtmosGrid, CPU, write_transport_binary, TransportBinaryReader

struct WindowStageTiming
    spectral_seconds  :: Float64
    qv_seconds        :: Float64
    mass_fix_seconds  :: Float64
    dry_basis_seconds :: Float64
    merge_seconds     :: Float64
    store_seconds     :: Float64
    total_seconds     :: Float64
end

struct DayTimingReport
    read_seconds      :: Float64
    window_reports    :: Vector{WindowStageTiming}
    next_day_seconds  :: Float64
    poisson_seconds   :: Float64
    qv_endpoint_seconds :: Float64
    pack_seconds      :: Float64
    write_seconds     :: Float64
    total_seconds     :: Float64
end

function Base.summary(report::DayTimingReport)
    return string("DayTimingReport(", length(report.window_reports), " windows, total=",
                  round(report.total_seconds, digits=2), " s)")
end

function Base.show(io::IO, report::DayTimingReport)
    spectral = sum(r.spectral_seconds for r in report.window_reports)
    qv = sum(r.qv_seconds for r in report.window_reports)
    mass_fix = sum(r.mass_fix_seconds for r in report.window_reports)
    dry_basis = sum(r.dry_basis_seconds for r in report.window_reports)
    merge = sum(r.merge_seconds for r in report.window_reports)
    store = sum(r.store_seconds for r in report.window_reports)
    print(io, summary(report), "\n",
          "├── read:         ", round(report.read_seconds, digits=2), " s\n",
          "├── spectral:     ", round(spectral, digits=2), " s\n",
          "├── qv load:      ", round(qv, digits=2), " s\n",
          "├── mass fix:     ", round(mass_fix, digits=2), " s\n",
          "├── dry basis:    ", round(dry_basis, digits=2), " s\n",
          "├── merge/store:  ", round(merge + store, digits=2), " s\n",
          "├── next day h0:  ", round(report.next_day_seconds, digits=2), " s\n",
          "├── poisson:      ", round(report.poisson_seconds, digits=2), " s\n",
          "├── qv endpoints: ", round(report.qv_endpoint_seconds, digits=2), " s\n",
          "├── pack:         ", round(report.pack_seconds, digits=2), " s\n",
          "└── write:        ", round(report.write_seconds, digits=2), " s")
end

function log_day_timing_report(report::DayTimingReport)
    spectral = sum(r.spectral_seconds for r in report.window_reports)
    qv = sum(r.qv_seconds for r in report.window_reports)
    mass_fix = sum(r.mass_fix_seconds for r in report.window_reports)
    dry_basis = sum(r.dry_basis_seconds for r in report.window_reports)
    merge = sum(r.merge_seconds for r in report.window_reports)
    store = sum(r.store_seconds for r in report.window_reports)
    @info @sprintf("  Timing summary: total %.2fs | read %.2fs | spectral %.2fs | qv %.2fs | mass-fix %.2fs | dry-basis %.2fs | merge %.2fs | store %.2fs | next-day %.2fs | poisson %.2fs | qv-end %.2fs | pack %.2fs | write %.2fs",
                   report.total_seconds,
                   report.read_seconds,
                   spectral,
                   qv,
                   mass_fix,
                   dry_basis,
                   merge,
                   store,
                   report.next_day_seconds,
                   report.poisson_seconds,
                   report.qv_endpoint_seconds,
                   report.pack_seconds,
                   report.write_seconds)
    return nothing
end

function output_transport_binary_v2_path(date::Date, out_dir::String, min_dp::Float64, ::Type{FT}) where FT
    dp_tag = @sprintf("merged%dPa", round(Int, min_dp))
    ft_tag = FT == Float64 ? "float64" : "float32"
    date_str = Dates.format(date, "yyyymmdd")
    return joinpath(out_dir, "era5_transport_v2_$(date_str)_$(dp_tag)_$(ft_tag).bin")
end

function build_transport_binary_grid(grid::LatLonTargetGeometry, vertical, ::Type{FT}) where FT
    vc = AtmosTransport.HybridSigmaPressure(FT.(vertical.merged_vc.A), FT.(vertical.merged_vc.B))
    return AtmosGrid(grid.mesh, vc, CPU(); FT=FT)
end

function process_window_profiled!(win_idx::Int,
                                  hour::Int,
                                  spec,
                                  grid::LatLonTargetGeometry,
                                  vertical,
                                  settings,
                                  transform::SpectralTransformWorkspace,
                                  merged::MergeWorkspace{FT},
                                  qv::AbstractQVWorkspace{FT},
                                  storage::WindowStorage{FT},
                                  ps_offsets::Vector{Float64}) where FT
    Nx = size(transform.sp, 1)
    Ny = size(transform.sp, 2)
    t_total = time()

    t0 = time()
    spectral_to_native_fields!(
        transform.m_arr, transform.am_arr, transform.bm_arr, transform.cm_arr, transform.sp,
        transform.u_cc, transform.v_cc, transform.u_stag, transform.v_stag, transform.dp,
        spec.lnsp_all[hour], spec.vo_by_hour[hour], spec.d_by_hour[hour],
        spec.T, vertical.level_range, vertical.ab, grid, settings.half_dt,
        transform.P_buf, transform.fft_buf, transform.field_2d,
        transform.P_buf_t, transform.fft_buf_t, transform.fft_out_t,
        transform.u_spec_t, transform.v_spec_t, transform.field_2d_t,
        transform.bfft_plans)
    spectral_seconds = time() - t0

    t0 = time()
    read_window_qv!(qv, win_idx, Nx, Ny, vertical.Nz_native)
    qv_seconds = time() - t0

    t0 = time()
    apply_mass_fix_if_needed!(qv, transform, grid, vertical, settings, ps_offsets, win_idx)
    mass_fix_seconds = time() - t0

    t0 = time()
    apply_dry_basis_if_needed!(settings.mass_basis, transform, qv)
    dry_basis_seconds = time() - t0

    t0 = time()
    merge_native_window!(merged, transform, qv, vertical, settings)
    merge_seconds = time() - t0

    t0 = time()
    store_window_fields!(storage, merged, transform.sp, qv, win_idx)
    store_seconds = time() - t0

    total_seconds = time() - t_total
    should_log_window(win_idx, length(storage.all_m)) &&
        @info(@sprintf("    Window %d/%d (hour %02d): %.2fs  [spectral=%.2fs qv=%.2fs merge=%.2fs store=%.2fs]  ps_offset=%+.3f Pa",
                       win_idx, length(storage.all_m), hour, total_seconds,
                       spectral_seconds, qv_seconds, merge_seconds, store_seconds,
                       ps_offsets[win_idx]))

    return WindowStageTiming(spectral_seconds, qv_seconds, mass_fix_seconds,
                             dry_basis_seconds, merge_seconds, store_seconds,
                             total_seconds)
end

function build_structured_transport_windows(storage::WindowStorage{FT},
                                            settings,
                                            grid::LatLonTargetGeometry,
                                            vertical,
                                            last_hour_next) where FT
    fill_qv_endpoints!(storage, last_hour_next)
    scratch = allocate_merge_workspace(grid, vertical.Nz_native, vertical.Nz, FT)
    Nt = length(storage.all_m)

    if settings.include_qv
        return [begin
            compute_window_deltas!(scratch, storage, win_idx, last_hour_next)
            (; m = storage.all_m[win_idx],
               am = storage.all_am[win_idx],
               bm = storage.all_bm[win_idx],
               cm = storage.all_cm[win_idx],
               ps = storage.all_ps[win_idx],
               qv_start = storage.all_qv_start[win_idx],
               qv_end = storage.all_qv_end[win_idx],
               dam = copy(scratch.dam_merged),
               dbm = copy(scratch.dbm_merged),
               dcm = copy(scratch.dcm_merged),
               dm = copy(scratch.dm_merged))
        end for win_idx in 1:Nt]
    else
        return [begin
            compute_window_deltas!(scratch, storage, win_idx, last_hour_next)
            (; m = storage.all_m[win_idx],
               am = storage.all_am[win_idx],
               bm = storage.all_bm[win_idx],
               cm = storage.all_cm[win_idx],
               ps = storage.all_ps[win_idx],
               dam = copy(scratch.dam_merged),
               dbm = copy(scratch.dbm_merged),
               dcm = copy(scratch.dcm_merged),
               dm = copy(scratch.dm_merged))
        end for win_idx in 1:Nt]
    end
end

function write_transport_binary_v2_from_storage!(bin_path::AbstractString,
                                                 grid::LatLonTargetGeometry,
                                                 vertical,
                                                 storage::WindowStorage{FT},
                                                 settings,
                                                 last_hour_next) where FT
    t_pack = time()
    windows = build_structured_transport_windows(storage, settings, grid, vertical, last_hour_next)
    pack_seconds = time() - t_pack

    transport_grid = build_transport_binary_grid(grid, vertical, FT)
    t_write = time()
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
                           humidity_sampling=:window_endpoints,
                           delta_semantics=:forward_window_endpoint_difference,
                           mass_basis=settings.mass_basis,
                           extra_header=Dict(
                               "poisson_balance_target_scale" => poisson_balance_target_scale(steps_per_met),
                               "poisson_balance_target_semantics" => "forward_window_mass_difference / (2 * steps_per_window)",
                           ),
                           threaded=true)
    write_seconds = time() - t_write
    return pack_seconds, write_seconds
end

function process_day_transport_binary_v2(date::Date,
                                         grid::LatLonTargetGeometry,
                                         settings,
                                         vertical;
                                         next_day_hour0=nothing)
    FT = settings.output_float_type
    Nz_native = vertical.Nz_native
    Nz = vertical.Nz
    Nx = nlon(grid)
    Ny = nlat(grid)
    steps_per_met = exact_steps_per_window(settings.met_interval, settings.dt)
    date_str = Dates.format(date, "yyyymmdd")

    vo_d_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_vo_d.gb")
    lnsp_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_lnsp.gb")

    if !isfile(vo_d_path) || !isfile(lnsp_path)
        @warn "Missing GRIB files for $date_str, skipping"
        return nothing
    end

    mkpath(settings.out_dir)
    bin_path = output_transport_binary_v2_path(date, settings.out_dir, settings.min_dp, FT)
    if isfile(bin_path)
        @info "  SKIP (exists): $(basename(bin_path))"
        return (path = bin_path, report = nothing)
    end

    t_day = time()
    t0 = time()
    @info "  Reading spectral data for $date_str..."
    spec = read_day_spectral_streaming(vo_d_path, lnsp_path; T_target=settings.T_target)
    read_seconds = time() - t0
    @info @sprintf("  Spectral data read: T=%d, %d hours (%.1fs)",
                   spec.T, spec.n_times, read_seconds)

    Nt = spec.n_times
    transform = allocate_transform_workspace(grid, spec.T, Nz_native)
    merged = allocate_merge_workspace(grid, Nz_native, Nz, FT)
    storage = allocate_window_storage(Nt, FT; include_qv=settings.include_qv)
    qv = allocate_qv_workspace(grid, settings, date, Nz_native, Nz, FT)
    ps_offsets = zeros(Float64, Nt + 1)
    window_reports = Vector{WindowStageTiming}(undef, Nt)

    log_mass_fix_configuration(settings)
    @info "  Computing spectral -> gridpoint -> merged for $Nt windows..."

    for (win_idx, hour) in enumerate(spec.hours)
        window_reports[win_idx] = process_window_profiled!(win_idx, hour, spec, grid, vertical,
                                                           settings, transform, merged, qv,
                                                           storage, ps_offsets)
    end

    if settings.mass_fix_enable
        @info @sprintf("  Mass-fix offsets (Pa) min/max/mean: %+.3f / %+.3f / %+.3f",
                       minimum(ps_offsets[1:Nt]),
                       maximum(ps_offsets[1:Nt]),
                       sum(ps_offsets[1:Nt]) / Nt)
    end

    t0 = time()
    last_hour_next = next_day_merged_fields(next_day_hour0, date, grid, vertical,
                                            settings, transform, merged, qv, ps_offsets)
    next_day_seconds = time() - t0

    t0 = time()
    apply_poisson_balance!(storage, last_hour_next, vertical, steps_per_met)
    poisson_seconds = time() - t0

    t0 = time()
    fill_qv_endpoints!(storage, last_hour_next)
    qv_endpoint_seconds = time() - t0

    pack_seconds, write_seconds = write_transport_binary_v2_from_storage!(bin_path, grid, vertical,
                                                                          storage, settings, last_hour_next)

    report = DayTimingReport(read_seconds, window_reports, next_day_seconds,
                             poisson_seconds, qv_endpoint_seconds,
                             pack_seconds, write_seconds, time() - t_day)
    log_day_timing_report(report)

    reader = TransportBinaryReader(bin_path; FT=FT)
    actual = filesize(bin_path)
    @info @sprintf("  Done: %s (%s, %.2f GB, %.1fs)", basename(bin_path), summary(reader), actual / 1e9, report.total_seconds)
    close(reader)

    return (path = bin_path, report = report)
end

"""
    LatLonSpectralTransportBinaryV2Target

Config-driven ERA5 spectral -> structured lat-lon transport-binary v2 target.
"""
struct LatLonSpectralTransportBinaryV2Target{G, S, V} <: AbstractSpectralTransportBinaryV2Target
    config_path :: String
    grid :: G
    settings :: S
    vertical :: V
    dates :: Vector{Date}
end

target_dates(target::LatLonSpectralTransportBinaryV2Target) = target.dates
target_summary(target::LatLonSpectralTransportBinaryV2Target) =
    "ERA5 spectral -> transport binary v2 lat-lon ($(target_summary(target.grid)))"

log_spectral_transport_binary_v2_configuration(target::LatLonSpectralTransportBinaryV2Target) =
    log_preprocessor_configuration(target.settings, target.grid, target.vertical)

function build_spectral_transport_binary_v2_target(::Val{:latlon},
                                                   config_path::AbstractString,
                                                   cfg,
                                                   argv::AbstractVector{<:AbstractString},
                                                   grid::LatLonTargetGeometry,
                                                   settings,
                                                   vertical,
                                                   dates;
                                                   FT::Type{T} = Float64) where T <: AbstractFloat
    return LatLonSpectralTransportBinaryV2Target(
        String(config_path),
        grid,
        settings,
        vertical,
        collect(dates),
    )
end

function process_spectral_transport_binary_v2_day(target::LatLonSpectralTransportBinaryV2Target,
                                                  date::Date)
    next_day_h0 = next_day_hour0(date, target.dates, target.settings.spectral_dir, target.settings.T_target)
    next_day_h0 !== nothing && @info("  Next day hour 0 available for last-window delta")
    return process_day_transport_binary_v2(date, target.grid, target.settings, target.vertical;
                                           next_day_hour0=next_day_h0)
end

function main_v2(argv=ARGS)
    base_logger = ConsoleLogger(stderr, Logging.Info; show_limited=false)
    global_logger(_FlushingLogger(base_logger))
    println(stderr, "[transport-v2] Logger installed, starting…")
    flush(stderr)
    atexit(() -> (try flush(stderr) catch; end; try flush(stdout) catch; end))

    if isempty(argv)
        println("""
        ERA5 spectral -> src transport-binary preprocessor

        Usage:
          julia -t8 --project=. $(PROGRAM_FILE) config.toml [--day 2021-12-01]
        """)
        return
    end

    target = build_spectral_transport_binary_v2_target(argv[1], argv[2:end]; FT=Float64)
    target isa LatLonSpectralTransportBinaryV2Target ||
        error("This script currently supports only LatLonTargetGeometry")
    run_spectral_transport_binary_v2_preprocessor(target)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_v2()
end
