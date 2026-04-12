#!/usr/bin/env julia

using Printf

const COMPARE_V4_SCRIPT_DIR = @__DIR__
const COMPARE_V4_REPO_ROOT = abspath(joinpath(COMPARE_V4_SCRIPT_DIR, "..", ".."))

include(joinpath(COMPARE_V4_REPO_ROOT, "src", "AtmosTransport.jl"))
using .AtmosTransport

mutable struct FieldComparisonStats
    max_abs  :: Float64
    max_rel  :: Float64
    sum_abs  :: Float64
    sum_sq   :: Float64
    ref_sum_sq :: Float64
    n_diff   :: Int64
    gt1      :: Int64
    gt10     :: Int64
    gt100    :: Int64
    n_total  :: Int64
end

FieldComparisonStats() = FieldComparisonStats(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0)

function update!(stats::FieldComparisonStats, a, b)
    @inbounds for i in eachindex(a, b)
        da = Float64(a[i] - b[i])
        ad = abs(da)
        stats.max_abs = max(stats.max_abs, ad)
        denom = max(abs(Float64(a[i])), abs(Float64(b[i])), 1e-30)
        stats.max_rel = max(stats.max_rel, ad / denom)
        stats.sum_abs += ad
        stats.sum_sq += da * da
        ref = Float64(a[i])
        stats.ref_sum_sq += ref * ref
        stats.n_diff += ad != 0 ? 1 : 0
        stats.gt1 += ad > 1 ? 1 : 0
        stats.gt10 += ad > 10 ? 1 : 0
        stats.gt100 += ad > 100 ? 1 : 0
        stats.n_total += 1
    end
    return stats
end

mean_abs(stats::FieldComparisonStats) = stats.n_total == 0 ? 0.0 : stats.sum_abs / stats.n_total
rms_diff(stats::FieldComparisonStats) = stats.n_total == 0 ? 0.0 : sqrt(stats.sum_sq / stats.n_total)
rel_rms(stats::FieldComparisonStats) = stats.ref_sum_sq == 0 ? 0.0 : sqrt(stats.sum_sq / stats.ref_sum_sq)
diff_fraction(stats::FieldComparisonStats) = stats.n_total == 0 ? 0.0 : stats.n_diff / stats.n_total

function header_summary(reader)
    h = reader.header
    return (
        version = h.version,
        dims = (h.Nx, h.Ny, h.Nz, h.Nt),
        mass_basis = h.mass_basis,
        header_bytes = h.header_bytes,
        lon_first = first(h.lons_f64),
        lon_last = last(h.lons_f64),
        lat_first = first(h.lats_f64),
        lat_last = last(h.lats_f64),
    )
end

wrapped_lon_offset_deg(old_lon::Float64, new_lon::Float64) =
    mod(old_lon - new_lon + 180.0, 360.0) - 180.0

function compare_headers(old_reader, new_reader)
    old = header_summary(old_reader)
    new = header_summary(new_reader)
    A_diff = maximum(abs.(old_reader.header.A_ifc .- new_reader.header.A_ifc))
    B_diff = maximum(abs.(old_reader.header.B_ifc .- new_reader.header.B_ifc))
    lon_shift = wrapped_lon_offset_deg(old.lon_first, new.lon_first)

    println("Header comparison")
    @printf("  version:      %d vs %d\n", old.version, new.version)
    @printf("  dims:         %s vs %s\n", string(old.dims), string(new.dims))
    @printf("  mass_basis:   %s vs %s\n", String(old.mass_basis), String(new.mass_basis))
    @printf("  header_bytes: %d vs %d\n", old.header_bytes, new.header_bytes)
    @printf("  A_ifc maxabs: %.6e\n", A_diff)
    @printf("  B_ifc maxabs: %.6e\n", B_diff)
    @printf("  lon range:    [%.6f, %.6f] vs [%.6f, %.6f]\n",
            old.lon_first, old.lon_last, new.lon_first, new.lon_last)
    @printf("  lat range:    [%.6f, %.6f] vs [%.6f, %.6f]\n",
            old.lat_first, old.lat_last, new.lat_first, new.lat_last)
    @printf("  wrapped lon offset (old-new, first center): %.6f deg\n", lon_shift)
end

function compare_fields(old_reader, new_reader)
    stats = Dict(
        "m" => FieldComparisonStats(),
        "ps" => FieldComparisonStats(),
        "am" => FieldComparisonStats(),
        "bm" => FieldComparisonStats(),
        "cm" => FieldComparisonStats(),
        "dam" => FieldComparisonStats(),
        "dbm" => FieldComparisonStats(),
        "dm" => FieldComparisonStats(),
    )

    for win in 1:window_count(old_reader)
        m_old, ps_old, flux_old = load_window!(old_reader, win)
        m_new, ps_new, flux_new = load_window!(new_reader, win)
        del_old = load_flux_delta_window!(old_reader, win)
        del_new = load_flux_delta_window!(new_reader, win)

        update!(stats["m"], m_old, m_new)
        update!(stats["ps"], ps_old, ps_new)
        update!(stats["am"], flux_old.am, flux_new.am)
        update!(stats["bm"], flux_old.bm, flux_new.bm)
        update!(stats["cm"], flux_old.cm, flux_new.cm)
        update!(stats["dam"], del_old.dam, del_new.dam)
        update!(stats["dbm"], del_old.dbm, del_new.dbm)
        update!(stats["dm"], del_old.dm, del_new.dm)
    end

    println("\nField comparison across all windows")
    println("  field   max_abs       mean_abs      rms_diff      rel_rms       diff_frac    >1       >10      >100")
    for name in ("m", "ps", "am", "bm", "cm", "dam", "dbm", "dm")
        s = stats[name]
        @printf("  %-4s  %11.4e  %11.4e  %11.4e  %11.4e  %11.4e  %8d %8d %8d\n",
                name, s.max_abs, mean_abs(s), rms_diff(s), rel_rms(s), diff_fraction(s),
                s.gt1, s.gt10, s.gt100)
    end

    return stats
end

function main(args)
    length(args) == 2 || error("Usage: julia --project=. $(PROGRAM_FILE) old.bin new.bin")
    old_path = expanduser(args[1])
    new_path = expanduser(args[2])
    isfile(old_path) || error("Old binary not found: $old_path")
    isfile(new_path) || error("New binary not found: $new_path")

    old_reader = ERA5BinaryReader(old_path; FT=Float32)
    new_reader = ERA5BinaryReader(new_path; FT=Float32)

    try
        compare_headers(old_reader, new_reader)
        compare_fields(old_reader, new_reader)
    finally
        close(old_reader)
        close(new_reader)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
