#!/usr/bin/env julia
# ===========================================================================
# OBSOLETE — DO NOT USE FOR NEW BINARIES (deprecated 2026-04-06)
# ===========================================================================
#
# This script PICKS native cm at merged interface boundaries (merge_interface_field!)
# and then smears the surface-BC residual via correct_cm_residual!. The result is a
# cm field that does NOT satisfy local continuity with the merged am/bm — most
# visibly at the polar caps, where the resulting binary causes the global Check_CFL
# pre-pass to reject all 5 halvings (era5_f64_debug_moist.toml). See
# `~/.claude/projects/-home-cfranken-code-gitHub-AtmosTransportModel/memory/tm5_preprocessing_comparison.md`
# for the full analysis.
#
# REPLACEMENTS:
#   - For core-only (no physics) binaries:
#       scripts/preprocessing/preprocess_spectral_v4_binary.jl
#       (goes spectral GRIB → binary directly, recomputes cm from continuity,
#        then Poisson-balances am/bm, then recomputes cm again)
#   - For full-physics binaries (convection, surface, QV, etc.):
#       scripts/preprocessing/preprocess_era5_daily.jl
#       (consumes spectral NetCDF + physics NetCDFs, also uses
#        recompute_cm_from_divergence! for cm)
#
# This file is kept ONLY for historical reference and to support reading old
# binaries that were generated with it. Do not generate new binaries with this.
# Run-time configs that point at binaries produced by this script should be
# regenerated with one of the replacements above.
#
# ===========================================================================
# Convert preprocessed mass-flux NetCDF → merged-level flat binary (v2)
#
# Reads native-level ERA5 mass flux NetCDF (from preprocess_spectral_massflux.jl),
# merges thin vertical levels, and writes a self-describing flat binary file
# with embedded QV, CMFMC, surface fields, and A/B coefficients.
#
# The binary format (v2) is fully self-describing: the model auto-detects
# grid dimensions, vertical coordinate, and available fields from the header.
# No manual `size` or `merge_min_thickness_Pa` needed in run configs.
#
# Modes:
#   1. Single file:
#      julia -t10 --project=. scripts/preprocessing/convert_merged_massflux_to_binary.jl \
#          input.nc [output.bin] [--min-dp=1000]
#
#   2. Batch (same dir as source):
#      MIN_DP=1000 MASSFLUX_DIR=~/data/AtmosTransport/met/era5/preprocessed_spectral_catrine \
#          julia -t10 --project=. scripts/preprocessing/convert_merged_massflux_to_binary.jl
#
# Output goes alongside source NetCDF (same directory), following the data layout:
#   ~/data/AtmosTransport/met/era5/<preprocessed_dir>/
#     massflux_era5_spectral_202112_float32.nc              ← source
#     massflux_era5_spectral_202112_merged1000Pa_float32.bin ← output
#
# Binary layout (v2):
#   [Header]  — 16384 bytes (JSON metadata, zero-padded)
#   [Window 1]: m | am | bm | cm | ps [| qv | cmfmc | pblh | t2m | ustar | hflux]
#   [Window 2]: same layout
#   ...
#
# Follows the same pattern as preprocess_geosfp_cs_v4.jl (GEOS-style binary).
# ===========================================================================

using NCDatasets
using JSON3
using Printf
using TOML

# ===========================================================================
# Minimal vertical coordinate types (self-contained, no package dependency)
# ===========================================================================

struct HybridSigmaPressure{FT}
    A :: Vector{FT}   # Nz+1 interface values
    B :: Vector{FT}   # Nz+1 interface values
end

n_levels(vc::HybridSigmaPressure) = length(vc.A) - 1
pressure_at_interface(vc::HybridSigmaPressure, k, p_s) = vc.A[k] + vc.B[k] * p_s
level_thickness(vc::HybridSigmaPressure, k, p_s) =
    pressure_at_interface(vc, k + 1, p_s) - pressure_at_interface(vc, k, p_s)
pressure_at_level(vc::HybridSigmaPressure, k, p_s) =
    (pressure_at_interface(vc, k, p_s) + pressure_at_interface(vc, k + 1, p_s)) / 2

"""
    merge_thin_levels(vc; min_thickness_Pa=1000, p_surface=101325)

Merge consecutive levels thinner than `min_thickness_Pa` into coarser layers.
Works inward from both ends (top-down + bottom-up), joins at meeting point.
Returns `(merged_vc, merge_map)`.
"""
function merge_thin_levels(vc::HybridSigmaPressure{FT};
                            min_thickness_Pa::Real = FT(1000),
                            p_surface::Real = FT(101325)) where FT
    Nz = n_levels(vc)
    ps = FT(p_surface)
    min_dp = FT(min_thickness_Pa)
    dp = [level_thickness(vc, k, ps) for k in 1:Nz]

    # Top-down pass
    top_ifaces = Int[1]
    acc = zero(FT)
    for k in 1:Nz
        acc += dp[k]
        if acc >= min_dp
            push!(top_ifaces, k + 1)
            acc = zero(FT)
        end
    end

    # Bottom-up pass
    bot_ifaces = Int[Nz + 1]
    acc = zero(FT)
    for k in Nz:-1:1
        acc += dp[k]
        if acc >= min_dp
            pushfirst!(bot_ifaces, k)
            acc = zero(FT)
        end
    end

    # Join: top-down until first bottom-up interface
    first_bot = bot_ifaces[1]
    keep_interfaces = Int[]
    for iface in top_ifaces
        iface <= first_bot && push!(keep_interfaces, iface)
    end
    if isempty(keep_interfaces) || keep_interfaces[end] < first_bot
        push!(keep_interfaces, first_bot)
    end
    for iface in bot_ifaces
        iface > keep_interfaces[end] && push!(keep_interfaces, iface)
    end

    # Build merged A/B
    A_merged = FT[vc.A[k] for k in keep_interfaces]
    B_merged = FT[vc.B[k] for k in keep_interfaces]
    merged_vc = HybridSigmaPressure(A_merged, B_merged)
    Nz_merged = n_levels(merged_vc)

    # Build merge_map: native level → merged level
    merge_map = Vector{Int}(undef, Nz)
    km = 1
    for k_native in 1:Nz
        while km < Nz_merged && keep_interfaces[km + 1] <= k_native
            km += 1
        end
        merge_map[k_native] = km
    end

    return merged_vc, merge_map
end

# ===========================================================================
# Threaded merging kernels
# ===========================================================================

"""Merge cell-center fields (m, am, bm): sum native levels per merged group.
Threaded over the k-slabs for the fill + accumulate."""
function merge_cell_field!(merged::Array{FT,3}, native::Array{FT,3},
                           mm::Vector{Int}) where FT
    Nz_merged = size(merged, 3)
    # Zero and accumulate per merged level (thread over merged levels)
    Threads.@threads for km in 1:Nz_merged
        @views merged[:, :, km] .= zero(FT)
    end
    # Accumulate (sequential over native k — each km touched once per k)
    @inbounds for k in 1:length(mm)
        km = mm[k]
        @views merged[:, :, km] .+= native[:, :, k]
    end
end

"""Merge interface fields (cm, cmfmc): pick at merged interface boundaries."""
function merge_interface_field!(merged::Array{FT,3}, native::Array{FT,3},
                                mm::Vector{Int}) where FT
    Nz_merged = maximum(mm)
    fill!(merged, zero(FT))
    @views merged[:, :, 1] .= native[:, :, 1]   # TOA
    @inbounds for km in 1:Nz_merged
        k_last = findlast(==(km), mm)
        @views merged[:, :, km + 1] .= native[:, :, k_last + 1]
    end
end

"""Mass-weighted merge of QV: qv_m[km] = Σ(qv[k]×m[k]) / Σ(m[k])."""
function merge_qv!(qv_merged::Array{FT,3}, qv_native::Array{FT,3},
                    m_native::Array{FT,3}, mm::Vector{Int}) where FT
    Nz_merged = size(qv_merged, 3)
    fill!(qv_merged, zero(FT))
    m_sum = zeros(FT, size(qv_merged))
    @inbounds for k in 1:length(mm)
        km = mm[k]
        @views begin
            qv_merged[:, :, km] .+= qv_native[:, :, k] .* m_native[:, :, k]
            m_sum[:, :, km]     .+= m_native[:, :, k]
        end
    end
    Threads.@threads for km in 1:Nz_merged
        @views @inbounds for j in 1:size(qv_merged, 2), i in 1:size(qv_merged, 1)
            ms = m_sum[i, j, km]
            qv_merged[i, j, km] = ms > zero(FT) ? qv_merged[i, j, km] / ms : zero(FT)
        end
    end
end

"""
Distribute continuity residual so cm[:,:,1]=0 (TOA) and cm[:,:,Nz+1]=0 (surface).
Correction at each interface is proportional to cumulative mass fraction above it.
Threaded over latitude.
"""
function correct_cm_residual!(cm::Array{FT,3}, m::Array{FT,3}) where FT
    Nx, Ny, Nz_plus1 = size(cm)
    Nz = Nz_plus1 - 1
    Threads.@threads for j in 1:Ny
        @inbounds for i in 1:Nx
            residual = cm[i, j, Nz + 1]
            abs(residual) < eps(FT) && continue
            col_mass = zero(FT)
            for k in 1:Nz
                col_mass += m[i, j, k]
            end
            col_mass < eps(FT) && continue
            cum_mass = zero(FT)
            cm[i, j, 1] = zero(FT)
            for k in 1:Nz
                cum_mass += m[i, j, k]
                cm[i, j, k + 1] -= residual * cum_mass / col_mass
            end
        end
    end
end

# ===========================================================================
# Binary write helper — uses unsafe_write for contiguous arrays
# ===========================================================================

"""Write a contiguous array to IO as raw bytes (unsafe_write for speed + reliability)."""
function write_array!(io::IO, arr::Array{FT}) where FT
    nb = sizeof(arr)
    GC.@preserve arr begin
        written = unsafe_write(io, pointer(arr), nb)
    end
    written == nb || error("Short write: expected $nb bytes, got $written")
    return written
end

# ===========================================================================
# Binary v2 writer
# ===========================================================================

const HEADER_SIZE_V2 = 16384

function write_v2_binary(bin_path::String, ds::NCDataset,
                         merged_vc::HybridSigmaPressure,
                         merge_map::Vector{Int},
                         min_dp::Float64)

    # Data float type from NetCDF attribute (NOT from vertical coordinate type)
    ft_str = get(ds.attrib, "float_type", "Float32")
    FT = ft_str == "Float64" ? Float64 : Float32

    # Native dimensions
    lons = Float64.(ds["lon"][:])
    lats = Float64.(ds["lat"][:])
    Nx = length(lons)
    Ny = length(lats)
    Nz_native = ds.dim["lev"]
    Nt = ds.dim["time"]

    Nz = n_levels(merged_vc)

    dt_seconds      = Float64(ds.attrib["dt_seconds"])
    half_dt_seconds = Float64(ds.attrib["half_dt_seconds"])
    steps_per_met   = Int(ds.attrib["steps_per_met_window"])
    level_top       = Int(ds.attrib["level_top"])
    level_bot       = Int(ds.attrib["level_bot"])
    ft_bytes        = sizeof(FT)

    # Detect available optional fields
    has_qv      = any(k -> haskey(ds, k), ("qv", "QV", "q"))
    has_cmfmc   = haskey(ds, "conv_mass_flux")
    has_surface = all(k -> haskey(ds, k), ("pblh", "t2m", "ustar", "hflux"))

    qv_varname = has_qv ? first(filter(k -> haskey(ds, k), ("qv", "QV", "q"))) : ""

    # Element counts (Int64 to avoid overflow in size calculation)
    n_m  = Int64(Nx) * Ny * Nz
    n_am = Int64(Nx + 1) * Ny * Nz
    n_bm = Int64(Nx) * (Ny + 1) * Nz
    n_cm = Int64(Nx) * Ny * (Nz + 1)
    n_ps = Int64(Nx) * Ny
    n_qv    = has_qv      ? Int64(Nx) * Ny * Nz       : Int64(0)
    n_cmfmc = has_cmfmc   ? Int64(Nx) * Ny * (Nz + 1) : Int64(0)
    n_sfc   = has_surface  ? Int64(Nx) * Ny            : Int64(0)

    elems_per_window = n_m + n_am + n_bm + n_cm + n_ps +
                       n_qv + n_cmfmc + 4 * n_sfc
    bytes_per_window = elems_per_window * ft_bytes
    total_bytes = Int64(HEADER_SIZE_V2) + bytes_per_window * Nt

    @info @sprintf("  Merged grid: %d×%d×%d (from %d native levels)", Nx, Ny, Nz, Nz_native)
    @info @sprintf("  Windows: %d, %.1f MB/win, %.2f GB total",
                   Nt, bytes_per_window / 1e6, total_bytes / 1e9)
    @info @sprintf("  Optional: QV=%s, CMFMC=%s, Surface=%s",
                   has_qv, has_cmfmc, has_surface)
    @info @sprintf("  Threads: %d", Threads.nthreads())

    # Check available disk space
    mkpath(dirname(bin_path))
    dir_stat = stat(dirname(bin_path))
    # (stat doesn't give free space; just warn about expected size)
    @info @sprintf("  Output: %s (%.2f GB needed)", bin_path, total_bytes / 1e9)

    # Build header
    header = Dict{String,Any}(
        "magic"                  => "MFLX",
        "version"                => 2,
        "header_bytes"           => HEADER_SIZE_V2,
        "Nx"                     => Nx,
        "Ny"                     => Ny,
        "Nz"                     => Nz,
        "Nt"                     => Nt,
        "float_type"             => ft_str,
        "float_bytes"            => ft_bytes,
        "window_bytes"           => bytes_per_window,
        "n_m"                    => n_m,
        "n_am"                   => n_am,
        "n_bm"                   => n_bm,
        "n_cm"                   => n_cm,
        "n_ps"                   => n_ps,
        "n_qv"                   => n_qv,
        "n_cmfmc"                => n_cmfmc,
        "n_pblh"                 => n_sfc,
        "n_t2m"                  => n_sfc,
        "n_ustar"                => n_sfc,
        "n_hflux"                => n_sfc,
        "dt_seconds"             => dt_seconds,
        "half_dt_seconds"        => half_dt_seconds,
        "steps_per_met_window"   => steps_per_met,
        "level_top"              => level_top,
        "level_bot"              => level_bot,
        "lons"                   => lons,
        "lats"                   => lats,
        # Vertical coordinate (self-describing)
        "A_ifc"                  => Float64.(merged_vc.A),
        "B_ifc"                  => Float64.(merged_vc.B),
        # Merge provenance
        "Nz_native"              => Nz_native,
        "merge_map"              => merge_map,
        "merge_min_thickness_Pa" => min_dp,
        # Feature flags
        "include_qv"             => has_qv,
        "include_cmfmc"          => has_cmfmc,
        "include_surface"        => has_surface,
    )
    header_json = JSON3.write(header)
    length(header_json) < HEADER_SIZE_V2 ||
        error("Header JSON too large ($(length(header_json)) >= $HEADER_SIZE_V2)")

    # Allocate merged work arrays
    m_merged  = Array{FT}(undef, Nx, Ny, Nz)
    am_merged = Array{FT}(undef, Nx + 1, Ny, Nz)
    bm_merged = Array{FT}(undef, Nx, Ny + 1, Nz)
    cm_merged = Array{FT}(undef, Nx, Ny, Nz + 1)

    qv_merged    = has_qv      ? Array{FT}(undef, Nx, Ny, Nz)     : nothing
    cmfmc_merged = has_cmfmc   ? Array{FT}(undef, Nx, Ny, Nz + 1) : nothing

    # Reusable 2D buffers for surface fields (avoid repeated allocation)
    sfc_buf = has_surface ? Array{FT}(undef, Nx, Ny) : nothing

    # Write binary
    @info "Writing: $bin_path"
    bytes_written = Int64(0)
    open(bin_path, "w") do io
        # Padded JSON header
        hdr_buf = zeros(UInt8, HEADER_SIZE_V2)
        copyto!(hdr_buf, 1, Vector{UInt8}(header_json), 1, length(header_json))
        write(io, hdr_buf)
        bytes_written += HEADER_SIZE_V2

        for win in 1:Nt
            t0 = time()

            # --- Read native data ---
            m_native  = FT.(ds["m"][:, :, :, win])
            am_native = FT.(ds["am"][:, :, :, win])
            bm_native = FT.(ds["bm"][:, :, :, win])
            cm_native = FT.(ds["cm"][:, :, :, win])
            ps_data   = Array{FT}(FT.(ds["ps"][:, :, win]))

            # --- Merge mass fluxes ---
            merge_cell_field!(m_merged, m_native, merge_map)
            merge_cell_field!(am_merged, am_native, merge_map)
            merge_cell_field!(bm_merged, bm_native, merge_map)
            merge_interface_field!(cm_merged, cm_native, merge_map)

            # Fix continuity residual (critical for Z-CFL)
            correct_cm_residual!(cm_merged, m_merged)

            # --- Write core fields (unsafe_write for contiguous arrays) ---
            bytes_written += write_array!(io, m_merged)
            bytes_written += write_array!(io, am_merged)
            bytes_written += write_array!(io, bm_merged)
            bytes_written += write_array!(io, cm_merged)
            bytes_written += write_array!(io, ps_data)

            # --- Optional: QV ---
            if has_qv
                qv_native = FT.(ds[qv_varname][:, :, :, win])
                merge_qv!(qv_merged, qv_native, m_native, merge_map)
                bytes_written += write_array!(io, qv_merged)
            end

            # --- Optional: CMFMC ---
            if has_cmfmc
                cmfmc_native = FT.(ds["conv_mass_flux"][:, :, :, win])
                merge_interface_field!(cmfmc_merged, cmfmc_native, merge_map)
                bytes_written += write_array!(io, cmfmc_merged)
            end

            # --- Optional: Surface fields (2D, no merge needed) ---
            if has_surface
                for varname in ("pblh", "t2m", "ustar", "hflux")
                    sfc_buf .= FT.(ds[varname][:, :, win])
                    bytes_written += write_array!(io, sfc_buf)
                end
            end

            t_win = round(time() - t0, digits=2)
            if win <= 3 || win == Nt || win % 20 == 0
                @info @sprintf("  Window %d/%d  (%.2fs)  [%.1f GB written]",
                               win, Nt, t_win, bytes_written / 1e9)
            end
        end

        flush(io)
    end

    actual_size = filesize(bin_path)
    @info @sprintf("Done: %s (%.2f GB)", bin_path, actual_size / 1e9)
    @info @sprintf("  bytes_written=%d, filesize=%d, expected=%d",
                   bytes_written, actual_size, total_bytes)
    actual_size == total_bytes ||
        error(@sprintf("SIZE MISMATCH: expected %d bytes, got %d. Output is corrupt!",
                       total_bytes, actual_size))
end

# ===========================================================================
# Merge summary: print level structure
# ===========================================================================

function print_merge_summary(vc_native, merged_vc, merge_map, min_dp)
    Nz_native = n_levels(vc_native)
    Nz_merged = n_levels(merged_vc)
    ps = 101325.0

    @info @sprintf("Merge summary: %d → %d levels (min_dp = %.0f Pa = %.1f hPa)",
                   Nz_native, Nz_merged, min_dp, min_dp / 100)

    # Print merged groups
    for km in 1:Nz_merged
        native_levels = findall(==(km), merge_map)
        dp_m = level_thickness(merged_vc, km, ps)
        p_mid = pressure_at_level(merged_vc, km, ps)
        n = length(native_levels)
        if n > 1
            @info @sprintf("  merged %3d: %2d native (k=%d..%d) dp=%7.0f Pa (%5.1f hPa) p=%8.0f Pa",
                          km, n, first(native_levels), last(native_levels),
                          dp_m, dp_m / 100, p_mid)
        end
    end

    # Min/max thickness of merged grid
    dp_all = [level_thickness(merged_vc, k, ps) for k in 1:Nz_merged]
    @info @sprintf("  Min thickness: %.0f Pa (%.1f hPa), Max: %.0f Pa (%.1f hPa)",
                   minimum(dp_all), minimum(dp_all) / 100,
                   maximum(dp_all), maximum(dp_all) / 100)
end

# ===========================================================================
# Load A/B coefficients from TOML
# ===========================================================================

function load_era5_vertical_coordinate(level_top::Int, level_bot::Int)
    coeff_path = joinpath(@__DIR__, "..", "..", "config", "era5_L137_coefficients.toml")
    isfile(coeff_path) || error("ERA5 coefficients not found: $coeff_path")
    cfg = TOML.parsefile(coeff_path)
    a_all = Float64.(cfg["coefficients"]["a"])   # 138 values (n=0..137)
    b_all = Float64.(cfg["coefficients"]["b"])
    # Level k uses interfaces k and k+1 (1-based)
    i_start = level_top
    i_end   = level_bot + 1
    a_ifc = a_all[i_start:i_end]   # Nz+1 values
    b_ifc = b_all[i_start:i_end]
    return HybridSigmaPressure(a_ifc, b_ifc)
end

# ===========================================================================
# Main: single file conversion
# ===========================================================================

function convert_merged(nc_path::String, bin_path::String, min_dp::Float64)
    @info "Reading: $nc_path"
    ds = NCDataset(nc_path, "r")

    level_top = Int(get(ds.attrib, "level_top", 1))
    level_bot = Int(get(ds.attrib, "level_bot", 137))
    ft_str = get(ds.attrib, "float_type", "Float32")
    FT = ft_str == "Float64" ? Float64 : Float32

    # Build native vertical coordinate and merge
    vc_native = load_era5_vertical_coordinate(level_top, level_bot)
    merged_vc, merge_map = merge_thin_levels(vc_native; min_thickness_Pa=min_dp)
    print_merge_summary(vc_native, merged_vc, merge_map, min_dp)

    # Write merged binary
    write_v2_binary(bin_path, ds, merged_vc, merge_map, min_dp)
    close(ds)
end

# ===========================================================================
# Batch conversion
# ===========================================================================

function batch_convert(dir::String, min_dp::Float64)
    files = filter(readdir(dir; join=true)) do f
        endswith(f, ".nc") && contains(basename(f), "massflux")
    end
    sort!(files; by=basename)
    isempty(files) && (@warn "No massflux*.nc files found in $dir"; return)

    dp_tag = @sprintf("merged%dPa", round(Int, min_dp))
    @info "Batch: $(length(files)) files, min_dp=$(min_dp) Pa → tag=$(dp_tag)"

    for nc_path in files
        # e.g. massflux_era5_spectral_202112_float32.nc → ..._merged1000Pa_float32.bin
        bin_name = replace(basename(nc_path), r"_float(32|64)\.nc$" =>
            SubstitutionString("_$(dp_tag)_float\\1.bin"))
        bin_path = joinpath(dir, bin_name)
        if isfile(bin_path) && filesize(bin_path) > 0
            @info "  Skipping (exists): $(basename(bin_path))"
            continue
        end
        convert_merged(nc_path, bin_path, min_dp)
    end
end

# ===========================================================================
# Entry point
# ===========================================================================

# Parse --min-dp from args
function parse_min_dp(args)
    for a in args
        m = match(r"^--min-dp=(\d+\.?\d*)$", a)
        m !== nothing && return parse(Float64, m[1])
    end
    return parse(Float64, get(ENV, "MIN_DP", "1000"))
end

@warn """
================================================================================
convert_merged_massflux_to_binary.jl is OBSOLETE (deprecated 2026-04-06).

This script PICKS native cm at merged interface boundaries and smears the
surface-BC residual via correct_cm_residual!. The result does NOT satisfy local
continuity with the merged am/bm and causes polar-cap mass drainage in the
runtime model (see era5_f64_debug_moist.toml failure mode).

USE INSTEAD:
  - preprocess_spectral_v4_binary.jl  (core only, spectral GRIB → binary direct)
  - preprocess_era5_daily.jl          (full physics: convection, surface, QV)

Both replacements use recompute_cm_from_divergence! which produces a cm that
exactly satisfies continuity with the merged am/bm.

Continuing in 5 seconds (Ctrl-C to abort)...
================================================================================
"""
sleep(5)

min_dp = parse_min_dp(ARGS)
file_args = filter(a -> !startswith(a, "--"), ARGS)

massflux_dir = get(ENV, "MASSFLUX_DIR", "")

if !isempty(massflux_dir)
    batch_convert(expanduser(massflux_dir), min_dp)
elseif !isempty(file_args)
    nc_path = expanduser(file_args[1])
    bin_path = if length(file_args) >= 2
        expanduser(file_args[2])
    else
        # Output alongside source, following data layout convention
        replace(nc_path, r"\.nc$" => @sprintf("_merged%dPa.bin", round(Int, min_dp)))
    end
    isfile(nc_path) || error("Input NetCDF not found: $nc_path")
    convert_merged(nc_path, bin_path, min_dp)
else
    println("""
    Convert ERA5 mass-flux NetCDF → merged-level binary (v2)

    Usage:
      julia -t10 --project=. $(PROGRAM_FILE) input.nc [output.bin] [--min-dp=1000]
      MIN_DP=500 MASSFLUX_DIR=~/data/AtmosTransport/met/era5/preprocessed_spectral_catrine \\
          julia -t10 --project=. $(PROGRAM_FILE)

    Default min_dp: 1000 Pa (10 hPa). Try 500 Pa (5 hPa) for finer resolution.
    Output goes alongside source NetCDF (same directory).
    """)
end
