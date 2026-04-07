#!/usr/bin/env julia
# ===========================================================================
# Repair an existing v2/v3 mass-flux binary by recomputing cm from continuity.
#
# Reads a binary produced by the (now obsolete) convert_merged_massflux_to_binary.jl
# pipeline. For each window, runs the same Poisson balance + cm recompute that
# preprocess_spectral_v4_binary.jl uses. Writes the result to a NEW binary
# file with provenance fields.
#
# This is a 100% drop-in fix for existing binaries that don't satisfy
# cm-continuity. Much faster than re-running the spectral preprocessor
# (~30 seconds vs ~12 minutes per day on this host).
#
# Usage:
#   julia -t8 --project=. scripts/preprocessing/repair_v2_binary_cm.jl \
#       INPUT.bin OUTPUT.bin
#
# Requirements:
#   - Input binary must have v2 header with embedded A_ifc, B_ifc
#   - Input must have m, am, bm, cm, ps fields
#   - The repair Poisson-balances am, bm against dm/dt = m_next - m_curr,
#     then recomputes cm from the balanced am/bm.
#   - For the LAST window of the input, dm/dt is taken as zero (no next).
# ===========================================================================

using JSON3
using FFTW
using Mmap
using Dates
using Printf
using Logging

# Flushing logger so output is visible when piped
struct _FlushingLogger{L<:AbstractLogger} <: AbstractLogger
    inner :: L
end
Logging.min_enabled_level(l::_FlushingLogger) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::_FlushingLogger, level, _module, group, id) =
    Logging.shouldlog(l.inner, level, _module, group, id)
Logging.catch_exceptions(l::_FlushingLogger) = Logging.catch_exceptions(l.inner)
function Logging.handle_message(l::_FlushingLogger, level, message, _module, group, id, file, line; kwargs...)
    Logging.handle_message(l.inner, level, message, _module, group, id, file, line; kwargs...)
    try; flush(stderr); catch; end
    try; flush(stdout); catch; end
    return nothing
end

# ===========================================================================
# Poisson balance (verbatim copy of preprocess_spectral_v4_binary.jl)
# ===========================================================================
function balance_mass_fluxes!(am::Array{FT,3}, bm::Array{FT,3},
                               dm_dt::Array{FT,3}) where FT
    Nx = size(am, 1) - 1
    Ny = size(bm, 2) - 1
    Nz = size(am, 3)

    fac = Array{Float64}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        fac[i, j] = 2.0 * (cos(2π * (i - 1) / Nx) + cos(2π * (j - 1) / Ny) - 2.0)
    end
    fac[1, 1] = 1.0

    psi = Array{Float64}(undef, Nx, Ny)
    residual = Array{Float64}(undef, Nx, Ny)

    n_balanced = 0
    max_residual = 0.0

    for k in 1:Nz
        @inbounds for j in 1:Ny, i in 1:Nx
            conv = (Float64(am[i, j, k]) - Float64(am[i+1, j, k])) +
                   (Float64(bm[i, j, k]) - Float64(bm[i, j+1, k]))
            residual[i, j] = conv - Float64(dm_dt[i, j, k])
        end

        max_res_k = maximum(abs, residual)
        max_residual = max(max_residual, max_res_k)
        max_res_k < 1e-10 && continue

        A = fft(complex.(residual))
        @inbounds for j in 1:Ny, i in 1:Nx
            A[i, j] /= fac[i, j]
        end
        A[1, 1] = 0.0 + 0.0im
        psi .= real.(ifft(A))

        @inbounds for j in 1:Ny
            u_wrap = psi[1, j] - psi[Nx, j]
            for i in 2:Nx
                du = (psi[i, j] - psi[i-1, j]) - u_wrap
                am[i, j, k] += FT(du)
            end
            am[1, j, k] += FT(0)
            am[Nx+1, j, k] += FT(0)
        end

        @inbounds for i in 1:Nx
            v_wrap = psi[i, 1] - psi[i, Ny]
            for j in 2:Ny
                dv = (psi[i, j] - psi[i, j-1]) - v_wrap
                bm[i, j, k] += FT(dv)
            end
        end

        n_balanced += 1
    end

    @info "  Poisson balance: corrected $n_balanced/$Nz levels, " *
          "max pre-balance residual: $(round(max_residual, sigdigits=3)) kg"
end

function recompute_cm_from_divergence!(cm::Array{FT,3}, am::Array{FT,3},
                                       bm::Array{FT,3};
                                       B_ifc::Vector{<:Real}) where FT
    Nx = size(cm, 1)
    Ny = size(cm, 2)
    Nz = size(cm, 3) - 1
    fill!(cm, zero(FT))

    @inbounds for j in 1:Ny, i in 1:Nx
        pit = 0.0
        for k in 1:Nz
            pit += (Float64(am[i+1, j, k]) - Float64(am[i, j, k])) +
                   (Float64(bm[i, j+1, k]) - Float64(bm[i, j, k]))
        end
        acc = 0.0
        for k in 1:Nz
            div_h = (Float64(am[i+1, j, k]) - Float64(am[i, j, k])) +
                    (Float64(bm[i, j+1, k]) - Float64(bm[i, j, k]))
            acc = acc - div_h + (Float64(B_ifc[k+1]) - Float64(B_ifc[k])) * pit
            cm[i, j, k+1] = FT(acc)
        end
    end
end

# ===========================================================================
# Main
# ===========================================================================
function main()
    base_logger = ConsoleLogger(stderr, Logging.Info; show_limited=false)
    global_logger(_FlushingLogger(base_logger))

    if length(ARGS) < 2
        println(stderr, """
        Usage: julia -t8 --project=. $(PROGRAM_FILE) INPUT.bin OUTPUT.bin

        Repairs an obsolete v2/v3 binary by:
        1. Reading m, am, bm, cm, ps for every window
        2. Computing dm/dt = m_next - m_curr (zero for last window)
        3. Poisson-balancing am, bm so conv = dm/dt at every cell
        4. Recomputing cm from continuity using B-correction
        5. Writing repaired v4 binary with provenance fields
        """)
        exit(1)
    end

    in_path  = expanduser(ARGS[1])
    out_path = expanduser(ARGS[2])
    isfile(in_path) || error("Input not found: $in_path")
    in_path == out_path && error("Input and output must differ")

    @info "Repairing binary: $in_path"
    @info "  Output: $out_path"

    # ----- Read header -----
    io_in = open(in_path, "r")
    max_hdr = 16384
    file_sz = filesize(in_path)
    hdr_bytes = read(io_in, min(max_hdr, file_sz))
    json_end = something(findfirst(==(0x00), hdr_bytes), length(hdr_bytes) + 1) - 1
    hdr = JSON3.read(String(hdr_bytes[1:json_end]))
    hdr_size_in = Int(get(hdr, :header_bytes, 16384))
    Nx = Int(hdr.Nx); Ny = Int(hdr.Ny); Nz = Int(hdr.Nz); Nt = Int(hdr.Nt)
    n_m  = Int(hdr.n_m)
    n_am = Int(hdr.n_am)
    n_bm = Int(hdr.n_bm)
    n_cm = Int(hdr.n_cm)
    n_ps = Int(hdr.n_ps)
    elems_per_window_in = n_m + n_am + n_bm + n_cm + n_ps
    # Skip optional fields (qv, cmfmc, surface, etc.) — we don't carry them over
    elems_per_window_in_total = Int(get(hdr, :window_bytes, 0)) ÷ 4
    if elems_per_window_in_total == 0
        # Fallback: derive from file size
        elems_per_window_in_total = (file_sz - hdr_size_in) ÷ (4 * Nt)
    end

    A_ifc = Float64.(collect(hdr.A_ifc))
    B_ifc = Float64.(collect(hdr.B_ifc))
    @info "  Grid: $(Nx)×$(Ny)×$(Nz), $Nt windows, elems_per_window=$elems_per_window_in_total"

    seek(io_in, hdr_size_in)
    raw = Mmap.mmap(io_in, Vector{Float32}, elems_per_window_in_total * Nt, hdr_size_in)

    # ----- Read all windows into memory -----
    FT = Float32
    all_m  = [Array{FT}(undef, Nx, Ny, Nz) for _ in 1:Nt]
    all_am = [Array{FT}(undef, Nx + 1, Ny, Nz) for _ in 1:Nt]
    all_bm = [Array{FT}(undef, Nx, Ny + 1, Nz) for _ in 1:Nt]
    all_cm = [Array{FT}(undef, Nx, Ny, Nz + 1) for _ in 1:Nt]
    all_ps = [Array{FT}(undef, Nx, Ny) for _ in 1:Nt]

    @info "  Reading $Nt windows..."
    for win in 1:Nt
        off = (win - 1) * elems_per_window_in_total
        copyto!(all_m[win],  1, raw, off + 1, n_m); off += n_m
        copyto!(all_am[win], 1, raw, off + 1, n_am); off += n_am
        copyto!(all_bm[win], 1, raw, off + 1, n_bm); off += n_bm
        copyto!(all_cm[win], 1, raw, off + 1, n_cm); off += n_cm
        copyto!(all_ps[win], 1, raw, off + 1, n_ps)
    end
    close(io_in)
    @info "  Read complete"

    # ----- Per-window: Poisson balance + cm recompute -----
    dm_dt_buf = Array{FT}(undef, Nx, Ny, Nz)
    for win in 1:Nt
        @info "  Window $win/$Nt: balance + recompute cm"
        if win < Nt
            dm_dt_buf .= all_m[win + 1] .- all_m[win]
        else
            fill!(dm_dt_buf, zero(FT))
        end
        balance_mass_fluxes!(all_am[win], all_bm[win], dm_dt_buf)
        # Re-zero pole bm faces
        @views all_bm[win][:, 1, :]    .= zero(FT)
        @views all_bm[win][:, Ny+1, :] .= zero(FT)
        recompute_cm_from_divergence!(all_cm[win], all_am[win], all_bm[win]; B_ifc=B_ifc)
        @views all_cm[win][:, :, 1]      .= zero(FT)
        @views all_cm[win][:, :, Nz + 1] .= zero(FT)
    end

    # ----- Compute dam, dbm, dm for v4 -----
    @info "  Computing v4 deltas..."
    all_dam = [Array{FT}(undef, Nx + 1, Ny, Nz) for _ in 1:Nt]
    all_dbm = [Array{FT}(undef, Nx, Ny + 1, Nz) for _ in 1:Nt]
    all_dm  = [Array{FT}(undef, Nx, Ny, Nz) for _ in 1:Nt]
    for win in 1:Nt
        if win < Nt
            all_dam[win] .= all_am[win + 1] .- all_am[win]
            all_dbm[win] .= all_bm[win + 1] .- all_bm[win]
            all_dm[win]  .= all_m[win + 1]  .- all_m[win]
        else
            fill!(all_dam[win], zero(FT))
            fill!(all_dbm[win], zero(FT))
            fill!(all_dm[win], zero(FT))
        end
    end

    # ----- Write repaired binary -----
    HEADER_SIZE = 16384
    n_dam = (Nx + 1) * Ny * Nz
    n_dbm = Nx * (Ny + 1) * Nz
    n_dm  = Nx * Ny * Nz
    elems_per_window_out = n_m + n_am + n_bm + n_cm + n_ps + n_dam + n_dbm + n_dm
    bytes_per_window = elems_per_window_out * sizeof(FT)
    total_bytes = HEADER_SIZE + bytes_per_window * Nt

    # Provenance fields
    script_path = abspath(@__FILE__)
    script_mtime = mtime(script_path)
    git_commit = try
        readchomp(`git -C $(dirname(script_path)) rev-parse HEAD`)
    catch
        "unknown"
    end
    git_dirty = try
        !isempty(readchomp(`git -C $(dirname(script_path)) status --porcelain`))
    catch
        false
    end

    out_hdr = Dict{String,Any}(
        "magic" => "MFLX", "version" => 4, "header_bytes" => HEADER_SIZE,
        "Nx" => Nx, "Ny" => Ny, "Nz" => Nz, "Nz_native" => Nz, "Nt" => Nt,
        "float_type" => "Float32", "float_bytes" => sizeof(FT),
        "window_bytes" => bytes_per_window,
        "n_m" => n_m, "n_am" => n_am, "n_bm" => n_bm, "n_cm" => n_cm, "n_ps" => n_ps,
        "n_qv" => 0, "n_cmfmc" => 0,
        "n_entu" => 0, "n_detu" => 0, "n_entd" => 0, "n_detd" => 0,
        "n_pblh" => 0, "n_t2m" => 0, "n_ustar" => 0, "n_hflux" => 0,
        "n_temperature" => 0,
        "n_dam" => n_dam, "n_dbm" => n_dbm, "n_dm" => n_dm,
        "include_flux_delta" => true,
        "include_qv" => false, "include_cmfmc" => false,
        "include_tm5conv" => false, "include_surface" => false,
        "include_temperature" => false,
        "dt_seconds" => Float64(get(hdr, :dt_seconds, 900.0)),
        "half_dt_seconds" => Float64(get(hdr, :half_dt_seconds, 450.0)),
        "steps_per_met_window" => Int(get(hdr, :steps_per_met_window, 4)),
        "level_top" => Int(get(hdr, :level_top, 1)),
        "level_bot" => Int(get(hdr, :level_bot, 137)),
        "lons" => collect(hdr.lons),
        "lats" => collect(hdr.lats),
        "A_ifc" => A_ifc, "B_ifc" => B_ifc,
        "merge_min_thickness_Pa" => Float64(get(hdr, :merge_min_thickness_Pa, 1000.0)),
        "var_names" => ["m","am","bm","cm","ps","dam","dbm","dm"],
        "date" => String(get(hdr, :date, "unknown")),
        "grid_convention" => "TM5",
        # Provenance
        "script_path" => script_path,
        "script_mtime_unix" => script_mtime,
        "git_commit" => git_commit,
        "git_dirty" => git_dirty,
        "creation_time" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        "repaired_from" => abspath(in_path),
    )
    out_hdr_json = JSON3.write(out_hdr)
    length(out_hdr_json) < HEADER_SIZE ||
        error("Header JSON too large: $(length(out_hdr_json)) >= $HEADER_SIZE")

    @info "  Writing repaired binary ($(round(total_bytes / 1e9, digits=2)) GB)..."
    open(out_path, "w") do io_out
        hdr_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(hdr_buf, 1, Vector{UInt8}(out_hdr_json), 1, length(out_hdr_json))
        write(io_out, hdr_buf)
        for win in 1:Nt
            write(io_out, all_m[win])
            write(io_out, all_am[win])
            write(io_out, all_bm[win])
            write(io_out, all_cm[win])
            write(io_out, all_ps[win])
            write(io_out, all_dam[win])
            write(io_out, all_dbm[win])
            write(io_out, all_dm[win])
        end
    end

    @info "  Repair complete: $out_path"
end

main()
