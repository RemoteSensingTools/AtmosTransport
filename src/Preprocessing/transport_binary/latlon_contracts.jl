# Shared structured lat-lon replay, Poisson-balance, and v4 writer helpers.

function next_day_merged_fields(next_day_hour0,
                                date::Date,
                                grid::LatLonTargetGeometry,
                                vertical,
                                settings,
                                transform::SpectralTransformWorkspace,
                                merged::MergeWorkspace{FT},
                                qv::AbstractQVWorkspace{FT},
                                ps_offsets::Vector{Float64}) where FT
    next_day_hour0 === nothing && return nothing
    Nx = size(transform.sp, 1)
    Ny = size(transform.sp, 2)
    @info "  Computing next day hour 0 for last-window delta..."

    spectral_to_native_fields!(
        transform.m_arr, transform.am_arr, transform.bm_arr, transform.cm_arr, transform.sp,
        transform.u_cc, transform.v_cc, transform.u_stag, transform.v_stag, transform.dp,
        next_day_hour0.lnsp, next_day_hour0.vo, next_day_hour0.d,
        next_day_hour0.T, vertical.level_range, vertical.ab, grid, settings.half_dt,
        transform.P_buf, transform.fft_buf, transform.field_2d,
        transform.P_buf_t, transform.fft_buf_t, transform.fft_out_t,
        transform.u_spec_t, transform.v_spec_t, transform.field_2d_t,
        transform.bfft_plans)

    read_next_day_qv!(qv, date, settings, Nx, Ny, vertical.Nz_native)
    apply_mass_fix_if_needed!(qv, transform, grid, vertical, settings, ps_offsets, length(ps_offsets))
    apply_dry_basis_if_needed!(settings.mass_basis, transform, qv)
    merge_native_window!(merged, transform, qv, vertical, settings)

    return (m=copy(merged.m_merged),
            am=copy(merged.am_merged),
            bm=copy(merged.bm_merged),
            cm=copy(merged.cm_merged),
            qv=settings.include_qv ? copy(qv.qv_merged) : nothing)
end

"""
    fill_window_mass_tendency!(dm_dt_buf, storage, last_hour_next, win_idx, steps_per_window)

Fill the cell-mass target used by the Poisson horizontal-flux balance step.

The stored `am/bm/cm` fields are half-sweep transport amounts. A full Strang
substep applies the horizontal fluxes twice, so the Poisson target must be the
forward window mass difference divided by `2 * steps_per_window`.
"""
function fill_window_mass_tendency!(dm_dt_buf::Array{FT, 3},
                                    storage::WindowStorage{FT},
                                    last_hour_next,
                                    win_idx::Int,
                                    steps_per_window::Int) where FT
    Nt = length(storage.all_m)
    scale = poisson_balance_target_scale(steps_per_window, FT)

    if win_idx < Nt
        dm_dt_buf .= (storage.all_m[win_idx + 1] .- storage.all_m[win_idx]) .* scale
    elseif last_hour_next !== nothing
        dm_dt_buf .= (last_hour_next.m .- storage.all_m[win_idx]) .* scale
    else
        fill!(dm_dt_buf, zero(FT))
    end

    return nothing
end

"""
    verify_storage_continuity_ll!(storage, last_hour_next, steps_per_window, ::Type{FT})

Plan 39 Commit E — write-time replay gate for structured LL storage.
Iterates every window k and asserts

    m[k] − 2·steps·(∇·am + ∇·bm + ∂_k cm) ≈ m[k+1]    (k < Nt)
    m[Nt] − 2·steps·(∇·am + ∇·bm + ∂_k cm) ≈ last_hour_next.m   (k == Nt, if available)
    m[Nt] − 2·steps·(∇·am + ∇·bm + ∂_k cm) ≈ m[Nt]              (otherwise zero-tendency fallback)

to within a Poisson-balance tolerance floor derived from `FT` (roughly
`1e-10` for `Float64`, `1e-4` for `Float32`). Errors loudly with a per-window
diagnostic if the contract is violated — this was the gate that would have
caught the dry-basis Δb×pit closure bug before it reached the runtime.

Bypass with env var `ATMOSTR_NO_WRITE_REPLAY_CHECK=1` for diagnostic runs.
"""
function verify_storage_continuity_ll!(storage::WindowStorage{FT},
                                        last_hour_next,
                                        steps_per_window::Int,
                                        ::Type{FT}) where FT
    if get(ENV, "ATMOSTR_NO_WRITE_REPLAY_CHECK", "0") == "1"
        @info "  Write-time replay gate SKIPPED (ATMOSTR_NO_WRITE_REPLAY_CHECK=1)"
        return nothing
    end
    Nt = length(storage.all_m)
    Nt == 0 && return nothing
    tol_rel = replay_tolerance(FT)
    div_scratch = Array{Float64}(undef, size(storage.all_m[1]))
    layout = structured_replay_layout()
    run_replay_gate(Nt; tol_rel=tol_rel,
                    summary_label="  Write-time replay gate",
                    failure_prefix="Write-time replay gate") do k
        m_next = if k < Nt
            storage.all_m[k + 1]
        elseif last_hour_next !== nothing
            last_hour_next.m
        else
            storage.all_m[k]
        end
        verify_window_continuity(layout, div_scratch,
                                 storage.all_m[k],
                                 storage.all_cm[k],
                                 m_next,
                                 steps_per_window,
                                 storage.all_am[k],
                                 storage.all_bm[k])
    end
    return nothing
end

"""
    apply_poisson_balance!(storage, last_hour_next, steps_per_window)

Apply the TM5-style Poisson horizontal-flux correction to every stored window
so horizontal convergence matches the per-half-sweep mass target implied by the
forward window endpoints.
"""
function apply_poisson_balance!(storage::WindowStorage{FT},
                                last_hour_next,
                                steps_per_window::Int) where FT
    Nx, Ny, Nz = size(storage.all_m[1])
    dm_dt_buf = Array{FT}(undef, Nx, Ny, Nz)
    div_scratch = Array{Float64}(undef, Nx, Ny, Nz)
    poisson_ws = LLPoissonWorkspace(Nx, Ny)
    replay_layout = structured_replay_layout()

    @info "  Applying Poisson mass flux balance..."
    for win_idx in eachindex(storage.all_m)
        fill_window_mass_tendency!(dm_dt_buf, storage, last_hour_next, win_idx, steps_per_window)
        balance_mass_fluxes!(storage.all_am[win_idx], storage.all_bm[win_idx], dm_dt_buf, poisson_ws)
        @views storage.all_bm[win_idx][:, 1, :] .= zero(FT)
        @views storage.all_bm[win_idx][:, Ny + 1, :] .= zero(FT)
        # Plan 39 dry-basis fix (2026-04-22): use explicit-dm closure, not
        # the hybrid Δb×pit one. The Δb×pit closure assumes
        # dm[k] = dB[k] × Σ_k dm[k], which holds under moist hybrid coords
        # but is violated by ~27% under dry basis because qv[k] varies with
        # level. That mismatch caused the 0.75% day-boundary air_mass jump
        # observed on F64 probe; see plan39_reconnect.md memory entry.
        recompute_cm_from_dm_target!(replay_layout, div_scratch,
                                     storage.all_cm[win_idx], storage.all_m[win_idx], dm_dt_buf,
                                     storage.all_am[win_idx], storage.all_bm[win_idx])
        @views storage.all_cm[win_idx][:, :, 1] .= zero(FT)
        @views storage.all_cm[win_idx][:, :, Nz + 1] .= zero(FT)
    end

    # Plan 39 Commit E: write-time replay gate. Under the `:window_constant`
    # contract, starting from `storage.all_m[k]` and integrating the stored
    # fluxes (am, bm, cm) over one window via palindrome continuity must
    # reproduce `storage.all_m[k+1]` (or `last_hour_next.m` for k=Nt) to
    # within the Poisson-balance tolerance floor. Fails loudly if the fix
    # regresses or a new preprocessor path breaks the contract.
    verify_storage_continuity_ll!(storage, last_hour_next, steps_per_window, FT)
    @info "  Poisson balance complete for $(length(storage.all_m)) windows"

    return nothing
end

"""
    compute_window_deltas!(merged, storage, win_idx, last_hour_next)

Form the forward-in-time `dam`, `dbm`, `dcm`, and `dm` payloads for one window.
"""
function compute_window_deltas!(merged::MergeWorkspace{FT},
                                storage::WindowStorage{FT},
                                win_idx::Int,
                                last_hour_next) where FT
    Nt = length(storage.all_m)

    if win_idx < Nt
        merged.dam_merged .= storage.all_am[win_idx + 1] .- storage.all_am[win_idx]
        merged.dbm_merged .= storage.all_bm[win_idx + 1] .- storage.all_bm[win_idx]
        merged.dcm_merged .= storage.all_cm[win_idx + 1] .- storage.all_cm[win_idx]
        merged.dm_merged  .= storage.all_m[win_idx + 1]  .- storage.all_m[win_idx]
    elseif last_hour_next !== nothing
        merged.dam_merged .= last_hour_next.am .- storage.all_am[win_idx]
        merged.dbm_merged .= last_hour_next.bm .- storage.all_bm[win_idx]
        merged.dcm_merged .= last_hour_next.cm .- storage.all_cm[win_idx]
        merged.dm_merged  .= last_hour_next.m  .- storage.all_m[win_idx]
    else
        fill!(merged.dam_merged, zero(FT))
        fill!(merged.dbm_merged, zero(FT))
        fill!(merged.dcm_merged, zero(FT))
        fill!(merged.dm_merged, zero(FT))
    end

    return nothing
end

function fill_qv_endpoints!(storage::WindowStorage{FT}, last_hour_next) where FT
    isempty(storage.all_qv_start) && return nothing
    Nt = length(storage.all_qv_start)

    for win_idx in 1:Nt-1
        storage.all_qv_end[win_idx] = copy(storage.all_qv_start[win_idx + 1])
    end

    if last_hour_next !== nothing && hasproperty(last_hour_next, :qv) && last_hour_next.qv !== nothing
        storage.all_qv_end[Nt] = copy(last_hour_next.qv)
    else
        storage.all_qv_end[Nt] = copy(storage.all_qv_start[Nt])
    end

    return nothing
end

"""
    write_window!(io, win_idx, storage, settings, merged, last_hour_next) -> Int64

Write one window's payload blocks to the output stream in v4 on-disk order.
"""
function write_window!(io::IO,
                       win_idx::Int,
                       storage::WindowStorage{FT},
                       settings,
                       merged::MergeWorkspace{FT},
                       last_hour_next) where FT
    bytes_written = Int64(0)
    bytes_written += write_array!(io, storage.all_m[win_idx])
    bytes_written += write_array!(io, storage.all_am[win_idx])
    bytes_written += write_array!(io, storage.all_bm[win_idx])
    bytes_written += write_array!(io, storage.all_cm[win_idx])
    bytes_written += write_array!(io, storage.all_ps[win_idx])
    if settings.include_qv
        bytes_written += write_array!(io, storage.all_qv_start[win_idx])
        bytes_written += write_array!(io, storage.all_qv_end[win_idx])
    end

    compute_window_deltas!(merged, storage, win_idx, last_hour_next)
    bytes_written += write_array!(io, merged.dam_merged)
    bytes_written += write_array!(io, merged.dbm_merged)
    bytes_written += write_array!(io, merged.dcm_merged)
    bytes_written += write_array!(io, merged.dm_merged)

    # Plan 24 Commit 4: TM5 convection sections (order must match
    # _transport_push_optional_sections! in TransportBinary.jl:557-578).
    if settings.tm5_convection_enable
        bytes_written += write_array!(io, storage.all_entu[win_idx])
        bytes_written += write_array!(io, storage.all_detu[win_idx])
        bytes_written += write_array!(io, storage.all_entd[win_idx])
        bytes_written += write_array!(io, storage.all_detd[win_idx])
    end

    return bytes_written
end

"""
    write_day_binary!(bin_path, header_json, storage, settings, merged, last_hour_next)

Write the padded header and all window payloads for one daily binary file.
Returns the total number of bytes written.
"""
function write_day_binary!(bin_path::String,
                           header_json,
                           storage::WindowStorage{FT},
                           settings,
                           merged::MergeWorkspace{FT},
                           last_hour_next) where FT
    @info "  Writing binary..."
    bytes_written = Int64(0)

    open(bin_path, "w") do io
        hdr_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(hdr_buf, 1, Vector{UInt8}(header_json), 1, length(header_json))
        write(io, hdr_buf)
        bytes_written += HEADER_SIZE

        for win_idx in eachindex(storage.all_m)
            bytes_written += write_window!(io, win_idx, storage, settings, merged, last_hour_next)
        end

        flush(io)
    end

    return bytes_written
end
