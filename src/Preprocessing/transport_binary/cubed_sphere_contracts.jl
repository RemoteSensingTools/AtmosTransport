# Cubed-sphere replay helpers shared by spectral and regrid CS preprocessors.

"""
    fill_cs_window_mass_tendency!(dm_panels, m_cur, m_next, steps_per_window)

Fill the CS Poisson-balance target for one window.

`m_cur` and `m_next` are explicit endpoint masses. The stored CS horizontal
fluxes are half-sweep amounts under Strang splitting, so the target is
`(m_next - m_cur) / (2 * steps_per_window)` per panel cell.
"""
@inline function fill_cs_window_mass_tendency!(dm_panels::NTuple{NP, <:AbstractArray{FT, 3}},
                                               m_cur::NTuple{NP, <:AbstractArray{FT, 3}},
                                               m_next::NTuple{NP, <:AbstractArray{FT, 3}},
                                               steps_per_window::Int) where {FT, NP}
    inv_two_steps = one(FT) / FT(2 * steps_per_window)
    for p in 1:NP
        @inbounds for idx in eachindex(dm_panels[p])
            dm_panels[p][idx] = (m_next[p][idx] - m_cur[p][idx]) * inv_two_steps
        end
    end
    return nothing
end

"""
    convert_cs_mass_target_to_delta!(m_target, m_cur)

Convert an in-place CS endpoint target into the on-disk `dm` payload.

The CS writer stores forward endpoint differences. Call this only after all
balance and replay checks that still need the absolute target endpoint.
"""
@inline function convert_cs_mass_target_to_delta!(m_target::NTuple{NP, <:AbstractArray{FT, 3}},
                                                  m_cur::NTuple{NP, <:AbstractArray{FT, 3}}) where {FT, NP}
    for p in 1:NP
        @inbounds for idx in eachindex(m_target[p])
            m_target[p][idx] -= m_cur[p][idx]
        end
    end
    return nothing
end

"""
    verify_write_replay_cs!(m_cur, am, bm, cm, m_next, steps_per_window, tol_rel, win_idx)

Run the CS write-time replay gate for one window and return its diagnostic.

The check integrates the stored panel-local fluxes from `m_cur` under the
runtime palindrome-continuity contract and verifies that the result matches the
explicit endpoint `m_next`. A failure here means the binary would produce a
runtime day-boundary or window-boundary mass inconsistency.
"""
function verify_write_replay_cs!(m_cur::NTuple{NP, <:AbstractArray{FT, 3}},
                                 am::NTuple{NP, <:AbstractArray},
                                 bm::NTuple{NP, <:AbstractArray},
                                 cm::NTuple{NP, <:AbstractArray},
                                 m_next::NTuple{NP, <:AbstractArray},
                                 steps_per_window::Int,
                                 tol_rel::Real,
                                 win_idx::Int) where {FT, NP}
    diag = verify_window_continuity_cs(m_cur, am, bm, cm, m_next, steps_per_window)
    diag.max_rel_err <= tol_rel ||
        error("Write-time replay gate FAILED for CS window $(win_idx): " *
              "rel=$(diag.max_rel_err) > tol=$(tol_rel) at cell $(diag.worst_idx) " *
              "(abs=$(diag.max_abs_err) kg). Stored CS fluxes do not integrate to " *
              "the target mass endpoint under palindrome continuity.")
    return diag
end
