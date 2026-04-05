# ===========================================================================
# Mass-only CFL pilot for LL adaptive refinement (TM5-style Check_CFL)
#
# Replays the Strang split sequence on m only, checking local face
# inequalities at every operator sweep:
#   - |beta| = |flux / m_donor| > beta_limit (0.95) → flux empties donor
#   - m_new <= m_floor = 32*eps(FT)*max(m_ref, 1) → Float32 noise floor
#
# Returns a refinement factor r over powers of two up to `max_r`. The caller scales
# am/bm/cm by 1/r and runs n_sub*r substeps instead of n_sub.
#
# All GPU kernels are Float32-only. No Float64 promotion in kernels.
# Cluster sums use compensated (Kahan) summation via _kahan_add.
#
# Reference: TM5 advectm_cfl.F90:205 (Check_CFL), advectm_cfl.F90:2083
# (dynamvm mass-only Y checker).
# ===========================================================================

using KernelAbstractions: @kernel, @index, synchronize, get_backend

# Import _cluster_sum and _kahan_add from parent module
# (These are defined in mass_flux_advection.jl and Architectures.jl)

# =====================================================================
# Pilot kernels: mass update + fail check (no tracers, no slopes)
# =====================================================================

@kernel function _mass_pilot_x_kernel!(
    m_new, @Const(m), @Const(am), fail_flags,
    Nx, @Const(cluster_sizes), beta_limit, r_inv
)
    i, j, k = @index(Global, NTuple)
    FT = eltype(m)
    @inbounds begin
        fail = false
        r = Int(cluster_sizes[j])

        if r == 1
            # Uniform row
            ip = i == Nx ? 1 : i + 1
            am_l = am[i, j, k] * r_inv
            am_r = am[ip, j, k] * r_inv
            m_new[i, j, k] = m[i, j, k] + am_l - am_r
            m_floor = FT(32) * eps(FT) * max(abs(m[i, j, k]), one(FT))
            if !(m_new[i, j, k] > m_floor)
                fail = true
            end

            # Beta at left face
            im = i == 1 ? Nx : i - 1
            m_donor = am[i, j, k] >= zero(FT) ? m[im, j, k] : m[i, j, k]
            if m_donor > zero(FT) && abs(am_l) / m_donor > beta_limit
                fail = true
            end
            # Beta at right face
            m_donor = am[ip, j, k] >= zero(FT) ? m[i, j, k] : m[ip == Nx ? 1 : ip, j, k]
            if m_donor > zero(FT) && abs(am_r) / m_donor > beta_limit
                fail = true
            end
        else
            # Reduced row: cluster-level update
            Nx_red = Nx ÷ r
            ic = (i - 1) ÷ r + 1
            m_ic = _cluster_sum(m, ic, j, k, r)

            am_l_idx = (ic - 1) * r + 1
            am_r_idx = ic * r + 1
            am_l = am[am_l_idx, j, k] * r_inv
            am_r = am[am_r_idx > Nx ? 1 : am_r_idx, j, k] * r_inv

            delta_m = am_l - am_r
            frac_m = abs(m_ic) > eps(FT) ? m[i, j, k] / m_ic : one(FT) / FT(r)
            m_new[i, j, k] = (m_ic + delta_m) * frac_m
            m_floor = FT(32) * eps(FT) * max(abs(m[i, j, k]), one(FT))
            if !(m_new[i, j, k] > m_floor)
                fail = true
            end

            # Beta at cluster faces
            ic_m = ic == 1 ? Nx_red : ic - 1
            m_donor_l = _cluster_sum(m, am[am_l_idx, j, k] >= zero(FT) ? ic_m : ic, j, k, r)
            if m_donor_l > zero(FT) && abs(am_l) / m_donor_l > beta_limit
                fail = true
            end
            ic_p = ic == Nx_red ? 1 : ic + 1
            m_donor_r = _cluster_sum(m, am[am_r_idx > Nx ? 1 : am_r_idx, j, k] >= zero(FT) ? ic : ic_p, j, k, r)
            if m_donor_r > zero(FT) && abs(am_r) / m_donor_r > beta_limit
                fail = true
            end
        end

        fail_flags[i, j, k] = fail
    end
end

@kernel function _mass_pilot_y_kernel!(
    m_new, @Const(m), @Const(bm), fail_flags, Ny, beta_limit, r_inv
)
    i, j, k = @index(Global, NTuple)
    FT = eltype(m)
    @inbounds begin
        fail = false

        bm_s = j > 1 ? bm[i, j, k] * r_inv : zero(FT)
        bm_n = j < Ny ? bm[i, j + 1, k] * r_inv : zero(FT)
        m_new[i, j, k] = m[i, j, k] + bm_s - bm_n
        m_floor = FT(32) * eps(FT) * max(abs(m[i, j, k]), one(FT))
        if !(m_new[i, j, k] > m_floor)
            fail = true
        end

        # South face beta
        if j > 1
            m_donor = bm[i, j, k] >= zero(FT) ? m[i, j - 1, k] : m[i, j, k]
            if m_donor > zero(FT) && abs(bm_s) / m_donor > beta_limit
                fail = true
            end
        end
        # North face beta
        if j < Ny
            m_donor = bm[i, j + 1, k] >= zero(FT) ? m[i, j, k] : m[i, j + 1, k]
            if m_donor > zero(FT) && abs(bm_n) / m_donor > beta_limit
                fail = true
            end
        end

        fail_flags[i, j, k] = fail
    end
end

@kernel function _mass_pilot_z_kernel!(
    m_new, @Const(m), @Const(cm), fail_flags, Nz, beta_limit, r_inv
)
    i, j, k = @index(Global, NTuple)
    FT = eltype(m)
    @inbounds begin
        fail = false

        cm_top = k > 1 ? cm[i, j, k] * r_inv : zero(FT)
        cm_bot = k < Nz ? cm[i, j, k + 1] * r_inv : zero(FT)
        m_new[i, j, k] = m[i, j, k] + cm_top - cm_bot
        m_floor = FT(32) * eps(FT) * max(abs(m[i, j, k]), one(FT))
        if !(m_new[i, j, k] > m_floor)
            fail = true
        end

        # Top face beta
        if k > 1
            m_donor = cm[i, j, k] > zero(FT) ? m[i, j, k - 1] : m[i, j, k]
            if m_donor > zero(FT) && abs(cm_top) / m_donor > beta_limit
                fail = true
            end
        end
        # Bottom face beta
        if k < Nz
            m_donor = cm[i, j, k + 1] > zero(FT) ? m[i, j, k] : m[i, j, k + 1]
            if m_donor > zero(FT) && abs(cm_bot) / m_donor > beta_limit
                fail = true
            end
        end

        fail_flags[i, j, k] = fail
    end
end

# =====================================================================
# Pilot driver: find minimum refinement factor r
# =====================================================================

# Lazy workspace
const _MASS_PILOT_M = Ref{Any}(nothing)
const _MASS_PILOT_MNEW = Ref{Any}(nothing)
const _MASS_PILOT_FAIL = Ref{Any}(nothing)

"""
    find_mass_cfl_refinement(m_ref, am, bm, cm, grid, cluster_sizes, n_sub;
                              beta_limit=0.95, max_r=64) -> Int

TM5-style evolving-mass preflight. Returns refinement factor r over powers of two
up to `max_r`.
Caller should scale am/bm/cm by 1/r and run n_sub*r substeps.
"""
function find_mass_cfl_refinement(m_ref::AbstractArray{FT,3},
                                   am, bm, cm, grid,
                                   cluster_sizes, n_sub;
                                   beta_limit::FT = FT(0.95),
                                   max_r::Int = 64) where FT
    Nx, Ny, Nz = size(m_ref)
    backend = get_backend(m_ref)

    # Lazy-allocate workspace
    if _MASS_PILOT_M[] === nothing || size(_MASS_PILOT_M[]) != size(m_ref)
        _MASS_PILOT_M[] = similar(m_ref)
        _MASS_PILOT_MNEW[] = similar(m_ref)
        _MASS_PILOT_FAIL[] = similar(m_ref, Bool)
    end
    m_pilot = _MASS_PILOT_M[]
    m_new = _MASS_PILOT_MNEW[]
    fail_flags = _MASS_PILOT_FAIL[]

    r = 1
    while r <= max_r
        copyto!(m_pilot, m_ref)
        r_inv = FT(1) / FT(r)
        n_eff = n_sub * r

        ok = true
        for _ in 1:n_eff
            # X-Y-Z-Z-Y-X Strang sequence
            for (op, dir_flux) in (
                (:X, am), (:Y, bm), (:Z, cm),
                (:Z, cm), (:Y, bm), (:X, am)
            )
                fill!(fail_flags, false)

                if op === :X
                    k! = _mass_pilot_x_kernel!(backend, 256)
                    k!(m_new, m_pilot, dir_flux, fail_flags,
                       Nx, cluster_sizes, beta_limit, r_inv;
                       ndrange=(Nx, Ny, Nz))
                elseif op === :Y
                    k! = _mass_pilot_y_kernel!(backend, 256)
                    k!(m_new, m_pilot, dir_flux, fail_flags,
                       Ny, beta_limit, r_inv;
                       ndrange=(Nx, Ny, Nz))
                else  # :Z
                    k! = _mass_pilot_z_kernel!(backend, 256)
                    k!(m_new, m_pilot, dir_flux, fail_flags,
                       Nz, beta_limit, r_inv;
                       ndrange=(Nx, Ny, Nz))
                end
                synchronize(backend)

                # Check fail flags
                if any(fail_flags)
                    ok = false
                    break
                end
                copyto!(m_pilot, m_new)
            end
            ok || break
        end

        if ok
            return r
        end

        r *= 2
    end

    @info "Mass-CFL pilot reached conservative fallback r=$max_r" maxlog=10
    return max_r
end
