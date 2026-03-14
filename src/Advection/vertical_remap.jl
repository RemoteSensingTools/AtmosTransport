# ---------------------------------------------------------------------------
# GCHP-Style Vertical Remapping for Cubed-Sphere Grids
#
# Replaces explicit Z-advection (cm-based Strang splitting) with a single
# conservative PPM vertical remap to the target pressure grid. This follows
# FV3's map1_q2 algorithm (fv_mapz.F90) and eliminates the dry/wet
# inconsistency between horizontal mass fluxes and vertical pressure closure.
#
# After horizontal-only Lin-Rood transport evolves air mass self-consistently
# from am/bm divergence, this remap redistributes tracers onto the target
# pressure levels (next met window or hybrid coordinates).
#
# Reference: fv_mapz.F90:map1_q2 (lines 1606-1692), ppm_profile (lines 2681-2935)
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

"""
    VerticalRemapWorkspace{FT, A3, A1}

GPU workspace for conservative PPM vertical remapping.

- `pe_tgt`: Target pressure edges from hybrid coords (6 panels, Nc x Nc x Nz+1)
- `dp_tgt`: Target layer thickness (6 panels, Nc x Nc x Nz)
- `m_save`: Saved air mass before horizontal transport (shared across tracers)
- `ak_dev`: Hybrid A coefficients on GPU (Nz+1)
- `bk_dev`: Hybrid B coefficients on GPU (Nz+1)
"""
struct VerticalRemapWorkspace{FT, A3 <: AbstractArray{FT,3},
                               A2 <: AbstractArray{FT,2},
                               A1 <: AbstractVector{FT}}
    pe_tgt  :: NTuple{6, A3}   # Target pressure edges (Nc, Nc, Nz+1)
    dp_tgt  :: NTuple{6, A3}   # Target layer thickness (Nc, Nc, Nz)
    pe_src  :: NTuple{6, A3}   # Source pressure edges from hybrid coords (Nc, Nc, Nz+1)
    m_save  :: NTuple{6, A3}   # Saved m for multi-tracer horizontal transport
    ps_tgt  :: NTuple{6, A2}   # Target surface pressure (Nc, Nc)
    ps_src  :: NTuple{6, A2}   # Source dry surface pressure (Nc, Nc)
    q_al    :: NTuple{6, A3}   # PPM left edge AL (Nc, Nc, Nz)
    q_ar    :: NTuple{6, A3}   # PPM right edge AR (Nc, Nc, Nz)
    q_a6    :: NTuple{6, A3}   # PPM curvature A6 (Nc, Nc, Nz)
    ak_dev  :: A1              # ak on GPU (Nz+1)
    bk_dev  :: A1              # bk on GPU (Nz+1)
end

function VerticalRemapWorkspace(grid::CubedSphereGrid{FT}, arch) where FT
    AT = array_type(arch)
    (; Nc, Hp, Nz) = grid
    N = Nc + 2Hp

    pe_tgt = ntuple(_ -> AT(zeros(FT, Nc, Nc, Nz + 1)), 6)
    dp_tgt = ntuple(_ -> AT(zeros(FT, Nc, Nc, Nz)),     6)
    pe_src = ntuple(_ -> AT(zeros(FT, Nc, Nc, Nz + 1)), 6)
    m_save = ntuple(_ -> AT(zeros(FT, N, N, Nz)),       6)
    ps_tgt = ntuple(_ -> AT(zeros(FT, Nc, Nc)),         6)
    ps_src = ntuple(_ -> AT(zeros(FT, Nc, Nc)),         6)
    q_al   = ntuple(_ -> AT(zeros(FT, Nc, Nc, Nz)),     6)
    q_ar   = ntuple(_ -> AT(zeros(FT, Nc, Nc, Nz)),     6)
    q_a6   = ntuple(_ -> AT(zeros(FT, Nc, Nc, Nz)),     6)

    ak_dev = AT(FT.(grid.vertical.A))
    bk_dev = AT(FT.(grid.vertical.B))

    return VerticalRemapWorkspace(pe_tgt, dp_tgt, pe_src, m_save, ps_tgt, ps_src,
                                  q_al, q_ar, q_a6, ak_dev, bk_dev)
end

# ---------------------------------------------------------------------------
# Target pressure computation kernel
# ---------------------------------------------------------------------------

"""
    _compute_target_pe_kernel!(pe_tgt, dp_tgt, ak, bk, ps_tgt, Nz)

Compute target pressure edges from hybrid coords: pe[i,j,k] = ak[k] + bk[k] * ps[i,j].
Also computes layer thickness dp[i,j,k] = pe[i,j,k+1] - pe[i,j,k].
"""
@kernel function _compute_target_pe_kernel!(pe_tgt, dp_tgt, @Const(ak), @Const(bk),
                                             @Const(ps_tgt), Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ps = ps_tgt[i, j]
        for k in 1:Nz+1
            pe_tgt[i, j, k] = ak[k] + bk[k] * ps
        end
        for k in 1:Nz
            dp_tgt[i, j, k] = pe_tgt[i, j, k+1] - pe_tgt[i, j, k]
        end
    end
end

# ---------------------------------------------------------------------------
# Inline helper: recompute standard monotonic slope dc[k] from source data
# Matches FV3 ppm_profile lines 2725-2735. Returns zero for k ≤ 1 or k ≥ Nz.
# ---------------------------------------------------------------------------
@inline function _vremap_dc(rm_src, m_src, ii, jj, k, Nz, g, a, zero_ft)
    FT = typeof(zero_ft)
    if k <= 1 || k >= Nz
        return zero_ft
    end
    @inbounds begin
        mkm1 = m_src[ii, jj, k-1]; dpkm1 = mkm1 * g / a
        mk   = m_src[ii, jj, k];   dpk   = mk * g / a
        mkp1 = m_src[ii, jj, k+1]; dpkp1 = mkp1 * g / a
        qkm1 = dpkm1 > FT(100) * eps(FT) ? rm_src[ii, jj, k-1] / mkm1 : zero_ft
        qk   = dpk   > FT(100) * eps(FT) ? rm_src[ii, jj, k]   / mk   : zero_ft
        qkp1 = dpkp1 > FT(100) * eps(FT) ? rm_src[ii, jj, k+1] / mkp1 : zero_ft

        d4  = dpkm1 + dpk
        d4p = dpk + dpkp1
        c1 = (dpkm1 + FT(0.5) * dpk) / d4p
        c2 = (dpkp1 + FT(0.5) * dpk) / d4
        df2 = dpk * (c1 * (qkp1 - qk) + c2 * (qk - qkm1)) / (d4 + dpkp1)

        q_max = max(qkm1, qk, qkp1)
        q_min = min(qkm1, qk, qkp1)
        sgn = df2 >= zero_ft ? one(FT) : -one(FT)
        return sgn * min(abs(df2), q_max - qk, qk - q_min)
    end
end

# ---------------------------------------------------------------------------
# PPM reconstruction kernel (FV3 ppm_profile, kord=7, iv=0)
# ---------------------------------------------------------------------------

"""
    _ppm_reconstruct_kernel!(q_al, q_ar, q_a6, rm_src, m_src, area, g_val, Hp, Nc, Nz)

Compute PPM parabolic coefficients (AL, AR, A6) for each column.
One thread per (i,j) column, sequential k-loop.
Follows fv_mapz.F90 `ppm_profile` with iv=0 (positive definite), kord=7.

PPM parabola: f(s) = AL + s*[(AR-AL) + A6*(1-s)], s ∈ [0,1]
where A6 = 3*(2*AA - (AL+AR)).

Phase ordering matches FV3 exactly:
  A: Monotonic slopes dc[k] → q_a6 scratch
  B: Interior edges AL[k] for k=3..Nz-1
  C: Top boundary edges (area-preserving cubic)
  D: Bottom boundary edges (area-preserving cubic)
  E: AR[k] = AL[k+1]
  F: A6 + lmt=0 for top 2 layers
  G: Huynh 2nd constraint + lmt=2 for interior (dc/h2 recomputed on-the-fly)
  H: A6 + lmt=0 for bottom 2 layers
"""
@kernel function _ppm_reconstruct_kernel!(q_al, q_ar, q_a6,
        @Const(rm_src), @Const(m_src), @Const(area), @Const(g_val), Hp, Nc, Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(q_al)
        a = area[i, j]
        g = g_val
        zft = zero(FT)

        # ================================================================
        # Phase A: Monotonic slopes dc[k] for k=2..Nz-1  →  q_a6
        # FV3 ppm_profile lines 2718-2735
        # ================================================================
        q_a6[i, j, 1]  = zft
        q_a6[i, j, Nz] = zft

        for k in 2:Nz-1
            mkm1 = m_src[ii, jj, k-1]; dpkm1 = mkm1 * g / a
            mk   = m_src[ii, jj, k];   dpk   = mk * g / a
            mkp1 = m_src[ii, jj, k+1]; dpkp1 = mkp1 * g / a
            qkm1 = dpkm1 > FT(100) * eps(FT) ? rm_src[ii, jj, k-1] / mkm1 : zft
            qk   = dpk   > FT(100) * eps(FT) ? rm_src[ii, jj, k]   / mk   : zft
            qkp1 = dpkp1 > FT(100) * eps(FT) ? rm_src[ii, jj, k+1] / mkp1 : zft

            d4  = dpkm1 + dpk
            d4p = dpk + dpkp1
            c1 = (dpkm1 + FT(0.5) * dpk) / d4p
            c2 = (dpkp1 + FT(0.5) * dpk) / d4
            df2 = dpk * (c1 * (qkp1 - qk) + c2 * (qk - qkm1)) / (d4 + dpkp1)

            q_max = max(qkm1, qk, qkp1)
            q_min = min(qkm1, qk, qkp1)
            sgn = df2 >= zft ? one(FT) : -one(FT)
            q_a6[i, j, k] = sgn * min(abs(df2), q_max - qk, qk - q_min)
        end

        # ================================================================
        # Phase B: Interior edges AL[k] for k=3..Nz-1  →  q_al
        # FV3 ppm_profile lines 2741-2750
        # Re-reads q, dp from rm_src/m_src; reads dc from q_a6.
        # ================================================================
        for k in 3:Nz-1
            mkm1 = m_src[ii, jj, k-1]; dpkm1 = mkm1 * g / a
            mk   = m_src[ii, jj, k];   dpk   = mk * g / a
            qkm1 = dpkm1 > FT(100) * eps(FT) ? rm_src[ii, jj, k-1] / mkm1 : zft
            qk   = dpk   > FT(100) * eps(FT) ? rm_src[ii, jj, k]   / mk   : zft
            dpkm2 = m_src[ii, jj, k-2] * g / a
            dpkp1 = m_src[ii, jj, k+1] * g / a

            dc_km1 = q_a6[i, j, k-1]
            dc_k   = q_a6[i, j, k]

            delq_km1 = qk - qkm1
            d4  = dpkm1 + dpk
            d4m = dpkm2 + dpkm1
            d4p = dpk + dpkp1

            c1 = delq_km1 * dpkm1 / d4
            a1 = d4m / (d4 + dpkm1)
            a2 = d4p / (d4 + dpk)
            q_al[i, j, k] = qkm1 + c1 + FT(2) / (d4m + d4p) *
                (dpk * (c1 * (a1 - a2) + a2 * dc_km1) - dpkm1 * a1 * dc_k)
        end

        # ================================================================
        # Phase C: Top boundary — area-preserving cubic, zero 2nd deriv at TOA
        # FV3 ppm_profile lines 2756-2781
        # ================================================================
        m1 = m_src[ii, jj, 1]; dp1 = m1 * g / a
        m2 = m_src[ii, jj, 2]; dp2 = m2 * g / a
        q1 = dp1 > FT(100) * eps(FT) ? rm_src[ii, jj, 1] / m1 : zft
        q2 = dp2 > FT(100) * eps(FT) ? rm_src[ii, jj, 2] / m2 : zft
        al3 = q_al[i, j, 3]   # from Phase B

        d1 = dp1; d2 = dp2
        qm = (d2 * q1 + d1 * q2) / (d1 + d2)
        dq = FT(2) * (q2 - q1) / (d1 + d2)
        c1_t = FT(4) * (al3 - qm - d2 * dq) /
            (d2 * (FT(2) * d2 * d2 + d1 * (d2 + FT(3) * d1)))
        c3 = dq - FT(0.5) * c1_t * (d2 * (FT(5) * d1 + d2) - FT(3) * d1 * d1)
        al_2 = qm - FT(0.25) * c1_t * d1 * d2 * (d2 + FT(3) * d1)
        al_1 = d1 * (FT(2) * c1_t * d1 * d1 - c3) + al_2

        # No over/undershoot (FV3 lines 2771-2772)
        al_2 = clamp(al_2, min(q1, q2), max(q1, q2))
        # iv=0 positivity (FV3 lines 2779-2781)
        al_1 = max(zft, al_1)
        al_2 = max(zft, al_2)
        # Boundary dc[1] (FV3 line 2773) — used by lmt=0 and Huynh h2
        dc_1_val = FT(0.5) * (al_2 - q1)

        q_al[i, j, 1] = al_1
        q_al[i, j, 2] = al_2
        q_a6[i, j, 1] = dc_1_val   # store boundary dc[1] in scratch

        # ================================================================
        # Phase D: Bottom boundary — area-preserving cubic, zero 2nd deriv at surface
        # FV3 ppm_profile lines 2796-2839
        # ================================================================
        mN   = m_src[ii, jj, Nz];   dpN   = mN * g / a
        mNm1 = m_src[ii, jj, Nz-1]; dpNm1 = mNm1 * g / a
        qN   = dpN   > FT(100) * eps(FT) ? rm_src[ii, jj, Nz]   / mN   : zft
        qNm1 = dpNm1 > FT(100) * eps(FT) ? rm_src[ii, jj, Nz-1] / mNm1 : zft
        al_Nm1 = q_al[i, j, Nz-1]   # from Phase B

        d1 = dpN; d2 = dpNm1
        qm = (d2 * qN + d1 * qNm1) / (d1 + d2)
        dq = FT(2) * (qNm1 - qN) / (d1 + d2)
        c1_b = (al_Nm1 - qm - d2 * dq) /
            (d2 * (FT(2) * d2 * d2 + d1 * (d2 + FT(3) * d1)))
        c3 = dq - FT(2) * c1_b * (d2 * (FT(5) * d1 + d2) - FT(3) * d1 * d1)
        al_N = qm - c1_b * d1 * d2 * (d2 + FT(3) * d1)
        ar_N = d1 * (FT(8) * c1_b * d1 * d1 - c3) + al_N

        # No over/undershoot (FV3 lines 2811-2812)
        al_N = clamp(al_N, min(qN, qNm1), max(qN, qNm1))
        # iv=0 positivity (FV3 lines 2836-2839)
        al_N = max(zft, al_N)
        ar_N = max(zft, ar_N)
        # Boundary dc[Nz] (FV3 line 2813) — used by lmt=0 and Huynh h2
        dc_N_val = FT(0.5) * (qN - al_N)

        q_al[i, j, Nz] = al_N
        q_ar[i, j, Nz] = ar_N
        q_a6[i, j, Nz] = dc_N_val   # store boundary dc[Nz] in scratch

        # ================================================================
        # Phase E: AR[k] = AL[k+1] for k=1..Nz-1
        # FV3 ppm_profile lines 2847-2851. AR[Nz] already set in Phase D.
        # ================================================================
        for k in 1:Nz-1
            q_ar[i, j, k] = q_al[i, j, k+1]
        end

        # ================================================================
        # Save boundary dc values before A6 overwrites q_a6
        # ================================================================
        dc_1_saved = q_a6[i, j, 1]    # boundary dc[1] from Phase C
        dc_N_saved = q_a6[i, j, Nz]   # boundary dc[Nz] from Phase D

        # ================================================================
        # Phase F: A6 + lmt=0 for top 2 layers (k=1,2)
        # FV3 ppm_profile lines 2857-2862 + ppm_limiters lmt=0
        # ================================================================
        for k_bnd in (1, 2)
            mk  = m_src[ii, jj, k_bnd]; dpk = mk * g / a
            qk  = dpk > FT(100) * eps(FT) ? rm_src[ii, jj, k_bnd] / mk : zft
            al  = q_al[i, j, k_bnd]
            ar  = q_ar[i, j, k_bnd]
            a6  = FT(3) * (FT(2) * qk - (al + ar))
            dc_k = q_a6[i, j, k_bnd]   # dc still valid (overwritten AFTER this read)

            # ppm_limiters lmt=0 (FV3 lines 2958-2977)
            if dc_k == zft
                al = qk; ar = qk; a6 = zft
            else
                da1  = ar - al
                da2  = da1 * da1
                a6da = a6 * da1
                if a6da < -da2
                    a6 = FT(3) * (al - qk)
                    ar = al - a6
                elseif a6da > da2
                    a6 = FT(3) * (ar - qk)
                    al = ar - a6
                end
            end

            q_al[i, j, k_bnd] = al
            q_ar[i, j, k_bnd] = ar
            q_a6[i, j, k_bnd] = a6   # overwrites dc → A6
        end

        # ================================================================
        # Phase G: Huynh 2nd constraint + lmt=2 for interior (k=3..Nz-2)
        # FV3 ppm_profile lines 2864-2910
        # dc and h2 recomputed on-the-fly to avoid scratch-space conflicts.
        # ================================================================
        for k in 3:Nz-2
            mk  = m_src[ii, jj, k]; dpk = mk * g / a
            qk  = dpk > FT(100) * eps(FT) ? rm_src[ii, jj, k] / mk : zft
            al  = q_al[i, j, k]
            ar  = q_ar[i, j, k]

            # Recompute dc[k] (standard formula, FV3 lines 2725-2735)
            dc_k = _vremap_dc(rm_src, m_src, ii, jj, k, Nz, g, a, zft)

            # --- h2[k-1] on-the-fly (FV3 lines 2868-2875) ---
            # h2[m] = 2*(dc[m+1]/dp[m+1] - dc[m-1]/dp[m-1]) /
            #          (dp[m]+0.5*(dp[m-1]+dp[m+1])) * dp[m]^2
            dc_km2 = (k - 2 == 1) ? dc_1_saved :
                _vremap_dc(rm_src, m_src, ii, jj, k - 2, Nz, g, a, zft)
            dp_km2 = m_src[ii, jj, k-2] * g / a
            dp_km1 = m_src[ii, jj, k-1] * g / a
            h2_km1 = FT(2) * (dc_k / dpk - dc_km2 / dp_km2) /
                (dp_km1 + FT(0.5) * (dp_km2 + dpk)) * dp_km1 * dp_km1

            # --- h2[k+1] on-the-fly ---
            dc_kp2 = (k + 2 == Nz) ? dc_N_saved :
                _vremap_dc(rm_src, m_src, ii, jj, k + 2, Nz, g, a, zft)
            dp_kp1 = m_src[ii, jj, k+1] * g / a
            dp_kp2 = m_src[ii, jj, k+2] * g / a
            h2_kp1 = FT(2) * (dc_kp2 / dp_kp2 - dc_k / dpk) /
                (dp_kp1 + FT(0.5) * (dpk + dp_kp2)) * dp_kp1 * dp_kp1

            # --- Huynh constraint (FV3 lines 2889-2901) ---
            fac = FT(1.5)
            pmp = FT(2) * dc_k

            # Right edge (AR)
            qmp = qk + pmp
            lac = qk + fac * h2_km1 + dc_k
            ar = clamp(ar, min(qk, qmp, lac), max(qk, qmp, lac))

            # Left edge (AL)
            qmp = qk - pmp
            lac = qk + fac * h2_kp1 - dc_k
            al = clamp(al, min(qk, qmp, lac), max(qk, qmp, lac))

            # Recompute A6 (FV3 line 2905)
            a6 = FT(3) * (FT(2) * qk - (al + ar))

            # --- Positive-definite lmt=2 (FV3 lines 2990-3010) ---
            da1 = ar - al
            if abs(da1) < -a6   # interior extremum exists (A6 < 0, |A6| > |da|)
                fmin = qk + FT(0.25) * da1 * da1 / a6 + a6 / FT(12)
                if fmin < zft
                    if qk < ar && qk < al
                        # Cell mean below both edges → flatten
                        al = qk; ar = qk; a6 = zft
                    elseif ar > al
                        # Keep AL, adjust AR
                        a6 = FT(3) * (al - qk)
                        ar = al - a6
                    else
                        # Keep AR, adjust AL
                        a6 = FT(3) * (ar - qk)
                        al = ar - a6
                    end
                end
            end

            q_al[i, j, k] = al
            q_ar[i, j, k] = ar
            q_a6[i, j, k] = a6
        end

        # ================================================================
        # Phase H: A6 + lmt=0 for bottom 2 layers (k=Nz-1, Nz)
        # FV3 ppm_profile lines 2928-2933 + ppm_limiters lmt=0
        # ================================================================
        for k_bnd in (Nz-1, Nz)
            mk  = m_src[ii, jj, k_bnd]; dpk = mk * g / a
            qk  = dpk > FT(100) * eps(FT) ? rm_src[ii, jj, k_bnd] / mk : zft
            al  = q_al[i, j, k_bnd]
            ar  = q_ar[i, j, k_bnd]
            a6  = FT(3) * (FT(2) * qk - (al + ar))

            # Recompute dc for lmt=0 zero-check
            dc_k = (k_bnd == Nz) ? dc_N_saved :
                _vremap_dc(rm_src, m_src, ii, jj, k_bnd, Nz, g, a, zft)

            # ppm_limiters lmt=0 (FV3 lines 2958-2977)
            if dc_k == zft
                al = qk; ar = qk; a6 = zft
            else
                da1  = ar - al
                da2  = da1 * da1
                a6da = a6 * da1
                if a6da < -da2
                    a6 = FT(3) * (al - qk)
                    ar = al - a6
                elseif a6da > da2
                    a6 = FT(3) * (ar - qk)
                    al = ar - a6
                end
            end

            q_al[i, j, k_bnd] = al
            q_ar[i, j, k_bnd] = ar
            q_a6[i, j, k_bnd] = a6
        end
    end
end

# ---------------------------------------------------------------------------
# Core vertical remap kernel (column-sequential, one thread per i,j)
# ---------------------------------------------------------------------------

"""
    _vertical_remap_column_kernel!(rm, rm_src, m_src, q_al, q_ar, q_a6,
                                    pe_tgt, dp_tgt, area, g, Hp, Nc, Nz)

Conservative PPM vertical remap for one panel of the cubed sphere.
For each column (i,j):
1. Compute source pressure edges from current air mass
2. Read pre-computed PPM coefficients (AL, AR, A6)
3. Integrate PPM polynomial over target layer intervals
4. Write remapped rm = q_remap * dp_tgt * area / g

Uses the general PPM integral: q_avg = AL + 0.5*(pl+pr)*(AR-AL+A6) - (A6/3)*(pl²+pl*pr+pr²)
where pl, pr are normalized coordinates within the source layer.
"""
@kernel function _vertical_remap_column_kernel!(rm,
        @Const(rm_src), @Const(m_src),
        @Const(q_al), @Const(q_ar), @Const(q_a6),
        @Const(pe_tgt), @Const(dp_tgt),
        @Const(area), @Const(g_val), Hp, Nc, Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(rm)
        a = area[i, j]
        g = g_val

        # Bottom source layer mixing ratio for extrapolation when target
        # extends below source (PS_tgt > PS_src between windows).
        m_bot = m_src[ii, jj, Nz]
        q_bot = m_bot > FT(100) * eps(FT) ?
            rm_src[ii, jj, Nz] / m_bot : zero(FT)

        # Source state: track current source layer index and running PE edges.
        # Use Float64 for PE accumulation to avoid precision loss on thin layers.
        # With Float32, eps(~100000 Pa) ≈ 0.004 Pa, but TOA layers can be < 0.01 Pa.
        ks = 1
        pe_s_lo = Float64(pe_tgt[i, j, 1])  # TOA = ptop
        dp_s_k64 = Float64(m_src[ii, jj, 1]) * Float64(g) / Float64(a)
        pe_s_hi = pe_s_lo + dp_s_k64

        for kt in 1:Nz
            pe_t_lo = Float64(pe_tgt[i, j, kt])
            pe_t_hi = Float64(pe_tgt[i, j, kt + 1])
            dp_t = dp_tgt[i, j, kt]

            if dp_t <= zero(FT)
                rm[ii, jj, kt] = zero(FT)
                continue
            end

            rm_accum = zero(FT)

            while pe_s_lo < pe_t_hi && ks <= Nz
                dp_s_k64 = Float64(m_src[ii, jj, ks]) * Float64(g) / Float64(a)
                dp_s_k = FT(dp_s_k64)
                pe_s_hi = pe_s_lo + dp_s_k64
                q_k = dp_s_k > FT(100) * eps(FT) ?
                    rm_src[ii, jj, ks] / m_src[ii, jj, ks] : zero(FT)

                p_lo = max(pe_t_lo, pe_s_lo)
                p_hi = min(pe_t_hi, pe_s_hi)

                if p_hi > p_lo && dp_s_k > FT(100) * eps(FT)
                    dp_overlap = FT(p_hi - p_lo)

                    # Check if this source layer is entirely contained in target
                    if pe_s_lo >= pe_t_lo && pe_s_hi <= pe_t_hi
                        # Whole source layer → use cell mean directly (avoids rounding)
                        rm_accum += q_k * dp_s_k
                    else
                        # Partial overlap → PPM integral
                        al = q_al[i, j, ks]
                        ar = q_ar[i, j, ks]
                        a6 = q_a6[i, j, ks]

                        # Normalized coordinates within source layer [0,1]
                        pl = FT((p_lo - pe_s_lo) / dp_s_k64)
                        pr = FT((p_hi - pe_s_lo) / dp_s_k64)

                        # PPM integral average: map1_q2 general formula
                        q_avg = al + FT(0.5) * (pl + pr) * (ar - al + a6) -
                                a6 / FT(3) * (pr * pr + pr * pl + pl * pl)
                        rm_accum += q_avg * dp_overlap
                    end
                end

                if pe_s_hi <= pe_t_hi
                    pe_s_lo = pe_s_hi
                    ks += 1
                else
                    break
                end
            end

            # Extrapolate when target extends below source column
            # (PS_tgt > PS_src between windows). Fill with bottom q.
            if ks > Nz && pe_s_lo < pe_t_hi
                dp_extra = FT(pe_t_hi - max(pe_t_lo, pe_s_lo))
                if dp_extra > zero(FT)
                    rm_accum += q_bot * dp_extra
                end
            end

            rm[ii, jj, kt] = rm_accum * a / g
        end
    end
end

# ---------------------------------------------------------------------------
# Hybrid-PE remap kernel: uses pre-computed source PE from hybrid coords
# instead of computing from m_src. This ensures source and target PE are
# both on the smooth hybrid grid (PE = ak + bk*PS_dry), eliminating noisy
# per-level displacement at the pure-pressure/hybrid boundary.
# q = rm/m still uses actual mass (correct mixing ratio).
# ---------------------------------------------------------------------------

"""
Conservative PPM vertical remap using pre-computed source pressure edges.
Same algorithm as `_vertical_remap_column_kernel!`, but reads source PE
from `pe_src` (hybrid coords) instead of computing from `m_src × g / area`.
This ensures source and target PE are both on the smooth hybrid grid,
eliminating noisy per-level displacement at the pure-pressure/hybrid boundary.
q = rm/m still uses actual mass (correct mixing ratio).
"""
@kernel function _vertical_remap_hybrid_pe_kernel!(rm,
        @Const(rm_src), @Const(m_src), @Const(pe_src),
        @Const(q_al), @Const(q_ar), @Const(q_a6),
        @Const(pe_tgt), @Const(dp_tgt),
        @Const(area), @Const(g_val), Hp, Nc, Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(rm)
        a = area[i, j]
        g = g_val

        # Bottom source layer mixing ratio for extrapolation when target
        # extends below source (PS_tgt > PS_src between windows).
        # GCHP avoids this by forcing pe2(npz+1)=pe1(npz+1); we can't,
        # so we extrapolate with constant q from the bottom source layer.
        m_bot = m_src[ii, jj, Nz]
        q_bot = m_bot > FT(100) * eps(FT) ?
            rm_src[ii, jj, Nz] / m_bot : zero(FT)

        ks = 1
        pe_s_lo = Float64(pe_src[i, j, 1])

        for kt in 1:Nz
            pe_t_lo = Float64(pe_tgt[i, j, kt])
            pe_t_hi = Float64(pe_tgt[i, j, kt + 1])
            dp_t = dp_tgt[i, j, kt]

            if dp_t <= zero(FT)
                rm[ii, jj, kt] = zero(FT)
                continue
            end

            rm_accum = zero(FT)

            while pe_s_lo < pe_t_hi && ks <= Nz
                pe_s_hi = Float64(pe_src[i, j, ks + 1])
                dp_s_k64 = pe_s_hi - Float64(pe_src[i, j, ks])
                dp_s_k = FT(dp_s_k64)
                mk = m_src[ii, jj, ks]
                q_k = mk > FT(100) * eps(FT) ?
                    rm_src[ii, jj, ks] / mk : zero(FT)

                p_lo = max(pe_t_lo, pe_s_lo)
                p_hi = min(pe_t_hi, pe_s_hi)

                if p_hi > p_lo && dp_s_k > FT(100) * eps(FT)
                    dp_overlap = FT(p_hi - p_lo)

                    if pe_s_lo >= pe_t_lo && pe_s_hi <= pe_t_hi
                        rm_accum += q_k * dp_s_k
                    else
                        al = q_al[i, j, ks]
                        ar = q_ar[i, j, ks]
                        a6 = q_a6[i, j, ks]

                        pl = FT((p_lo - pe_s_lo) / dp_s_k64)
                        pr = FT((p_hi - pe_s_lo) / dp_s_k64)

                        q_avg = al + FT(0.5) * (pl + pr) * (ar - al + a6) -
                                a6 / FT(3) * (pr * pr + pr * pl + pl * pl)
                        rm_accum += q_avg * dp_overlap
                    end
                end

                if pe_s_hi <= pe_t_hi
                    pe_s_lo = pe_s_hi
                    ks += 1
                else
                    break
                end
            end

            # Extrapolate when target extends below source column
            # (PS_tgt > PS_src). Fill with bottom source layer's q.
            if ks > Nz && pe_s_lo < pe_t_hi
                dp_extra = FT(pe_t_hi - max(pe_t_lo, pe_s_lo))
                if dp_extra > zero(FT)
                    rm_accum += q_bot * dp_extra
                end
            end

            rm[ii, jj, kt] = rm_accum * a / g
        end
    end
end

# ---------------------------------------------------------------------------
# Panel-level dispatch
# ---------------------------------------------------------------------------

"""
    vertical_remap_cs!(rm_panels, m_src_panels, ws_vr, ws, gc, grid;
                       flat=false, hybrid_pe=false)

Apply conservative vertical remapping for all 6 CS panels.
Remaps tracer `rm` from source pressure to target pressure.

When `hybrid_pe=true`, uses pre-computed `ws_vr.pe_src` (from hybrid coords)
for source pressure edges instead of deriving them from `m_src`. This ensures
source and target PE are both on the smooth hybrid grid, eliminating noisy
displacement at the pure-pressure/hybrid transition.

`m_src_panels` is the SAVED post-horizontal air mass — shared across all tracers
and NOT modified by this function. The remap only modifies `rm_panels`.
"""
function vertical_remap_cs!(rm_panels, m_src_panels, ws_vr::VerticalRemapWorkspace,
                              ws, gc, grid; flat::Bool=false, hybrid_pe::Bool=false)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    backend = get_backend(rm_panels[1])
    recon_k! = flat ? _flat_reconstruct_kernel!(backend, 256) :
                      _ppm_reconstruct_kernel!(backend, 256)

    for p in 1:6
        # Copy rm to buffer (source tracer mass), m_src is read-only
        copyto!(ws.rm_buf, rm_panels[p])

        # Phase 1: Reconstruction → compute AL, AR, A6 per column
        if flat
            recon_k!(ws_vr.q_al[p], ws_vr.q_ar[p], ws_vr.q_a6[p],
                     ws.rm_buf, m_src_panels[p], Hp, Nc, Nz; ndrange=(Nc, Nc))
        else
            recon_k!(ws_vr.q_al[p], ws_vr.q_ar[p], ws_vr.q_a6[p],
                     ws.rm_buf, m_src_panels[p],
                     gc.area[p], gc.gravity, Hp, Nc, Nz; ndrange=(Nc, Nc))
        end
        synchronize(backend)

        # Phase 2: PPM integration → remap rm onto target pressure levels
        if hybrid_pe
            remap_k! = _vertical_remap_hybrid_pe_kernel!(backend, 256)
            remap_k!(rm_panels[p], ws.rm_buf, m_src_panels[p],
                     ws_vr.pe_src[p],
                     ws_vr.q_al[p], ws_vr.q_ar[p], ws_vr.q_a6[p],
                     ws_vr.pe_tgt[p], ws_vr.dp_tgt[p],
                     gc.area[p], gc.gravity, Hp, Nc, Nz; ndrange=(Nc, Nc))
        else
            remap_k! = _vertical_remap_column_kernel!(backend, 256)
            remap_k!(rm_panels[p], ws.rm_buf, m_src_panels[p],
                     ws_vr.q_al[p], ws_vr.q_ar[p], ws_vr.q_a6[p],
                     ws_vr.pe_tgt[p], ws_vr.dp_tgt[p],
                     gc.area[p], gc.gravity, Hp, Nc, Nz; ndrange=(Nc, Nc))
        end
        synchronize(backend)
    end
    return nothing
end

"""Zeroth-order reconstruction: AL=AR=q, A6=0. Diagnostic mode — preserves
cell means for fully-contained layers, uses first-order averaging for partial overlaps."""
@kernel function _flat_reconstruct_kernel!(q_al, q_ar, q_a6,
        @Const(rm_src), @Const(m_src), Hp, Nc, Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(q_al)
        for k in 1:Nz
            mk = m_src[ii, jj, k]
            qk = mk > FT(100) * eps(FT) ? rm_src[ii, jj, k] / mk : zero(FT)
            q_al[i, j, k] = qk
            q_ar[i, j, k] = qk
            q_a6[i, j, k] = zero(FT)
        end
    end
end

"""
    compute_target_pressure!(ws_vr, phys, sched, grid; has_next=true)

Compute target pressure edges for vertical remapping.
- If `has_next`: target PS = sum(next_delp) per column → steer toward next met window
- Otherwise: use post-advection PS with hybrid coordinates
"""
function compute_target_pressure_from_next_delp!(ws_vr::VerticalRemapWorkspace,
                                                   ng_delp, gc, grid)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    FT = eltype(ws_vr.ak_dev)
    backend = get_backend(ws_vr.ak_dev)
    ptop = FT(grid.vertical.A[1])  # TOA pressure (1.0 Pa for GEOS L72)

    # Compute ps_tgt from next-window DELP (sum over all levels)
    ps_k! = _ps_from_delp_kernel!(backend, 256)
    pe_k! = _compute_target_pe_kernel!(backend, 256)

    for p in 1:6
        ps_k!(ws_vr.ps_tgt[p], ng_delp[p], ptop, Hp, Nc, Nz; ndrange=(Nc, Nc))
    end
    synchronize(backend)

    for p in 1:6
        pe_k!(ws_vr.pe_tgt[p], ws_vr.dp_tgt[p], ws_vr.ak_dev, ws_vr.bk_dev,
              ws_vr.ps_tgt[p], Nz; ndrange=(Nc, Nc))
    end
    synchronize(backend)
    return nothing
end

"""
    compute_target_pressure_from_mass!(ws_vr, m_panels, gc, grid)

Compute target pressure from post-advection air mass (fallback for last window).
PS = sum(m * g / area) per column, then pe = ak + bk * ps.
"""
function compute_target_pressure_from_mass!(ws_vr::VerticalRemapWorkspace,
                                              m_panels, gc, grid)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    FT = eltype(ws_vr.ak_dev)
    backend = get_backend(ws_vr.ak_dev)
    ptop = FT(grid.vertical.A[1])  # TOA pressure (1.0 Pa for GEOS L72)

    ps_m_k! = _ps_from_mass_kernel!(backend, 256)
    pe_k!   = _compute_target_pe_kernel!(backend, 256)

    for p in 1:6
        ps_m_k!(ws_vr.ps_tgt[p], m_panels[p], gc.area[p], gc.gravity, ptop,
                Hp, Nc, Nz; ndrange=(Nc, Nc))
    end
    synchronize(backend)

    for p in 1:6
        pe_k!(ws_vr.pe_tgt[p], ws_vr.dp_tgt[p], ws_vr.ak_dev, ws_vr.bk_dev,
              ws_vr.ps_tgt[p], Nz; ndrange=(Nc, Nc))
    end
    synchronize(backend)
    return nothing
end

# ---------------------------------------------------------------------------
# Helper kernels
# ---------------------------------------------------------------------------

"""Compute surface pressure as ptop + sum of DELP over all levels."""
@kernel function _ps_from_delp_kernel!(ps_tgt, @Const(delp), ptop, Hp, Nc, Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(ps_tgt)
        s = zero(FT)
        for k in 1:Nz
            s += delp[ii, jj, k]
        end
        ps_tgt[i, j] = s + ptop
    end
end

"""Compute surface pressure from air mass: PS = ptop + sum(m * g / area)."""
@kernel function _ps_from_mass_kernel!(ps_tgt, @Const(m), @Const(area), g_val, ptop, Hp, Nc, Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(ps_tgt)
        a = area[i, j]
        g = g_val
        s = zero(FT)
        for k in 1:Nz
            s += m[ii, jj, k] * g / a
        end
        ps_tgt[i, j] = s + ptop
    end
end

"""
    update_air_mass_from_target!(m_panels, ws_vr, gc, grid)

Set air mass to target state: m[k] = dp_tgt[k] * area / g.
"""
function update_air_mass_from_target!(m_panels, ws_vr::VerticalRemapWorkspace, gc, grid)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    backend = get_backend(m_panels[1])
    k! = _set_mass_from_dp_kernel!(backend, 256)

    for p in 1:6
        k!(m_panels[p], ws_vr.dp_tgt[p], gc.area[p], gc.gravity,
           Hp, Nc, Nz; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)
    return nothing
end

@kernel function _set_mass_from_dp_kernel!(m, @Const(dp_tgt), @Const(area), g_val, Hp, Nc, Nz)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        m[ii, jj, k] = dp_tgt[i, j, k] * area[i, j] / g_val
    end
end

# ---------------------------------------------------------------------------
# Direct target PE from DELP (cumulative sum, bypasses hybrid formula)
# ---------------------------------------------------------------------------

"""
    compute_target_pressure_from_delp_direct!(ws_vr, ng_delp, gc, grid)

Build target PE as cumulative sum of next-window DELP, NOT via hybrid ak+bk*ps.
This avoids artificial redistribution when dry-air correction makes the actual
layer thicknesses differ from the hybrid formula.

pe_tgt[1] = ptop; pe_tgt[k+1] = pe_tgt[k] + delp[k]; dp_tgt[k] = delp[k].
"""
function compute_target_pressure_from_delp_direct!(ws_vr::VerticalRemapWorkspace,
                                                     ng_delp, gc, grid)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    FT = eltype(ws_vr.ak_dev)
    backend = get_backend(ws_vr.ak_dev)
    ptop = FT(grid.vertical.A[1])

    k! = _pe_from_delp_direct_kernel!(backend, 256)
    for p in 1:6
        k!(ws_vr.pe_tgt[p], ws_vr.dp_tgt[p], ng_delp[p],
           ptop, Hp, Nc, Nz; ndrange=(Nc, Nc))
    end
    synchronize(backend)
    return nothing
end

@kernel function _pe_from_delp_direct_kernel!(pe_tgt, dp_tgt, @Const(delp),
                                                ptop, Hp, Nc, Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(pe_tgt)
        # Use Float64 accumulator to avoid precision loss in cumsum.
        pe_acc = Float64(ptop)
        pe_tgt[i, j, 1] = FT(pe_acc)
        for k in 1:Nz
            dp = Float64(delp[ii, jj, k])
            pe_acc += dp
            pe_tgt[i, j, k + 1] = FT(pe_acc)
            dp_tgt[i, j, k] = FT(dp)
        end
    end
end

"""
    compute_target_pressure_from_mass_direct!(ws_vr, m_panels, gc, grid)

Build target PE from current air mass as cumulative sum (identity remap).
Used for last window when no next DELP is available.
pe_tgt[k+1] = pe_tgt[k] + m[k]*g/area → source = target → remap is no-op.
"""
function compute_target_pressure_from_mass_direct!(ws_vr::VerticalRemapWorkspace,
                                                     m_panels, gc, grid)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    FT = eltype(ws_vr.ak_dev)
    backend = get_backend(ws_vr.ak_dev)
    ptop = FT(grid.vertical.A[1])

    k! = _pe_from_mass_direct_kernel!(backend, 256)
    for p in 1:6
        k!(ws_vr.pe_tgt[p], ws_vr.dp_tgt[p], m_panels[p],
           gc.area[p], gc.gravity, ptop, Hp, Nc, Nz; ndrange=(Nc, Nc))
    end
    synchronize(backend)
    return nothing
end

"""
    compute_target_pe_from_hybrid_coords!(ws_vr, ng_delp, qv_panels, gc, grid)

Build target PE using GCHP's approach: `PE[k] = ak[k] + bk[k] * PS_dry`.

Computes `PS_dry = sum(delp * (1 - qv))` for each column, then reconstructs
target pressure edge values directly from the hybrid coordinate definition.
This is physically exact: pure-pressure levels get `PE = ak` (zero noise),
hybrid levels vary smoothly with `PS_dry` via the bk coefficient.

Unlike the cumsum approach followed by `fix_target_bottom_pe!`, this avoids
per-level QV fluctuations contaminating the target PE and eliminates all
ad-hoc scaling. Reference: GCHP `DryPLE` computation in fvdycore.
"""
function compute_target_pe_from_hybrid_coords!(ws_vr::VerticalRemapWorkspace,
                                                 ng_delp, qv_panels, gc, grid)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    FT = eltype(ws_vr.ak_dev)
    backend = get_backend(ws_vr.ak_dev)
    ptop = FT(grid.vertical.A[1])

    k! = _pe_from_hybrid_coords_kernel!(backend, 256)
    for p in 1:6
        k!(ws_vr.pe_tgt[p], ws_vr.dp_tgt[p], ng_delp[p], qv_panels[p],
           ws_vr.ak_dev, ws_vr.bk_dev, ptop, Hp, Nc, Nz; ndrange=(Nc, Nc))
    end
    synchronize(backend)
    return nothing
end

@kernel function _pe_from_hybrid_coords_kernel!(pe_tgt, dp_tgt, @Const(delp), @Const(qv),
                                                  @Const(ak), @Const(bk),
                                                  ptop, Hp, Nc, Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(pe_tgt)

        # Step 1: Compute dry surface pressure
        # PS_dry = ptop + sum(delp * (1 - qv))
        # sum(delp*(1-qv)) is column thickness; add ptop for full surface pressure
        ps_dry = ptop
        for k in 1:Nz
            ps_dry += delp[ii, jj, k] * (FT(1) - qv[ii, jj, k])
        end

        # Step 2: Rebuild PE from hybrid coordinates: PE[k] = ak[k] + bk[k] * PS_dry
        for k in 1:Nz + 1
            pe_tgt[i, j, k] = ak[k] + bk[k] * ps_dry
        end

        # Step 3: dp_tgt[k] = PE[k+1] - PE[k]
        for k in 1:Nz
            dp_tgt[i, j, k] = pe_tgt[i, j, k + 1] - pe_tgt[i, j, k]
        end
    end
end

# ---------------------------------------------------------------------------
# Pure GCHP approach: PE from PS + ak/bk (not from raw DELP accumulation)
# ---------------------------------------------------------------------------

"""
    _pe_from_ps_hybrid_kernel!(pe, dp, ps_dry_out, delp, qv, ak, bk, Hp, Nc, Nz)

Compute pressure edges using the pure GCHP algorithm:
1. PS_total = ptop + sum(DELP)  (total surface pressure from met DELP)
2. DELP_hybrid[k] = (ak[k+1]-ak[k]) + (bk[k+1]-bk[k]) × PS_total  (exact hybrid formula)
3. PS_dry = ptop + sum(DELP_hybrid[k] × (1 - QV[k]))  (dry surface pressure)
4. PE[k] = ak[k] + bk[k] × PS_dry  (smooth hybrid PE)

All intermediate computation in Float64 for precision. This ensures PE is
exactly on the hybrid grid with no per-level Float32 noise from DELP accumulation.
Pure-pressure levels (bk=0) get PE=ak exactly.

Reference: GCHP `GCHPctmEnv_GridCompMod.F90:calculate_ple` with SPHU argument.
"""
@kernel function _pe_from_ps_hybrid_kernel!(pe, dp, ps_dry_out,
        @Const(delp), @Const(qv), @Const(ak), @Const(bk), Hp, Nc, Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(pe)

        # Step 1: Total surface pressure from met DELP
        ps_total = Float64(ak[1])  # ptop
        for k in 1:Nz
            ps_total += Float64(delp[ii, jj, k])
        end

        # Step 2: Hybrid DELP from PS + ak/bk, then dry surface pressure
        # DELP_hybrid[k] = PE[k+1] - PE[k] = (ak[k+1]-ak[k]) + (bk[k+1]-bk[k]) × PS
        ps_dry = Float64(ak[1])  # ptop
        for k in 1:Nz
            delp_hybrid = (Float64(ak[k + 1]) - Float64(ak[k])) +
                          (Float64(bk[k + 1]) - Float64(bk[k])) * ps_total
            ps_dry += delp_hybrid * (1.0 - Float64(qv[ii, jj, k]))
        end
        ps_dry_out[i, j] = FT(ps_dry)

        # Step 3: PE and dp from hybrid coordinates
        for k in 1:Nz + 1
            pe[i, j, k] = FT(Float64(ak[k]) + Float64(bk[k]) * ps_dry)
        end
        for k in 1:Nz
            dp[i, j, k] = pe[i, j, k + 1] - pe[i, j, k]
        end
    end
end

"""
    compute_source_pe_from_hybrid!(ws_vr, delp_panels, qv_panels, gc, grid)

Compute SOURCE pressure edges using the pure GCHP hybrid approach.
Writes to `ws_vr.pe_src` and `ws_vr.ps_src`. Uses current-window DELP and QV.
"""
function compute_source_pe_from_hybrid!(ws_vr::VerticalRemapWorkspace,
                                         delp_panels, qv_panels, gc, grid)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    backend = get_backend(ws_vr.ak_dev)

    # pe_src has no dp counterpart in workspace — use a temporary view of dp_tgt
    # that will be overwritten by target PE computation later. Or we compute dp
    # into a scratch buffer. Since we only need pe_src (not dp_src) for the remap
    # kernel, we can write dp to dp_tgt as scratch (overwritten by target PE next).
    k! = _pe_from_ps_hybrid_kernel!(backend, 256)
    for p in 1:6
        k!(ws_vr.pe_src[p], ws_vr.dp_tgt[p], ws_vr.ps_src[p],
           delp_panels[p], qv_panels[p],
           ws_vr.ak_dev, ws_vr.bk_dev, Hp, Nc, Nz; ndrange=(Nc, Nc))
    end
    synchronize(backend)
    return nothing
end

"""
    compute_target_pe_from_ps_hybrid!(ws_vr, delp_panels, qv_panels, gc, grid)

Compute TARGET pressure edges using the pure GCHP hybrid approach.
Writes to `ws_vr.pe_tgt`, `ws_vr.dp_tgt`, and `ws_vr.ps_tgt`.
Uses next-window DELP and current-window QV (<0.1% approximation).
"""
function compute_target_pe_from_ps_hybrid!(ws_vr::VerticalRemapWorkspace,
                                             delp_panels, qv_panels, gc, grid)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    backend = get_backend(ws_vr.ak_dev)

    k! = _pe_from_ps_hybrid_kernel!(backend, 256)
    for p in 1:6
        k!(ws_vr.pe_tgt[p], ws_vr.dp_tgt[p], ws_vr.ps_tgt[p],
           delp_panels[p], qv_panels[p],
           ws_vr.ak_dev, ws_vr.bk_dev, Hp, Nc, Nz; ndrange=(Nc, Nc))
    end
    synchronize(backend)
    return nothing
end

"""
    compute_target_pressure_from_dry_delp_direct!(ws_vr, ng_delp, qv_panels, gc, grid)

Build target PE as cumulative sum of DRY next-window DELP: `delp × (1 - qv)`.
Uses current-window QV as approximation (QV changes <0.1% between hourly windows).
Ensures target PE is on the same dry basis as source PE (from dry air mass).
"""
function compute_target_pressure_from_dry_delp_direct!(ws_vr::VerticalRemapWorkspace,
                                                         ng_delp, qv_panels, gc, grid)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    FT = eltype(ws_vr.ak_dev)
    backend = get_backend(ws_vr.ak_dev)
    ptop = FT(grid.vertical.A[1])

    k! = _pe_from_dry_delp_direct_kernel!(backend, 256)
    for p in 1:6
        k!(ws_vr.pe_tgt[p], ws_vr.dp_tgt[p], ng_delp[p], qv_panels[p],
           ptop, Hp, Nc, Nz; ndrange=(Nc, Nc))
    end
    synchronize(backend)
    return nothing
end

@kernel function _pe_from_dry_delp_direct_kernel!(pe_tgt, dp_tgt, @Const(delp), @Const(qv),
                                                    ptop, Hp, Nc, Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(pe_tgt)
        # Use Float64 accumulator to avoid precision loss in cumsum.
        pe_acc = Float64(ptop)
        pe_tgt[i, j, 1] = FT(pe_acc)
        for k in 1:Nz
            dp = Float64(delp[ii, jj, k]) * (1.0 - Float64(qv[ii, jj, k]))
            pe_acc += dp
            pe_tgt[i, j, k + 1] = FT(pe_acc)
            dp_tgt[i, j, k] = FT(dp)
        end
    end
end

@kernel function _pe_from_mass_direct_kernel!(pe_tgt, dp_tgt, @Const(m),
                                                @Const(area), g_val, ptop, Hp, Nc, Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(pe_tgt)
        a = Float64(area[i, j])
        g = Float64(g_val)
        # Use Float64 accumulator to avoid precision loss in cumsum.
        pe_acc = Float64(ptop)
        pe_tgt[i, j, 1] = FT(pe_acc)
        for k in 1:Nz
            dp = Float64(m[ii, jj, k]) * g / a
            pe_acc += dp
            pe_tgt[i, j, k + 1] = FT(pe_acc)
            dp_tgt[i, j, k] = FT(dp)
        end
    end
end

# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

"""
    fix_target_bottom_pe!(ws_vr, m_src_panels, gc, grid)

Fix target PE to match source column pressure while preserving pure-pressure
levels.  Pure-pressure levels (bk[k]==0 && bk[k+1]==0) get dp_tgt set from
m_src directly (identity remap — no vertical transport where DELP is invariant).
Only hybrid levels (bk > 0) are scaled to absorb the PS difference from
horizontal mass divergence.  This eliminates a ~0.23 ppm/window noise floor
that the old uniform-scaling approach applied to all levels.
"""
function fix_target_bottom_pe!(ws_vr::VerticalRemapWorkspace,
                                 m_src_panels, gc, grid)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    FT = eltype(ws_vr.ak_dev)
    backend = get_backend(ws_vr.ak_dev)
    ptop = FT(grid.vertical.A[1])
    k! = _fix_column_pe_kernel!(backend, 256)

    for p in 1:6
        k!(ws_vr.pe_tgt[p], ws_vr.dp_tgt[p], m_src_panels[p],
           gc.area[p], gc.gravity, ws_vr.bk_dev, ptop, Hp, Nc, Nz;
           ndrange=(Nc, Nc))
    end
    synchronize(backend)
    return nothing
end

@kernel function _fix_column_pe_kernel!(pe_tgt, dp_tgt, @Const(m_src),
                                          @Const(area), g_val, @Const(bk),
                                          ptop, Hp, Nc, Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(pe_tgt)
        a = Float64(area[i, j])
        g = Float64(g_val)

        # Find last pure-pressure level: bk[k] ≈ 0 && bk[k+1] ≈ 0.
        # Use threshold to handle near-zero bk (e.g. 8e-9 at GEOS-IT k=42).
        # These levels have fixed DELP regardless of surface pressure,
        # so their target dp should equal source dp (identity remap).
        bk_thr = FT(1e-6)
        n_fixed = 0
        for k in 1:Nz
            if abs(bk[k]) < bk_thr && abs(bk[k + 1]) < bk_thr
                n_fixed = k
            else
                break
            end
        end

        # Source total dry pressure and pure-pressure subtotal
        ps_src64 = Float64(ptop)
        pe_fixed_src = Float64(ptop)
        for k in 1:Nz
            dp_src_k = Float64(m_src[ii, jj, k]) * g / a
            ps_src64 += dp_src_k
            if k <= n_fixed
                pe_fixed_src += dp_src_k
            end
        end

        # Target hybrid subtotal (before fix)
        pe_hybrid_raw = 0.0
        for k in n_fixed + 1:Nz
            pe_hybrid_raw += Float64(dp_tgt[i, j, k])
        end

        # Scale factor for hybrid levels only:
        # hybrid dp must sum to (ps_src - pe_fixed_src)
        ps_hybrid_target = ps_src64 - pe_fixed_src
        scale64 = if pe_hybrid_raw > 100.0 * Float64(eps(FT))
            ps_hybrid_target / pe_hybrid_raw
        else
            1.0
        end

        # Rebuild dp_tgt and pe_tgt:
        #  - pure-pressure levels: dp from m_src (identity remap)
        #  - hybrid levels: original dp_tgt × scale
        pe_acc = Float64(ptop)
        pe_tgt[i, j, 1] = FT(pe_acc)
        for k in 1:Nz
            if k <= n_fixed
                dp_tgt[i, j, k] = FT(Float64(m_src[ii, jj, k]) * g / a)
            else
                dp_tgt[i, j, k] = FT(Float64(dp_tgt[i, j, k]) * scale64)
            end
            pe_acc += Float64(dp_tgt[i, j, k])
            pe_tgt[i, j, k + 1] = FT(pe_acc)
        end
    end
end
