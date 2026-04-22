# ---------------------------------------------------------------------------
# TM5 column solver — plan 23 Commit 2.
#
# Backend-agnostic Julia implementation of the Tiedtke 1989 mass-flux
# convection matrix builder + partial-pivot LU solve, transcribed once
# from the TM5-4DVAR reference at deps/tm5/base/src/tm5_conv.F90:32–341
# into AtmosTransport orientation (k=1=TOA, k=Nz=surface; plan 23
# principle 1 — preprocessor writes forcings in this orientation, so
# the solver has zero runtime orientation logic).
#
# Algorithm overview (matches the Fortran line-by-line):
#
#   1. Diagnose cloud dims (icltop, iclbas, icllfs) from `detu` / `entd`.
#   2. Compute active range [icltop, Nz] (active sub-block of conv1).
#      If icltop > Nz (no convection in this column), identity-solve.
#   3. Build the flux-divergence matrix `conv1 = I - dt·D` within the
#      active range. Outside the active range, conv1 is identity
#      (by construction).
#   4. Factorize conv1[icltop:Nz, icltop:Nz] with partial pivoting.
#      Store the permutation vector in `pivots[icltop:Nz]` (plan 23
#      principle 3 — plan 19's adjoint replays this factorization
#      with trans='T'; do NOT fuse the permutation into the solved
#      coefficients).
#   5. Back-substitute each tracer's vertical profile within the
#      active range. Levels above icltop are untouched (identity
#      action).
#
# The matrix is dense lower + upper triangular within the active
# window (no banded structure to exploit; see
# artifacts/plan23/matrix_structure.md). Factorization cost is
# O(lmc³) where lmc = Nz - icltop + 1 is the active sub-block size.
# Solve cost is O(lmc² · Nt) per column.
# ---------------------------------------------------------------------------

"""
    _tm5_diagnose_cloud_dims(detu, entd, Nz) -> (icltop, iclbas, icllfs)

Diagnose cloud top, cloud base, and level of free sinking from the
updraft detrainment and downdraft entrainment profiles. All
indices in AtmosTransport orientation (k=1=TOA, k=Nz=surface).

- `icltop` = smallest k with `detu[k] > 0` (highest-altitude active
  updraft detrainment level; set to `Nz + 1` if no convection).
- `iclbas` = largest k with `detu[k] > 0` (lowest-altitude active
  updraft detrainment level; set to 0 if no convection).
- `icllfs` = smallest k with `entd[k] > 0` (highest-altitude
  downdraft entrainment level; set to `Nz + 1` if no downdraft).

Maps to TM5-4DVAR's `ConvCloudDim`
([`tmphys_convec.F90:39–132`](../../../deps/tm5/base/src/tmphys_convec.F90)).
TM5's "cloud top is HIGHEST level with detu > 0 searching top-down"
corresponds in AtmosTransport to "smallest k with detu[k] > 0
searching from k=1". The invariant `iclbas >= icltop` (Fortran
`iclbas > icltop`) becomes `iclbas >= icltop` in AtmosTransport
indexing.

`entu` is not needed for cloud-dim diagnosis — TM5 upstream uses
detu for the cloud top/base and entd for the LFS.
"""
@inline function _tm5_diagnose_cloud_dims(detu_col, entd_col,
                                           Nz::Integer)
    icltop = Nz + 1   # sentinel "no convection"
    iclbas = 0
    icllfs = Nz + 1

    @inbounds for k in 1:Nz
        if detu_col[k] > 0
            icltop == Nz + 1 && (icltop = k)
            iclbas = k
        end
        if entd_col[k] > 0 && icllfs == Nz + 1
            icllfs = k
        end
    end
    return (icltop, iclbas, icllfs)
end

"""
    _tm5_build_conv1!(conv1, entu_col, detu_col, entd_col, detd_col,
                      m_col, icltop, icllfs, dt, Nz) -> nothing

Build the transport matrix `conv1 = I - dt · D` in place. On entry,
`conv1` may hold arbitrary data; this function zeros it and writes
the identity + active-window entries. Indices run 1..Nz in
AtmosTransport orientation.

Follows [`tm5_conv.F90:32–191`](../../../deps/tm5/base/src/tm5_conv.F90)
block by block. Intermediate `fu` (updraft subplane) and `f`
(downdraft + subsidence) arrays are stack-allocated via `MVector`
from StaticArrays… or are they? For an arbitrary `Nz` we can't.
Commit 2 uses plain `Array`s sized `(Nz+1, Nz)` (TM5 Fortran's
`f(0:lmx, 1:lmx)` pattern, shifted by +1); perf tuning in Commit 4
can decide whether to batch-allocate these inside the workspace or
keep them per-column (the matrices are small — `Nz × Nz` floats ≤
72² × 4 bytes ≈ 20 KiB per column).

Reindex rules (AtmosTransport ↔ TM5; applied inline):

- TM5 `do k = 1, li` (surface → cloud top) becomes AtmosTransport
  `for k_atm in Nz:-1:icltop_atm` (surface → cloud top, k decreasing).
- TM5 `do k = ld, 2, -1` (level of free sinking → 2) becomes
  AtmosTransport `for k_atm in icllfs_atm:(Nz-1)` (level of free
  sinking → second-from-bottom, k increasing).
- TM5 `do k = 1, lmx-1` (surface → TOA-1) becomes AtmosTransport
  `for k_atm in Nz:-1:2`.

Mass-flux boundary: TM5 `amu(0) = 0` ("flux below the bottom
boundary") maps to AtmosTransport `amu_atIFC[Nz+1] = 0` (flux at
the surface interface, below the bottom-most air-mass layer).
"""
function _tm5_build_conv1!(conv1::AbstractMatrix{FT},
                           entu_col::AbstractVector{FT},
                           detu_col::AbstractVector{FT},
                           entd_col::AbstractVector{FT},
                           detd_col::AbstractVector{FT},
                           m_col::AbstractVector{FT},
                           icltop::Integer, icllfs::Integer,
                           dt::FT, Nz::Integer) where {FT}
    # Working storage: fu and f are (Nz+1, Nz) to mirror TM5's
    # f(0:lmx, 1:lmx) / fu(0:lmx, 1:lmx). We index `f[k+1, kk]`
    # for the TM5 `f(k, kk)` (shift the row index by +1).
    # amu and amd likewise get shape (Nz+1,) with index convention
    # `amu[k+1]` ↔ TM5 `amu(k)`.
    #
    # We allocate here in Commit 2 because the function is
    # backend-agnostic and used only in CPU tests. Commit 4 will
    # either thread these through the workspace or use MVector for
    # compile-time sizes.
    fu  = zeros(FT, Nz + 1, Nz)
    f   = zeros(FT, Nz + 1, Nz)
    amu = zeros(FT, Nz + 1)
    amd = zeros(FT, Nz + 1)

    # Index convention in AtmosTransport orientation (k=1=TOA,
    # k=Nz=surface):
    # - Layers use k ∈ 1:Nz with k_atm = Nz+1-k_tm5.
    # - Interface-valued quantities (amu, amd) use a length-(Nz+1)
    #   vector where amu[k_atm] is the mass flux at the top of layer
    #   k_atm, and amu[Nz+1] is the bottom boundary (TM5 amu(0) = 0).
    # - f and fu use (Nz+1, Nz); f[k_atm, kk_atm] is TM5 f(k, kk)
    #   with k_atm = Nz+1-k_tm5 (layer index shifted). The extra
    #   row at index Nz+1 represents TM5's f(0, kk) / fu(0, kk).

    # --- Updraft pass (AtmosTransport surface → cloud top) ------
    # For each layer k_atm iterated surface-up, compute amu[k_atm],
    # zxi, and propagate fu rows.
    @inbounds for k_atm in Nz:-1:icltop
        # TM5:  amu(k)   = amu(k-1) + entu(k) - detu(k)
        # Atm:  amu[k_atm] = amu[k_atm+1] + entu[k_atm] - detu[k_atm]
        amu[k_atm] = amu[k_atm + 1] + entu_col[k_atm] - detu_col[k_atm]
        if amu[k_atm] > 0
            denom = amu[k_atm + 1] + entu_col[k_atm]
            zxi = max(zero(FT), one(FT) - detu_col[k_atm] / denom)
        else
            amu[k_atm] = zero(FT)
            zxi = zero(FT)
        end
        # TM5:  fu(k, kk) = fu(k-1, kk) * zxi   for kk = 1..k-1
        # Atm:  fu[k_atm, kk_atm] = fu[k_atm+1, kk_atm] * zxi for
        #       kk_atm iterating "TM5 1..k-1" in AtmosTransport order.
        # TM5 kk = 1..k-1 is "layers strictly below the current one
        # in TM5" = "layers strictly surface-ward of current" in
        # AtmosTransport = "kk_atm in (k_atm+1):Nz".
        for kk_atm in (k_atm + 1):Nz
            fu[k_atm, kk_atm] = fu[k_atm + 1, kk_atm] * zxi
        end
        # TM5:  fu(k, k)   = entu(k) / m(k) * zxi   (diagonal)
        # Atm:  fu[k_atm, k_atm] = entu[k_atm] / m[k_atm] * zxi
        fu[k_atm, k_atm] = entu_col[k_atm] / m_col[k_atm] * zxi
    end

    # --- Downdraft pass (AtmosTransport icllfs → Nz-1) ----------
    # TM5: do k = ld, 2, -1    (ld = icllfs_tm5; iterate top-down)
    # Atm: iterate k_atm in icllfs:(Nz-1), bottom-up.
    # TM5 "k-1" = layer below current in TM5 = k_atm+1 in AtmosTransport.
    # TM5 kk = k+1..ld = layers ABOVE current in TM5 = layers TOA-ward =
    #   kk_atm in icllfs:(k_atm-1)  (above current in AtmosTransport).
    # The "result row" `f(k-1, kk)` is "layer below current in TM5" =
    #   f[k_atm+1, kk_atm] in AtmosTransport.
    if icllfs <= Nz
        @inbounds for k_atm in icllfs:(Nz - 1)
            # TM5:  amd(k-1) = amd(k) - entd(k) + detd(k)
            # Atm:  amd[k_atm+1] = amd[k_atm] - entd[k_atm] + detd[k_atm]
            amd[k_atm + 1] = amd[k_atm] - entd_col[k_atm] + detd_col[k_atm]
            if amd[k_atm + 1] < 0
                denom = amd[k_atm] - entd_col[k_atm]
                zxi = max(zero(FT), one(FT) + detd_col[k_atm] / denom)
            else
                amd[k_atm + 1] = zero(FT)
                zxi = zero(FT)
            end
            # TM5:  f(k-1, kk) = f(k, kk) * zxi   for kk = k+1..ld
            # Atm:  f[k_atm+1, kk_atm] = f[k_atm, kk_atm] * zxi
            #       for kk_atm in icllfs:(k_atm-1)
            for kk_atm in icllfs:(k_atm - 1)
                f[k_atm + 1, kk_atm] = f[k_atm, kk_atm] * zxi
            end
            # TM5:  f(k-1, k) = -entd(k) / m(k) * zxi
            # Atm:  f[k_atm+1, k_atm] = -entd[k_atm] / m[k_atm] * zxi
            f[k_atm + 1, k_atm] = -entd_col[k_atm] / m_col[k_atm] * zxi
        end
    end

    # --- Combine updraft + downdraft + subsidence --------------
    # TM5: do k=1, lmx-1 → Atm: for k_atm = Nz:-1:2
    #      f(k, kk) += fu(k, kk) for all kk
    #      f(k, k+1) -= amu(k) / m(k+1)        [subsidence, updraft]
    #      f(k, k)   -= amd(k) / m(k)          [subsidence, downdraft]
    # TM5 kk in 1..lmx → Atm: kk_atm in 1..Nz (any layer).
    # TM5 k+1 = layer above current in TM5 = k_atm-1 in Atm.
    @inbounds for k_atm in Nz:-1:2
        for kk_atm in 1:Nz
            f[k_atm, kk_atm] += fu[k_atm, kk_atm]
        end
        f[k_atm, k_atm - 1] -= amu[k_atm] / m_col[k_atm - 1]
        f[k_atm, k_atm]     -= amd[k_atm] / m_col[k_atm]
    end

    # --- Final assembly: conv1(k, kk) = -dt · (f(k-1, kk) - f(k, kk)) + I
    # TM5: do k=1, lmx → Atm: for k_atm = 1:Nz (any order).
    # TM5 "f(k-1, kk)" = "row below current in TM5" = f[k_atm+1, kk_atm].
    fill!(conv1, zero(FT))
    @inbounds for k_atm in 1:Nz
        for kk_atm in 1:Nz
            fdiff = f[k_atm + 1, kk_atm] - f[k_atm, kk_atm]
            if fdiff != zero(FT)
                conv1[k_atm, kk_atm] = -dt * fdiff
            end
        end
        conv1[k_atm, k_atm] += one(FT)
    end
    return nothing
end

"""
    _tm5_lu!(conv1, pivots, Nz) -> nothing

In-place partial-pivot Gaussian elimination on `conv1[1:Nz, 1:Nz]`.
Writes the L and U factors into the same matrix and the row
permutation into `pivots[1:Nz]`.

We factorize the full matrix rather than restricting to
`[icltop, Nz]` because the matrix has non-identity entries beyond
the strict cloud window: the combine + subsidence step
(`_tm5_build_conv1!` lines covering `do k = Nz:-1:2`) propagates
fluxes into rows outside `[icltop, Nz]` whenever the updraft or
downdraft passes touch adjacent layers. Identity rows (above the
actively-modified range) factorize trivially — no row swap is
needed and the back-substitution is a no-op. The `O(Nz³)` worst
case is still cheap at production `Nz ≤ 72` (~370k flops / column).

Partial pivoting is retained per plan 23 principle 3 for adjoint
replay, even though TM5's diagonally-dominant conv1 rarely needs
swaps in practice.
"""
function _tm5_lu!(conv1::AbstractMatrix{FT},
                  pivots::AbstractVector{<:Integer},
                  Nz::Integer) where {FT}
    Nz == 0 && return nothing

    @inbounds for k in 1:Nz
        # Find the pivot row within column k, range [k, Nz].
        piv = k
        pivmag = abs(conv1[k, k])
        for r in (k + 1):Nz
            mag = abs(conv1[r, k])
            if mag > pivmag
                piv = r
                pivmag = mag
            end
        end
        pivots[k] = piv
        if piv != k
            for c in 1:Nz
                tmp = conv1[k, c]
                conv1[k, c] = conv1[piv, c]
                conv1[piv, c] = tmp
            end
        end
        # Diagonal is guaranteed non-zero by diagonal dominance,
        # but protect against degenerate input just in case.
        diag = conv1[k, k]
        if diag == zero(FT)
            # Singular column — leave matrix untouched; solve will
            # propagate NaN/Inf. Upstream validators (preprocessor
            # sanity probe, Commit 3) should have caught this.
            continue
        end
        invd = one(FT) / diag
        for r in (k + 1):Nz
            conv1[r, k] *= invd
        end
        for r in (k + 1):Nz
            lrk = conv1[r, k]
            for c in (k + 1):Nz
                conv1[r, c] -= lrk * conv1[k, c]
            end
        end
    end
    return nothing
end

"""
    _tm5_solve!(rm_col, conv1, pivots, Nz, Nt) -> nothing

Apply the factored `conv1` + stored pivots to `rm_col` (shape
`(Nz, Nt)`). Back-substitutes in place for each tracer `1..Nt`.
"""
function _tm5_solve!(rm_col::AbstractMatrix{FT},
                     conv1::AbstractMatrix{FT},
                     pivots::AbstractVector{<:Integer},
                     Nz::Integer,
                     Nt::Integer) where {FT}
    Nz == 0 && return nothing
    @inbounds for t in 1:Nt
        # Apply permutation.
        for k in 1:Nz
            piv = pivots[k]
            if piv != k
                tmp = rm_col[k, t]
                rm_col[k, t] = rm_col[piv, t]
                rm_col[piv, t] = tmp
            end
        end
        # Forward solve L y = b (unit-diagonal L).
        for k in 1:Nz
            s = rm_col[k, t]
            for j in 1:(k - 1)
                s -= conv1[k, j] * rm_col[j, t]
            end
            rm_col[k, t] = s
        end
        # Back solve U x = y.
        for k in Nz:-1:1
            s = rm_col[k, t]
            for j in (k + 1):Nz
                s -= conv1[k, j] * rm_col[j, t]
            end
            rm_col[k, t] = s / conv1[k, k]
        end
    end
    return nothing
end

"""
    _tm5_solve_column!(rm_col, m_col, entu_col, detu_col, entd_col, detd_col,
                       conv1_buf, pivots_buf, cloud_dims, dt) -> nothing

Backend-agnostic full column solver. Shapes:

- `rm_col      :: AbstractMatrix{FT}`  (Nz, Nt) — tracer mixing ratios; updated in place.
- `m_col       :: AbstractVector{FT}`  (Nz,)    — layer air mass per unit area (kg/m²).
- `entu_col, detu_col, entd_col, detd_col :: AbstractVector{FT}` (Nz,) — forcings.
- `conv1_buf   :: AbstractMatrix{FT}`  (Nz, Nz) — scratch matrix.
- `pivots_buf  :: AbstractVector{Int}` (Nz,)    — scratch permutation.
- `cloud_dims  :: AbstractVector{Int}` (3,)     — output (icltop, iclbas, icllfs)
                                                  in AtmosTransport orientation.
- `dt          :: FT`                            — step length in seconds.

Zero-forcing (`entu = detu = entd = detd = 0`) short-circuits to
identity. Zero-size columns (`Nz = 0`) are a no-op.
"""
function _tm5_solve_column!(rm_col::AbstractMatrix{FT},
                            m_col::AbstractVector{FT},
                            entu_col::AbstractVector{FT},
                            detu_col::AbstractVector{FT},
                            entd_col::AbstractVector{FT},
                            detd_col::AbstractVector{FT},
                            conv1_buf::AbstractMatrix{FT},
                            pivots_buf::AbstractVector{<:Integer},
                            cloud_dims::AbstractVector{<:Integer},
                            dt::FT) where {FT}
    Nz = length(m_col)
    Nz == 0 && return nothing
    Nt = size(rm_col, 2)

    icltop, iclbas, icllfs = _tm5_diagnose_cloud_dims(detu_col, entd_col, Nz)
    cloud_dims[1] = icltop
    cloud_dims[2] = iclbas
    cloud_dims[3] = icllfs

    # No convection in this column → identity transform.
    if icltop > Nz
        return nothing
    end

    _tm5_build_conv1!(conv1_buf,
                      entu_col, detu_col, entd_col, detd_col, m_col,
                      icltop, icllfs, dt, Nz)
    _tm5_lu!(conv1_buf, pivots_buf, Nz)
    _tm5_solve!(rm_col, conv1_buf, pivots_buf, Nz, Nt)
    return nothing
end
