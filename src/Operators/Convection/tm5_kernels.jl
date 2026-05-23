# ---------------------------------------------------------------------------
# TM5 convection kernels — plan 23 Commit 4 + storage-redesign Commit 4.
#
# Thin KernelAbstractions wrappers around `_tm5_solve_column!`
# (tm5_column_solve.jl).  One kernel per topology:
#
#   _tm5_column_kernel!             — LatLon 4D `(Nx, Ny, Nz, Nt)` state.
#   _tm5_faceindexed_column_kernel! — ReducedGaussian 3D `(ncells, Nz, Nt)` state.
#   _tm5_cs_panel_column_kernel!    — CubedSphere per-panel 4D
#                                     `(Nc+2Hp, Nc+2Hp, Nz, Nt)` state.
#
# Storage-plan Commit 4: every kernel runs on a 1D ndrange of `B`
# tile cells. The host wraps the launch in a tile loop, biasing
# the per-tile global cell index by `tile_offset`. The workspace
# slabs are flat `(Nz, Nz, B)` / `(Nz, B)` / `(3, B)` /
# `(Nz+1, B)` and are indexed by the local tile slot
# `t = @index(Global)`. This decouples GPU memory from grid
# resolution.
#
# Each kernel:
#   1. Reads `t = @index(Global)` (the local tile slot).
#   2. Computes `c_global = tile_offset + t` (the topology cell index).
#   3. Returns early if `c_global > N_total` (trailing tile guard).
#   4. Decodes `c_global` to topology coordinates (column-major;
#      LatLon → (i, j); RG → c; CS → (c1, c2) and halo-shifted (i, j)).
#   5. Slices per-column views of the workspace using `t` (local).
#   6. Calls `_tm5_solve_column!`.
#
# CLAUDE.md gotcha: the kernel uses `@view` for per-column slices
# because KA + CUDA tolerate SubArrays here — the surrounding
# kernel body is non-trivial.  No allocation inside the kernel
# (mandatory for GPU correctness, plan 23 principle 4).
# ---------------------------------------------------------------------------

# LatLon (structured 4D): state.tracers_raw :: (Nx, Ny, Nz, Nt),
# air_mass :: (Nx, Ny, Nz), forcing fields :: (Nx, Ny, Nz),
# tile workspace conv1 :: (Nz, Nz, B), pivots :: (Nz, B),
# cloud_dims :: (3, B), f :: (Nz, Nz, B) aliased to conv1,
# amu/amd :: (Nz+1, B).
@kernel function _tm5_column_kernel!(
    q_raw, @Const(air_mass),
    @Const(entu), @Const(detu), @Const(entd), @Const(detd),
    @Const(cell_areas_y),
    conv1_ws, pivots_ws, cloud_dims_ws,
    f_ws, amu_ws, amd_ws,
    tile_offset::Int, Nx::Int, dt,
)
    # The host clamps `ndrange = min(B, N_total - tile_off)` so
    # `t` is always in 1..ndrange and `c_global` is always a valid
    # cell index. No trailing-tile guard needed (KA also forbids
    # `return` inside a kernel body).
    t = @index(Global)
    c_global = tile_offset + t
    # Column-major decode: c_global = i + Nx*(j-1).
    i = ((c_global - 1) % Nx) + 1
    j = ((c_global - 1) ÷ Nx) + 1
    rm_col    = @view q_raw[i, j, :, :]
    m_col     = @view air_mass[i, j, :]
    entu_col  = @view entu[i, j, :]
    detu_col  = @view detu[i, j, :]
    entd_col  = @view entd[i, j, :]
    detd_col  = @view detd[i, j, :]
    conv1_col  = @view conv1_ws[:, :, t]
    pivots_col = @view pivots_ws[:, t]
    cloud_col  = @view cloud_dims_ws[:, t]
    f_col      = @view f_ws[:, :, t]
    amu_col    = @view amu_ws[:, t]
    amd_col    = @view amd_ws[:, t]
    cell_area  = cell_areas_y[j]
    _tm5_solve_column!(rm_col, m_col,
                        entu_col, detu_col, entd_col, detd_col,
                        conv1_col, pivots_col, cloud_col, dt;
                        cell_area = cell_area,
                        f_buf = f_col,
                        amu_buf = amu_col, amd_buf = amd_col)
end

# Face-indexed ReducedGaussian: state.tracers_raw :: (ncells, Nz, Nt).
# Single-axis cell index — no decode needed.
@kernel function _tm5_faceindexed_column_kernel!(
    q_raw, @Const(air_mass),
    @Const(entu), @Const(detu), @Const(entd), @Const(detd),
    @Const(cell_areas),
    conv1_ws, pivots_ws, cloud_dims_ws,
    f_ws, amu_ws, amd_ws,
    tile_offset::Int, dt,
)
    t = @index(Global)
    c_global = tile_offset + t
    rm_col    = @view q_raw[c_global, :, :]
    m_col     = @view air_mass[c_global, :]
    entu_col  = @view entu[c_global, :]
    detu_col  = @view detu[c_global, :]
    entd_col  = @view entd[c_global, :]
    detd_col  = @view detd[c_global, :]
    conv1_col  = @view conv1_ws[:, :, t]
    pivots_col = @view pivots_ws[:, t]
    cloud_col  = @view cloud_dims_ws[:, t]
    f_col      = @view f_ws[:, :, t]
    amu_col    = @view amu_ws[:, t]
    amd_col    = @view amd_ws[:, t]
    cell_area  = cell_areas[c_global]
    _tm5_solve_column!(rm_col, m_col,
                        entu_col, detu_col, entd_col, detd_col,
                        conv1_col, pivots_col, cloud_col, dt;
                        cell_area = cell_area,
                        f_buf = f_col,
                        amu_buf = amu_col, amd_buf = amd_col)
end

# CubedSphere panel: q_raw_panel :: (Nc+2Hp, Nc+2Hp, Nz, Nt),
# air_mass_panel :: (Nc+2Hp, Nc+2Hp, Nz), forcing fields
# :: (Nc, Nc, Nz) (halo-free per panel). The workspace is shared
# across panels — `apply_convection!` launches one panel at a time
# and KA stream ordering keeps panel n+1 from starting until panel
# n's writes are visible.
@kernel function _tm5_cs_panel_column_kernel!(
    q_raw_panel, @Const(air_mass_panel),
    @Const(entu_panel), @Const(detu_panel),
    @Const(entd_panel), @Const(detd_panel),
    @Const(cell_areas_panel),
    conv1_panel, pivots_panel, cloud_panel,
    f_panel, amu_panel, amd_panel,
    Hp::Int, tile_offset::Int, Nc::Int, dt,
)
    t = @index(Global)
    c_global = tile_offset + t
    # Column-major decode: c_global = c1 + Nc*(c2-1).
    c1 = ((c_global - 1) % Nc) + 1
    c2 = ((c_global - 1) ÷ Nc) + 1
    # Halo-offset indices into the padded state arrays.
    i = c1 + Hp
    j = c2 + Hp
    rm_col    = @view q_raw_panel[i, j, :, :]
    m_col     = @view air_mass_panel[i, j, :]
    entu_col  = @view entu_panel[c1, c2, :]
    detu_col  = @view detu_panel[c1, c2, :]
    entd_col  = @view entd_panel[c1, c2, :]
    detd_col  = @view detd_panel[c1, c2, :]
    conv1_col  = @view conv1_panel[:, :, t]
    pivots_col = @view pivots_panel[:, t]
    cloud_col  = @view cloud_panel[:, t]
    f_col      = @view f_panel[:, :, t]
    amu_col    = @view amu_panel[:, t]
    amd_col    = @view amd_panel[:, t]
    cell_area  = cell_areas_panel[c1, c2]
    _tm5_solve_column!(rm_col, m_col,
                        entu_col, detu_col, entd_col, detd_col,
                        conv1_col, pivots_col, cloud_col, dt;
                        cell_area = cell_area,
                        f_buf = f_col,
                        amu_buf = amu_col, amd_buf = amd_col)
end

# ---------------------------------------------------------------------------
# Workgroup-collaborative kernels — one workgroup per column, WG_SIZE threads
# cooperatively factor an Nz×Nz matrix in `@localmem`.
#
# Why these exist: the per-thread kernels above run one serial LU per column
# per GPU thread. On a column of Nz=85 that is ~200 k flops in a single
# thread of an SM — fundamentally a poor GPU pattern (no warp-level
# parallelism inside the LU; ~28 KB per-thread working set; branch
# divergence on `icltop_eff`).
#
# The collaborative kernels put the column's matrix in workgroup-local
# memory and have WG_SIZE threads cooperatively (a) build, (b) factor
# with partial pivoting, (c) apply pivots + forward solve + back solve.
# Measured on a C180 panel (Nz=85, Nt=2, L40S): the collaborative
# kernel runs at 53–63 ms / panel versus 397–692 ms for the per-thread
# baseline — an 8–11× speedup that is bit-exact to baseline within
# Float32 rounding (≈ 7e-7 max abs deviation).
#
# Compile-time `LMAX_CONV` parameter:
#   `@localmem` allocations need compile-time sizes, so the matrix size
#   is propagated via `Val{LMAX_CONV}`. The host extracts `LMAX_CONV = size(air_mass)[end]`
#   and constructs `Val(LMAX_CONV)` at kernel-launch time; KA compiles one
#   variant per unique `LMAX_CONV` it sees. This keeps the per-workgroup
#   shared-memory footprint exactly `LMAX_CONV² · sizeof(FT) + O(LMAX_CONV)`.
#
#   Realistic LMAX_CONV values: 60, 72, 85, 91, 137. The largest fitting under
#   Metal's 32 KB threadgroup-memory limit is ~85 (28.9 KB for A_loc plus
#   ~3 KB scratch). For LMAX_CONV > 85 the dispatch falls back to the per-thread
#   kernel above; a runtime assertion at the host side enforces this.
#
# `NT_MAX` constant:
#   The per-column RHS storage `q_loc` is sized `(LMAX_CONV, NT_MAX)` for a
#   compile-time max tracer count of 8. Production runs with more than
#   8 tracers fall back to the per-thread kernel (runtime check at the
#   host side). The number was chosen so that the total `@localmem`
#   footprint at LMAX_CONV=85, NT_MAX=8 stays ≤ 32 KB (28.9 KB for A_loc plus
#   2.7 KB for q_loc plus ~0.7 KB scratch = ~32.3 KB; a soft warning,
#   not a hard error — Metal threadgroup memory varies by chip).
#
# Implementation notes:
#   - The matrix build (Phase 1) runs sequentially in thread 1 because the
#     `amu[k]` and `amd[k]` recurrences have a serial dependency. The
#     inner `kk` loops on each `k` could be parallelised; left as future
#     work because the build is only ~3.5% of the LU's flop count.
#   - The pivot search inside the LU (Phase 3) also runs in thread 1.
#     A parallel-reduction pivot search would shave a few µs per outer
#     iteration; left as future work for the same reason.
#   - The shared-memory layout aliases A_loc and the build's intermediate
#     `f` matrix, matching `_tm5_build_conv1!`'s production behaviour.
# ---------------------------------------------------------------------------

const _TM5_COLLAB_WG_SIZE = 32
# Compile-time tracer-slot count for the per-workgroup RHS storage
# `q_loc`. Production CO2 inversions use Nt ∈ {1..6}; configurations
# with Nt > 6 fall back to the per-thread kernel.
#
# This number is bounded by Metal's 32 KiB threadgroup-memory budget:
# total per-workgroup `@localmem` is `4·Nz² + (4·NT_MAX + 16)·Nz + 16`
# bytes (A_loc + q_loc + piv_loc + bmass_loc + amu/amd + icl_top/lfs).
# At Nz=85 the max NT_MAX that fits Metal's 32 768 bytes is 6 (giving
# 32 316 B); NT_MAX=8 would need 32 996 B and overflow the M1/M2/M3
# baseline. Confirmed by an external code review on 2026-05-22.
const _TM5_COLLAB_NT_MAX  = 6

"""
    _tm5_collab_supports(lmax_conv, Nt) -> Bool

True when the collaborative kernels can run for the given convection-
matrix size and tracer count. The host uses this to decide between
the new collab path and the legacy per-thread path; outside the
supported envelope the legacy kernels remain numerically correct,
just slower.

The `lmax_conv ≤ 85` ceiling is set by Metal's 32 KB threadgroup-
memory baseline: at lmax_conv=85, NT_MAX=6 the per-workgroup
`@localmem` footprint is ~32.3 KB. With NT_MAX dropped to 6 in this
patch, lmax_conv=85 fits Metal comfortably. lmax_conv=91 would need
~36 KB and refuses to launch on Apple Silicon — so the envelope is
set to 85 here.

Critically, the gate is on **lmax_conv**, not on the underlying
total `Nz`. This unlocks ml91/tropo60 and ml137/tropo34a-style
configurations (TM5's own approach): a high-resolution vertical
grid (Nz=91 or 137) coupled with a smaller convection ceiling that
fits the threadgroup budget. The kernel allocates `@localmem` at
`(lmax_conv, lmax_conv)`; layers `k ≤ Nz - lmax_conv` are pass-
through (no convection writes there). Production must verify via
`scripts/diagnostics/per_column_depth_histogram.jl` that the
binary's deepest observed convection fits inside `[Nz - lmax_conv +
1, Nz]` so stratospheric overshoots aren't silently dropped.
"""
@inline _tm5_collab_supports(lmax_conv::Integer, Nt::Integer) =
    lmax_conv > 0 && lmax_conv <= 85 && Nt > 0 && Nt <= _TM5_COLLAB_NT_MAX

# The collaborative-solve body is inlined verbatim into all three
# topology kernels below. We tried sharing it via a Julia macro and a
# `@inline` helper, but KA's `@kernel` only recognises `@synchronize`
# / `@index` / `@localmem` when they appear directly in its own AST —
# it does not re-process macro expansions, so a shared helper produces
# unbound-`__ctx__` errors at runtime.
#
# The trade-off: ~150 lines of mechanical duplication per topology vs.
# a fragile macro that depends on KA's internals. Duplication is the
# safer choice given the bit-exact contract.
#
# Topology-specific differences across the three copies are limited
# to four lines per kernel:
#   - the index decode (LL → (i, j); RG → c; CS → (c1, c2))
#   - the `cell_areas_*[…]` lookup
#   - the `_tm5_read_*` / `_tm5_write_q!` Val tag (:ll / :rg / :cs)
#   - the halo offset Hp (always 0 for LL / RG; mesh.Hp for CS)
# Everything else (boundary init, cloud-dim diagnosis, bmass, build,
# LU, pivot-apply, forward/back solve, store-back) is identical.
#
# Shared-memory allocations per kernel:
#   A_loc      : @localmem Float32 (LMAX_CONV, LMAX_CONV)
#   q_loc      : @localmem Float32 (LMAX_CONV, NT_MAX)
#   piv_loc    : @localmem Int32   (LMAX_CONV,)
#   bmass_loc  : @localmem Float32 (LMAX_CONV,)
#   amu_loc    : @localmem Float32 (LMAX_CONV + 1,)
#   amd_loc    : @localmem Float32 (LMAX_CONV + 1,)
#   icl_top    : @localmem Int32   (1,)
#   icl_lfs    : @localmem Int32   (1,)


# LatLon collaborative kernel.  `q_raw :: (Nx, Ny, Nz, Nt)`.
# One workgroup per column, WG_SIZE threads cooperate; ndrange
# = WG_SIZE × (Nx · Ny).  The kernel body is inlined verbatim
# below; see the comment block above for why we don't share a
# helper.
@kernel function _tm5_column_collab_kernel!(
    q_raw_arr,            # (Nx, Ny, Nz, Nt)
    @Const(air_mass_arr), # (Nx, Ny, Nz)
    @Const(entu), @Const(detu), @Const(entd), @Const(detd),
    @Const(cell_areas_y), # (Ny,) — LL contract
    Nx::Int, Nz::Int, Nt::Int, dt::Float32, ::Val{LMAX_CONV},
) where {LMAX_CONV}
    @uniform g = @index(Group)
    t = @index(Local)

    A_loc     = @localmem Float32 (LMAX_CONV, LMAX_CONV)
    q_loc     = @localmem Float32 (LMAX_CONV, _TM5_COLLAB_NT_MAX)
    piv_loc   = @localmem Int32   (LMAX_CONV,)
    bmass_loc = @localmem Float32 (LMAX_CONV,)
    amu_loc   = @localmem Float32 (LMAX_CONV + 1,)
    amd_loc   = @localmem Float32 (LMAX_CONV + 1,)
    icl_top   = @localmem Int32   (1,)
    icl_lfs   = @localmem Int32   (1,)

    # Topology-specific decode (LL).
    @uniform i = ((g - 1) % Nx) + 1
    @uniform j = ((g - 1) ÷ Nx) + 1
    @uniform Hp = 0
    @uniform area = cell_areas_y[j]
    @uniform k_shift = Nz - LMAX_CONV
    # WG_SIZE is _TM5_COLLAB_WG_SIZE (32); used directly below to satisfy KA scoping.

    # ---- 1. Boundary init (parallel) -------------------------
    @inbounds for k in t:_TM5_COLLAB_WG_SIZE:(LMAX_CONV + 1)
        amu_loc[k] = 0f0
        amd_loc[k] = 0f0
    end

    # ---- 2. Cloud-dim diagnosis (serial in thread 1) ---------
    if t == 1
        icl_top[1] = Int32(LMAX_CONV + 1)
        icl_lfs[1] = Int32(LMAX_CONV + 1)
        @inbounds for k in 1:LMAX_CONV
            d = _tm5_read_forcing(detu, i, j, Hp, k_shift + k, Val(:ll))
            if d > 0f0 && icl_top[1] == LMAX_CONV + 1
                icl_top[1] = Int32(k)
            end
            e = _tm5_read_forcing(entd, i, j, Hp, k_shift + k, Val(:ll))
            if e > 0f0 && icl_lfs[1] == LMAX_CONV + 1
                icl_lfs[1] = Int32(k)
            end
        end
    end
    @synchronize

    icltop = Int(icl_top[1])
    icllfs = Int(icl_lfs[1])
    icltop_eff = min(icllfs, max(icltop, 2) - 1)
    k_lo = max(icltop_eff, 1)
    no_conv = icltop > LMAX_CONV

    # ---- 3. bmass per layer (parallel) -----------------------
    @inbounds for k in t:_TM5_COLLAB_WG_SIZE:LMAX_CONV
        bmass_loc[k] = _tm5_read_mass(air_mass_arr, i, j, Hp, k_shift + k, Val(:ll)) / area
    end
    @synchronize

    # ---- 4. Matrix build -------------------------------------
    if !no_conv
        @inbounds for idx in t:_TM5_COLLAB_WG_SIZE:(LMAX_CONV * LMAX_CONV)
            k_  = (idx - 1) % LMAX_CONV + 1
            kk_ = (idx - 1) ÷ LMAX_CONV + 1
            if k_ >= k_lo && kk_ >= k_lo
                A_loc[k_, kk_] = 0f0
            elseif k_ == kk_
                A_loc[k_, kk_] = 1f0
            else
                A_loc[k_, kk_] = 0f0
            end
        end
        @synchronize

        if t == 1
            # Updraft pass.
            @inbounds for k in LMAX_CONV:-1:icltop
                e = _tm5_read_forcing(entu, i, j, Hp, k_shift + k, Val(:ll))
                d = _tm5_read_forcing(detu, i, j, Hp, k_shift + k, Val(:ll))
                amu_loc[k] = amu_loc[k + 1] + e - d
                zxi = 0f0
                if amu_loc[k] > 0f0
                    denom = amu_loc[k + 1] + e
                    zxi = max(0f0, 1f0 - d / denom)
                else
                    amu_loc[k] = 0f0
                end
                for kk in (k + 1):LMAX_CONV
                    f_below = k == LMAX_CONV ? 0f0 : A_loc[k + 1, kk]
                    A_loc[k, kk] = f_below * zxi
                end
                bmass_k = bmass_loc[k]
                A_loc[k, k] = e / bmass_k * zxi
            end
            # Downdraft pass.
            if icllfs <= LMAX_CONV
                @inbounds for k in icllfs:(LMAX_CONV - 1)
                    e = _tm5_read_forcing(entd, i, j, Hp, k_shift + k, Val(:ll))
                    d = _tm5_read_forcing(detd, i, j, Hp, k_shift + k, Val(:ll))
                    amd_loc[k + 1] = amd_loc[k] - e + d
                    zxi = 0f0
                    if amd_loc[k + 1] < 0f0
                        denom = amd_loc[k] - e
                        zxi = max(0f0, 1f0 + d / denom)
                    else
                        amd_loc[k + 1] = 0f0
                    end
                    for kk in icllfs:(k - 1)
                        A_loc[k + 1, kk] = A_loc[k, kk] * zxi
                    end
                    bmass_k = bmass_loc[k]
                    A_loc[k + 1, k] = -e / bmass_k * zxi
                end
            end
            # Subsidence subtraction.
            @inbounds for k in LMAX_CONV:-1:2
                bmass_above = bmass_loc[k - 1]
                bmass_k = bmass_loc[k]
                A_loc[k, k - 1] -= amu_loc[k] / bmass_above
                A_loc[k, k]     -= amd_loc[k] / bmass_k
            end
            # Final f -> conv1 = I - dt·(f_below - f).
            @inbounds for k in 1:LMAX_CONV
                for kk in 1:LMAX_CONV
                    if k < k_lo || kk < k_lo
                        A_loc[k, kk] = (k == kk) ? 1f0 : 0f0
                        continue
                    end
                    f_below = k == LMAX_CONV ? 0f0 : A_loc[k + 1, kk]
                    fdiff = f_below - A_loc[k, kk]
                    A_loc[k, kk] = -dt * fdiff
                end
                A_loc[k, k] += 1f0
            end
        end
    end
    @synchronize

    # ---- 5. Load RHS (parallel) ------------------------------
    @inbounds for idx in t:_TM5_COLLAB_WG_SIZE:(LMAX_CONV * Nt)
        k_  = (idx - 1) % LMAX_CONV + 1
        tt = (idx - 1) ÷ LMAX_CONV + 1
        q_loc[k_, tt] = _tm5_read_q(q_raw_arr, i, j, Hp, k_shift + k_, tt, Val(:ll))
    end
    @synchronize

    # ---- 6. LU + pivot apply + forward/back solve ------------
    if !no_conv
        for k in k_lo:LMAX_CONV
            if t == 1
                piv = k
                pivmag = abs(A_loc[k, k])
                @inbounds for r in (k + 1):LMAX_CONV
                    m_ = abs(A_loc[r, k])
                    if m_ > pivmag
                        piv = r
                        pivmag = m_
                    end
                end
                piv_loc[k] = Int32(piv)
            end
            @synchronize
            piv = Int(piv_loc[k])

            if piv != k
                @inbounds for cc in (k_lo + t - 1):_TM5_COLLAB_WG_SIZE:LMAX_CONV
                    tmp = A_loc[k, cc]
                    A_loc[k, cc] = A_loc[piv, cc]
                    A_loc[piv, cc] = tmp
                end
            end
            @synchronize

            diag_val = A_loc[k, k]
            @inbounds for r in (k + t):_TM5_COLLAB_WG_SIZE:LMAX_CONV
                A_loc[r, k] /= diag_val
            end
            @synchronize

            @inbounds for cc in (k + t):_TM5_COLLAB_WG_SIZE:LMAX_CONV
                akc = A_loc[k, cc]
                for r in (k + 1):LMAX_CONV
                    A_loc[r, cc] -= A_loc[r, k] * akc
                end
            end
            @synchronize
        end

        if t == 1
            @inbounds for k in k_lo:LMAX_CONV
                piv = Int(piv_loc[k])
                if piv != k
                    for tt in 1:Nt
                        tmp = q_loc[k, tt]
                        q_loc[k, tt] = q_loc[piv, tt]
                        q_loc[piv, tt] = tmp
                    end
                end
            end
        end
        @synchronize

        @inbounds for tt in t:_TM5_COLLAB_WG_SIZE:Nt
            for k in k_lo:LMAX_CONV
                s = q_loc[k, tt]
                for j2 in k_lo:(k - 1)
                    s -= A_loc[k, j2] * q_loc[j2, tt]
                end
                q_loc[k, tt] = s
            end
        end
        @synchronize

        @inbounds for tt in t:_TM5_COLLAB_WG_SIZE:Nt
            for k in LMAX_CONV:-1:k_lo
                s = q_loc[k, tt]
                for j2 in (k + 1):LMAX_CONV
                    s -= A_loc[k, j2] * q_loc[j2, tt]
                end
                q_loc[k, tt] = s / A_loc[k, k]
            end
        end
        @synchronize
    end

    # ---- 7. Store back (parallel) ----------------------------
    @inbounds for idx in t:_TM5_COLLAB_WG_SIZE:(LMAX_CONV * Nt)
        k_  = (idx - 1) % LMAX_CONV + 1
        tt = (idx - 1) ÷ LMAX_CONV + 1
        _tm5_write_q!(q_raw_arr, i, j, Hp, k_shift + k_, tt, q_loc[k_, tt], Val(:ll))
    end
end

# Face-indexed (ReducedGaussian) collaborative kernel.
# `q_raw :: (ncells, Nz, Nt)`.  See LL kernel above for why the
# body is inlined verbatim instead of factored into a helper.
@kernel function _tm5_faceindexed_column_collab_kernel!(
    q_raw_arr,            # (ncells, Nz, Nt)
    @Const(air_mass_arr), # (ncells, Nz)
    @Const(entu), @Const(detu), @Const(entd), @Const(detd),
    @Const(cell_areas),   # (ncells,)
    Nz::Int, Nt::Int, dt::Float32, ::Val{LMAX_CONV},
) where {LMAX_CONV}
    @uniform g = @index(Group)
    t = @index(Local)

    A_loc     = @localmem Float32 (LMAX_CONV, LMAX_CONV)
    q_loc     = @localmem Float32 (LMAX_CONV, _TM5_COLLAB_NT_MAX)
    piv_loc   = @localmem Int32   (LMAX_CONV,)
    bmass_loc = @localmem Float32 (LMAX_CONV,)
    amu_loc   = @localmem Float32 (LMAX_CONV + 1,)
    amd_loc   = @localmem Float32 (LMAX_CONV + 1,)
    icl_top   = @localmem Int32   (1,)
    icl_lfs   = @localmem Int32   (1,)

    # Topology-specific decode (RG).  `b` is unused but kept so the
    # `_tm5_*` helpers see a uniform signature across topologies.
    @uniform c = g
    @uniform b = 0
    @uniform Hp = 0
    @uniform area = cell_areas[c]
    @uniform k_shift = Nz - LMAX_CONV
    # WG_SIZE is _TM5_COLLAB_WG_SIZE (32); used directly below to satisfy KA scoping.

    @inbounds for k in t:_TM5_COLLAB_WG_SIZE:(LMAX_CONV + 1)
        amu_loc[k] = 0f0
        amd_loc[k] = 0f0
    end

    if t == 1
        icl_top[1] = Int32(LMAX_CONV + 1)
        icl_lfs[1] = Int32(LMAX_CONV + 1)
        @inbounds for k in 1:LMAX_CONV
            d = _tm5_read_forcing(detu, c, b, Hp, k_shift + k, Val(:rg))
            if d > 0f0 && icl_top[1] == LMAX_CONV + 1
                icl_top[1] = Int32(k)
            end
            e = _tm5_read_forcing(entd, c, b, Hp, k_shift + k, Val(:rg))
            if e > 0f0 && icl_lfs[1] == LMAX_CONV + 1
                icl_lfs[1] = Int32(k)
            end
        end
    end
    @synchronize

    icltop = Int(icl_top[1])
    icllfs = Int(icl_lfs[1])
    icltop_eff = min(icllfs, max(icltop, 2) - 1)
    k_lo = max(icltop_eff, 1)
    no_conv = icltop > LMAX_CONV

    @inbounds for k in t:_TM5_COLLAB_WG_SIZE:LMAX_CONV
        bmass_loc[k] = _tm5_read_mass(air_mass_arr, c, b, Hp, k_shift + k, Val(:rg)) / area
    end
    @synchronize

    if !no_conv
        @inbounds for idx in t:_TM5_COLLAB_WG_SIZE:(LMAX_CONV * LMAX_CONV)
            k_  = (idx - 1) % LMAX_CONV + 1
            kk_ = (idx - 1) ÷ LMAX_CONV + 1
            if k_ >= k_lo && kk_ >= k_lo
                A_loc[k_, kk_] = 0f0
            elseif k_ == kk_
                A_loc[k_, kk_] = 1f0
            else
                A_loc[k_, kk_] = 0f0
            end
        end
        @synchronize

        if t == 1
            @inbounds for k in LMAX_CONV:-1:icltop
                e = _tm5_read_forcing(entu, c, b, Hp, k_shift + k, Val(:rg))
                d = _tm5_read_forcing(detu, c, b, Hp, k_shift + k, Val(:rg))
                amu_loc[k] = amu_loc[k + 1] + e - d
                zxi = 0f0
                if amu_loc[k] > 0f0
                    denom = amu_loc[k + 1] + e
                    zxi = max(0f0, 1f0 - d / denom)
                else
                    amu_loc[k] = 0f0
                end
                for kk in (k + 1):LMAX_CONV
                    f_below = k == LMAX_CONV ? 0f0 : A_loc[k + 1, kk]
                    A_loc[k, kk] = f_below * zxi
                end
                bmass_k = bmass_loc[k]
                A_loc[k, k] = e / bmass_k * zxi
            end
            if icllfs <= LMAX_CONV
                @inbounds for k in icllfs:(LMAX_CONV - 1)
                    e = _tm5_read_forcing(entd, c, b, Hp, k_shift + k, Val(:rg))
                    d = _tm5_read_forcing(detd, c, b, Hp, k_shift + k, Val(:rg))
                    amd_loc[k + 1] = amd_loc[k] - e + d
                    zxi = 0f0
                    if amd_loc[k + 1] < 0f0
                        denom = amd_loc[k] - e
                        zxi = max(0f0, 1f0 + d / denom)
                    else
                        amd_loc[k + 1] = 0f0
                    end
                    for kk in icllfs:(k - 1)
                        A_loc[k + 1, kk] = A_loc[k, kk] * zxi
                    end
                    bmass_k = bmass_loc[k]
                    A_loc[k + 1, k] = -e / bmass_k * zxi
                end
            end
            @inbounds for k in LMAX_CONV:-1:2
                bmass_above = bmass_loc[k - 1]
                bmass_k = bmass_loc[k]
                A_loc[k, k - 1] -= amu_loc[k] / bmass_above
                A_loc[k, k]     -= amd_loc[k] / bmass_k
            end
            @inbounds for k in 1:LMAX_CONV
                for kk in 1:LMAX_CONV
                    if k < k_lo || kk < k_lo
                        A_loc[k, kk] = (k == kk) ? 1f0 : 0f0
                        continue
                    end
                    f_below = k == LMAX_CONV ? 0f0 : A_loc[k + 1, kk]
                    fdiff = f_below - A_loc[k, kk]
                    A_loc[k, kk] = -dt * fdiff
                end
                A_loc[k, k] += 1f0
            end
        end
    end
    @synchronize

    @inbounds for idx in t:_TM5_COLLAB_WG_SIZE:(LMAX_CONV * Nt)
        k_  = (idx - 1) % LMAX_CONV + 1
        tt = (idx - 1) ÷ LMAX_CONV + 1
        q_loc[k_, tt] = _tm5_read_q(q_raw_arr, c, b, Hp, k_shift + k_, tt, Val(:rg))
    end
    @synchronize

    if !no_conv
        for k in k_lo:LMAX_CONV
            if t == 1
                piv = k
                pivmag = abs(A_loc[k, k])
                @inbounds for r in (k + 1):LMAX_CONV
                    m_ = abs(A_loc[r, k])
                    if m_ > pivmag
                        piv = r
                        pivmag = m_
                    end
                end
                piv_loc[k] = Int32(piv)
            end
            @synchronize
            piv = Int(piv_loc[k])

            if piv != k
                @inbounds for cc in (k_lo + t - 1):_TM5_COLLAB_WG_SIZE:LMAX_CONV
                    tmp = A_loc[k, cc]
                    A_loc[k, cc] = A_loc[piv, cc]
                    A_loc[piv, cc] = tmp
                end
            end
            @synchronize

            diag_val = A_loc[k, k]
            @inbounds for r in (k + t):_TM5_COLLAB_WG_SIZE:LMAX_CONV
                A_loc[r, k] /= diag_val
            end
            @synchronize

            @inbounds for cc in (k + t):_TM5_COLLAB_WG_SIZE:LMAX_CONV
                akc = A_loc[k, cc]
                for r in (k + 1):LMAX_CONV
                    A_loc[r, cc] -= A_loc[r, k] * akc
                end
            end
            @synchronize
        end

        if t == 1
            @inbounds for k in k_lo:LMAX_CONV
                piv = Int(piv_loc[k])
                if piv != k
                    for tt in 1:Nt
                        tmp = q_loc[k, tt]
                        q_loc[k, tt] = q_loc[piv, tt]
                        q_loc[piv, tt] = tmp
                    end
                end
            end
        end
        @synchronize

        @inbounds for tt in t:_TM5_COLLAB_WG_SIZE:Nt
            for k in k_lo:LMAX_CONV
                s = q_loc[k, tt]
                for j2 in k_lo:(k - 1)
                    s -= A_loc[k, j2] * q_loc[j2, tt]
                end
                q_loc[k, tt] = s
            end
        end
        @synchronize

        @inbounds for tt in t:_TM5_COLLAB_WG_SIZE:Nt
            for k in LMAX_CONV:-1:k_lo
                s = q_loc[k, tt]
                for j2 in (k + 1):LMAX_CONV
                    s -= A_loc[k, j2] * q_loc[j2, tt]
                end
                q_loc[k, tt] = s / A_loc[k, k]
            end
        end
        @synchronize
    end

    @inbounds for idx in t:_TM5_COLLAB_WG_SIZE:(LMAX_CONV * Nt)
        k_  = (idx - 1) % LMAX_CONV + 1
        tt = (idx - 1) ÷ LMAX_CONV + 1
        _tm5_write_q!(q_raw_arr, c, b, Hp, k_shift + k_, tt, q_loc[k_, tt], Val(:rg))
    end
end

# Cubed-sphere panel collaborative kernel.  `q_raw_arr :: (Nc + 2Hp,
# Nc + 2Hp, Nz, Nt)` — the halo offset `Hp` is applied by the
# topology helpers when reading/writing q_raw and air_mass.  Forcings
# (entu/…) are halo-free (Nc, Nc, Nz).  See LL kernel above for why
# the body is inlined verbatim instead of factored into a helper.
@kernel function _tm5_cs_panel_column_collab_kernel!(
    q_raw_arr,                          # (Nc+2Hp, Nc+2Hp, Nz, Nt)
    @Const(air_mass_arr),               # (Nc+2Hp, Nc+2Hp, Nz)
    @Const(entu), @Const(detu),
    @Const(entd), @Const(detd),
    @Const(cell_areas_panel),           # (Nc, Nc)
    Hp::Int, Nc::Int, Nz::Int, Nt::Int, dt::Float32, ::Val{LMAX_CONV},
) where {LMAX_CONV}
    @uniform g = @index(Group)
    t = @index(Local)

    A_loc     = @localmem Float32 (LMAX_CONV, LMAX_CONV)
    q_loc     = @localmem Float32 (LMAX_CONV, _TM5_COLLAB_NT_MAX)
    piv_loc   = @localmem Int32   (LMAX_CONV,)
    bmass_loc = @localmem Float32 (LMAX_CONV,)
    amu_loc   = @localmem Float32 (LMAX_CONV + 1,)
    amd_loc   = @localmem Float32 (LMAX_CONV + 1,)
    icl_top   = @localmem Int32   (1,)
    icl_lfs   = @localmem Int32   (1,)

    # Topology-specific decode (CS).  `Hp` comes from the kernel
    # argument; the topology helpers fold it into the array index.
    @uniform c1 = ((g - 1) % Nc) + 1
    @uniform c2 = ((g - 1) ÷ Nc) + 1
    @uniform area = cell_areas_panel[c1, c2]
    @uniform k_shift = Nz - LMAX_CONV
    # WG_SIZE is _TM5_COLLAB_WG_SIZE (32); used directly below to satisfy KA scoping.

    @inbounds for k in t:_TM5_COLLAB_WG_SIZE:(LMAX_CONV + 1)
        amu_loc[k] = 0f0
        amd_loc[k] = 0f0
    end

    if t == 1
        icl_top[1] = Int32(LMAX_CONV + 1)
        icl_lfs[1] = Int32(LMAX_CONV + 1)
        @inbounds for k in 1:LMAX_CONV
            d = _tm5_read_forcing(detu, c1, c2, Hp, k_shift + k, Val(:cs))
            if d > 0f0 && icl_top[1] == LMAX_CONV + 1
                icl_top[1] = Int32(k)
            end
            e = _tm5_read_forcing(entd, c1, c2, Hp, k_shift + k, Val(:cs))
            if e > 0f0 && icl_lfs[1] == LMAX_CONV + 1
                icl_lfs[1] = Int32(k)
            end
        end
    end
    @synchronize

    icltop = Int(icl_top[1])
    icllfs = Int(icl_lfs[1])
    icltop_eff = min(icllfs, max(icltop, 2) - 1)
    k_lo = max(icltop_eff, 1)
    no_conv = icltop > LMAX_CONV

    @inbounds for k in t:_TM5_COLLAB_WG_SIZE:LMAX_CONV
        bmass_loc[k] = _tm5_read_mass(air_mass_arr, c1, c2, Hp, k_shift + k, Val(:cs)) / area
    end
    @synchronize

    if !no_conv
        @inbounds for idx in t:_TM5_COLLAB_WG_SIZE:(LMAX_CONV * LMAX_CONV)
            k_  = (idx - 1) % LMAX_CONV + 1
            kk_ = (idx - 1) ÷ LMAX_CONV + 1
            if k_ >= k_lo && kk_ >= k_lo
                A_loc[k_, kk_] = 0f0
            elseif k_ == kk_
                A_loc[k_, kk_] = 1f0
            else
                A_loc[k_, kk_] = 0f0
            end
        end
        @synchronize

        if t == 1
            @inbounds for k in LMAX_CONV:-1:icltop
                e = _tm5_read_forcing(entu, c1, c2, Hp, k_shift + k, Val(:cs))
                d = _tm5_read_forcing(detu, c1, c2, Hp, k_shift + k, Val(:cs))
                amu_loc[k] = amu_loc[k + 1] + e - d
                zxi = 0f0
                if amu_loc[k] > 0f0
                    denom = amu_loc[k + 1] + e
                    zxi = max(0f0, 1f0 - d / denom)
                else
                    amu_loc[k] = 0f0
                end
                for kk in (k + 1):LMAX_CONV
                    f_below = k == LMAX_CONV ? 0f0 : A_loc[k + 1, kk]
                    A_loc[k, kk] = f_below * zxi
                end
                bmass_k = bmass_loc[k]
                A_loc[k, k] = e / bmass_k * zxi
            end
            if icllfs <= LMAX_CONV
                @inbounds for k in icllfs:(LMAX_CONV - 1)
                    e = _tm5_read_forcing(entd, c1, c2, Hp, k_shift + k, Val(:cs))
                    d = _tm5_read_forcing(detd, c1, c2, Hp, k_shift + k, Val(:cs))
                    amd_loc[k + 1] = amd_loc[k] - e + d
                    zxi = 0f0
                    if amd_loc[k + 1] < 0f0
                        denom = amd_loc[k] - e
                        zxi = max(0f0, 1f0 + d / denom)
                    else
                        amd_loc[k + 1] = 0f0
                    end
                    for kk in icllfs:(k - 1)
                        A_loc[k + 1, kk] = A_loc[k, kk] * zxi
                    end
                    bmass_k = bmass_loc[k]
                    A_loc[k + 1, k] = -e / bmass_k * zxi
                end
            end
            @inbounds for k in LMAX_CONV:-1:2
                bmass_above = bmass_loc[k - 1]
                bmass_k = bmass_loc[k]
                A_loc[k, k - 1] -= amu_loc[k] / bmass_above
                A_loc[k, k]     -= amd_loc[k] / bmass_k
            end
            @inbounds for k in 1:LMAX_CONV
                for kk in 1:LMAX_CONV
                    if k < k_lo || kk < k_lo
                        A_loc[k, kk] = (k == kk) ? 1f0 : 0f0
                        continue
                    end
                    f_below = k == LMAX_CONV ? 0f0 : A_loc[k + 1, kk]
                    fdiff = f_below - A_loc[k, kk]
                    A_loc[k, kk] = -dt * fdiff
                end
                A_loc[k, k] += 1f0
            end
        end
    end
    @synchronize

    @inbounds for idx in t:_TM5_COLLAB_WG_SIZE:(LMAX_CONV * Nt)
        k_  = (idx - 1) % LMAX_CONV + 1
        tt = (idx - 1) ÷ LMAX_CONV + 1
        q_loc[k_, tt] = _tm5_read_q(q_raw_arr, c1, c2, Hp, k_shift + k_, tt, Val(:cs))
    end
    @synchronize

    if !no_conv
        for k in k_lo:LMAX_CONV
            if t == 1
                piv = k
                pivmag = abs(A_loc[k, k])
                @inbounds for r in (k + 1):LMAX_CONV
                    m_ = abs(A_loc[r, k])
                    if m_ > pivmag
                        piv = r
                        pivmag = m_
                    end
                end
                piv_loc[k] = Int32(piv)
            end
            @synchronize
            piv = Int(piv_loc[k])

            if piv != k
                @inbounds for cc in (k_lo + t - 1):_TM5_COLLAB_WG_SIZE:LMAX_CONV
                    tmp = A_loc[k, cc]
                    A_loc[k, cc] = A_loc[piv, cc]
                    A_loc[piv, cc] = tmp
                end
            end
            @synchronize

            diag_val = A_loc[k, k]
            @inbounds for r in (k + t):_TM5_COLLAB_WG_SIZE:LMAX_CONV
                A_loc[r, k] /= diag_val
            end
            @synchronize

            @inbounds for cc in (k + t):_TM5_COLLAB_WG_SIZE:LMAX_CONV
                akc = A_loc[k, cc]
                for r in (k + 1):LMAX_CONV
                    A_loc[r, cc] -= A_loc[r, k] * akc
                end
            end
            @synchronize
        end

        if t == 1
            @inbounds for k in k_lo:LMAX_CONV
                piv = Int(piv_loc[k])
                if piv != k
                    for tt in 1:Nt
                        tmp = q_loc[k, tt]
                        q_loc[k, tt] = q_loc[piv, tt]
                        q_loc[piv, tt] = tmp
                    end
                end
            end
        end
        @synchronize

        @inbounds for tt in t:_TM5_COLLAB_WG_SIZE:Nt
            for k in k_lo:LMAX_CONV
                s = q_loc[k, tt]
                for j2 in k_lo:(k - 1)
                    s -= A_loc[k, j2] * q_loc[j2, tt]
                end
                q_loc[k, tt] = s
            end
        end
        @synchronize

        @inbounds for tt in t:_TM5_COLLAB_WG_SIZE:Nt
            for k in LMAX_CONV:-1:k_lo
                s = q_loc[k, tt]
                for j2 in (k + 1):LMAX_CONV
                    s -= A_loc[k, j2] * q_loc[j2, tt]
                end
                q_loc[k, tt] = s / A_loc[k, k]
            end
        end
        @synchronize
    end

    @inbounds for idx in t:_TM5_COLLAB_WG_SIZE:(LMAX_CONV * Nt)
        k_  = (idx - 1) % LMAX_CONV + 1
        tt = (idx - 1) ÷ LMAX_CONV + 1
        _tm5_write_q!(q_raw_arr, c1, c2, Hp, k_shift + k_, tt, q_loc[k_, tt], Val(:cs))
    end
end

# ---- Topology-specific array readers/writers -------------------------
@inline _tm5_read_forcing(arr, i, j, Hp, k, ::Val{:ll}) = arr[i, j, k]
@inline _tm5_read_forcing(arr, c, _, Hp, k, ::Val{:rg}) = arr[c, k]
@inline _tm5_read_forcing(arr, c1, c2, Hp, k, ::Val{:cs}) = arr[c1, c2, k]

@inline _tm5_read_mass(arr, i, j, Hp, k, ::Val{:ll}) = arr[i, j, k]
@inline _tm5_read_mass(arr, c, _, Hp, k, ::Val{:rg}) = arr[c, k]
@inline _tm5_read_mass(arr, c1, c2, Hp, k, ::Val{:cs}) = arr[c1 + Hp, c2 + Hp, k]

@inline _tm5_read_q(arr, i, j, Hp, k, tt, ::Val{:ll}) = arr[i, j, k, tt]
@inline _tm5_read_q(arr, c, _, Hp, k, tt, ::Val{:rg}) = arr[c, k, tt]
@inline _tm5_read_q(arr, c1, c2, Hp, k, tt, ::Val{:cs}) = arr[c1 + Hp, c2 + Hp, k, tt]

@inline function _tm5_write_q!(arr, i, j, Hp, k, tt, v, ::Val{:ll})
    arr[i, j, k, tt] = v
    return nothing
end
@inline function _tm5_write_q!(arr, c, _, Hp, k, tt, v, ::Val{:rg})
    arr[c, k, tt] = v
    return nothing
end
@inline function _tm5_write_q!(arr, c1, c2, Hp, k, tt, v, ::Val{:cs})
    arr[c1 + Hp, c2 + Hp, k, tt] = v
    return nothing
end
