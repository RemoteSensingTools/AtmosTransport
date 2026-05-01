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
    _tm5_solve_column!(rm_col, m_col,
                        entu_col, detu_col, entd_col, detd_col,
                        conv1_col, pivots_col, cloud_col, dt;
                        f_buf = f_col,
                        amu_buf = amu_col, amd_buf = amd_col)
end

# Face-indexed ReducedGaussian: state.tracers_raw :: (ncells, Nz, Nt).
# Single-axis cell index — no decode needed.
@kernel function _tm5_faceindexed_column_kernel!(
    q_raw, @Const(air_mass),
    @Const(entu), @Const(detu), @Const(entd), @Const(detd),
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
    _tm5_solve_column!(rm_col, m_col,
                        entu_col, detu_col, entd_col, detd_col,
                        conv1_col, pivots_col, cloud_col, dt;
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
    _tm5_solve_column!(rm_col, m_col,
                        entu_col, detu_col, entd_col, detd_col,
                        conv1_col, pivots_col, cloud_col, dt;
                        f_buf = f_col,
                        amu_buf = amu_col, amd_buf = amd_col)
end
