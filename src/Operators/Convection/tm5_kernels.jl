# ---------------------------------------------------------------------------
# TM5 convection kernels — plan 23 Commit 4.
#
# Thin KernelAbstractions wrappers around `_tm5_solve_column!`
# (tm5_column_solve.jl).  One kernel per topology:
#
#   _tm5_column_kernel!             — LatLon 4D `(Nx, Ny, Nz, Nt)` state,
#                                     ndrange `(Nx, Ny)`.
#   _tm5_faceindexed_column_kernel! — ReducedGaussian 3D
#                                     `(ncells, Nz, Nt)` state,
#                                     ndrange `ncells`.
#   _tm5_cs_panel_column_kernel!    — CubedSphere per-panel 4D
#                                     `(Nc+2Hp, Nc+2Hp, Nz, Nt)` state,
#                                     ndrange `(Nc, Nc)` × 6 panels.
#
# Each kernel slices the pre-allocated `TM5Workspace` buffers per
# thread and dispatches to `_tm5_solve_column!`.  No allocation
# inside the kernel — mandatory for GPU correctness (plan 23
# principle 4).
#
# CLAUDE.md gotcha: the kernel uses explicit column indices
# (not `view`s stored in local variables) because KA kernels on
# CUDA.jl cannot survive SubArray construction inside their
# bodies.  Each thread reads from `state.tracers_raw`, `air_mass`,
# forcing fields via direct indexing, passes per-column views of
# the workspace slabs into `_tm5_solve_column!`, which iterates
# columns in-place.
# ---------------------------------------------------------------------------

# LatLon (structured 4D): state.tracers_raw :: (Nx, Ny, Nz, Nt),
# air_mass :: (Nx, Ny, Nz), forcing fields :: (Nx, Ny, Nz),
# conv1 workspace :: (Nz, Nz, Nx, Ny), pivots :: (Nz, Nx, Ny),
# cloud_dims :: (3, Nx, Ny), f/fu :: (Nz+1, Nz, Nx, Ny),
# amu/amd :: (Nz+1, Nx, Ny).
@kernel function _tm5_column_kernel!(
    q_raw, @Const(air_mass),
    @Const(entu), @Const(detu), @Const(entd), @Const(detd),
    conv1_ws, pivots_ws, cloud_dims_ws,
    f_ws, fu_ws, amu_ws, amd_ws,
    dt,
)
    i, j = @index(Global, NTuple)
    rm_col    = @view q_raw[i, j, :, :]
    m_col     = @view air_mass[i, j, :]
    entu_col  = @view entu[i, j, :]
    detu_col  = @view detu[i, j, :]
    entd_col  = @view entd[i, j, :]
    detd_col  = @view detd[i, j, :]
    conv1_col = @view conv1_ws[:, :, i, j]
    pivots_col = @view pivots_ws[:, i, j]
    cloud_col = @view cloud_dims_ws[:, i, j]
    f_col     = @view f_ws[:, :, i, j]
    fu_col    = @view fu_ws[:, :, i, j]
    amu_col   = @view amu_ws[:, i, j]
    amd_col   = @view amd_ws[:, i, j]
    _tm5_solve_column!(rm_col, m_col,
                        entu_col, detu_col, entd_col, detd_col,
                        conv1_col, pivots_col, cloud_col, dt;
                        fu_buf = fu_col, f_buf = f_col,
                        amu_buf = amu_col, amd_buf = amd_col)
end

# Face-indexed ReducedGaussian: state.tracers_raw :: (ncells, Nz, Nt).
@kernel function _tm5_faceindexed_column_kernel!(
    q_raw, @Const(air_mass),
    @Const(entu), @Const(detu), @Const(entd), @Const(detd),
    conv1_ws, pivots_ws, cloud_dims_ws,
    f_ws, fu_ws, amu_ws, amd_ws,
    dt,
)
    c = @index(Global)
    rm_col    = @view q_raw[c, :, :]
    m_col     = @view air_mass[c, :]
    entu_col  = @view entu[c, :]
    detu_col  = @view detu[c, :]
    entd_col  = @view entd[c, :]
    detd_col  = @view detd[c, :]
    conv1_col = @view conv1_ws[:, :, c]
    pivots_col = @view pivots_ws[:, c]
    cloud_col = @view cloud_dims_ws[:, c]
    f_col     = @view f_ws[:, :, c]
    fu_col    = @view fu_ws[:, :, c]
    amu_col   = @view amu_ws[:, c]
    amd_col   = @view amd_ws[:, c]
    _tm5_solve_column!(rm_col, m_col,
                        entu_col, detu_col, entd_col, detd_col,
                        conv1_col, pivots_col, cloud_col, dt;
                        fu_buf = fu_col, f_buf = f_col,
                        amu_buf = amu_col, amd_buf = amd_col)
end

# CubedSphere panel: q_raw_panel :: (Nc+2Hp, Nc+2Hp, Nz, Nt),
# air_mass_panel :: (Nc+2Hp, Nc+2Hp, Nz), forcing fields
# :: (Nc, Nc, Nz) (halo-free per panel). Conv1 / scratch
# per-panel shapes mirror forcings: (Nz, Nz, Nc, Nc), etc.
# The kernel runs on ndrange `(Nc, Nc)` — halos are untouched
# because convection is column-local.  The caller is responsible
# for converting panel-halo-space indices to halo-free indices
# for the forcing payloads.
@kernel function _tm5_cs_panel_column_kernel!(
    q_raw_panel, @Const(air_mass_panel),
    @Const(entu_panel), @Const(detu_panel),
    @Const(entd_panel), @Const(detd_panel),
    conv1_panel, pivots_panel, cloud_panel,
    f_panel, fu_panel, amu_panel, amd_panel,
    Hp::Int, dt,
)
    c1, c2 = @index(Global, NTuple)
    # Halo-offset indices into the padded state arrays.
    i = c1 + Hp
    j = c2 + Hp
    rm_col    = @view q_raw_panel[i, j, :, :]
    m_col     = @view air_mass_panel[i, j, :]
    entu_col  = @view entu_panel[c1, c2, :]
    detu_col  = @view detu_panel[c1, c2, :]
    entd_col  = @view entd_panel[c1, c2, :]
    detd_col  = @view detd_panel[c1, c2, :]
    conv1_col = @view conv1_panel[:, :, c1, c2]
    pivots_col = @view pivots_panel[:, c1, c2]
    cloud_col = @view cloud_panel[:, c1, c2]
    f_col     = @view f_panel[:, :, c1, c2]
    fu_col    = @view fu_panel[:, :, c1, c2]
    amu_col   = @view amu_panel[:, c1, c2]
    amd_col   = @view amd_panel[:, c1, c2]
    _tm5_solve_column!(rm_col, m_col,
                        entu_col, detu_col, entd_col, detd_col,
                        conv1_col, pivots_col, cloud_col, dt;
                        fu_buf = fu_col, f_buf = f_col,
                        amu_buf = amu_col, amd_buf = amd_col)
end
