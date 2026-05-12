# ---------------------------------------------------------------------------
# Adjoint of CS convection (CMFMC + TM5).
#
# Reverse-mode of `CMFMCConvection` and `TM5Convection` column operators.
# Contains:
#   * TM5 forward Thomas/LU solves (`_tm5_solve_vector!`,
#     `_tm5_solve_column_vector!`) used by the reverse-mode forward
#     replay path.
#   * TM5 transpose solves (`_tm5_solve_vector_transpose!`,
#     `_tm5_solve_column_vector_adjoint!`).
#   * Per-panel column kernels for both TM5 and CMFMC (forward + adjoint).
#   * Workspace + forcing validation helpers.
#   * `_apply_cs_convection_forward!` and `_apply_cs_convection_adjoint!`
#     dispatch arms for `NoConvection` / `CMFMCConvection` / `TM5Convection`.
#
# Relocated from `src/Adjoints/Adjoints.jl` lines 1723-2333 unchanged
# in Plan 26 P0.2; no semantic change.
# ---------------------------------------------------------------------------

function _tm5_solve_vector!(rm_col, conv1, pivots, Nz::Integer;
                            icltop_eff::Integer = 1)
    Nz == 0 && return nothing
    k_lo = max(Int(icltop_eff), 1)
    @inbounds begin
        for k in k_lo:Nz
            piv = pivots[k]
            if piv != k
                tmp = rm_col[k]
                rm_col[k] = rm_col[piv]
                rm_col[piv] = tmp
            end
        end
        for k in k_lo:Nz
            s = rm_col[k]
            for j in k_lo:(k - 1)
                s -= conv1[k, j] * rm_col[j]
            end
            rm_col[k] = s
        end
        for k in Nz:-1:k_lo
            s = rm_col[k]
            for j in (k + 1):Nz
                s -= conv1[k, j] * rm_col[j]
            end
            rm_col[k] = s / conv1[k, k]
        end
    end
    return nothing
end

function _tm5_solve_vector_transpose!(lambda_col, conv1, pivots, Nz::Integer;
                                      icltop_eff::Integer = 1)
    Nz == 0 && return nothing
    k_lo = max(Int(icltop_eff), 1)
    @inbounds begin
        # U' z = lambda, where U is stored in the upper triangle.
        for k in k_lo:Nz
            s = lambda_col[k]
            for j in k_lo:(k - 1)
                s -= conv1[j, k] * lambda_col[j]
            end
            lambda_col[k] = s / conv1[k, k]
        end
        # L' y = z, where L is unit diagonal and stored below the diagonal.
        for k in Nz:-1:k_lo
            s = lambda_col[k]
            for j in (k + 1):Nz
                s -= conv1[j, k] * lambda_col[j]
            end
            lambda_col[k] = s
        end
        # Forward solve applies pivots in ascending order; the transpose
        # applies the inverse permutation, so replay swaps in reverse.
        for k in Nz:-1:k_lo
            piv = pivots[k]
            if piv != k
                tmp = lambda_col[k]
                lambda_col[k] = lambda_col[piv]
                lambda_col[piv] = tmp
            end
        end
    end
    return nothing
end

@inline function _tm5_effective_cloud_top(icltop, icllfs)
    return min(Int(icllfs), max(Int(icltop), 2) - 1)
end

function _tm5_solve_column_vector!(rm_col, m_col,
                                   entu_col, detu_col, entd_col, detd_col,
                                   conv1_buf, pivots_buf, cloud_dims, dt;
                                   cell_area = one(eltype(rm_col)),
                                   f_buf = conv1_buf,
                                   amu_buf,
                                   amd_buf)
    FT = eltype(rm_col)
    Nz = length(m_col)
    Nz == 0 && return nothing
    icltop, iclbas, icllfs = _tm5_diagnose_cloud_dims(detu_col, entd_col, Nz)
    cloud_dims[1] = icltop
    cloud_dims[2] = iclbas
    cloud_dims[3] = icllfs
    icltop > Nz && return nothing

    icltop_eff = _tm5_effective_cloud_top(icltop, icllfs)
    _tm5_build_conv1!(conv1_buf,
                      entu_col, detu_col, entd_col, detd_col, m_col,
                      icltop, icllfs, FT(dt), Nz;
                      cell_area = FT(cell_area),
                      f = f_buf, amu = amu_buf, amd = amd_buf)
    _tm5_lu!(conv1_buf, pivots_buf, Nz; icltop_eff = icltop_eff)
    _tm5_solve_vector!(rm_col, conv1_buf, pivots_buf, Nz;
                       icltop_eff = icltop_eff)
    return nothing
end

function _tm5_solve_column_vector_adjoint!(lambda_col, m_col,
                                           entu_col, detu_col, entd_col, detd_col,
                                           conv1_buf, pivots_buf, cloud_dims, dt;
                                           cell_area = one(eltype(lambda_col)),
                                           f_buf = conv1_buf,
                                           amu_buf,
                                           amd_buf)
    FT = eltype(lambda_col)
    Nz = length(m_col)
    Nz == 0 && return nothing
    icltop, iclbas, icllfs = _tm5_diagnose_cloud_dims(detu_col, entd_col, Nz)
    cloud_dims[1] = icltop
    cloud_dims[2] = iclbas
    cloud_dims[3] = icllfs
    icltop > Nz && return nothing

    icltop_eff = _tm5_effective_cloud_top(icltop, icllfs)
    _tm5_build_conv1!(conv1_buf,
                      entu_col, detu_col, entd_col, detd_col, m_col,
                      icltop, icllfs, FT(dt), Nz;
                      cell_area = FT(cell_area),
                      f = f_buf, amu = amu_buf, amd = amd_buf)
    _tm5_lu!(conv1_buf, pivots_buf, Nz; icltop_eff = icltop_eff)
    _tm5_solve_vector_transpose!(lambda_col, conv1_buf, pivots_buf, Nz;
                                 icltop_eff = icltop_eff)
    return nothing
end

@kernel function _tm5_cs_panel_column_single_kernel!(
    q_raw_panel, @Const(air_mass_panel),
    @Const(entu_panel), @Const(detu_panel),
    @Const(entd_panel), @Const(detd_panel),
    @Const(cell_areas_panel),
    conv1_panel, pivots_panel, cloud_panel,
    f_panel, amu_panel, amd_panel,
    Hp::Int, tile_offset::Int, Nc::Int, dt)
    t = @index(Global)
    c_global = tile_offset + t
    c1 = ((c_global - 1) % Nc) + 1
    c2 = ((c_global - 1) ÷ Nc) + 1
    i = c1 + Hp
    j = c2 + Hp
    @inbounds begin
        rm_col = @view q_raw_panel[i, j, :]
        m_col = @view air_mass_panel[i, j, :]
        entu_col = @view entu_panel[c1, c2, :]
        detu_col = @view detu_panel[c1, c2, :]
        entd_col = @view entd_panel[c1, c2, :]
        detd_col = @view detd_panel[c1, c2, :]
        conv1_col = @view conv1_panel[:, :, t]
        pivots_col = @view pivots_panel[:, t]
        cloud_col = @view cloud_panel[:, t]
        f_col = @view f_panel[:, :, t]
        amu_col = @view amu_panel[:, t]
        amd_col = @view amd_panel[:, t]
        _tm5_solve_column_vector!(
            rm_col, m_col, entu_col, detu_col, entd_col, detd_col,
            conv1_col, pivots_col, cloud_col, dt;
            cell_area = cell_areas_panel[c1, c2],
            f_buf = f_col, amu_buf = amu_col, amd_buf = amd_col)
    end
end

@kernel function _tm5_cs_panel_column_adjoint_kernel!(
    lambda_panel, @Const(air_mass_panel),
    @Const(entu_panel), @Const(detu_panel),
    @Const(entd_panel), @Const(detd_panel),
    @Const(cell_areas_panel),
    conv1_panel, pivots_panel, cloud_panel,
    f_panel, amu_panel, amd_panel,
    Hp::Int, tile_offset::Int, Nc::Int, dt)
    t = @index(Global)
    c_global = tile_offset + t
    c1 = ((c_global - 1) % Nc) + 1
    c2 = ((c_global - 1) ÷ Nc) + 1
    i = c1 + Hp
    j = c2 + Hp
    @inbounds begin
        lambda_col = @view lambda_panel[i, j, :]
        m_col = @view air_mass_panel[i, j, :]
        entu_col = @view entu_panel[c1, c2, :]
        detu_col = @view detu_panel[c1, c2, :]
        entd_col = @view entd_panel[c1, c2, :]
        detd_col = @view detd_panel[c1, c2, :]
        conv1_col = @view conv1_panel[:, :, t]
        pivots_col = @view pivots_panel[:, t]
        cloud_col = @view cloud_panel[:, t]
        f_col = @view f_panel[:, :, t]
        amu_col = @view amu_panel[:, t]
        amd_col = @view amd_panel[:, t]
        _tm5_solve_column_vector_adjoint!(
            lambda_col, m_col, entu_col, detu_col, entd_col, detd_col,
            conv1_col, pivots_col, cloud_col, dt;
            cell_area = cell_areas_panel[c1, c2],
            f_buf = f_col, amu_buf = amu_col, amd_buf = amd_col)
    end
end

@inline function _cmfmc_panel_dtrain(cmfmc_panel, dtrain_panel,
                                     i, j, k, ::Val{true})
    return dtrain_panel[i, j, k]
end

@inline function _cmfmc_panel_dtrain(cmfmc_panel, dtrain_panel,
                                     i, j, k, ::Val{false})
    FT = eltype(cmfmc_panel)
    return max(zero(FT), cmfmc_panel[i, j, k + 1] - cmfmc_panel[i, j, k])
end

@inline function _cmfmc_cloud_base(cmfmc_panel, i, j, Nz::Int, tiny)
    cldbase_k = 0
    @inbounds for k in 1:Nz
        cmfmc_bot_k = cmfmc_panel[i, j, k + 1]
        if abs(cmfmc_bot_k) > tiny
            cldbase_k = k
            break
        end
    end
    return cldbase_k
end

@kernel function _cmfmc_cs_panel_column_single_kernel!(
    rm_panel,
    @Const(air_mass_panel),
    @Const(cmfmc_panel),
    @Const(dtrain_panel),
    @Const(cell_areas_panel),
    qc_scratch_panel,
    Nz::Int,
    dt,
    Hp::Int,
    ::Val{has_dtrain}) where has_dtrain
    i, j = @index(Global, NTuple)
    FT = eltype(rm_panel)
    tiny = FT(1e-30)
    ii = i + Hp
    jj = j + Hp
    cell_area = FT(cell_areas_panel[i, j])
    dt_ft = FT(dt)

    @inbounds begin
        cldbase_k = _cmfmc_cloud_base(cmfmc_panel, i, j, Nz, tiny)
        if cldbase_k != 0
            if cldbase_k < Nz
                m_cb = air_mass_panel[ii, jj, cldbase_k]
                q_cldbase = m_cb > tiny ? rm_panel[ii, jj, cldbase_k] / m_cb : zero(FT)
                cmfmc_at_cldbase = cmfmc_panel[i, j, cldbase_k + 1]
                if cmfmc_at_cldbase > tiny
                    qb_num = zero(FT)
                    mb = zero(FT)
                    for k in (cldbase_k + 1):Nz
                        m_k = air_mass_panel[ii, jj, k]
                        q_k = m_k > tiny ? rm_panel[ii, jj, k] / m_k : zero(FT)
                        qb_num += q_k * m_k
                        mb += m_k
                    end
                    if mb > zero(FT)
                        qb = qb_num / mb
                        qc_mixed = (mb * qb + cmfmc_at_cldbase * q_cldbase * dt_ft) /
                                   (mb + cmfmc_at_cldbase * dt_ft)
                        for k in (cldbase_k + 1):Nz
                            rm_panel[ii, jj, k] = qc_mixed * air_mass_panel[ii, jj, k]
                        end
                    end
                end
            end

            qc_below = zero(FT)
            for k in Nz:-1:1
                m_k = air_mass_panel[ii, jj, k]
                q_k = m_k > tiny ? rm_panel[ii, jj, k] / m_k : zero(FT)
                cmfmc_bot = k < Nz ? cmfmc_panel[i, j, k + 1] : zero(FT)
                cmfmc_top = cmfmc_panel[i, j, k]
                dtrain_k = _cmfmc_panel_dtrain(cmfmc_panel, dtrain_panel,
                                               i, j, k, Val(has_dtrain))
                cmout = cmfmc_top + dtrain_k
                cmfmc_bot_eff = min(cmfmc_bot, cmout)
                entrn = cmout - cmfmc_bot_eff
                qc = cmout > tiny ?
                     (cmfmc_bot_eff * qc_below + entrn * q_k) / cmout :
                     q_k
                qc_scratch_panel[ii, jj, k] = qc
                qc_below = qc
            end

            q_env_prev = zero(FT)
            for k in 1:Nz
                m_k = air_mass_panel[ii, jj, k]
                q_k = m_k > tiny ? rm_panel[ii, jj, k] / m_k : zero(FT)
                bmass = m_k / cell_area
                cmfmc_top = cmfmc_panel[i, j, k]
                dtrain_k = _cmfmc_panel_dtrain(cmfmc_panel, dtrain_panel,
                                               i, j, k, Val(has_dtrain))
                qc_post = qc_scratch_panel[ii, jj, k]
                q_new = if k > 1 && bmass > tiny
                    q_k + (dt_ft / bmass) *
                          (cmfmc_top * (q_env_prev - q_k) +
                           dtrain_k * (qc_post - q_k))
                elseif bmass > tiny
                    q_k + (dt_ft / bmass) * dtrain_k * (qc_post - q_k)
                else
                    q_k
                end
                q_env_prev = q_k
                rm_panel[ii, jj, k] = q_new * m_k
            end
        end
    end
end

@kernel function _cmfmc_cs_panel_column_single_adjoint_kernel!(
    lambda_panel,
    @Const(air_mass_panel),
    @Const(cmfmc_panel),
    @Const(dtrain_panel),
    @Const(cell_areas_panel),
    lambda_qc_panel,
    Nz::Int,
    dt,
    Hp::Int,
    ::Val{has_dtrain}) where has_dtrain
    i, j = @index(Global, NTuple)
    FT = eltype(lambda_panel)
    tiny = FT(1e-30)
    ii = i + Hp
    jj = j + Hp
    cell_area = FT(cell_areas_panel[i, j])
    dt_ft = FT(dt)

    @inbounds begin
        for k in 1:Nz
            lambda_qc_panel[ii, jj, k] = zero(FT)
        end

        # Transpose the top-to-bottom environment tendency pass.
        for k in 1:Nz
            m_k = air_mass_panel[ii, jj, k]
            lambda_out = lambda_panel[ii, jj, k]
            lambda_panel[ii, jj, k] = zero(FT)
            if m_k > tiny
                lambda_qnew = lambda_out * m_k
                bmass = m_k / cell_area
                dtrain_k = _cmfmc_panel_dtrain(cmfmc_panel, dtrain_panel,
                                               i, j, k, Val(has_dtrain))
                if bmass > tiny
                    alpha = dt_ft / bmass
                    if k > 1
                        cmfmc_top = cmfmc_panel[i, j, k]
                        lambda_panel[ii, jj, k] +=
                            lambda_qnew *
                            (one(FT) - alpha * (cmfmc_top + dtrain_k)) / m_k
                        m_prev = air_mass_panel[ii, jj, k - 1]
                        if m_prev > tiny
                            lambda_panel[ii, jj, k - 1] +=
                                lambda_qnew * alpha * cmfmc_top / m_prev
                        end
                        lambda_qc_panel[ii, jj, k] =
                            lambda_qnew * alpha * dtrain_k
                    else
                        lambda_panel[ii, jj, k] +=
                            lambda_qnew * (one(FT) - alpha * dtrain_k) / m_k
                        lambda_qc_panel[ii, jj, k] =
                            lambda_qnew * alpha * dtrain_k
                    end
                else
                    lambda_panel[ii, jj, k] += lambda_qnew / m_k
                end
            end
        end

        # Transpose the bottom-to-top updraft recurrence.
        for k in 1:Nz
            lambda_qc = lambda_qc_panel[ii, jj, k]
            cmfmc_bot = k < Nz ? cmfmc_panel[i, j, k + 1] : zero(FT)
            cmfmc_top = cmfmc_panel[i, j, k]
            dtrain_k = _cmfmc_panel_dtrain(cmfmc_panel, dtrain_panel,
                                           i, j, k, Val(has_dtrain))
            cmout = cmfmc_top + dtrain_k
            cmfmc_bot_eff = min(cmfmc_bot, cmout)
            entrn = cmout - cmfmc_bot_eff
            coeff_below = cmout > tiny ? cmfmc_bot_eff / cmout : zero(FT)
            coeff_q = cmout > tiny ? entrn / cmout : one(FT)
            m_k = air_mass_panel[ii, jj, k]
            if m_k > tiny
                lambda_panel[ii, jj, k] += lambda_qc * coeff_q / m_k
            end
            if k < Nz
                lambda_qc_panel[ii, jj, k + 1] += lambda_qc * coeff_below
            end
        end

        # Transpose the optional well-mixed sub-cloud preprocessing.
        cldbase_k = _cmfmc_cloud_base(cmfmc_panel, i, j, Nz, tiny)
        if cldbase_k != 0 && cldbase_k < Nz
            cmfmc_at_cldbase = cmfmc_panel[i, j, cldbase_k + 1]
            if cmfmc_at_cldbase > tiny
                mb = zero(FT)
                lambda_mixed = zero(FT)
                for k in (cldbase_k + 1):Nz
                    m_k = air_mass_panel[ii, jj, k]
                    mb += m_k
                    lambda_mixed += lambda_panel[ii, jj, k] * m_k
                end
                if mb > zero(FT)
                    gamma = cmfmc_at_cldbase * dt_ft
                    denom = mb + gamma
                    coeff_sub = lambda_mixed / denom
                    for k in (cldbase_k + 1):Nz
                        m_k = air_mass_panel[ii, jj, k]
                        lambda_panel[ii, jj, k] = m_k > tiny ? coeff_sub : zero(FT)
                    end
                    m_cb = air_mass_panel[ii, jj, cldbase_k]
                    if m_cb > tiny
                        lambda_panel[ii, jj, cldbase_k] +=
                            lambda_mixed * gamma / denom / m_cb
                    end
                end
            end
        end
    end
end

function _require_cmfmc_convection_workspace(workspace)
    workspace isa CMFMCWorkspace || throw(ArgumentError(
        "CS CMFMC adjoint convection requires a `CMFMCWorkspace`; got $(typeof(workspace))"))
    workspace.cell_metrics === nothing && throw(ArgumentError(
        "CS CMFMC adjoint convection requires `workspace.cell_metrics` with per-panel cell areas"))
    return workspace
end

function _assert_cmfmc_adjoint_forcing(forcing)
    forcing isa ConvectionForcing || throw(ArgumentError(
        "CS CMFMC adjoint convection requires a `ConvectionForcing`; got $(typeof(forcing))"))
    forcing.cmfmc === nothing && throw(ArgumentError(
        "CS CMFMC adjoint convection requires `forcing.cmfmc` panel fields"))
    return forcing.cmfmc, forcing.dtrain
end

function _require_tm5_convection_workspace(workspace)
    workspace isa TM5Workspace || throw(ArgumentError(
        "CS TM5 adjoint convection requires a `TM5Workspace`; got $(typeof(workspace))"))
    workspace.cell_metrics === nothing && throw(ArgumentError(
        "CS TM5 adjoint convection requires `workspace.cell_metrics` with per-panel cell areas"))
    return workspace
end

function _assert_tm5_adjoint_forcing(forcing)
    forcing isa ConvectionForcing || throw(ArgumentError(
        "CS TM5 adjoint convection requires a `ConvectionForcing`; got $(typeof(forcing))"))
    forcing.tm5_fields === nothing && throw(ArgumentError(
        "CS TM5 adjoint convection requires `forcing.tm5_fields` (:entu, :detu, :entd, :detd)"))
    return forcing.tm5_fields
end

_require_cs_convection_workspace(::NoConvection, workspace) = nothing
_require_cs_convection_workspace(::CMFMCConvection, workspace) =
    _require_cmfmc_convection_workspace(workspace)
_require_cs_convection_workspace(::TM5Convection, workspace) =
    _require_tm5_convection_workspace(workspace)
function _require_cs_convection_workspace(op, workspace)
    throw(ArgumentError("CS adjoint footprint supports `NoConvection`, " *
                        "`CMFMCConvection`, and `TM5Convection`; got $(typeof(op))"))
end

function _apply_cs_convection_forward!(panels_rm, panels_m, forcing,
                                       ::NoConvection, dt, workspace,
                                       mesh::CubedSphereMesh)
    return nothing
end

function _apply_cs_convection_forward!(panels_rm, panels_m, forcing,
                                       ::CMFMCConvection, dt,
                                       workspace::CMFMCWorkspace,
                                       mesh::CubedSphereMesh)
    cmfmc, dtrain = _assert_cmfmc_adjoint_forcing(forcing)
    _require_cmfmc_convection_workspace(workspace)
    cell_areas = workspace.cell_metrics
    invalidate_cmfmc_cache!(workspace)
    n_sub = _get_or_compute_n_sub!(workspace, cmfmc, panels_m, cell_areas, dt)
    has_dtrain = dtrain !== nothing
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(panels_rm[1], 3)
    backend = get_backend(panels_rm[1])
    kernel! = _cmfmc_cs_panel_column_single_kernel!(backend, (16, 16))
    FT = eltype(panels_rm[1])
    sdt = FT(dt) / FT(n_sub)
    @inbounds for _ in 1:n_sub
        for p in 1:6
            dtrain_panel = has_dtrain ? dtrain[p] : cmfmc[p]
            kernel!(panels_rm[p], panels_m[p], cmfmc[p], dtrain_panel,
                    cell_areas[p], workspace.qc_scratch[p],
                    Nz, sdt, Hp, Val(has_dtrain);
                    ndrange = (Nc, Nc))
        end
    end
    synchronize(backend)
    return nothing
end

function _apply_cs_convection_forward!(panels_rm, panels_m, forcing,
                                       ::TM5Convection, dt,
                                       workspace::TM5Workspace,
                                       mesh::CubedSphereMesh)
    tm5 = _assert_tm5_adjoint_forcing(forcing)
    _require_tm5_convection_workspace(workspace)
    cell_areas = workspace.cell_metrics
    Nc = mesh.Nc
    Hp = mesh.Hp
    N_total = Nc * Nc
    B = size(workspace.conv1, 3)
    backend = get_backend(panels_rm[1])
    kernel! = _tm5_cs_panel_column_single_kernel!(backend)
    FT = eltype(panels_rm[1])
    @inbounds for p in 1:6
        for tile_off in 0:B:(N_total - 1)
            n = min(B, N_total - tile_off)
            kernel!(panels_rm[p], panels_m[p],
                    tm5.entu[p], tm5.detu[p], tm5.entd[p], tm5.detd[p],
                    cell_areas[p],
                    workspace.conv1, workspace.pivots, workspace.cloud_dims,
                    workspace.f_scratch,
                    workspace.amu_scratch, workspace.amd_scratch,
                    Hp, Int(tile_off), Nc, FT(dt);
                    ndrange = n)
        end
    end
    synchronize(backend)
    return nothing
end

function _apply_cs_convection_forward!(panels_rm, panels_m, forcing,
                                       op, dt, workspace,
                                       mesh::CubedSphereMesh)
    throw(ArgumentError("CS adjoint footprint forward helper supports `NoConvection` " *
                        "`CMFMCConvection`, and `TM5Convection`; got $(typeof(op))"))
end

function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       ::NoConvection, dt, workspace,
                                       mesh::CubedSphereMesh)
    return nothing
end

function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       ::CMFMCConvection, dt,
                                       workspace::CMFMCWorkspace,
                                       mesh::CubedSphereMesh)
    cmfmc, dtrain = _assert_cmfmc_adjoint_forcing(forcing)
    _require_cmfmc_convection_workspace(workspace)
    cell_areas = workspace.cell_metrics
    invalidate_cmfmc_cache!(workspace)
    n_sub = _get_or_compute_n_sub!(workspace, cmfmc, panels_m, cell_areas, dt)
    has_dtrain = dtrain !== nothing
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(lambda_panels[1], 3)
    backend = get_backend(lambda_panels[1])
    kernel! = _cmfmc_cs_panel_column_single_adjoint_kernel!(backend, (16, 16))
    FT = eltype(lambda_panels[1])
    sdt = FT(dt) / FT(n_sub)
    @inbounds for _ in 1:n_sub
        for p in 1:6
            dtrain_panel = has_dtrain ? dtrain[p] : cmfmc[p]
            kernel!(lambda_panels[p], panels_m[p], cmfmc[p], dtrain_panel,
                    cell_areas[p], workspace.qc_scratch[p],
                    Nz, sdt, Hp, Val(has_dtrain);
                    ndrange = (Nc, Nc))
        end
    end
    synchronize(backend)
    return nothing
end

function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       ::TM5Convection, dt,
                                       workspace::TM5Workspace,
                                       mesh::CubedSphereMesh)
    tm5 = _assert_tm5_adjoint_forcing(forcing)
    _require_tm5_convection_workspace(workspace)
    cell_areas = workspace.cell_metrics
    Nc = mesh.Nc
    Hp = mesh.Hp
    N_total = Nc * Nc
    B = size(workspace.conv1, 3)
    backend = get_backend(lambda_panels[1])
    kernel! = _tm5_cs_panel_column_adjoint_kernel!(backend)
    FT = eltype(lambda_panels[1])
    @inbounds for p in 1:6
        for tile_off in 0:B:(N_total - 1)
            n = min(B, N_total - tile_off)
            kernel!(lambda_panels[p], panels_m[p],
                    tm5.entu[p], tm5.detu[p], tm5.entd[p], tm5.detd[p],
                    cell_areas[p],
                    workspace.conv1, workspace.pivots, workspace.cloud_dims,
                    workspace.f_scratch,
                    workspace.amu_scratch, workspace.amd_scratch,
                    Hp, Int(tile_off), Nc, FT(dt);
                    ndrange = n)
        end
    end
    synchronize(backend)
    return nothing
end

function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       op, dt, workspace,
                                       mesh::CubedSphereMesh)
    throw(ArgumentError("CS adjoint footprint reverse helper supports `NoConvection` " *
                        "`CMFMCConvection`, and `TM5Convection`; got $(typeof(op))"))
end
