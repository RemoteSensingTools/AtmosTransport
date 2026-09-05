function _review_reference_fv_tp_2d_cs!(rm_panels, m_panels, am_panels, bm_panels,
                       mesh::CubedSphereMesh, ::Val{ORD}, ws, ws_lr::LinRoodWorkspace;
                       damp_coeff=0.0) where ORD
    Nc = mesh.Nc; Hp = mesh.Hp; Nz = size(rm_panels[1], 3)
    N = Nc + 2Hp
    backend = get_backend(rm_panels[1])

    # Optional divergence damping
    damp_coeff > 0 && apply_divergence_damping_cs!(rm_panels, m_panels, mesh, ws, damp_coeff)

    # Pre-instantiate all kernels (avoid repeated compilation inside loops)
    init_k!    = _init_q_buf_kernel!(backend, 256)
    y_face_k!  = _ppm_y_face_kernel!(backend, 256)
    x_face_k!  = _ppm_x_face_kernel!(backend, 256)
    xq_face_k! = _ppm_x_face_from_q_kernel!(backend, 256)
    yq_face_k! = _ppm_y_face_from_q_kernel!(backend, 256)
    pre_y_k!   = _pre_advect_y_kernel!(backend, 256)
    pre_x_k!   = _pre_advect_x_kernel!(backend, 256)
    update_k!  = _linrood_update_kernel!(backend, 256)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1: Edge halos + Y-corners → inner Y-PPM + pre-advect q_i
    # ═══════════════════════════════════════════════════════════════════════
    fill_panel_halos!(rm_panels, mesh)
    fill_panel_halos!(m_panels, mesh)
    copy_corners!(rm_panels, mesh, 2)
    copy_corners!(m_panels, mesh, 2)

    # Initialize q_buf with original mixing ratio (halos persist for outer PPM)
    for p in eachindex(ws_lr.q_buf)
        init_k!(ws_lr.q_buf[p], rm_panels[p], m_panels[p]; ndrange=(N, N, Nz))
    end
    synchronize(backend)

    for p in eachindex(ws_lr.fy_in)
        y_face_k!(ws_lr.fy_in[p], rm_panels[p], m_panels[p], bm_panels[p],
                  Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
        pre_y_k!(ws_lr.q_buf[p], rm_panels[p], m_panels[p], bm_panels[p],
                 ws_lr.fy_in[p], Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: X-corners → outer X-PPM on q_i + inner X-PPM + pre-advect q_j
    # ═══════════════════════════════════════════════════════════════════════
    copy_corners!(ws_lr.q_buf, mesh, 1)
    copy_corners!(rm_panels, mesh, 1)
    copy_corners!(m_panels, mesh, 1)

    for p in eachindex(ws_lr.fx_out)
        xq_face_k!(ws_lr.fx_out[p], ws_lr.q_buf[p], am_panels[p], m_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
        x_face_k!(ws_lr.fx_in[p], rm_panels[p], m_panels[p], am_panels[p],
                  Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
    end
    synchronize(backend)

    # Re-initialize q_buf (halos retain original q; interior overwritten with q_j)
    for p in eachindex(ws_lr.q_buf)
        init_k!(ws_lr.q_buf[p], rm_panels[p], m_panels[p]; ndrange=(N, N, Nz))
    end
    synchronize(backend)

    for p in eachindex(ws_lr.q_buf)
        pre_x_k!(ws_lr.q_buf[p], rm_panels[p], m_panels[p], am_panels[p],
                 ws_lr.fx_in[p], Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 3: Y-corners on q_j → outer Y-PPM → averaged update
    # ═══════════════════════════════════════════════════════════════════════
    copy_corners!(ws_lr.q_buf, mesh, 2)

    for p in eachindex(ws_lr.fx_in)
        yq_face_k!(ws_lr.fy_out[p], ws_lr.q_buf[p], bm_panels[p], m_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
        update_k!(ws.rm_A, ws.m_A,
                  rm_panels[p], m_panels[p], am_panels[p], bm_panels[p],
                  ws_lr.fx_in[p], ws_lr.fx_out[p], ws_lr.fy_in[p], ws_lr.fy_out[p],
                  Hp; ndrange=(Nc, Nc, Nz))
        synchronize(backend)  # required: ws.rm_A/m_A reused across panels
        _copy_interior!(rm_panels[p], ws.rm_A, Nc, Hp, Nz)
        _copy_interior!(m_panels[p], ws.m_A, Nc, Hp, Nz)
    end

    return nothing
end
