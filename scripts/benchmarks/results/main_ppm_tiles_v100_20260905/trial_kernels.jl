module SweepTrial
using KernelAbstractions: @kernel, @index, @Const
using AtmosTransport.Operators.Advection: TracerView, _xface_tracer_flux, _yface_tracer_flux, _zface_tracer_flux

@kernel function parallel_x!(rm_new_4d, @Const(rm_4d),
                                    m_new, @Const(m),
                                    @Const(am), scheme, Nc, Hp, Nt, flux_scale)
    ii, jj, k, tt = @index(Global, NTuple)
    t = Int32(tt)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        am_l = flux_scale * am[i, j, k]
        am_r = flux_scale * am[i + 1, j, k]
        Nx_padded = Int32(Nc + 2 * Hp)
        t == Int32(1) && (m_new[i, j, k] = m[i, j, k] + am_l - am_r)
        rm_t = TracerView(rm_4d, t)
        flux_L = _xface_tracer_flux(Int32(i), j, k, rm_t, m, am_l, scheme, Nx_padded)
        flux_R = _xface_tracer_flux(Int32(i) + Int32(1), j, k, rm_t, m, am_r, scheme, Nx_padded)
        rm_new_4d[i, j, k, t] = rm_4d[i, j, k, t] + flux_L - flux_R
    end
end

@kernel function parallel_y!(rm_new_4d, @Const(rm_4d),
                                    m_new, @Const(m),
                                    @Const(bm), scheme, Nc, Hp, Nt, flux_scale)
    ii, jj, k, tt = @index(Global, NTuple)
    t = Int32(tt)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bm_s = flux_scale * bm[i, j, k]
        bm_n = flux_scale * bm[i, j + 1, k]
        Ny_padded = Int32(Nc + 2 * Hp)
        t == Int32(1) && (m_new[i, j, k] = m[i, j, k] + bm_s - bm_n)
        rm_t = TracerView(rm_4d, t)
        flux_S = _yface_tracer_flux(i, Int32(j), k, rm_t, m, bm_s, scheme, Ny_padded)
        flux_N = _yface_tracer_flux(i, Int32(j) + Int32(1), k, rm_t, m, bm_n, scheme, Ny_padded)
        rm_new_4d[i, j, k, t] = rm_4d[i, j, k, t] + flux_S - flux_N
    end
end

@kernel function parallel_z!(rm_new_4d, @Const(rm_4d),
                                    m_new, @Const(m),
                                    @Const(cm), scheme, Nz, Hp, Nt, flux_scale)
    ii, jj, k, tt = @index(Global, NTuple)
    t = Int32(tt)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        cm_t = flux_scale * cm[i, j, k]
        cm_b = flux_scale * cm[i, j, k + 1]
        t == Int32(1) && (m_new[i, j, k] = m[i, j, k] + cm_t - cm_b)
        rm_t = TracerView(rm_4d, t)
        flux_T = _zface_tracer_flux(i, j, Int32(k), rm_t, m, cm_t, scheme, Int32(Nz))
        flux_B = _zface_tracer_flux(i, j, Int32(k) + Int32(1), rm_t, m, cm_b, scheme, Int32(Nz))
        rm_new_4d[i, j, k, t] = rm_4d[i, j, k, t] + flux_T - flux_B
    end
end

end
