# ---------------------------------------------------------------------------
# GPU emission kernels for lat-lon grids
#
# Pressure-dependent surface-layer emission injection using
# KernelAbstractions for CPU/GPU portability.
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend

@kernel function _emit_surface_kernel!(c, @Const(flux_2d), @Const(area_j),
                                       @Const(ps_2d), ΔA_sfc, ΔB_sfc,
                                       g, dt_window, mol_ratio, Nz)
    i, j = @index(Global, NTuple)
    FT = eltype(c)
    @inbounds begin
        f = flux_2d[i, j]
        if f != zero(FT)
            Δp = ΔA_sfc + ΔB_sfc * ps_2d[i, j]
            m_air = Δp * area_j[j] / g
            ΔM = f * dt_window * area_j[j]
            c[i, j, Nz] += ΔM * mol_ratio / m_air
        end
    end
end

"""
    apply_emissions_window!(c, flux_dev, area_j_dev, ps_dev, A_coeff, B_coeff,
                             Nz, g, dt_window; molar_mass=M_CO2)

GPU-accelerated surface emission injection for lat-lon grids.
Uses pressure-dependent air mass at the surface layer.

This is the GPU-optimized path; for CPU-only use, see
`apply_surface_flux!(tracers, source::SurfaceFlux{LatLonLayout}, grid, dt)`.
"""
function apply_emissions_window!(c, flux_dev, area_j_dev, ps_dev,
                                  A_coeff, B_coeff, Nz, g, dt_window;
                                  molar_mass=M_CO2)
    FT_e = eltype(c)
    ΔA = FT_e(A_coeff[Nz + 1] - A_coeff[Nz])
    ΔB = FT_e(B_coeff[Nz + 1] - B_coeff[Nz])
    mol = FT_e(M_AIR / molar_mass)
    backend = get_backend(c)
    Nx, Ny = size(c, 1), size(c, 2)
    k! = _emit_surface_kernel!(backend, 256)
    k!(c, flux_dev, area_j_dev, ps_dev, ΔA, ΔB, FT_e(g), FT_e(dt_window), mol, Nz;
       ndrange=(Nx, Ny))
    synchronize(backend)
end
