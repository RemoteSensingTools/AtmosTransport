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

# ---------------------------------------------------------------------------
# GPU emission kernel — PBL-distributed (lat-lon grids)
#
# Distributes surface flux across PBL levels proportionally to air mass (DELP).
# For VMR tracers, this gives uniform Δc = flux * dt * mol_ratio * g / pbl_delp
# across all full PBL levels (area cancels in the VMR calculation).
#
# Uses same isothermal hydrostatic approximation as cubed_sphere_emission.jl:
#   dz_k = DELP_k * R_d * T_ref / (p_mid_k * g)  with T_ref = 280 K
# ---------------------------------------------------------------------------

@kernel function _emit_surface_pbl_kernel!(c, @Const(flux_2d), @Const(delp),
                                            @Const(pblh_2d), g, dt_window,
                                            mol_ratio, Nz, R_T_over_g)
    i, j = @index(Global, NTuple)
    FT = eltype(c)
    @inbounds begin
        f = flux_2d[i, j]
        if f != zero(FT)
            h_pbl = max(FT(pblh_2d[i, j]), FT(100))

            # Compute surface pressure from DELP column
            ps = zero(FT)
            for k in 1:Nz
                ps += delp[i, j, k]
            end

            # Pass 1: walk surface→TOA, find total DELP within PBL
            pbl_delp = zero(FT)
            z_accum = zero(FT)
            p_bot = ps
            done = false
            for k_off in 0:Nz-1
                if !done
                    k = Nz - k_off    # surface (Nz) → TOA (1)
                    delp_k = delp[i, j, k]
                    p_top_k = p_bot - delp_k
                    p_mid = max((p_bot + p_top_k) / FT(2), FT(1))
                    dz = delp_k * FT(R_T_over_g) / p_mid

                    if z_accum + dz > h_pbl
                        frac = (h_pbl - z_accum) / dz
                        pbl_delp += delp_k * frac
                        done = true
                    else
                        pbl_delp += delp_k
                        z_accum += dz
                        p_bot = p_top_k
                    end
                end
            end

            # Fallback: zero PBL DELP → inject into bottom cell only
            if pbl_delp <= zero(FT)
                delp_sfc = delp[i, j, Nz]
                c[i, j, Nz] += f * dt_window * mol_ratio * g / delp_sfc
            else
                # VMR increment: Δc = flux * dt * mol_ratio * g / pbl_delp
                Δc_full = f * dt_window * mol_ratio * g / pbl_delp

                # Pass 2: distribute across PBL levels
                z_accum = zero(FT)
                p_bot = ps
                done = false
                for k_off in 0:Nz-1
                    if !done
                        k = Nz - k_off
                        delp_k = delp[i, j, k]
                        p_top_k = p_bot - delp_k
                        p_mid = max((p_bot + p_top_k) / FT(2), FT(1))
                        dz = delp_k * FT(R_T_over_g) / p_mid

                        if z_accum + dz > h_pbl
                            frac = (h_pbl - z_accum) / dz
                            c[i, j, k] += Δc_full * frac
                            done = true
                        else
                            c[i, j, k] += Δc_full
                            z_accum += dz
                            p_bot = p_top_k
                        end
                    end
                end
            end
        end
    end
end

"""
    apply_emissions_window_pbl!(c, flux_dev, delp_dev, pblh_dev,
                                 g, dt_window; molar_mass=M_CO2)

GPU-accelerated PBL-distributed surface emission injection for lat-lon grids.
Distributes surface flux across boundary layer levels proportionally to air mass.
Falls back to bottom-cell injection if PBL height is zero or missing.
"""
function apply_emissions_window_pbl!(c, flux_dev, delp_dev, pblh_dev,
                                      g, dt_window; molar_mass=M_CO2)
    FT_e = eltype(c)
    mol = FT_e(M_AIR / molar_mass)
    Nz = size(c, 3)
    backend = get_backend(c)
    Nx, Ny = size(c, 1), size(c, 2)
    k! = _emit_surface_pbl_kernel!(backend, 256)
    k!(c, flux_dev, delp_dev, pblh_dev, FT_e(g), FT_e(dt_window), mol, Nz,
       FT_e(_EMIT_R_T_OVER_G); ndrange=(Nx, Ny))
    synchronize(backend)
end
