# ---------------------------------------------------------------------------
# apply_surface_flux! for SurfaceFlux{CubedSphereLayout} and
# TimeVaryingSurfaceFlux{CubedSphereLayout}
#
# GPU path uses KernelAbstractions; CPU path uses plain loops.
#
# Two injection modes:
#   1. Bottom-cell only (apply_surface_flux!) — original, used when PBLH unavailable
#   2. PBL-distributed  (apply_surface_flux_pbl!) — spreads emission across boundary
#      layer levels proportionally to air mass (DELP). Gives uniform Δq across PBL,
#      matching GEOS-Chem's effective treatment where rapid PBL mixing distributes
#      surface emissions throughout the boundary layer.
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend

# ---------------------------------------------------------------------------
# GPU emission kernel — bottom cell only (original)
# ---------------------------------------------------------------------------

@kernel function _emit_cs_kernel!(rm, @Const(flux), @Const(area), dt_window, mol_ratio, Hp)
    i, j = @index(Global, NTuple)
    @inbounds begin
        f = flux[i, j]
        if f != zero(eltype(rm))
            rm[Hp + i, Hp + j, size(rm, 3)] += f * dt_window * area[i, j] * mol_ratio
        end
    end
end

# ---------------------------------------------------------------------------
# GPU emission kernel — PBL-distributed
#
# Uses isothermal hydrostatic approximation to convert PBLH (meters) to
# pressure levels:  dz_k = DELP_k * R_d * T_ref / (p_mid_k * g)
# with T_ref = 280 K (adequate for lowest ~3 km).
#
# Two passes over the vertical:
#   Pass 1: walk surface→TOA, accumulate height, find total DELP within PBL
#   Pass 2: distribute emission proportionally to DELP in PBL levels
# Both passes break early at PBL top (typically 3-10 levels).
# ---------------------------------------------------------------------------

@kernel function _emit_cs_pbl_kernel!(rm, @Const(flux), @Const(area), @Const(delp),
                                       @Const(pblh), dt_window, mol_ratio, Hp, Nz,
                                       R_T_over_g)
    i, j = @index(Global, NTuple)
    FT = eltype(rm)
    @inbounds begin
        f = flux[i, j]
        if f != zero(FT)
            ii = Hp + i
            jj = Hp + j

            # PBL height (meters), clamped to minimum 100 m
            h_pbl = max(FT(pblh[ii, jj]), FT(100))

            # Compute surface pressure from DELP column
            ps = zero(FT)
            for k in 1:Nz
                ps += delp[ii, jj, k]
            end

            # Pass 1: walk surface→TOA, find total DELP within PBL
            pbl_delp = zero(FT)
            z_accum = zero(FT)
            p_bot = ps
            done = false
            for k_off in 0:Nz-1
                if !done
                    k = Nz - k_off    # surface (Nz) → TOA (1)
                    delp_k = delp[ii, jj, k]
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

            # Total emission in rm units
            total_rm = f * dt_window * area[i, j] * mol_ratio

            # Fallback: if pbl_delp is zero, inject into bottom cell only
            if pbl_delp <= zero(FT)
                rm[ii, jj, Nz] += total_rm
            else
                # Pass 2: distribute proportionally to DELP within PBL
                z_accum = zero(FT)
                p_bot = ps
                done = false
                for k_off in 0:Nz-1
                    if !done
                        k = Nz - k_off
                        delp_k = delp[ii, jj, k]
                        p_top_k = p_bot - delp_k
                        p_mid = max((p_bot + p_top_k) / FT(2), FT(1))
                        dz = delp_k * FT(R_T_over_g) / p_mid

                        if z_accum + dz > h_pbl
                            frac = (h_pbl - z_accum) / dz
                            rm[ii, jj, k] += total_rm * (delp_k * frac) / pbl_delp
                            done = true
                        else
                            rm[ii, jj, k] += total_rm * delp_k / pbl_delp
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
    apply_surface_flux!(rm_panels, source::SurfaceFlux{CubedSphereLayout}, area_panels, dt, Nc, Hp)

Inject cubed-sphere surface fluxes into haloed tracer panels.
`rm_panels` is NTuple{6} of haloed 3D arrays (mixing-ratio × air-mass).
`area_panels` is NTuple{6} of (Nc × Nc) cell area arrays.

Works on both CPU and GPU via KernelAbstractions dispatch.
"""
function apply_surface_flux!(rm_panels::NTuple{6}, source::SurfaceFlux{CubedSphereLayout, FT},
                              area_panels::NTuple{6},
                              dt, Nc::Int, Hp::Int) where FT
    mol_ratio = FT(M_AIR / source.molar_mass)
    backend = get_backend(rm_panels[1])
    k! = _emit_cs_kernel!(backend, 256)
    for p in 1:6
        k!(rm_panels[p], source.flux[p], area_panels[p],
           FT(dt), mol_ratio, Hp; ndrange=(Nc, Nc))
    end
    synchronize(backend)
end

function apply_surface_flux!(rm_panels::NTuple{6}, source::TimeVaryingSurfaceFlux{CubedSphereLayout, FT},
                              area_panels::NTuple{6},
                              dt, Nc::Int, Hp::Int) where FT
    panels = flux_data(source)
    mol_ratio = FT(M_AIR / source.molar_mass)
    backend = get_backend(rm_panels[1])
    k! = _emit_cs_kernel!(backend, 256)
    for p in 1:6
        k!(rm_panels[p], panels[p], area_panels[p],
           FT(dt), mol_ratio, Hp; ndrange=(Nc, Nc))
    end
    synchronize(backend)
end

# ---------------------------------------------------------------------------
# PBL-distributed emission dispatch
# ---------------------------------------------------------------------------

# R_d * T_ref / g  with  R_d = 1004/3.5 = 286.857,  T_ref = 280 K,  g = 9.81
const _EMIT_R_T_OVER_G = (1004.0 / 3.5) * 280.0 / 9.81

"""
    apply_surface_flux_pbl!(rm_panels, flux_panels, area_panels, delp_panels, pblh_panels,
                             dt, mol_ratio, Nc, Hp)

Distribute surface flux across PBL levels proportionally to air mass (DELP).
Falls back to bottom-cell injection if PBL height is zero or missing.
"""
function apply_surface_flux_pbl!(rm_panels::NTuple{6}, flux_panels::NTuple{6},
                                  area_panels::NTuple{6}, delp_panels::NTuple{6},
                                  pblh_panels::NTuple{6},
                                  dt::FT, mol_ratio::FT, Nc::Int, Hp::Int) where FT
    Nz = size(rm_panels[1], 3)
    backend = get_backend(rm_panels[1])
    k! = _emit_cs_pbl_kernel!(backend, 256)
    for p in 1:6
        k!(rm_panels[p], flux_panels[p], area_panels[p], delp_panels[p],
           pblh_panels[p], dt, mol_ratio, Hp, Nz, FT(_EMIT_R_T_OVER_G);
           ndrange=(Nc, Nc))
    end
    synchronize(backend)
end
