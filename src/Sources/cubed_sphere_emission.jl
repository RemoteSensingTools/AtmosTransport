# ---------------------------------------------------------------------------
# apply_surface_flux! for SurfaceFlux{CubedSphereLayout} and
# TimeVaryingSurfaceFlux{CubedSphereLayout}
#
# GPU path uses KernelAbstractions; CPU path uses plain loops.
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend

# ---------------------------------------------------------------------------
# GPU emission kernel for cubed-sphere panels
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
