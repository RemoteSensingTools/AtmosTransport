# ---------------------------------------------------------------------------
# CubedSphereEmission — panel-based surface emission for cubed-sphere grids
#
# Dispatches apply_surface_flux! on CubedSphereGrid via multiple dispatch.
# GPU path uses KernelAbstractions; CPU path uses plain loops.
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend

"""
$(TYPEDEF)

Surface emission regridded to cubed-sphere panels.
Each panel has an (Nc × Nc) flux field in kg/m²/s.

$(FIELDS)
"""
struct CubedSphereEmission{FT, A <: AbstractMatrix{FT}} <: AbstractGriddedEmission{FT}
    "emission flux panels [kg/m²/s], NTuple{6, Nc×Nc}"
    flux_panels :: NTuple{6, A}
    "tracer name (e.g. :co2)"
    species     :: Symbol
    "human-readable label"
    label       :: String
end

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
    apply_surface_flux!(rm_panels, source::CubedSphereEmission, area_panels, dt, Nc, Hp)

Inject cubed-sphere surface emissions into haloed tracer panels.
`rm_panels` is NTuple{6} of haloed 3D arrays (mixing-ratio × air-mass).
`area_panels` is NTuple{6} of (Nc × Nc) cell area arrays.

Works on both CPU and GPU via KernelAbstractions dispatch.
"""
function apply_surface_flux!(rm_panels::NTuple{6}, source::CubedSphereEmission{FT},
                              area_panels::NTuple{6},
                              dt, Nc::Int, Hp::Int) where FT
    mol_ratio = FT(1e6 * M_AIR / M_CO2)
    backend = get_backend(rm_panels[1])
    k! = _emit_cs_kernel!(backend, 256)
    for p in 1:6
        k!(rm_panels[p], source.flux_panels[p], area_panels[p],
           FT(dt), mol_ratio, Hp; ndrange=(Nc, Nc))
    end
    synchronize(backend)
end
