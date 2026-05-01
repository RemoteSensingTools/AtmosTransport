# ---------------------------------------------------------------------------
# PBLSurfaceForcing - raw per-window surface fields used to derive PBL Kz.
#
# The transport binary stores the raw meteorological fields, not Kz. Runtime
# diffusion derives a panel-native Kz cache from these fields and the current
# dry air mass whenever the met window advances.
# ---------------------------------------------------------------------------

"""
    PBLSurfaceForcing(pblh, ustar, hflux, t2m)

Container for raw surface fields used by the PBL diffusion closure.

Fields are topology-shaped 2D arrays:

- structured: `(Nx, Ny)`
- face-indexed: `(ncell,)` when such a path is added
- cubed sphere: `NTuple{6, <:AbstractMatrix}` with one `(Nc, Nc)` panel

Units follow the canonical runtime contract:

- `pblh`  - boundary-layer height [m]
- `ustar` - friction velocity [m s^-1]
- `hflux` - upward sensible heat flux [W m^-2]
- `t2m`   - 2 m air temperature [K]
"""
struct PBLSurfaceForcing{P, U, H, T}
    pblh  :: P
    ustar :: U
    hflux :: H
    t2m   :: T
end

PBLSurfaceForcing(; pblh, ustar, hflux, t2m) =
    PBLSurfaceForcing(pblh, ustar, hflux, t2m)

has_pbl_surface_forcing(f::PBLSurfaceForcing) = true
has_pbl_surface_forcing(::Nothing) = false

function Adapt.adapt_structure(_to, f::PBLSurfaceForcing)
    # Surface fields are consumed by host-side Kz refresh logic, not kernels.
    # Keeping them host-resident avoids unnecessary device transfers and still
    # lets window adaptation move the large advection/convection payloads.
    return f
end

export PBLSurfaceForcing, has_pbl_surface_forcing
