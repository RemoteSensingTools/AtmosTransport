"""
    Convection

Convective transport parameterizations with paired discrete adjoints.

# Interface contract

    convect!(tracers, met, grid, conv::AbstractConvection, Δt)
    adjoint_convect!(adj_tracers, met, grid, conv::AbstractConvection, Δt)
"""
module Convection

using DocStringExtensions

using ..Grids: AbstractGrid
using ..Fields: AbstractField

export AbstractConvection, TiedtkeConvection, NoConvection, RASConvection, TM5MatrixConvection
export convect!, adjoint_convect!, invalidate_ras_cfl_cache!
export AbstractTracerSolubility, InertTracer, SolubleTracer
export tracer_solubility, wet_scavenge_fraction

"""
$(TYPEDEF)

Supertype for convective transport parameterizations.

Available concrete types:
- [`NoConvection`](@ref): no-op pass-through
- [`TiedtkeConvection`](@ref): simplified upwind mass-flux (CMFMC only)
- [`RASConvection`](@ref): Relaxed Arakawa-Schubert with entrainment/detrainment (CMFMC + DTRAIN)
- [`TM5MatrixConvection`](@ref): TM5-faithful matrix scheme with implicit LU solve (entu/detu/entd/detd)

# Adding a new scheme

1. Define `struct MyScheme <: AbstractConvection end`
2. Implement `convect!` methods for `CubedSphereGrid` and `LatitudeLongitudeGrid`
3. Add `_needs_convection(::MyScheme) = true` in `run_implementations.jl`
4. If the scheme needs DTRAIN: add `_needs_dtrain(::MyScheme) = true`
5. Add a `type = "myscheme"` branch in `_build_convection` (configuration.jl)
"""
abstract type AbstractConvection end

"""
$(TYPEDEF)

No convection (pass-through). Adjoint is also a no-op.
"""
struct NoConvection <: AbstractConvection end

convect!(tracers, met, grid, ::NoConvection, Δt) = nothing
adjoint_convect!(adj_tracers, met, grid, ::NoConvection, Δt) = nothing

"""
$(TYPEDEF)

Tiedtke (1989) mass-flux convection scheme, as used in TM5.
Mass fluxes come from met data (fixed), so the operator is linear in tracers
and the adjoint is the transpose of the mass-flux redistribution matrix.

The forward operator uses prescribed convective mass fluxes from met data
(`met.conv_mass_flux`) to redistribute tracers vertically via upwind
mass-flux transport. See `tiedtke_convection.jl` for implementation.
"""
struct TiedtkeConvection <: AbstractConvection end

"""
$(TYPEDEF)

Relaxed Arakawa-Schubert (RAS) convection scheme (Moorthi & Suarez 1992),
as implemented in GEOS-Chem (`convection_mod.F90`).

Uses both CMFMC (updraft mass flux at interfaces) and DTRAIN (detraining
mass flux at layer centers) from GEOS met data. Entrainment is diagnosed
from the mass balance: ENTRN = max(0, CMFMC + DTRAIN - CMFMC_below).

If DTRAIN data is unavailable at runtime, falls back to Tiedtke-style
CMFMC-only transport with a warning.

See `ras_convection.jl` for the full algorithm and references.
"""
struct RASConvection <: AbstractConvection end

"""
$(TYPEDEF)

TM5-faithful matrix convection scheme (Heimann & Keeling, Tiedtke 1989).

Builds a full Nz×Nz transfer matrix per column from 4 met fields (updraft/downdraft
entrainment and detrainment), then applies via implicit LU solve. This is the exact
algorithm used in TM5 (tm5_conv.F90).

# Key differences from `TiedtkeConvection`:
- Uses 4 met fields (entu, detu, entd, detd) instead of 1 (CMFMC)
- Builds Nz×Nz transfer matrix (non-local transport)
- Implicit solve (unconditionally stable, no CFL limit)
- Exact mass conservation to machine precision

# Fields
- `lmax_conv::Int`: maximum level for convection (0 = use full Nz)

See `tm5_matrix_convection.jl` for the matrix builder algorithm.
"""
struct TM5MatrixConvection <: AbstractConvection
    lmax_conv::Int
end
TM5MatrixConvection(; lmax_conv::Int=0) = TM5MatrixConvection(lmax_conv)

include("scavenging.jl")
include("tiedtke_convection.jl")
include("tiedtke_convection_adjoint.jl")
include("ras_convection.jl")
include("tm5_matrix_convection.jl")
include("tm5_matrix_convection_adjoint.jl")

end # module Convection
