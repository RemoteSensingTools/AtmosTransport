# ---------------------------------------------------------------------------
# Wet scavenging traits for convective transport
#
# Trait system for species-dependent wet removal during convection.
# Inert tracers (CO₂, SF₆, ²²²Rn) experience no scavenging — all
# updraft air is conserved. Soluble tracers (future: HNO₃, aerosols)
# will use Henry's law dissolution + retention efficiency.
#
# The trait is queried inside the convection kernel to determine what
# fraction of the updraft tracer mass is removed by precipitation.
#
# References:
#   Jacob (2000), doi:10.1023/A:1006207024468 — Henry's law scavenging
#   Liu et al. (2001), doi:10.1029/2000JD900265 — GEOS-Chem wet deposition
# ---------------------------------------------------------------------------

using DocStringExtensions

"""
$(TYPEDEF)

Supertype for tracer solubility traits used in convective wet scavenging.

Subtypes determine how much tracer mass is removed from the convective
updraft by precipitation at each level.
"""
abstract type AbstractTracerSolubility end

"""
$(TYPEDEF)

Inert tracer — no wet scavenging. All updraft tracer mass is conserved
during convective transport. Appropriate for CO₂, SF₆, ²²²Rn, CH₄.
"""
struct InertTracer <: AbstractTracerSolubility end

"""
$(TYPEDEF)

Soluble tracer — subject to wet scavenging via Henry's law dissolution
and retention in cloud condensate. Not yet implemented; will require
precipitation flux fields (PFICU, PFLCU, DQRCU) from met data.

$(FIELDS)
"""
struct SolubleTracer <: AbstractTracerSolubility
    "Henry's law constant [mol/L/atm]"
    henry_constant       :: Float64
    "fraction of dissolved tracer retained in updraft after rain-out (0–1)"
    retention_efficiency :: Float64
end

"""
$(SIGNATURES)

Return the solubility trait for a given tracer species. Defaults to
[`InertTracer`](@ref) (no scavenging) for all species. Override this
method for soluble species when wet deposition is implemented.

# Example
```julia
tracer_solubility(:sf6)        # → InertTracer()
tracer_solubility(:hno3)       # → SolubleTracer(2.1e5, 0.62) (future)
```
"""
tracer_solubility(::Symbol) = InertTracer()

"""
$(SIGNATURES)

Compute the fraction of updraft tracer mass removed by wet scavenging
at a given level. Returns `0.0` for inert tracers (no removal).

For soluble tracers (not yet implemented), this would depend on
temperature, precipitation flux, and Henry's law constant.
"""
wet_scavenge_fraction(::InertTracer, args...) = 0.0

"""
$(SIGNATURES)

Wet scavenging fraction for soluble tracers. Not yet implemented —
raises an error. Will require precipitation flux fields from GEOS met data
(PFICU, PFLCU from A3mstE collection).
"""
function wet_scavenge_fraction(::SolubleTracer, T, precip_flux, Δp)
    error("Soluble tracer wet scavenging not yet implemented. " *
          "Requires PFICU/PFLCU precipitation fields from A3mstE.")
end
