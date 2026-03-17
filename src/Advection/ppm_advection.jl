# ---------------------------------------------------------------------------
# Putman & Lin (2007) PPM Advection Scheme
#
# Piecewise Parabolic Method for tracer transport on structured grids
# (lat-lon and cubed-sphere). Extends the Lin-Rood flux-form transport scheme
# with three-order accurate subgrid distributions.
#
# Paper: "Finite-volume transport on various cubed-sphere grids"
#        Putman & Lin, Journal of Computational Physics, 227:55-78 (2007)
#
# Variants:
#   ORD=4: Optimized PPM (LR96 PPM + minmod limiter)
#   ORD=5: PPM with Huynh's 2nd constraint (quasi-monotonic, improved errors)
#   ORD=6: Quasi-5th order (Suresh & Huynh, non-monotonic, best L∞)
#   ORD=7: RECOMMENDED — ORD=5 + special edge treatment at CS face discontinuities
#
# Recommended choice: ORD=7 for both lat-lon and cubed-sphere grids.
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Supertype for Putman & Lin PPM advection variants.
"""
abstract type AbstractPPMScheme <: AbstractAdvectionScheme end

"""
$(TYPEDEF)

Piecewise Parabolic Method advection scheme (Putman & Lin, 2007).

# Parameters

- `ORD ∈ {4, 5, 6, 7}`: Subgrid distribution method
  - **ORD=4** (Putman & Lin Sec. 4): Optimized PPM (LR96 PPM + minmod)
  - **ORD=5** (Putman & Lin Sec. 4): PPM with Huynh's 2nd constraint (quasi-monotonic)
  - **ORD=6** (Putman & Lin Appendix B): Quasi-5th order, non-monotonic (better L∞ errors)
  - **ORD=7** (Putman & Lin Appendix C): **RECOMMENDED** — ORD=5 + special edge treatment
    for gnomonic cubed-sphere face discontinuities

# Interface

Like other advection schemes, this supports both forward and adjoint methods,
with grid-specific implementations for LatitudeLongitudeGrid and CubedSphereGrid.

For the cubed-sphere mass-flux path (production code), dispatch on `Val{ORD}`
inside kernels for compile-time specialization.

# Example

```julia
# Create advection scheme with ORD=7 (recommended)
scheme = PPMAdvection{7}()

# Use in run loop (currently not yet integrated with main advect! interface)
# Cubed-sphere mass flux: strang_split_massflux_ppm!(rm, m, am, bm, cm, grid, Val(7), ws)
```
"""
struct PPMAdvection{ORD} <: AbstractPPMScheme
    damp_coeff::Float64
    use_linrood::Bool
    use_vertical_remap::Bool
    use_gchp::Bool
    function PPMAdvection{ORD}(; damp_coeff::Real=0.0, use_linrood::Bool=false,
                                 use_vertical_remap::Bool=false,
                                 use_gchp::Bool=false) where {ORD}
        ORD ∈ (4, 5, 6, 7) || throw(ArgumentError("ORD must be in {4,5,6,7}, got $ORD"))
        return new{ORD}(Float64(damp_coeff), use_linrood, use_vertical_remap, use_gchp)
    end
end

# Convenience constructors
PPMAdvection(ORD::Int; damp_coeff::Real=0.0, use_linrood::Bool=false,
             use_vertical_remap::Bool=false, use_gchp::Bool=false) =
    PPMAdvection{ORD}(; damp_coeff, use_linrood, use_vertical_remap, use_gchp)

"""
    PPMAdvection{7}()

Return the recommended PPM advection scheme (ORD=7).
"""
PPMAdvection(; damp_coeff::Real=0.0) = PPMAdvection{7}(; damp_coeff)

function Base.show(io::IO, s::PPMAdvection{ORD}) where {ORD}
    damp = s.damp_coeff > 0 ? ", damp=$(s.damp_coeff)" : ""
    print(io, "PPMAdvection{$ORD} (Putman & Lin, ORD=$ORD$damp)")
end
