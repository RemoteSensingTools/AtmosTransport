# ---------------------------------------------------------------------------
# Tiedtke (1989) mass-flux convection scheme — forward
#
# Uses updraft/downdraft mass fluxes from met data to redistribute tracers
# vertically. The mass fluxes are external (from ECMWF/GEOS reanalysis),
# so the operation is LINEAR in tracer concentration.
#
# The convective tracer flux at interface k is:
#   F[k] = M_net[k] * q_upwind
# where M_net is the net convective mass flux (positive upward) and
# q_upwind is selected by the sign of M_net.
#
# Layer tendency:
#   q_new[k] = q_old[k] + Δt * g / Δp[k] * (F[k+1] - F[k])
#
# Mass conservation is guaranteed because the flux telescopes:
#   Σ_k Δp[k] * q_new[k] = Σ_k Δp[k] * q_old[k]
# when F[1] = F[Nz+1] = 0 (no flux at top/bottom boundaries).
# ---------------------------------------------------------------------------

using ..Fields: interior, AbstractField
using ..Grids: grid_size, Δz, floattype, LatitudeLongitudeGrid


"""
    conv_tracer_data(t)

Return the modifiable 3D array for a tracer (Field or raw array).
"""
conv_tracer_data(t) = t isa AbstractField ? interior(t) : t

"""
    has_conv_mass_flux(met)

Check whether `met` provides convective mass flux data.
Returns `false` for `nothing` or any object without a `conv_mass_flux` field.
"""
has_conv_mass_flux(met) = met !== nothing && hasproperty(met, :conv_mass_flux)

"""
    convect!(tracers::NamedTuple, met, grid::LatitudeLongitudeGrid,
             conv::TiedtkeConvection, Δt)

Apply Tiedtke mass-flux convection to all tracers in-place.

# Met data

`met` should be a NamedTuple (or similar) with field `conv_mass_flux`:
a 3D array of size `(Nx, Ny, Nz+1)` containing the net convective mass flux
[kg/m²/s] at each interface level, positive upward.

If `met` is `nothing` or lacks `conv_mass_flux`, this is a no-op.

# Algorithm

For each column `(i, j)`, computes upwind fluxes at interior interfaces
and updates the tracer mixing ratio:

    q[k] += Δt * g / Δp[k] * (F[k+1] - F[k])

where `F[k] = M_net[k] * q_upwind` with upwind selection based on the
sign of `M_net[k]`.

Interface indexing:
- Interface `k` sits between layer `k-1` (above) and layer `k` (below).
- `M > 0` (upward): tracer sourced from layer `k` (below the interface).
- `M < 0` (downward): tracer sourced from layer `k-1` (above the interface).

Zero-flux boundary conditions at top (`k=1`) and surface (`k=Nz+1`)
ensure mass conservation.
"""
function convect!(tracers::NamedTuple, met, grid::LatitudeLongitudeGrid,
                  conv::TiedtkeConvection, Δt)
    has_conv_mass_flux(met) || return nothing

    mf = met.conv_mass_flux

    gs = grid_size(grid)
    Nx, Ny, Nz = gs.Nx, gs.Ny, gs.Nz
    FT = floattype(grid)
    grav = grid.gravity

    col  = Vector{FT}(undef, Nz)
    flux = Vector{FT}(undef, Nz + 1)

    for tracer in values(tracers)
        arr = conv_tracer_data(tracer)

        for j in 1:Ny, i in 1:Nx
            # Extract column
            @inbounds for k in 1:Nz
                col[k] = arr[i, j, k]
            end

            # Compute upwind fluxes at interfaces
            @inbounds begin
                flux[1]      = zero(FT)   # no flux through model top
                flux[Nz + 1] = zero(FT)   # no flux through surface

                for k in 2:Nz
                    M = FT(mf[i, j, k])
                    # M > 0 (upward): tracer from layer k (below interface)
                    # M < 0 (downward): tracer from layer k-1 (above interface)
                    flux[k] = M >= zero(FT) ? M * col[k] : M * col[k - 1]
                end
            end

            # Apply tendency: q_new = q_old + Δt * g / Δp * (F_below - F_above)
            @inbounds for k in 1:Nz
                Δp_k = FT(Δz(k, grid))
                arr[i, j, k] = col[k] + FT(Δt) * grav / Δp_k * (flux[k + 1] - flux[k])
            end
        end
    end

    return nothing
end
