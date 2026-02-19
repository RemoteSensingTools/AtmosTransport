# ---------------------------------------------------------------------------
# Discrete adjoint of Tiedtke convection
#
# Since mass fluxes are fixed (from met data), the forward operator is a
# linear map:  q_new = L * q_old
#
# The adjoint is the exact transpose:  λ_old = L^T * λ_new
#
# Parity: ⟨L^T λ, δq⟩ = ⟨λ, L δq⟩ to machine precision (see test_convection.jl
# "Adjoint identity (exact discrete transpose)").
#
# Following TM5/NICAM-TM (Niwa et al., 2017), the discrete adjoint approach
# is used for the convection process because the operator is already linear
# in tracer concentration.
#
# Implementation uses a scatter pattern: for each output level k, scatter
# λ_new[k] back to source levels weighted by the transposed forward
# stencil coefficients.
# ---------------------------------------------------------------------------

using ..Fields: interior, AbstractField
using ..Grids: grid_size, Δz, floattype, LatitudeLongitudeGrid

"""
$(SIGNATURES)

Discrete adjoint of `convect!` for the Tiedtke mass-flux scheme.

Since the forward operator is linear in tracer concentration (mass fluxes are
fixed from met data), the adjoint is the exact matrix transpose. This gives
machine-precision dot-product identity:  ⟨L^T λ, δq⟩ = ⟨λ, L δq⟩.

The forward update for level k is:
    q_new[k] = q[k] + Δt·g/Δp[k] · (flux[k+1] - flux[k])
where flux[k] = M[k]·q_upwind.

The adjoint scatters λ_new[k] to λ_old via the transposed coefficients.
"""
function adjoint_convect!(adj_tracers::NamedTuple, met, grid::LatitudeLongitudeGrid,
                          conv::TiedtkeConvection, Δt)
    has_conv_mass_flux(met) || return nothing

    mf = met.conv_mass_flux

    gs = grid_size(grid)
    Nx, Ny, Nz = gs.Nx, gs.Ny, gs.Nz
    FT = floattype(grid)
    grav = grid.gravity

    λ_new = Vector{FT}(undef, Nz)
    λ_old = Vector{FT}(undef, Nz)

    for adj_tracer in values(adj_tracers)
        arr = conv_tracer_data(adj_tracer)

        for j in 1:Ny, i in 1:Nx
            @inbounds for k in 1:Nz
                λ_new[k] = arr[i, j, k]
                λ_old[k] = λ_new[k]   # identity contribution
            end

            # Scatter adjoint contributions from each output level
            @inbounds for k in 1:Nz
                Δp_k = FT(Δz(k, grid))
                fac = FT(Δt) * grav / Δp_k
                λv = λ_new[k]

                # +fac * flux[k+1] contribution:
                # flux[k+1] is non-zero for k+1 ∈ [2, Nz], i.e. k ∈ [1, Nz-1]
                if k < Nz
                    M = FT(mf[i, j, k + 1])
                    if M >= zero(FT)
                        # flux[k+1] = M * q[k+1] (upward: source from below)
                        # ∂q_new[k]/∂q[k+1] = +fac * M
                        λ_old[k + 1] += fac * M * λv
                    else
                        # flux[k+1] = M * q[k] (downward: source from above)
                        # ∂q_new[k]/∂q[k] = +fac * M
                        λ_old[k] += fac * M * λv
                    end
                end

                # -fac * flux[k] contribution:
                # flux[k] is non-zero for k ∈ [2, Nz]
                if k >= 2
                    M = FT(mf[i, j, k])
                    if M >= zero(FT)
                        # flux[k] = M * q[k] (upward: source from below)
                        # ∂q_new[k]/∂q[k] = -fac * M
                        λ_old[k] -= fac * M * λv
                    else
                        # flux[k] = M * q[k-1] (downward: source from above)
                        # ∂q_new[k]/∂q[k-1] = -fac * M
                        λ_old[k - 1] -= fac * M * λv
                    end
                end
            end

            @inbounds for k in 1:Nz
                arr[i, j, k] = λ_old[k]
            end
        end
    end

    return nothing
end
