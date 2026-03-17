# ---------------------------------------------------------------------------
# TM5 matrix convection scheme — adjoint (discrete transpose)
#
# The forward operator is: rm_new = conv1^{-1} * rm_old
# The adjoint is:          λ_old = conv1^{-T} * λ_new
#
# This is exact because the forward operator is linear (conv1 is fixed from
# met data, independent of tracer state). The adjoint satisfies the
# dot-product identity to machine precision:
#   ⟨conv1^{-T} λ, δq⟩ = ⟨λ, conv1^{-1} δq⟩
#
# Reference: TM5_Conv_Apply in deps/tm5/base/src/tm5_conv.F90 lines 192-336
# uses `trans='T'` argument to LAPACK dGeTrs for the reverse/adjoint run.
# ---------------------------------------------------------------------------

using ..Fields: interior, AbstractField
using ..Grids: grid_size, Δz, floattype, LatitudeLongitudeGrid
using ..Parameters: PlanetParameters
using LinearAlgebra: lu!, ldiv!, transpose!

"""
    adjoint_convect!(adj_tracers, tm5conv_data, delp, conv::TM5MatrixConvection,
                     grid::LatitudeLongitudeGrid, dt, planet; kwargs...)

Adjoint of TM5 matrix convection for lat-lon grids.

Builds the same conv1 matrix as the forward, then solves the transposed system:
  conv1^T * λ_old = λ_new

This is equivalent to LAPACK `dGeTrs` with `trans='T'`.
"""
function adjoint_convect!(adj_tracers::NamedTuple, tm5conv_data::NamedTuple, delp,
                           conv::TM5MatrixConvection, grid::LatitudeLongitudeGrid, dt,
                           planet::PlanetParameters;
                           dtrain_panels=nothing, workspace=nothing)
    FT = floattype(grid)
    Nx, Ny, Nz = size(delp)
    grav = FT(planet.gravity)
    dt_FT = FT(dt)

    lmax = conv.lmax_conv > 0 ? min(conv.lmax_conv, Nz) : Nz

    entu_data = tm5conv_data.entu
    detu_data = tm5conv_data.detu
    entd_data = tm5conv_data.entd
    detd_data = tm5conv_data.detd

    # Process each column independently
    Threads.@threads for idx in 1:Nx*Ny
        j = div(idx - 1, Nx) + 1
        i = mod(idx - 1, Nx) + 1

        # Column work arrays (TM5 bottom-to-top convention)
        m_col     = Vector{FT}(undef, lmax)
        entu_col  = Vector{FT}(undef, lmax)
        detu_col  = Vector{FT}(undef, lmax)
        entd_col  = Vector{FT}(undef, lmax)
        detd_col  = Vector{FT}(undef, lmax)
        conv1     = Matrix{FT}(undef, lmax, lmax)

        # Extract and reverse to TM5 convention
        @inbounds for k in 1:lmax
            k_rev = Nz + 1 - k
            m_col[k]    = delp[i, j, k_rev] / grav
            entu_col[k] = entu_data[i, j, k_rev]
            detu_col[k] = detu_data[i, j, k_rev]
            entd_col[k] = entd_data[i, j, k_rev]
            detd_col[k] = detd_data[i, j, k_rev]
        end

        li, ld = _conv_cloud_dim(detu_col, entd_col, lmax)
        (li == 0 && ld == 0) && continue

        lmc = tm5_conv_matrix!(conv1, m_col, entu_col, detu_col,
                               entd_col, detd_col, lmax, li, ld, dt_FT)
        lmc == 0 && continue

        # LU factorize (same matrix as forward)
        conv1_sub = @view conv1[1:lmc, 1:lmc]
        F = lu!(conv1_sub)

        # Apply transposed solve to each adjoint tracer
        for tracer in values(adj_tracers)
            arr = _tm5_tracer_data(tracer)

            # Forward chain: q →(×m)→ rm →(conv1\)→ rm →(÷m)→ q
            # Adjoint (reverse): λ_q →(÷m)→ λ_rm →(conv1^T\)→ λ_rm →(×m)→ λ_q
            λ_col = Vector{FT}(undef, lmc)
            @inbounds for k in 1:lmc
                k_rev = Nz + 1 - k
                λ_col[k] = arr[i, j, k_rev] / m_col[k]  # adjoint of (÷m): divide by m
            end

            # Solve transposed system: conv1^T * λ_old = λ_new
            ldiv!(transpose(F), λ_col)

            # Adjoint of (×m): multiply by m
            @inbounds for k in 1:lmc
                k_rev = Nz + 1 - k
                arr[i, j, k_rev] = λ_col[k] * m_col[k]
            end
        end
    end

    return nothing
end
