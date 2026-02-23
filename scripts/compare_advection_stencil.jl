#!/usr/bin/env julia
# ===========================================================================
# Stencil-level comparison: Julia SlopesAdvection vs TM5 Fortran advectx
#
# Demonstrates that both implementations produce identical results when
# given the same input fields. Translates TM5's mass-based formulation
# to concentration form and verifies agreement.
#
# TM5 variables:
#   rm  = tracer mass per cell      (our c * m)
#   rxm = x-slope of tracer mass    (our slope * m)
#   am  = mass flux at face [kg/s]  (our u * m_face / Δx * Δt)
#   m   = air mass per cell [kg]
#
# Russell-Lerner flux (advectx.F90 lines 662-674):
#   if am >= 0:  α = am/m[i],      f = α*(rm[i] + (1-α)*rxm[i])
#   else:        α = am/m[i+1],    f = α*(rm[i+1] - (1+α)*rxm[i+1])
#
# In concentration form (c = rm/m, s = rxm/m):
#   if u >= 0:  α = u*Δt/Δx,  flux = u*(c[i] + (1-α)*s[i]/2)
#   else:       α = u*Δt/Δx,  flux = u*(c[i+1] - (1+α)*s[i+1]/2)
#
# This is exactly what slopes_advection.jl implements.
# ===========================================================================

using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection
using LinearAlgebra

@info "=" ^ 60
@info "Stencil Comparison: Julia SlopesAdvection vs TM5 advectx"
@info "=" ^ 60

# ---------------------------------------------------------------------------
# Setup: small grid, known field, known velocity
# ---------------------------------------------------------------------------
const FT = Float64
Nx, Ny, Nz = 32, 16, 5

vc = HybridSigmaPressure(
    FT[0.0, 5000.0, 10000.0, 20000.0, 50000.0, 101325.0],
    FT[0.0, 0.0, 0.1, 0.3, 0.7, 1.0])

grid = LatitudeLongitudeGrid(CPU(); FT,
    size = (Nx, Ny, Nz),
    longitude = (-180.0, 180.0),
    latitude = (-90.0, 90.0),
    vertical = vc)

Δt_test = FT(300.0)

# Gaussian blob initial condition
c0 = zeros(FT, Nx, Ny, Nz)
for k in 1:Nz, j in 1:Ny, i in 1:Nx
    x = (i - Nx/2)^2 / 20.0
    y = (j - Ny/2)^2 / 10.0
    z = (k - 3)^2 / 2.0
    c0[i, j, k] = FT(1.0) + FT(0.5) * exp(-(x + y + z))
end

# Spatially varying velocity (periodic, staggered)
u = zeros(FT, Nx + 1, Ny, Nz)
for k in 1:Nz, j in 1:Ny, i in 1:Nx+1
    u[i, j, k] = FT(5.0) * sin(2π * (i - 1) / Nx) + FT(2.0)
end

v = zeros(FT, Nx, Ny + 1, Nz)
w = zeros(FT, Nx, Ny, Nz + 1)
velocities = (; u, v, w)

# ===========================================================================
# 1. Julia SlopesAdvection (no limiter — linear, for exact comparison)
# ===========================================================================
@info "\n--- Julia SlopesAdvection (no limiter) ---"
tracers_julia = (; c = copy(c0))
scheme_nolim = SlopesAdvection(use_limiter=false)
advect_x!(tracers_julia, velocities, grid, scheme_nolim, Δt_test)
c_julia = copy(tracers_julia.c)

# ===========================================================================
# 2. TM5 Fortran stencil (translated to Julia, concentration form)
# ===========================================================================
@info "--- TM5 stencil (translated, concentration form) ---"

function tm5_advect_x_stencil!(c_out, c_in, u, grid, Δt)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    FT = eltype(c_in)

    for k in 1:Nz, j in 1:Ny
        Δx_j = Δx(1, j, grid)

        for i in 1:Nx
            i_prev = i == 1 ? Nx : i - 1
            i_next = i == Nx ? 1 : i + 1
            i_pp   = i_next == Nx ? 1 : i_next + 1
            i_mm   = i_prev == 1 ? Nx : i_prev - 1

            # Russell-Lerner slope (centered difference, no limiter)
            s_i      = (c_in[i_next, j, k] - c_in[i_prev, j, k]) / 2
            s_i_next = (c_in[i_pp, j, k]   - c_in[i, j, k])      / 2
            s_i_prev = (c_in[i, j, k]      - c_in[i_mm, j, k])   / 2

            u_right = u[i + 1, j, k]
            u_left  = u[i, j, k]

            # TM5 stencil in concentration form:
            # α = u * Δt / Δx  (Courant number)
            # if u >= 0: flux = u * (c[i] + (1 - α) * s[i] / 2)
            # if u <  0: flux = u * (c[i+1] - (1 + α) * s[i+1] / 2)
            if u_right >= 0
                α_r = u_right * Δt / Δx_j
                flux_right = u_right * (c_in[i, j, k] + (1 - α_r) * s_i / 2)
            else
                α_r = u_right * Δt / Δx_j
                flux_right = u_right * (c_in[i_next, j, k] - (1 + α_r) * s_i_next / 2)
            end

            if u_left >= 0
                α_l = u_left * Δt / Δx_j
                flux_left = u_left * (c_in[i_prev, j, k] + (1 - α_l) * s_i_prev / 2)
            else
                α_l = u_left * Δt / Δx_j
                flux_left = u_left * (c_in[i, j, k] - (1 + α_l) * s_i / 2)
            end

            c_out[i, j, k] = c_in[i, j, k] - Δt / Δx_j * (flux_right - flux_left)
        end
    end
end

c_tm5 = similar(c0)
tm5_advect_x_stencil!(c_tm5, c0, u, grid, Δt_test)

# ===========================================================================
# 3. Compare
# ===========================================================================
@info "\n--- Comparison ---"

max_abs_diff = maximum(abs.(c_julia .- c_tm5))
max_rel_diff = maximum(abs.(c_julia .- c_tm5) ./ max.(abs.(c_julia), FT(1e-30)))
rmse = sqrt(sum((c_julia .- c_tm5).^2) / length(c_julia))

@info "  Max absolute difference: $(max_abs_diff)"
@info "  Max relative difference: $(max_rel_diff)"
@info "  RMSE:                    $(rmse)"

if max_abs_diff < 1e-12
    @info "  PASS: Julia and TM5 stencils agree to machine precision"
else
    @warn "  FAIL: Stencils disagree — investigate"
end

# ===========================================================================
# 4. Also compare with limiter
# ===========================================================================
@info "\n--- With minmod limiter ---"

function tm5_advect_x_stencil_lim!(c_out, c_in, u, grid, Δt)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    FT = eltype(c_in)

    minmod(a, b, c) = begin
        if a > 0 && b > 0 && c > 0
            min(a, b, c)
        elseif a < 0 && b < 0 && c < 0
            max(a, b, c)
        else
            zero(a)
        end
    end

    for k in 1:Nz, j in 1:Ny
        Δx_j = Δx(1, j, grid)

        for i in 1:Nx
            i_prev = i == 1 ? Nx : i - 1
            i_next = i == Nx ? 1 : i + 1
            i_pp   = i_next == Nx ? 1 : i_next + 1
            i_mm   = i_prev == 1 ? Nx : i_prev - 1

            s_i = minmod(
                (c_in[i_next, j, k] - c_in[i_prev, j, k]) / 2,
                2 * (c_in[i_next, j, k] - c_in[i, j, k]),
                2 * (c_in[i, j, k] - c_in[i_prev, j, k]))

            s_i_next = minmod(
                (c_in[i_pp, j, k] - c_in[i, j, k]) / 2,
                2 * (c_in[i_pp, j, k] - c_in[i_next, j, k]),
                2 * (c_in[i_next, j, k] - c_in[i, j, k]))

            s_i_prev = minmod(
                (c_in[i, j, k] - c_in[i_mm, j, k]) / 2,
                2 * (c_in[i, j, k] - c_in[i_prev, j, k]),
                2 * (c_in[i_prev, j, k] - c_in[i_mm, j, k]))

            u_right = u[i + 1, j, k]
            u_left  = u[i, j, k]

            if u_right >= 0
                α_r = u_right * Δt / Δx_j
                flux_right = u_right * (c_in[i, j, k] + (1 - α_r) * s_i / 2)
            else
                α_r = u_right * Δt / Δx_j
                flux_right = u_right * (c_in[i_next, j, k] - (1 + α_r) * s_i_next / 2)
            end

            if u_left >= 0
                α_l = u_left * Δt / Δx_j
                flux_left = u_left * (c_in[i_prev, j, k] + (1 - α_l) * s_i_prev / 2)
            else
                α_l = u_left * Δt / Δx_j
                flux_left = u_left * (c_in[i, j, k] - (1 + α_l) * s_i / 2)
            end

            c_out[i, j, k] = c_in[i, j, k] - Δt / Δx_j * (flux_right - flux_left)
        end
    end
end

tracers_julia_lim = (; c = copy(c0))
scheme_lim = SlopesAdvection(use_limiter=true)
advect_x!(tracers_julia_lim, velocities, grid, scheme_lim, Δt_test)

c_tm5_lim = similar(c0)
tm5_advect_x_stencil_lim!(c_tm5_lim, c0, u, grid, Δt_test)

max_abs_lim = maximum(abs.(tracers_julia_lim.c .- c_tm5_lim))
max_rel_lim = maximum(abs.(tracers_julia_lim.c .- c_tm5_lim) ./ max.(abs.(tracers_julia_lim.c), FT(1e-30)))

@info "  Max absolute difference (limiter): $(max_abs_lim)"
@info "  Max relative difference (limiter): $(max_rel_lim)"

if max_abs_lim < 1e-12
    @info "  PASS: Limiter stencils agree to machine precision"
else
    @warn "  FAIL: Limiter stencils disagree — investigate"
end

# ===========================================================================
# 5. Mass conservation check
# ===========================================================================
@info "\n--- Mass conservation ---"
mass0 = sum(c0)
mass_julia = sum(c_julia)
mass_tm5 = sum(c_tm5)
@info "  Initial mass:     $(mass0)"
@info "  Julia after x:    $(mass_julia) (change: $(abs(mass_julia - mass0) / mass0))"
@info "  TM5 after x:      $(mass_tm5) (change: $(abs(mass_tm5 - mass0) / mass0))"

# ===========================================================================
# 6. Adjoint dot-product test: ⟨Lᵀλ, δc⟩ = ⟨λ, L δc⟩
# ===========================================================================
@info "\n--- Adjoint dot-product test ---"

using Random
Random.seed!(42)
δc = randn(FT, Nx, Ny, Nz) .* FT(0.01)
λ_init = randn(FT, Nx, Ny, Nz) .* FT(0.01)

scheme_nolim_adj = SlopesAdvection(use_limiter=false)
tracers_fwd = (; c = copy(δc))
advect_x!(tracers_fwd, velocities, grid, scheme_nolim_adj, Δt_test)
Lδc = copy(tracers_fwd.c)

adj_tracers = (; c = copy(λ_init))
adjoint_advect_x!(adj_tracers, velocities, grid, scheme_nolim_adj, Δt_test)
LTλ = copy(adj_tracers.c)

lhs = dot(LTλ, δc)
rhs = dot(λ_init, Lδc)
ratio = lhs / rhs
@info "  ⟨Lᵀλ, δc⟩  = $(lhs)"
@info "  ⟨λ, L δc⟩  = $(rhs)"
@info "  Ratio:       $(ratio)"
@info "  |1 - ratio|: $(abs(1 - ratio))"

if abs(1 - ratio) < 1e-12
    @info "  PASS: Adjoint identity satisfied to machine precision"
else
    @warn "  FAIL: Adjoint identity violated"
end

@info "\n" * "=" ^ 60
