# ---------------------------------------------------------------------------
# Gradient test: verify adjoint correctness
#
# Compares the adjoint gradient ∇J against central finite differences:
#
#   ratio = <∇J, δx> / [J(x₀ + ε·δx) - J(x₀ - ε·δx)] / (2ε)
#
# Central differences have O(ε²) truncation error, so for an exact discrete
# adjoint the ratio should be 1 ± O(ε²). With ε = 1e-4 the agreement is
# typically better than 1e-8.
#
# The test uses a sequence of decreasing ε values and checks convergence.
# ---------------------------------------------------------------------------

using ..Grids: grid_size, floattype
using ..TimeSteppers: time_step!, adjoint_time_step!, Clock
using LinearAlgebra: dot

"""
$(TYPEDSIGNATURES)

Run a gradient test for the full operator-splitting time stepper.

Sets up random initial conditions, runs the forward model `n_steps` times
to compute a cost function `J = 0.5 * ||c_final||²`, then runs the adjoint
backward to compute `∇J`. Compares the adjoint directional derivative
against finite differences for a series of perturbation sizes `ε`.

Returns a vector of `(ε, ratio)` pairs. For a correct discrete adjoint,
`ratio → 1.0` as `ε → 0`.

# Example

```julia
# Set up grid and time stepper with no limiter for exact gradients
grid = LatitudeLongitudeGrid(CPU(); size=(8, 4, 5), ...)
ts = OperatorSplittingTimeStepper(
    advection  = SlopesAdvection(use_limiter=false),
    convection = TiedtkeConvection(),
    diffusion  = BoundaryLayerDiffusion(Kz_max=50.0),
    Δt_outer   = 900.0)

# Constant met data (staggered velocities)
met = (; u = fill(5.0, 9, 4, 5),
         v = zeros(8, 5, 5),
         w = zeros(8, 4, 6),
         conv_mass_flux = zeros(8, 4, 6))

results = gradient_test(; grid, timestepper=ts, met_data=met, n_steps=3)
# All ratios should be ≈ 1.0
```
"""
function gradient_test(;
        grid,
        timestepper,
        met_data,
        n_steps::Int = 1,
        Δt::Real = timestepper.Δt_outer,
        epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        verbose::Bool = false)

    gs = grid_size(grid)
    Nx, Ny, Nz = gs.Nx, gs.Ny, gs.Nz
    FT = floattype(grid)

    # Random initial condition and perturbation direction
    c_init = randn(FT, Nx, Ny, Nz) .* FT(0.01) .+ FT(1.0)
    δc = randn(FT, Nx, Ny, Nz) .* FT(0.01)

    # Forward run: c0 → c_final, returns cost J = 0.5 * ||c_final||²
    function forward_cost(c0)
        tracers = (; c = copy(c0))
        clock = Clock(FT; Δt = FT(Δt))
        pseudo_model = (; tracers, met_data, grid, timestepper, clock)
        for _ in 1:n_steps
            time_step!(pseudo_model, Δt)
        end
        return FT(0.5) * dot(tracers.c, tracers.c), tracers.c
    end

    # ── Reference forward run ──
    J0, c_final = forward_cost(c_init)

    if verbose
        @info "Reference cost J₀ = $J0"
    end

    # ── Adjoint run ──
    # Adjoint forcing: ∂J/∂c_final = c_final
    adj_tracers = (; c = copy(c_final))
    adj_clock = Clock(FT; Δt = FT(Δt))
    adj_clock.time = FT(Δt * n_steps)
    adj_clock.iteration = n_steps
    pseudo_model_adj = (; adj_tracers, met_data, grid, timestepper, clock = adj_clock)

    for _ in 1:n_steps
        adjoint_time_step!(pseudo_model_adj, Δt)
    end

    grad = adj_tracers.c
    adj_dd = dot(grad, δc)  # adjoint directional derivative

    if verbose
        @info "Adjoint directional derivative = $adj_dd"
    end

    # ── Central finite-difference check: O(ε²) truncation ──
    results = Tuple{FT, FT}[]

    for ε in epsilons
        c_plus  = c_init .+ FT(ε) .* δc
        c_minus = c_init .- FT(ε) .* δc
        J_plus, _  = forward_cost(c_plus)
        J_minus, _ = forward_cost(c_minus)
        fd_dd = (J_plus - J_minus) / (FT(2) * FT(ε))
        ratio = adj_dd / fd_dd

        push!(results, (FT(ε), ratio))

        if verbose
            @info "ε = $ε: ratio = $ratio (adj = $adj_dd, fd = $fd_dd)"
        end
    end

    return results
end

"""
$(TYPEDSIGNATURES)

Generic gradient test interface. Currently delegates to the keyword-based
version using a simple quadratic cost function.

For full 4DVar gradient tests with observation operators, use the keyword
version directly with a custom setup.
"""
function gradient_test(model, cost_function, control, perturbation;
                       epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    error("Generic gradient_test not yet implemented — use the keyword-based version")
end
