using Test
using LinearAlgebra: dot
using AtmosTransportModel
using AtmosTransportModel.Advection
using AtmosTransportModel.Grids

@testset "Advection" begin
    @test SlopesAdvection() isa AbstractAdvectionScheme
    @test SlopesAdvection(use_limiter=false) isa AbstractAdvectionScheme
    @test UpwindAdvection() isa AbstractAdvectionScheme

    # Forward/adjoint stubs should error with informative messages
    @test_throws ErrorException advect!(nothing, nothing, nothing, SlopesAdvection(), 1.0)
    @test_throws ErrorException adjoint_advect!(nothing, nothing, nothing, SlopesAdvection(), 1.0)

    @testset "UpwindAdvection on LatitudeLongitudeGrid" begin
        vc = HybridSigmaPressure(
            [0.0, 5000.0, 10000.0, 20000.0, 50000.0, 101325.0],
            [0.0, 0.0, 0.1, 0.3, 0.7, 1.0])
        grid = LatitudeLongitudeGrid(CPU();
            size = (36, 18, 5),
            longitude = (-180, 180),
            latitude = (-90, 90),
            vertical = vc)
        scheme = UpwindAdvection()
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

        # 1D advection test: uniform wind in x, Gaussian initial condition, periodic domain
        @testset "1D advection (uniform wind, Gaussian)" begin
            # Uniform u > 0 (eastward), zero v and w
            u = fill(10.0, Nx + 1, Ny, Nz)
            v = zeros(Nx, Ny + 1, Nz)
            w = zeros(Nx, Ny, Nz + 1)
            velocities = (; u, v, w)

            # Gaussian in x at center of domain, constant in y and z
            c = zeros(Nx, Ny, Nz)
            x_center = (Nx + 1) / 2
            σ = 3.0
            for i in 1:Nx, j in 1:Ny, k in 1:Nz
                c[i, j, k] = exp(-((i - x_center)^2) / (2 * σ^2))
            end
            tracers = (; c)

            mass_before = sum(c)
            Δt = 60.0  # 1 minute
            advect_x!(tracers, velocities, grid, scheme, Δt)
            mass_after = sum(tracers.c)
            @test mass_after ≈ mass_before rtol = 1e-10
        end

        @testset "Mass conservation (x, y, z)" begin
            u = fill(5.0, Nx + 1, Ny, Nz)
            v = zeros(Nx, Ny + 1, Nz)
            w = zeros(Nx, Ny, Nz + 1)
            velocities = (; u, v, w)
            c = rand(Nx, Ny, Nz) .+ 0.1
            tracers = (; c)

            mass0 = sum(c)
            advect_x!(tracers, velocities, grid, scheme, 30.0)
            @test sum(tracers.c) ≈ mass0 rtol = 1e-10

            # Reset and test y (zero v at boundaries for mass conservation)
            # Spherical y-advection conserves physical mass (weighted by Δsinφ),
            # not simple sum(c).
            c = rand(Nx, Ny, Nz) .+ 0.1
            tracers = (; c)
            v = fill(0.5, Nx, Ny + 1, Nz)
            v[:, 1, :] .= 0
            v[:, end, :] .= 0
            u = zeros(Nx + 1, Ny, Nz)
            w = zeros(Nx, Ny, Nz + 1)
            velocities = (; u, v, w)
            φᶠ = Array(grid.φᶠ)
            phys_mass(cc) = sum(abs(sind(φᶠ[j+1]) - sind(φᶠ[j])) * cc[i,j,k]
                                for i in 1:Nx, j in 1:Ny, k in 1:Nz)
            mass0 = phys_mass(c)
            advect_y!(tracers, velocities, grid, scheme, 30.0)
            @test phys_mass(tracers.c) ≈ mass0 rtol = 1e-10

            # Reset and test z (zero w at top/bottom for mass conservation)
            # Conserved quantity is Δz-weighted mass (pressure thickness × concentration)
            c = rand(Nx, Ny, Nz) .+ 0.1
            tracers = (; c)
            w = fill(-10.0, Nx, Ny, Nz + 1)
            w[:, :, 1] .= 0
            w[:, :, end] .= 0
            u = zeros(Nx + 1, Ny, Nz)
            v = zeros(Nx, Ny + 1, Nz)
            velocities = (; u, v, w)
            mass0 = sum(Δz(k, grid) * c[i, j, k] for i in 1:Nx, j in 1:Ny, k in 1:Nz)
            advect_z!(tracers, velocities, grid, scheme, 30.0)
            mass_after = sum(Δz(k, grid) * tracers.c[i, j, k] for i in 1:Nx, j in 1:Ny, k in 1:Nz)
            @test mass_after ≈ mass0 rtol = 1e-10
        end

        @testset "Adjoint identity (gradient test)" begin
            # dot(A' * λ, δc) ≈ dot(λ, A * δc)
            u = fill(10.0, Nx + 1, Ny, Nz)
            v = zeros(Nx, Ny + 1, Nz)
            w = zeros(Nx, Ny, Nz + 1)
            velocities = (; u, v, w)

            c = rand(Nx, Ny, Nz) .+ 0.1
            tracers = (; c)
            λ = rand(Nx, Ny, Nz)
            adj_tracers = (; c = copy(λ))
            δc = rand(Nx, Ny, Nz) .* 0.01

            Δt = 60.0

            # Forward perturbation: A * δc
            tracers_pert = (; c = copy(δc))
            advect_x!(tracers_pert, velocities, grid, scheme, Δt)
            Aδc = tracers_pert.c

            # Adjoint: A' * λ
            adjoint_advect_x!(adj_tracers, velocities, grid, scheme, Δt)
            Atλ = adj_tracers.c

            # Adjoint identity: dot(A'λ, δc) ≈ dot(λ, Aδc) — exact for linear upwind
            lhs = dot(Atλ, δc)
            rhs = dot(λ, Aδc)
            @test lhs ≈ rhs rtol = 1e-10

            # Also test y and z (zero velocity at boundaries)
            v = fill(1.0, Nx, Ny + 1, Nz)
            v[:, 1, :] .= 0
            v[:, end, :] .= 0
            u = zeros(Nx + 1, Ny, Nz)
            w = zeros(Nx, Ny, Nz + 1)
            velocities = (; u, v, w)
            c = rand(Nx, Ny, Nz) .+ 0.1
            tracers = (; c)
            λ = rand(Nx, Ny, Nz)
            adj_tracers = (; c = copy(λ))
            δc = rand(Nx, Ny, Nz) .* 0.01
            tracers_pert = (; c = copy(δc))
            advect_y!(tracers_pert, velocities, grid, scheme, Δt)
            adjoint_advect_y!(adj_tracers, velocities, grid, scheme, Δt)
            @test dot(adj_tracers.c, δc) ≈ dot(λ, tracers_pert.c) rtol = 1e-10

            w = fill(-5.0, Nx, Ny, Nz + 1)
            w[:, :, 1] .= 0
            w[:, :, end] .= 0
            u = zeros(Nx + 1, Ny, Nz)
            v = zeros(Nx, Ny + 1, Nz)
            velocities = (; u, v, w)
            c = rand(Nx, Ny, Nz) .+ 0.1
            tracers = (; c)
            λ = rand(Nx, Ny, Nz)
            adj_tracers = (; c = copy(λ))
            δc = rand(Nx, Ny, Nz) .* 0.01
            tracers_pert = (; c = copy(δc))
            advect_z!(tracers_pert, velocities, grid, scheme, Δt)
            adjoint_advect_z!(adj_tracers, velocities, grid, scheme, Δt)
            @test dot(adj_tracers.c, δc) ≈ dot(λ, tracers_pert.c) rtol = 1e-10
        end
    end

    @testset "SlopesAdvection on LatitudeLongitudeGrid" begin
        vc = HybridSigmaPressure(
            [0.0, 5000.0, 10000.0, 20000.0, 50000.0, 101325.0],
            [0.0, 0.0, 0.1, 0.3, 0.7, 1.0])
        grid = LatitudeLongitudeGrid(CPU();
            size = (36, 18, 5),
            longitude = (-180, 180),
            latitude = (-90, 90),
            vertical = vc)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

        @testset "1D advection (uniform wind, Gaussian)" begin
            for scheme in (SlopesAdvection(), SlopesAdvection(use_limiter=false))
                u = fill(10.0, Nx + 1, Ny, Nz)
                v = zeros(Nx, Ny + 1, Nz)
                w = zeros(Nx, Ny, Nz + 1)
                velocities = (; u, v, w)

                c = zeros(Nx, Ny, Nz)
                x_center = (Nx + 1) / 2
                σ = 3.0
                for i in 1:Nx, j in 1:Ny, k in 1:Nz
                    c[i, j, k] = exp(-((i - x_center)^2) / (2 * σ^2))
                end
                tracers = (; c)

                mass_before = sum(c)
                Δt = 60.0
                advect_x!(tracers, velocities, grid, scheme, Δt)
                mass_after = sum(tracers.c)
                @test mass_after ≈ mass_before rtol = 1e-10
            end
        end

        @testset "Mass conservation (x, y, z)" begin
            for scheme in (SlopesAdvection(), SlopesAdvection(use_limiter=false))
                u = fill(5.0, Nx + 1, Ny, Nz)
                v = zeros(Nx, Ny + 1, Nz)
                w = zeros(Nx, Ny, Nz + 1)
                velocities = (; u, v, w)
                c = rand(Nx, Ny, Nz) .+ 0.1
                tracers = (; c)

                mass0 = sum(c)
                advect_x!(tracers, velocities, grid, scheme, 30.0)
                @test sum(tracers.c) ≈ mass0 rtol = 1e-10

                c = rand(Nx, Ny, Nz) .+ 0.1
                tracers = (; c)
                v = fill(0.5, Nx, Ny + 1, Nz)
                v[:, 1, :] .= 0
                v[:, end, :] .= 0
                u = zeros(Nx + 1, Ny, Nz)
                w = zeros(Nx, Ny, Nz + 1)
                velocities = (; u, v, w)
                mass0 = sum(c)
                advect_y!(tracers, velocities, grid, scheme, 30.0)
                @test sum(tracers.c) ≈ mass0 rtol = 1e-10

                c = rand(Nx, Ny, Nz) .+ 0.1
                tracers = (; c)
                w = fill(-10.0, Nx, Ny, Nz + 1)
                w[:, :, 1] .= 0
                w[:, :, end] .= 0
                u = zeros(Nx + 1, Ny, Nz)
                v = zeros(Nx, Ny + 1, Nz)
                velocities = (; u, v, w)
                mass0 = sum(Δz(k, grid) * c[i, j, k] for i in 1:Nx, j in 1:Ny, k in 1:Nz)
                advect_z!(tracers, velocities, grid, scheme, 30.0)
                mass_after = sum(Δz(k, grid) * tracers.c[i, j, k] for i in 1:Nx, j in 1:Ny, k in 1:Nz)
                @test mass_after ≈ mass0 rtol = 1e-10
            end
        end

        @testset "Slopes more accurate than upwind (less diffusion)" begin
            # Advect a Gaussian N steps; slopes should preserve peak better than upwind
            u = fill(10.0, Nx + 1, Ny, Nz)
            v = zeros(Nx, Ny + 1, Nz)
            w = zeros(Nx, Ny, Nz + 1)
            velocities = (; u, v, w)

            c_init = zeros(Nx, Ny, Nz)
            x_center = (Nx + 1) / 2
            σ = 3.0
            for i in 1:Nx, j in 1:Ny, k in 1:Nz
                c_init[i, j, k] = exp(-((i - x_center)^2) / (2 * σ^2))
            end

            N_steps = 20
            Δt = 60.0

            # Upwind
            c_upwind = copy(c_init)
            tracers_upwind = (; c = c_upwind)
            for _ in 1:N_steps
                advect_x!(tracers_upwind, velocities, grid, UpwindAdvection(), Δt)
            end
            peak_upwind = maximum(tracers_upwind.c)

            # Slopes (with limiter)
            c_slopes = copy(c_init)
            tracers_slopes = (; c = c_slopes)
            for _ in 1:N_steps
                advect_x!(tracers_slopes, velocities, grid, SlopesAdvection(), Δt)
            end
            peak_slopes = maximum(tracers_slopes.c)

            # Slopes should be less diffusive (higher peak)
            @test peak_slopes > peak_upwind
        end

        @testset "use_limiter=false vs use_limiter=true" begin
            # Both should conserve mass; limiter may reduce oscillations
            u = fill(10.0, Nx + 1, Ny, Nz)
            v = zeros(Nx, Ny + 1, Nz)
            w = zeros(Nx, Ny, Nz + 1)
            velocities = (; u, v, w)

            c_init = zeros(Nx, Ny, Nz)
            x_center = (Nx + 1) / 2
            σ = 2.0  # Sharper Gaussian to potentially trigger oscillations
            for i in 1:Nx, j in 1:Ny, k in 1:Nz
                c_init[i, j, k] = exp(-((i - x_center)^2) / (2 * σ^2))
            end

            mass0 = sum(c_init)
            N_steps = 10
            Δt = 60.0

            for scheme in (SlopesAdvection(use_limiter=false), SlopesAdvection(use_limiter=true))
                c = copy(c_init)
                tracers = (; c)
                for _ in 1:N_steps
                    advect_x!(tracers, velocities, grid, scheme, Δt)
                end
                @test sum(tracers.c) ≈ mass0 rtol = 1e-10
            end
        end

        @testset "Adjoint identity — no limiter (exact transpose)" begin
            scheme = SlopesAdvection(use_limiter=false)
            Δt = 60.0

            # x-direction (periodic)
            u = randn(Nx + 1, Ny, Nz) .* 5.0
            v = zeros(Nx, Ny + 1, Nz)
            w = zeros(Nx, Ny, Nz + 1)
            velocities = (; u, v, w)

            δc = randn(Nx, Ny, Nz) .* 0.01
            λ  = randn(Nx, Ny, Nz)
            tracers_pert = (; c = copy(δc))
            adj_tracers  = (; c = copy(λ))
            advect_x!(tracers_pert, velocities, grid, scheme, Δt)
            adjoint_advect_x!(adj_tracers, velocities, grid, scheme, Δt)
            @test dot(adj_tracers.c, δc) ≈ dot(λ, tracers_pert.c) rtol = 1e-10

            # y-direction (bounded, zero v at boundaries)
            v = randn(Nx, Ny + 1, Nz) .* 2.0
            v[:, 1, :]   .= 0
            v[:, end, :] .= 0
            u = zeros(Nx + 1, Ny, Nz)
            w = zeros(Nx, Ny, Nz + 1)
            velocities = (; u, v, w)
            δc = randn(Nx, Ny, Nz) .* 0.01
            λ  = randn(Nx, Ny, Nz)
            tracers_pert = (; c = copy(δc))
            adj_tracers  = (; c = copy(λ))
            advect_y!(tracers_pert, velocities, grid, scheme, Δt)
            adjoint_advect_y!(adj_tracers, velocities, grid, scheme, Δt)
            @test dot(adj_tracers.c, δc) ≈ dot(λ, tracers_pert.c) rtol = 1e-10

            # z-direction (bounded, zero w at top/bottom)
            w = randn(Nx, Ny, Nz + 1) .* 3.0
            w[:, :, 1]   .= 0
            w[:, :, end] .= 0
            u = zeros(Nx + 1, Ny, Nz)
            v = zeros(Nx, Ny + 1, Nz)
            velocities = (; u, v, w)
            δc = randn(Nx, Ny, Nz) .* 0.01
            λ  = randn(Nx, Ny, Nz)
            tracers_pert = (; c = copy(δc))
            adj_tracers  = (; c = copy(λ))
            advect_z!(tracers_pert, velocities, grid, scheme, Δt)
            adjoint_advect_z!(adj_tracers, velocities, grid, scheme, Δt)
            @test dot(adj_tracers.c, δc) ≈ dot(λ, tracers_pert.c) rtol = 1e-10
        end

        @testset "Adjoint — with limiter (TM5 continuous adjoint)" begin
            # Continuous adjoint: forward code with negated velocity (Niwa et al. 2017).
            # Not the exact transpose, but a well-established approximation that ensures
            # monotonicity of adjoint sensitivities. The dot-product identity is
            # approximate due to the nonlinearity of the flux limiter.
            scheme = SlopesAdvection(use_limiter=true)
            Δt = 60.0

            u = fill(5.0, Nx + 1, Ny, Nz)
            v = zeros(Nx, Ny + 1, Nz)
            w = zeros(Nx, Ny, Nz + 1)
            velocities = (; u, v, w)

            # Smooth Gaussian fields — limiter should mostly select centered diff
            δc = zeros(Nx, Ny, Nz)
            x_center = (Nx + 1) / 2
            σ = 5.0
            for i in 1:Nx, j in 1:Ny, k in 1:Nz
                δc[i, j, k] = 0.01 * exp(-((i - x_center)^2) / (2 * σ^2))
            end
            λ = zeros(Nx, Ny, Nz)
            for i in 1:Nx, j in 1:Ny, k in 1:Nz
                λ[i, j, k] = exp(-((i - x_center + 3)^2) / (2 * σ^2))
            end

            tracers_pert = (; c = copy(δc))
            adj_tracers  = (; c = copy(λ))
            advect_x!(tracers_pert, velocities, grid, scheme, Δt)
            adjoint_advect_x!(adj_tracers, velocities, grid, scheme, Δt)

            lhs = dot(adj_tracers.c, δc)
            rhs = dot(λ, tracers_pert.c)
            # Continuous adjoint: not exact, but should be reasonable
            @test abs(lhs - rhs) / max(abs(lhs), abs(rhs)) < 0.3
        end
    end

    @testset "Transport direction: w > 0 moves mass downward (increasing k)" begin
        vc = HybridSigmaPressure(
            [0.0, 20000.0, 40000.0, 60000.0, 80000.0, 101325.0],
            [0.0, 0.0, 0.1, 0.3, 0.7, 1.0])
        grid = LatitudeLongitudeGrid(CPU();
            size = (4, 4, 5),
            longitude = (-180, 180),
            latitude = (-90, 90),
            vertical = vc)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

        for scheme in [UpwindAdvection(), SlopesAdvection(use_limiter = false)]
            label = scheme isa UpwindAdvection ? "Upwind" : "Slopes"

            @testset "$label: w > 0 → downward transport" begin
                c = zeros(Nx, Ny, Nz)
                c[:, :, 1] .= 100.0
                sum_k1_before = sum(c[:, :, 1])
                tracers = (; c)

                u = zeros(Nx + 1, Ny, Nz)
                v = zeros(Nx, Ny + 1, Nz)
                w = fill(5000.0, Nx, Ny, Nz + 1)  # w > 0 = downward
                w[:, :, 1] .= 0
                w[:, :, end] .= 0
                velocities = (; u, v, w)

                advect_z!(tracers, velocities, grid, scheme, 1.0)

                @test sum(tracers.c[:, :, 1]) < sum_k1_before
                @test sum(tracers.c[:, :, 2]) > 0
            end

            @testset "$label: w < 0 → upward transport" begin
                c = zeros(Nx, Ny, Nz)
                c[:, :, Nz] .= 100.0
                sum_kN_before = sum(c[:, :, Nz])
                tracers = (; c)

                u = zeros(Nx + 1, Ny, Nz)
                v = zeros(Nx, Ny + 1, Nz)
                w = fill(-5000.0, Nx, Ny, Nz + 1)  # w < 0 = upward
                w[:, :, 1] .= 0
                w[:, :, end] .= 0
                velocities = (; u, v, w)

                advect_z!(tracers, velocities, grid, scheme, 1.0)

                @test sum(tracers.c[:, :, Nz]) < sum_kN_before
                @test sum(tracers.c[:, :, Nz - 1]) > 0
            end
        end
    end
end
