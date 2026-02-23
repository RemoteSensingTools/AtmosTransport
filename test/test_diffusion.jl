using Test
using LinearAlgebra: dot
using AtmosTransport.Diffusion
using AtmosTransport.Grids

@testset "Diffusion" begin
    @test NoDiffusion() isa AbstractDiffusion
    @test BoundaryLayerDiffusion() isa AbstractDiffusion

    # NoDiffusion should be a no-op (not error)
    @test diffuse!(nothing, nothing, nothing, NoDiffusion(), 1.0) === nothing
    @test adjoint_diffuse!(nothing, nothing, nothing, NoDiffusion(), 1.0) === nothing

    @testset "BoundaryLayerDiffusion on LatitudeLongitudeGrid" begin
        vc = HybridSigmaPressure(
            [0.0, 5000.0, 10000.0, 20000.0, 50000.0, 101325.0],
            [0.0, 0.0, 0.1, 0.3, 0.7, 1.0])
        grid = LatitudeLongitudeGrid(CPU();
            size = (36, 18, 5),
            longitude = (-180, 180),
            latitude = (-90, 90),
            vertical = vc)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        diff = BoundaryLayerDiffusion(Kz_max = 500.0)
        Δt = 3600.0

        @testset "Column smoothing (step function)" begin
            # Initial step: high in bottom half, low in top half
            c = zeros(Nx, Ny, Nz)
            for i in 1:Nx, j in 1:Ny, k in 1:Nz
                c[i, j, k] = k > Nz ÷ 2 ? 1.0 : 0.0
            end
            tracers = (; c)

            # Diffuse for many steps to get noticeable smoothing
            for _ in 1:50
                diffuse!(tracers, nothing, grid, diff, Δt)
            end

            # Verify smoothing: vertical gradient should be reduced
            # Top half should have increased from 0, bottom half decreased from 1
            c_final = tracers.c
            top_avg = sum(c_final[:, :, 1:(Nz ÷ 2)]) / (Nx * Ny * (Nz ÷ 2))
            bot_avg = sum(c_final[:, :, (Nz ÷ 2 + 1):Nz]) / (Nx * Ny * (Nz - Nz ÷ 2))
            @test top_avg > 0.01
            @test bot_avg < 0.99
            @test top_avg < bot_avg
        end

        @testset "Mass conservation" begin
            c = rand(Nx, Ny, Nz) .+ 0.1
            tracers = (; c)

            mass_before = sum(Δz(k, grid) * c[i, j, k] for i in 1:Nx, j in 1:Ny, k in 1:Nz)
            diffuse!(tracers, nothing, grid, diff, Δt)
            mass_after = sum(Δz(k, grid) * tracers.c[i, j, k] for i in 1:Nx, j in 1:Ny, k in 1:Nz)
            @test mass_after ≈ mass_before rtol = 1e-10
        end

        @testset "Adjoint identity (gradient test)" begin
            # dot(A'λ, δc) ≈ dot(λ, A*δc)
            # A = forward diffusion operator (implicit solve)
            # Use smaller Kz for better numerical precision in adjoint test
            diff_adj = BoundaryLayerDiffusion(Kz_max = 50.0)
            c = rand(Nx, Ny, Nz) .+ 0.1
            tracers = (; c)
            λ = rand(Nx, Ny, Nz)
            adj_tracers = (; c = copy(λ))
            δc = rand(Nx, Ny, Nz) .* 0.01

            # Forward perturbation: A * δc
            tracers_pert = (; c = copy(δc))
            diffuse!(tracers_pert, nothing, grid, diff_adj, Δt)
            Aδc = tracers_pert.c

            # Adjoint: A' * λ (in-place overwrites adj_tracers)
            adjoint_diffuse!(adj_tracers, nothing, grid, diff_adj, Δt)
            Atλ = adj_tracers.c

            # Adjoint identity: dot(A'λ, δc) ≈ dot(λ, Aδc)
            # With the transposed Thomas solve, this should be machine precision
            lhs = dot(Atλ, δc)
            rhs = dot(λ, Aδc)
            @test lhs ≈ rhs rtol = 1e-10
        end
    end
end
