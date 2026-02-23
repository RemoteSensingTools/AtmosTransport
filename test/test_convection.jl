using Test
using LinearAlgebra: dot
using AtmosTransport.Convection
using AtmosTransport.Grids

@testset "Convection" begin
    @test NoConvection() isa AbstractConvection
    @test TiedtkeConvection() isa AbstractConvection

    # NoConvection should be a no-op
    @test convect!(nothing, nothing, nothing, NoConvection(), 1.0) === nothing
    @test adjoint_convect!(nothing, nothing, nothing, NoConvection(), 1.0) === nothing

    @testset "TiedtkeConvection on LatitudeLongitudeGrid" begin
        vc = HybridSigmaPressure(
            [0.0, 5000.0, 10000.0, 20000.0, 50000.0, 101325.0],
            [0.0, 0.0, 0.1, 0.3, 0.7, 1.0])
        grid = LatitudeLongitudeGrid(CPU();
            size = (4, 4, 5),
            longitude = (-180, 180),
            latitude = (-90, 90),
            vertical = vc)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        conv = TiedtkeConvection()
        Δt = 3600.0

        @testset "No-flux gives no change" begin
            c = rand(Nx, Ny, Nz) .+ 0.1
            c_orig = copy(c)
            tracers = (; c)

            mass_flux = zeros(Nx, Ny, Nz + 1)
            met = (; conv_mass_flux = mass_flux)

            convect!(tracers, met, grid, conv, Δt)
            @test tracers.c ≈ c_orig
        end

        @testset "No-op when met has no convective fluxes" begin
            c = rand(Nx, Ny, Nz) .+ 0.1
            c_orig = copy(c)
            tracers = (; c)

            # met = nothing → no-op
            convect!(tracers, nothing, grid, conv, Δt)
            @test tracers.c ≈ c_orig

            # met without conv_mass_flux field → no-op
            convect!(tracers, (;), grid, conv, Δt)
            @test tracers.c ≈ c_orig
        end

        @testset "Mass conservation" begin
            c = rand(Nx, Ny, Nz) .+ 0.1
            tracers = (; c)

            # Non-trivial mass flux profile (positive = upward)
            mass_flux = zeros(Nx, Ny, Nz + 1)
            for j in 1:Ny, i in 1:Nx
                for k in 2:Nz
                    mass_flux[i, j, k] = 0.01   # 0.01 kg/m²/s upward
                end
            end
            met = (; conv_mass_flux = mass_flux)

            # Δp-weighted mass before
            mass_before = sum(Δz(k, grid) * c[i, j, k]
                              for i in 1:Nx, j in 1:Ny, k in 1:Nz)

            convect!(tracers, met, grid, conv, Δt)

            mass_after = sum(Δz(k, grid) * tracers.c[i, j, k]
                             for i in 1:Nx, j in 1:Ny, k in 1:Nz)

            @test mass_after ≈ mass_before rtol = 1e-10
        end

        @testset "Mass conservation with mixed up/down fluxes" begin
            c = rand(Nx, Ny, Nz) .+ 0.1
            tracers = (; c)

            # Some interfaces have upward flux, others downward
            mass_flux = zeros(Nx, Ny, Nz + 1)
            for j in 1:Ny, i in 1:Nx
                mass_flux[i, j, 2] =  0.005
                mass_flux[i, j, 3] = -0.003
                mass_flux[i, j, 4] =  0.008
            end
            met = (; conv_mass_flux = mass_flux)

            mass_before = sum(Δz(k, grid) * c[i, j, k]
                              for i in 1:Nx, j in 1:Ny, k in 1:Nz)

            convect!(tracers, met, grid, conv, Δt)

            mass_after = sum(Δz(k, grid) * tracers.c[i, j, k]
                             for i in 1:Nx, j in 1:Ny, k in 1:Nz)

            @test mass_after ≈ mass_before rtol = 1e-10
        end

        @testset "Upward flux moves tracer upward" begin
            # Tracer concentrated near the surface (layer Nz = 5)
            c = zeros(Nx, Ny, Nz)
            for j in 1:Ny, i in 1:Nx
                c[i, j, Nz] = 1.0
            end
            tracers = (; c)

            # Uniform upward mass flux at interior interfaces
            mass_flux = zeros(Nx, Ny, Nz + 1)
            for j in 1:Ny, i in 1:Nx
                for k in 2:Nz
                    mass_flux[i, j, k] = 0.01
                end
            end
            met = (; conv_mass_flux = mass_flux)

            # Apply convection for several steps
            for _ in 1:10
                convect!(tracers, met, grid, conv, Δt)
            end

            # Tracer should have moved upward: decrease at surface, appear above
            surface_avg = sum(tracers.c[:, :, Nz]) / (Nx * Ny)
            upper_avg   = sum(tracers.c[:, :, Nz - 1]) / (Nx * Ny)

            @test surface_avg < 1.0   # tracer decreased at surface
            @test upper_avg > 0.0     # tracer appeared in layer above
        end

        @testset "Adjoint identity (exact discrete transpose)" begin
            # dot(L^T λ, δq) ≈ dot(λ, L δq) to machine precision
            # since convection is linear in tracers

            # Non-trivial mass flux profile with both up and down
            mass_flux = zeros(Nx, Ny, Nz + 1)
            for j in 1:Ny, i in 1:Nx
                mass_flux[i, j, 2] =  0.008
                mass_flux[i, j, 3] = -0.003
                mass_flux[i, j, 4] =  0.005
            end
            met = (; conv_mass_flux = mass_flux)

            δq = randn(Nx, Ny, Nz) .* 0.01
            λ  = randn(Nx, Ny, Nz)

            # Forward: L * δq
            fwd = (; c = copy(δq))
            convect!(fwd, met, grid, conv, Δt)

            # Adjoint: L^T * λ
            adj = (; c = copy(λ))
            adjoint_convect!(adj, met, grid, conv, Δt)

            lhs = dot(adj.c, δq)
            rhs = dot(λ, fwd.c)
            @test lhs ≈ rhs rtol = 1e-12
        end

        @testset "Adjoint no-op when met is nothing" begin
            λ = rand(Nx, Ny, Nz)
            λ_orig = copy(λ)
            adj = (; c = λ)
            adjoint_convect!(adj, nothing, grid, conv, Δt)
            @test adj.c == λ_orig
        end

        @testset "Multiple tracers are all updated" begin
            a = rand(Nx, Ny, Nz) .+ 0.1
            b = rand(Nx, Ny, Nz) .+ 0.5
            a_orig = copy(a)
            b_orig = copy(b)
            tracers = (; a, b)

            mass_flux = zeros(Nx, Ny, Nz + 1)
            for j in 1:Ny, i in 1:Nx
                for k in 2:Nz
                    mass_flux[i, j, k] = 0.01
                end
            end
            met = (; conv_mass_flux = mass_flux)

            convect!(tracers, met, grid, conv, Δt)

            # Both tracers should have changed
            @test !(tracers.a ≈ a_orig)
            @test !(tracers.b ≈ b_orig)

            # Both should conserve mass
            mass_a_before = sum(Δz(k, grid) * a_orig[i, j, k]
                                for i in 1:Nx, j in 1:Ny, k in 1:Nz)
            mass_a_after  = sum(Δz(k, grid) * tracers.a[i, j, k]
                                for i in 1:Nx, j in 1:Ny, k in 1:Nz)
            @test mass_a_after ≈ mass_a_before rtol = 1e-10

            mass_b_before = sum(Δz(k, grid) * b_orig[i, j, k]
                                for i in 1:Nx, j in 1:Ny, k in 1:Nz)
            mass_b_after  = sum(Δz(k, grid) * tracers.b[i, j, k]
                                for i in 1:Nx, j in 1:Ny, k in 1:Nz)
            @test mass_b_after ≈ mass_b_before rtol = 1e-10
        end
    end
end
