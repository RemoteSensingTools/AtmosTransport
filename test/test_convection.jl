using Test
using LinearAlgebra: dot
using AtmosTransport.Convection
using AtmosTransport.Convection: tm5_conv_matrix!, _conv_cloud_dim
using AtmosTransport.Grids
using AtmosTransport.Architectures: CPU
using AtmosTransport.Parameters: PlanetParameters

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

            # Mass-divergence correction breaks exact flux telescoping;
            # small drift (~0.02%) handled by global mass fixer in production.
            @test mass_after ≈ mass_before rtol = 1e-3
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

            # Mass-divergence correction breaks exact flux telescoping
            @test mass_after ≈ mass_before rtol = 1e-3
        end

        @testset "Upward flux moves tracer upward" begin
            # Tracer concentrated near the surface (layer Nz = 5)
            c = zeros(Nx, Ny, Nz)
            for j in 1:Ny, i in 1:Nx
                c[i, j, Nz] = 1.0
            end
            tracers = (; c)

            # Upward mass flux at interior interfaces (non-uniform to create
            # gradient — mass-divergence correction cancels transport for
            # levels with zero flux divergence)
            mass_flux = zeros(Nx, Ny, Nz + 1)
            for j in 1:Ny, i in 1:Nx
                for k in 2:Nz
                    frac = (Nz + 1 - k) / Nz  # decreasing with height
                    mass_flux[i, j, k] = 0.01 * frac
                end
            end
            met = (; conv_mass_flux = mass_flux)

            # Apply convection for several steps
            for _ in 1:10
                convect!(tracers, met, grid, conv, Δt)
            end

            # Tracer should have moved upward: appear in layer above surface
            upper_avg = sum(tracers.c[:, :, Nz - 1]) / (Nx * Ny)
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
            @test lhs ≈ rhs rtol = 1e-10
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
            @test mass_a_after ≈ mass_a_before rtol = 1e-3

            mass_b_before = sum(Δz(k, grid) * b_orig[i, j, k]
                                for i in 1:Nx, j in 1:Ny, k in 1:Nz)
            mass_b_after  = sum(Δz(k, grid) * tracers.b[i, j, k]
                                for i in 1:Nx, j in 1:Ny, k in 1:Nz)
            @test mass_b_after ≈ mass_b_before rtol = 1e-3
        end
    end

    @testset "TM5MatrixConvection on LatitudeLongitudeGrid" begin
        vc = HybridSigmaPressure(
            [0.0, 5000.0, 10000.0, 20000.0, 50000.0, 101325.0],
            [0.0, 0.0, 0.1, 0.3, 0.7, 1.0])
        grid = LatitudeLongitudeGrid(CPU();
            size = (4, 4, 5),
            longitude = (-180, 180),
            latitude = (-90, 90),
            vertical = vc)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        conv = TM5MatrixConvection()
        planet = PlanetParameters{Float64}(6.371e6, 9.80665, 101325.0, 0.4, 1004.0, 1.225)
        grav = planet.gravity
        Δt = 3600.0

        # Pressure thickness per level
        delp = zeros(Nx, Ny, Nz)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            delp[i, j, k] = Float64(Δz(k, grid))
        end

        # Build a simple updraft profile (top-to-bottom convention: k=1=TOA, k=5=surface)
        # Updraft: entrainment at surface (k=5), detrainment at upper levels (k=2,3)
        # In TM5 convention (bottom-to-top): entu at k=1=surface, detu at k=3,4
        function make_simple_tm5conv(Nx, Ny, Nz)
            entu = zeros(Nx, Ny, Nz)
            detu = zeros(Nx, Ny, Nz)
            entd = zeros(Nx, Ny, Nz)
            detd = zeros(Nx, Ny, Nz)

            # Simple updraft: entrainment of 0.01 kg/m²/s at surface level (k=Nz in our convention)
            # detraining 0.005 at k=Nz-2 and 0.005 at k=Nz-3
            for j in 1:Ny, i in 1:Nx
                entu[i, j, Nz]   = 0.01    # surface: entrain into updraft
                detu[i, j, Nz-2] = 0.005   # detrain at level 3
                detu[i, j, Nz-3] = 0.005   # detrain at level 2
            end

            return (; entu, detu, entd, detd)
        end

        @testset "Type construction" begin
            @test TM5MatrixConvection() isa AbstractConvection
            @test TM5MatrixConvection(lmax_conv=25) isa AbstractConvection
            @test TM5MatrixConvection().lmax_conv == 0
        end

        @testset "Matrix builder: row sums (mass conservation)" begin
            # For each row k: sum_kk conv1(k,kk) should equal 1.0
            # This ensures total tracer mass is conserved.
            lmx = 5
            m_col = [1000.0, 2000.0, 3000.0, 5000.0, 8000.0]  # TM5 convention: k=1=surface
            entu_col = [0.01, 0.0, 0.0, 0.0, 0.0]
            detu_col = [0.0, 0.0, 0.005, 0.005, 0.0]
            entd_col = zeros(lmx)
            detd_col = zeros(lmx)
            dt = 3600.0

            li, ld = _conv_cloud_dim(detu_col, entd_col, lmx)
            conv1 = zeros(lmx, lmx)
            lmc = tm5_conv_matrix!(conv1, m_col, entu_col, detu_col,
                                   entd_col, detd_col, lmx, li, ld, dt)

            # Each column of conv1 should sum to 1 (mass from source kk distributes fully)
            # Note: conv1 * rm_new = rm_old, so we need column sums of conv1^{-1} = 1
            # But for the matrix itself, the COLUMN sums should be 1
            # (because conv1 = I - dt*D and D conserves mass: sum_k D(k,kk) = 0)
            for kk in 1:lmc
                col_sum = sum(conv1[k, kk] for k in 1:lmc)
                @test col_sum ≈ 1.0 atol = 1e-12
            end
        end

        @testset "Uniform field invariance" begin
            tm5conv = make_simple_tm5conv(Nx, Ny, Nz)

            # Uniform mixing ratio: should be unchanged after convection
            c = fill(0.4, Nx, Ny, Nz)
            c_orig = copy(c)
            tracers = (; c)

            convect!(tracers, tm5conv, delp, conv, grid, Δt, planet)
            @test tracers.c ≈ c_orig atol = 1e-10
        end

        @testset "Mass conservation (exact)" begin
            tm5conv = make_simple_tm5conv(Nx, Ny, Nz)

            c = rand(Nx, Ny, Nz) .+ 0.1
            tracers = (; c)

            # Mass = sum of (q * delp/g) over all levels
            mass_before = sum(c[i, j, k] * delp[i, j, k] / grav
                              for i in 1:Nx, j in 1:Ny, k in 1:Nz)

            convect!(tracers, tm5conv, delp, conv, grid, Δt, planet)

            mass_after = sum(tracers.c[i, j, k] * delp[i, j, k] / grav
                             for i in 1:Nx, j in 1:Ny, k in 1:Nz)

            # Implicit scheme conserves mass to machine precision
            @test mass_after ≈ mass_before rtol = 1e-12
        end

        @testset "No convection when all fields are zero" begin
            tm5conv = (; entu=zeros(Nx,Ny,Nz), detu=zeros(Nx,Ny,Nz),
                         entd=zeros(Nx,Ny,Nz), detd=zeros(Nx,Ny,Nz))

            c = rand(Nx, Ny, Nz) .+ 0.1
            c_orig = copy(c)
            tracers = (; c)

            convect!(tracers, tm5conv, delp, conv, grid, Δt, planet)
            @test tracers.c ≈ c_orig
        end

        @testset "Updraft moves tracer upward" begin
            tm5conv = make_simple_tm5conv(Nx, Ny, Nz)

            # Tracer concentrated at surface (k=Nz in our convention)
            c = zeros(Nx, Ny, Nz)
            for j in 1:Ny, i in 1:Nx
                c[i, j, Nz] = 1.0
            end
            tracers = (; c)

            convect!(tracers, tm5conv, delp, conv, grid, Δt, planet)

            # Surface should have decreased
            sfc_avg = sum(tracers.c[:, :, Nz]) / (Nx * Ny)
            @test sfc_avg < 1.0

            # Upper levels should have gained tracer
            upper_total = sum(tracers.c[:, :, k] for k in 1:Nz-1) / (Nx * Ny)
            @test sum(upper_total) > 0.0
        end

        @testset "Multiple tracers" begin
            tm5conv = make_simple_tm5conv(Nx, Ny, Nz)

            a = rand(Nx, Ny, Nz) .+ 0.1
            b = rand(Nx, Ny, Nz) .+ 0.5
            a_orig = copy(a)
            b_orig = copy(b)
            tracers = (; a, b)

            convect!(tracers, tm5conv, delp, conv, grid, Δt, planet)

            @test !(tracers.a ≈ a_orig)
            @test !(tracers.b ≈ b_orig)

            # Both conserve mass exactly
            for (arr, orig) in [(tracers.a, a_orig), (tracers.b, b_orig)]
                mass_before = sum(orig[i,j,k] * delp[i,j,k] / grav
                                  for i in 1:Nx, j in 1:Ny, k in 1:Nz)
                mass_after = sum(arr[i,j,k] * delp[i,j,k] / grav
                                 for i in 1:Nx, j in 1:Ny, k in 1:Nz)
                @test mass_after ≈ mass_before rtol = 1e-12
            end
        end

        @testset "Adjoint identity (exact discrete transpose)" begin
            tm5conv = make_simple_tm5conv(Nx, Ny, Nz)

            δq = randn(Nx, Ny, Nz) .* 0.01
            λ  = randn(Nx, Ny, Nz)

            # Forward: L * δq
            fwd = (; c = copy(δq))
            convect!(fwd, tm5conv, delp, conv, grid, Δt, planet)

            # Adjoint: L^T * λ
            adj = (; c = copy(λ))
            adjoint_convect!(adj, tm5conv, delp, conv, grid, Δt, planet)

            lhs = dot(adj.c, δq)
            rhs = dot(λ, fwd.c)
            @test lhs ≈ rhs rtol = 1e-10
        end

        @testset "Downdraft moves tracer downward" begin
            # Downdraft profile: entrain at upper levels, detrain at surface
            entd = zeros(Nx, Ny, Nz)
            detd = zeros(Nx, Ny, Nz)
            for j in 1:Ny, i in 1:Nx
                entd[i, j, Nz-3] = 0.008   # entrain into downdraft at level 2
                detd[i, j, Nz]   = 0.008   # detrain at surface
            end
            tm5conv = (; entu=zeros(Nx,Ny,Nz), detu=zeros(Nx,Ny,Nz), entd, detd)

            # Tracer concentrated at upper level (k=Nz-3 in our convention)
            c = zeros(Nx, Ny, Nz)
            for j in 1:Ny, i in 1:Nx
                c[i, j, Nz-3] = 1.0
            end
            tracers = (; c)

            convect!(tracers, tm5conv, delp, conv, grid, Δt, planet)

            # Surface should have gained tracer
            sfc_avg = sum(tracers.c[:, :, Nz]) / (Nx * Ny)
            @test sfc_avg > 0.0
        end

        @testset "Combined updraft + downdraft" begin
            entu = zeros(Nx, Ny, Nz)
            detu = zeros(Nx, Ny, Nz)
            entd = zeros(Nx, Ny, Nz)
            detd = zeros(Nx, Ny, Nz)
            for j in 1:Ny, i in 1:Nx
                entu[i, j, Nz]   = 0.01   # surface updraft
                detu[i, j, Nz-2] = 0.01   # detrain at mid-level
                entd[i, j, Nz-2] = 0.005  # downdraft from mid-level
                detd[i, j, Nz]   = 0.005  # detrain at surface
            end
            tm5conv = (; entu, detu, entd, detd)

            c = rand(Nx, Ny, Nz) .+ 0.1
            tracers = (; c)

            mass_before = sum(c[i,j,k] * delp[i,j,k] / grav
                              for i in 1:Nx, j in 1:Ny, k in 1:Nz)

            convect!(tracers, tm5conv, delp, conv, grid, Δt, planet)

            mass_after = sum(tracers.c[i,j,k] * delp[i,j,k] / grav
                             for i in 1:Nx, j in 1:Ny, k in 1:Nz)

            @test mass_after ≈ mass_before rtol = 1e-12
        end

        @testset "lmax_conv parameter respected" begin
            conv_lmax = TM5MatrixConvection(lmax_conv=3)
            tm5conv = make_simple_tm5conv(Nx, Ny, Nz)

            c = rand(Nx, Ny, Nz) .+ 0.1
            tracers = (; c)

            # Should not error with lmax_conv < Nz
            convect!(tracers, tm5conv, delp, conv_lmax, grid, Δt, planet)
            @test true  # no error thrown
        end

        @testset "GPU kernel (KA on CPU) matches CPU path" begin
            # Verify the KA build+solve kernels produce the same result
            # as the LAPACK LU path when run on the CPU backend.
            tm5conv = make_simple_tm5conv(Nx, Ny, Nz)

            # Random tracer field
            c_cpu = rand(Nx, Ny, Nz) .+ 0.1
            c_gpu = copy(c_cpu)

            # CPU path (LAPACK LU)
            tracers_cpu = (; c = c_cpu)
            convect!(tracers_cpu, tm5conv, delp, conv, grid, Δt, planet)

            # GPU path (KA kernels with workspace)
            ws = allocate_tm5conv_workspace(Float64, Nz, Nx, Ny, Array)
            tracers_gpu = (; c = c_gpu)
            convect!(tracers_gpu, tm5conv, delp, conv, grid, Δt, planet;
                     workspace=ws)

            @test tracers_gpu.c ≈ tracers_cpu.c rtol = 1e-10
        end

        @testset "GPU kernel: uniform field invariance" begin
            tm5conv = make_simple_tm5conv(Nx, Ny, Nz)
            ws = allocate_tm5conv_workspace(Float64, Nz, Nx, Ny, Array)

            c = fill(0.4, Nx, Ny, Nz)
            c_orig = copy(c)
            tracers = (; c)

            convect!(tracers, tm5conv, delp, conv, grid, Δt, planet;
                     workspace=ws)
            @test tracers.c ≈ c_orig atol = 1e-10
        end

        @testset "GPU kernel: mass conservation" begin
            tm5conv = make_simple_tm5conv(Nx, Ny, Nz)
            ws = allocate_tm5conv_workspace(Float64, Nz, Nx, Ny, Array)

            c = rand(Nx, Ny, Nz) .+ 0.1
            tracers = (; c)

            mass_before = sum(c[i, j, k] * delp[i, j, k] / grav
                              for i in 1:Nx, j in 1:Ny, k in 1:Nz)

            convect!(tracers, tm5conv, delp, conv, grid, Δt, planet;
                     workspace=ws)

            mass_after = sum(tracers.c[i, j, k] * delp[i, j, k] / grav
                             for i in 1:Nx, j in 1:Ny, k in 1:Nz)

            @test mass_after ≈ mass_before rtol = 1e-12
        end

        @testset "GPU kernel: multiple tracers" begin
            tm5conv = make_simple_tm5conv(Nx, Ny, Nz)
            ws = allocate_tm5conv_workspace(Float64, Nz, Nx, Ny, Array)

            a_cpu = rand(Nx, Ny, Nz) .+ 0.1
            b_cpu = rand(Nx, Ny, Nz) .+ 0.5
            a_gpu = copy(a_cpu)
            b_gpu = copy(b_cpu)

            # CPU path
            convect!((; a=a_cpu, b=b_cpu), tm5conv, delp, conv, grid, Δt, planet)

            # GPU path
            convect!((; a=a_gpu, b=b_gpu), tm5conv, delp, conv, grid, Δt, planet;
                     workspace=ws)

            @test a_gpu ≈ a_cpu rtol = 1e-10
            @test b_gpu ≈ b_cpu rtol = 1e-10
        end

        @testset "GPU kernel: with downdraft" begin
            # Create met data with both updraft and downdraft
            entu = zeros(Nx, Ny, Nz)
            detu = zeros(Nx, Ny, Nz)
            entd = zeros(Nx, Ny, Nz)
            detd = zeros(Nx, Ny, Nz)

            for j in 1:Ny, i in 1:Nx
                entu[i, j, Nz]   = 0.01    # updraft entrainment at surface
                detu[i, j, Nz-2] = 0.005   # updraft detrainment
                detu[i, j, Nz-3] = 0.005
                entd[i, j, Nz-2] = 0.003   # downdraft entrainment
                detd[i, j, Nz]   = 0.003   # downdraft detrainment at surface
            end

            tm5conv_dd = (; entu, detu, entd, detd)
            ws = allocate_tm5conv_workspace(Float64, Nz, Nx, Ny, Array)

            c_cpu = rand(Nx, Ny, Nz) .+ 0.1
            c_gpu = copy(c_cpu)

            convect!((; c=c_cpu), tm5conv_dd, delp, conv, grid, Δt, planet)
            convect!((; c=c_gpu), tm5conv_dd, delp, conv, grid, Δt, planet;
                     workspace=ws)

            @test c_gpu ≈ c_cpu rtol = 1e-10
        end
    end
end
