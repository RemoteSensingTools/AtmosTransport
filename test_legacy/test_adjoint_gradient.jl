using Test
using AtmosTransport.Adjoint
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection
using AtmosTransport.Convection
using AtmosTransport.Diffusion
using AtmosTransport.Chemistry
using AtmosTransport.TimeSteppers

@testset "Adjoint infrastructure" begin
    @test StoreAllCheckpointer() isa AbstractCheckpointer
    @test RevolveCheckpointer() isa AbstractCheckpointer
    @test RevolveCheckpointer(n_snapshots=20) isa AbstractCheckpointer
end

@testset "run_adjoint! with StoreAllCheckpointer" begin
    using AtmosTransport.Grids: grid_size, floattype
    using AtmosTransport.TimeSteppers: Clock
    using LinearAlgebra: dot

    vc = HybridSigmaPressure(
        [0.0, 5000.0, 10000.0, 20000.0, 50000.0, 101325.0],
        [0.0, 0.0, 0.1, 0.3, 0.7, 1.0])
    grid = LatitudeLongitudeGrid(CPU();
        size = (8, 4, 5),
        longitude = (-180, 180),
        latitude = (-90, 90),
        vertical = vc)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    FT = floattype(grid)

    ts = OperatorSplittingTimeStepper(
        advection  = SlopesAdvection(use_limiter=false),
        convection = NoConvection(),
        diffusion  = NoDiffusion(),
        chemistry  = NoChemistry(),
        Δt_outer   = 900.0)

    u = fill(5.0, Nx + 1, Ny, Nz)
    v = zeros(Nx, Ny + 1, Nz)
    w = zeros(Nx, Ny, Nz + 1)
    met = (; u, v, w)

    c_init = randn(FT, Nx, Ny, Nz) .* FT(0.01) .+ FT(1.0)
    tracers = (; c = copy(c_init))
    adj_tracers = (; c = similar(c_init))
    clock = Clock(FT; Δt = FT(900.0))
    model = (; tracers, adj_tracers, met_data = met, grid, timestepper = ts, clock)

    # J = 0.5 * ||c_final||² => ∂J/∂c_final = c_final
    cost_gradient_fn = tracers -> (; (k => copy(v) for (k, v) in pairs(tracers))...)

    grad = run_adjoint!(model, met, StoreAllCheckpointer(), 3, 900.0; cost_gradient_fn)

    @test grad isa NamedTuple
    @test haskey(grad, :c)
    @test size(grad.c) == (Nx, Ny, Nz)
    @test !all(iszero, grad.c)
end

@testset "Gradient test — full operator splitting" begin
    vc = HybridSigmaPressure(
        [0.0, 5000.0, 10000.0, 20000.0, 50000.0, 101325.0],
        [0.0, 0.0, 0.1, 0.3, 0.7, 1.0])
    grid = LatitudeLongitudeGrid(CPU();
        size = (8, 4, 5),
        longitude = (-180, 180),
        latitude = (-90, 90),
        vertical = vc)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    @testset "Advection only (no limiter, exact adjoint)" begin
        ts = OperatorSplittingTimeStepper(
            advection  = SlopesAdvection(use_limiter=false),
            convection = NoConvection(),
            diffusion  = NoDiffusion(),
            chemistry  = NoChemistry(),
            Δt_outer   = 900.0)

        # Constant eastward wind
        u = fill(5.0, Nx + 1, Ny, Nz)
        v = zeros(Nx, Ny + 1, Nz)
        w = zeros(Nx, Ny, Nz + 1)
        met = (; u, v, w)

        results = gradient_test(; grid, timestepper=ts, met_data=met,
                                  n_steps=3, Δt=900.0)
        # Exact discrete adjoint: ratio should be 1.0 to machine precision
        for (ε, ratio) in results
            if ε >= 1e-6
                @test abs(ratio - 1.0) < 1e-5
            end
        end
    end

    @testset "Advection + Convection (no limiter)" begin
        ts = OperatorSplittingTimeStepper(
            advection  = SlopesAdvection(use_limiter=false),
            convection = TiedtkeConvection(),
            diffusion  = NoDiffusion(),
            chemistry  = NoChemistry(),
            Δt_outer   = 900.0)

        u = fill(3.0, Nx + 1, Ny, Nz)
        v = zeros(Nx, Ny + 1, Nz)
        w = zeros(Nx, Ny, Nz + 1)

        # Small upward convective mass flux
        mf = zeros(Nx, Ny, Nz + 1)
        for j in 1:Ny, i in 1:Nx
            mf[i, j, 2] =  0.005
            mf[i, j, 3] = -0.002
            mf[i, j, 4] =  0.003
        end
        met = (; u, v, w, conv_mass_flux = mf)

        results = gradient_test(; grid, timestepper=ts, met_data=met,
                                  n_steps=2, Δt=900.0)
        for (ε, ratio) in results
            if ε >= 1e-6
                # Convection subcycling introduces more numerical noise
                @test abs(ratio - 1.0) < 1e-3
            end
        end
    end

    @testset "Advection + Diffusion (no limiter)" begin
        ts = OperatorSplittingTimeStepper(
            advection  = SlopesAdvection(use_limiter=false),
            convection = NoConvection(),
            diffusion  = BoundaryLayerDiffusion(Kz_max=50.0),
            chemistry  = NoChemistry(),
            Δt_outer   = 900.0)

        u = fill(2.0, Nx + 1, Ny, Nz)
        v = zeros(Nx, Ny + 1, Nz)
        w = zeros(Nx, Ny, Nz + 1)
        met = (; u, v, w)

        results = gradient_test(; grid, timestepper=ts, met_data=met,
                                  n_steps=2, Δt=900.0)
        for (ε, ratio) in results
            if ε >= 1e-6
                @test abs(ratio - 1.0) < 1e-5
            end
        end
    end

    @testset "Full physics (advection + convection + diffusion, no limiter)" begin
        ts = OperatorSplittingTimeStepper(
            advection  = SlopesAdvection(use_limiter=false),
            convection = TiedtkeConvection(),
            diffusion  = BoundaryLayerDiffusion(Kz_max=50.0),
            chemistry  = NoChemistry(),
            Δt_outer   = 900.0)

        u = fill(2.0, Nx + 1, Ny, Nz)
        v = fill(-1.0, Nx, Ny + 1, Nz)
        w = zeros(Nx, Ny, Nz + 1)

        mf = zeros(Nx, Ny, Nz + 1)
        for j in 1:Ny, i in 1:Nx
            mf[i, j, 2] = 0.003
            mf[i, j, 3] = 0.001
        end
        met = (; u, v, w, conv_mass_flux = mf)

        results = gradient_test(; grid, timestepper=ts, met_data=met,
                                  n_steps=3, Δt=900.0, verbose=false)
        for (ε, ratio) in results
            if ε >= 1e-5
                @test abs(ratio - 1.0) < 1e-4
            end
        end
    end

    @testset "With limiter (approximate, TM5 continuous adjoint)" begin
        ts = OperatorSplittingTimeStepper(
            advection  = SlopesAdvection(use_limiter=true),
            convection = TiedtkeConvection(),
            diffusion  = BoundaryLayerDiffusion(Kz_max=50.0),
            chemistry  = NoChemistry(),
            Δt_outer   = 900.0)

        u = fill(2.0, Nx + 1, Ny, Nz)
        v = zeros(Nx, Ny + 1, Nz)
        w = zeros(Nx, Ny, Nz + 1)

        mf = zeros(Nx, Ny, Nz + 1)
        for j in 1:Ny, i in 1:Nx
            mf[i, j, 2] = 0.002
        end
        met = (; u, v, w, conv_mass_flux = mf)

        results = gradient_test(; grid, timestepper=ts, met_data=met,
                                  n_steps=2, Δt=900.0)
        # Continuous adjoint: approximate, but ratio should be in a reasonable range
        for (ε, ratio) in results
            if ε >= 1e-4
                @test abs(ratio - 1.0) < 0.5
            end
        end
    end
end
