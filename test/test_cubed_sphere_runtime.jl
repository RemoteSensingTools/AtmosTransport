#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

function write_driven_cs_binary(path::AbstractString;
                                FT::Type{<:AbstractFloat} = Float64,
                                Nc::Int = 4,
                                Nz::Int = 2,
                                window_mass_scales::Tuple{Vararg{Real}} = (1, 1),
                                convection_windows = nothing,
                                dtrain_windows = nothing)
    convection_windows !== nothing &&
        length(convection_windows) == length(window_mass_scales) ||
        convection_windows === nothing || throw(ArgumentError("convection_windows length must match window_mass_scales"))
    dtrain_windows !== nothing &&
        length(dtrain_windows) == length(window_mass_scales) ||
        dtrain_windows === nothing || throw(ArgumentError("dtrain_windows length must match window_mass_scales"))
    vc = if Nz == 2
        HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    elseif Nz == 5
        HybridSigmaPressure(FT[0, 100, 300, 600, 1000, 2000],
                            FT[0, 0, 0.1, 0.3, 0.7, 1])
    else
        HybridSigmaPressure(FT.(collect(range(0, stop = 2000, length = Nz + 1))),
                            FT.(collect(range(0, stop = 1, length = Nz + 1))))
    end
    writer = AtmosTransport.MetDrivers.open_streaming_cs_transport_binary(
        path, Nc, 6, Nz, length(window_mass_scales), vc;
        FT = FT,
        dt_met_seconds = 3600.0,
        steps_per_window = 2,
        mass_basis = :dry,
        include_cmfmc = convection_windows !== nothing,
        include_dtrain = dtrain_windows !== nothing,
    )

    for (win, scale) in enumerate(window_mass_scales)
        window = (
            m  = ntuple(_ -> fill(FT(scale), Nc, Nc, Nz), 6),
            am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), 6),
            bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), 6),
            cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6),
            ps = ntuple(_ -> fill(FT(90000), Nc, Nc), 6),
        )
        if convection_windows !== nothing
            window = merge(window, (cmfmc = convection_windows[win],))
        end
        if dtrain_windows !== nothing
            window = merge(window, (dtrain = dtrain_windows[win],))
        end
        AtmosTransport.MetDrivers.write_streaming_cs_window!(writer, window, Nc, 6)
    end

    AtmosTransport.MetDrivers.close_streaming_transport_binary!(writer)
    return nothing
end

function make_cs_cmfmc_panels(FT::Type{<:AbstractFloat}, Nc::Int, Nz::Int;
                              peak = FT(0.02),
                              top_detrain = FT(0.01))
    cmfmc = ntuple(6) do _
        arr = zeros(FT, Nc, Nc, Nz + 1)
        if Nz >= 5
            arr[:, :, 4] .= peak * FT(0.5)
            arr[:, :, 3] .= peak
            arr[:, :, 2] .= peak * FT(0.5)
        else
            arr[:, :, min(2, Nz)] .= peak
        end
        arr
    end
    dtrain = ntuple(6) do _
        arr = zeros(FT, Nc, Nc, Nz)
        arr[:, :, 1] .= top_detrain
        arr
    end
    return cmfmc, dtrain
end

@testset "CubedSphere transport driver + DrivenSimulation" begin
    mktemp() do path, io
        close(io)
        write_driven_cs_binary(path; FT=Float64, window_mass_scales=(1, 1))

        reader = CubedSphereBinaryReader(path; FT=Float64)
        @test grid_type(reader) == :cubed_sphere
        @test horizontal_topology(reader) == :structureddirectional
        @test window_count(reader) == 2
        @test mass_basis(reader) == :dry
        @test !has_qv(reader)
        @test !has_flux_delta(reader)
        @test !has_cmfmc(reader)

        grid = load_grid(reader; FT=Float64, arch=CPU(), Hp=1)
        @test grid.horizontal isa CubedSphereMesh
        @test grid.horizontal.Nc == 4
        @test grid.horizontal.Hp == 1

        close(reader)

        driver = CubedSphereTransportDriver(path; FT=Float64, arch=CPU(), Hp=1)
        @test total_windows(driver) == 2
        @test window_dt(driver) == 3600.0
        @test steps_per_window(driver) == 2
        @test air_mass_basis(driver) == :dry

        window = load_transport_window(driver, 1)
        @test window isa CubedSphereTransportWindow{DryBasis}
        @test size(window.air_mass[1]) == (6, 6, 2)
        @test size(window.fluxes.am[1]) == (7, 6, 2)
        @test size(window.fluxes.bm[1]) == (6, 7, 2)
        @test size(window.fluxes.cm[1]) == (6, 6, 3)
        @test !has_humidity_endpoints(window)

        mesh = driver_grid(driver).horizontal
        tracer_panels = ntuple(p -> window.air_mass[p] .* 400e-6, 6)
        state = CubedSphereState(DryBasis, mesh, window.air_mass; CO2 = tracer_panels)
        fluxes = allocate_face_fluxes(mesh, 2; FT=Float64, basis=DryBasis)
        model = @inferred TransportModel(state, fluxes, driver_grid(driver), UpwindScheme())
        sim = DrivenSimulation(model, driver; start_window=1, stop_window=2)

        @test sim.interpolate_fluxes_within_window == false
        @test sim.Δt == 1800.0
        @test sim.steps_per_window == 2
        @test window_index(sim) == 1
        @test current_qv(sim) === nothing

        m0 = total_air_mass(sim.model.state)
        rm0 = total_mass(sim.model.state, :CO2)

        run!(sim)

        @test sim.iteration == 4
        @test sim.time == 7200.0
        @test window_index(sim) == 2
        @test total_air_mass(sim.model.state) ≈ m0 atol=eps(Float64) * m0 * 10
        @test total_mass(sim.model.state, :CO2) ≈ rm0 atol=eps(Float64) * rm0 * 10

        close(driver)
    end
end

@testset "CubedSphere runtime supports CMFMC convection without promoting to Float64" begin
    FT = Float32
    Nc, Nz = 4, 5
    cmfmc, dtrain = make_cs_cmfmc_panels(FT, Nc, Nz)

    mktemp() do path, io
        close(io)
        write_driven_cs_binary(path;
                               FT = FT,
                               Nc = Nc,
                               Nz = Nz,
                               window_mass_scales = (FT(1e16),),
                               convection_windows = (cmfmc,),
                               dtrain_windows = (dtrain,))

        reader = CubedSphereBinaryReader(path; FT = FT)
        @test has_cmfmc(reader)
        close(reader)

        driver = CubedSphereTransportDriver(path; FT = FT, arch = CPU(), Hp = 1)
        window = load_transport_window(driver, 1)
        mesh = driver_grid(driver).horizontal
        @test window.convection !== nothing
        @test eltype(window.convection.cmfmc[1]) === FT

        tracer_panels = ntuple(6) do p
            rm = zeros(FT, size(window.air_mass[p]))
            @views rm[mesh.Hp + 1:mesh.Hp + mesh.Nc, mesh.Hp + 1:mesh.Hp + mesh.Nc, Nz] .=
                FT(1e-6) .* window.air_mass[p][mesh.Hp + 1:mesh.Hp + mesh.Nc, mesh.Hp + 1:mesh.Hp + mesh.Nc, Nz]
            rm
        end

        state = CubedSphereState(DryBasis, mesh, window.air_mass; CO2 = tracer_panels)
        fluxes = allocate_face_fluxes(mesh, Nz; FT = FT, basis = DryBasis)
        model = TransportModel(state, fluxes, driver_grid(driver), UpwindScheme();
                               convection = CMFMCConvection())
        sim = DrivenSimulation(model, driver; start_window = 1, stop_window = 1)

        @test typeof(sim.Δt) === FT
        @test eltype(sim.model.convection_forcing.cmfmc[1]) === FT

        rm0 = total_mass(sim.model.state, :CO2)
        run!(sim)

        @test sim.iteration == 2
        @test total_mass(sim.model.state, :CO2) ≈ rm0 rtol = 1f-5
        for p in 1:6
            panel = get_tracer(sim.model.state, :CO2)[p]
            @test any(panel[mesh.Hp + 1:mesh.Hp + mesh.Nc, mesh.Hp + 1:mesh.Hp + mesh.Nc, 1] .> 0)
        end

        close(driver)
    end
end

@testset "CubedSphere runtime supports diffusion plus surface sources" begin
    mktemp() do path, io
        close(io)
        write_driven_cs_binary(path; FT=Float64, window_mass_scales=(1,))

        driver = CubedSphereTransportDriver(path; FT=Float64, arch=CPU(), Hp=1)
        window = load_transport_window(driver, 1)
        mesh = driver_grid(driver).horizontal

        tracer_panels = ntuple(6) do p
            rm = zeros(Float64, size(window.air_mass[p]))
            @views rm[mesh.Hp + 1:mesh.Hp + mesh.Nc, mesh.Hp + 1:mesh.Hp + mesh.Nc, 2] .= 100.0
            rm
        end

        state = CubedSphereState(DryBasis, mesh, window.air_mass; CO2=tracer_panels)
        fluxes = allocate_face_fluxes(mesh, 2; FT=Float64, basis=DryBasis)
        kz = CubedSphereField(ntuple(_ -> ConstantField{Float64, 3}(1.0), 6))
        diffusion = ImplicitVerticalDiffusion(; kz_field=kz)
        model = TransportModel(state, fluxes, driver_grid(driver), UpwindScheme();
                               diffusion=diffusion)
        for p in 1:6
            fill!(model.workspace.dz_scratch[p], 100.0)
        end

        source = SurfaceFluxSource(:CO2, ntuple(_ -> fill(2.0, mesh.Nc, mesh.Nc), 6))
        sim = DrivenSimulation(model, driver;
                               start_window=1,
                               stop_window=1,
                               surface_sources=(source,))

        m0 = total_mass(sim.model.state, :CO2)
        run!(sim)

        @test sim.iteration == 2
        @test sim.time == 3600.0
        for p in 1:6
            panel = get_tracer(sim.model.state, :CO2)[p]
            @test all(panel[mesh.Hp + 1:mesh.Hp + mesh.Nc, mesh.Hp + 1:mesh.Hp + mesh.Nc, 1] .> 0.0)
        end
        @test total_mass(sim.model.state, :CO2) ≈ m0 + 6 * mesh.Nc * mesh.Nc * 2.0 * 1800.0 * 2 rtol=1e-12

        close(driver)
    end
end
