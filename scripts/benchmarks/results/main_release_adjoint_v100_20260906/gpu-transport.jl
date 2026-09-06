using CUDA, Test, AtmosTransport
CUDA.functional(true) || error("CUDA unavailable")
@assert length(collect(CUDA.devices())) == 1
@assert occursin("V100", CUDA.name(CUDA.device()))
CUDA.allowscalar(false)
# Reuse the committed fixture definitions without rerunning their testsets.
Base.include(ex -> ex isa Expr && ex.head == :macrocall && ex.args[1] == Symbol("@testset") ? nothing : ex,
    @__MODULE__, joinpath(@__DIR__, "test/core/test_cs_ppm_adjoint_footprint.jl"))
@testset "V100 transporting footprint gradients" begin
    for FT in (Float64, Float32), scheme in (AT.PPMScheme(AT.NoLimiter()), AT.PPMScheme(), AT.LinRoodPPMScheme(5), AT.LinRoodPPMScheme(7))
        @testset "$FT $scheme" begin
            mesh, m, rm, am, bm, cm = _transport_cs_problem(; Nc=4, Nz=4, nsteps=3, FT)
            _fill_smooth_tracer!(rm, m, mesh)
            rates = [ntuple(p -> [FT(sin(0.31s+0.17p+0.23i-0.19j)) for i in 1:mesh.Nc, j in 1:mesh.Nc], 6) for s in 1:3]
            gm, grm = map(CuArray, m), map(CuArray, rm)
            gam, gbm, gcm, grates = map(_to_gpu_steps, (am,bm,cm,rates))
            obj = AT.CSColumnMeanObjective(1,1,1)
            dt = FT(1.5)
            cpu = AT.cs_surface_emission_footprint(rm,m,am,bm,cm,mesh,obj; scheme,dt)
            gpu = AT.cs_surface_emission_footprint(grm,gm,gam,gbm,gcm,mesh,obj; scheme,dt)
            predicted = _dot_footprint(gpu,grates)
            tolerance = FT == Float64 ? 1e-10 : 2e-5
            for s in 1:3, p in 1:6
                @test gpu.footprints[s][p] isa CuArray
                @test isapprox(Array(gpu.footprints[s][p]),cpu.footprints[s][p]; rtol=tolerance,atol=tolerance*1e-3)
            end
            if FT == Float64
                eps_dir = FT(2e-6)
                jp = AT.run_cs_footprint_forward(grm,gm,gam,gbm,gcm,mesh,obj;scheme,dt,emission_rates=_scaled_rates(grates,eps_dir))
                jm = AT.run_cs_footprint_forward(grm,gm,gam,gbm,gcm,mesh,obj;scheme,dt,emission_rates=_scaled_rates(grates,-eps_dir))
                fd = (jp-jm)/(2eps_dir)
                println("GRADIENT ",FT," ",scheme," predicted=",predicted," fd=",fd," relative_error=",abs((predicted-fd)/fd))
                @test isapprox(predicted,fd;rtol=1e-6,atol=1e-9)
            end
            for checkpoint in (AT.StrideCheckpoint(2),AT.RevolveCheckpoint())
                replay = AT.cs_surface_emission_footprint(grm,gm,gam,gbm,gcm,mesh,obj; scheme,dt,checkpoint)
                for s in 1:3, p in 1:6
                    @test isapprox(Array(replay.footprints[s][p]),Array(gpu.footprints[s][p]);rtol=tolerance,atol=tolerance*1e-3)
                end
            end
            CUDA.synchronize()
            println("COMPLETED ",FT," ",scheme)
            flush(stdout)
        end
    end
end
println("GPU_TRANSPORT_GRADIENTS_PASSED")
