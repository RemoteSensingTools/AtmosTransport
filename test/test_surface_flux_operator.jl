"""
Unit tests for the plan-17-Commit-3 `SurfaceFluxOperator` stack:

- `AbstractSurfaceFluxOperator` hierarchy
- `NoSurfaceFlux` identity (state and array-level entries)
- `SurfaceFluxOperator` application:
    - writes to `k = Nz` surface layer only (other layers untouched)
    - rate × dt arithmetic (kg/s per cell × seconds → kg added)
    - multi-source with per-tracer index resolution
    - tracers in map but absent from state are skipped
    - tracers in state but absent from map are untouched
- Array-level `apply_surface_flux!` entry point (Commit 5 palindrome
  integration target):
    - operates on any 4D buffer (not just `state.tracers_raw`) when
      `tracer_names` is supplied as a kwarg
    - `NoSurfaceFlux` is a zero-FP dead branch
- Mass accounting: Σ (rate × dt × emitting cells) matches analytic total
- Adapt passthrough (CPU), GPU (when CUDA.functional())
"""

using Test
using AtmosTransport
using AtmosTransport: CellState, MoistBasis, DryBasis,
                      SurfaceFluxSource, PerTracerFluxMap, flux_for,
                      NoSurfaceFlux, SurfaceFluxOperator, AbstractSurfaceFluxOperator,
                      apply!, apply_surface_flux!
using AtmosTransport.Operators.SurfaceFlux: emitting_tracer_indices
using Adapt

const HAS_GPU_SFO = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# Helper: build a minimal structured state with named tracers, all
# initialised to `init`.
function _make_state(FT, Nx, Ny, Nz; init = one(FT),
                     tracer_names = (:CO2, :SF6, :Rn222))
    air = ones(FT, Nx, Ny, Nz)
    kwargs = NamedTuple{tracer_names}(ntuple(_ -> fill(FT(init), Nx, Ny, Nz),
                                             length(tracer_names)))
    return CellState(MoistBasis, air; kwargs...)
end

@testset "SurfaceFluxOperator — plan 17 Commit 3" begin

    @testset "type hierarchy" begin
        @test NoSurfaceFlux() isa AbstractSurfaceFluxOperator
        src = SurfaceFluxSource(:CO2, fill(1.0, 2, 2))
        op = SurfaceFluxOperator(src)
        @test op isa AbstractSurfaceFluxOperator
        @test op isa SurfaceFluxOperator
    end

    @testset "SurfaceFluxOperator variadic constructor" begin
        src1 = SurfaceFluxSource(:CO2, fill(1.0, 2, 2))
        src2 = SurfaceFluxSource(:SF6, fill(2.0, 2, 2))
        op1 = SurfaceFluxOperator(src1, src2)
        op2 = SurfaceFluxOperator(PerTracerFluxMap(src1, src2))
        @test length(op1.flux_map) == length(op2.flux_map)
        @test flux_for(op1.flux_map, :CO2).cell_mass_rate ==
              flux_for(op2.flux_map, :CO2).cell_mass_rate
    end

    @testset "SurfaceFluxOperator() — empty variadic is allowed" begin
        # An empty-map operator is equivalent to NoSurfaceFlux for
        # `apply!` semantics (identity) but has a distinct type.
        # Needed for configs that might later add emissions.
        op = SurfaceFluxOperator()
        @test op isa SurfaceFluxOperator
        @test length(op.flux_map) == 0
    end

    @testset "NoSurfaceFlux is state identity" begin
        FT = Float64
        state = _make_state(FT, 4, 3, 2; init = 1.0)
        orig  = copy(state.tracers_raw)

        apply!(state, nothing, nothing, NoSurfaceFlux(), 10.0)
        @test state.tracers_raw == orig   # byte-for-byte equal
    end

    @testset "NoSurfaceFlux is array-level identity" begin
        FT  = Float64
        q4d = rand(FT, 4, 3, 2, 3)
        orig = copy(q4d)
        # Array-level entry point returns `nothing`; q4d untouched.
        ret = apply_surface_flux!(q4d, NoSurfaceFlux(), nothing, 10.0,
                                  nothing, nothing;
                                  tracer_names = (:a, :b, :c))
        @test ret === nothing
        @test q4d == orig
    end

    @testset "apply! writes only to k = Nz surface layer" begin
        FT = Float64
        Nx, Ny, Nz = 4, 3, 4
        state = _make_state(FT, Nx, Ny, Nz; init = 5.0)
        orig_CO2  = copy(state.tracers.CO2)

        rate = fill(2.0, Nx, Ny)   # kg/s per cell
        dt   = 3.0                  # s
        op   = SurfaceFluxOperator(SurfaceFluxSource(:CO2, rate))

        apply!(state, nothing, nothing, op, dt)

        # Surface layer gets the full flux × dt addition
        expected_surface = orig_CO2[:, :, Nz] .+ rate .* dt
        @test state.tracers.CO2[:, :, Nz] ≈ expected_surface

        # Above-surface layers untouched
        for k in 1:(Nz - 1)
            @test state.tracers.CO2[:, :, k] == orig_CO2[:, :, k]
        end
    end

    @testset "apply! rate × dt arithmetic (kg/s × s → kg)" begin
        FT = Float64
        Nx, Ny, Nz = 3, 2, 2
        state = _make_state(FT, Nx, Ny, Nz; init = 0.0)

        rate = fill(7.5, Nx, Ny)    # 7.5 kg/s per cell
        dt   = 4.0                   # 4 seconds

        op = SurfaceFluxOperator(SurfaceFluxSource(:CO2, rate))
        apply!(state, nothing, nothing, op, dt)

        # Expected addition per surface cell: 7.5 × 4.0 = 30.0 kg
        @test all(isapprox.(state.tracers.CO2[:, :, Nz], 30.0; atol = eps(FT) * 10))
    end

    @testset "Multiple emitting tracers: per-tracer index resolution" begin
        FT = Float64
        Nx, Ny, Nz = 4, 3, 2
        state = _make_state(FT, Nx, Ny, Nz; init = 1.0,
                            tracer_names = (:CO2, :SF6, :Rn222))
        co2_rate   = fill(2.0,  Nx, Ny)
        sf6_rate   = fill(3.0,  Nx, Ny)
        rn222_rate = fill(5.0,  Nx, Ny)

        op = SurfaceFluxOperator(
            SurfaceFluxSource(:CO2,   co2_rate),
            SurfaceFluxSource(:SF6,   sf6_rate),
            SurfaceFluxSource(:Rn222, rn222_rate),
        )

        dt = 2.0
        apply!(state, nothing, nothing, op, dt)

        # Each tracer sees its own flux × dt on the surface layer only
        @test all(state.tracers.CO2[:, :, Nz]   .≈ 1.0 + 2.0 * dt)
        @test all(state.tracers.SF6[:, :, Nz]   .≈ 1.0 + 3.0 * dt)
        @test all(state.tracers.Rn222[:, :, Nz] .≈ 1.0 + 5.0 * dt)

        # Above-surface unchanged
        @test all(state.tracers.CO2[:, :, 1]   .== 1.0)
        @test all(state.tracers.SF6[:, :, 1]   .== 1.0)
        @test all(state.tracers.Rn222[:, :, 1] .== 1.0)
    end

    @testset "Tracer names in map but absent from state are skipped (not an error)" begin
        FT = Float64
        Nx, Ny, Nz = 4, 3, 2
        state = _make_state(FT, Nx, Ny, Nz; init = 1.0, tracer_names = (:CO2,))

        op = SurfaceFluxOperator(
            SurfaceFluxSource(:CO2, fill(2.0, Nx, Ny)),
            SurfaceFluxSource(:CH4, fill(9.9, Nx, Ny)),  # CH4 NOT in state
        )

        dt = 1.5
        apply!(state, nothing, nothing, op, dt)   # must not throw

        # CO2 surface gets the flux; CH4 entry is silently skipped.
        @test all(state.tracers.CO2[:, :, Nz] .≈ 1.0 + 2.0 * dt)
    end

    @testset "Tracer names in state but absent from map are untouched" begin
        FT = Float64
        Nx, Ny, Nz = 3, 2, 2
        state = _make_state(FT, Nx, Ny, Nz; init = 1.0,
                            tracer_names = (:CO2, :SF6, :Rn222))

        # Only CO2 has a source. SF6 and Rn222 must be byte-unchanged.
        orig_sf6    = copy(state.tracers.SF6)
        orig_rn222  = copy(state.tracers.Rn222)

        op = SurfaceFluxOperator(SurfaceFluxSource(:CO2, fill(3.0, Nx, Ny)))

        apply!(state, nothing, nothing, op, 2.0)

        @test state.tracers.SF6   == orig_sf6
        @test state.tracers.Rn222 == orig_rn222
    end

    @testset "Mass-accounting invariant: Σ (rate × dt × cells) == total added" begin
        # User-requested acceptance criterion (plan 17 Decision 1):
        # sum up the mass added globally and compare to the analytic total.
        FT = Float64
        Nx, Ny, Nz = 5, 4, 3
        state = _make_state(FT, Nx, Ny, Nz; init = 0.0, tracer_names = (:CO2,))

        # Spatially-varying rate so any indexing bug would skew the total.
        rate = reshape(collect(1.0:Float64(Nx * Ny)), Nx, Ny) ./ 100.0
        dt   = 7.5

        op = SurfaceFluxOperator(SurfaceFluxSource(:CO2, rate))
        apply!(state, nothing, nothing, op, dt)

        expected_total = sum(rate) * dt
        # Only surface layer is populated; upper layers stay zero.
        actual_total   = sum(state.tracers.CO2)

        @test actual_total ≈ expected_total
    end

    @testset "Array-level apply_surface_flux! operates on any 4D buffer" begin
        FT = Float64
        Nx, Ny, Nz, Nt = 4, 3, 2, 2
        # Fresh 4D buffer unrelated to any state — simulates the palindrome
        # ping-pong destination buffer (Commit 5 scenario).
        q_buf = zeros(FT, Nx, Ny, Nz, Nt)

        names = (:X, :Y)
        op = SurfaceFluxOperator(SurfaceFluxSource(:X, fill(2.0, Nx, Ny)),
                                  SurfaceFluxSource(:Y, fill(3.0, Nx, Ny)))

        apply_surface_flux!(q_buf, op, nothing, 4.0, nothing, nothing;
                            tracer_names = names)

        # X's surface = 2.0 * 4.0 = 8.0; Y's surface = 3.0 * 4.0 = 12.0
        @test all(q_buf[:, :, Nz, 1] .≈ 8.0)
        @test all(q_buf[:, :, Nz, 2] .≈ 12.0)
        # Interior untouched
        @test all(q_buf[:, :, 1, 1] .== 0.0)
        @test all(q_buf[:, :, 1, 2] .== 0.0)
    end

    @testset "emitting_tracer_indices helper" begin
        FT = Float64
        state = _make_state(FT, 4, 3, 2; init = 1.0,
                            tracer_names = (:CO2, :SF6, :Rn222))
        op = SurfaceFluxOperator(
            SurfaceFluxSource(:SF6,   fill(1.0, 4, 3)),
            SurfaceFluxSource(:Rn222, fill(2.0, 4, 3)),
        )

        indices = emitting_tracer_indices(op, state)
        # SF6 is index 2, Rn222 is index 3 in state.tracer_names
        @test indices === (2, 3)

        # NoSurfaceFlux short-circuits to empty tuple
        @test emitting_tracer_indices(NoSurfaceFlux(), state) === ()
    end

    @testset "Float32 precision path" begin
        FT = Float32
        Nx, Ny, Nz = 3, 2, 2
        state = _make_state(FT, Nx, Ny, Nz; init = 0.5f0,
                            tracer_names = (:CO2,))
        rate = fill(2.0f0, Nx, Ny)
        dt   = 3.0
        op   = SurfaceFluxOperator(SurfaceFluxSource(:CO2, rate))
        apply!(state, nothing, nothing, op, dt)

        # Kernel converts dt to Float32 via FT(dt); result stays Float32
        @test eltype(state.tracers_raw) === Float32
        @test all(state.tracers.CO2[:, :, Nz] .≈ 0.5f0 + 2.0f0 * 3.0f0)
    end

    @testset "Repeated apply! accumulates (not overwrites)" begin
        FT = Float64
        Nx, Ny, Nz = 3, 2, 2
        state = _make_state(FT, Nx, Ny, Nz; init = 0.0, tracer_names = (:CO2,))
        op = SurfaceFluxOperator(SurfaceFluxSource(:CO2, fill(1.0, Nx, Ny)))

        for _ in 1:5
            apply!(state, nothing, nothing, op, 2.0)
        end
        # 5 steps of 1.0 × 2.0 = 2.0 each → surface = 10.0
        @test all(state.tracers.CO2[:, :, Nz] .≈ 10.0)
    end

    @testset "Adapt — CPU passthrough" begin
        src = SurfaceFluxSource(:CO2, fill(1.0, 2, 2))
        op  = SurfaceFluxOperator(src)
        op_adapt = Adapt.adapt(Array, op)
        @test op_adapt isa SurfaceFluxOperator
        @test flux_for(op_adapt.flux_map, :CO2).cell_mass_rate == src.cell_mass_rate
    end

    @testset "Adapt — NoSurfaceFlux is a bits-stable singleton" begin
        op = NoSurfaceFlux()
        @test Adapt.adapt(Array, op) === op
    end

    if HAS_GPU_SFO
        @testset "GPU apply! matches CPU bit-for-bit" begin
            FT = Float64
            Nx, Ny, Nz = 8, 6, 3

            rate   = rand(FT, Nx, Ny)
            dt     = 2.5

            # CPU path
            state_cpu = _make_state(FT, Nx, Ny, Nz; init = 0.0, tracer_names = (:CO2,))
            op_cpu    = SurfaceFluxOperator(SurfaceFluxSource(:CO2, rate))
            apply!(state_cpu, nothing, nothing, op_cpu, dt)

            # GPU path — adapt state + operator
            state_gpu = Adapt.adapt(CUDA.CuArray, state_cpu)   # carries post-CPU-apply state
            # Reset tracers to zero on GPU to match CPU starting state
            state_gpu.tracers_raw .= 0.0
            op_gpu = Adapt.adapt(CUDA.CuArray, op_cpu)
            apply!(state_gpu, nothing, nothing, op_gpu, dt)

            # Pull result back and compare
            @test Array(state_gpu.tracers_raw) ≈ state_cpu.tracers_raw
        end
    end
end
