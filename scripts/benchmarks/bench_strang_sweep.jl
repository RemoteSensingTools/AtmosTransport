#!/usr/bin/env julia
# ===========================================================================
# Strang split per-step benchmark
#
# Measures per-step wall time for strang_split! (LatLon, structured path)
# on a synthetic problem. Use this to compare ping-pong refactor
# before/after. Reports median, mean, std for Upwind / Slopes / PPM.
#
# Usage:
#   julia --project=. scripts/benchmarks/bench_strang_sweep.jl [size] [backend]
# where
#   size    ∈ {small, medium, large}   (default: medium)
#   backend ∈ {cpu, gpu}               (default: cpu; gpu requires CUDA)
#
# On GPU, only Float32 is reported because L40S has no Float64 units.
# On CPU both Float64 and Float32 are reported.
# ===========================================================================

using AtmosTransport
using AtmosTransport: AdvectionWorkspace, strang_split!,
    UpwindScheme, SlopesScheme, PPMScheme, MonotoneLimiter,
    LatLonMesh, AtmosGrid, HybridSigmaPressure, CPU,
    cell_areas_by_latitude, gravity, reference_pressure, level_thickness,
    CellState, StructuredFaceFluxState, DryBasis
using Statistics
using Printf

const SIZE    = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :medium
const BACKEND = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :cpu

if BACKEND === :gpu
    using CUDA
    CUDA.functional() || error("CUDA requested but not functional")
end

const DIMS = Dict(
    :small  => (Nx =  72, Ny =  36, Nz =  4, Nt = 1,  n_steps = 50),
    :medium => (Nx = 288, Ny = 144, Nz = 32, Nt = 5,  n_steps = 20),
    :large  => (Nx = 576, Ny = 288, Nz = 72, Nt = 10, n_steps = 10),
)[SIZE]

function build_problem(::Type{FT}, cfg) where {FT}
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    mesh = LatLonMesh(; Nx = Nx, Ny = Ny, FT = FT)
    # Pure-sigma coordinate: A = 0 everywhere, B linear from 0 (TOA) to 1 (surface).
    # This keeps interface pressures monotonic regardless of Nz.
    A_ifc = zeros(FT, Nz + 1)
    B_ifc = FT.(collect(range(0.0, 1.0, length = Nz + 1)))
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, CPU(); FT = FT)

    g  = gravity(grid)
    ps = reference_pressure(grid)
    areas = cell_areas_by_latitude(mesh)

    m = zeros(FT, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        m[i, j, k] = FT(level_thickness(vc, k, ps)) * areas[j] / g
    end

    # Multi-tracer: distinct latitudinal gradient per tracer
    rms = [similar(m) for _ in 1:cfg.Nt]
    @inbounds for t in 1:cfg.Nt
        chi0 = FT(100e-6) * t
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            lat_frac = FT(j - 1) / FT(Ny - 1)
            rms[t][i, j, k] = m[i, j, k] * (chi0 + FT(200e-6) * lat_frac)
        end
    end

    # Synthetic face fluxes at ~CFL 0.3 on smallest cell mass
    m_min = minimum(m)
    cfl = FT(0.3)
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:(Nx + 1)
        lat = FT(-90) + (FT(j) - FT(0.5)) * (FT(180) / FT(Ny))
        am[i, j, k] = cfl * m_min * cos(lat * FT(pi) / FT(180))
    end
    am[:, 1, :]  .= zero(FT)
    am[:, Ny, :] .= zero(FT)
    @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
        lat = FT(-90) + FT(j - 1) * (FT(180) / FT(Ny))
        bm[i, j, k] = cfl * m_min * FT(0.1) * sin(FT(2) * lat * FT(pi) / FT(180))
    end
    cm = zeros(FT, Nx, Ny, Nz + 1)

    return grid, m, rms, am, bm, cm
end

function run_benchmark(::Type{FT}, scheme, cfg, backend::Symbol) where {FT}
    grid, m_cpu, rms_cpu, am_cpu, bm_cpu, cm_cpu = build_problem(FT, cfg)

    tracer_syms = ntuple(t -> Symbol("tr$t"), cfg.Nt)

    if backend === :gpu
        # Move to GPU. The grid stays a CPU object (it's immutable
        # metadata); only the prognostic arrays live on device.
        m      = CUDA.CuArray(m_cpu)
        am     = CUDA.CuArray(am_cpu)
        bm     = CUDA.CuArray(bm_cpu)
        cm     = CUDA.CuArray(cm_cpu)
        tracer_vals = ntuple(t -> CUDA.CuArray(rms_cpu[t]), cfg.Nt)
    else
        m      = copy(m_cpu)
        am     = copy(am_cpu)
        bm     = copy(bm_cpu)
        cm     = copy(cm_cpu)
        tracer_vals = ntuple(t -> copy(rms_cpu[t]), cfg.Nt)
    end

    tracers = NamedTuple{tracer_syms}(tracer_vals)
    state   = CellState(DryBasis, m; tracers...)
    fluxes  = StructuredFaceFluxState(am, bm, cm)
    ws      = AdvectionWorkspace(state.air_mass; n_tracers = cfg.Nt)

    # Warmup (compile + JIT)
    for _ in 1:3
        strang_split!(state, fluxes, grid, scheme; workspace = ws)
    end
    backend === :gpu && CUDA.synchronize()

    times = Float64[]
    for _ in 1:cfg.n_steps
        if backend === :gpu
            CUDA.synchronize()
            t0 = time_ns()
            strang_split!(state, fluxes, grid, scheme; workspace = ws)
            CUDA.synchronize()
            push!(times, (time_ns() - t0) * 1e-9)
        else
            t = @elapsed strang_split!(state, fluxes, grid, scheme; workspace = ws)
            push!(times, t)
        end
    end

    return median(times) * 1e3, mean(times) * 1e3, std(times) * 1e3
end

function main()
    # On GPU (L40S), skip Float64 — no F64 units.
    ft_list = BACKEND === :gpu ? (Float32,) : (Float64, Float32)

    @printf("Strang sweep benchmark (%s / %s: Nx=%d Ny=%d Nz=%d Nt=%d, steps=%d)\n",
            SIZE, BACKEND, DIMS.Nx, DIMS.Ny, DIMS.Nz, DIMS.Nt, DIMS.n_steps)
    @printf("%-10s %-10s %12s %12s %12s\n",
            "FT", "Scheme", "median(ms)", "mean(ms)", "std(ms)")
    println("-"^60)

    for FT in ft_list
        for (scheme, tag) in (
            (UpwindScheme(),                      "Upwind"),
            (SlopesScheme(MonotoneLimiter()),     "Slopes"),
            (PPMScheme(MonotoneLimiter()),        "PPM"),
        )
            med, avg, sd = run_benchmark(FT, scheme, DIMS, BACKEND)
            @printf("%-10s %-10s %12.3f %12.3f %12.3f\n",
                    string(FT), tag, med, avg, sd)
        end
    end
end

main()
