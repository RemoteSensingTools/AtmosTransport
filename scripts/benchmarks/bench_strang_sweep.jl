#!/usr/bin/env julia
# ===========================================================================
# Strang split per-step benchmark
#
# Measures per-step wall time for strang_split! (LatLon, structured path)
# on a synthetic problem. Reports median ± MAD (robust to outliers).
#
# Usage:
#   julia --project=. scripts/benchmarks/bench_strang_sweep.jl [size] [backend] [--events]
# where
#   size    ∈ {small, medium, large}   (default: medium)
#   backend ∈ {cpu, gpu}               (default: cpu; gpu requires CUDA)
#   --events                           on GPU: also report CUDA-event-based
#                                      device time (vs. host wall time).
#
# On GPU, only Float32 is reported because L40S has no Float64 units.
# On CPU both Float64 and Float32 are reported.
#
# With --events on GPU:
#   - host(ms): wall time measured with synchronize bracketing; includes
#              host-side kernel-launch overhead + any synchronize cost
#   - cuda(ms): pure GPU time measured via CUDA events; kernel work only
#   - delta = host - cuda: time the CPU spent on Julia dispatch, kernel
#              launches, and blocked on synchronize. Sync removal (plan 13)
#              targets this delta.
# ===========================================================================

using AtmosTransport
using AtmosTransport: AdvectionWorkspace, strang_split!,
    UpwindScheme, SlopesScheme, PPMScheme, MonotoneLimiter,
    LatLonMesh, AtmosGrid, HybridSigmaPressure, CPU,
    cell_areas_by_latitude, gravity, reference_pressure, level_thickness,
    CellState, StructuredFaceFluxState, DryBasis
using Statistics
using Printf

const POSITIONAL = filter(a -> !startswith(a, "--"), ARGS)
const SIZE    = length(POSITIONAL) >= 1 ? Symbol(POSITIONAL[1]) : :medium
const BACKEND = length(POSITIONAL) >= 2 ? Symbol(POSITIONAL[2]) : :cpu
const EVENTS  = "--events" in ARGS

if BACKEND === :gpu
    using CUDA
    CUDA.functional() || error("CUDA requested but not functional")
    # Wrap CUDA.@elapsed in a regular function so it's only parsed when
    # CUDA is loaded. run_benchmark can then reference gpu_event_time
    # at compile time on CPU without pulling in CUDA.
    @eval gpu_event_time(f::Function) = CUDA.@elapsed f()
end

const DIMS = Dict(
    :small  => (Nx =  72, Ny =  36, Nz =  4, Nt = 1,  n_steps = 50),
    :medium => (Nx = 288, Ny = 144, Nz = 32, Nt = 5,  n_steps = 40),
    :large  => (Nx = 576, Ny = 288, Nz = 72, Nt = 10, n_steps = 60),
)[SIZE]

"Median absolute deviation (robust scale)."
median_abs_dev(v::AbstractVector) = median(abs.(v .- median(v)))

function build_problem(::Type{FT}, cfg) where {FT}
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    mesh = LatLonMesh(; Nx = Nx, Ny = Ny, FT = FT)
    # Pure-sigma coordinate: A = 0 everywhere, B linear from 0 (TOA) to 1 (surface).
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

    rms = [similar(m) for _ in 1:cfg.Nt]
    @inbounds for t in 1:cfg.Nt
        chi0 = FT(100e-6) * t
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            lat_frac = FT(j - 1) / FT(Ny - 1)
            rms[t][i, j, k] = m[i, j, k] * (chi0 + FT(200e-6) * lat_frac)
        end
    end

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

function run_benchmark(::Type{FT}, scheme, cfg, backend::Symbol, use_events::Bool) where {FT}
    grid, m_cpu, rms_cpu, am_cpu, bm_cpu, cm_cpu = build_problem(FT, cfg)

    tracer_syms = ntuple(t -> Symbol("tr$t"), cfg.Nt)

    if backend === :gpu
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

    host_times_ms = Float64[]
    evt_times_ms  = Float64[]
    for _ in 1:cfg.n_steps
        if backend === :gpu
            CUDA.synchronize()
            t0 = time_ns()
            if use_events
                # CUDA events bracket the call on-device; returns elapsed
                # device time in seconds. Isolates GPU work from host-side
                # launch + synchronize overhead.
                evt_s = gpu_event_time(() ->
                    strang_split!(state, fluxes, grid, scheme; workspace = ws))
                push!(evt_times_ms, evt_s * 1e3)
            else
                strang_split!(state, fluxes, grid, scheme; workspace = ws)
            end
            CUDA.synchronize()
            push!(host_times_ms, (time_ns() - t0) * 1e-6)
        else
            t = @elapsed strang_split!(state, fluxes, grid, scheme; workspace = ws)
            push!(host_times_ms, t * 1e3)
        end
    end

    med_host = median(host_times_ms)
    mad_host = median_abs_dev(host_times_ms)
    med_evt  = isempty(evt_times_ms) ? NaN : median(evt_times_ms)
    mad_evt  = isempty(evt_times_ms) ? NaN : median_abs_dev(evt_times_ms)
    return med_host, mad_host, med_evt, mad_evt
end

function main()
    ft_list = BACKEND === :gpu ? (Float32,) : (Float64, Float32)
    show_events = EVENTS && BACKEND === :gpu

    @printf("Strang sweep benchmark (%s / %s: Nx=%d Ny=%d Nz=%d Nt=%d, steps=%d%s)\n",
            SIZE, BACKEND, DIMS.Nx, DIMS.Ny, DIMS.Nz, DIMS.Nt, DIMS.n_steps,
            show_events ? ", events" : "")
    if show_events
        @printf("%-10s %-10s %12s %12s %12s %12s %12s\n",
                "FT", "Scheme", "host(ms)", "host MAD", "cuda(ms)", "cuda MAD", "Δhost-cuda")
    else
        @printf("%-10s %-10s %12s %12s\n",
                "FT", "Scheme", "median(ms)", "MAD(ms)")
    end
    println("-"^80)

    for FT in ft_list
        for (scheme, tag) in (
            (UpwindScheme(),                      "Upwind"),
            (SlopesScheme(MonotoneLimiter()),     "Slopes"),
            (PPMScheme(MonotoneLimiter()),        "PPM"),
        )
            med_host, mad_host, med_evt, mad_evt = run_benchmark(FT, scheme, DIMS, BACKEND, EVENTS)
            if show_events
                @printf("%-10s %-10s %12.3f %12.3f %12.3f %12.3f %12.3f\n",
                        string(FT), tag, med_host, mad_host, med_evt, mad_evt,
                        med_host - med_evt)
            else
                @printf("%-10s %-10s %12.3f %12.3f\n",
                        string(FT), tag, med_host, mad_host)
            end
        end
    end
end

main()
