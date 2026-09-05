#!/usr/bin/env julia
# CPU-only investigation; does not change the production solver or launch GPU work.
# CUDA_VISIBLE_DEVICES='' julia --project=. scripts/benchmarks/probe_matrix_convection_cpu.jl [results.toml]
using AtmosTransport
using KernelAbstractions: CPU, synchronize
using Random, Test, LinearAlgebra, Statistics, TOML
const C = AtmosTransport.Operators.Convection

"""Partial-pivot LU specialized to an upper-Hessenberg active block (prototype)."""
function hessenberg_lu!(A::AbstractMatrix{FT}, pivots, n; icltop_eff=1) where FT
    lo = max(icltop_eff, 1)
    @inbounds for k in lo:n
        p = k < n && abs(A[k+1, k]) > abs(A[k, k]) ? k+1 : k
        pivots[k] = p
        if p != k
            # Retain previous L multipliers, including fill from earlier swaps.
            for j in lo:n
                A[k, j], A[p, j] = A[p, j], A[k, j]
            end
        end
        d = A[k, k]
        (iszero(d) || k == n) && continue
        A[k+1, k] *= one(FT) / d
        l = A[k+1, k]
        for j in (k+1):n
            A[k+1, j] -= l * A[k, j]
        end
    end
    nothing
end

function tiled_solve!(rhs, lu, pivots, width; lo=1)
    n, nt = size(rhs)
    for first in 1:width:nt
        last = min(nt, first + width - 1)
        C._tm5_solve!(@view(rhs[:, first:last]), lu, pivots, n, last-first+1;
                      icltop_eff=lo)
    end
    nothing
end

function fixture(rng, FT, n, top)
    cmfmc = zeros(FT, 1, 1, n+1)
    dtrain = zeros(FT, 1, 1, n)
    cmfmc[1, 1, (top+1):n] .= FT(0.1) .* rand(rng, FT, n-top)
    dtrain[1, 1, top:n] .= FT(0.01) .* rand(rng, FT, n-top+1)
    entu, detu = similar(dtrain), similar(dtrain)
    C._derive_cmfmc_matrix_rates_ll_kernel!(CPU(), (1, 1))(
        entu, detu, cmfmc, dtrain, n; ndrange=(1, 1))
    synchronize(CPU())
    m = FT(50) .+ FT(150) .* rand(rng, FT, n)
    (; entu=vec(entu), detu=vec(detu), zero=zeros(FT, n), m, dt=FT(1800), n)
end

function build!(A, f, amu, amd)
    top, _, lfs = C._tm5_diagnose_cloud_dims(f.detu, f.zero, f.n)
    C._tm5_build_conv1!(A, f.entu, f.detu, f.zero, f.zero, f.m,
                       top, lfs, f.dt, f.n; f=A, amu, amd)
    min(lfs, max(top, 2)-1)
end

function validate()
    rng = MersenneTwister(9042026)
    swaps = 0
    @testset "CPU matrix convection optimization probes" begin
        for FT in (Float32, Float64), n in (25, 60, 85, 137), top in (1, 4), trial in 1:3
            f = fixture(rng, FT, n, top)
            A = Matrix{FT}(I, n, n)
            lo = build!(A, f, zeros(FT, n+1), zeros(FT, n+1))
            @test all(iszero(A[i,j]) for j in 1:n for i in (j+2):n)
            dense, fast = copy(A), copy(A)
            pd, ph = collect(1:n), collect(1:n)
            C._tm5_lu!(dense, pd, n; icltop_eff=lo)
            hessenberg_lu!(fast, ph, n; icltop_eff=lo)
            @test pd == ph
            @test dense == fast
            for nt in (1, 6, 7, 12, 32, 65)
                rhs = rand(rng, FT, n, nt) .* f.m
                ref = copy(rhs)
                C._tm5_solve!(ref, dense, pd, n, nt; icltop_eff=lo)
                for width in (4, 6, 16, 32)
                    actual = copy(rhs)
                    tiled_solve!(actual, fast, ph, width; lo)
                    @test reinterpret(floatbits(FT), vec(actual)) == reinterpret(floatbits(FT), vec(ref))
                end
                @test norm(Float64.(A)*Float64.(ref)-Float64.(rhs)) / norm(rhs) < 100eps(FT)
                @test minimum(ref) >= 0
                @test maximum(abs.(sum(ref; dims=1)-sum(rhs; dims=1)) ./ sum(rhs; dims=1)) < 100eps(FT)
            end
        end
        # Synthetic nonsingular Hessenberg matrices force nontrivial adjacent
        # pivots; physical convection often needs none. Retain the general solve.
        for FT in (Float32, Float64), n in (25, 85), trial in 1:20
            A = triu(randn(rng, FT, n, n), -1)
            A[1,1] = FT(0.01)
            A[2,1] = FT(2)
            dense, fast = copy(A), copy(A)
            pd, ph = zeros(Int, n), zeros(Int, n)
            C._tm5_lu!(dense, pd, n)
            hessenberg_lu!(fast, ph, n)
            swaps += count(pd .!= 1:n)
            @test pd == ph
            @test dense == fast
            rhs = randn(rng, FT, n, 7)
            ref, actual = copy(rhs), copy(rhs)
            C._tm5_solve!(ref, dense, pd, n, 7)
            tiled_solve!(actual, fast, ph, 6)
            @test reinterpret(floatbits(FT), vec(actual)) == reinterpret(floatbits(FT), vec(ref))
            @test norm(Float64.(A)*Float64.(actual)-Float64.(rhs)) /
                  (norm(A)*norm(actual)+norm(rhs)) < 100eps(FT)
        end
        @test swaps > 0
        # RHS batching also applies to the general dense TM5 factorization.
        for FT in (Float32, Float64)
            n, nt = 85, 65
            A = randn(rng, FT, n, n) + FT(n)*I
            piv = zeros(Int, n)
            C._tm5_lu!(A, piv, n)
            rhs = randn(rng, FT, n, nt)
            ref = copy(rhs)
            C._tm5_solve!(ref, A, piv, n, nt)
            for width in (4, 6, 16, 32)
                actual = copy(rhs)
                tiled_solve!(actual, A, piv, width)
                @test reinterpret(floatbits(FT), vec(actual)) == reinterpret(floatbits(FT), vec(ref))
            end
        end
    end
    swaps
end

floatbits(::Type{Float32}) = UInt32
floatbits(::Type{Float64}) = UInt64

function median_us(f; repetitions=30, samples=9)
    for _ in 1:5
        f()
    end
    timings = Float64[]
    for _ in 1:samples
        elapsed = @elapsed begin
            for _ in 1:repetitions
                f()
            end
        end
        push!(timings, elapsed * 1e6 / repetitions)
    end
    median(timings)
end

function benchmark()
    rng = MersenneTwister(9042026)
    results = []
    for n in (60, 85, 137), nt in (6, 32, 65)
        FT = Float32
        f = fixture(rng, FT, n, 1)
        amu, amd = zeros(FT, n+1), zeros(FT, n+1)
        A, work = Matrix{FT}(I, n, n), zeros(FT, n, n)
        lo = build!(A, f, amu, amd)
        piv = zeros(Int, n)
        rhs = rand(rng, FT, n, nt) .* f.m
        q = similar(rhs)
        factor_times, full_times = Float64[], Float64[]
        for factor! in (C._tm5_lu!, hessenberg_lu!)
            push!(factor_times, median_us() do
                copyto!(work, A)
                factor!(work, piv, n; icltop_eff=lo)
            end)
            push!(full_times, median_us() do
                build!(work, f, amu, amd)
                factor!(work, piv, n; icltop_eff=lo)
                copyto!(q, rhs)
                tiled_solve!(q, work, piv, 6; lo)
            end)
        end
        push!(results, Dict("levels"=>n, "tracers"=>nt,
            "dense_factor_us"=>factor_times[1], "hessenberg_factor_us"=>factor_times[2],
            "dense_column_us"=>full_times[1], "hessenberg_column_us"=>full_times[2],
            "factor_speedup"=>factor_times[1]/factor_times[2],
            "column_speedup"=>full_times[1]/full_times[2]))
    end
    results
end

function main()
    swaps = validate()
    results = Dict("julia_version"=>string(VERSION), "gpu_used"=>false,
        "cpu"=>Sys.cpu_info()[1].model, "julia_threads"=>Threads.nthreads(),
        "pivot_stress_swaps"=>swaps,
        "scope"=>"Synthetic CPU columns; timings include buffer copies; no GPU speed claim",
        "benchmark"=>benchmark())
    TOML.print(stdout, results)
    if !isempty(ARGS)
        open(ARGS[1], "w") do io
            TOML.print(io, results)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
