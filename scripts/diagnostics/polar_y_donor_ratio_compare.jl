#!/usr/bin/env julia

# Compare polar Y donor ratios |bm|/m_donor between two ERA5 binaries.
# Replays mass-only Strang sweeps (X-Y-Z-Z-Y-X) and logs Y1/Y2 pre-sweep stats.

using Printf

using AtmosTransport
using AtmosTransport.IO: MassFluxBinaryReader, load_window!
using AtmosTransport.Grids: compute_reduced_grid_tm5
using KernelAbstractions: get_backend, synchronize

const MA = AtmosTransport.Advection

struct SweepYStats
    substep::Int
    sweep::String
    max_all::Float64
    loc_all::NTuple{3,Int}
    max_polar::Float64
    loc_polar::NTuple{3,Int}
    max_nonpolar::Float64
    loc_nonpolar::NTuple{3,Int}
    max_all_relaxed::Float64
    loc_all_relaxed::NTuple{3,Int}
end

struct TraceResult
    path::String
    Nx::Int
    Ny::Int
    Nz::Int
    window::Int
    n_substeps::Int
    polar_band::Int
    stats::Vector{SweepYStats}
end

function parse_cli(argv::Vector{String})
    positional = String[]
    window = 1
    n_substeps = 4
    polar_band = 3
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--window"
            i == length(argv) && error("--window requires an integer")
            window = parse(Int, argv[i + 1])
            i += 2
        elseif a == "--n-substeps"
            i == length(argv) && error("--n-substeps requires an integer")
            n_substeps = parse(Int, argv[i + 1])
            i += 2
        elseif a == "--polar-band"
            i == length(argv) && error("--polar-band requires an integer")
            polar_band = parse(Int, argv[i + 1])
            i += 2
        else
            push!(positional, a)
            i += 1
        end
    end
    length(positional) == 2 || error("Usage: ... <file_A.bin> <file_B.bin> [--window N] [--n-substeps N] [--polar-band N]")
    return positional[1], positional[2], window, n_substeps, polar_band
end

function load_window_fields(path::String, window::Int)
    r = MassFluxBinaryReader(path, Float64)
    try
        Nx, Ny, Nz = r.Nx, r.Ny, r.Nz
        m = Array{Float64}(undef, Nx, Ny, Nz)
        am = Array{Float64}(undef, Nx + 1, Ny, Nz)
        bm = Array{Float64}(undef, Nx, Ny + 1, Nz)
        cm = Array{Float64}(undef, Nx, Ny, Nz + 1)
        ps = Array{Float64}(undef, Nx, Ny)
        load_window!(m, am, bm, cm, ps, r, window)
        return m, am, bm, cm, r.lats
    finally
        close(r)
    end
end

function y_stats(m::Array{Float64,3}, bm::Array{Float64,3}, polar_band::Int, substep::Int, sweep::String)
    Nx, Ny, Nz = size(m)
    pb = clamp(polar_band, 1, max(1, Ny ÷ 2))

    max_all = -Inf
    max_polar = -Inf
    max_nonpolar = -Inf
    max_all_relaxed = -Inf
    loc_all = (0, 0, 0)
    loc_polar = (0, 0, 0)
    loc_nonpolar = (0, 0, 0)
    loc_all_relaxed = (0, 0, 0)

    @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
        bm_v = bm[i, j, k]
        donor_j = bm_v >= 0 ? j - 1 : j
        md = bm_v >= 0 ? m[i, j - 1, k] : m[i, j, k]
        if md <= 0
            continue
        end
        c = abs(bm_v) / md
        if c > max_all
            max_all = c
            loc_all = (i, j, k)
        end

        is_polar = donor_j <= pb || donor_j >= Ny - pb + 1
        if is_polar
            if c > max_polar
                max_polar = c
                loc_polar = (i, j, k)
            end
            c_relaxed = c / 6.0
            if c_relaxed > max_all_relaxed
                max_all_relaxed = c_relaxed
                loc_all_relaxed = (i, j, k)
            end
        else
            if c > max_nonpolar
                max_nonpolar = c
                loc_nonpolar = (i, j, k)
            end
            if c > max_all_relaxed
                max_all_relaxed = c
                loc_all_relaxed = (i, j, k)
            end
        end
    end

    return SweepYStats(substep, sweep, max_all, loc_all, max_polar, loc_polar,
                       max_nonpolar, loc_nonpolar, max_all_relaxed, loc_all_relaxed)
end

function trace_file(path::String; window::Int = 1, n_substeps::Int = 4, polar_band::Int = 3)
    m, am, bm, cm, lats = load_window_fields(path, window)
    Nx, Ny, Nz = size(m)
    rg = compute_reduced_grid_tm5(Nx, lats)
    cluster_sizes = Int32.(rg === nothing ? ones(Int, Ny) : rg.cluster_sizes)

    m_pilot = copy(m)
    m_buf = similar(m)
    backend = get_backend(m_pilot)
    kx! = MA._mass_only_x_kernel!(backend, 256)
    ky! = MA._mass_only_y_kernel!(backend, 256)
    kz! = MA._mass_only_z_kernel!(backend, 256)

    stats = SweepYStats[]
    for sub in 1:n_substeps
        # X1
        kx!(m_buf, m_pilot, am, Int32(Nx), cluster_sizes; ndrange=size(m_pilot))
        synchronize(backend)
        copyto!(m_pilot, m_buf)

        # pre Y1
        push!(stats, y_stats(m_pilot, bm, polar_band, sub, "Y1"))

        # Y1
        ky!(m_buf, m_pilot, bm, Int32(Ny); ndrange=size(m_pilot))
        synchronize(backend)
        copyto!(m_pilot, m_buf)

        # Z1
        kz!(m_buf, m_pilot, cm, Int32(Nz); ndrange=size(m_pilot))
        synchronize(backend)
        copyto!(m_pilot, m_buf)
        # Z2
        kz!(m_buf, m_pilot, cm, Int32(Nz); ndrange=size(m_pilot))
        synchronize(backend)
        copyto!(m_pilot, m_buf)

        # pre Y2
        push!(stats, y_stats(m_pilot, bm, polar_band, sub, "Y2"))

        # Y2
        ky!(m_buf, m_pilot, bm, Int32(Ny); ndrange=size(m_pilot))
        synchronize(backend)
        copyto!(m_pilot, m_buf)

        # X2
        kx!(m_buf, m_pilot, am, Int32(Nx), cluster_sizes; ndrange=size(m_pilot))
        synchronize(backend)
        copyto!(m_pilot, m_buf)
    end

    return TraceResult(path, Nx, Ny, Nz, window, n_substeps, polar_band, stats)
end

function print_result(r::TraceResult, label::String)
    println("\n=== $label ===")
    println("file: $(r.path)")
    @printf("dims: %dx%dx%d | window=%d | substeps=%d | polar_band=%d\n",
            r.Nx, r.Ny, r.Nz, r.window, r.n_substeps, r.polar_band)
    println("sub  sweep   max_all      max_polar    max_nonpolar max_all_relaxed")
    for s in r.stats
        @printf("%3d  %-4s  %10.6f   %10.6f   %10.6f   %10.6f\n",
                s.substep, s.sweep, s.max_all, s.max_polar, s.max_nonpolar, s.max_all_relaxed)
    end
    worst = findmax(map(x -> x.max_polar, r.stats))
    ws = r.stats[worst[2]]
    @printf("worst polar ratio: %.6f at substep=%d %s loc=%s\n",
            ws.max_polar, ws.substep, ws.sweep, string(ws.loc_polar))
end

function print_compare(a::TraceResult, b::TraceResult)
    @assert length(a.stats) == length(b.stats)
    println("\n=== A/B ratio (A / B) ===")
    println("sub  sweep   polar_ratio   all_ratio     relaxed_ratio")
    for i in eachindex(a.stats)
        sa, sb = a.stats[i], b.stats[i]
        @printf("%3d  %-4s  %10.4f   %10.4f   %10.4f\n",
                sa.substep, sa.sweep,
                sa.max_polar / sb.max_polar,
                sa.max_all / sb.max_all,
                sa.max_all_relaxed / sb.max_all_relaxed)
    end
end

function main()
    file_a, file_b, window, n_substeps, polar_band = parse_cli(ARGS)
    A = trace_file(file_a; window=window, n_substeps=n_substeps, polar_band=polar_band)
    B = trace_file(file_b; window=window, n_substeps=n_substeps, polar_band=polar_band)
    print_result(A, "A")
    print_result(B, "B")
    print_compare(A, B)
end

main()
