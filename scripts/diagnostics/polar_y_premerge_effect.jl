#!/usr/bin/env julia

# Estimate impact of lon-binning applied immediately before each Y sweep
# using mass-only Strang evolution driven by binary am/bm/cm.

using Printf
using KernelAbstractions: get_backend, synchronize

using AtmosTransport
using AtmosTransport.IO: MassFluxBinaryReader, load_window!
using AtmosTransport.Grids: compute_reduced_grid_tm5

const MA = AtmosTransport.Advection

function parse_cli(argv::Vector{String})
    length(argv) >= 2 || error("Usage: ... <file_A.bin> <file_B.bin> [--window N] [--substeps N] [--cap-rows N] [--r N]")
    a = argv[1]
    b = argv[2]
    window = 1
    substeps = 4
    cap_rows = 2
    rbin = 720
    i = 3
    while i <= length(argv)
        x = argv[i]
        if x == "--window"
            window = parse(Int, argv[i + 1]); i += 2
        elseif x == "--substeps"
            substeps = parse(Int, argv[i + 1]); i += 2
        elseif x == "--cap-rows"
            cap_rows = parse(Int, argv[i + 1]); i += 2
        elseif x == "--r"
            rbin = parse(Int, argv[i + 1]); i += 2
        else
            error("Unknown arg: $x")
        end
    end
    return a, b, window, substeps, cap_rows, rbin
end

function load_fields(path::String, window::Int)
    rdr = MassFluxBinaryReader(path, Float64)
    Nx, Ny, Nz = rdr.Nx, rdr.Ny, rdr.Nz
    m = Array{Float64}(undef, Nx, Ny, Nz)
    am = Array{Float64}(undef, Nx + 1, Ny, Nz)
    bm = Array{Float64}(undef, Nx, Ny + 1, Nz)
    cm = Array{Float64}(undef, Nx, Ny, Nz + 1)
    ps = Array{Float64}(undef, Nx, Ny)
    load_window!(m, am, bm, cm, ps, rdr, window)
    lats = copy(rdr.lats)
    close(rdr)
    return m, am, bm, cm, lats
end

function y_metrics(m::Array{Float64,3}, bm::Array{Float64,3}, cap_rows::Int, rbin::Int)
    Nx, Ny, Nz = size(m)
    @assert Nx % rbin == 0 "r must divide Nx"
    ngr = Nx ÷ rbin
    is_polar(dj) = (dj <= cap_rows) || (dj >= Ny - cap_rows + 1)

    raw_max = 0.0
    net_max = 0.0
    gross_max = 0.0
    @inbounds for k in 1:Nz, j in 2:Ny
        for g in 1:ngr
            ist = (g - 1) * rbin + 1
            ied = g * rbin
            bsum = 0.0
            babs = 0.0
            msum = 0.0
            for i in ist:ied
                b = bm[i, j, k]
                dj = b >= 0 ? j - 1 : j
                is_polar(dj) || continue
                md = b >= 0 ? m[i, j - 1, k] : m[i, j, k]
                if md <= 0
                    continue
                end
                c = abs(b) / md
                raw_max = max(raw_max, c)
                bsum += b
                babs += abs(b)
                msum += md
            end
            if msum > 0
                net_max = max(net_max, abs(bsum) / msum)
                gross_max = max(gross_max, babs / msum)
            end
        end
    end
    return raw_max, net_max, gross_max
end

function run_case(path::String; window::Int, substeps::Int, cap_rows::Int, rbin::Int)
    m, am, bm, cm, lats = load_fields(path, window)
    Nx, Ny, _ = size(m)
    rg = compute_reduced_grid_tm5(Nx, lats)
    cluster_sizes = Int32.(rg === nothing ? ones(Int, Ny) : rg.cluster_sizes)
    m_pilot = copy(m)
    m_buf = similar(m)
    backend = get_backend(m_pilot)
    kx! = MA._mass_only_x_kernel!(backend, 256)
    ky! = MA._mass_only_y_kernel!(backend, 256)
    kz! = MA._mass_only_z_kernel!(backend, 256)

    rows = String[]
    for sub in 1:substeps
        # X1
        kx!(m_buf, m_pilot, am, Int32(Nx), cluster_sizes; ndrange=size(m_pilot))
        synchronize(backend); copyto!(m_pilot, m_buf)

        # Pre-Y1 metrics
        raw, net, gross = y_metrics(m_pilot, bm, cap_rows, rbin)
        push!(rows, @sprintf("%2d Y1 raw=%10.4f net=%10.4f gross=%10.4f red(net)=%.2fx red(gross)=%.2fx",
                             sub, raw, net, gross, raw / max(net, 1e-30), raw / max(gross, 1e-30)))

        # Y1, Z1, Z2
        ky!(m_buf, m_pilot, bm, Int32(Ny); ndrange=size(m_pilot))
        synchronize(backend); copyto!(m_pilot, m_buf)
        kz!(m_buf, m_pilot, cm, Int32(size(m, 3)); ndrange=size(m_pilot))
        synchronize(backend); copyto!(m_pilot, m_buf)
        kz!(m_buf, m_pilot, cm, Int32(size(m, 3)); ndrange=size(m_pilot))
        synchronize(backend); copyto!(m_pilot, m_buf)

        # Pre-Y2 metrics
        raw, net, gross = y_metrics(m_pilot, bm, cap_rows, rbin)
        push!(rows, @sprintf("%2d Y2 raw=%10.4f net=%10.4f gross=%10.4f red(net)=%.2fx red(gross)=%.2fx",
                             sub, raw, net, gross, raw / max(net, 1e-30), raw / max(gross, 1e-30)))

        # Y2, X2
        ky!(m_buf, m_pilot, bm, Int32(Ny); ndrange=size(m_pilot))
        synchronize(backend); copyto!(m_pilot, m_buf)
        kx!(m_buf, m_pilot, am, Int32(Nx), cluster_sizes; ndrange=size(m_pilot))
        synchronize(backend); copyto!(m_pilot, m_buf)
    end
    return rows
end

function main()
    a, b, window, substeps, cap_rows, rbin = parse_cli(ARGS)
    println("pre-Y merge estimate: window=$window substeps=$substeps cap_rows=$cap_rows r=$rbin")
    println("\nA: $a")
    for ln in run_case(a; window=window, substeps=substeps, cap_rows=cap_rows, rbin=rbin)
        println(ln)
    end
    println("\nB: $b")
    for ln in run_case(b; window=window, substeps=substeps, cap_rows=cap_rows, rbin=rbin)
        println(ln)
    end
end

main()
