#!/usr/bin/env julia

using AtmosTransport.IO: MassFluxBinaryReader, load_window!

function parse_cli(argv::Vector{String})
    length(argv) >= 1 || error("Usage: polar_y_binning_impact.jl <file.bin> [--window N] [--cap-rows N]")
    file = argv[1]
    window = 1
    cap_rows = 2
    i = 2
    while i <= length(argv)
        a = argv[i]
        if a == "--window"
            i == length(argv) && error("--window requires integer")
            window = parse(Int, argv[i + 1]); i += 2
        elseif a == "--cap-rows"
            i == length(argv) && error("--cap-rows requires integer")
            cap_rows = parse(Int, argv[i + 1]); i += 2
        else
            error("Unknown arg: $a")
        end
    end
    return file, window, cap_rows
end

function main()
    file, window, cap_rows = parse_cli(ARGS)
    r = MassFluxBinaryReader(file, Float64)
    Nx, Ny, Nz = r.Nx, r.Ny, r.Nz
    m = Array{Float64}(undef, Nx, Ny, Nz)
    am = Array{Float64}(undef, Nx + 1, Ny, Nz)
    bm = Array{Float64}(undef, Nx, Ny + 1, Nz)
    cm = Array{Float64}(undef, Nx, Ny, Nz + 1)
    ps = Array{Float64}(undef, Nx, Ny)
    load_window!(m, am, bm, cm, ps, r, window)
    close(r)

    donor_is_polar(dj) = (dj <= cap_rows) || (dj >= Ny - cap_rows + 1)

    raw_max = 0.0
    raw_loc = (0, 0, 0, 0)
    @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
        b = bm[i, j, k]
        dj = b >= 0 ? j - 1 : j
        donor_is_polar(dj) || continue
        md = b >= 0 ? m[i, j - 1, k] : m[i, j, k]
        md > 0 || continue
        c = abs(b) / md
        if c > raw_max
            raw_max = c
            raw_loc = (i, j, k, dj)
        end
    end

    println("file=$file")
    println("dims=$(Nx)x$(Ny)x$(Nz) window=$window cap_rows=$cap_rows")
    println("raw polar max |bm|/m = $raw_max at (i,j,k,donor_j)=$raw_loc")

    for rr in (1, 2, 4, 8, 10, 20, 30, 60, 120, 180, 360, 720)
        Nx % rr == 0 || continue
        ngr = Nx ÷ rr
        gmax_net = 0.0
        gmax_gross = 0.0
        @inbounds for k in 1:Nz, j in 2:Ny
            for g in 1:ngr
                ist = (g - 1) * rr + 1
                ied = g * rr
                bsum = 0.0
                babs_sum = 0.0
                msum = 0.0
                for i in ist:ied
                    b = bm[i, j, k]
                    dj = b >= 0 ? j - 1 : j
                    donor_is_polar(dj) || continue
                    bsum += b
                    babs_sum += abs(b)
                    msum += (b >= 0 ? m[i, j - 1, k] : m[i, j, k])
                end
                if msum > 0
                    c_net = abs(bsum) / msum
                    c_gross = babs_sum / msum
                    gmax_net = max(gmax_net, c_net)
                    gmax_gross = max(gmax_gross, c_gross)
                end
            end
        end
        red_net = raw_max > 0 ? raw_max / gmax_net : NaN
        red_gross = raw_max > 0 ? raw_max / gmax_gross : NaN
        println("r=$(lpad(rr, 3)) net_max= $gmax_net gross_max= $gmax_gross  reduction_x(net)=$red_net reduction_x(gross)=$red_gross")
    end
end

main()
