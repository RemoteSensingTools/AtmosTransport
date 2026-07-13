#!/usr/bin/env julia
# Estimate CS adjoint tape size without building the full tape.

using Printf

try
    @eval using AtmosTransport
    @eval using AtmosTransport.Operators.Advection: fill_panel_halos!
catch
    include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
    @eval using .AtmosTransport
    @eval using .AtmosTransport.Operators.Advection: fill_panel_halos!
end

const USAGE = """
Usage: julia --project=. scripts/benchmarks/bench_adjoint_memory.jl \\
           [--ncs C24,C48,C90] [--nz N] [--nsteps N] \\
           [--scheme upwind|slopes|ppm] [--float-type Float32|Float64]
"""

function _parse_args(argv)
    ncs = [24, 48, 90]
    nz = 72
    nsteps = 48
    scheme = :upwind
    FT = Float32
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--ncs" && i + 1 <= length(argv)
            ncs = _parse_nc.(split(argv[i + 1], ","))
            i += 2
        elseif arg == "--nz" && i + 1 <= length(argv)
            nz = parse(Int, argv[i + 1])
            i += 2
        elseif arg == "--nsteps" && i + 1 <= length(argv)
            nsteps = parse(Int, argv[i + 1])
            i += 2
        elseif arg == "--scheme" && i + 1 <= length(argv)
            scheme = Symbol(argv[i + 1])
            scheme in (:upwind, :slopes, :ppm) ||
                error("--scheme must be upwind, slopes, or ppm")
            i += 2
        elseif arg == "--float-type" && i + 1 <= length(argv)
            value = argv[i + 1]
            FT = value == "Float32" ? Float32 :
                 value == "Float64" ? Float64 :
                 error("--float-type must be Float32 or Float64")
            i += 2
        elseif arg in ("-h", "--help")
            println(USAGE)
            exit(0)
        else
            error("Unknown argument `$arg`.\n$USAGE")
        end
    end
    return (; ncs, nz, nsteps, scheme, FT)
end

function _parse_nc(value::AbstractString)
    stripped = strip(value)
    isempty(stripped) && error("empty grid entry in --ncs")
    return startswith(lowercase(stripped), "c") ?
        parse(Int, stripped[2:end]) :
        parse(Int, stripped)
end

_scheme(::Val{:upwind}) = UpwindScheme()
_scheme(::Val{:slopes}) = SlopesScheme(NoLimiter())
_scheme(::Val{:ppm}) = PPMScheme(NoLimiter())
_scheme(s::Symbol) = _scheme(Val(s))

_halo_width(::Val{:upwind}) = 1
_halo_width(::Val{:slopes}) = 2
_halo_width(::Val{:ppm}) = 3
_halo_width(s::Symbol) = _halo_width(Val(s))

function _problem(Nc::Int, Nz::Int, nsteps::Int, FT, scheme::Symbol)
    mesh = CubedSphereMesh(; Nc=Nc, Hp=_halo_width(scheme), FT=FT)
    N = mesh.geometry.Nc + 2mesh.Hp
    panels_m = ntuple(_ -> fill(FT(1), N, N, Nz), 6)
    fill_panel_halos!(panels_m, mesh; dir=0)
    panels_am = Vector{Any}(undef, nsteps)
    panels_bm = Vector{Any}(undef, nsteps)
    panels_cm = Vector{Any}(undef, nsteps)
    for step in 1:nsteps
        panels_am[step] = ntuple(_ -> begin
            a = zeros(FT, N + 1, N, Nz)
            a[mesh.Hp + 1:mesh.Hp + Nc + 1, mesh.Hp + 1:mesh.Hp + Nc, :] .= FT(0.05)
            a
        end, 6)
        panels_bm[step] = ntuple(_ -> zeros(FT, N, N + 1, Nz), 6)
        panels_cm[step] = ntuple(_ -> zeros(FT, N, N, Nz + 1), 6)
    end
    return mesh, panels_m, panels_am, panels_bm, panels_cm
end

function _pretty_bytes(n::Integer)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = Float64(n)
    idx = 1
    while value >= 1024 && idx < length(units)
        value /= 1024
        idx += 1
    end
    return @sprintf("%.2f %s", value, units[idx])
end

function main(argv=ARGS)
    opts = _parse_args(argv)
    @printf("%-6s %-5s %-7s %-8s %-8s %-8s %-12s %-12s\n",
            "Grid", "Nz", "Steps", "Sweeps", "States", "Records",
            "State bytes", "Tape bytes")
    for Nc in opts.ncs
        mesh, panels_m, panels_am, panels_bm, panels_cm =
            _problem(Nc, opts.nz, opts.nsteps, opts.FT, opts.scheme)
        est = cs_tape_byte_estimate(
            panels_m, panels_am, panels_bm, panels_cm, mesh, _scheme(opts.scheme))
        @printf("%-6s %-5d %-7d %-8d %-8d %-8d %-12s %-12s\n",
                "C$Nc", opts.nz, opts.nsteps, est.sweep_records,
                est.state_records, est.total_records,
                _pretty_bytes(est.bytes_per_state),
                _pretty_bytes(est.state_bytes))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
