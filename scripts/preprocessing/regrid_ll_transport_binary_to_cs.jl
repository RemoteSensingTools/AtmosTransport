#!/usr/bin/env julia
# ===========================================================================
# Regrid an LL transport binary to a cubed-sphere transport binary.
#
# Thin CLI over `Preprocessing.regrid_ll_binary_to_cs` (plan 40 Commit 3).
# All of the real work — conservative regrid, wind recovery, panel-local
# rotation, flux reconstruction, per-level mass consistency, CS Poisson
# balance, cm diagnosis, streaming v4 write — lives in
# `src/Preprocessing/binary_pipeline.jl:1545`.
#
# Usage:
#   julia -t16 --project=. \
#       scripts/preprocessing/regrid_ll_transport_binary_to_cs.jl \
#       --input  <path/to/ll.bin> \
#       --output <path/to/cs.bin> \
#       --Nc 48
#       [--float-type Float32|Float64]   # default Float64 (matches LL source)
#       [--met-interval 3600.0]          # seconds per met window
#       [--dt 900.0]                     # transport timestep
#       [--mass-basis dry|moist]         # invariant 14: default dry
#       [--cache-dir ~/.cache/AtmosTransport/cr_regridding]
# ===========================================================================

using Logging
using Dates

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Preprocessing: regrid_ll_binary_to_cs, build_target_geometry

const USAGE = """
Usage: julia --project=. scripts/preprocessing/regrid_ll_transport_binary_to_cs.jl \\
           --input <ll.bin> --output <cs.bin> --Nc <int>
           [--float-type Float32|Float64] [--met-interval 3600.0] [--dt 900.0]
           [--mass-basis dry|moist] [--cache-dir <path>]
"""

function _parse_args(argv)
    input = nothing
    output = nothing
    Nc = 0
    float_type = "Float64"
    met_interval = 3600.0
    dt = 900.0
    mass_basis = "dry"
    cache_dir = ""

    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--input" && i + 1 <= length(argv)
            input = expanduser(argv[i + 1]); i += 2
        elseif arg == "--output" && i + 1 <= length(argv)
            output = expanduser(argv[i + 1]); i += 2
        elseif arg == "--Nc" && i + 1 <= length(argv)
            Nc = parse(Int, argv[i + 1]); i += 2
        elseif arg == "--float-type" && i + 1 <= length(argv)
            float_type = argv[i + 1]; i += 2
        elseif arg == "--met-interval" && i + 1 <= length(argv)
            met_interval = parse(Float64, argv[i + 1]); i += 2
        elseif arg == "--dt" && i + 1 <= length(argv)
            dt = parse(Float64, argv[i + 1]); i += 2
        elseif arg == "--mass-basis" && i + 1 <= length(argv)
            mass_basis = argv[i + 1]; i += 2
        elseif arg == "--cache-dir" && i + 1 <= length(argv)
            cache_dir = expanduser(argv[i + 1]); i += 2
        elseif arg in ("-h", "--help")
            println(USAGE); exit(0)
        else
            error("Unknown argument `$(arg)`.\n$(USAGE)")
        end
    end

    input  === nothing && error("--input required.\n$(USAGE)")
    output === nothing && error("--output required.\n$(USAGE)")
    Nc > 0 ||               error("--Nc required (positive integer).\n$(USAGE)")
    isfile(input) ||        error("Input binary not found: $(input)")
    float_type in ("Float32", "Float64") ||
        error("--float-type must be Float32 or Float64, got $(float_type)")
    mass_basis in ("dry", "moist") ||
        error("--mass-basis must be dry or moist, got $(mass_basis)")

    return (; input, output, Nc, float_type, met_interval, dt, mass_basis, cache_dir)
end

function main()
    global_logger(ConsoleLogger(stderr, Logging.Info; show_limited = false))
    opts = _parse_args(ARGS)

    FT = opts.float_type == "Float32" ? Float32 : Float64

    cfg_grid = Dict{String, Any}("Nc" => opts.Nc)
    isempty(opts.cache_dir) || (cfg_grid["regridder_cache_dir"] = opts.cache_dir)
    cs_grid = build_target_geometry(Val(:cubed_sphere), cfg_grid, FT)

    @info "LL → CS transport-binary regrid"
    @info "  input:        $(opts.input)"
    @info "  output:       $(opts.output)"
    @info "  target:       C$(opts.Nc) gnomonic CS"
    @info "  float type:   $(opts.float_type)"
    @info "  met_interval: $(opts.met_interval) s"
    @info "  dt:           $(opts.dt) s"
    @info "  mass_basis:   $(opts.mass_basis)"

    regrid_ll_binary_to_cs(opts.input, cs_grid, opts.output;
                           FT           = FT,
                           met_interval = opts.met_interval,
                           dt           = opts.dt,
                           mass_basis   = Symbol(opts.mass_basis))

    return opts.output
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
