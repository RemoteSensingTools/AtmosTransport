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
# Timestep metadata (`dt_met_seconds`, `steps_per_window`) is read from the
# source binary header — there is no `--met-interval` or `--dt` flag,
# because resampling to a different substep count is not a safe binary-to-
# binary operation (flux_kind = :substep_mass_amount).
#
# Usage:
#   julia -t16 --project=. \
#       scripts/preprocessing/regrid_ll_transport_binary_to_cs.jl \
#       --input  <path/to/ll.bin> \
#       --output <path/to/cs.bin> \
#       --Nc 48
#       [--float-type Float32|Float64]   # default Float64 (matches LL source)
#       [--mass-basis dry|moist]         # default: match source header
#       [--convention gnomonic|geos_native]
#       [--cache-dir ~/.cache/AtmosTransport/cr_regridding]
#       [--steps-per-window 12]          # override source's substep count
#                                         # (smaller per-substep flux; needed
#                                         # for high-res CS output that
#                                         # otherwise fails the positivity gate)
# ===========================================================================

using Logging
using Dates

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Preprocessing: regrid_ll_binary_to_cs, build_target_geometry

const USAGE = """
Usage: julia --project=. scripts/preprocessing/regrid_ll_transport_binary_to_cs.jl \\
           --input <ll.bin> --output <cs.bin> --Nc <int>
           [--float-type Float32|Float64] [--mass-basis dry|moist]
           [--convention gnomonic|geos_native] [--cache-dir <path>]
           [--steps-per-window <int>] [--allow-positivity-violation]
"""

function _parse_args(argv)
    input = nothing
    output = nothing
    Nc = 0
    float_type = "Float64"
    mass_basis = nothing     # nothing = match source header
    convention = "gnomonic"
    cache_dir = ""
    steps_per_window = nothing  # nothing = match source header
    require_substep_positivity = true

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
        elseif arg == "--mass-basis" && i + 1 <= length(argv)
            mass_basis = argv[i + 1]; i += 2
        elseif arg == "--convention" && i + 1 <= length(argv)
            convention = argv[i + 1]; i += 2
        elseif arg == "--cache-dir" && i + 1 <= length(argv)
            cache_dir = expanduser(argv[i + 1]); i += 2
        elseif arg == "--steps-per-window" && i + 1 <= length(argv)
            steps_per_window = parse(Int, argv[i + 1]); i += 2
        elseif arg == "--allow-positivity-violation"
            require_substep_positivity = false; i += 1
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
    mass_basis === nothing || mass_basis in ("dry", "moist") ||
        error("--mass-basis must be dry or moist, got $(mass_basis)")
    norm_convention = lowercase(replace(convention, '-' => '_', ' ' => '_'))
    norm_convention in ("gnomonic", "gnomic", "geos_native", "geosnative") ||
        error("--convention must be gnomonic or geos_native, got $(convention)")
    convention = norm_convention in ("geos_native", "geosnative") ? "geos_native" : "gnomonic"

    steps_per_window === nothing || steps_per_window >= 1 ||
        error("--steps-per-window must be ≥ 1, got $(steps_per_window)")

    return (; input, output, Nc, float_type, mass_basis, convention, cache_dir,
              steps_per_window, require_substep_positivity)
end

function main()
    global_logger(ConsoleLogger(stderr, Logging.Info; show_limited = false))
    opts = _parse_args(ARGS)

    FT = opts.float_type == "Float32" ? Float32 : Float64

    cfg_grid = Dict{String, Any}(
        "Nc" => opts.Nc,
        "panel_convention" => opts.convention,
    )
    isempty(opts.cache_dir) || (cfg_grid["regridder_cache_dir"] = opts.cache_dir)
    cs_grid = build_target_geometry(Val(:cubed_sphere), cfg_grid, FT)

    basis_sym = opts.mass_basis === nothing ? nothing : Symbol(opts.mass_basis)

    @info "LL → CS transport-binary regrid"
    @info "  input:      $(opts.input)"
    @info "  output:     $(opts.output)"
    @info "  target:     C$(opts.Nc) $(opts.convention) CS"
    @info "  float type: $(opts.float_type)"
    @info "  mass_basis: $(basis_sym === nothing ? "(match source)" : basis_sym)"

    opts.steps_per_window === nothing ||
        @info "  steps_per_window override: $(opts.steps_per_window)"

    regrid_ll_binary_to_cs(opts.input, cs_grid, opts.output;
                           FT         = FT,
                           mass_basis = basis_sym,
                           steps_per_window = opts.steps_per_window,
                           require_substep_positivity = opts.require_substep_positivity)

    return opts.output
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
