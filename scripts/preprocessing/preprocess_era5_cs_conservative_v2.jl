#!/usr/bin/env julia
#
# Thin CLI wrapper around the dispatch-based conservative LatLon -> cubed-sphere
# transport-binary preprocessor. The CS conservative path is the first concrete
# target implementation of the transport-binary v2 preprocessing interface.
#
# Usage:
#   julia --project=. scripts/preprocessing/preprocess_era5_cs_conservative_v2.jl \
#       --input <latlon_binary.bin> --output <cs_binary.bin> \
#       --Nc 90 [--cache-dir <path>]

using Logging

include(joinpath(@__DIR__, "..", "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2
using .AtmosTransportV2.Regridding

include(joinpath(@__DIR__, "transport_binary_v2_dispatch.jl"))
include(joinpath(@__DIR__, "transport_binary_v2_cs_conservative.jl"))

function main(argv=ARGS)
    base_logger = ConsoleLogger(stderr, Logging.Info; show_limited=false)
    global_logger(base_logger)
    println(stderr, "[transport-v2-cs-conservative] Logger installed, starting…")
    flush(stderr)

    target = build_transport_binary_v2_target(:cubed_sphere_conservative, argv; FT=Float64)
    result = run_transport_binary_v2_preprocessor(target)
    println("\nDone: $(result.path)")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
