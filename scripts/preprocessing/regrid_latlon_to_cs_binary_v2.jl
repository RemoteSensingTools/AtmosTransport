#!/usr/bin/env julia
#
# Thin CLI wrapper around the dispatch-based bilinear LatLon -> cubed-sphere
# transport-binary preprocessor. This preserves the historical fast path for
# cross-grid validation while routing through the stable transport-binary v2
# target interface.
#
# Usage:
#   julia --project=. scripts/preprocessing/regrid_latlon_to_cs_binary_v2.jl \
#       --input <latlon_binary.bin> --output <cs_binary.bin> \
#       --Nc 90

using Logging

include(joinpath(@__DIR__, "..", "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2

include(joinpath(@__DIR__, "transport_binary_v2_dispatch.jl"))
include(joinpath(@__DIR__, "transport_binary_v2_cs_bilinear.jl"))

function main(argv=ARGS)
    base_logger = ConsoleLogger(stderr, Logging.Info; show_limited=false)
    global_logger(base_logger)
    println(stderr, "[transport-v2-cs-bilinear] Logger installed, starting…")
    flush(stderr)

    target = build_transport_binary_v2_target(:cubed_sphere_bilinear, argv; FT=Float64)
    result = run_transport_binary_v2_preprocessor(target)
    println("\nDone: $(result.path)")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
