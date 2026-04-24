#!/usr/bin/env julia
#
# Thin CLI over `AtmosTransport.inspect_binary` (plan 40 Commit 5).
# Auto-detects LL/RG vs CS binaries, runs load-time gates, prints a
# capability-augmented report, and returns the capability NamedTuple.

using ArgParse

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

function _argparse_settings()
    s = ArgParseSettings(
        description = "Inspect a transport binary — header, grid, capability summary, and driver compatibility.",
        prog = "inspect_transport_binary.jl")
    @add_arg_table! s begin
        "--allow-legacy"
            action = :store_true
            help = "Demote contract-violation ArgumentError to warning, so " *
                   "pre-plan-39 binaries without the 8 self-describing fields " *
                   "can be inspected (runtime behavior NOT trusted)."
        "path"
            arg_type = String
            required = true
            help = "path to the transport binary .bin file (LL/RG or CS)"
    end
    return s
end

function main(args)
    parsed = parse_args(args, _argparse_settings())
    path = abspath(parsed["path"])

    if parsed["allow-legacy"]
        ENV["ATMOSTR_ALLOW_LEGACY_BINARY"] = "1"
        @info "inspect: --allow-legacy set; contract violations demoted to warnings"
    end

    # `inspect_binary` prints a rich report (header, geometry, semantics,
    # payload sections, capability rows with ✓/✗) and returns a
    # `binary_capabilities` NamedTuple we could use for scripting.
    inspect_binary(path)

    # Driver-compatibility probe: does the binary produce a working
    # runtime driver? Useful for catching semantic mismatches beyond
    # payload-section presence (e.g. contract-field validation).
    println()
    try
        if inspect_binary(path; io = devnull).grid_type === :cubed_sphere
            driver = CubedSphereTransportDriver(path; FT = Float64, arch = CPU())
            println("Driver: OK (CubedSphereTransportDriver)")
            println(driver)
            close(driver)
        else
            driver = TransportBinaryDriver(path; FT = Float64, arch = CPU())
            println("Driver: OK (TransportBinaryDriver)")
            println(driver)
            close(driver)
        end
    catch err
        println("Driver: incompatible with current src runtime")
        println("  ", sprint(showerror, err))
    end

    return 0
end

exit(main(ARGS))
