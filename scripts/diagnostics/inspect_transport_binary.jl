#!/usr/bin/env julia

using ArgParse

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

function _print_semantics(reader::TransportBinaryReader)
    println("Semantics:")
    println("  source_flux_sampling = ", source_flux_sampling(reader))
    println("  air_mass_sampling    = ", air_mass_sampling(reader))
    println("  flux_sampling        = ", flux_sampling(reader))
    println("  flux_kind            = ", flux_kind(reader))
    println("  humidity_sampling    = ", humidity_sampling(reader))
    println("  delta_semantics      = ", delta_semantics(reader))
end

function _argparse_settings()
    s = ArgParseSettings(
        description = "Inspect a transport binary — header, grid, semantics, and driver compatibility.",
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
            help = "path to the transport binary .bin file"
    end
    return s
end

function main(args)
    parsed = parse_args(args, _argparse_settings())
    path = abspath(parsed["path"])
    isfile(path) || throw(ArgumentError("transport binary not found: $(path)"))

    if parsed["allow-legacy"]
        ENV["ATMOSTR_ALLOW_LEGACY_BINARY"] = "1"
        @info "inspect: --allow-legacy set; contract violations demoted to warnings"
    end

    reader = TransportBinaryReader(path; FT=Float64)
    println(reader)
    println()
    println("Header:")
    println(reader.header)
    println()
    println("Grid:")
    println(load_grid(reader; FT=Float64, arch=CPU()).horizontal)
    println()
    _print_semantics(reader)
    println()
    println("Sections:")
    println("  payload_sections = ", join(String.(reader.header.payload_sections), ", "))
    println("  has_qv           = ", has_qv(reader))
    println("  has_qv_endpoints = ", has_qv_endpoints(reader))
    println("  has_flux_delta   = ", has_flux_delta(reader))
    println()

    try
        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU())
        println("Driver: OK")
        println(driver)
        close(driver)
    catch err
        println("Driver: incompatible with current src runtime")
        println("  ", sprint(showerror, err))
    end

    close(reader)
    return 0
end

exit(main(ARGS))
