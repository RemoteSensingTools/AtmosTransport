#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

function _usage()
    println("Usage: julia --project=. scripts/diagnostics/inspect_transport_binary_v2.jl <path/to/file.bin>")
end

function _print_semantics(reader::TransportBinaryReader)
    println("Semantics:")
    println("  source_flux_sampling = ", source_flux_sampling(reader))
    println("  air_mass_sampling    = ", air_mass_sampling(reader))
    println("  flux_sampling        = ", flux_sampling(reader))
    println("  flux_kind            = ", flux_kind(reader))
    println("  humidity_sampling    = ", humidity_sampling(reader))
    println("  delta_semantics      = ", delta_semantics(reader))
end

function main(args)
    if length(args) != 1
        _usage()
        return 1
    end

    path = abspath(args[1])
    isfile(path) || throw(ArgumentError("transport binary not found: $(path)"))

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
