#!/usr/bin/env julia

using Printf

include(joinpath(@__DIR__, "..", "run_transport_binary_v2.jl"))
using .AtmosTransportV2

function main()
    length(ARGS) >= 1 || error("Usage: julia --project=. scripts/diagnostics/check_window_endpoint_v2.jl binary_path [scheme=upwind] [nwindows=6]")
    binary_path = expanduser(ARGS[1])
    scheme_name = length(ARGS) >= 2 ? Symbol(lowercase(ARGS[2])) : :upwind
    nwindows = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 6

    driver = TransportBinaryDriver(binary_path; FT=Float64, arch=CPU())
    try
        init_cfg = Dict{String, Any}("kind" => "uniform", "background" => 4.0e-4)
        model = make_model(driver; FT=Float64, scheme_name=scheme_name, tracer_name=:CO2, init_cfg=init_cfg)
        sim = DrivenSimulation(model, driver; start_window=1, stop_window=total_windows(driver), initialize_air_mass=true)

        local_max = 0.0
        global_max = 0.0
        last_window = min(nwindows, total_windows(driver) - 1)

        for win in 1:last_window
            run_window!(sim)
            nxt = load_transport_window(driver, win + 1)
            local_rel = maximum(abs.(model.state.air_mass .- nxt.air_mass)) / maximum(abs.(nxt.air_mass))
            global_rel = abs(sum(model.state.air_mass) - sum(nxt.air_mass)) / sum(nxt.air_mass)
            local_max = max(local_max, local_rel)
            global_max = max(global_max, global_rel)
            @info @sprintf("window %d -> %d: local air-mass mismatch %.3e | global %.3e", win, win + 1, local_rel, global_rel)
        end

        @info @sprintf("max local mismatch over %d checked windows: %.3e", last_window, local_max)
        @info @sprintf("max global mismatch over %d checked windows: %.3e", last_window, global_max)
    finally
        close(driver)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
