using Test
using TOML

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.Grids: HybridSigmaPressure, n_levels
using .AtmosTransport.Preprocessing:
    ECHLEVS_ML137_CFL94,
    ECHLEVS_ML137_CFL85,
    build_vertical_setup,
    select_levels_echlevs

function merge_map_from_spans(spans::Vector{UnitRange{Int}}, nz::Int)
    mm = zeros(Int, nz)
    for (g, span) in enumerate(spans)
        mm[span] .= g
    end
    return mm
end

interfaces_from_spans(spans::Vector{UnitRange{Int}}, vc) =
    vcat([vc.A[first(span)] for span in spans], vc.A[last(spans[end]) + 1])

function synthetic_l137_vc()
    a = collect(Float64, 0:137)
    b = zeros(Float64, 138)
    return HybridSigmaPressure(a, b)
end

function write_synthetic_l137_coefficients(path::AbstractString)
    open(path, "w") do io
        TOML.print(io, Dict(
            "coefficients" => Dict(
                "a" => collect(Float64, 0:137),
                "b" => zeros(Float64, 138),
            ),
        ))
    end
end

@testset "CFL-oriented ERA5 L137 echlev presets" begin
    vc = synthetic_l137_vc()

    spans_94 = vcat(
        [1:12, 13:24],
        [s:min(s + 3, 53) for s in 25:4:53],
        [k:k for k in 54:137],
    )
    selected_94, merge_map_94 = select_levels_echlevs(vc, ECHLEVS_ML137_CFL94)

    @test n_levels(selected_94) == 94
    @test length(ECHLEVS_ML137_CFL94) == 95
    @test merge_map_94 == merge_map_from_spans(spans_94, 137)
    @test selected_94.A == interfaces_from_spans(spans_94, vc)

    spans_85 = vcat(
        [1:12, 13:24],
        [s:min(s + 3, 53) for s in 25:4:53],
        [s:(s + 1) for s in 54:2:70],
        [k:k for k in 72:137],
    )
    selected_85, merge_map_85 = select_levels_echlevs(vc, ECHLEVS_ML137_CFL85)

    @test n_levels(selected_85) == 85
    @test length(ECHLEVS_ML137_CFL85) == 86
    @test merge_map_85 == merge_map_from_spans(spans_85, 137)
    @test selected_85.A == interfaces_from_spans(spans_85, vc)
end

@testset "CFL-oriented echlev presets are available through config" begin
    mktempdir() do dir
        coeff_path = joinpath(dir, "synthetic_l137.toml")
        write_synthetic_l137_coefficients(coeff_path)

        vertical_94 = build_vertical_setup(
            coeff_path, 1:137, 1000.0, Dict("echlevs" => "ml137_94L"))
        vertical_85 = build_vertical_setup(
            coeff_path, 1:137, 1000.0, Dict("echlevs" => "ml137_85L"))
        vertical_cfl94 = build_vertical_setup(
            coeff_path, 1:137, 1000.0, Dict("echlevs" => "ml137_cfl94"))
        vertical_cfl85 = build_vertical_setup(
            coeff_path, 1:137, 1000.0, Dict("echlevs" => "ml137_cfl85"))

        @test vertical_94.Nz == 94
        @test vertical_85.Nz == 85
        @test vertical_cfl94.Nz == 94
        @test vertical_cfl85.Nz == 85
        @test vertical_94.merge_map == vertical_cfl94.merge_map
        @test vertical_85.merge_map == vertical_cfl85.merge_map
        @test vertical_94.vertical_mapping_method == :merge_map
        @test vertical_85.vertical_mapping_method == :merge_map
    end
end
