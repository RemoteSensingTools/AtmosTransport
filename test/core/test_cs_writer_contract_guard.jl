#!/usr/bin/env julia
# Tests for the cubed-sphere writer-contract guard. The writer must emit every
# runtime-read contract key when the file is created.

using Test

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.MetDrivers: validate_cs_writer_contract!

@testset "CS writer-contract guard" begin
    @testset "validate_cs_writer_contract! accepts a complete header" begin
        h = Dict{String,Any}("runtime_substep_contract" => "binary_schedule",
                             "preprocessor_contract" => "streaming_cs_v4",
                             "adaptive_substeps" => true)
        @test validate_cs_writer_contract!(h) === nothing
    end

    @testset "validate_cs_writer_contract! rejects a missing key" begin
        for drop in ("runtime_substep_contract", "preprocessor_contract", "adaptive_substeps")
            h = Dict{String,Any}("runtime_substep_contract" => "binary_schedule",
                                 "preprocessor_contract" => "streaming_cs_v4",
                                 "adaptive_substeps" => false)
            delete!(h, drop)
            err = try; validate_cs_writer_contract!(h); nothing
                  catch e; e end
            @test err isa ErrorException
            @test occursin(drop, sprint(showerror, err))
        end
    end
end
