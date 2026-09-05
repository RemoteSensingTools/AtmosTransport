#!/usr/bin/env julia
# Tests for the cubed-sphere writer-contract guard (2026-05-31 contract audit):
#   1. `validate_cs_writer_contract!` — the write-time assertion that every
#      runtime-read CS header key is present.
#   2. `patch_cs_runtime_substep_contract.jl` — the in-place header-patch tool
#      that retro-fits the `runtime_substep_contract` flag onto existing binaries
#      without touching the payload.
#
# The integration path (open_streaming_cs_transport_binary actually emitting the
# keys) is exercised by the payload tests; here we unit-test the guard logic and
# the data-safety-critical patch tool against synthetic headers.

using Test
using JSON3

using AtmosTransport
using .AtmosTransport.MetDrivers: validate_cs_writer_contract!

const _PATCH_TOOL = joinpath(@__DIR__, "..", "..", "scripts", "diagnostics",
                             "patch_cs_runtime_substep_contract.jl")
include(_PATCH_TOOL)   # exposes patch!, _read_header (guarded main, not run)

"""Write a synthetic CS binary: JSON header + 0x00 terminator + zero pad to
`header_bytes`, then `payload`. Mirrors `open_streaming_cs_transport_binary`."""
function _write_synthetic(path; header_bytes::Int, hdr::AbstractDict,
                          payload::Vector{UInt8} = UInt8[])
    merged = merge(Dict{String,Any}("header_bytes" => header_bytes), hdr)
    json = JSON3.write(merged)
    @assert ncodeunits(json) < header_bytes "synthetic header too big for test"
    buf = zeros(UInt8, header_bytes)
    copyto!(buf, 1, codeunits(json), 1, ncodeunits(json))  # rest stays 0x00
    open(path, "w") do io
        write(io, buf)
        write(io, payload)
    end
    return path
end

"""A minimal header the patch tool's `_validate_patch_target` accepts: a
cubed-sphere binary with a positive-integer per-window schedule whose length is
`nwin`, whose maximum equals `steps_per_window`, and a matching `nwindow`. Merge
per-test extras on top."""
cs_hdr(extra::AbstractDict = Dict{String,Any}(); spw::Int = 18, nwin::Int = 2) =
    merge(Dict{String,Any}("grid_type" => "cubed_sphere",
                           "steps_per_window" => spw,
                           "steps_per_window_by_window" => fill(spw, nwin),
                           "nwindow" => nwin), extra)

@testset "CS writer-contract guard" begin
    @testset "validate_cs_writer_contract! accepts a complete header" begin
        h = Dict{String,Any}("runtime_substep_contract" => "binary_schedule",
                             "preprocessor_contract" => "streaming_cs_v5",
                             "adaptive_substeps" => true)
        @test validate_cs_writer_contract!(h) === nothing
    end

    @testset "validate_cs_writer_contract! rejects a missing key" begin
        for drop in ("runtime_substep_contract", "preprocessor_contract", "adaptive_substeps")
            h = Dict{String,Any}("runtime_substep_contract" => "binary_schedule",
                                 "preprocessor_contract" => "streaming_cs_v5",
                                 "adaptive_substeps" => false)
            delete!(h, drop)
            err = try; validate_cs_writer_contract!(h); nothing
                  catch e; e end
            @test err isa ErrorException
            @test occursin(drop, sprint(showerror, err))
        end
    end
end

@testset "patch_cs_runtime_substep_contract tool" begin
    HB = 131072
    SENTINEL = UInt8[0xDE, 0xAD, 0xBE, 0xEF]

    mktempdir() do dir
        @testset "adds the flag, preserves header_bytes + payload + terminator" begin
            p = joinpath(dir, "a.bin")
            _write_synthetic(p; header_bytes = HB,
                             hdr = cs_hdr(Dict("global_mass_pin_target_kg" => nothing);
                                          spw = 35),
                             payload = SENTINEL)
            @test patch!(p; apply = true) === :patched
            dict, hb = _read_header(p)
            @test dict["runtime_substep_contract"] == "binary_schedule"
            @test hb == HB                                   # header_bytes untouched
            @test dict["steps_per_window"] == 35             # other keys survive
            # null terminator sits right after the JSON
            raw = read(p)
            json = JSON3.write(Dict{String,Any}(String(k) => v for (k,v) in dict))
            @test raw[ncodeunits(json) + 1] == 0x00
            # payload at offset header_bytes is byte-for-byte intact
            @test raw[HB+1:HB+4] == SENTINEL
        end

        @testset "idempotent: a patched binary is skipped" begin
            p = joinpath(dir, "b.bin")
            _write_synthetic(p; header_bytes = HB,
                             hdr = cs_hdr(Dict("runtime_substep_contract" => "binary_schedule")),
                             payload = SENTINEL)
            before = read(p)
            @test patch!(p; apply = true) === :skip
            @test read(p) == before                          # bit-identical
        end

        @testset "conflict: unexpected existing value is not overwritten" begin
            p = joinpath(dir, "c.bin")
            _write_synthetic(p; header_bytes = HB,
                             hdr = cs_hdr(Dict("runtime_substep_contract" => "something_else")))
            before = read(p)
            @test patch!(p; apply = true) === :conflict
            @test read(p) == before
        end

        @testset "overflow guard: rejects when the key would not fit" begin
            # A contract-valid CS header padded to within ~8 bytes of
            # header_bytes: the base fits, but the added ~45-byte contract key
            # overflows. Size the filler from the actual base JSON length.
            small = 256
            probe = JSON3.write(merge(Dict{String,Any}("header_bytes" => small),
                                      cs_hdr(; spw = 1, nwin = 1)))
            filler = "x"^max(small - ncodeunits(probe) - 17, 0)  # -9 pad wrapper, -8 free
            p = joinpath(dir, "d.bin")
            _write_synthetic(p; header_bytes = small,
                             hdr = cs_hdr(Dict("pad" => filler); spw = 1, nwin = 1))
            before = read(p)
            @test patch!(p; apply = true) === :overflow
            @test read(p) == before                          # untouched on overflow
        end

        @testset "dry run never writes" begin
            p = joinpath(dir, "e.bin")
            _write_synthetic(p; header_bytes = HB, hdr = cs_hdr(; spw = 18),
                             payload = SENTINEL)
            before = read(p)
            @test patch!(p; apply = false) === :would
            @test read(p) == before
        end
    end
end
