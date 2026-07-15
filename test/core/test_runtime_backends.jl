#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

const Arch = AtmosTransport.Architectures
const Runner = AtmosTransport.Models.DrivenRunner

struct MockCUDAArray end
struct MockMetalArray end

@testset "runtime backend selection" begin
    cpu = Arch.runtime_backend_from_config(Dict("backend" => "cpu"))
    @test cpu isa Arch.CPUBackend
    @test !Arch.is_gpu_backend(cpu)
    @test Arch.backend_array_adapter(cpu) === Array
    @test Arch.backend_label(cpu) == "CPU"
    @test Arch.array_adapter_for(zeros(Float32, 2, 2)) === Array

    # Working-tree CLI scripts `include` AtmosTransport after loading CUDA or
    # Metal. In that mode package extensions cannot attach to the local module,
    # so adapter discovery must also recognize arrays through loaded runtimes.
    loaded = Dict{Base.PkgId, Any}(
        Arch._RUNTIME_PACKAGE_IDS.CUDA => (CuArray = MockCUDAArray,),
        Arch._RUNTIME_PACKAGE_IDS.Metal => (MtlArray = MockMetalArray,),
    )
    @test Arch._loaded_gpu_adapter(MockCUDAArray(), loaded) === MockCUDAArray
    @test Arch._loaded_gpu_adapter(MockMetalArray(), loaded) === MockMetalArray
    @test Arch._loaded_gpu_adapter(zeros(Float32, 2), loaded) === Array

    @test Arch.runtime_backend_from_config(Dict("use_gpu" => false)) isa Arch.CPUBackend
    @test_throws ArgumentError Arch.runtime_backend_from_config(Dict("use_gpu" => true,
                                                                      "backend" => "cpu"))
    @test_throws ArgumentError Arch.runtime_backend_from_config(Dict("backend" => "rocm"))

    architecture_source = read(joinpath(@__DIR__, "..", "..", "src", "Architectures.jl"), String)
    driven_source = read(joinpath(@__DIR__, "..", "..", "src", "Models", "DrivenSimulation.jl"), String)
    @test !occursin(r"\bisdefined\(Main|\bgetproperty\(Main|Core\.eval\(Main", architecture_source)
    @test !occursin(r"\bisdefined\(Main|\bgetproperty\(Main|Core\.eval\(Main", driven_source)
end

@testset "Metal requires Float32" begin
    metal = Arch.runtime_backend_from_config(Dict("backend" => "metal"))
    @test metal isa Arch.MetalGPUBackend
    @test Arch.assert_backend_float_type!(metal, Float32) === nothing
    @test_throws ArgumentError Arch.assert_backend_float_type!(metal, Float64)
end

@testset "runtime kernels avoid hard Float64 accumulation" begin
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    files = [
        "src/Operators/Advection/VerticalRemap.jl",
        "src/Kernels/ColumnKernels.jl",
        "src/MetDrivers/ERA5/VerticalClosure.jl",
        "src/Operators/Convection/cmfmc_kernels.jl",
    ]
    forbidden = r"Float64\(|zero\(Float64\)|::Float64"
    for file in files
        src = read(joinpath(repo, file), String)
        @test !occursin(forbidden, src)
    end
end

@testset "DrivenRunner CPU backend helpers" begin
    cfg = Dict("architecture" => Dict("backend" => "cpu"),
               "numerics" => Dict("float_type" => "Float64"))
    @test Runner._cfg_use_gpu(cfg) == false
    @test Runner._backend_array_adapter(cfg) === Array
    @test Runner._backend_label(cfg) == "CPU"
    @test Runner._synchronize_backend!(cfg) === nothing
end
