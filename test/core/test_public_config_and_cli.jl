#!/usr/bin/env julia

using Test
using JSON3
using TOML
using Dates

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function _script_module(name::Symbol, relative_path::AbstractString)
    mod = Module(name)
    Core.eval(mod, :(include(path::AbstractString) = Base.include($mod, path)))
    Core.eval(mod, :(eval(expr) = Core.eval($mod, expr)))
    withenv("ATMOSTR_NO_AUTO_THREADS" => "1") do
        Base.include(mod, joinpath(REPO_ROOT, relative_path))
    end
    return mod
end

@testset "runtime editor schema follows public config contract" begin
    schema_path = joinpath(REPO_ROOT, "schemas", "atmos_transport_run.schema.json")
    schema = JSON3.read(read(schema_path, String))
    @test schema[Symbol("\$schema")] == "http://json-schema.org/draft-07/schema#"

    properties = schema.properties
    @test hasproperty(properties, :advection)
    @test !hasproperty(properties.run.properties, :scheme)

    @test Set(String.(properties.advection.properties.scheme.enum)) ==
          Set(["upwind", "slopes", "ppm", "linrood", "none"])
    @test Set(String.(properties.convection.properties.kind.enum)) ==
          Set(["none", "tm5", "cmfmc", "cmfmc_matrix"])
    @test Set(String.(properties.diffusion.properties.kind.enum)) == Set([
        "none", "constant", "tm5_beljaars_viterbo_local_kz",
        "beljaars_viterbo_local_kz", "pbl",
        "geoschem_holtslag_boville_vdiff", "precomputed_kz",
    ])

    output = schema.definitions.output.properties
    for key in (:enabled, :format, :path, :hours, :cadence_hours, :split,
                :deflate_level, :shuffle, :fields)
        @test hasproperty(output, key)
    end
    for retired in (:mode, :path_template, :frequency, :provenance)
        @test !hasproperty(output, retired)
    end
    temporal_schemes = schema.definitions.surface_flux.properties.temporal_scheme.enum
    @test Set(String.(temporal_schemes)) == Set(["stepwise", "linear", "conservative"])

    for example in ("atmos_transport_schema_demo.toml", "minimal_template.toml")
        cfg = TOML.parsefile(joinpath(REPO_ROOT, "config", "examples", example))
        @test haskey(cfg, "advection")
        @test !haskey(get(cfg, "run", Dict{String, Any}()), "scheme")
        @test haskey(cfg["output"], "path")
        @test haskey(cfg["output"], "hours")
    end
end

@testset "canonical command-line parsers fail closed" begin
    cfg = joinpath(REPO_ROOT, "config", "examples", "minimal_template.toml")

    run_cli = _script_module(:RunTransportCLITest, "scripts/run_transport.jl")
    run_parse = getfield(run_cli, :_parse_cli)
    @test run_parse(["--help"]) === nothing
    @test run_parse([cfg]) == cfg
    @test_throws ArgumentError run_parse(String[])
    @test_throws ArgumentError run_parse([cfg, "unexpected"])
    @test_throws ArgumentError run_parse(["--unknown"])

    preprocess_cli = _script_module(
        :PreprocessTransportCLITest,
        "scripts/preprocessing/preprocess_transport_binary.jl",
    )
    preprocess_parse = getfield(preprocess_cli, :_parse_cli)
    @test preprocess_parse([cfg, "--day", "2021-12-01"])[2] == "2021-12-01"
    @test preprocess_parse([
        cfg, "--start", "2021-12-01", "--end", "2021-12-03",
    ])[3:4] == ("2021-12-01", "2021-12-03")
    @test_throws ArgumentError preprocess_parse([cfg, "--unknown"])
    @test_throws ArgumentError preprocess_parse([cfg, "--day"])
    @test_throws ArgumentError preprocess_parse([
        cfg, "--day", "2021-12-01", "--start", "2021-12-01", "--end", "2021-12-03",
    ])
    @test_throws ArgumentError preprocess_parse([cfg, "--start", "2021-12-03"])
    @test_throws ArgumentError preprocess_parse([
        cfg, "--start", "2021-12-03", "--end", "2021-12-01",
    ])

    download_cli = _script_module(:DownloadDataCLITest, "scripts/downloads/download_data.jl")
    download_parse = getfield(download_cli, :_parse_cli)
    opts = download_parse([
        cfg, "--start", "2021-12-01", "--end", "2021-12-03", "--dry-run",
    ])
    @test opts.start_date == Date(2021, 12, 1)
    @test opts.end_date == Date(2021, 12, 3)
    @test opts.dry_run
    @test !opts.verify
    @test_throws ArgumentError download_parse([cfg, "--unknown"])
    @test_throws ArgumentError download_parse([cfg, "--start"])
    @test_throws ArgumentError download_parse([cfg, "--dry-run", "--dry-run"])
    @test_throws ArgumentError download_parse([
        cfg, "--start", "2021-12-03", "--end", "2021-12-01",
    ])
end
