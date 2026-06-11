# ---------------------------------------------------------------------------
# TracerReferences — reference-state (anomaly transport) plumbing tests
#
# Plan 45 Stage 0/1: the carrier itself, its wiring into CubedSphereState,
# constructor defaults (all-REF_NONE = raw path), Adapt pass-through, and the
# [tracers.X.transport] config parse. The kind-flag contract (path selection
# by flag, never by `q_ref == 0`) is asserted explicitly.
# ---------------------------------------------------------------------------

using Test
using Adapt
using AtmosTransport
using AtmosTransport.State: TracerReferences, REF_NONE, REF_GLOBAL_MEAN,
    tracer_reference_kind, tracer_reference_value, set_tracer_reference!,
    any_tracer_referenced, DryBasis, tracer_index, set_uniform_mixing_ratio!
using AtmosTransport.Grids: CubedSphereMesh
using AtmosTransport: HybridSigmaPressure, AtmosGrid, allocate_face_fluxes,
    strang_split!, PPMScheme
using AtmosTransport.Models.DrivenRunner: _parse_tracer_specs,
    _tracer_transport_cfg, TransportTracerSpec

@testset "TracerReferences carrier" begin
    refs = TracerReferences(3)
    @test length(refs.kind) == 3
    @test all(==(REF_NONE), refs.kind)
    @test all(==(0.0), refs.q_ref)
    @test !any_tracer_referenced(refs)
    @test tracer_reference_value(refs, 1) === nothing

    set_tracer_reference!(refs, 2, REF_GLOBAL_MEAN, 412.5e-6)
    @test any_tracer_referenced(refs)
    @test tracer_reference_kind(refs, 2) == REF_GLOBAL_MEAN
    @test tracer_reference_value(refs, 2) == 412.5e-6
    # q_ref is stored F64 regardless of input type
    @test refs.q_ref[2] isa Float64

    # kind-flag contract: a referenced tracer with q_ref == 0 still reports a
    # value (0.0), NOT `nothing` — the referenced code path must run.
    set_tracer_reference!(refs, 3, REF_GLOBAL_MEAN, 0.0)
    @test tracer_reference_value(refs, 3) === 0.0
    @test tracer_reference_value(refs, 1) === nothing   # unreferenced untouched

    @test_throws ArgumentError set_tracer_reference!(refs, 1, 0x7f, 1.0)
    @test_throws DimensionMismatch TracerReferences(fill(REF_NONE, 2), zeros(3))
end

function _mini_cs_state(; Nc = 4, Hp = 1, Nz = 3, FT = Float32)
    mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp)
    Np = Nc + 2Hp
    air = ntuple(_ -> ones(FT, Np, Np, Nz), 6)
    co2 = ntuple(_ -> fill(FT(400e-6), Np, Np, Nz), 6)
    sf6 = ntuple(_ -> fill(FT(10e-12), Np, Np, Nz), 6)
    state = CubedSphereState(DryBasis, mesh, air; co2 = co2, sf6 = sf6)
    return state
end

@testset "CubedSphereState carries tracer_refs (default raw)" begin
    state = _mini_cs_state()
    @test :tracer_refs in propertynames(state)
    refs = state.tracer_refs
    @test refs isa TracerReferences
    @test length(refs.kind) == 2
    @test !any_tracer_referenced(refs)
    @test tracer_reference_value(state, 1) === nothing
    @test tracer_reference_kind(state, 2) == REF_NONE

    # in-place mutation through the state-level carrier
    set_tracer_reference!(state.tracer_refs, 1, REF_GLOBAL_MEAN, 4.0e-4)
    @test tracer_reference_value(state, 1) === 4.0e-4

    # Adapt passes the SAME host carrier through (mutation stays visible)
    adapted = Adapt.adapt(Array, state)
    @test adapted.tracer_refs === state.tracer_refs
    @test tracer_reference_value(adapted, 1) === 4.0e-4

    # carrier length must match the tracer axis
    @test_throws DimensionMismatch CubedSphereState(
        DryBasis, state.air_mass, state.tracers_raw, state.tracer_names;
        halo_width = state.halo_width, tracer_refs = TracerReferences(5))
end

@testset "[tracers.X.transport] config parse" begin
    base = Dict{String, Any}(
        "tracers" => Dict{String, Any}(
            "co2" => Dict{String, Any}(
                "init" => Dict{String, Any}("kind" => "uniform",
                                            "background" => 4.0e-4))))

    specs = _parse_tracer_specs(base)
    @test length(specs) == 1
    @test specs[1].reference_kind === :none
    @test specs[1].reference_cadence === :fixed

    # explicit reference = "none" parses (with any cadence)
    cfg = deepcopy(base)
    cfg["tracers"]["co2"]["transport"] =
        Dict{String, Any}("reference" => "none", "reference_cadence" => "daily")
    specs = _parse_tracer_specs(cfg)
    @test specs[1].reference_kind === :none
    @test specs[1].reference_cadence === :daily

    # unknown key, unknown reference value, unknown cadence: parse-time errors
    for bad in (Dict{String, Any}("referenc" => "none"),
                Dict{String, Any}("reference" => "column_mean"),
                Dict{String, Any}("reference_cadence" => "hourly"))
        cfg = deepcopy(base)
        cfg["tracers"]["co2"]["transport"] = bad
        @test_throws ArgumentError _parse_tracer_specs(cfg)
    end

    # Stage-0 guard: reference="global_mean" rejected until seeding ships
    # (plan 45 Stage 2 removes this — flip the test to expect success then)
    cfg = deepcopy(base)
    cfg["tracers"]["co2"]["transport"] = Dict{String, Any}("reference" => "global_mean")
    @test_throws ArgumentError _parse_tracer_specs(cfg)

    # back-compat 3-arg spec constructor defaults to the raw path
    spec = TransportTracerSpec(:x, Dict{String, Any}(), Dict{String, Any}())
    @test spec.reference_kind === :none && spec.reference_cadence === :fixed
end

@testset "validate_config preflights [tracers.X.transport]" begin
    cfg = Dict{String, Any}(
        "tracers" => Dict{String, Any}(
            "co2" => Dict{String, Any}(
                "init" => Dict{String, Any}("kind" => "uniform",
                                            "background" => 4.0e-4),
                "transport" => Dict{String, Any}("referenc" => "none"))))  # typo
    ok, errors = AtmosTransport.validate_config(cfg)
    @test !ok
    @test any(occursin("transport", e) for e in errors)
end

@testset "split-sweep runtime guard rejects referenced tracers" begin
    state = _mini_cs_state(; Nz = 2)
    Nc, Hp, Nz = 4, 1, 2
    mesh = CubedSphereMesh(; FT = Float32, Nc = Nc, Hp = Hp)
    vc = HybridSigmaPressure(Float32[0, 100, 300], Float32[0, 0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = Float32)
    fluxes = allocate_face_fluxes(mesh, Nz; FT = Float32, basis = DryBasis)

    # unreferenced: reaches the workspace check, NOT the reference guard
    err_unref = try
        strang_split!(state, fluxes, grid, PPMScheme(); workspace = nothing)
        nothing
    catch e
        e
    end
    @test err_unref isa ArgumentError
    @test !occursin("reference-state", err_unref.msg)

    # referenced: the split-sweep guard fires first with the actionable message
    set_tracer_reference!(state.tracer_refs, 1, REF_GLOBAL_MEAN, 4.0e-4)
    err_ref = try
        strang_split!(state, fluxes, grid, PPMScheme(); workspace = nothing)
        nothing
    catch e
        e
    end
    @test err_ref isa ArgumentError
    @test occursin("reference-state", err_ref.msg)
    @test occursin("linrood", err_ref.msg)
end

@testset "_raw/_full accessors recover the physical field" begin
    state = _mini_cs_state(; FT = Float32)
    m = state.air_mass
    Hp = state.halo_width

    # raw == get_tracer (alias), full == raw for unreferenced (zero-copy)
    @test get_tracer_raw(state, :co2)[1] === get_tracer(state, :co2)[1]
    @test get_tracer_full(state, :co2)[1] === get_tracer(state, :co2)[1]
    tm_raw = total_mass_full(state, :co2)
    @test tm_raw isa Float64

    # synthetic reference: convert the stored field to an anomaly against
    # q_ref, then assert the _full accessors recover the original physical
    # field / burden to F64 round-off.
    q_ref = 4.0e-4
    full_before = deepcopy(get_tracer(state, :co2))           # physical field
    burden_before = total_mass_full(state, :co2)
    raw = get_tracer_raw(state, :co2)
    for p in 1:6
        raw[p] .= raw[p] .- Float32(q_ref) .* m[p]            # seed anomaly
    end
    set_tracer_reference!(state.tracer_refs, tracer_index(state, :co2),
                          REF_GLOBAL_MEAN, q_ref)

    recovered = get_tracer_full(state, :co2)
    @test all(all(isapprox.(recovered[p], full_before[p];
                        rtol = 4 * eps(Float32))) for p in 1:6)
    @test isapprox(total_mass_full(state, :co2), burden_before;
                   rtol = 8 * eps(Float32))
    vmr = mixing_ratio_full(state, :co2)
    @test all(all(isapprox.(vmr[p], full_before[p] ./ m[p];
                        rtol = 4 * eps(Float32))) for p in 1:6)

    # the OTHER tracer stays untouched and raw-pathed
    @test get_tracer_full(state, :sf6)[1] === get_tracer(state, :sf6)[1]

    # full-field semantic writes are rejected for referenced tracers
    @test_throws ArgumentError set_uniform_mixing_ratio!(state, :co2, 1.0e-6)
    @test set_uniform_mixing_ratio!(state, :sf6, 1.0e-12) === nothing
end

@testset "accessor discipline: no bare accessors in output namespaces" begin
    # Physical-field namespaces must use get_tracer_full/_raw etc. explicitly —
    # a bare `get_tracer(`/`total_mass(`/`mixing_ratio(` there would silently
    # read ANOMALY mass for referenced tracers. Same freshness-gate pattern as
    # test_readme_current.jl: grep the sources, fail loudly with file:line.
    src_root = normpath(joinpath(@__DIR__, "..", "..", "src"))
    gated_dirs = [joinpath(src_root, "Output"),
                  joinpath(src_root, "Models"),
                  normpath(joinpath(@__DIR__, "..", "..", "scripts", "diagnostics"))]
    bare = r"(?<![_a-zA-Z])(get_tracer|total_mass|mixing_ratio)\("
    offenders = String[]
    for dir in gated_dirs
        isdir(dir) || continue
        for (root, _, files) in walkdir(dir)
            for f in files
                endswith(f, ".jl") || continue
                path = joinpath(root, f)
                for (i, line) in enumerate(eachline(path))
                    startswith(strip(line), "#") && continue
                    occursin(bare, line) &&
                        push!(offenders, "$(path):$(i): $(strip(line))")
                end
            end
        end
    end
    if !isempty(offenders)
        @info "bare tracer accessors in gated namespaces (use _raw/_full):" offenders
    end
    @test isempty(offenders)
end

println("test_tracer_references.jl OK")
