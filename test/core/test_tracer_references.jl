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
    _tracer_transport_cfg, TransportTracerSpec,
    _seed_tracer_references!, _validate_tracer_reference_compat,
    is_offset_invariant, _apply_reference_cadence!, _reference_cadence_callbacks
using AtmosTransport.Models: RuntimePhysicsRecipe
using AtmosTransport: LinRoodPPMScheme, NoDiffusion, NoConvection, NoChemistry,
    CMFMCConvection, ExponentialDecay, TM5Convection, CMFMCMatrixConvection,
    ImplicitVerticalDiffusion, ConstantField

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

    # Stage 2: reference="global_mean" parses (seeding + gates are wired)
    cfg = deepcopy(base)
    cfg["tracers"]["co2"]["transport"] = Dict{String, Any}("reference" => "global_mean")
    specs = _parse_tracer_specs(cfg)
    @test specs[1].reference_kind === :global_mean
    @test specs[1].reference_cadence === :fixed

    # Stage 5: non-fixed cadences parse for referenced tracers
    cfg = deepcopy(base)
    cfg["tracers"]["co2"]["transport"] =
        Dict{String, Any}("reference" => "global_mean", "reference_cadence" => "daily")
    specs = _parse_tracer_specs(cfg)
    @test specs[1].reference_kind === :global_mean
    @test specs[1].reference_cadence === :daily

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

@testset "Stage 2: IC seeding converts full mass to anomaly exactly" begin
    state = _mini_cs_state(; FT = Float32)
    # make co2 spatially structured so the test is not a trivial uniform field
    raw = get_tracer_raw(state, :co2)
    for p in 1:6
        raw[p] .+= Float32(1e-5) .* reshape(collect(Float32, 1:size(raw[p], 3)), 1, 1, :)
    end
    burden_before = total_mass_full(state, :co2)
    vmr_before = mixing_ratio_full(state, :co2)

    specs = (TransportTracerSpec(:co2, Dict{String, Any}(), Dict{String, Any}(),
                                 :global_mean, :fixed),
             TransportTracerSpec(:sf6, Dict{String, Any}(), Dict{String, Any}()),)
    _seed_tracer_references!(state, specs)

    idx = tracer_index(state, :co2)
    @test tracer_reference_kind(state.tracer_refs, idx) == REF_GLOBAL_MEAN
    q_ref = tracer_reference_value(state, idx)
    @test q_ref isa Float64 && q_ref > 0

    # physical burden + VMR unchanged by seeding (to FT roundoff of the store)
    @test isapprox(total_mass_full(state, :co2), burden_before;
                   rtol = 8 * eps(Float32))
    vmr_after = mixing_ratio_full(state, :co2)
    @test all(all(isapprox.(vmr_after[p], vmr_before[p]; atol = 4e-4 * eps(Float32) * 100))
              for p in 1:6)

    # the anomaly store straddles zero (mean removed) and is small vs q_ref
    raw_after = get_tracer_raw(state, idx)
    anom_min = minimum(minimum.(raw_after))
    anom_max = maximum(maximum.(raw_after))
    @test anom_min < 0 < anom_max
    @test max(abs(anom_min), abs(anom_max)) < 0.2 * q_ref   # mean removed

    # sf6 untouched (kind REF_NONE, store unchanged semantics)
    @test tracer_reference_value(state, tracer_index(state, :sf6)) === nothing

    # IC=0 tracer: q_ref = 0 but the REFERENCED path is installed (kind flag)
    state0 = _mini_cs_state(; FT = Float32)
    raw0 = get_tracer_raw(state0, :co2)
    for p in 1:6; raw0[p] .= 0; end
    _seed_tracer_references!(state0, (TransportTracerSpec(
        :co2, Dict{String, Any}(), Dict{String, Any}(), :global_mean, :fixed),))
    i0 = tracer_index(state0, :co2)
    @test tracer_reference_kind(state0.tracer_refs, i0) == REF_GLOBAL_MEAN
    @test tracer_reference_value(state0, i0) === 0.0
    @test all(all(iszero, raw0[p]) for p in 1:6)   # anomaly == full == 0
end

@testset "Stage 2: compatibility gates name the offending operator" begin
    refspec = TransportTracerSpec(:co2, Dict{String, Any}(), Dict{String, Any}(),
                                  :global_mean, :fixed)
    nospec = TransportTracerSpec(:rn222, Dict{String, Any}(), Dict{String, Any}())
    lr = LinRoodPPMScheme{7}()
    ok_recipe = RuntimePhysicsRecipe(lr, NoDiffusion(), NoConvection(), NoChemistry())

    # all-invariant recipe: passes
    @test _validate_tracer_reference_compat((refspec, nospec), ok_recipe;
                                            air_mass_reset_mode = :none) === nothing
    # unreferenced specs: no checks at all (any recipe passes)
    bad_adv = RuntimePhysicsRecipe(PPMScheme(), NoDiffusion(), NoConvection(), NoChemistry())
    @test _validate_tracer_reference_compat((nospec,), bad_adv;
                                            air_mass_reset_mode = :preserve_vmr) === nothing

    # split-sweep advection rejected with actionable message
    err = try
        _validate_tracer_reference_compat((refspec,), bad_adv;
                                          air_mass_reset_mode = :none)
    catch e; e end
    @test err isa ArgumentError && occursin("linrood", err.msg)

    # clamped CMFMC rejected; unclamped accepted
    clamp_recipe = RuntimePhysicsRecipe(lr, NoDiffusion(),
                                        CMFMCConvection(clamp = true), NoChemistry())
    @test_throws ArgumentError _validate_tracer_reference_compat(
        (refspec,), clamp_recipe; air_mass_reset_mode = :none)
    noclamp = RuntimePhysicsRecipe(lr, NoDiffusion(),
                                   CMFMCConvection(clamp = false), NoChemistry())
    @test _validate_tracer_reference_compat((refspec,), noclamp;
                                            air_mass_reset_mode = :none) === nothing

    # decay: rejected only when it acts on the REFERENCED tracer
    decay_other = ExponentialDecay(; rn222 = 330350.0)
    okr = RuntimePhysicsRecipe(lr, NoDiffusion(), NoConvection(), decay_other)
    @test _validate_tracer_reference_compat((refspec,), okr;
                                            air_mass_reset_mode = :none) === nothing
    decay_ref = ExponentialDecay(; co2 = 1.0e6)
    badr = RuntimePhysicsRecipe(lr, NoDiffusion(), NoConvection(), decay_ref)
    @test_throws ArgumentError _validate_tracer_reference_compat(
        (refspec,), badr; air_mass_reset_mode = :none)

    # preserve-VMR window reset rejected; preserve_tracer_mass accepted
    # (the CS reset absorbs the reference shift — see the absorb testset)
    @test_throws ArgumentError _validate_tracer_reference_compat(
        (refspec,), ok_recipe; air_mass_reset_mode = :preserve_vmr)
    @test _validate_tracer_reference_compat(
        (refspec,), ok_recipe; air_mass_reset_mode = :preserve_tracer_mass) === nothing
    @test _validate_tracer_reference_compat(
        (refspec,), ok_recipe; air_mass_reset_mode = "preserve_tracer_mass") === nothing

    # pin the full trait table (codex finding: TM5 merge path n_merge > 1
    # disaggregates with fine_old/super_old tracer RATIOS — nonlinear, must
    # be rejected; n_merge = 1 is the bit-exact linear path)
    diff_op = ImplicitVerticalDiffusion(; kz_field = ConstantField{Float64, 2}(1.0))
    @test is_offset_invariant(diff_op, :co2)
    tm5_exact = TM5Convection(; n_merge = 1)
    tm5_merge = TM5Convection(; n_merge = 3)
    @test is_offset_invariant(tm5_exact, :co2)
    @test !is_offset_invariant(tm5_merge, :co2)
    mtx_exact = CMFMCMatrixConvection(; n_merge = 1)
    mtx_merge = CMFMCMatrixConvection(; n_merge = 3)
    @test is_offset_invariant(mtx_exact, :co2)
    @test !is_offset_invariant(mtx_merge, :co2)
    @test_throws ArgumentError _validate_tracer_reference_compat(
        (refspec,), RuntimePhysicsRecipe(lr, NoDiffusion(), mtx_merge, NoChemistry());
        air_mass_reset_mode = :none)
end

@testset "Stage 3: fillz negativity gate on anomaly stores" begin
    fillz! = AtmosTransport.Operators.Advection._fillz_rm_panels!
    Nc, Hp, Nz = 4, 1, 3
    mesh = CubedSphereMesh(; FT = Float32, Nc = Nc, Hp = Hp)
    Np = Nc + 2Hp
    m = ntuple(_ -> ones(Float32, Np, Np, Nz), 6)
    q_ref = 4.0e-4

    # signed anomaly, but q_full = q_anom + q_ref > 0 everywhere
    rm = ntuple(_ -> Float32(1e-5) .* (rand(Float32, Np, Np, Nz) .- 0.5f0), 6)
    before = deepcopy(rm)
    fillz!(rm, m, mesh; q_ref = q_ref)
    # gate skips: stored anomaly is bit-identical (no rm→q→rm round-trip)
    @test all(rm[p] == before[p] for p in 1:6)

    # unreferenced call on the same panels DOES run the round-trip (baseline
    # behavior preserved; values may or may not change, but the call works)
    fillz!(rm, m, mesh)

    # inject a genuine full-field negative: q_anom < -q_ref in one interior cell
    rm2 = ntuple(_ -> Float32(1e-5) .* (rand(Float32, Np, Np, Nz) .- 0.5f0), 6)
    rm2[1][Hp + 2, Hp + 2, 1] = Float32(-2 * q_ref)   # q_full = -q_ref < 0
    fillz!(rm2, m, mesh; q_ref = q_ref)
    # fillz fired and fixed the FULL field: q_full ≥ 0 on the interior.
    # ACCEPTANCE NOTE (plan 45 / codex review): the fire path is "physical
    # positivity with FT reconstruction" via the delta-form scratch — NOT an
    # F64-exact full-field repair. Cells fillz does not modify receive exactly
    # zero delta; modified cells carry FT-scale reconstruction rounding.
    qfull_min = minimum(minimum(view(rm2[p], Hp+1:Hp+Nc, Hp+1:Hp+Nc, :) ./
                                view(m[p],  Hp+1:Hp+Nc, Hp+1:Hp+Nc, :)) for p in 1:6) + q_ref
    @test qfull_min >= -4 * eps(Float32)
end

@testset "Stage 5: re-reference cadence hook" begin
    state = _mini_cs_state(; FT = Float32)
    raw = get_tracer_raw(state, :co2)
    for p in 1:6   # structured field
        raw[p] .+= Float32(2e-5) .* reshape(collect(Float32, 1:size(raw[p], 3)), 1, 1, :)
    end
    specs = (TransportTracerSpec(:co2, Dict{String, Any}(), Dict{String, Any}(),
                                 :global_mean, :daily),
             TransportTracerSpec(:sf6, Dict{String, Any}(), Dict{String, Any}()),)
    _seed_tracer_references!(state, specs)
    idx = tracer_index(state, :co2)
    q_ref0 = tracer_reference_value(state, idx)

    # drift the anomaly mean (simulates accumulated emission since the seed)
    drift = Float32(5e-6)
    for p in 1:6
        raw[p] .+= drift .* state.air_mass[p]
    end
    burden_drifted = total_mass_full(state, :co2)
    # the hook's contract: Δ = anomaly mean AT CALL TIME (which includes any
    # bounded FT residual the seed left behind, not the synthetic literal)
    mean_before = AtmosTransport.State.mass_weighted_global_mean_vmr(
        raw, state.air_mass, state.halo_width)

    # wrong boundary: no-op
    _apply_reference_cadence!(state, specs, :per_window)
    @test tracer_reference_value(state, idx) === q_ref0

    # matching boundary: q_ref absorbs the pre-hook anomaly mean; burden
    # invariant to BOUNDED FT shift roundoff (plan 45 Stage-5 contract)
    _apply_reference_cadence!(state, specs, :daily)
    q_ref1 = tracer_reference_value(state, idx)
    @test q_ref1 - q_ref0 == mean_before
    # absolute bound from the actual FT cast: per-cell |err| ≤ eps(FT)·|Δ·m|
    # (muladd: one rounding at the shift scale), summed over interior cells
    Hp = state.halo_width
    shift_err_bound = sum(sum(abs, view(state.air_mass[p],
                                        Hp+1:size(state.air_mass[p],1)-Hp,
                                        Hp+1:size(state.air_mass[p],2)-Hp, :))
                          for p in 1:6) * abs(mean_before) * eps(Float32) * 4
    @test abs(total_mass_full(state, :co2) - burden_drifted) <= shift_err_bound
    # anomaly mean re-centred at ~0 (post-shift mean ≪ the applied drift)
    post_mean = AtmosTransport.State.mass_weighted_global_mean_vmr(
        raw, state.air_mass, state.halo_width)
    @test abs(post_mean) < 1e-3 * drift

    # unreferenced tracer untouched throughout
    @test tracer_reference_value(state, tracer_index(state, :sf6)) === nothing

    # callbacks: empty unless some referenced tracer asks for per_window
    @test _reference_cadence_callbacks(specs) === NamedTuple()
    pw = (TransportTracerSpec(:co2, Dict{String, Any}(), Dict{String, Any}(),
                              :global_mean, :per_window),)
    cbs = _reference_cadence_callbacks(pw)
    @test haskey(cbs, :reference_cadence)

    # callback timing: fires exactly at the window-end iteration, not before
    state2 = _mini_cs_state(; FT = Float32)
    raw2 = get_tracer_raw(state2, :co2)
    for p in 1:6
        raw2[p] .+= Float32(2e-5) .* reshape(collect(Float32, 1:size(raw2[p], 3)), 1, 1, :)
    end
    pw2 = (TransportTracerSpec(:co2, Dict{String, Any}(), Dict{String, Any}(),
                               :global_mean, :per_window),)
    _seed_tracer_references!(state2, pw2)
    i2 = tracer_index(state2, :co2)
    for p in 1:6   # drift again so a firing visibly moves q_ref
        raw2[p] .+= Float32(3e-6) .* state2.air_mass[p]
    end
    q_before = tracer_reference_value(state2, i2)
    cb = _reference_cadence_callbacks(pw2).reference_cadence
    FakeSim = (model = (state = state2,), iteration = 3,
               current_window_end_iteration = 8)
    cb(FakeSim)                      # mid-window: must NOT fire
    @test tracer_reference_value(state2, i2) === q_before
    FakeSimEnd = (model = (state = state2,), iteration = 8,
                  current_window_end_iteration = 8)
    cb(FakeSimEnd)                   # boundary: fires exactly once
    q_after = tracer_reference_value(state2, i2)
    @test q_after != q_before
    cb(FakeSimEnd)                   # idempotent-ish: mean now ~0, tiny move
    @test abs(tracer_reference_value(state2, i2) - q_after) < 1e-3 * abs(q_after - q_before)
end

@testset "preserve_tracer_mass reset absorbs the reference shift" begin
    state = _mini_cs_state(; FT = Float32, Nz = 3)
    raw = get_tracer_raw(state, :co2)
    for p in 1:6
        raw[p] .+= Float32(2e-5) .* reshape(collect(Float32, 1:size(raw[p], 3)), 1, 1, :)
    end
    specs = (TransportTracerSpec(:co2, Dict{String, Any}(), Dict{String, Any}(),
                                 :global_mean, :fixed),)
    _seed_tracer_references!(state, specs)
    burden_co2 = total_mass_full(state, :co2)         # referenced (F64)
    burden_sf6 = total_mass_full(state, :sf6)         # unreferenced control

    # an air-mass "binary endpoint" 0.3% off the carried state
    mesh = CubedSphereMesh(; FT = Float32, Nc = 4, Hp = 1)
    new_m = ntuple(p -> state.air_mass[p] .* Float32(1.003), 6)
    AtmosTransport.Models._reset_air_mass_preserve_tracer_mass!(state, new_m, mesh)

    # FULL physical burden preserved for the REFERENCED tracer (the absorb
    # term) and exactly for the unreferenced one (stored mass untouched)
    @test isapprox(total_mass_full(state, :co2), burden_co2;
                   rtol = 8 * eps(Float32))
    @test total_mass_full(state, :sf6) == burden_sf6
    # air mass actually moved to the endpoint
    @test state.air_mass[1][2, 2, 1] ≈ new_m[1][2, 2, 1]

    # referenced with q_ref == 0 (IC=0 tracer): the absorb term is exact zero
    # — stored mass bit-identical through the reset (kind-flag path, not a
    # value test)
    state0 = _mini_cs_state(; FT = Float32, Nz = 3)
    raw0 = get_tracer_raw(state0, :co2)
    for p in 1:6; raw0[p] .= 0; end
    _seed_tracer_references!(state0, (TransportTracerSpec(
        :co2, Dict{String, Any}(), Dict{String, Any}(), :global_mean, :fixed),))
    new_m0 = ntuple(p -> state0.air_mass[p] .* Float32(1.003), 6)
    AtmosTransport.Models._reset_air_mass_preserve_tracer_mass!(state0, new_m0, mesh)
    @test all(all(iszero, get_tracer_raw(state0, :co2)[p]) for p in 1:6)
end

println("test_tracer_references.jl OK")
