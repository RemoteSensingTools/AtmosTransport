#!/usr/bin/env julia

using Test
using TOML

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport
const TapeMod = AtmosTransport.Tape

# ---------------------------------------------------------------------------
# Plan 26 Phase A.1 — `MmapCSTapeStorage` on-disk tape roundtrip tests.
#
# Verifies:
#   * Slot allocation bumps the cursor by the expected number of bytes.
#   * stage_panels! → _tape_panels yields bit-exact equality.
#   * Multiple staged slots stay disjoint and readable in any order.
#   * Heterogeneous panel shapes (am / bm / cm-shaped tapes) survive the
#     roundtrip with the per-panel offset bookkeeping.
#   * finalize_tape! emits a manifest.toml that captures every record and
#     marks the storage `finalised`.
#   * The CPU mmap path is bit-exact against the in-memory
#     DeviceCSTapeStorage path for cs_surface_emission_footprint across
#     upwind, linear PPM, monotone PPM, and emission-bearing midpoints.
# ---------------------------------------------------------------------------

function _shape_panel(::Val{:m}, Nc, Hp, Nz, FT)
    N = Nc + 2Hp
    a = zeros(FT, N, N, Nz)
    for k in 1:Nz, j in 1:N, i in 1:N
        a[i, j, k] = FT(1000 + 100i + 10j + k)
    end
    a
end

_shape_panel(::Val{:am}, Nc, Hp, Nz, FT) =
    fill!(zeros(FT, Nc + 2Hp + 1, Nc + 2Hp, Nz), FT(0.5))
_shape_panel(::Val{:bm}, Nc, Hp, Nz, FT) =
    fill!(zeros(FT, Nc + 2Hp, Nc + 2Hp + 1, Nz), FT(-0.25))
_shape_panel(::Val{:cm}, Nc, Hp, Nz, FT) =
    fill!(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz + 1), FT(0.125))

function _build_panels(kind::Symbol; Nc = 3, Hp = 3, Nz = 4, FT = Float32, seed = 1)
    return ntuple(p -> begin
        a = _shape_panel(Val(kind), Nc, Hp, Nz, FT)
        a .+= FT(0.001) * FT(seed) * FT(p)
        a
    end, 6)
end

@testset "MmapCSTapeStorage roundtrip" begin
    @testset "single-slot bit-exact roundtrip" begin
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            src = _build_panels(:m; Nc = 4, Hp = 3, Nz = 5, FT = Float64)
            slot = TapeMod._stage_panels(storage, src)

            @test slot.record_id == 1
            @test slot.eltype === Float64
            expected_offsets = let
                stride = sizeof(Float64) * length(src[1])
                (0, stride, 2stride, 3stride, 4stride, 5stride)
            end
            @test slot.offsets == expected_offsets
            @test storage.cursor == 6 * sizeof(Float64) * length(src[1])

            got = TapeMod._tape_panels(slot)
            @test length(got) == 6
            for p in 1:6
                @test got[p] == src[p]
                @test eltype(got[p]) === Float64
                @test size(got[p]) == size(src[p])
            end

            TapeMod.finalize_tape!(storage)
            @test isfile(joinpath(dir, "manifest.toml"))
            @test filesize(joinpath(dir, "records.bin")) == storage.cursor

            m = TOML.parsefile(joinpath(dir, "manifest.toml"))
            @test m["meta"]["finalised"] === true
            @test m["meta"]["record_count"] == 1
            @test m["meta"]["version"] == "v1"
            @test m["meta"]["endianness"] in ("little", "big")
            @test length(m["record"]) == 1
            @test m["record"][1]["eltype"] == "Float64"
            @test m["record"][1]["offsets"] == collect(expected_offsets)
        end
    end

    @testset "multiple slots stay disjoint and out-of-order readable" begin
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            srcs = [_build_panels(:m; Nc = 3, Hp = 3, Nz = 3, FT = Float32, seed = s)
                    for s in 1:4]
            slots = [TapeMod._stage_panels(storage, src) for src in srcs]

            # Read in reverse to mirror the adjoint walk order.
            for idx in 4:-1:1
                got = TapeMod._tape_panels(slots[idx])
                for p in 1:6
                    @test got[p] == srcs[idx][p]
                end
            end

            # Each slot should sit at a unique base offset.
            base_offsets = [slot.offsets[1] for slot in slots]
            @test length(unique(base_offsets)) == 4
            @test issorted(base_offsets)
            TapeMod.finalize_tape!(storage)
        end
    end

    @testset "heterogeneous panel shapes (am, bm, cm)" begin
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            am = _build_panels(:am; FT = Float32, seed = 7)
            bm = _build_panels(:bm; FT = Float32, seed = 8)
            cm = _build_panels(:cm; FT = Float32, seed = 9)

            slot_am = TapeMod._stage_panels(storage, am)
            slot_bm = TapeMod._stage_panels(storage, bm)
            slot_cm = TapeMod._stage_panels(storage, cm)

            for (slot, src) in ((slot_am, am), (slot_bm, bm), (slot_cm, cm))
                got = TapeMod._tape_panels(slot)
                for p in 1:6
                    @test size(got[p]) == size(src[p])
                    @test got[p] == src[p]
                end
            end
            TapeMod.finalize_tape!(storage)
        end
    end

    @testset "finaliser is idempotent and survives missing files" begin
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            TapeMod._stage_panels(storage,
                _build_panels(:m; Nc = 2, Hp = 2, Nz = 2, FT = Float32))
            TapeMod.finalize_tape!(storage)
            @test storage.finalised === true
            @test storage.closed === true
            # Second call is a safe no-op.
            TapeMod.finalize_tape!(storage)
            @test storage.closed === true
        end
    end

    @testset "default :mmap dispatch returns a temp-dir MmapCSTapeStorage" begin
        s = TapeMod._tape_storage(:mmap)
        try
            @test s isa TapeMod.MmapCSTapeStorage
            @test isdir(s.dir)
            @test s.cleanup_on_finalize === true
        finally
            TapeMod.finalize_tape!(s)
        end
        # Temp dir should be removed after finalize_tape! when cleanup is on.
        @test !isdir(s.dir)
    end

    @testset "argument errors on closed / finalised storage" begin
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            TapeMod.finalize_tape!(storage)
            @test_throws ArgumentError TapeMod._stage_panels(storage,
                _build_panels(:m; Nc = 2, Hp = 2, Nz = 2, FT = Float32))
        end
    end
end

# ---------------------------------------------------------------------------
# Cross-policy parity: :mmap must be bit-exact against :device on CPU.
# ---------------------------------------------------------------------------

const Adv = AT.Operators.Advection

function _trivial_problem(; Nc = 4, Nz = 3, nsteps = 2, FT = Float64,
                          nontrivial::Bool = false)
    mesh = AT.CubedSphereMesh(Nc = Nc, Hp = 3, FT = FT)
    N = mesh.Nc + 2 * mesh.Hp
    Hp = mesh.Hp
    panels_m = ntuple(p -> begin
        m = zeros(FT, N, N, Nz)
        for k in 1:Nz, j in 1:N, i in 1:N
            m[i, j, k] = FT(2.0 + 0.25 * k + 0.01 * p)
        end
        m
    end, 6)
    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    Adv.fill_panel_halos!(panels_m, mesh; dir = 0)
    Adv.fill_panel_halos!(panels_rm, mesh; dir = 0)

    if nontrivial
        am_steps = [ntuple(p -> begin
            a = zeros(FT, N + 1, N, Nz)
            for k in 1:Nz, j in Hp + 1:Hp + Nc, i in Hp + 1:Hp + Nc + 1
                a[i, j, k] = FT(0.015) *
                    sin(FT(0.2 * step + 0.3 * p + 0.7 * i + 0.4 * j + 0.2 * k))
            end
            a
        end, 6) for step in 1:nsteps]
        bm_steps = [ntuple(p -> begin
            b = zeros(FT, N, N + 1, Nz)
            for k in 1:Nz, j in Hp + 1:Hp + Nc + 1, i in Hp + 1:Hp + Nc
                b[i, j, k] = FT(0.012) *
                    cos(FT(0.3 * step + 0.5 * p + 0.4 * i + 0.6 * j + 0.1 * k))
            end
            b
        end, 6) for step in 1:nsteps]
        cm_steps = [ntuple(p -> begin
            c = zeros(FT, N, N, Nz + 1)
            for k in 2:Nz, j in Hp + 1:Hp + Nc, i in Hp + 1:Hp + Nc
                c[i, j, k] = -FT(0.010) *
                    (one(FT) + FT(0.1) * sin(FT(i + j + k + p + step)))
            end
            c
        end, 6) for step in 1:nsteps]
    else
        am_steps = [ntuple(_ -> zeros(FT, N + 1, N, Nz), 6) for _ in 1:nsteps]
        bm_steps = [ntuple(_ -> zeros(FT, N, N + 1, Nz), 6) for _ in 1:nsteps]
        cm_steps = [ntuple(_ -> zeros(FT, N, N, Nz + 1), 6) for _ in 1:nsteps]
    end
    return mesh, panels_m, panels_rm, am_steps, bm_steps, cm_steps
end

function _max_footprint_diff(a, b)
    nsteps = length(a.footprints)
    return maximum(maximum(abs, a.footprints[s][p] .- b.footprints[s][p])
                   for s in 1:nsteps, p in 1:6)
end

@testset "cs_surface_emission_footprint :mmap parity with :device" begin
    @testset "$(nameof(typeof(scheme))) ($(nontrivial ? "transport" : "constant"))" for
        scheme in (AT.UpwindScheme(),
                   AT.SlopesScheme(AT.NoLimiter()),
                   AT.PPMScheme(AT.NoLimiter()),
                   AT.PPMScheme(AT.MonotoneLimiter())),
        nontrivial in (false, true)

        mesh, panels_m, panels_rm, am, bm, cm =
            _trivial_problem(Nc = 4, Nz = 3, nsteps = 2; nontrivial = nontrivial)
        obj = AT.CSLayerMeanObjective(1, 2, 2, 3)
        dev = AT.cs_surface_emission_footprint(panels_rm, panels_m, am, bm, cm,
            mesh, obj; scheme = scheme, dt = 1.0, tape_storage = :device)
        mmap = AT.cs_surface_emission_footprint(panels_rm, panels_m, am, bm, cm,
            mesh, obj; scheme = scheme, dt = 1.0, tape_storage = :mmap)
        @test _max_footprint_diff(dev, mmap) == 0
    end

    @testset "from_seed parity" begin
        mesh, panels_m, _, am, bm, cm =
            _trivial_problem(Nc = 4, Nz = 3, nsteps = 2; nontrivial = true)
        N = mesh.Nc + 2mesh.Hp
        seed = ntuple(p -> begin
            a = zeros(Float64, N, N, 3)
            a[mesh.Hp + 2, mesh.Hp + 2, 3] = 0.5
            a
        end, 6)
        dev = AT.cs_surface_emission_footprint_from_seed(seed, panels_m, am, bm, cm,
            mesh; scheme = AT.PPMScheme(AT.NoLimiter()), dt = 1.0,
            tape_storage = :device)
        mmap = AT.cs_surface_emission_footprint_from_seed(seed, panels_m, am, bm, cm,
            mesh; scheme = AT.PPMScheme(AT.NoLimiter()), dt = 1.0,
            tape_storage = :mmap)
        @test _max_footprint_diff(dev, mmap) == 0
    end
end
