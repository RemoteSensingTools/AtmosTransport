#!/usr/bin/env julia
# Focused regression tests for the per-window CS preprocessor contract
# surface exported from `cubed_sphere_contracts.jl`.
#
# Builds tiny 6-panel synthetic windows that trip each gate (replay,
# positivity-CFL, m <= 0, NaN/Inf mass, NaN/Inf flux) and confirms the
# corresponding helper fires with the documented diagnostic. Also locks in
# the round-2 (45b87f3) regression that `summarize_cs_positivity_status`
# does not throw `InexactError` on a non-finite ratio under either policy.

using Test
using Logging: with_logger, NullLogger

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Preprocessing: verify_substep_positivity_cs!,
                                       verify_cs_window_contract!,
                                       init_cs_positivity_accumulator,
                                       update_cs_positivity_accumulator,
                                       summarize_cs_positivity_status

# Build a 6-panel CS window whose horizontal fluxes vanish and whose vertical
# fluxes encode the per-cell mass tendency exactly. The result satisfies the
# write-time replay gate to F64 ULP, so the positivity scan can be exercised
# in isolation through `verify_cs_window_contract!` without tripping the
# replay short-circuit first.
function build_clean_cs_window(FT::Type; Nc::Int = 4, Nz::Int = 3, steps::Int = 2,
                                m_base::Real = 1e9, dm_scale::Real = 1e4)
    panels = ntuple(6) do p
        m_cur = fill(FT(m_base), Nc, Nc, Nz)
        am = zeros(FT, Nc + 1, Nc, Nz)
        bm = zeros(FT, Nc, Nc + 1, Nz)
        cm = zeros(FT, Nc, Nc, Nz + 1)
        dm = Array{FT}(undef, Nc, Nc, Nz)
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            dm[i, j, k] = FT(dm_scale) * sinpi(FT(i) / Nc) *
                          cospi(FT(j) / Nc) * FT(k / Nz) * FT(1 + 0.05 * p)
        end
        for j in 1:Nc, i in 1:Nc
            acc = 0.0
            for k in 1:Nz
                acc -= Float64(dm[i, j, k])
                cm[i, j, k + 1] = FT(acc)
            end
        end
        m_next = m_cur .+ FT(2 * steps) .* dm
        (; m_cur, am, bm, cm, m_next)
    end
    return (m_cur = ntuple(p -> panels[p].m_cur, 6),
            am = ntuple(p -> panels[p].am, 6),
            bm = ntuple(p -> panels[p].bm, 6),
            cm = ntuple(p -> panels[p].cm, 6),
            m_next = ntuple(p -> panels[p].m_next, 6),
            steps = steps)
end

# Discard log output from `summarize_cs_positivity_status` during the
# require=false / Inf branches — the tests assert return values, not text.
with_quiet_logger(f) = with_logger(f, NullLogger())

@testset "CS preprocessor contract gates" begin

    # ------------------------------------------------------------------
    # verify_substep_positivity_cs!
    # ------------------------------------------------------------------

    @testset "positivity: clean window passes" begin
        for FT in (Float32, Float64)
            w = build_clean_cs_window(FT)
            diag = verify_substep_positivity_cs!(w.m_cur, w.am, w.bm, w.cm;
                                                  cfl_limit = 0.95)
            @test diag.ok
            @test isfinite(diag.ratio)
            @test diag.ratio < 0.95
            @test diag.location isa NTuple{4, Int}
        end
    end

    @testset "positivity: x-face CFL violation reported with direction and location" begin
        w = build_clean_cs_window(Float64)
        am = ntuple(6) do p
            arr = copy(w.am[p])
            p == 3 && (arr[3, 2, 1] = 0.99e9)  # right face of cell (2,2,1)
            arr
        end
        diag = verify_substep_positivity_cs!(w.m_cur, am, w.bm, w.cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test diag.direction === :x
        @test diag.ratio ≈ 0.99 atol = 1e-12
        @test diag.location == (3, 2, 2, 1)
    end

    @testset "positivity: y-face CFL violation reported with direction and location" begin
        w = build_clean_cs_window(Float64)
        bm = ntuple(6) do p
            arr = copy(w.bm[p])
            p == 5 && (arr[2, 3, 2] = 0.97e9)  # top face of cell (2,2,2)
            arr
        end
        diag = verify_substep_positivity_cs!(w.m_cur, w.am, bm, w.cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test diag.direction === :y
        @test diag.ratio ≈ 0.97 atol = 1e-12
        @test diag.location == (5, 2, 2, 2)
    end

    @testset "positivity: z-face violation is included in the scan" begin
        # Drive a single vertical interface flux up enough to dominate the
        # tiny baseline z-ratio from the clean-window dm divergence.
        w = build_clean_cs_window(Float64)
        cm = ntuple(6) do p
            arr = copy(w.cm[p])
            p == 6 && (arr[1, 1, 3] = 0.98e9)  # interface k=3 of cell (1,1,2)
            arr
        end
        diag = verify_substep_positivity_cs!(w.m_cur, w.am, w.bm, cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test diag.direction === :z
        # The clean-window baseline contributes a few units to cm at this
        # interface, so the observed ratio is slightly above 0.98 — loose
        # tolerance is fine, the point is that the z-direction violation
        # surfaces rather than being silently dominated by tiny x/y ratios.
        @test diag.ratio ≈ 0.98 atol = 1e-3
        @test diag.location == (6, 1, 1, 2)
    end

    @testset "positivity: m <= 0 is flagged Inf regardless of flux magnitude (round-2 fix)" begin
        w = build_clean_cs_window(Float64)
        m_cur = ntuple(6) do p
            arr = copy(w.m_cur[p])
            p == 1 && (arr[2, 3, 1] = 0.0)
            arr
        end
        diag = verify_substep_positivity_cs!(m_cur, w.am, w.bm, w.cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test isinf(diag.ratio)
        # The kernel pins the report to the first non-positive cell it
        # encounters; subsequent cells cannot make `Inf` worse.
        @test diag.direction === :x
        @test diag.location == (1, 2, 3, 1)
    end

    @testset "positivity: NaN mass yields Inf ratio without slipping through (round-2 fix)" begin
        # Before the round-2 fix the kernel branched on `mi <= 0`, which is
        # `false` for `NaN` in Julia — so a `NaN`-mass cell would fall through
        # to the divisor branch where `NaN / NaN = NaN`, and `NaN > worst_ratio`
        # is also false, silently dropping the cell.
        w = build_clean_cs_window(Float64)
        m_cur = ntuple(6) do p
            arr = copy(w.m_cur[p])
            p == 2 && (arr[1, 1, 1] = NaN)
            arr
        end
        diag = verify_substep_positivity_cs!(m_cur, w.am, w.bm, w.cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test isinf(diag.ratio)
        @test diag.location == (2, 1, 1, 1)
    end

    @testset "positivity: NaN flux yields Inf ratio (round-2 fix)" begin
        w = build_clean_cs_window(Float64)
        am = ntuple(6) do p
            arr = copy(w.am[p])
            p == 6 && (arr[3, 3, 2] = NaN)
            arr
        end
        diag = verify_substep_positivity_cs!(w.m_cur, am, w.bm, w.cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test isinf(diag.ratio)
    end

    @testset "positivity: Inf flux yields Inf ratio (round-2 fix)" begin
        w = build_clean_cs_window(Float64)
        cm = ntuple(6) do p
            arr = copy(w.cm[p])
            p == 4 && (arr[2, 2, 2] = Inf)
            arr
        end
        diag = verify_substep_positivity_cs!(w.m_cur, w.am, w.bm, cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test isinf(diag.ratio)
    end

    @testset "positivity: halo_width skips halo cells" begin
        # Build a haloed buffer with the violation seeded in the halo region;
        # the interior is clean (all-zero fluxes, m = 1e9), so positivity
        # must report ok = true.
        FT = Float64
        Nc = 4
        Nz = 3
        Hp = 1
        m = ntuple(_ -> begin
            arr = fill(FT(1e9), Nc + 2Hp, Nc + 2Hp, Nz)
            arr[1, 1, 1] = -1.0  # in the halo
            arr
        end, 6)
        am = ntuple(_ -> zeros(FT, Nc + 2Hp + 1, Nc + 2Hp, Nz), 6)
        bm = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp + 1, Nz), 6)
        cm = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz + 1), 6)
        diag = verify_substep_positivity_cs!(m, am, bm, cm;
                                              cfl_limit = 0.95, halo_width = Hp)
        @test diag.ok
        @test diag.ratio == 0.0
        # Same buffer with halo_width = 0 must catch the same violation.
        diag0 = verify_substep_positivity_cs!(m, am, bm, cm;
                                               cfl_limit = 0.95, halo_width = 0)
        @test !diag0.ok
        @test isinf(diag0.ratio)
        @test diag0.location == (1, 1, 1, 1)
    end

    # ------------------------------------------------------------------
    # verify_cs_window_contract!
    # ------------------------------------------------------------------

    @testset "wrapper: clean window returns both diagnostics" begin
        w = build_clean_cs_window(Float64)
        result = verify_cs_window_contract!(w.m_cur, w.am, w.bm, w.cm,
                                             w.m_next, w.steps, 1;
                                             replay_tol = 1e-12,
                                             positivity_cfl_limit = 0.95)
        @test result.replay.max_rel_err <= 1e-12
        @test result.positivity.ok
    end

    @testset "wrapper: replay failure errors before positivity is reached" begin
        w = build_clean_cs_window(Float64)
        cm_broken = ntuple(6) do p
            arr = copy(w.cm[p])
            p == 4 && (arr[2, 2, 2] += 1e4)
            arr
        end
        @test_throws ErrorException verify_cs_window_contract!(
            w.m_cur, w.am, w.bm, cm_broken, w.m_next, w.steps, 7;
            replay_tol = 1e-12, positivity_cfl_limit = 0.95,
        )
    end

    @testset "wrapper: positivity failure with passing replay is non-fatal (caller policy)" begin
        # Closed-loop uniform shift on one (j, k) row preserves per-cell
        # divergence on the perturbed row but drives outgoing/m up to 0.99.
        # `verify_cs_window_contract!` must return both diagnostics so the
        # caller — not the wrapper — decides whether to error or warn after
        # aggregating across windows.
        w = build_clean_cs_window(Float64)
        am = ntuple(6) do p
            arr = copy(w.am[p])
            p == 2 && (arr[:, 1, 1] .= 0.99e9)
            arr
        end
        result = verify_cs_window_contract!(w.m_cur, am, w.bm, w.cm,
                                             w.m_next, w.steps, 3;
                                             replay_tol = 1e-12,
                                             positivity_cfl_limit = 0.95)
        @test result.replay.max_rel_err <= 1e-12
        @test !result.positivity.ok
        @test result.positivity.direction === :x
        @test result.positivity.ratio ≈ 0.99 atol = 1e-12
        @test result.positivity.location == (2, 1, 1, 1)
    end

    # ------------------------------------------------------------------
    # accumulator
    # ------------------------------------------------------------------

    @testset "accumulator: tracks worst window across the loop" begin
        worst = init_cs_positivity_accumulator()
        @test worst.ratio == 0.0
        @test worst.direction === :none
        @test worst.win == 0
        @test worst.location == (0, 0, 0, 0)

        worst = update_cs_positivity_accumulator(worst,
            (direction = :x, ratio = 0.3, location = (1, 2, 3, 4), ok = true), 5)
        @test worst.ratio ≈ 0.3
        @test worst.direction === :x
        @test worst.win == 5
        @test worst.location == (1, 2, 3, 4)

        # Smaller ratio is ignored — the older worst is preserved.
        worst = update_cs_positivity_accumulator(worst,
            (direction = :y, ratio = 0.1, location = (2, 1, 1, 1), ok = true), 6)
        @test worst.ratio ≈ 0.3
        @test worst.win == 5

        worst = update_cs_positivity_accumulator(worst,
            (direction = :z, ratio = Inf, location = (3, 2, 2, 2), ok = false), 7)
        @test isinf(worst.ratio)
        @test worst.direction === :z
        @test worst.win == 7
        @test worst.location == (3, 2, 2, 2)
    end

    @testset "accumulator: direction === nothing is normalized to :none" begin
        worst = update_cs_positivity_accumulator(init_cs_positivity_accumulator(),
            (direction = nothing, ratio = 0.5, location = (1, 1, 1, 1), ok = true), 1)
        @test worst.direction === :none
        @test worst.ratio ≈ 0.5
    end

    # ------------------------------------------------------------------
    # summarize_cs_positivity_status
    # ------------------------------------------------------------------

    @testset "summary: ratio within limit returns nothing" begin
        worst = (ratio = 0.5, direction = :x, win = 1, location = (1, 1, 1, 1))
        @test summarize_cs_positivity_status(worst; cfl_limit = 0.95,
                                              steps_per_window = 8) === nothing
    end

    @testset "summary: finite violation + require=true errors with rescue advice" begin
        worst = (ratio = 1.5, direction = :x, win = 1, location = (1, 1, 1, 1))
        err = try
            summarize_cs_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = true)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        # The error message must include the recommended steps_per_window
        # (ceil(1.5 / 0.95) * 8 = 16) so the operator can act on it.
        @test occursin("steps_per_window=16", err.msg)
    end

    @testset "summary: finite violation + require=false warns and returns nothing" begin
        worst = (ratio = 1.5, direction = :x, win = 1, location = (1, 1, 1, 1))
        r = with_quiet_logger() do
            summarize_cs_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = false)
        end
        @test r === nothing
    end

    @testset "summary: Inf ratio + require=true throws ErrorException, NOT InexactError (round-2 fix)" begin
        # Before the round-2 fix this branch hit `ceil(Int, Inf)` and threw
        # `InexactError(:Int64, Inf)` BEFORE the intended error/warn path.
        # That broke the `require_substep_positivity = false` escape hatch for
        # exactly the failure modes that produce `Inf`/`NaN` ratios — the only
        # ones an operator might legitimately want to record-and-continue.
        worst = (ratio = Inf, direction = :z, win = 4, location = (2, 3, 3, 2))
        err = try
            summarize_cs_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = true)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test !(err isa InexactError)
        @test occursin("non-finite", err.msg)
    end

    @testset "summary: Inf ratio + require=false warns (no InexactError) (round-2 fix)" begin
        worst = (ratio = Inf, direction = :z, win = 4, location = (2, 3, 3, 2))
        r = with_quiet_logger() do
            summarize_cs_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = false)
        end
        @test r === nothing
    end

    @testset "summary: quarantine_path is deleted on error" begin
        worst = (ratio = 1.5, direction = :x, win = 1, location = (1, 1, 1, 1))
        tmp = tempname() * ".bin"
        write(tmp, b"contents")
        @test isfile(tmp)
        @test_throws ErrorException summarize_cs_positivity_status(
            worst; cfl_limit = 0.95, steps_per_window = 8,
            require_substep_positivity = true, quarantine_path = tmp,
        )
        @test !isfile(tmp)
    end

    @testset "summary: missing quarantine_path is benign" begin
        worst = (ratio = 1.5, direction = :x, win = 1, location = (1, 1, 1, 1))
        missing_path = tempname() * ".bin"
        @test !isfile(missing_path)
        @test_throws ErrorException summarize_cs_positivity_status(
            worst; cfl_limit = 0.95, steps_per_window = 8,
            require_substep_positivity = true, quarantine_path = missing_path,
        )
    end
end
