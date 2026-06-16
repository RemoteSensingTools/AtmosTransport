"""
F32 emission rounding accuracy: naive vs Kahan vs F64.

Motivation: `q_raw[surface] += rate * dt` in Float32 silently rounds away
emission when the per-step increment is small relative to the background
tracer mass.  The rounding is SYSTEMATICALLY NEGATIVE:

  * Regime A ("complete loss"): rate*dt < eps32*background/2
    → fl(background + rate*dt) == background exactly, every step lost.
    Global budget deficit is 100% of emission. Sign: always negative.

  * Regime B ("partial rounding"): rate*dt ≈ eps32*background
    → sign depends on the fractional part of (rate*dt / ULP(background)):
      frac < 0.5 → rounds DOWN → systematic negative (undercount)
      frac > 0.5 → rounds UP   → systematic positive (overcount)
    Which regime a cell falls in is determined by the rate/ULP ratio.
    The production SF6 2.5% deficit is Regime A (complete loss).

Kahan compensated summation carries the rounding debt in a compensation
variable (`c`): `y = x - c; t = s + y; c = (t - s) - y; s = t`.
The residual after N steps is bounded by one ULP, independent of N.
This test quantifies the error magnitude and verifies Kahan's correction.
"""

using Test

# ---------------------------------------------------------------------------
# Pure-Julia Kahan helpers (matching what the KA kernels do per-thread).
# ---------------------------------------------------------------------------
function naive_accumulate(background::T, rate_per_step::T, N::Int) where {T}
    s = background
    for _ in 1:N
        s += rate_per_step
    end
    return s
end

function kahan_accumulate(background::T, rate_per_step::T, N::Int) where {T}
    s = background
    c = zero(T)
    for _ in 1:N
        x = rate_per_step
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    end
    return s
end

@testset "Kahan vs naive F32 surface-flux accumulation" begin

    # ------------------------------------------------------------------
    # Regime A — complete loss: rate*dt << ULP(background)/2.
    # Every addition is rounded back to background; naive result is exact
    # background with zero emission accumulated.
    # Expected error: always negative, magnitude ≈ total emission.
    # Kahan residual is bounded by one ULP of background (the carry buffer
    # drains at most one ULP's worth of debt per step).
    # ------------------------------------------------------------------
    @testset "Regime A: complete loss (rate << ULP/2)" begin
        # background = 400 ppm CO2; ULP ≈ 3.05e-5.
        # rate = 1e-7 per step — rate/ULP ≈ 0.003, well below 0.5 → complete loss.
        background = Float32(400.0)
        rate       = Float32(1e-7)
        N          = 10_000

        exact_emission = Float64(rate) * N
        exact_total    = Float64(background) + exact_emission

        naive_total = naive_accumulate(background, rate, N)
        kahan_total = kahan_accumulate(background, rate, N)

        naive_deficit = Float64(naive_total) - exact_total   # ≈ −exact_emission
        kahan_deficit = Float64(kahan_total) - exact_total   # ≈ 0

        # Naive: complete loss → deficit ≈ −100% of emission; always negative.
        @test naive_deficit ≈ -exact_emission  rtol=0.01
        @test naive_deficit < 0

        # Kahan: residual bounded by one ULP of background (the carry buffer
        # holds at most ULP/2 of unresolved debt at any point).
        ulp_background = Float64(eps(background))
        @test abs(kahan_deficit) ≤ ulp_background
    end

    # ------------------------------------------------------------------
    # Regime B — sign depends on fractional(rate / ULP):
    #   * frac < 0.5 → rounds DOWN → negative error (systematic undercount)
    #   * frac > 0.5 → rounds UP   → positive error (systematic overcount)
    # Both are systematic (same direction for all N steps since rate and
    # background change slowly).  Kahan corrects either case to one ULP.
    # ------------------------------------------------------------------
    @testset "Regime B-neg: rate < ULP/2 (rounds down, negative error)" begin
        background = Float32(1000.0)
        rate       = Float32(0.3 * eps(background))   # 0.3 × ULP < 0.5 → rounds down
        N          = 100_000

        exact_emission = Float64(rate) * N
        exact_total    = Float64(background) + exact_emission

        naive_total = naive_accumulate(background, rate, N)
        kahan_total = kahan_accumulate(background, rate, N)

        naive_deficit = Float64(naive_total) - exact_total
        kahan_deficit = Float64(kahan_total) - exact_total

        # Rounds down every step → deficit is negative, ~70% of emission lost.
        @test naive_deficit < 0
        @test abs(naive_deficit / exact_emission) > 0.50

        # Kahan: residual ≤ 1 ULP of background.
        @test abs(kahan_deficit) ≤ Float64(eps(background))
    end

    @testset "Regime B-pos: rate > ULP/2 (rounds up, positive error)" begin
        background = Float32(1000.0)
        rate       = Float32(0.8 * eps(background))   # 0.8 × ULP > 0.5 → rounds up
        N          = 100_000

        exact_emission = Float64(rate) * N
        exact_total    = Float64(background) + exact_emission

        naive_total = naive_accumulate(background, rate, N)
        kahan_total = kahan_accumulate(background, rate, N)

        naive_surplus = Float64(naive_total) - exact_total
        kahan_deficit = Float64(kahan_total) - exact_total

        # Rounds up every step → surplus is positive, ~25% extra emission.
        @test naive_surplus > 0
        @test abs(naive_surplus / exact_emission) > 0.10

        # Kahan still corrects: residual ≤ 1 ULP.
        @test abs(kahan_deficit) ≤ Float64(eps(background))
    end

    # ------------------------------------------------------------------
    # F32 vs F64 comparison: the "ground truth" for why F64 fixes the bug
    # (but F64 is unavailable on L40S / Metal GPUs, hence Kahan is needed).
    # ------------------------------------------------------------------
    @testset "F32 vs F64 vs Kahan comparison" begin
        background = Float32(1e4)
        rate       = Float32(1e-4)   # rate/ULP(1e4) ≈ 0.8 → partial loss regime
        N          = 50_000

        exact_emission = Float64(rate) * N          # = 5.0
        exact_total    = Float64(background) + exact_emission

        # F64 ground truth (no rounding loss at these magnitudes).
        f64_total = naive_accumulate(Float64(background), Float64(rate), N)
        f64_deficit = f64_total - exact_total
        @test abs(f64_deficit / exact_emission) < 1e-10

        # F32 naive: substantial loss.
        f32_total   = naive_accumulate(background, rate, N)
        f32_deficit = Float64(f32_total) - exact_total
        @test f32_deficit < 0
        @test abs(f32_deficit / exact_emission) > 0.01   # > 1% loss

        # Kahan F32: approaches F64 accuracy.
        kahan_total   = kahan_accumulate(background, rate, N)
        kahan_deficit = Float64(kahan_total) - exact_total
        @test abs(kahan_deficit / exact_emission) < abs(f32_deficit / exact_emission) / 10
        @test abs(kahan_deficit / exact_emission) < 1e-3
    end

    # ------------------------------------------------------------------
    # Sign-bias check: in Regime A the error is ALWAYS negative (never
    # positive), confirming the global budget always shows a deficit.
    # ------------------------------------------------------------------
    @testset "Error is always non-positive in Regime A" begin
        background = Float32(400.0)
        # Sweep a range of rates all well below ULP/2.
        ulp_half = 0.5 * eps(background)
        for rate_fraction in (0.001, 0.01, 0.05, 0.1, 0.2, 0.4)
            rate = Float32(rate_fraction * ulp_half)
            naive_total = naive_accumulate(background, rate, 10_000)
            # naive total ≤ background (no step ever overshoots).
            @test Float64(naive_total) ≤ Float64(background)
        end
    end

    # ------------------------------------------------------------------
    # Kahan compensation array allocator: `_alloc_flux_comp_from_series`
    # must match the backend of its input (the P2 Codex finding).
    # ------------------------------------------------------------------
    @testset "Kahan compensation backend matches series type" begin
        using AtmosTransport.Operators.SurfaceFlux: _alloc_flux_comp_from_series,
                                                    _alloc_flux_comp

        # Static rate: `zero(rate)` should preserve Array type.
        rate2d = rand(Float32, 4, 5)
        comp2d = _alloc_flux_comp(rate2d)
        @test comp2d isa Array{Float32}
        @test size(comp2d) == (4, 5)
        @test all(iszero, comp2d)

        # Series: `similar` must drop the time dimension and stay same array type.
        series = rand(Float32, 4, 5, 12)      # (Nx, Ny, Nt)
        comp   = _alloc_flux_comp_from_series(series)
        @test comp isa Array{Float32}
        @test size(comp) == (4, 5)
        @test all(iszero, comp)

        # NTuple{6} series: each panel drops its time dimension.
        panels = ntuple(p -> rand(Float32, 3, 3, 6), Val(6))
        comps  = _alloc_flux_comp_from_series(panels)
        @test comps isa NTuple{6}
        for p in 1:6
            @test comps[p] isa Array{Float32}
            @test size(comps[p]) == (3, 3)
            @test all(iszero, comps[p])
        end
    end

end
