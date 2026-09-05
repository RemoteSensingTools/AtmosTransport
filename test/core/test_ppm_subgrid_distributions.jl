#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Regression tests for the PPM cell-edge reconstruction used by the LinRood
# (ORD=5 / ORD=7) cubed-sphere advection path.
#
# History: ORD=5 (and therefore ORD=7, which reuses it) silently degenerated to
# a 2nd-order 2-point average — `huynh_second_constraint(q_im, q_i, q_i, …)`
# passed `q_i` as both centre and right edge, so the clamp collapsed to the plain
# difference `q_i - q_im` and the parabola edges became `(q_im+q_i)/2`. ORD=6's
# coefficients also summed to 58/60, breaking constant preservation. These tests
# pin the corrected 4th-order interpolation, constant/linear exactness for every
# ORD, the genuine higher-order behaviour of ORD=5/7, and forward/adjoint
# (dual-number) consistency.
# ---------------------------------------------------------------------------

using Test

using AtmosTransport
const ADV = AtmosTransport.Operators.Advection

const _edges = ADV._ppm_edge_values

@testset "PPM edge reconstruction" begin
    @testset "constant field is preserved exactly (every ORD)" begin
        for ord in (4, 5, 6, 7)
            qL, qR = _edges(1.0, 1.0, 1.0, 1.0, 1.0, Val(ord))
            @test qL ≈ 1.0
            @test qR ≈ 1.0
        end
    end

    @testset "linear field is reconstructed exactly (every ORD)" begin
        # q = x at centres x = -2..2 → edge at -1/2 is -0.5, at +1/2 is +0.5.
        for ord in (4, 5, 6, 7)
            qL, qR = _edges(-2.0, -1.0, 0.0, 1.0, 2.0, Val(ord))
            @test qL ≈ -0.5
            @test qR ≈ 0.5
        end
    end

    @testset "ORD=5/7 use the 4th-order interpolation, not 2-point averaging" begin
        # The exact 4th-order PPM weights are (-1/12, 7/12, 7/12, -1/12) for q_L
        # and (0, -1/12, 7/12, 7/12, -1/12) for q_R.
        q = (0.3, 1.1, 2.7, 1.9, 0.4)
        p1, p2 = 7 / 12, -1 / 12
        wantL = p2 * (q[1] + q[4]) + p1 * (q[2] + q[3])
        wantR = p2 * (q[2] + q[5]) + p1 * (q[3] + q[4])
        for ord in (5, 7)
            qL, qR = _edges(q..., Val(ord))
            @test qL ≈ wantL
            @test qR ≈ wantR
            # ...and decidedly NOT the degenerate 2-point average (the old bug).
            @test !(qL ≈ (q[2] + q[3]) / 2)
            @test !(qR ≈ (q[3] + q[4]) / 2)
        end

        # On a cubic the degenerate scheme returned the minmod/average value
        # (±0.5); the 4th-order scheme returns ≈0 — a concrete higher-order check.
        qL5, qR5 = _edges(-8.0, -1.0, 0.0, 1.0, 8.0, Val(5))
        @test abs(qL5) < 1e-12
        @test abs(qR5) < 1e-12
    end

    @testset "ORD=6 weights sum to one (constant preservation restored)" begin
        # Regression: the old (1/30,13/60,13/60,9/20,1/20) weights summed to 58/60.
        qL, qR = _edges(1.0, 1.0, 1.0, 1.0, 1.0, Val(6))
        @test qL ≈ 1.0
        @test qR ≈ 1.0
        # 5th-order upwind right-edge stencil (2,-13,47,27,-3)/60.
        q = (0.3, 1.1, 2.7, 1.9, 0.4)
        _, qR6 = _edges(q..., Val(6))
        @test qR6 ≈ (2q[1] - 13q[2] + 47q[3] + 27q[4] - 3q[5]) / 60
    end

    @testset "Float32 stays finite and consistent" begin
        for ord in (4, 5, 6, 7)
            qL, qR = _edges(1.0f0, 2.0f0, 3.0f0, 2.0f0, 1.0f0, Val(ord))
            @test isfinite(qL) && isfinite(qR)
            @test qL isa Float32 && qR isa Float32
        end
        # constant preserved at Float32
        for ord in (4, 5, 6, 7)
            qL, qR = _edges(5.0f0, 5.0f0, 5.0f0, 5.0f0, 5.0f0, Val(ord))
            @test qL ≈ 5.0f0
            @test qR ≈ 5.0f0
        end
    end

    @testset "ORD=5 forward/adjoint (dual-number) consistency" begin
        # The dual-number edge reconstruction must return the forward value and a
        # gradient matching a central finite difference of the forward kernel.
        q = [0.3, 1.1, 2.7, 1.9, 0.4]
        ds = ntuple(n -> ADV._d6_var(q[n], Val(n)), 5)
        qL_d6, qR_d6 = ADV._ppm_edge_values_ord5_d6(ds...)
        fL, fR = _edges(q..., Val(5))
        @test qL_d6.v ≈ fL
        @test qR_d6.v ≈ fR
        h = 1e-6
        for (slot, dual) in ((1, qL_d6), (2, qR_d6))
            for n in 1:5
                qp = copy(q); qp[n] += h
                qm = copy(q); qm[n] -= h
                fd = (_edges(qp..., Val(5))[slot] - _edges(qm..., Val(5))[slot]) / (2h)
                @test dual.g[n] ≈ fd atol = 1e-6
            end
        end
    end

    @testset "extremum flattening still applies (quasi-monotone limiter)" begin
        # _apply_monotonicity flattens the reconstruction to the cell mean when the
        # edges straddle a local extremum (q_R, q_L on opposite sides of c).
        c = 2.0
        qLf, qRf = ADV._apply_monotonicity(1.5, 1.5, c)  # both below c → extremum
        @test qLf == c
        @test qRf == c
        # monotone stencil: edges pass through unchanged
        qLp, qRp = ADV._apply_monotonicity(1.0, 3.0, c)
        @test qLp == 1.0
        @test qRp == 3.0
    end
end
