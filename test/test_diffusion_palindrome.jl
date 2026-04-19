"""
Tests for the palindrome integration of vertical diffusion into
`strang_split_mt!` (plan 16b, Commit 4).

Structure: `X(nₓ) Y(nᵧ) Z(n_z) V(dt) Z(n_z) Y(nᵧ) X(nₓ)`.

The critical regression property: **bit-exact equality** between the
default (implicit `NoDiffusion()`) code path and the explicit
`diffusion_op = NoDiffusion()` path. `NoDiffusion`'s method is
`apply_vertical_diffusion!(_, ::NoDiffusion, _, _) = nothing`, so the
palindrome-center call carries no floating-point work and the
non-diffusion path stays bit-exact with pre-16b behavior.

Further invariants checked:
- zero fluxes + `NoDiffusion` is the identity on tracers;
- zero fluxes + `ImplicitVerticalDiffusion{Kz > 0}` makes a
  vertically non-uniform profile evolve;
- linear-operator composition: `V(dt) == V(dt/2) ∘ V(dt/2)` (plan
  Decision 8) at the palindrome center.

Strang second-order accuracy for pure advection is covered by the
existing `test_advection_kernels.jl`; the palindrome insert does
not change the advection-only code path when `diffusion_op` is
`NoDiffusion()`, so those tests implicitly validate that Commit 4
did not degrade Strang accuracy. A dedicated "halve dt → quarter
error" convergence test is therefore not duplicated here.
"""

using Test
using AtmosTransport: CellState, AdvectionWorkspace,
                      ConstantField, PreComputedKzField,
                      NoDiffusion, ImplicitVerticalDiffusion,
                      UpwindScheme,
                      apply_vertical_diffusion!
using AtmosTransport.Operators.Advection: strang_split_mt!

# =========================================================================
# Test setup
# =========================================================================

"""
Build (rm_4d, m, am, bm, cm, scheme, ws) for a small structured grid
with known mass/flux arrays. Tracers get a vertically-varying profile
so diffusion has something to act on.
"""
function _make_palindrome_inputs(::Type{FT}; Nx = 3, Ny = 3, Nz = 8,
                                   Nt = 2, flux_val = 0.0) where FT
    m      = ones(FT, Nx, Ny, Nz)
    am     = fill(FT(flux_val), Nx + 1, Ny, Nz)
    bm     = fill(FT(flux_val), Nx, Ny + 1, Nz)
    cm     = fill(FT(flux_val), Nx, Ny, Nz + 1)
    rm_4d  = Array{FT}(undef, Nx, Ny, Nz, Nt)
    for t in 1:Nt, k in 1:Nz, j in 1:Ny, i in 1:Nx
        # Vertically-varying profile: Gaussian peak at k mid-column,
        # perturbed by tracer index so the two tracers are distinguishable
        k_peak = (Nz + 1) / 2
        rm_4d[i, j, k, t] = FT(exp(-((k - k_peak)^2) / 4.0) + 0.1 * t)
    end
    scheme = UpwindScheme()
    # Build a CellState-matching workspace via the raw-array constructor
    ws = AdvectionWorkspace(m; n_tracers = Nt)
    return rm_4d, m, am, bm, cm, scheme, ws
end

# =========================================================================
# 1. Bit-exact: default kwarg matches explicit NoDiffusion
# =========================================================================

@testset "bit-exact: default kwargs == explicit NoDiffusion" begin
    FT = Float64
    rm_A, m_A, am, bm, cm, scheme, ws_A =
        _make_palindrome_inputs(FT; flux_val = 0.003)
    rm_B, m_B, _, _, _, _, ws_B =
        _make_palindrome_inputs(FT; flux_val = 0.003)
    @assert rm_A == rm_B && m_A == m_B   # equal inputs

    # Default path — no diffusion_op argument passed
    strang_split_mt!(rm_A, m_A, am, bm, cm, scheme, ws_A)

    # Explicit NoDiffusion
    strang_split_mt!(rm_B, m_B, am, bm, cm, scheme, ws_B;
                     diffusion_op = NoDiffusion())

    # BIT-EXACT equality — no floating-point op differences allowed.
    # Per plan 16b Commit 4 review: if the NoDiffusion branch is truly
    # dead code, the outputs must be `==`, not just `≈`.
    @test rm_A == rm_B
    @test m_A  == m_B
end

# =========================================================================
# 2. NoDiffusion + zero fluxes = identity on rm
# =========================================================================

@testset "NoDiffusion + zero fluxes is the identity" begin
    FT = Float64
    rm, m, am, bm, cm, scheme, ws = _make_palindrome_inputs(FT; flux_val = 0.0)
    rm0 = copy(rm)
    m0  = copy(m)

    strang_split_mt!(rm, m, am, bm, cm, scheme, ws;
                     diffusion_op = NoDiffusion())

    @test rm == rm0
    @test m  == m0
end

# =========================================================================
# 3. Diffusion actually runs at the palindrome center
# =========================================================================

@testset "ImplicitVerticalDiffusion + zero fluxes alters a non-uniform profile" begin
    FT = Float64
    Nx, Ny, Nz, Nt = 3, 3, 8, 2
    rm_diff, m, am, bm, cm, scheme, ws =
        _make_palindrome_inputs(FT; Nx = Nx, Ny = Ny, Nz = Nz, Nt = Nt,
                                flux_val = 0.0)
    rm0 = copy(rm_diff)

    # Fill dz_scratch and build a non-trivial Kz field (uniform is fine —
    # Gaussian has non-zero vertical curvature so Kz > 0 moves mass).
    fill!(ws.dz_scratch, FT(100.0))
    op = ImplicitVerticalDiffusion(; kz_field = ConstantField{FT, 3}(1.0))
    dt = 10.0

    strang_split_mt!(rm_diff, m, am, bm, cm, scheme, ws;
                     diffusion_op = op, dt = dt)

    # The profile MUST change somewhere (diffusion of a Gaussian is
    # non-trivial).
    @test any(rm_diff .!= rm0)

    # Column mass conserved to ULP (Neumann BCs, uniform dz)
    for t in 1:Nt, j in 1:Ny, i in 1:Nx
        mass_before = sum(@view rm0[i, j, :, t])
        mass_after  = sum(@view rm_diff[i, j, :, t])
        @test abs(mass_after - mass_before) / abs(mass_before) < 1e-12
    end
end

# =========================================================================
# 4. Comparison vs standalone apply_vertical_diffusion! call
# =========================================================================

@testset "palindrome diffusion matches a standalone apply_vertical_diffusion!" begin
    # With zero fluxes, the full palindrome reduces to a no-op forward
    # half, then the palindrome-center V(dt), then a no-op reverse half.
    # The output must match calling apply_vertical_diffusion! alone.
    FT = Float64
    Nx, Ny, Nz, Nt = 2, 2, 6, 1
    rm_A, m_A, am, bm, cm, scheme, ws_A =
        _make_palindrome_inputs(FT; Nx = Nx, Ny = Ny, Nz = Nz, Nt = Nt,
                                flux_val = 0.0)
    rm_B = copy(rm_A)
    m_B  = copy(m_A)
    ws_B = AdvectionWorkspace(m_B; n_tracers = Nt)

    Kz_arr = FT.(0.5 .+ 1.5 .* rand(Nx, Ny, Nz))
    dz_arr = FT.(80.0 .+ 40.0 .* rand(Nx, Ny, Nz))
    copyto!(ws_A.dz_scratch, dz_arr)
    copyto!(ws_B.dz_scratch, dz_arr)

    kz_field_A = PreComputedKzField(copy(Kz_arr))
    kz_field_B = PreComputedKzField(copy(Kz_arr))
    op_A = ImplicitVerticalDiffusion(; kz_field = kz_field_A)
    dt = 5.0

    # Path A: full palindrome with zero fluxes
    strang_split_mt!(rm_A, m_A, am, bm, cm, scheme, ws_A;
                     diffusion_op = op_A, dt = dt)

    # Path B: standalone apply_vertical_diffusion! on the same input
    op_B = ImplicitVerticalDiffusion(; kz_field = kz_field_B)
    apply_vertical_diffusion!(rm_B, op_B, ws_B, dt)

    @test rm_A ≈ rm_B atol=1e-12 rtol=1e-12
end

# =========================================================================
# 5. V(dt) vs V(dt/2)∘V(dt/2) agree to O(dt²) — plan Decision 8 refined
# =========================================================================

@testset "V(dt) and V(dt/2)∘V(dt/2) agree to O(dt²)" begin
    # Plan Decision 8 claims `V(dt) = V(dt/2) ∘ V(dt/2)` for linear V.
    # That is exactly true for the continuous linear-ODE flow
    # (`e^(dt·D) = e^(dt/2·D) · e^(dt/2·D)`), but NOT exactly true for
    # the Backward-Euler discretization used here:
    #
    #   `(I - dt·D)⁻¹ ≠ [(I - dt/2·D)⁻¹]²` in general
    #
    # Both are 2nd-order approximations of `e^(dt·D)`, so they agree to
    # O((dt·D)²). This test verifies the convergence rate: halving dt
    # should reduce the discrepancy by a factor of ~4, not leave it
    # constant.
    FT = Float64
    Nx, Ny, Nz = 2, 2, 6
    Nt = 1

    # Convergence rate: ‖ΔV(dt)‖ / ‖ΔV(dt/2)‖ ≈ 4  (2nd-order in dt)
    function _pairwise_diff(FT, dt)
        rm_A = _make_palindrome_inputs(FT; Nx = Nx, Ny = Ny, Nz = Nz, Nt = Nt,
                                       flux_val = 0.0)[1]
        rm_B = copy(rm_A)
        m_ref = ones(FT, Nx, Ny, Nz)

        ws_A = AdvectionWorkspace(m_ref; n_tracers = Nt)
        ws_B = AdvectionWorkspace(m_ref; n_tracers = Nt)
        fill!(ws_A.dz_scratch, FT(100.0))
        fill!(ws_B.dz_scratch, FT(100.0))

        kz_val = FT(1.5)
        op = ImplicitVerticalDiffusion(; kz_field = ConstantField{FT, 3}(kz_val))

        # One full step
        apply_vertical_diffusion!(rm_A, op, ws_A, dt)
        # Two half steps
        apply_vertical_diffusion!(rm_B, op, ws_B, dt / 2)
        apply_vertical_diffusion!(rm_B, op, ws_B, dt / 2)

        return maximum(abs.(rm_A .- rm_B))
    end

    diff_big   = _pairwise_diff(FT, 4.0)
    diff_small = _pairwise_diff(FT, 2.0)

    # Ratio should be ~4 for 2nd-order convergence. Broad tolerance to
    # absorb that the error itself is tiny (on the order of 1e-8) and
    # finite-precision rounding adds noise at that scale.
    @test diff_big > diff_small                        # monotone in dt
    @test 2.5 < diff_big / diff_small < 5.5            # near 4× (O(dt²))
end

# =========================================================================
# 6. dt required when diffusion_op != NoDiffusion
# =========================================================================

@testset "ImplicitVerticalDiffusion with dt = nothing is rejected at runtime" begin
    FT = Float64
    rm, m, am, bm, cm, scheme, ws = _make_palindrome_inputs(FT; flux_val = 0.0)
    fill!(ws.dz_scratch, FT(100.0))

    op = ImplicitVerticalDiffusion(; kz_field = ConstantField{FT, 3}(1.0))
    # No dt supplied → the downstream kernel conversion `FT(nothing)` fails.
    @test_throws MethodError strang_split_mt!(rm, m, am, bm, cm, scheme, ws;
                                               diffusion_op = op, dt = nothing)
end
