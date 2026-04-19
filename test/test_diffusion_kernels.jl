"""
Unit tests for the vertical diffusion solver infrastructure
(plan 16b, Commit 2).

Three layers:

1. `build_diffusion_coefficients` — pure reference Backward-Euler
   coefficient builder. Checked against hand-expanded formulas.
2. `solve_tridiagonal!` — generic Thomas solve. Checked against
   small hand-built systems, a dense `Tridiagonal` solve, and the
   adjoint identity via the documented transposition rule.
3. `_vertical_diffusion_kernel!` — KA kernel that inlines the
   coefficient formulas. Verified against a pure-Julia reference
   that composes (1) + (2) per column. Analytic-diffusion Gaussian
   broadening and Neumann-BC mass conservation close the loop.
"""

using Test
using LinearAlgebra: Tridiagonal, dot
using AtmosTransport: solve_tridiagonal!, build_diffusion_coefficients,
                      PreComputedKzField, ConstantField
using AtmosTransport.Operators.Diffusion: _vertical_diffusion_kernel!
using KernelAbstractions: get_backend, synchronize

# =========================================================================
# 1. build_diffusion_coefficients — corner cases
# =========================================================================

@testset "build_diffusion_coefficients" begin

    @testset "uniform Kz, uniform dz — hand-computed" begin
        Nz = 5
        Kz = fill(1.0, Nz)
        dz = fill(1.0, Nz)
        dt = 0.1
        a, b, c = build_diffusion_coefficients(Kz, dz, dt)

        # Interior: dz_above = dz_below = 1.0, Kz_above = Kz_below = 1.0
        # D_above = D_below = 1 / (1*1) = 1
        # a_k = -0.1, b_k = 1 + 0.2 = 1.2, c_k = -0.1 (interior)
        # Boundaries: D_above=0 at k=1 (→ a_1=0, b_1 = 1+0.1 = 1.1, c_1 = -0.1)
        #             D_below=0 at k=Nz (→ c_Nz=0, b_Nz = 1.1, a_Nz = -0.1)
        @test a[1] === 0.0
        @test a[2] ≈ -0.1 atol=1e-14
        @test a[5] ≈ -0.1 atol=1e-14
        @test b[1] ≈ 1.1 atol=1e-14
        @test b[3] ≈ 1.2 atol=1e-14
        @test b[5] ≈ 1.1 atol=1e-14
        @test c[1] ≈ -0.1 atol=1e-14
        @test c[4] ≈ -0.1 atol=1e-14
        @test c[5] === 0.0
    end

    @testset "Neumann BCs: a[1] = 0, c[Nz] = 0" begin
        Kz = [2.0, 3.0, 4.0]
        dz = [0.5, 1.0, 1.5]
        a, b, c = build_diffusion_coefficients(Kz, dz, 0.2)
        @test a[1] === 0.0
        @test c[end] === 0.0
    end

    @testset "dt = 0 → identity (b = 1, a = c = 0)" begin
        Kz = rand(4) .+ 0.1
        dz = rand(4) .+ 0.1
        a, b, c = build_diffusion_coefficients(Kz, dz, 0.0)
        @test all(b .== 1.0)
        @test all(a .== 0.0)
        @test all(c .== 0.0)
    end

    @testset "varying Kz — asymmetric a vs c at the same interface" begin
        # At an interface between k and k+1, the flux is the same but gets
        # normalized by different dz[k] vs dz[k+1] — so c_k and a_{k+1}
        # should scale accordingly.
        Nz = 3
        Kz = [1.0, 2.0, 3.0]
        dz = [1.0, 2.0, 4.0]
        dt = 1.0
        a, b, c = build_diffusion_coefficients(Kz, dz, dt)

        # Interface between k=1 and k=2:
        # Kz_int = (1+2)/2 = 1.5, dz_int = (1+2)/2 = 1.5
        # D_below at k=1: 1.5 / (1 * 1.5) = 1.0  → c[1] = -1.0
        # D_above at k=2: 1.5 / (2 * 1.5) = 0.5  → a[2] = -0.5
        @test c[1] ≈ -1.0 atol=1e-14
        @test a[2] ≈ -0.5 atol=1e-14
    end

    @testset "dimension mismatch throws" begin
        @test_throws DimensionMismatch build_diffusion_coefficients(
            [1.0, 2.0], [1.0, 2.0, 3.0], 0.1)
    end

    @testset "type stability" begin
        Kz = Float64[1.0, 2.0, 3.0]
        dz = Float64[1.0, 2.0, 3.0]
        out = @inferred build_diffusion_coefficients(Kz, dz, 0.5)
        @test eltype(out[1]) === Float64

        Kz32 = Float32[1.0f0, 2.0f0, 3.0f0]
        dz32 = Float32[1.0f0, 2.0f0, 3.0f0]
        out32 = @inferred build_diffusion_coefficients(Kz32, dz32, 0.5)
        @test eltype(out32[1]) === Float32
    end
end

# =========================================================================
# 2. solve_tridiagonal! — correctness
# =========================================================================

@testset "solve_tridiagonal!" begin

    @testset "identity matrix returns d" begin
        Nz = 4
        a = zeros(Float64, Nz)
        b = ones(Float64, Nz)
        c = zeros(Float64, Nz)
        d = [1.0, 2.0, 3.0, 4.0]
        x = similar(d)
        w = similar(d)
        solve_tridiagonal!(x, a, b, c, d, w)
        @test x == d
    end

    @testset "matches Julia's Tridiagonal \\ d on random SPD system" begin
        Nz = 10
        # Symmetric diagonally-dominant tridiag: guaranteed solvable
        a = [0.0; -rand(Nz - 1) .* 0.2]
        b = 2.0 .+ rand(Nz) .* 0.1
        c = [-rand(Nz - 1) .* 0.2; 0.0]     # length Nz; last element ignored
        # Build dense reference from a[2:end] (sub-diag), b, c[1:end-1] (super-diag)
        T = Tridiagonal(a[2:end], b, c[1:end-1])
        d = rand(Nz) .* 10
        x_ref = T \ d

        x = similar(d)
        w = similar(d)
        solve_tridiagonal!(x, a, b, c, d, w)
        @test x ≈ x_ref atol=1e-10 rtol=1e-10
    end

    @testset "does not mutate a, b, c, d" begin
        Nz = 5
        a = [0.0; -0.1; -0.2; -0.3; -0.4]
        b = [1.2; 1.5; 1.7; 1.9; 2.1]
        c = [-0.1; -0.2; -0.3; -0.4; 0.0]
        d = [1.0, 2.0, 3.0, 4.0, 5.0]
        a0, b0, c0, d0 = copy(a), copy(b), copy(c), copy(d)

        x = similar(d)
        w = similar(d)
        solve_tridiagonal!(x, a, b, c, d, w)
        @test a == a0
        @test b == b0
        @test c == c0
        @test d == d0
    end

    @testset "dimension mismatch throws with @boundscheck on" begin
        a = zeros(3)
        b = ones(3)
        c = zeros(3)
        d = ones(3)
        x = zeros(3)
        w = zeros(2)   # too short
        @test_throws DimensionMismatch solve_tridiagonal!(x, a, b, c, d, w)
    end

    @testset "type stability" begin
        Nz = 5
        a = zeros(Nz); b = ones(Nz); c = zeros(Nz); d = ones(Nz)
        x = similar(d); w = similar(d)
        @test_nowarn @inferred solve_tridiagonal!(x, a, b, c, d, w)
    end
end

# =========================================================================
# 3. Adjoint-structure test: transposition rule yields L^T
# =========================================================================

@testset "Adjoint transposition rule verifies L^T" begin
    # Build a non-symmetric forward tridiagonal L via the diffusion
    # coefficient builder. Then construct L^T via the documented
    # transposition rule and verify the adjoint identity
    #     ⟨y, L x⟩ = ⟨e, x⟩
    # where L^T y = e for arbitrary x and arbitrary e.
    Nz = 8
    Kz = [0.5, 1.0, 1.5, 2.0, 1.7, 1.3, 0.9, 0.4]
    dz = [1.0, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]
    dt = 0.1
    a, b, c = build_diffusion_coefficients(Kz, dz, dt)

    # Forward L as dense for the identity check
    L = Tridiagonal(a[2:end], b, c[1:end-1])

    # Transposed coefficients per the documented rule:
    #   a_T[k] = c[k-1]     for k ≥ 2
    #   b_T[k] = b[k]
    #   c_T[k] = a[k+1]     for k ≤ Nz-1
    a_T = zeros(Nz)
    b_T = copy(b)
    c_T = zeros(Nz)
    for k in 2:Nz
        a_T[k] = c[k - 1]
    end
    for k in 1:Nz-1
        c_T[k] = a[k + 1]
    end
    L_T = Tridiagonal(a_T[2:end], b_T, c_T[1:end-1])

    # Sanity: L_T must equal L' (dense transpose)
    @test Matrix(L_T) ≈ Matrix(L)' atol=1e-14

    # Adjoint identity via Thomas solves:
    #   solve L x = d  → x
    #   solve L^T y = e → y
    #   check ⟨y, L x⟩ == ⟨e, x⟩
    d_rhs = Float64[1, -2, 3, -4, 5, -6, 7, -8]
    e_rhs = Float64[2, 0.5, -1.0, 1.2, 0.3, -0.7, 0.9, -0.4]

    x = similar(d_rhs); w = similar(d_rhs)
    solve_tridiagonal!(x, a, b, c, d_rhs, w)

    y = similar(e_rhs); w2 = similar(e_rhs)
    solve_tridiagonal!(y, a_T, b_T, c_T, e_rhs, w2)

    # Primary identity: ⟨y, L x⟩ = ⟨y, d⟩  = ⟨e, x⟩ ?
    # Since L x = d by construction, ⟨y, L x⟩ = ⟨y, d⟩. And
    # ⟨e, x⟩ should equal ⟨y, d⟩ iff L_T y = e and y = L^{-T} e.
    lhs = dot(y, L * x)
    rhs = dot(e_rhs, x)
    @test lhs ≈ rhs rtol=1e-10
end

# =========================================================================
# 4. Kernel vs. pure-Julia reference
# =========================================================================

"""
Pure-Julia column reference: apply one Backward-Euler diffusion step
using `build_diffusion_coefficients` + `solve_tridiagonal!` per column.
Mutates `q_in` in place with the diffused values.
"""
function reference_diffusion_step!(q_in::AbstractArray{FT, 4},
                                    Kz_arr::AbstractArray{FT, 3},
                                    dz_arr::AbstractArray{FT, 3},
                                    dt) where FT
    Nx, Ny, Nz, Nt = size(q_in)
    w = Vector{FT}(undef, Nz)
    for t in 1:Nt, j in 1:Ny, i in 1:Nx
        Kz_col = @view Kz_arr[i, j, :]
        dz_col = @view dz_arr[i, j, :]
        a, b, c = build_diffusion_coefficients(collect(Kz_col), collect(dz_col), dt)
        d = collect(@view q_in[i, j, :, t])
        x = similar(d)
        solve_tridiagonal!(x, a, b, c, d, w)
        q_in[i, j, :, t] .= x
    end
    return q_in
end

@testset "KA kernel matches pure-Julia reference (CPU)" begin
    FT = Float64
    Nx, Ny, Nz, Nt = 3, 2, 6, 2
    dt = 0.5

    # Random but sane Kz in [0.1, 2] m²/s and dz in [50, 500] m
    Kz_arr = FT.(0.1 .+ 1.9 .* rand(Nx, Ny, Nz))
    dz_arr = FT.(50.0 .+ 450.0 .* rand(Nx, Ny, Nz))
    q_in = FT.(rand(Nx, Ny, Nz, Nt))

    q_kernel = copy(q_in)
    q_ref    = copy(q_in)

    # Reference
    reference_diffusion_step!(q_ref, Kz_arr, dz_arr, dt)

    # Kernel
    kz_field = PreComputedKzField(Kz_arr)
    w_scratch = similar(Kz_arr)
    backend = get_backend(q_kernel)
    kernel = _vertical_diffusion_kernel!(backend, (4, 4, 1))
    kernel(q_kernel, kz_field, dz_arr, w_scratch, FT(dt), Nz;
           ndrange = (Nx, Ny, Nt))
    synchronize(backend)

    # Tight agreement — same arithmetic order within a column
    @test q_kernel ≈ q_ref atol=1e-12 rtol=1e-12
end

# =========================================================================
# 5. Analytic: Gaussian broadening under uniform K
# =========================================================================

"""
Fitted (second-moment) variance of profile `q` on grid `z`, treating
`q` as a non-negative density. Used for the Gaussian-broadening test.
"""
function _fitted_variance(q::AbstractVector, z::AbstractVector)
    mass = sum(q)
    μ = sum(q .* z) / mass
    return sum(q .* (z .- μ).^2) / mass
end

@testset "Gaussian broadening (Backward Euler, small K dt/dz²)" begin
    FT = Float64
    Nz = 201
    dz_val = 1.0          # m
    Kz_val = 1.0          # m²/s
    dt = 0.5              # s  — so K·dt = 0.5 m²
    # Backward Euler variance grows by 2·K·dt per step (same as exact diffusion)
    # to first order in K·dt/dz². Here K·dt/dz² = 0.5, modest but not tiny.

    # Cell-center heights
    z = collect(0.5:dz_val:Nz*dz_val - 0.5)
    # Gaussian centered in the middle of the column
    μ0 = z[Nz ÷ 2 + 1]
    σ0² = 25.0
    q_col = exp.(-(z .- μ0).^2 ./ (2 * σ0²))

    # 1×1×Nz×1 layout
    q = reshape(copy(q_col), 1, 1, Nz, 1)
    Kz_arr = fill(Kz_val, 1, 1, Nz)
    dz_arr = fill(dz_val, 1, 1, Nz)

    kz_field = PreComputedKzField(Kz_arr)
    w_scratch = similar(Kz_arr)
    backend = get_backend(q)
    kernel = _vertical_diffusion_kernel!(backend, (1, 1, 1))
    kernel(q, kz_field, dz_arr, w_scratch, FT(dt), Nz; ndrange = (1, 1, 1))
    synchronize(backend)

    σ_new² = _fitted_variance(vec(q[1, 1, :, 1]), z)
    σ_expected² = σ0² + 2 * Kz_val * dt

    # 5% tolerance at K·dt/dz² = 0.5. Tighter would require finer grid or
    # smaller dt; this is a sanity check that the operator's sign, BCs, and
    # flux normalization are correct.
    @test abs(σ_new² - σ_expected²) / σ_expected² < 0.05
end

# =========================================================================
# 6. Mass conservation under Neumann BCs
# =========================================================================

@testset "Neumann BCs conserve column mass to machine precision" begin
    FT = Float64
    Nx, Ny, Nz, Nt = 2, 2, 12, 2
    dt = 1.0

    Kz_arr = FT.(0.1 .+ 2.0 .* rand(Nx, Ny, Nz))
    dz_arr = fill(FT(100.0), Nx, Ny, Nz)          # uniform dz
    q_in   = FT.(rand(Nx, Ny, Nz, Nt))

    mass_before = [sum(q_in[i, j, :, t]) for i in 1:Nx, j in 1:Ny, t in 1:Nt]

    kz_field = PreComputedKzField(Kz_arr)
    w_scratch = similar(Kz_arr)
    backend = get_backend(q_in)
    kernel = _vertical_diffusion_kernel!(backend, (4, 4, 1))
    kernel(q_in, kz_field, dz_arr, w_scratch, FT(dt), Nz;
           ndrange = (Nx, Ny, Nt))
    synchronize(backend)

    mass_after = [sum(q_in[i, j, :, t]) for i in 1:Nx, j in 1:Ny, t in 1:Nt]

    # For uniform dz, Σq is proportional to column-integrated mass.
    # Backward-Euler with Neumann BCs conserves exactly (up to ULP).
    @test maximum(abs.(mass_after .- mass_before) ./ mass_before) < 1e-12
end

# =========================================================================
# 7. ConstantField Kz also works (spans plan-16a interface)
# =========================================================================

@testset "Kernel accepts ConstantField{FT, 3} Kz" begin
    FT = Float64
    Nx, Ny, Nz, Nt = 2, 2, 5, 1
    dt = 0.5
    dz_arr = fill(FT(100.0), Nx, Ny, Nz)
    q_in   = FT.(rand(Nx, Ny, Nz, Nt))

    q_kernel = copy(q_in)
    q_ref    = copy(q_in)

    # Reference with a uniform Kz array
    Kz_val = 1.5
    Kz_arr = fill(FT(Kz_val), Nx, Ny, Nz)
    reference_diffusion_step!(q_ref, Kz_arr, dz_arr, dt)

    # Kernel with ConstantField{FT, 3}
    kz_field = ConstantField{FT, 3}(Kz_val)
    w_scratch = similar(Kz_arr)
    backend = get_backend(q_kernel)
    kernel = _vertical_diffusion_kernel!(backend, (4, 4, 1))
    kernel(q_kernel, kz_field, dz_arr, w_scratch, FT(dt), Nz;
           ndrange = (Nx, Ny, Nt))
    synchronize(backend)

    @test q_kernel ≈ q_ref atol=1e-12 rtol=1e-12
end
