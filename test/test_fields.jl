"""
Unit tests for the `TimeVaryingField` abstraction (plan 16a).

Validates the minimum concrete type `ConstantField{FT, N}` against
the interface specified in `docs/plans/TIME_VARYING_FIELD_MODEL.md`:

1. `field_value(f, idx)` returns the stored scalar regardless of index
2. `update_field!(f, t)` is a no-op (returns `f`, no state change)
3. Type stability at every rank used by downstream plans (N ∈ {0, 2, 3})
4. Kernel-safety on CPU (callable inside a KA kernel)

Plan 16b adds N = 3 concrete types (`ProfileKzField`, etc.) and
extends this suite.
"""

using Test
using AtmosTransport: AbstractTimeVaryingField, ConstantField, ProfileKzField,
                      PreComputedKzField, DerivedKzField, PBLPhysicsParameters,
                      field_value, update_field!
using AtmosTransport.State.Fields: _beljaars_viterbo_kz, _obukhov_length,
                                    _prandtl_inverse
using KernelAbstractions: @kernel, @index, get_backend, synchronize

@testset "TimeVaryingField — ConstantField" begin

    @testset "scalar (N=0) — Float64" begin
        f = ConstantField{Float64, 0}(2.098e-6)
        @test f isa AbstractTimeVaryingField{Float64, 0}
        @test field_value(f, ()) === 2.098e-6
        @test update_field!(f, 0.0) === f
        @test update_field!(f, 1.0e9) === f
        # idempotent updates leave field_value unchanged
        update_field!(f, 1.0)
        @test field_value(f, ()) === 2.098e-6
    end

    @testset "scalar (N=0) — Float32" begin
        f = ConstantField{Float32, 0}(3.0f0)
        @test field_value(f, ()) === 3.0f0
        @test update_field!(f, 0.0) === f
    end

    @testset "Real → FT coercion in constructor" begin
        # Integer / other-float input coerced to FT
        f = ConstantField{Float64, 0}(1)
        @test field_value(f, ()) === 1.0
        g = ConstantField{Float32, 0}(2.5)   # Float64 → Float32
        @test field_value(g, ()) === 2.5f0
    end

    @testset "surface (N=2)" begin
        f = ConstantField{Float64, 2}(42.0)
        @test f isa AbstractTimeVaryingField{Float64, 2}
        @test field_value(f, (1, 1)) === 42.0
        @test field_value(f, (1000, 500)) === 42.0
    end

    @testset "volume (N=3) — anticipating Kz in plan 16b" begin
        f = ConstantField{Float64, 3}(1.5)
        @test f isa AbstractTimeVaryingField{Float64, 3}
        @test field_value(f, (1, 1, 1)) === 1.5
        @test field_value(f, (144, 72, 32)) === 1.5
    end

    @testset "rank-mismatched index does not dispatch" begin
        f = ConstantField{Float64, 3}(1.0)
        # Cannot call field_value with a 2-tuple on a rank-3 field
        @test_throws MethodError field_value(f, (1, 1))
        # Cannot call with empty tuple on rank-3
        @test_throws MethodError field_value(f, ())
    end

    @testset "type stability" begin
        f = ConstantField{Float64, 0}(1.0)
        @test @inferred(field_value(f, ())) === 1.0

        g = ConstantField{Float32, 3}(0.5f0)
        @test @inferred(field_value(g, (1, 2, 3))) === 0.5f0
    end

    @testset "kernel-safety — CPU backend" begin
        # field_value must be callable from inside a KA kernel without
        # allocation or dynamic dispatch. Launch a trivial kernel that
        # writes the field's value into every cell of an output array.
        f = ConstantField{Float64, 0}(7.25)
        out = zeros(Float64, 4, 3, 2)
        backend = get_backend(out)

        @kernel function _fill_from_scalar_field!(out, field)
            i, j, k = @index(Global, NTuple)
            @inbounds out[i, j, k] = field_value(field, ())
        end

        _fill_from_scalar_field!(backend, 64)(out, f; ndrange = size(out))
        synchronize(backend)
        @test all(out .== 7.25)

        # Rank-3 field
        g = ConstantField{Float64, 3}(-1.0)
        fill!(out, 0.0)
        @kernel function _fill_from_volume_field!(out, field)
            i, j, k = @index(Global, NTuple)
            @inbounds out[i, j, k] = field_value(field, (i, j, k))
        end
        _fill_from_volume_field!(backend, 64)(out, g; ndrange = size(out))
        synchronize(backend)
        @test all(out .== -1.0)
    end
end

@testset "TimeVaryingField — ProfileKzField" begin

    @testset "construction + type bounds" begin
        profile = [0.1, 0.5, 1.0, 2.0, 5.0]
        f = ProfileKzField(profile)
        @test f isa AbstractTimeVaryingField{Float64, 3}
        @test f isa ProfileKzField{Float64}

        g = ProfileKzField(Float32[0.0, 1.0, 2.0])
        @test g isa AbstractTimeVaryingField{Float32, 3}
    end

    @testset "field_value selects the k coordinate" begin
        profile = [10.0, 20.0, 30.0, 40.0, 50.0]
        f = ProfileKzField(profile)
        @test field_value(f, (1, 1, 1)) === 10.0
        @test field_value(f, (1, 1, 2)) === 20.0
        @test field_value(f, (1, 1, 3)) === 30.0
        @test field_value(f, (1, 1, 5)) === 50.0
    end

    @testset "field_value ignores i, j" begin
        profile = [1.0, 2.0, 3.0]
        f = ProfileKzField(profile)
        # Same k, different (i, j) → same value
        @test field_value(f, (1,   1,  2)) === 2.0
        @test field_value(f, (100, 50, 2)) === 2.0
        @test field_value(f, (1,   72, 2)) === 2.0
        @test field_value(f, (144, 1,  2)) === 2.0
    end

    @testset "update_field! is a no-op" begin
        profile = [1.0, 2.0, 3.0]
        f = ProfileKzField(profile)
        @test update_field!(f, 0.0) === f
        @test update_field!(f, 1.0e9) === f
        # Profile unchanged
        @test f.profile == [1.0, 2.0, 3.0]
        # Subsequent reads still return the originals
        @test field_value(f, (1, 1, 1)) === 1.0
        @test field_value(f, (1, 1, 3)) === 3.0
    end

    @testset "type stability" begin
        f = ProfileKzField([1.0, 2.0, 3.0])
        @test @inferred(field_value(f, (1, 1, 1))) === 1.0

        g = ProfileKzField(Float32[0.5f0, 1.5f0])
        @test @inferred(field_value(g, (1, 1, 2))) === 1.5f0
    end

    @testset "rank-mismatched index does not dispatch" begin
        f = ProfileKzField([1.0, 2.0, 3.0])
        @test_throws MethodError field_value(f, (1, 1))
        @test_throws MethodError field_value(f, ())
    end

    @testset "kernel-safety — CPU backend" begin
        # Verify field_value is callable from inside a KA kernel.
        # Write a k-varying profile into every column of an output array.
        profile = [10.0, 20.0, 30.0, 40.0]
        f = ProfileKzField(profile)
        out = zeros(Float64, 3, 2, 4)
        backend = get_backend(out)

        @kernel function _fill_from_profile_field!(out, field)
            i, j, k = @index(Global, NTuple)
            @inbounds out[i, j, k] = field_value(field, (i, j, k))
        end

        _fill_from_profile_field!(backend, 64)(out, f; ndrange = size(out))
        synchronize(backend)

        # Every column should match the profile
        for i in 1:3, j in 1:2
            @test out[i, j, :] == profile
        end
    end
end

@testset "TimeVaryingField — PreComputedKzField" begin

    @testset "construction + type bounds" begin
        data = rand(Float64, 4, 3, 5)
        f = PreComputedKzField(data)
        @test f isa AbstractTimeVaryingField{Float64, 3}
        @test f isa PreComputedKzField{Float64}

        data32 = rand(Float32, 2, 2, 3)
        g = PreComputedKzField(data32)
        @test g isa AbstractTimeVaryingField{Float32, 3}
    end

    @testset "field_value respects (i, j, k) independently" begin
        # Construct data where each cell's value is a unique fingerprint
        data = Array{Float64, 3}(undef, 4, 3, 5)
        for i in 1:4, j in 1:3, k in 1:5
            data[i, j, k] = 100 * i + 10 * j + k
        end
        f = PreComputedKzField(data)

        @test field_value(f, (1, 1, 1)) === 111.0
        @test field_value(f, (4, 3, 5)) === 435.0
        @test field_value(f, (2, 1, 3)) === 213.0
        # Varying only i or only j must change the value (vs. ProfileKzField)
        @test field_value(f, (1, 1, 2)) !== field_value(f, (2, 1, 2))
        @test field_value(f, (1, 1, 2)) !== field_value(f, (1, 2, 2))
    end

    @testset "update_field! is a no-op" begin
        data = [1.0; 2.0; 3.0 ;; 4.0; 5.0; 6.0 ;;; 7.0; 8.0; 9.0 ;; 10.0; 11.0; 12.0]
        # Built as 3×2×2. Content doesn't matter; we test identity + preservation.
        f = PreComputedKzField(data)
        pre = copy(f.data)
        @test update_field!(f, 0.0) === f
        @test update_field!(f, 1.0e9) === f
        @test f.data == pre
    end

    @testset "caller-owned storage: external mutation visible" begin
        # The field does not own the array — mutating the backing array
        # (e.g. when met-window advances) is immediately visible through
        # field_value.
        data = zeros(Float64, 2, 2, 2)
        f = PreComputedKzField(data)
        @test field_value(f, (1, 1, 1)) === 0.0

        data[1, 1, 1] = 42.0
        @test field_value(f, (1, 1, 1)) === 42.0

        fill!(data, -1.0)
        @test field_value(f, (2, 2, 2)) === -1.0
    end

    @testset "type stability" begin
        data = rand(Float64, 3, 3, 3)
        f = PreComputedKzField(data)
        @test @inferred(field_value(f, (1, 1, 1))) isa Float64

        data32 = rand(Float32, 2, 2, 2)
        g = PreComputedKzField(data32)
        @test @inferred(field_value(g, (1, 1, 1))) isa Float32
    end

    @testset "rank-mismatched construction rejected" begin
        # 2D or 4D arrays must not be accepted — rank-3 is the interface contract
        @test_throws MethodError PreComputedKzField(rand(Float64, 3, 3))
        @test_throws MethodError PreComputedKzField(rand(Float64, 2, 2, 2, 2))
    end

    @testset "kernel-safety — CPU backend" begin
        # Launch a KA kernel that reads every element of the backing array
        # through field_value and copies it into an output buffer. If the
        # field path matches direct array indexing, the output equals the
        # input elementwise.
        Nx, Ny, Nz = 3, 4, 5
        src = reshape(collect(1.0:Nx*Ny*Nz), Nx, Ny, Nz)
        f = PreComputedKzField(src)
        out = zeros(Float64, Nx, Ny, Nz)
        backend = get_backend(out)

        @kernel function _copy_from_volume_field!(out, field)
            i, j, k = @index(Global, NTuple)
            @inbounds out[i, j, k] = field_value(field, (i, j, k))
        end

        _copy_from_volume_field!(backend, 64)(out, f; ndrange = size(out))
        synchronize(backend)
        @test out == src
    end
end

@testset "TimeVaryingField — PBLPhysicsParameters defaults" begin
    p = PBLPhysicsParameters{Float64}()
    @test p.β_h      === 15.0
    @test p.Kz_bg    === 0.1
    @test p.Kz_min   === 0.01
    @test p.Kz_max   === 500.0
    @test p.kappa_vk === 0.41
    @test p.gravity  === 9.80665
    @test p.cp_dry   === 1004.64
    @test p.rho_ref  === 1.225

    # FT propagation
    p32 = PBLPhysicsParameters{Float32}()
    @test p32.β_h isa Float32
    @test p32.Kz_bg === 0.1f0

    # kwarg overrides
    p_custom = PBLPhysicsParameters{Float64}(β_h = 20.0, Kz_min = 0.001)
    @test p_custom.β_h === 20.0
    @test p_custom.Kz_min === 0.001
    @test p_custom.Kz_bg === 0.1   # defaults preserved
end

@testset "TimeVaryingField — _beljaars_viterbo_kz" begin
    p = PBLPhysicsParameters{Float64}()

    @testset "above 1.2 h_pbl returns Kz_bg exactly" begin
        h_pbl = 1000.0
        for z in (1200.0, 1500.0, 5000.0, 1e5)
            Kz = _beljaars_viterbo_kz(z, h_pbl, 0.3, -50.0, 1.0, p)
            @test Kz === p.Kz_bg
        end
    end

    @testset "stable BL spot value (z < h)" begin
        # L_ob > 0 (stable). Kz = u* κ z (1-z/h)² / (1 + 5z/L)
        h_pbl = 1000.0
        us    = 0.3
        z     = 100.0
        L_ob  = 50.0
        expected = us * p.kappa_vk * z * (1 - z/h_pbl)^2 / (1 + 5 * z / L_ob)
        Kz = _beljaars_viterbo_kz(z, h_pbl, us, L_ob, 1.0, p)
        @test Kz ≈ expected atol=1e-12
        @test p.Kz_min <= Kz <= p.Kz_max  # inside clamps
    end

    @testset "unstable surface layer spot value (z < 0.1 h)" begin
        # L_ob < 0. z < 0.1 h → surface-layer branch.
        # Kz = u* κ z (1-z/h)² (1 - β_h z/L)^(1/3) × Pr_inv
        h_pbl, us, z, L_ob, Pr_inv = 1000.0, 0.3, 50.0, -40.0, 2.0
        expected = us * p.kappa_vk * z * (1 - z/h_pbl)^2 *
                   cbrt(1 - p.β_h * z / L_ob) * Pr_inv
        Kz = _beljaars_viterbo_kz(z, h_pbl, us, L_ob, Pr_inv, p)
        @test Kz ≈ expected atol=1e-12
    end

    @testset "unstable mixed layer spot value (0.1 h < z < h)" begin
        # L_ob < 0. z ≥ 0.1 h → mixed-layer branch.
        # w_m = u* (1 - 0.1 β_h h/L)^(1/3); Kz = w_m κ z (1-z/h)² × Pr_inv
        h_pbl, us, z, L_ob, Pr_inv = 1000.0, 0.3, 500.0, -40.0, 2.0
        w_m = us * cbrt(1 - 0.1 * p.β_h * h_pbl / L_ob)
        expected = w_m * p.kappa_vk * z * (1 - z/h_pbl)^2 * Pr_inv
        Kz = _beljaars_viterbo_kz(z, h_pbl, us, L_ob, Pr_inv, p)
        @test Kz ≈ expected atol=1e-12
    end

    @testset "clamping to Kz_max" begin
        # Very small L_ob (strongly unstable), large Pr_inv → would blow past 500
        h_pbl, us, z, L_ob, Pr_inv = 1000.0, 2.0, 500.0, -5.0, 10.0
        Kz = _beljaars_viterbo_kz(z, h_pbl, us, L_ob, Pr_inv, p)
        @test Kz === p.Kz_max
    end

    @testset "clamping to Kz_min" begin
        # Very small ustar → Kz → 0; gets clamped up to Kz_min
        h_pbl, us, z, L_ob, Pr_inv = 1000.0, 1e-6, 10.0, 50.0, 1.0
        Kz = _beljaars_viterbo_kz(z, h_pbl, us, L_ob, Pr_inv, p)
        @test Kz === p.Kz_min
    end

    @testset "taper zone (h ≤ z < 1.2 h): linear blend to Kz_bg" begin
        # At z = h_pbl exactly, taper has frac = (1.2h - h) / (0.2h) = 1 and
        # z_eff = min(z, h-1) = h-1, so (1-z_eff/h)² ≈ 1/h² — the raw Kz is
        # tiny. After clamp + taper-blend at frac=1 the returned value equals
        # the clamped in-PBL value. At z = 1.1h, frac = 0.5 → 50/50 blend.
        # At z = 1.2h, frac = 0 → Kz_bg.
        h_pbl, us, L_ob, Pr_inv = 1000.0, 0.3, 50.0, 1.0
        Kz_at_h   = _beljaars_viterbo_kz(h_pbl,     h_pbl, us, L_ob, Pr_inv, p)
        Kz_at_11h = _beljaars_viterbo_kz(1.1*h_pbl, h_pbl, us, L_ob, Pr_inv, p)
        Kz_at_12h = _beljaars_viterbo_kz(1.2*h_pbl, h_pbl, us, L_ob, Pr_inv, p)

        @test Kz_at_12h === p.Kz_bg
        # Blend formula: Kz = Kz_bg + frac × (Kz_in_pbl_clamped - Kz_bg)
        expected_11h = p.Kz_bg + 0.5 * (Kz_at_h - p.Kz_bg)
        @test Kz_at_11h ≈ expected_11h atol=1e-12
        # Direction of the blend depends on whether the clamped in-PBL value
        # exceeds Kz_bg (common in strong mixing) or sits below (as here,
        # where (1-zh)² ≈ 1/h² drives the raw Kz below Kz_min so it clamps
        # to Kz_min = 0.01 < Kz_bg = 0.1 → Kz increases through the taper).
        @test Kz_at_h <= Kz_at_11h <= Kz_at_12h
    end

    @testset "type stability" begin
        p64 = PBLPhysicsParameters{Float64}()
        @test @inferred(_beljaars_viterbo_kz(100.0, 1000.0, 0.3, -40.0, 1.5, p64)) isa Float64
        p32 = PBLPhysicsParameters{Float32}()
        @test @inferred(_beljaars_viterbo_kz(100f0, 1000f0, 0.3f0, -40f0, 1.5f0, p32)) isa Float32
    end
end

@testset "TimeVaryingField — _obukhov_length" begin
    p = PBLPhysicsParameters{Float64}()

    @testset "positive hflux → unstable (L < 0)" begin
        L_ob, H_kin = _obukhov_length(100.0, 0.3, 295.0, p)
        @test H_kin > 0
        @test L_ob < 0
    end

    @testset "negative hflux → stable (L > 0)" begin
        L_ob, H_kin = _obukhov_length(-50.0, 0.3, 280.0, p)
        @test H_kin < 0
        @test L_ob > 0
    end

    @testset "zero hflux → finite large L via safety offset" begin
        L_ob, H_kin = _obukhov_length(0.0, 0.3, 285.0, p)
        @test H_kin === 0.0
        @test isfinite(L_ob)
        @test abs(L_ob) > 1e9   # very large, no singularity
    end

    @testset "hand-verified spot value" begin
        # hflux=100, u*=0.3, t2m=295
        # H_kin = 100 / (1.225 * 1004.64) = 0.0813 K m/s
        # L_ob = -295 × 0.027 / (0.41 × 9.80665 × 0.0813 (+ offset))
        hflux, us, t2m = 100.0, 0.3, 295.0
        H_kin_ref = hflux / (p.rho_ref * p.cp_dry)
        H_safe_ref = H_kin_ref + sign(H_kin_ref + 1e-20) * 1e-10
        L_ob_ref = -t2m * us^3 / (p.kappa_vk * p.gravity * H_safe_ref)
        L_ob, H_kin = _obukhov_length(hflux, us, t2m, p)
        @test H_kin ≈ H_kin_ref atol=1e-12
        @test L_ob  ≈ L_ob_ref  atol=1e-6
    end
end

@testset "TimeVaryingField — _prandtl_inverse" begin
    p = PBLPhysicsParameters{Float64}()

    @testset "stable branch returns 1" begin
        # L_ob > 0, H_kin can be anything
        @test _prandtl_inverse(1000.0, 0.3, 0.1, 285.0, 50.0, p) === 1.0
        @test _prandtl_inverse(1000.0, 0.3, -0.1, 285.0, 50.0, p) === 1.0
    end

    @testset "non-positive H_kin returns 1" begin
        # Even with L_ob < 0 (unstable), if H_kin ≤ 0 formula doesn't apply
        @test _prandtl_inverse(1000.0, 0.3, 0.0, 285.0, -50.0, p) === 1.0
        @test _prandtl_inverse(1000.0, 0.3, -0.05, 285.0, -50.0, p) === 1.0
    end

    @testset "tiny h_pbl returns 1" begin
        @test _prandtl_inverse(5.0, 0.3, 0.1, 285.0, -50.0, p) === 1.0
    end

    @testset "unstable convective → > 1" begin
        # Classic daytime setup
        Pr_inv = _prandtl_inverse(1000.0, 0.3, 0.08, 295.0, -25.0, p)
        @test Pr_inv > 1.0
        @test isfinite(Pr_inv)
    end
end

# Helpers for DerivedKzField tests — module scope so `where {FT}` binds correctly
_make_delp_field(Nx, Ny, delp_col::Vector{FT}) where FT = begin
    data = Array{FT}(undef, Nx, Ny, length(delp_col))
    for i in 1:Nx, j in 1:Ny, k in eachindex(delp_col)
        data[i, j, k] = delp_col[k]
    end
    PreComputedKzField(data)
end

_constant_surface(::Type{FT}; pblh, ustar, hflux, t2m) where FT = (
    pblh  = ConstantField{FT, 2}(pblh),
    ustar = ConstantField{FT, 2}(ustar),
    hflux = ConstantField{FT, 2}(hflux),
    t2m   = ConstantField{FT, 2}(t2m),
)

@testset "TimeVaryingField — DerivedKzField" begin

    @testset "construction + type bounds" begin
        FT = Float64
        Nx, Ny, Nz = 2, 3, 5
        surface = _constant_surface(FT; pblh=1000.0, ustar=0.3, hflux=100.0, t2m=295.0)
        delp    = _make_delp_field(Nx, Ny, fill(20000.0, Nz))
        cache   = zeros(FT, Nx, Ny, Nz)
        f       = DerivedKzField(; surface, delp, cache)

        @test f isa AbstractTimeVaryingField{Float64, 3}
        @test f isa DerivedKzField{Float64}
        @test f.params isa PBLPhysicsParameters{Float64}
    end

    @testset "constructor rejects surface fields of wrong rank" begin
        FT = Float64
        bad_surface = (
            pblh  = ConstantField{FT, 3}(1000.0),   # should be rank-2
            ustar = ConstantField{FT, 2}(0.3),
            hflux = ConstantField{FT, 2}(100.0),
            t2m   = ConstantField{FT, 2}(295.0),
        )
        delp  = _make_delp_field(2, 2, fill(20000.0, 3))
        cache = zeros(FT, 2, 2, 3)
        @test_throws ArgumentError DerivedKzField(;
            surface=bad_surface, delp=delp, cache=cache)
    end

    @testset "update_field! populates cache" begin
        FT = Float64
        Nx, Ny, Nz = 2, 2, 10
        surface = _constant_surface(FT; pblh=3000.0, ustar=0.3,
                                    hflux=100.0, t2m=295.0)
        # 10 layers, roughly 10 kPa each, surface = 101325 Pa
        delp    = _make_delp_field(Nx, Ny, fill(10132.5, Nz))
        cache   = fill(-1.0, Nx, Ny, Nz)   # sentinel so we see writes
        f       = DerivedKzField(; surface, delp, cache)

        # Before update — sentinel present
        @test all(f.cache .== -1.0)

        update_field!(f, 0.0)

        # After update — all cells written, no negatives
        @test all(f.cache .>= 0.0)
        @test all(isfinite, f.cache)
        # Horizontal uniformity (all (i,j) columns have same surface fields and delp)
        @test f.cache[1, 1, :] == f.cache[2, 2, :]
        @test f.cache[1, 2, :] == f.cache[2, 1, :]
        # All within clamps [Kz_min, Kz_max]
        @test all(f.cache .>= f.params.Kz_min - 1e-12)
        @test all(f.cache .<= f.params.Kz_max + 1e-12)
    end

    @testset "high levels (above 1.2 h_pbl) → Kz_bg" begin
        # With h_pbl = 500 m and tall layers, the top cells sit above 1.2h
        FT = Float64
        Nx, Ny, Nz = 2, 2, 10
        surface = _constant_surface(FT; pblh=500.0, ustar=0.3,
                                    hflux=100.0, t2m=295.0)
        delp    = _make_delp_field(Nx, Ny, fill(10132.5, Nz))
        cache   = zeros(FT, Nx, Ny, Nz)
        f       = DerivedKzField(; surface, delp, cache)
        update_field!(f, 0.0)

        # k=1 is TOA; top few levels are well above 1.2 × 500 = 600 m
        @test f.cache[1, 1, 1] === f.params.Kz_bg
        @test f.cache[1, 1, 2] === f.params.Kz_bg
    end

    @testset "field_value reads from cache" begin
        FT = Float64
        Nx, Ny, Nz = 2, 2, 5
        surface = _constant_surface(FT; pblh=1500.0, ustar=0.3,
                                    hflux=100.0, t2m=295.0)
        delp    = _make_delp_field(Nx, Ny, fill(20265.0, Nz))
        cache   = zeros(FT, Nx, Ny, Nz)
        f       = DerivedKzField(; surface, delp, cache)
        update_field!(f, 0.0)

        for i in 1:Nx, j in 1:Ny, k in 1:Nz
            @test field_value(f, (i, j, k)) === f.cache[i, j, k]
        end
    end

    @testset "changing surface fields changes cache on next update" begin
        # Mutable surface field: wrap hflux in a PreComputedKzField
        # so we can flip it, re-update, and see a different cache.
        FT = Float64
        Nx, Ny, Nz = 2, 2, 5

        hflux_arr = fill(100.0, Nx, Ny)
        # Use a trivial 2D "constant-but-mutable" wrapper: simplest is to
        # just re-construct the field with a new ConstantField value and
        # re-call update_field!.
        surface_pos = _constant_surface(FT; pblh=1500.0, ustar=0.3,
                                        hflux=100.0, t2m=295.0)
        surface_neg = _constant_surface(FT; pblh=1500.0, ustar=0.3,
                                        hflux=-50.0, t2m=295.0)
        delp = _make_delp_field(Nx, Ny, fill(20265.0, Nz))

        cache1 = zeros(FT, Nx, Ny, Nz)
        f1 = DerivedKzField(; surface=surface_pos, delp, cache=cache1)
        update_field!(f1, 0.0)

        cache2 = zeros(FT, Nx, Ny, Nz)
        f2 = DerivedKzField(; surface=surface_neg, delp, cache=cache2)
        update_field!(f2, 0.0)

        # The PBL-level cells (below 1.2 × 1500 = 1800 m) must differ —
        # unstable vs stable branch → substantially different Kz.
        # The top of the column is above taper for both, so those cells
        # are the same (Kz_bg). Check the bottom cell (surface layer).
        @test cache1[1, 1, Nz] != cache2[1, 1, Nz]
    end

    @testset "type stability of field_value" begin
        FT = Float64
        surface = _constant_surface(FT; pblh=1000.0, ustar=0.3,
                                    hflux=100.0, t2m=295.0)
        delp    = _make_delp_field(2, 2, fill(20000.0, 3))
        cache   = zeros(FT, 2, 2, 3)
        f       = DerivedKzField(; surface, delp, cache)
        update_field!(f, 0.0)
        @test @inferred(field_value(f, (1, 1, 1))) isa Float64
    end
end
