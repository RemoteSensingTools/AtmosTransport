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
                      PreComputedKzField, field_value, update_field!
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
