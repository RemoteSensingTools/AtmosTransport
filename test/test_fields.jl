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
using AtmosTransport: AbstractTimeVaryingField, ConstantField,
                      field_value, update_field!
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
