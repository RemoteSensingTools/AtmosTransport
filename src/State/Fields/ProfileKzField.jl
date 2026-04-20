"""
    ProfileKzField{FT, V}(profile::AbstractVector{FT})

A rank-3 `AbstractTimeVaryingField{FT, 3}` backed by a vertical
vector of length `Nz`. `field_value(f, (i, j, k))` returns
`f.profile[k]` — horizontally uniform, vertically varying. Time-
independent (constant profile); `update_field!` is a no-op.

Useful for idealized test cases: Chapter-style exponential Kz
profiles (`K0 * exp(-z/H)`), canonical PBL-capped profiles, or any
column diagnostic where the Kz shape is known analytically and
horizontal variation is irrelevant. Not the operational path —
operational Kz comes from `DerivedKzField` (Beljaars-Viterbo) or
`PreComputedKzField` (from binary).

# Storage

Parametric on the vector type `V <: AbstractVector{FT}`. The
default constructor accepts a host `Vector{FT}`; `Adapt.adapt` can
then convert the field to device storage (e.g., `CuArray{FT, 1}`)
for GPU kernel launches. `field_value` dispatches through the
vector's `getindex`, which is kernel-safe on all supported
backends once the vector lives on-device.

# Examples
```julia
# CPU — single-scale exponential Kz
Kz_profile = 1.0 .* exp.(-(0:33) ./ 5.0)
f = ProfileKzField(collect(Kz_profile))

field_value(f, (1,   1,   1))   # 1.0
field_value(f, (144, 72,  1))   # 1.0  — same k, different (i,j)
field_value(f, (1,   1,  10))   # ≈ 0.135

# GPU — adapt to a CuArray backing before kernel launch
# using CUDA, Adapt
# f_gpu = Adapt.adapt(CuArray, f)   # f_gpu.profile isa CuArray{Float64, 1}
```
"""
struct ProfileKzField{FT <: AbstractFloat,
                      V <: AbstractVector{FT}} <: AbstractTimeVaryingField{FT, 3}
    profile :: V
end

# Julia's auto-generated outer constructor `ProfileKzField(::V)` infers
# both type parameters from the vector's `eltype` and concrete type —
# no explicit outer method needed.

@inline field_value(f::ProfileKzField, idx::NTuple{3, Int}) =
    @inbounds f.profile[idx[3]]

update_field!(f::ProfileKzField, ::Real) = f

# Adapt hook: convert the backing vector to the requested device.
# This is how KA kernels receive a GPU-usable `ProfileKzField` —
# they call `Adapt.adapt(to_gpu_array, field)` before capturing it,
# and `f.profile` becomes a `CuArray` / `MtlArray` / etc.
Adapt.adapt_structure(to, f::ProfileKzField) =
    ProfileKzField(Adapt.adapt(to, f.profile))
