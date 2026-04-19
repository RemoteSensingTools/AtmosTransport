"""
    ProfileKzField{FT}(profile::AbstractVector{FT})

A rank-3 `AbstractTimeVaryingField{FT, 3}` backed by a vertical
`Vector{FT}` of length `Nz`. `field_value(f, (i, j, k))` returns
`f.profile[k]` — horizontally uniform, vertically varying. Time-
independent (constant profile); `update_field!` is a no-op.

Useful for idealized test cases: Chapter-style exponential Kz
profiles (`K0 * exp(-z/H)`), canonical PBL-capped profiles, or any
column diagnostic where the Kz shape is known analytically and
horizontal variation is irrelevant. Not the operational path —
operational Kz comes from `DerivedKzField` (Beljaars-Viterbo) or
`PreComputedKzField` (from binary).

# Storage

`profile` is kept as a host `Vector{FT}`. `field_value` indexes
through a scalar-valued accessor, so kernel code that reads
`field_value(f, idx)` captures only the per-call scalar. The
vector itself is not accessed directly from device code in
diffusion operators that consume this field via `field_value`.

# Examples
```julia
# Single-scale exponential Kz
Kz_profile = 1.0 .* exp.(-(0:33) ./ 5.0)
f = ProfileKzField(collect(Kz_profile))

field_value(f, (1,   1,   1))   # 1.0
field_value(f, (144, 72,  1))   # 1.0  — same k, different (i,j)
field_value(f, (1,   1,  10))   # ≈ 0.135
```
"""
struct ProfileKzField{FT <: AbstractFloat} <: AbstractTimeVaryingField{FT, 3}
    profile :: Vector{FT}
end

@inline field_value(f::ProfileKzField{FT}, idx::NTuple{3, Int}) where {FT} =
    @inbounds f.profile[idx[3]]

update_field!(f::ProfileKzField, ::Real) = f
