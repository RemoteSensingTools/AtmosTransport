"""
    PreComputedKzField{FT, A}(data::AbstractArray{FT, 3})

A rank-3 `AbstractTimeVaryingField{FT, 3}` backed directly by a
3D array. `field_value(f, (i, j, k))` returns `f.data[i, j, k]`
— the no-abstraction path for Kz profiles read from a pre-computed
source (met binary snapshot, GCHP-style 3D diffusivity export, etc.).

Time variation, if any, is handled by the caller: mutate `f.data`
in place when the current met window advances, then call (or skip)
`update_field!` — this method is a no-op and does not own the
storage. A more structured stepwise-in-time backing (e.g. a 4D
buffer with an internal window index) can be added later as a
separate concrete type; keeping this one minimal makes the contract
transparent.

# Backend

`A <: AbstractArray{FT, 3}` is parametric: `Array{FT, 3}` for CPU,
`CuArray{FT, 3}` / `MtlArray{FT, 3}` for GPU runs. `field_value`
indexes the array element-wise, which is kernel-safe on all three
backends.

# Examples
```julia
# CPU: static 3D Kz snapshot
Kz = fill(1.0, 144, 72, 34)
f  = PreComputedKzField(Kz)

field_value(f, (1, 1, 1))      # 1.0
field_value(f, (144, 72, 34))  # 1.0

# Updating the snapshot between met windows (caller-owned)
Kz .= 2.0
field_value(f, (1, 1, 1))      # 2.0  — same f, updated data
```
"""
struct PreComputedKzField{FT <: AbstractFloat,
                          A <: AbstractArray} <: AbstractTimeVaryingField{FT, 3}
    data :: A

    function PreComputedKzField{FT, A}(data::A) where {FT, A <: AbstractArray{FT, 3}}
        new{FT, A}(data)
    end
end

PreComputedKzField(data::A) where {FT, A <: AbstractArray{FT, 3}} =
    PreComputedKzField{FT, A}(data)

@inline field_value(f::PreComputedKzField, idx::NTuple{3, Int}) =
    @inbounds f.data[idx[1], idx[2], idx[3]]

update_field!(f::PreComputedKzField, ::Real) = f

# Adapt hook: lets KA kernels receive a device-shaped `PreComputedKzField`.
# When `data::CuArray{...}` (non-bitstype, holds a host-side pointer
# wrapper), `Adapt.adapt(to, f.data)` becomes a `CuDeviceArray` usable
# from inside the kernel. Mirrors the `ProfileKzField` pattern.
Adapt.adapt_structure(to, f::PreComputedKzField) =
    PreComputedKzField(Adapt.adapt(to, f.data))
