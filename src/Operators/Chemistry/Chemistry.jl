"""
    Chemistry

Source/sink operators for tracer transformations (decay, photolysis, ...).

Type hierarchy (plan 15):

    AbstractChemistryOperator
    ├── NoChemistry                     — identity / inert tracers
    ├── ExponentialDecay{FT, N}         — multi-tracer first-order decay
    └── CompositeChemistry              — sequential composition

Interface (OPERATOR_COMPOSITION.md §6):

    apply!(state::CellState, meteo, grid, op::AbstractChemistryOperator, dt;
           workspace=nothing)

The operator mutates `state.tracers_raw` in place and returns `state`.
`meteo`, `grid`, and `workspace` are accepted for interface conformance
and may be `nothing` for operators that do not need them (pure decay).

Multi-tracer decay is fused into a single KernelAbstractions kernel —
see `chemistry_kernels.jl`. Tracers not listed in the operator's
`tracer_names` are left untouched.
"""
module Chemistry

using KernelAbstractions: get_backend, synchronize

using ...State: CellState
using ...State: ntracers, tracer_index, tracer_names
using ...State: AbstractTimeVaryingField, ConstantField, field_value, update_field!
using ...MetDrivers: current_time
import ..apply!

export AbstractChemistryOperator, NoChemistry, ExponentialDecay, CompositeChemistry
export chemistry_block!

include("chemistry_kernels.jl")

# =========================================================================
# Type hierarchy
# =========================================================================

abstract type AbstractChemistryOperator end

"""
    NoChemistry()

Identity operator — `apply!` is a no-op. Default for runs without active
chemistry.
"""
struct NoChemistry <: AbstractChemistryOperator end

"""
    ExponentialDecay{FT, N, R}(decay_rates, tracer_names)

Multi-tracer first-order decay: `c *= exp(-rate * dt)` applied in-place
to every selected tracer at every cell. Exact for constant rate and any
`dt`; unconditionally stable; trivially parallel.

# Fields
- `decay_rates  :: R` — an `NTuple{N, <: AbstractTimeVaryingField{FT, 0}}`
  of rate-valued fields, one per selected tracer [1/s]. `apply!` calls
  `update_field!` on each rate before launching the kernel.
- `tracer_names :: NTuple{N, Symbol}` — which tracers this operator applies to

# Construction
```julia
ExponentialDecay(; Rn222 = 330_350.4)                   # from half-lives [s]
ExponentialDecay(Float32; Rn222 = 330_350.4, Kr85 = 3.394e8)
```
The keyword constructor converts half-life `T` to decay rate
`λ = log(2) / T` (first-order exponential decay) and wraps each rate
in a `ConstantField{FT, 0}`. Future plans may pass time-varying rates
(e.g. temperature-dependent reaction rates) through the same field
interface.

Common isotopes:
- ²²²Rn: half-life = 330_350.4 s (3.8235 days) → λ ≈ 2.098e-6 s⁻¹
- ⁸⁵Kr:  half-life = 3.394e8 s (10.76 years)  → λ ≈ 2.042e-9 s⁻¹
"""
struct ExponentialDecay{FT, N, R} <: AbstractChemistryOperator
    decay_rates  :: R
    tracer_names :: NTuple{N, Symbol}

    function ExponentialDecay{FT, N, R}(rates::R, names::NTuple{N, Symbol}) where {FT, N, R}
        R <: NTuple{N, AbstractTimeVaryingField{FT, 0}} ||
            throw(ArgumentError("ExponentialDecay: decay_rates must be an " *
                "NTuple{$N, <:AbstractTimeVaryingField{$FT, 0}}, got $R"))
        return new{FT, N, R}(rates, names)
    end
end

"Keyword constructor: `ExponentialDecay(; Rn222 = half_life_seconds, ...)`."
function ExponentialDecay(FT::Type{<:AbstractFloat} = Float64; half_lives...)
    nt = NamedTuple(half_lives)
    names = keys(nt)
    N = length(names)
    rates = ntuple(i -> ConstantField{FT, 0}(FT(log(2) / nt[i])), N)
    return ExponentialDecay{FT, N, typeof(rates)}(rates, names)
end

"""
    CompositeChemistry(schemes...)
    CompositeChemistry(schemes::Tuple)

Apply multiple chemistry operators sequentially. Used when different
species need independent transformations or when different operator
types (decay + photolysis + ...) must run in a prescribed order.

```julia
chem = CompositeChemistry(
    ExponentialDecay(; Rn222 = 330_350.4),
    ExponentialDecay(; Kr85  = 3.394e8),
)
```
"""
struct CompositeChemistry{S <: Tuple} <: AbstractChemistryOperator
    schemes :: S
end

CompositeChemistry(schemes::AbstractChemistryOperator...) = CompositeChemistry(schemes)

# =========================================================================
# apply! dispatch
# =========================================================================

"""
    apply!(state::CellState, meteo, grid, op::NoChemistry, dt; workspace=nothing)

No-op — returns `state` unchanged.
"""
function apply!(state::CellState, meteo, grid, ::NoChemistry, dt;
                workspace = nothing)
    return state
end

"""
    apply!(state::CellState, meteo, grid, op::ExponentialDecay, dt; workspace=nothing)

Decay every tracer listed in `op.tracer_names` by `exp(-rate * dt)` in
place. `meteo`, `grid`, and `workspace` are unused (accepted for
interface conformance with other operators).

Throws `ArgumentError` if any name in `op.tracer_names` is not carried by
`state`.
"""
function apply!(state::CellState, meteo, grid,
                op::ExponentialDecay{FT, N}, dt;
                workspace = nothing) where {FT, N}
    N == 0 && return state

    # Resolve names → indices at call time.
    indices = ntuple(N) do n
        idx = tracer_index(state, op.tracer_names[n])
        if idx === nothing
            throw(ArgumentError("ExponentialDecay: tracer $(op.tracer_names[n]) " *
                "not present in state (tracer_names = $(tracer_names(state)))"))
        end
        Int32(idx)
    end

    # Refresh rate caches for the current time, then materialize to scalars
    # for the kernel. `ConstantField.update_field!` ignores `t`; once
    # non-constant rate fields (e.g. `StepwiseField{FT, 0}` for time-varying
    # decay rates) are wired in, this is where they consume simulation time.
    # For `meteo === nothing` (test fixtures, direct TransportModel callers
    # without a met driver), fall back to `zero(FT)`; the stub at
    # `AbstractMetDriver.jl:77` already returns `0.0` so concrete drivers
    # wanting to drive time-varying fields must override.
    t = meteo === nothing ? zero(FT) : FT(current_time(meteo))
    rates = ntuple(N) do n
        r = op.decay_rates[n]
        update_field!(r, t)
        field_value(r, ())
    end

    raw = state.tracers_raw
    backend = get_backend(raw)
    kernel! = _exp_decay_kernel!(backend, 256)

    # Launch across the spatial axes; the trailing tracer axis is handled
    # by the kernel's inner loop over `indices`.
    spatial_shape = ntuple(i -> size(raw, i), ndims(raw) - 1)
    kernel!(raw, indices, rates, FT(dt), Int32(N);
            ndrange = spatial_shape)
    synchronize(backend)
    return state
end

"""
    apply!(state::CellState, meteo, grid, op::CompositeChemistry, dt; workspace=nothing)

Apply each sub-operator in order.
"""
function apply!(state::CellState, meteo, grid,
                op::CompositeChemistry, dt;
                workspace = nothing)
    for sub in op.schemes
        apply!(state, meteo, grid, sub, dt; workspace = workspace)
    end
    return state
end

# =========================================================================
# chemistry_block! — step-level block composer (plan 15 Decision 7)
# =========================================================================

include("chemistry_block.jl")

end # module Chemistry
