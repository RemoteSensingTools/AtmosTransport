# TimeVaryingField — Abstract Model

**Status:** v1. Authoritative interface for rate-like and field-like
inputs to physics operators (chemistry decay rates, Kz diffusion fields,
future emission strengths, photolysis rates, ...). Introduced by
plan 16a; exercised first by the `ExponentialDecay` rate retrofit
(plan 16a Commit 2); extended by plan 16b with 3D concrete types for
Kz sourcing.

---

## 1. Purpose

Physics operators need scalar or array-valued inputs that may be
constant, stepwise-constant in time, interpolated, or derived at
runtime from other state. The kernel implementing the operator should
not branch on input source. `AbstractTimeVaryingField{FT, N}` is the
common interface: an object that can be:

1. **Updated** for the current simulation time (CPU-side, possibly
   expensive — e.g. interpolation from file windows, or recomputation
   from meteorology).
2. **Read** at a spatial index inside a kernel (kernel-safe,
   allocation-free, type-stable).

Different concrete types back different sources (constant, stepwise
from a preprocessed binary, derived from surface fields, ...) with
the same kernel code downstream.

## 2. Abstract type

```julia
abstract type AbstractTimeVaryingField{FT, N} end
```

Type parameters:

- `FT <: AbstractFloat` — element type of the field's values
- `N :: Int` — spatial rank of the field:
  - `N = 0` — scalar (e.g. a decay rate for a single tracer)
  - `N = 2` — surface field (e.g. `PBLH(lat, lon)`)
  - `N = 3` — volume field (e.g. `Kz(lat, lon, level)`)

Higher ranks are possible but no use case motivates them yet.

## 3. Interface

### 3.1 Required methods (every concrete type)

```julia
field_value(f::AbstractTimeVaryingField{FT, N}, idx::NTuple{N, Int}) -> FT
```

Return the field's current value at spatial index `idx`.

**Contract:**
- Must be **kernel-safe**: callable from inside a KernelAbstractions
  `@kernel`, allocation-free, type-stable, no dynamic dispatch.
- Must be **pure** with respect to `f`'s cached state — does not mutate.
- May read from a cache populated by the most recent `update_field!`
  call.
- For `N = 0`, `idx` is the empty tuple `()`.

```julia
update_field!(f::AbstractTimeVaryingField, t::Real) -> f
```

Refresh any internal caches so that subsequent `field_value` calls
return the field's value at simulation time `t`.

**Contract:**
- Runs on CPU OR device (concrete-type-dependent); NOT required to be
  kernel-safe itself.
- May be expensive (interpolation, meteorology-dependent recomputation).
- Called **once per `apply!`** by the operator, before any kernel launch
  that reads the field.
- For time-independent fields (e.g. `ConstantField`), this is a no-op.
- Returns `f` for chaining.

### 3.2 Optional methods (future concrete types)

Reserved for later concrete types; not required by `ConstantField`:

```julia
time_bounds(f) -> (t_min::Float64, t_max::Float64)
```
Range of simulation times for which the field is valid. Querying
outside this range is an error. For time-independent fields, returns
`(-Inf, Inf)`.

```julia
integral_between(f, t1::Real, t2::Real, idx::NTuple{N, Int}) -> FT
```
Time integral of the field over `[t1, t2]` at spatial index `idx`.
Needed for flux-like fields where the operator applies the time-
integrated value (not the instantaneous one). Chemistry and diffusion
use the instantaneous interface (`field_value` after `update_field!`)
and do NOT need `integral_between`.

```julia
requires_subfluxing(f) -> Bool
```
For operators that may need finer time steps than the outer dt
(advection CFL-limited; future convection). Returns `true` if the
field's temporal resolution is finer than the operator's outer dt.
Chemistry and diffusion do not consult this.

## 4. Concrete types (plan 16a scope)

### 4.1 `ConstantField{FT, N}`

```julia
struct ConstantField{FT, N} <: AbstractTimeVaryingField{FT, N}
    value :: FT
end
```

A scalar value presented as a field of rank `N`. `field_value` ignores
its index and returns the stored scalar. `update_field!` is a no-op.

**Storage:** one scalar. No arrays. Backend-agnostic by construction
— a scalar is a scalar on CPU and GPU.

**Use cases (plan 16a):** chemistry decay rates (one number per
tracer). `ExponentialDecay` stores its rates as
`NTuple{N, ConstantField{FT, 0}}`.

**Future use cases (plan 16b):** idealized constant-Kz diffusion tests
via `ConstantField{FT, 3}`.

## 5. Concrete types (future scope — not implemented in plan 16a)

Listed here so the spec is complete; these are 16b+.

### 5.1 `StepwiseField{FT, N, A, T}` (plan 16b)

Stepwise-constant in time, with a sequence of `(t_i, snapshot_i)` pairs.
`update_field!(f, t)` picks the snapshot for the window containing `t`
and caches a pointer (or copies into a device-resident buffer).
`field_value` reads from the cached snapshot.

### 5.2 `PreComputedKzField` / `DerivedKzField` (plan 16b)

Concrete types for diffusion Kz sourcing. Thin wrapper over
`StepwiseField` and a derived recomputation from surface fields
respectively.

## 6. Kernel-safety contract (expanded)

`field_value(f, idx)` must satisfy all of:

1. No heap allocation.
2. No dynamic dispatch — the concrete type `typeof(f)` must be known
   statically at the call site.
3. Constant arithmetic intensity per call (no loops over unrelated
   data inside `field_value`).
4. Pure with respect to `f` — reads only; no mutation, no I/O.
5. Returns an `FT`. (Not a view, not a boxed value.)

Everything that requires side effects, allocation, or variable cost
lives in `update_field!` and runs on the host (or as an explicit
device kernel the operator launches before its main kernel).

## 7. Lifecycle in an operator

```julia
function apply!(state, meteo, grid, op::SomeOperator, dt; workspace)
    t = current_time(meteo)
    for f in fields_of(op)
        update_field!(f, t)              # CPU-side refresh
    end
    launch_kernel!(..., op_fields..., dt) # field_value called inside
    return state
end
```

The operator owns the call to `update_field!`. The kernel only reads
through `field_value`. No operator should call `value_at(f, t, idx)`
inside a kernel — that's not part of the required interface.

## 8. Test contract

Every `AbstractTimeVaryingField` concrete type must have tests for:

1. **Interface completeness** — `field_value` and `update_field!`
   exist and behave per §3.1.
2. **Kernel-safety** — `field_value` is callable from a KA kernel
   without error on both CPU and GPU backends.
3. **Type stability** — `@code_warntype` clean for `field_value`.
4. **Idempotent `update_field!`** — calling twice with the same `t`
   produces the same result as calling once.

## 9. Version history

- **v1** (plan 16a, 2026-04-18) — initial spec. `ConstantField{FT, N}`
  only concrete type. Interface minimum: `field_value`,
  `update_field!`. `integral_between` / `time_bounds` /
  `requires_subfluxing` reserved for future use.
