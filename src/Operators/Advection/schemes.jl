# ---------------------------------------------------------------------------
# Advection scheme type hierarchy
#
# Organized by reconstruction family (constant, linear, quadratic) as
# abstract supertypes, with concrete schemes as leaf types.  Julia's
# multiple dispatch selects the right @inline face-flux function at
# compile time — the GPU kernel sees a monomorphic code path.
#
# Dispatch works at two levels:
#
#   f(::AbstractConstantScheme)  — shared fallback for all constant schemes
#   f(::UpwindScheme)            — specialized to first-order upwind
#
# Limiters are explicit policy objects carried as type parameters on
# schemes that need them (linear, quadratic).  This avoids runtime
# `if limiter ...` branches inside GPU kernels.
#
# References
# ----------
# - Godunov (1959): piecewise-constant upwind (first-order donor cell)
# - van Leer (1977): MUSCL — piecewise-linear with slope limiters
# - Russell & Lerner (1981): slopes advection for atmospheric tracers
#   (TM5 advectx__slopes / advecty__slopes)
# - Colella & Woodward (1984): piecewise parabolic method (PPM)
# - Putman & Lin (2007): PPM on cubed-sphere grids (FV3 / GCHP)
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize

# ---- Reconstruction family supertypes ------------------------------------

"""
    AbstractAdvectionScheme

Root abstract type for all advection operators in the mass-flux transport core.

Every concrete scheme belongs to one of three reconstruction families:

    AbstractAdvectionScheme
    ├── AbstractConstantScheme    (order 0: donor-cell / upwind)
    ├── AbstractLinearScheme      (order 1: van Leer slopes / MUSCL)
    └── AbstractQuadraticScheme   (order 2: PPM / Prather moments)

This hierarchy enables orthogonal dispatch:
- **Reconstruction order** selects the face-flux `@inline` function
- **Limiter type** (carried as a type parameter) selects the slope/moment limiter
- **Grid topology** is handled by the kernel shell (structured vs face-indexed)
- **CS execution style** is handled separately (`strang_split_cs!` sweep shell vs
  Lin-Rood / FV3 horizontal update)
- **Backend** (CPU/GPU) is handled by KernelAbstractions.jl

# Implementing a new scheme

1. Subtype one of the three families
2. Implement `_xface_tracer_flux`, `_yface_tracer_flux`, `_zface_tracer_flux`
   (see `reconstruction.jl`)
3. The universal kernel shells in `structured_kernels.jl` will automatically
   dispatch to your face-flux functions at compile time

# Example

```julia
scheme = SlopesScheme(MonotoneLimiter())
strang_split!(state, fluxes, grid, scheme; workspace=ws)
```
"""
abstract type AbstractAdvectionScheme end

"""
    AbstractConstantScheme <: AbstractAdvectionScheme

Piecewise-constant (order 0) reconstruction family.

The face value equals the donor cell mean — the cell upwind of the mass flux.
This is the simplest conservative finite-volume scheme and the reference
implementation for the generic kernel shells.

Concrete subtypes: [`UpwindScheme`](@ref)
"""
abstract type AbstractConstantScheme <: AbstractAdvectionScheme end

"""
    AbstractLinearScheme <: AbstractAdvectionScheme

Piecewise-linear (order 1) reconstruction family (van Leer 1977, MUSCL).

The subcell profile in each cell is ``q(x) = \\bar{q} + s_x (x - x_c)``
where ``s_x`` is a limited slope.  The face flux is the Courant-fraction
weighted integral of this profile over the swept volume (see
`_slopes_face_flux` in `reconstruction.jl`).

Concrete subtypes: [`SlopesScheme`](@ref)
"""
abstract type AbstractLinearScheme <: AbstractAdvectionScheme end

"""
    AbstractQuadraticScheme <: AbstractAdvectionScheme

Piecewise-quadratic (order 2) reconstruction family.

Includes PPM (Colella & Woodward 1984; Putman & Lin 2007) and Prather
second-moment schemes.  The subcell profile is a parabola constrained
by the cell mean and (limited) edge values.

Concrete subtypes: [`PPMScheme`](@ref) (stub — kernels not yet implemented)
"""
abstract type AbstractQuadraticScheme <: AbstractAdvectionScheme end

# ---- Limiters ------------------------------------------------------------

"""
    AbstractLimiter

Policy object controlling slope and moment limiting in linear and quadratic
advection schemes.

Limiters are carried as type parameters on scheme structs (e.g.,
`SlopesScheme{MonotoneLimiter}`), enabling compile-time specialization
with zero runtime branches on GPU.

Available limiters:
- [`NoLimiter`](@ref): unlimited centered slopes (second-order, may oscillate)
- [`MonotoneLimiter`](@ref): van Leer minmod (monotone, TVD)
- [`PositivityLimiter`](@ref): ensures non-negative face values

See `limiters.jl` for the `@inline` implementations.
"""
abstract type AbstractLimiter end

"""
    NoLimiter <: AbstractLimiter

No limiting applied.  The slope is the full centered difference
``s = (c_{i+1} - c_{i-1}) / 2``.  Second-order accurate but may
produce new extrema (oscillations) near sharp gradients.
"""
struct NoLimiter        <: AbstractLimiter end

"""
    MonotoneLimiter <: AbstractLimiter

Van Leer minmod limiter (van Leer 1977; Sweby 1984).

Limits the centered slope against the two one-sided differences
scaled by 2, using the three-argument minmod function:

```math
s = \\text{minmod}\\bigl(\\tfrac{c_{i+1} - c_{i-1}}{2},\\;
    2(c_{i+1} - c_i),\\; 2(c_i - c_{i-1})\\bigr)
```

This is TVD (total variation diminishing) and preserves monotonicity.
The TM5 `advectx__slopes` / `advecty__slopes` routines use this limiter.
"""
struct MonotoneLimiter  <: AbstractLimiter end

"""
    PositivityLimiter <: AbstractLimiter

Limits the slope to keep the reconstructed face values non-negative:
``s = \\text{minmod}(s, c_i)``, ensuring ``c_i \\pm s/2 \\geq 0``.

Weaker than `MonotoneLimiter` but sufficient for species that must
remain positive (e.g., tracer mixing ratios).
"""
struct PositivityLimiter <: AbstractLimiter end

# ---- Concrete schemes ----------------------------------------------------

"""
    UpwindScheme <: AbstractConstantScheme

First-order donor-cell (Godunov) upwind scheme.

The tracer flux through a face is simply the mass flux times the mixing
ratio of the upstream cell:

```math
F_q = \\begin{cases}
  F \\cdot c_L & \\text{if } F \\geq 0 \\\\
  F \\cdot c_R & \\text{if } F < 0
\\end{cases}
```

where ``F`` is the mass flux [kg/s] and ``c = r_m / m`` is the mixing ratio.

Properties: conservative, monotone, first-order accurate.  Strongly diffusive
but useful as a reference and for positivity-critical applications.

# Example
```julia
scheme = UpwindScheme()
```
"""
struct UpwindScheme <: AbstractConstantScheme end

"""
    NoAdvection <: AbstractAdvectionScheme

Identity scheme — `apply!` is a no-op. Default for runs that need to
isolate other operators (convection, chemistry, diagnostics) from the
advective transport block, e.g. a "convection-alone" timing
experiment or a synthetic-state regression.

When `NoAdvection` is selected:
- The advection sweeps are skipped on all three topologies
  (`LatLonMesh`, `ReducedGaussianMesh`, `CubedSphereMesh`).
- Diffusion is *allowed*: with no advection to wrap, the V(dt)
  diffusion step is applied directly through the mass-flux VMR
  kernels, which are mass-conserving on their own and need no
  Strang palindrome. This is the natural "diffusion-only"
  experimental setup.
- Surface emissions are *rejected* with an actionable
  `ArgumentError`. They are integrated as Strang half-steps
  wrapping the advection block, so running them without advection
  silently drops 2nd-order accuracy. To run a convection-only
  configuration omit `[tracers.*.surface_flux]` blocks.
- The CS workspace is auto-allocated by the `TransportModel`
  constructor when diffusion is active (and is `nothing` when
  diffusion is also off).

# Example
```julia
scheme = NoAdvection()
```
"""
struct NoAdvection <: AbstractAdvectionScheme end

"""
    SlopesScheme{L <: AbstractLimiter} <: AbstractLinearScheme

Van Leer / Russell–Lerner slopes advection (Russell & Lerner 1981).

Reconstructs a piecewise-linear subcell profile in each cell using a
limited slope, then integrates the Courant-fraction swept volume to
compute the face flux.  Second-order accurate with `MonotoneLimiter`.

This is the method used by TM5 (`advectx__slopes`, `advecty__slopes`)
for horizontal transport of atmospheric tracers.

The face tracer flux for positive mass flux ``F > 0`` (left donor):

```math
F_q = \\alpha \\bigl(r_{m,L} + (1 - \\alpha)\\, s_{x,L}\\bigr)
```

where ``\\alpha = F / m_L`` is the Courant fraction, ``r_{m,L}`` is the
donor cell tracer mass, and ``s_{x,L}`` is the limited first moment
``s_x = m \\cdot \\text{slope}(c_{i-1}, c_i, c_{i+1})``.

See `_slopes_face_flux` in `reconstruction.jl` for the full
derivation and `_limited_slope` in `limiters.jl` for the
limiter implementations.

# Fields
- `limiter::L` — slope/moment limiting policy (default: `MonotoneLimiter()`)

# Examples
```julia
SlopesScheme()                       # monotone-limited (default, matches TM5)
SlopesScheme(NoLimiter())            # unlimited (2nd order, may oscillate)
SlopesScheme(PositivityLimiter())    # positivity-preserving
```
"""
struct SlopesScheme{L <: AbstractLimiter} <: AbstractLinearScheme
    limiter::L
end
SlopesScheme() = SlopesScheme(MonotoneLimiter())

"""
    PPMScheme{L <: AbstractLimiter} <: AbstractQuadraticScheme

Piecewise Parabolic Method (Colella & Woodward 1984; Putman & Lin 2007).

Reconstructs a parabolic subcell profile constrained by the cell mean
and limited edge values.  Third-order accurate in smooth regions with
appropriate limiting.

**Status**: structured-grid face-flux kernels are implemented and covered by
kernel tests. `PPMScheme` is not yet part of the official real-data
reference path, and face-connected support is still unimplemented.

# Fields
- `limiter::L` — parabolic profile limiting policy (default: `MonotoneLimiter()`)

# Example
```julia
PPMScheme()                          # monotone-limited PPM
PPMScheme(NoLimiter())               # unlimited (high-order, may oscillate)
```
"""
struct PPMScheme{L <: AbstractLimiter} <: AbstractQuadraticScheme
    limiter::L
end
PPMScheme() = PPMScheme(MonotoneLimiter())

"""
    LinRoodPPMScheme{ORD} <: AbstractAdvectionScheme

Cubed-sphere Lin-Rood / FV3-style cross-term PPM advection with compile-time
PPM order `ORD`.

This is distinct from [`PPMScheme`](@ref): `PPMScheme` participates in the
standard Strang split implemented by `strang_split_cs!`, while
`LinRoodPPMScheme` selects the FV3-style horizontal Lin-Rood update
(`fv_tp_2d_cs!`) paired with the existing vertical upwind sweep.

Supported orders currently match the implemented PPM edge-value families in
`ppm_subgrid_distributions.jl`:

- `ORD = 5` — Huynh-constrained PPM
- `ORD = 7` — order-5 interior with special cubed-sphere face treatment

# Fields
- `fillz :: Bool` (default `true`) — run the GCHP `fillz` positivity fixer
  after each sweep of the palindrome.

  `fillz = true` (GCHP-faithful): transient advection undershoots (negative
  mixing ratios just downwind of sharp gradients) are repaired by borrowing
  mass between column levels. **Pro:** strictly non-negative output fields;
  bit-comparable to GCHP's transport at sharp gradients. **Con:** the
  `rm → q → fillz → q → rm` round-trip is NOT Float32-conservative — each
  repair injects a little net mass. Measured: a sharp IC=0 tracer
  (co2_fossil) gains +5% of emission at t=3 h, saturating to ~0.3% at month
  scale; the injection accounts for the surplus to 3 decimals
  (fillz/surplus = 1.000, see `ATMOSTR_FILLZ_MASS_DIAG`).

  `fillz = false`: undershoots are left in place as small signed transients.
  **Pro:** flux-form advection then conserves tracer mass EXACTLY
  (telescoping) — the right choice for conservation-critical budget and
  inversion work; also removes the one non-differentiable operation from the
  forward (better-behaved 4D-Var adjoint). Every production operator
  (LinRood limiter, vertical donor-cell, implicit diffusion, TM5/CMFMC
  convection, exponential decay) is sign-safe, so negatives are benign in
  this offline model. **Con:** output VMR fields can dip slightly negative
  near sharp sources (cosmetic, bounded by the undershoot scale); deviates
  from GCHP's exact transport behavior; NOT safe if a nonlinear/positivity-
  requiring chemistry operator is ever added.

  Set from TOML via `[advection] fillz = false` (only valid with
  `scheme = "linrood"`). See docs/reference/ADVECTION_SCHEMES.md.

# Examples
```julia
LinRoodPPMScheme()                  # default ORD=5, fillz=true
LinRoodPPMScheme(7)                 # ORD=7 cubed-sphere boundary treatment
LinRoodPPMScheme(7; fillz = false)  # exactly mass-conserving, signed transients
```
"""
struct LinRoodPPMScheme{ORD} <: AbstractAdvectionScheme
    fillz :: Bool
end

# Back-compat: the fieldless-era forms `LinRoodPPMScheme{7}()` /
# `LinRoodPPMScheme()` keep working and mean fillz=true (GCHP-faithful).
LinRoodPPMScheme{ORD}() where {ORD} = LinRoodPPMScheme{ORD}(true)

function LinRoodPPMScheme(order::Integer = 5; fillz::Bool = true)
    order in (5, 7) || throw(ArgumentError(
        "LinRoodPPMScheme supports ORD=5 or ORD=7, got ORD=$(order)"))
    return LinRoodPPMScheme{Int(order)}(fillz)
end

# ---- Cubed-sphere execution style + capability traits -------------------

abstract type AbstractCSAdvectionStyle end

struct CSSplitSweepStyle <: AbstractCSAdvectionStyle end
struct CSLinRoodStyle    <: AbstractCSAdvectionStyle end

@inline cs_advection_style(::AbstractAdvectionScheme) = CSSplitSweepStyle()
@inline cs_advection_style(::LinRoodPPMScheme)        = CSLinRoodStyle()
# `NoAdvection` doesn't run any CS execution path; the explicit
# specialization returns the split-sweep style so callers that
# branch on the trait don't choke on an unhandled scheme.
@inline cs_advection_style(::NoAdvection)             = CSSplitSweepStyle()

"""
    required_halo_width(scheme) -> Int

Return the minimum cubed-sphere halo width needed by `scheme`'s horizontal
stencil. This is a capability query, not a reconstruction-order query: several
schemes can share the same polynomial family while using different CS execution
paths.
"""
@inline required_halo_width(::AbstractConstantScheme)  = 1
@inline required_halo_width(::AbstractLinearScheme)    = 2
@inline required_halo_width(::AbstractQuadraticScheme) = 3
@inline required_halo_width(::LinRoodPPMScheme)        = 3
# `NoAdvection` doesn't read any horizontal stencil, so any halo
# width works. Reporting 0 keeps the workspace sizer from
# allocating CS halo scratch beyond what convection/diffusion need.
@inline required_halo_width(::NoAdvection)             = 0

# ---- Reconstruction order query -----------------------------------------

"""
    reconstruction_order(scheme) → Int

Return the polynomial order of the subcell reconstruction:
- 0 for constant (upwind)
- 1 for linear (slopes)
- 2 for quadratic (PPM)

Useful for diagnostics and for selecting stencil widths in
multi-tracer kernel fusion.
"""
@inline reconstruction_order(::AbstractConstantScheme)  = 0
@inline reconstruction_order(::AbstractLinearScheme)    = 1
@inline reconstruction_order(::AbstractQuadraticScheme) = 2
@inline reconstruction_order(::LinRoodPPMScheme)        = 2
# `NoAdvection` has no reconstruction; report -1 as a clear
# "not applicable" sentinel.
@inline reconstruction_order(::NoAdvection)             = -1

export AbstractAdvectionScheme
export AbstractConstantScheme, AbstractLinearScheme, AbstractQuadraticScheme
export AbstractLimiter, NoLimiter, MonotoneLimiter, PositivityLimiter
export UpwindScheme, SlopesScheme, PPMScheme, LinRoodPPMScheme, NoAdvection
export reconstruction_order, required_halo_width
