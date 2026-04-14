# ---------------------------------------------------------------------------
# Slope and moment limiters — @inline, branchless, GPU-safe
#
# These functions enforce monotonicity or positivity constraints on the
# piecewise-linear (or higher) subcell reconstruction.  They are the
# key ingredient that makes second-order advection stable and non-oscillatory.
#
# Why branchless?
# ---------------
# GPU warps execute in lockstep (SIMT).  An `if/else` branch causes warp
# divergence — threads that take different paths are serialized.  Using
# `ifelse(cond, a, b)` ensures ALL threads compute BOTH values and select
# one, keeping the warp fully utilized.  The compiler eliminates dead
# computation when the condition is statically known (e.g., scheme type).
#
# References
# ----------
# - van Leer (1977), "Towards the ultimate conservative difference
#   scheme. IV", J. Comp. Phys., 23:276–299.
#   → Original MUSCL slope limiter framework.
#
# - Sweby (1984), "High resolution schemes using flux limiters for
#   hyperbolic conservation laws", SIAM J. Numer. Anal., 21:995–1011.
#   → TVD (total variation diminishing) conditions for flux limiters.
#
# - Zalesak (1979), "Fully multidimensional flux-corrected transport
#   algorithms for fluids", J. Comp. Phys., 31:335–362.
#   → Flux-corrected transport and positivity preservation.
#
# - Russell & Lerner (1981), "A new finite-differencing scheme for the
#   tracer transport equation", J. Appl. Meteor., 20:1483–1498.
#   → Slopes advection for atmospheric tracers (TM5 heritage).
# ---------------------------------------------------------------------------

# ---- Minmod primitives ---------------------------------------------------

"""
    _minmod_slope(a, b) → typeof(a)

Two-argument minmod: returns the value with smaller magnitude if both
have the same sign, zero otherwise.

```math
\\text{minmod}(a, b) = \\begin{cases}
  \\text{sign}(a) \\min(|a|, |b|) & \\text{if } \\text{sign}(a) = \\text{sign}(b) \\\\
  0 & \\text{otherwise}
\\end{cases}
```

Branchless implementation using `ifelse` for GPU safety.
"""
@inline function _minmod_slope(a, b)
    return ifelse(a * b <= zero(a), zero(a),
                  ifelse(abs(a) < abs(b), a, b))
end

"""
    _minmod3(a, b, c) → typeof(a)

Three-argument minmod: returns the value with smallest magnitude if all
three share the same sign, zero otherwise.

```math
\\text{minmod}(a, b, c) = \\begin{cases}
  \\min(a, b, c)  & \\text{if } a, b, c > 0 \\\\
  \\max(a, b, c)  & \\text{if } a, b, c < 0 \\\\
  0              & \\text{otherwise}
\\end{cases}
```

This is the core primitive of the van Leer / Sweby TVD framework.
The three arguments are typically:

1. The centered slope: ``s_c = (c_{i+1} - c_{i-1}) / 2``
2. Twice the right one-sided slope: ``2 (c_{i+1} - c_i)``
3. Twice the left one-sided slope: ``2 (c_i - c_{i-1})``

The factor of 2 comes from the Sweby TVD region boundary — it is the
maximum slope that still yields monotone face values when the Courant
number reaches 1 (Sweby 1984, Theorem 4.1).
"""
@inline function _minmod3(a, b, c)
    all_pos = (a > zero(a)) & (b > zero(b)) & (c > zero(c))
    all_neg = (a < zero(a)) & (b < zero(b)) & (c < zero(c))
    return ifelse(all_pos, min(a, b, c),
           ifelse(all_neg, max(a, b, c), zero(a)))
end

# ---- Slope limiters (operating on mixing ratio) --------------------------
#
# Given three consecutive cell-mean mixing ratios (c_left, c_center, c_right),
# compute a limited slope for the center cell.  The slope is the coefficient
# in the piecewise-linear reconstruction:
#
#   q(x) = c_center + slope · (x - x_center) / Δx
#
# In mass-flux advection, this slope is multiplied by the cell mass to form
# the "first moment" sx = m · slope, which is then used in the Courant-fraction
# flux formula (see reconstruction.jl).

"""
    _limited_slope(c_left, c_center, c_right, ::MonotoneLimiter)

Van Leer / Sweby monotone slope limiter (van Leer 1977; Sweby 1984).

Computes the centered slope and limits it against twice the one-sided
slopes using the three-argument minmod:

```math
s = \\text{minmod}\\!\\left(
    \\frac{c_{i+1} - c_{i-1}}{2},\\;
    2(c_{i+1} - c_i),\\;
    2(c_i - c_{i-1})
\\right)
```

This ensures the reconstruction is TVD (total variation diminishing):
the reconstructed face values ``c_i \\pm s/2`` lie between the
neighboring cell means, preventing new extrema.

This is the limiter used by TM5's `advectx__slopes` and
`advecty__slopes` routines (Russell & Lerner 1981).
"""
@inline function _limited_slope(c_left, c_center, c_right, ::MonotoneLimiter)
    sc = (c_right - c_left) / 2
    return _minmod3(sc,
                    2 * (c_right  - c_center),
                    2 * (c_center - c_left))
end

"""
    _limited_slope(c_left, c_center, c_right, ::NoLimiter)

Unlimited centered slope: ``s = (c_{i+1} - c_{i-1}) / 2``.

Second-order accurate but may introduce new extrema (Gibbs-type
oscillations) near sharp gradients.  Useful for smooth test problems
and as a reference for limiter comparison.
"""
@inline function _limited_slope(c_left, c_center, c_right, ::NoLimiter)
    return (c_right - c_left) / 2
end

"""
    _limited_slope(c_left, c_center, c_right, ::PositivityLimiter)

Positivity-preserving slope limiter.

Limits the centered slope against the center value to ensure
``c_i - |s|/2 \\geq 0``:

```math
s = \\text{minmod}\\!\\left(\\frac{c_{i+1} - c_{i-1}}{2},\\; c_i\\right)
```

Weaker than `MonotoneLimiter` (allows new maxima) but guarantees
non-negative reconstructed values — important for species mixing
ratios that must remain ≥ 0.
"""
@inline function _limited_slope(c_left, c_center, c_right, ::PositivityLimiter)
    sc = (c_right - c_left) / 2
    return _minmod_slope(sc, c_center)
end

# ---- Moment limiters (operating on tracer mass) --------------------------
#
# After computing the slope in mixing-ratio space, we convert to the "first
# moment" sx = m · slope.  This moment must also be limited to prevent the
# Courant-fraction flux formula from producing negative tracer mass in the
# donor cell.
#
# The moment limiter clamps:  -rm ≤ sx ≤ rm
#
# This ensures that even when α → 1 (full Courant), the outgoing tracer
# flux F_q = α(rm + (1-α)·sx) cannot exceed rm (the cell's tracer mass).

"""
    _limited_moment(sx, rm_cell, ::MonotoneLimiter)

Clamp the first moment ``s_x`` to the interval ``[-r_m, r_m]``:

```math
s_x^{\\text{lim}} = \\max\\bigl(\\min(s_x,\\; r_m),\\; -r_m\\bigr)
```

where ``r_m`` is the cell's tracer mass.  This guarantees that the
reconstructed profile integrated over any Courant fraction ``\\alpha \\in [0,1]``
yields a non-negative, bounded tracer flux.

This is the TM5 moment limiter applied in `advectx__slopes` after the
slope computation (Russell & Lerner 1981, eq. 12).
"""
@inline function _limited_moment(sx, rm_cell, ::MonotoneLimiter)
    return max(min(sx, rm_cell), -rm_cell)
end

"""
    _limited_moment(sx, rm_cell, ::NoLimiter)

No moment limiting — returns ``s_x`` unchanged.
"""
@inline _limited_moment(sx, rm_cell, ::NoLimiter) = sx

"""
    _limited_moment(sx, rm_cell, ::PositivityLimiter)

Same clamp as `MonotoneLimiter`: ``s_x \\in [-r_m, r_m]``.
Ensures non-negative face values.
"""
@inline _limited_moment(sx, rm_cell, ::PositivityLimiter) = max(min(sx, rm_cell), -rm_cell)
