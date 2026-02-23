# # Advection Theory: From Concentration to Mass-Flux Formulation
#
# This document explains the mathematical framework behind the advection schemes
# in AtmosTransport.jl, their relationship to the TM5 reference model, and
# the critical design decision to adopt TM5's mass-flux formulation for
# mass-conserving operator-split transport.
#
# ## Background: Atmospheric Tracer Transport
#
# Atmospheric transport of trace gases (CO₂, CH₄, etc.) is governed by the
# continuity equation:
#
# ```math
# \frac{\partial (\rho q)}{\partial t} + \nabla \cdot (\rho q \mathbf{v}) = S
# ```
#
# where ``\rho`` is air density, ``q`` is the tracer mixing ratio, ``\mathbf{v}``
# is the 3D wind vector, and ``S`` represents sources and sinks.
#
# In pressure coordinates (hybrid sigma-pressure), this becomes:
#
# ```math
# \frac{\partial (m \cdot c)}{\partial t} + \nabla_p \cdot (m \cdot c \cdot \mathbf{v}) = S
# ```
#
# where ``m = \Delta p \cdot A / g`` is the air mass per grid cell and ``c`` is the
# tracer concentration (e.g., ppmv).

# ## Two Formulations: Concentration vs. Mass-Flux
#
# ### Concentration-based (Eulerian)
#
# The simplest approach advects concentration ``c`` directly:
#
# ```math
# c_{new} = c_{old} - \frac{\Delta t}{\Delta x} \left( F_R - F_L \right)
# ```
#
# where ``F = u \cdot (c + (1-\alpha) s / 2)`` is the Russell-Lerner flux with
# slope ``s`` and Courant number ``\alpha = u \Delta t / \Delta x``.
#
# **Problem**: In operator splitting (X→Y→Z→Z→Y→X), each 1D step creates mass
# divergence that is not compensated. After x-advection, the effective air mass
# in each cell has changed, but ``c`` doesn't know about it. This causes extreme
# values even with perfect global mass conservation.
#
# ### Mass-flux formulation (TM5's approach)
#
# TM5 advects **tracer mass** ``r_m = m \cdot c`` alongside **air mass** ``m``:
#
# ```math
# \begin{aligned}
# \alpha &= \frac{a_m}{m_{donor}} \quad \text{(mass-based Courant number)} \\
# f &= \alpha \left( r_m + (1-\alpha) \cdot s_{rm} \right) \quad \text{(tracer mass flux)} \\
# r_{m,new} &= r_m + f_{in} - f_{out} \\
# m_{new} &= m + a_{m,in} - a_{m,out} \\
# c_{new} &= r_{m,new} / m_{new}
# \end{aligned}
# ```
#
# The division by ``m_{new}`` naturally handles the mass divergence created by
# operator splitting. This is the key advantage: **the denominator automatically
# accounts for air mass changes in each directional step**.

# ## The Four Critical Differences
#
# Through careful comparison with TM5's Fortran source code (`advectx.F90`,
# `advecty.F90`, `advectz.F90`, `advect_tools.F90`), we identified four
# architectural differences between a naive concentration-based scheme and TM5:
#
# ### 1. Prognostic Variable
#
# | | TM5 | Concentration-based |
# |:---|:---|:---|
# | Advected quantity | Tracer mass ``r_m`` (kg) | Concentration ``c`` (ppmv) |
# | Courant number | ``\alpha = a_m / m_{donor}`` (mass-based) | ``\alpha = u \Delta t / \Delta x`` (velocity-based) |
# | Air mass | Co-evolved ``m_{new} = m + a_{m,in} - a_{m,out}`` | Not tracked |
# | Recovery | ``c = r_m / m_{new}`` | Direct output |
#
# ### 2. Continuous Mass Tracking Through Operator Splitting
#
# TM5 passes the updated ``m`` from x-advection into y-advection, then into
# z-advection, and so on through the full Strang split (X-Y-Z-Z-Y-X). ``m`` is
# **never reset** during the split — it is only re-initialized from surface
# pressure at the start of a new meteorological window.
#
# A post-hoc correction approach that resets ``\Delta p`` at each Strang split
# breaks this continuity and introduces mass drift.
#
# ### 3. Mass Fluxes vs. Velocities
#
# TM5 works with mass fluxes ``a_m = \Delta t / 2 \cdot p_u`` (kg per half-timestep),
# where ``p_u`` is derived from ECMWF spectral mass fluxes. The vertical mass
# flux ``c_m`` is computed from horizontal convergence via the continuity equation
# (TM5's `dynam0` subroutine), ensuring column mass conservation.
#
# Our implementation computes mass fluxes from gridpoint winds:
# ``a_m = \Delta t / 2 \cdot u \cdot \Delta p_{face} \cdot \Delta y / g``,
# with ``c_m`` derived from horizontal convergence using B-coefficient weighting.
#
# ### 4. Slope Computation
#
# TM5 maintains slopes (``r_{xm}``, ``r_{ym}``, ``r_{zm}``) as **prognostic
# variables** that evolve via their own transport equations. Our implementation
# computes slopes diagnostically from the **concentration** field each step,
# then scales by ``m``:
#
# ```math
# s_{rm} = m \cdot \frac{c_{i+1} - c_{i-1}}{2}
# ```
#
# Computing slopes from concentration rather than from ``r_m`` directly is
# essential: it ensures that a spatially uniform concentration field is exactly
# preserved despite spatially varying air mass.

# ## The Continuity-Consistent Vertical Mass Flux
#
# A key feature of the mass-flux formulation is that the vertical mass flux
# ``c_m`` is not taken from the meteorological data directly, but derived from
# the continuity equation. This follows TM5's `dynam0` subroutine.
#
# Given horizontal mass fluxes ``a_m`` and ``b_m``, the horizontal convergence
# into each cell is:
#
# ```math
# \text{conv}_{i,j,k} = a_{m,in} - a_{m,out} + b_{m,in} - b_{m,out}
# ```
#
# The column-integrated convergence is ``\text{pit} = \sum_k \text{conv}_k``,
# which represents the surface pressure tendency.
#
# In hybrid sigma-pressure coordinates, the fraction of this tendency felt by
# each layer is:
#
# ```math
# b_t(k) = \frac{B(k+1) - B(k)}{B(N_z+1) - B(1)}
# ```
#
# The vertical mass flux at interface ``k+1`` (between layers ``k`` and ``k+1``)
# is accumulated from the top of the atmosphere:
#
# ```math
# c_m(k+1) = c_m(k) + \text{conv}(k) - b_t(k) \cdot \text{pit}
# ```
#
# with ``c_m(1) = 0`` (no flux at the top) and ``c_m(N_z+1) \approx 0`` (no flux
# at the surface, guaranteed by construction since ``\sum_k b_t(k) = 1``).
#
# This ensures that the 3D mass flux field is **exactly divergence-free** in the
# column integral, which is critical for mass conservation.

# ## CFL Subcycling
#
# When the mass-based Courant number ``|\alpha| = |a_m / m_{donor}| > 1``, the
# advection becomes unstable. TM5 handles this by dividing the mass flux by
# ``n_{loop}`` and running the advection ``n_{loop}`` times (see
# `advectx_get_nloop` in TM5).
#
# Our implementation does the same: `advect_x_massflux_subcycled!` checks the
# maximum ``|\alpha|`` across all faces and subdivides accordingly.

# ## Validation Results
#
# The mass-flux implementation achieves:
#
# | Test | Result |
# |:---|:---|
# | X/Y/Z mass conservation | Machine precision (< 1e-15 relative) |
# | Uniform field preservation | < 4e-13 deviation (was ~1.0 with concentration-based) |
# | 10-step Strang split mass drift | 7.3e-5% (was 0.91% with post-hoc correction) |
# | Positivity with limiter | Zero negative cells |
# | CFL subcycling | Correct automatic subdivision |
# | Full test suite | 209/209 tests passing |
#
# These results confirm that the mass-flux formulation resolves the fundamental
# architectural differences that prevented agreement with TM5.

# ## Relationship to Other Models
#
# | Model | Formulation | Mass fluxes | Adjoint |
# |:---|:---|:---|:---|
# | TM5 | Mass-flux (``r_m``, ``m``) | From ECMWF spectral harmonics | Hand-coded discrete |
# | GEOS-Chem | Concentration-based | From GEOS-FP native fluxes | AD-based |
# | LMDZ | Mass-flux | From GCM dynamics | None (ensemble) |
# | **AtmosTransport.jl** | **Mass-flux (TM5-faithful)** | **From winds or native** | **Hand-coded discrete** |
#
# Our model combines TM5's proven mass-flux formulation with Julia's performance
# and flexibility. It supports multiple meteorological drivers (ERA5, GEOS-FP,
# MERRA-2) on multiple grids (lat-lon, cubed-sphere) with a single codebase.

# ## Key References
#
# - Russell, G.L. and Lerner, J.A. (1981). A new finite-differencing scheme for
#   the tracer transport equation. *J. Appl. Meteorol.*, 20, 1483-1498.
# - Bregman, A., et al. (2003). Comparing the effect of different mass flux
#   methods on the stratospheric age-of-air. *ACP*, 3, 447-457.
# - Krol, M., et al. (2005). The two-way nested global chemistry-transport
#   zoom model TM5. *ACP*, 5, 417-432.
# - Martin, S.T., et al. (2022). GEOS-Chem High Performance (GCHP v13.3.1).
#   *GMD*, 15, 8731-8748.
