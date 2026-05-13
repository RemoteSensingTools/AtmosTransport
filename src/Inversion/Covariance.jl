# ---------------------------------------------------------------------------
# Plan 26 P0.B1 — surface-flux background-error covariance B for the CS
# 4D-Var path.
#
# Preconditioned-4D-Var convention:
#     x   = x_b + B^(1/2) * χ              (physical-space control)
#     B   = B^(1/2) * (B^(1/2))^T          (symmetric PSD)
#     J   = 0.5 ‖χ‖²  +  0.5 (Hx − y)^T R^(−1) (Hx − y)
#     ∇_χ = χ + (B^(1/2))^T H^T R^(−1) (Hx − y)
#
# Factorization used here:
#     B^(1/2) = D · L            (NOT symmetric in general)
#     B       = D · L² · D       (with L² ≡ C, the correlation)
#
# Where:
#   * D is the diagonal `(per-cell σ)` scaling.
#   * L is a symmetric square root of the correlation operator C:
#       - For `DiagonalCSCovariance`, L = I (no correlation), so
#         B = D² and B^(1/2) is self-adjoint.
#       - For `IsotropicGaussianCSCovariance`, C is panel-local periodic
#         Gaussian; L is its symmetric spectral square root, computed
#         once at construction as `sqrt(C̃)` of the FFT eigenvalues of
#         the wrapped Gaussian kernel.
#
# v1 scope — explicitly out:
#   * Cross-panel correlation. Each panel is smoothed independently
#     with periodic-on-panel boundary conditions. Edge wrap-around at
#     panel corners is a known v1 limitation (NOTES.md "Phase B —
#     cross-panel correlation"); a v2 Schur-complement or low-rank
#     eigen-correction lands later.
#   * Temporal correlation. v1 operates on a single CSSurfaceFluxControl
#     `value` shape (`NTuple{6, Matrix{FT}}`). Multi-window temporal
#     smoothing comes with B2.
#   * GPU storage. The FFTW path is CPU-only. GPU support requires a
#     CUFFT path (gated on KernelAbstractions backend); deferred.
# ---------------------------------------------------------------------------

import FFTW

"""
    AbstractCSSurfaceFluxCovariance{FT, A}

Supertype for CS surface-flux background-error covariance operators.
Concrete subtypes implement [`apply_B_half!`](@ref) and
[`apply_B_half_adjoint!`](@ref) on a single
[`CSSurfaceFluxControl`](@ref)-shaped tuple `NTuple{6, A}` of `(Nc, Nc)`
matrices.
"""
abstract type AbstractCSSurfaceFluxCovariance{FT <: AbstractFloat,
                                              A <: AbstractMatrix{FT}} end

# ---------------------------------------------------------------------------
# DiagonalCSCovariance — `B = D²`, no correlation.
# ---------------------------------------------------------------------------

"""
    DiagonalCSCovariance(sigma::NTuple{6, AbstractMatrix})

Pure-diagonal CS surface-flux covariance `B = diag(σ)²`. `sigma[p][i, j]`
is the background standard deviation at panel `p`, interior cell
`(i, j)`. All six panels must share the same `(Nc, Nc)` shape and every
entry must be finite and strictly positive.

This is the cleanest baseline; `B^(1/2)` is self-adjoint, so
[`apply_B_half_adjoint!`](@ref) reduces to [`apply_B_half!`](@ref).
"""
struct DiagonalCSCovariance{FT, A} <: AbstractCSSurfaceFluxCovariance{FT, A}
    sigma::NTuple{6, A}
    Nc::Int
end

function DiagonalCSCovariance(sigma::NTuple{6, A}) where {FT <: AbstractFloat,
                                                          A <: AbstractMatrix{FT}}
    sh = size(sigma[1])
    sh[1] == sh[2] || throw(DimensionMismatch(
        "DiagonalCSCovariance requires square panels, got $sh"))
    Nc = sh[1]
    @inbounds for p in 1:6
        size(sigma[p]) == (Nc, Nc) || throw(DimensionMismatch(
            "DiagonalCSCovariance sigma panel $p has shape " *
            "$(size(sigma[p])); expected $((Nc, Nc))"))
        _validate_positive_finite(sigma[p], "DiagonalCSCovariance sigma panel $p")
    end
    return DiagonalCSCovariance{FT, A}(sigma, Nc)
end

# ---------------------------------------------------------------------------
# IsotropicGaussianCSCovariance — `B = D · C · D` with panel-local
# periodic Gaussian correlation `C` of length `L_cells` (in interior
# cells). Built from the spectral square root of the wrapped Gaussian.
# ---------------------------------------------------------------------------

"""
    IsotropicGaussianCSCovariance(sigma::NTuple{6, AbstractMatrix},
                                  correlation_length_cells::Real)

Panel-local CS surface-flux covariance `B = D · C · D` with
isotropic separable Gaussian correlation of length
`correlation_length_cells` (in interior-cell units).

Each panel's `(Nc, Nc)` grid is treated as periodic and the
correlation matrix `C` is the circulant whose first row is the
wrapped Gaussian. `L = C^(1/2)` (the symmetric square root) is
applied via FFT: in spectral space `L` is multiplication by
`sqrt(C̃)` where `C̃` is the real, non-negative spectral eigenvalues
of `C`. The factorization is `B^(1/2) = D · L`, so
`(B^(1/2))^T = L^T · D = L · D` (since `D` and `L` are both
symmetric).

The wrapped-Gaussian normalization pins `C[i, i] = 1`, hence
`B[i, i] = σ_i²` — the diagonal of `B` recovers the per-cell
variance, independent of correlation length.

**v1 limitations** (noted in `docs/plans/26_TM5_STYLE_INVERSION/NOTES.md`
under "Phase B — cross-panel correlation"):

- Cross-panel correlation is dropped. Each panel is smoothed in
  isolation; the implicit wrap-around at panel boundaries leaves
  edge artefacts that v2 will address.
- The FFT path is CPU-only.
"""
struct IsotropicGaussianCSCovariance{FT, A} <: AbstractCSSurfaceFluxCovariance{FT, A}
    sigma::NTuple{6, A}
    correlation_length_cells::FT
    Nc::Int
    # Spectral square root of the panel-local 2D Gaussian correlation
    # eigenvalues. Real, non-negative, length `Nc × Nc` per panel.
    L_transfer_sqrt::Matrix{FT}
end

function IsotropicGaussianCSCovariance(sigma::NTuple{6, A},
                                       correlation_length_cells::Real
                                       ) where {FT <: AbstractFloat,
                                                A <: AbstractMatrix{FT}}
    isfinite(correlation_length_cells) || throw(ArgumentError(
        "IsotropicGaussianCSCovariance correlation_length_cells must be " *
        "finite, got $correlation_length_cells"))
    correlation_length_cells > 0 || throw(ArgumentError(
        "IsotropicGaussianCSCovariance correlation_length_cells must be " *
        "positive, got $correlation_length_cells"))
    sh = size(sigma[1])
    sh[1] == sh[2] || throw(DimensionMismatch(
        "IsotropicGaussianCSCovariance requires square panels, got $sh"))
    Nc = sh[1]
    @inbounds for p in 1:6
        size(sigma[p]) == (Nc, Nc) || throw(DimensionMismatch(
            "IsotropicGaussianCSCovariance sigma panel $p has shape " *
            "$(size(sigma[p])); expected $((Nc, Nc))"))
        _validate_positive_finite(sigma[p],
            "IsotropicGaussianCSCovariance sigma panel $p")
    end
    L_cells = FT(correlation_length_cells)
    transfer_sqrt = _gaussian_transfer_sqrt_2d(Nc, L_cells)
    return IsotropicGaussianCSCovariance{FT, A}(sigma, L_cells, Nc, transfer_sqrt)
end

# ---------------------------------------------------------------------------
# Spectral helpers — wrapped 1D periodic Gaussian + 2D separable
# spectral square root.
# ---------------------------------------------------------------------------

# Periodic 1D Gaussian kernel `k[n]` for `n ∈ 0..Nc-1`, summed over
# enough periodic images that the tails are below Float64 precision.
# Returns the unnormalized kernel; the caller normalizes diagonal-of-C
# to 1 by dividing by `k[1]`.
function _gaussian_periodic_kernel(Nc::Int, L::FT) where FT
    inv2L2 = inv(2 * L * L)
    # Number of wraps. For `L < Nc/2` two images each side already give
    # < exp(-Nc²/2L²) ≪ 1e-15. Safety floor of 3 in case L ≳ Nc.
    M = max(3, ceil(Int, 4 * L / Nc) + 1)
    kernel = zeros(FT, Nc)
    @inbounds for n in 0:(Nc - 1)
        s = zero(FT)
        for m in (-M):M
            d = n + m * Nc
            s += exp(-FT(d) * FT(d) * inv2L2)
        end
        kernel[n + 1] = s
    end
    return kernel
end

# Spectral square root of the panel-local 2D isotropic Gaussian
# correlation. Returns `sqrt(C̃[kx, ky])` where `C̃[kx, ky] =
# C̃_1d[kx] · C̃_1d[ky]` is the separable spectral density of the
# wrapped Gaussian, normalized so that `C[i, i] = 1` (kernel diagonal
# matches a unit correlation).
function _gaussian_transfer_sqrt_2d(Nc::Int, L::FT) where FT
    k1d = _gaussian_periodic_kernel(Nc, L)
    # Normalize so `C[0, 0] = k[1] = 1` after division. `k[1]` is the
    # diagonal value (n = 0) and is strictly positive.
    k1d ./= k1d[1]
    # FFT eigenvalues of a circulant matrix are the FFT of its first
    # row. The kernel is real and symmetric (`k[Nc - n] = k[n]` modulo
    # periodicity), so the spectrum is real. Clamp any tiny negative
    # roundoff to zero before sqrt.
    spec_complex = FFTW.fft(k1d)
    spec_1d = real.(spec_complex)
    @inbounds for k in eachindex(spec_1d)
        spec_1d[k] = max(spec_1d[k], zero(FT))
    end
    sqrt_spec = sqrt.(spec_1d)
    # Outer product gives the 2D separable spectral square root.
    out = Matrix{FT}(undef, Nc, Nc)
    @inbounds for ky in 1:Nc, kx in 1:Nc
        out[kx, ky] = sqrt_spec[kx] * sqrt_spec[ky]
    end
    return out
end

# ---------------------------------------------------------------------------
# Forward + adjoint application
# ---------------------------------------------------------------------------

"""
    apply_B_half!(y::NTuple{6, AbstractMatrix},
                  cov::AbstractCSSurfaceFluxCovariance,
                  chi::NTuple{6, AbstractMatrix}) -> y

Apply the (non-symmetric) square root `B^(1/2)` to `chi` and write the
result into `y`. Concretely, `y = D · L · chi`, where `D` is the
diagonal scaling and `L` is the symmetric correlation square root.
`y` and `chi` may alias; the implementation writes through a scratch
buffer when needed.
"""
function apply_B_half! end

"""
    apply_B_half_adjoint!(g_chi::NTuple{6, AbstractMatrix},
                          cov::AbstractCSSurfaceFluxCovariance,
                          g_phys::NTuple{6, AbstractMatrix}) -> g_chi

Apply the adjoint `(B^(1/2))^T` to `g_phys`. With `B^(1/2) = D · L` and
`D`, `L` both symmetric, this is `g_chi = L · D · g_phys`.
"""
function apply_B_half_adjoint! end

# ---- DiagonalCSCovariance ----------------------------------------------------

function apply_B_half!(y::NTuple{6, A},
                      cov::DiagonalCSCovariance{FT, A},
                      chi::NTuple{6, A}) where {FT, A}
    _validate_panel_shapes(y, cov.Nc, "y")
    _validate_panel_shapes(chi, cov.Nc, "chi")
    @inbounds for p in 1:6
        @. y[p] = cov.sigma[p] * chi[p]
    end
    return y
end

# `B^(1/2) = D` is self-adjoint for the diagonal case.
function apply_B_half_adjoint!(g_chi::NTuple{6, A},
                               cov::DiagonalCSCovariance{FT, A},
                               g_phys::NTuple{6, A}) where {FT, A}
    return apply_B_half!(g_chi, cov, g_phys)
end

# ---- IsotropicGaussianCSCovariance ------------------------------------------

# Apply the symmetric correlation square root `L` to a single panel
# in-place via 2D FFT. Real input, real output (within float noise).
function _apply_L_panel!(buf::Matrix{Complex{FT}}, input::AbstractMatrix{FT},
                        output::AbstractMatrix{FT},
                        transfer_sqrt::Matrix{FT}) where FT
    Nc = size(transfer_sqrt, 1)
    @inbounds for j in 1:Nc, i in 1:Nc
        buf[i, j] = Complex{FT}(input[i, j])
    end
    FFTW.fft!(buf)
    @inbounds for j in 1:Nc, i in 1:Nc
        buf[i, j] *= transfer_sqrt[i, j]
    end
    FFTW.ifft!(buf)
    @inbounds for j in 1:Nc, i in 1:Nc
        output[i, j] = real(buf[i, j])
    end
    return output
end

function apply_B_half!(y::NTuple{6, A},
                      cov::IsotropicGaussianCSCovariance{FT, A},
                      chi::NTuple{6, A}) where {FT, A}
    _validate_panel_shapes(y, cov.Nc, "y")
    _validate_panel_shapes(chi, cov.Nc, "chi")
    Nc = cov.Nc
    buf = Matrix{Complex{FT}}(undef, Nc, Nc)
    smooth = Matrix{FT}(undef, Nc, Nc)
    @inbounds for p in 1:6
        # B^(1/2) = D · L. Apply L first (smooth chi), then multiply
        # by D (per-cell sigma). `smooth` holds `L · chi[p]`.
        _apply_L_panel!(buf, chi[p], smooth, cov.L_transfer_sqrt)
        @. y[p] = cov.sigma[p] * smooth
    end
    return y
end

function apply_B_half_adjoint!(g_chi::NTuple{6, A},
                               cov::IsotropicGaussianCSCovariance{FT, A},
                               g_phys::NTuple{6, A}) where {FT, A}
    _validate_panel_shapes(g_chi, cov.Nc, "g_chi")
    _validate_panel_shapes(g_phys, cov.Nc, "g_phys")
    Nc = cov.Nc
    buf = Matrix{Complex{FT}}(undef, Nc, Nc)
    scaled = Matrix{FT}(undef, Nc, Nc)
    @inbounds for p in 1:6
        # (B^(1/2))^T = L · D. Apply D first (per-cell sigma), then L.
        @. scaled = cov.sigma[p] * g_phys[p]
        _apply_L_panel!(buf, scaled, g_chi[p], cov.L_transfer_sqrt)
    end
    return g_chi
end

# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

function _validate_positive_finite(arr::AbstractMatrix, name::AbstractString)
    @inbounds for j in axes(arr, 2), i in axes(arr, 1)
        v = arr[i, j]
        isfinite(v) || throw(ArgumentError(
            "$name has non-finite entry at ($i, $j): $v"))
        v > 0 || throw(ArgumentError(
            "$name has non-positive entry at ($i, $j): $v"))
    end
    return nothing
end

function _validate_panel_shapes(panels::NTuple{6, <:AbstractMatrix},
                                Nc::Int, name::AbstractString)
    @inbounds for p in 1:6
        size(panels[p]) == (Nc, Nc) || throw(DimensionMismatch(
            "$name panel $p has shape $(size(panels[p])); expected $((Nc, Nc))"))
    end
    return nothing
end
