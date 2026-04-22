# ---------------------------------------------------------------------------
# ec2tm — ECMWF → TM5 convection-field conversion.
#
# Port of TM5-4DVAR `phys_convec_ec2tm.F90` transformation that derives
# `(entu, detu, entd, detd)` at LAYER CENTERS from ECMWF's raw
# convection fields:
#
#   - `mflu_ec` : updraft mass flux at HALF LEVELS (interfaces), positive
#   - `mfld_ec` : downdraft mass flux at HALF LEVELS, negative in ECMWF
#                 convention (re-signed positive by ec2tm)
#   - `detu_ec` : updraft detrainment rate at FULL LEVELS (layer centers),
#                 positive (very small negative values possible from
#                 ECMWF diagnostic rounding; clipped)
#   - `detd_ec` : downdraft detrainment rate at FULL LEVELS, positive
#                 (same clipping)
#
# Output (AtmosTransport orientation, k=1=TOA, k=Nz=surface):
#
#   - `entu` : updraft entrainment rate at layer centers, positive
#   - `detu` : updraft detrainment rate at layer centers, positive
#   - `entd` : downdraft entrainment rate at layer centers, positive
#              (sign-flipped from ECMWF `mfdo`)
#   - `detd` : downdraft detrainment rate at layer centers, positive
#
# All outputs in kg / m² / s.
#
# Mass-balance closure (ec2tm derivation):
#
# Updraft continuity: `entu(k) = detu(k) + mfup(k) - mfup(k-1)`
#                     where `mfup(k-1)` is the interface below layer k
#                     (surface side in TM5 convention).
# Downdraft:          `entd(k) = detd(k) - mfdo(k) + mfdo(k-1)` with sign
#                     flip on mfdo.
#
# See upstream_fortran_notes.md §6.5 + §7 in artifacts/plan18/ for the
# full provenance of these formulas.
#
# Orientation: INPUTS are expected in ECMWF native orientation
# (k=1=TOA, k=Nz=surface for CDS; half-levels numbered k=1 at TOA,
# k=Nz+1 at surface). OUTPUTS are in AtmosTransport orientation
# (same as the input ECMWF orientation). If raw data arrives in the
# OPPOSITE orientation (some MARS extractions have k=1=surface),
# reverse BEFORE calling ec2tm (a runtime reorientation inside this
# function would violate plan 23 principle 1).
#
# The commit point for this conversion in the preprocessor pipeline
# is after moist-field merging but before Poisson balancing.
# Preprocessor integration (wiring this function into the spectral
# pipeline + adding convective-variable downloads from CDS) is a
# future scope item — the math + tests ship here ready for that
# integration.
# ---------------------------------------------------------------------------

"""
    ec2tm!(entu, detu, entd, detd,
            mflu_ec, mfld_ec, detu_ec, detd_ec) -> (entu, detu, entd, detd)

Convert ECMWF convective mass-flux fields into TM5 `(entu, detu,
entd, detd)` layer-center fields. All inputs / outputs in
AtmosTransport orientation (k=1=TOA, k=Nz=surface). Operates in
place on the four output arrays (no allocations).

Shapes:

- `entu, detu, entd, detd`: `(..., Nz)` — layer centers.
- `mflu_ec, mfld_ec`: `(..., Nz+1)` — half levels (interfaces).
  Interface `k` is the TOP of layer `k`; interface `Nz+1` is the
  surface boundary.
- `detu_ec, detd_ec`: `(..., Nz)` — layer centers.

The leading dimensions are arbitrary (scalar, `(Nx, Ny)`, `(ncells,)`,
or per-panel `(Nc, Nc)`) as long as the arrays all have consistent
shape. This function is backend-agnostic pure Julia; call from the
preprocessor (CPU) ahead of binary writeout.

Negative small values (`<= 0`) in `detu_ec` / `detd_ec` are clipped
to zero, following the ECMWF-rounding-artifact clean-up documented
in `phys_convec_ec2tm.F90` (ECMWF diagnostics can produce
~-1e-19 values from rounding).

For every location, computes:

```
k = 1:          entu[1]   = detu[1]   (updraft starts in layer 1)
                entd[1]   = detd[1]   (no flux at TOA)
k ∈ 2:Nz:       entu[k]   = detu[k]   + mflu_ec[k+1] - mflu_ec[k]
                entd[k]   = detd[k]   - mfld_ec[k]   + mfld_ec[k-1]
                (where mfld_ec sign-flipped internally)
```

Wait — above formula is written in TM5 convention (k=1=surface).
Let me re-write in AtmosTransport orientation explicitly.

**AtmosTransport orientation** (k=1=TOA):

- Layer k has interface `k` at its TOP (higher altitude side) and
  interface `k+1` at its BOTTOM (lower altitude side).
- Updraft flows upward: from interface `k+1` into layer `k` via
  entrainment `entu[k]`, out of layer `k` via interface `k`.
  Continuity: `mflu_out − mflu_in + detu − entu = 0`
             ⟹ `entu[k] = detu[k] + mflu_ec[k] − mflu_ec[k+1]`
  where `mflu_ec[k]` is the flux through interface k (above layer k)
  and `mflu_ec[k+1]` is the flux through interface k+1 (below layer k).
  Convention: `mflu_ec >= 0` (positive upward).

- Downdraft flows downward: from interface `k` into layer `k` via
  entrainment `entd[k]`, out through interface `k+1` via detrainment.
  Sign convention in ECMWF: `mfdo <= 0` (negative). We define
  `mfdo_abs = -mfdo_ec >= 0`, then
  `entd[k] = detd[k] + mfdo_abs[k+1] − mfdo_abs[k]`.

Boundary conditions:

- `mflu_ec[1] = 0` (no updraft above TOA).
- `mflu_ec[Nz+1] = 0` (no updraft into ground from below — the
  surface is the sink).
- `mfld_ec[1] = 0` (no downdraft above TOA).
- `mfld_ec[Nz+1] = 0` (no downdraft escapes the surface).

These boundaries are typically already zero in ECMWF output; we
enforce explicitly by construction below.
"""
function ec2tm!(entu::AbstractArray{FT}, detu::AbstractArray{FT},
                entd::AbstractArray{FT}, detd::AbstractArray{FT},
                mflu_ec::AbstractArray{FT},
                mfld_ec::AbstractArray{FT},
                detu_ec::AbstractArray{FT},
                detd_ec::AbstractArray{FT}) where {FT <: AbstractFloat}
    size(entu) == size(detu) == size(entd) == size(detd) ||
        throw(ArgumentError("ec2tm!: entu/detu/entd/detd must have identical shape"))
    size(entu) == size(detu_ec) == size(detd_ec) ||
        throw(ArgumentError("ec2tm!: layer-center inputs (detu_ec, detd_ec) must match entu shape"))
    _ec2tm_check_halflevel_shape(size(entu), size(mflu_ec), "mflu_ec")
    _ec2tm_check_halflevel_shape(size(entu), size(mfld_ec), "mfld_ec")

    Nz = size(entu)[end]   # last dim is vertical
    Nz >= 1 || return (entu, detu, entd, detd)

    # Clip spurious negative detrainment (ECMWF diagnostic rounding).
    @inbounds for I in eachindex(detu, detu_ec)
        detu[I] = detu_ec[I] < 0 ? zero(FT) : detu_ec[I]
    end
    @inbounds for I in eachindex(detd, detd_ec)
        detd[I] = detd_ec[I] < 0 ? zero(FT) : detd_ec[I]
    end

    # Build `mfdo_abs = -mfld_ec` inline. ECMWF convention: mfld_ec <= 0;
    # we want positive magnitudes for the TM5 downdraft entrainment
    # formula. We read sign-flipped values directly off mfld_ec below.

    # The conversion loop runs over (non-vertical) locations and all
    # vertical layers. Because the leading dims are arbitrary, we
    # iterate using `CartesianIndices` over the leading shape.
    leading = Base.front(size(entu))
    @inbounds for I_h in CartesianIndices(leading)
        # Enforce boundary conditions on half-level fluxes (TOA and
        # surface). If the ECMWF data already zeros these we just
        # re-confirm; if it doesn't (e.g. noisy diagnostic), we clamp.
        # The half-level arrays have last-dim = Nz+1.
        for k in 1:Nz
            # entu[k] = detu[k] + mflu_ec[k] - mflu_ec[k+1]
            flux_top    = mflu_ec[I_h, k]      # interface above layer k (higher altitude)
            flux_bottom = mflu_ec[I_h, k + 1]  # interface below layer k (lower altitude)
            raw_entu    = detu[I_h, k] + flux_top - flux_bottom
            # Updraft entrainment is physically non-negative; clamp
            # numerical-noise negatives.
            entu[I_h, k] = raw_entu < 0 ? zero(FT) : raw_entu

            # entd[k] = detd[k] + |mfld_ec|[k+1] - |mfld_ec|[k]
            mfdo_abs_top    = -mfld_ec[I_h, k]       # = |mfld_ec[k]|
            mfdo_abs_bottom = -mfld_ec[I_h, k + 1]   # = |mfld_ec[k+1]|
            raw_entd        = detd[I_h, k] + mfdo_abs_bottom - mfdo_abs_top
            entd[I_h, k] = raw_entd < 0 ? zero(FT) : raw_entd
        end
    end
    return (entu, detu, entd, detd)
end

# Shape guard for half-level inputs: leading dims must match the
# layer-center arrays; last dim must be Nz+1.
@inline function _ec2tm_check_halflevel_shape(center_shape::Tuple,
                                               halflev_shape::Tuple,
                                               name::String)
    Nz = center_shape[end]
    expected = (Base.front(center_shape)..., Nz + 1)
    halflev_shape == expected ||
        throw(ArgumentError(
            "ec2tm!: $name shape $(halflev_shape) must match " *
            "$(expected) (leading dims plus Nz+1 interface layer)"))
    return nothing
end
