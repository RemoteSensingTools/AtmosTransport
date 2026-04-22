# `TM5Convection` Basis Policy (Commit 0 Decision)

**Question:** Does `TM5Convection` accept `CellState{MoistBasis}`
only, `CellState{DryBasis}` only, or both?

**Reference:**

- CLAUDE.md Invariant 9 — moist vs dry mass-flux conventions in
  met data.
- [`src/Operators/Convection/CMFMCConvection.jl:19–24`](../../src/Operators/Convection/CMFMCConvection.jl#L19-L24)
  — CMFMCConvection's basis stance: "basis-polymorphic, driver
  is responsible for basis correction upstream".
- [`artifacts/plan18/upstream_fortran_notes.md`](../plan18/upstream_fortran_notes.md)
  §6.5 + §7 — ECMWF input is total (moist), TM5 ec2tm conversion
  keeps moist convention.

## Decision: basis-polymorphic, identical to CMFMCConvection

`TM5Convection` dispatches on `CellState{<:AbstractMassBasis}`
without constraining to `MoistBasis` or `DryBasis`. The operator
trusts the caller / driver to supply `(entu, detu, entd, detd)`
on the same basis as `state.air_mass`.

**Why this stance matches CMFMC:**

1. The matrix builder operates on `m(k)` (air mass per layer) and
   four flux-rate fields, all in consistent units of `kg/m²/s`.
   Whether those kg are dry or total air is a calibration choice
   of the preprocessor, not a mathematical property of the kernel.
2. `conv1 = I - dt · D` treats mixing ratios `rm / m` identically
   regardless of basis — the matrix is unitless because the `dt ·
   f` products are dimensionless flux ratios per unit mass.
3. CMFMCConvection ships as basis-polymorphic with explicit
   docstring acknowledgement that driver + preprocessor provide
   basis coherence. TM5Convection must mirror that contract so
   sim-level code (`with_convection(model, op)`) works the same
   way for both.
4. Writing a moist-only variant would require (a) a
   runtime moist/dry conversion helper on `(entu, detu, entd,
   detd)` or (b) a second binary layout, both of which drift from
   the plan's "no future refactors" discipline. The caller's
   existing basis contract (set by the preprocessor + driver)
   already covers this.

**Implication for preprocessor (Commit 3):**

- `phys_convec_ec2tm.F90` port must be explicit about which basis
  the four output fields are on.
- Upstream ECMWF fields are total (moist). Port keeps that by
  default and writes `mass_basis = :moist` in the binary header
  alongside the four sections.
- A dry-basis preprocessor variant is outside plan 23 scope
  (listed in the Risk register / latent deferral); would ship
  as a separate plan following the agent's persistent-feedback
  guidance on dry-basis preference
  (auto-memory `feedback_dry_basis_default.md`).

**Implication for runtime validator (Commit 1):**

- `_validate_convection_window!(::TM5Convection, window, driver)`
  rejects windows where `window.convection.tm5_fields === nothing`
  but does **not** reject on basis mismatch — that's caller's
  contract, enforced at preprocessor + `ConvectionForcing`
  construction time.
- Error message names the fix: "TM5Convection requires
  `window.convection.tm5_fields` from a transport binary
  preprocessed with TM5 sections. Preprocess with
  `scripts/preprocessing/preprocess_spectral_v4_binary.jl` and
  `tm5_convection = true` in the config, or fall back to
  `CMFMCConvection()` if you have GEOS-FP CMFMC data instead."

**Implication for adjoint (plan 19 latent):**

- The adjoint kernel inherits the same basis polymorphism. No
  special casing needed.

## Status

Commit 0 decision: polymorphic on basis. Commits 1 / 3 / 4 /
6 / 7 all respect this stance. No per-basis dispatch in the
TM5Convection hierarchy.
