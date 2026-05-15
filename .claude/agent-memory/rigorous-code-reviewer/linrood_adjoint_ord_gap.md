---
name: linrood-adjoint-ord-gap
description: RESOLVED 2026-05-15 — LinRoodPPMScheme adjoint now supports ORD ∈ {5, 7}; ORD=7 silent-incorrectness gap closed (Plan-25 Commit 3b)
metadata:
  type: project
---

**RESOLVED 2026-05-15.** The hardcoded `Val(5)` in the LinRood adjoint
chain is gone. `LinRoodPPMScheme(7)` now produces correct adjoints
end-to-end:

- 4 new ORD=7 adjoint kernels in `src/Operators/Advection/linrood_adjoint_kernels.jl`
  (`_ppm_{x,y}_face_kernel_adjoint_ord7!` and the from-q variants),
  driven by `_apply_ord7_boundary_d6` and `_linrood_ppm_face_*_ord7`
  helpers. The interior chain is bit-equal to ORD=5 (verified by
  dedicated bit-parity tests); the panel-edge correction is the
  transpose of the forward linear `_apply_ord7_boundary`.
- The 4 public wrappers (`apply_ppm_{x,y}_face_adjoint!` and the
  from-q variants) now dispatch on `Val{ORD}` with ORD ∈ {5, 7}.
- `_CSLinRoodHorizRecord` carries `ORD` as a 6th type parameter so
  `_apply_cs_linrood_horizontal_adjoint!` reads it from dispatch and
  forwards `Val(ORD)` to the face-kernel adjoints automatically.
- `_record_cs_linrood_tape` no longer rejects ORD=7; the previous
  `ORD == 5 || throw(ArgumentError(...))` guard is now an
  `(ORD == 5 || ORD == 7) || throw(...)` admission gate.
- FD-vs-adjoint parity verified at panel-edge AND interior faces for
  all 4 kernels (`test_linrood_kernel_adjoints.jl`). Stride-vs-Full
  parity verified for the LinRood ORD=7 tape (`test_cs_stride_checkpoint.jl`).
  End-to-end `cs_surface_emission_footprint` FD parity verified
  (`test_linrood_adjoint_integration.jl`).

**Historical note (for understanding the gap that was there).** The
adjoint tape (`_record_linrood_horizontal_substep!` in
`src/Adjoints/LinRoodTape.jl`) used to hardcode `Val(5)` for the
face kernels and the single-panel adjoint, while the forward-only
path `_linrood_run_forward_step!` honoured `Val(ORD)`. Without a
guard, ORD=7 forward + ORD=5 adjoint silently produced a wrong
gradient. The fix runs `Val(ORD)` through the tape signature and
binds it into the record type so the dispatch can't drift.

**How to apply:** No longer a flag-on-review item. Future reviews
of LinRood adjoint code should verify the `Val(ORD)` plumbing
matches the forward path; if a NEW ORD is added (e.g. ORD=4 or
ORD=6 ever ship as adjoint-capable schemes), the four public
wrappers and the `_record_cs_linrood_tape` admission gate need
updating to recognize it.
