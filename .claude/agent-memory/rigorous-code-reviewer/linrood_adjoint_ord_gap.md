---
name: linrood-adjoint-ord-gap
description: LinRoodPPMScheme adjoint hardcodes Val(5); ORD=7 silently produces wrong adjoints with no runtime guard
metadata:
  type: project
---

The LinRood adjoint tape (`_record_linrood_horizontal_substep!` in
`src/Adjoints/LinRoodTape.jl`) hardcodes `Val(5)` for the face kernels
(y_face_k!, xq_face_k!, x_face_k!, yq_face_k!) and the single-panel
adjoint (`apply_linrood_horizontal_adjoint_single_panel!`).

Forward-only path `_linrood_run_forward_step!` uses `Val(ORD)`,
honoring the user's choice of `LinRoodPPMScheme(5)` vs
`LinRoodPPMScheme(7)`. There is no dispatch guard or runtime check —
calling the adjoint API with `LinRoodPPMScheme(7)` silently runs
ORD=5 stencils inside the tape and ORD=7 in the FD reference, so
parity tests would visibly fail but a production user wouldn't get
an explicit "ORD=7 not supported" error.

**Why:** noted as "ORD=7 + copy_corners reverse" on the resume
checklist in user MEMORY.md (Plan 25 follow-up); not blocking
Plan 26 A.3 but worth surfacing on every review until it's gated.

**How to apply:** when reviewing LinRood adjoint changes, check that
`scheme isa LinRoodPPMScheme{5}` is asserted somewhere in the entry
path (or that `Val(5)` is replaced by `Val(ORD)`). Until then,
flag ORD=7 as a known silent-incorrectness vector. Production C24
LA-footprint runs documented in MEMORY use the default ORD=5, so
no current run is at risk.
