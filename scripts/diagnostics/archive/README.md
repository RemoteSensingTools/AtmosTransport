# Archived diagnostics — concluded experiments

These scripts are preserved for provenance only; they are **not** maintained
and may not run against the current code. They were one-off investigations
whose conclusions shipped elsewhere:

- `fingerfix_proto_*`, `prototype_pfix_balanced.jl`, `prototype_remap_vs_cm.jl`
  — the 2026-06 SH-UTLS flux-fingering investigation. Conclusion: the
  `omega_consistent` cm closure (production, `geos_cm_closure="omega_consistent"`)
  and the MERRA-2 / wind-derived path; see
  `docs/reference/GEOS_MASS_FLUX_UTLS_FINGERING.md`. The `prototype_remap_vs_cm.jl`
  result (remap ≡ cm-advection, ratio 1.00) refuted the "remap is the fix"
  hypothesis — recorded in the memo, prototype no longer needed.
