# Orphan tier triage (2026-06-12)

The orphan tier is CI-excluded (`test/runtests.jl` runs it only with
`--orphan`). Each file was run standalone; the result **refutes the earlier
"60–70 % dead or subsumed" estimate** — 16 of 23 PASS and **none of the 16
has a same-named core counterpart** (i.e. unique, currently-lost coverage).

## PASS — unique, promotable (16)

These pass standalone and duplicate no `test/core/` file by name. Candidates
to promote into `test/core/` (recovers coverage). Promote in small batches and
re-run the full suite — they were tiered out during the suite reorg
(`e3ef729a`) for unrecorded reasons (possibly runtime or unvetted parallel
interaction), so verify CI time / ordering before bulk promotion.

`test_convection_types`, `test_cs_poisson_projection_cadence`,
`test_diffusion_kernels`, `test_diffusion_palindrome`,
`test_emissions_palindrome`, `test_era5_surface_reader`, `test_fields`,
`test_gchp_vdiff_binary_payload`, `test_output_snapshots` (import-rot already
fixed this session), `test_per_tracer_flux_map`,
`test_pressure_overlap_vertical_remap`, `test_qv_longitude_normalization`,
`test_surface_flux_operator`, `test_tm5_sparsity_above_icltop`,
`test_transport_model_diffusion`, `test_transport_model_emissions`.

## ROT — fail on stale imports/API, logic likely sound (5)

Same failure class as `test_output_snapshots` was (healthy suite, drifted
imports). Fix imports, then re-triage as PASS → promote.

`test_cmfmc_convection`, `test_cubed_sphere_runtime`, `test_current_time`,
`test_diffusion_operator`, `test_run_transport_binary_recipe`.

## BROKEN — needs investigation (2)

Did not produce a test summary (load/compile-time failure). Investigate
individually; may reference deleted APIs.

`test_chemistry`, `test_convection_forcing`.

## Recommendation

Promote the 16 PASS files in 2–3 batches (verify full-suite time stays
acceptable), fix + promote the 5 ROT, triage the 2 BROKEN last. Net: recovers
~21 files of unique coverage that CI currently skips.
