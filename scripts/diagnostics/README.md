# Diagnostics scripts

Inspection, validation, and analysis tools. Most are reusable; a few are
study one-offs. Concluded experiment prototypes live in `archive/`.

## Canonical / reusable

| Tool | Purpose |
|---|---|
| `inspect_transport_binary.jl` | Header, geometry, capability summary, load-time gates, driver-compat probe. The first thing to run on any binary. |
| `true_mass_balance.py` | **Authoritative** tracer mass budget — uses the model's logged `kg_air_equiv/s` emission rate and the F64 `<tracer>_total_mass` output variable, NOT a hardcoded rate or an F32 field integral (see `docs/reference/MASS_BALANCE.md` and the "always check true mass balance" rule). |
| `split_substep_analysis.jl` | Per-direction (x/y/z) substep requirements from any CS binary (`flux_kind`-aware). Backs the split-schedule payoff (BACKLOG 11c). |
| `compare_nc_bitident.py` | Variable-level byte comparison of two snapshot NetCDFs (the bit-identical regression gate). |

## Mass-balance checkers (overlapping — consolidation candidate, BACKLOG 14)

`true_mass_balance.py` is the canonical one. These others predate it and
overlap; prefer `true_mass_balance.py` for new work. Consolidating them into
one parametric tool is deferred (each carries slightly different reference
comparisons that need a deliberate merge): `check_mass_balance_dec2021.jl`,
`check_mass_conservation.py`, `tracer_mass_balance_vs_gc.py`,
`tracer_budget_closure.jl`, `stage4_surplus_verdict.py`.

## Wind / flux normalization

`compare_c180_binary_winds.jl`, `compare_c180_binary_mass_fluxes.jl`,
`diagnose_south_africa_wind_profiles.jl`, `compare_era_c180_binary_to_raw_uv.jl`
— all `flux_kind`-aware via `MetDrivers.flux_application_seconds` /
`flux_storage_substep_scale` (never hand-roll `dt/(2*steps)`; wrong by
`2*steps` on full-window binaries).

## archive/

Concluded experiment prototypes (SH-UTLS fingering investigation, etc.).
Preserved for provenance; not maintained.
