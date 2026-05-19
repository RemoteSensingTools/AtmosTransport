# Legacy Runtime Configs

This directory holds old runtime TOMLs that do not satisfy the canonical
`scripts/run_transport.jl` input schema.

Active run configs must declare `[input]` with either `binary_paths` or
`folder` plus `start_date` and `end_date`. Files here are preserved for
provenance only; migrate one back to `config/runs/` only after updating it to
the current schema.
