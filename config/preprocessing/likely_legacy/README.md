# Likely Legacy Preprocessing Configs

This folder is a quarantine area, not a deletion queue.

Files here either reference preprocessor scripts that no longer exist in
`scripts/preprocessing/`, or describe older GEOS/Catrine preprocessing pathways
that have not yet been re-established in the unified `src/Preprocessing`
pipeline.

Active preprocessing configs must declare either `[source].toml` for a typed
native source or `[input].spectral_dir` for ERA5 spectral, plus an explicit
`[grid].type`.

Before reactivating a config:

1. Update the command header to use an existing entrypoint.
2. Confirm the TOML schema matches `scripts/preprocessing/preprocess_transport_binary.jl`
   or document the replacement workflow.
3. Run a one-day smoke test and move the config back to `config/preprocessing/`.
