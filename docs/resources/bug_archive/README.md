# Bug archive

Historical investigative postmortems migrated from `docs_legacy/` during
the pre-main cleanup. Each file is preserved as-is at its date; none
describe current code, all document how a specific bug was found and
fixed. Useful as reference when similar issues recur or when a new
contributor is auditing past decisions.

## Entries

- **[SESSION1_GRID_CM_FIX_2026-04-03.md](SESSION1_GRID_CM_FIX_2026-04-03.md)**
  Grid-convention fix (cell centers at ±89.75° vs ±90°) and Float64
  accumulation of `cm` in the ERA5 spectral preprocessor.
- **[SESSION2_NAN_BLOCKING_2026-04-03.md](SESSION2_NAN_BLOCKING_2026-04-03.md)**
  NaN blocker during `/n_sub` removal experiments; red/blue team
  analysis proving TM5 computes `am` per 450s, not 1800s. Diagnostic
  slopes on uniform fields → negative `rm` risks.
- **[ERA5_LL_TRANSPORT_FIXES_2026-03-30.md](ERA5_LL_TRANSPORT_FIXES_2026-03-30.md)**
  6-bug report on ERA5 lat-lon spectral transport: missing `cos(φ)`
  in mass flux, Strang-split + `m`-evolve consistency, reduced-grid
  distribute-back, TM5 band definitions, double `cm` residual
  correction. Canonical debugging methodology reference.
- **[TM5_4DVAR_DRY_VMR_AUDIT_2026-04-06.md](TM5_4DVAR_DRY_VMR_AUDIT_2026-04-06.md)**
  Audit of TM5's dry-air handling with code line references to the
  4D-Var Fortran. Identifies stale config flags and maps where
  dry-air logic actually lives vs dead code.
- **[AGENT_CHAT_LOG.md](AGENT_CHAT_LOG.md)**
  Append-only communication channel between CLAUDE and CODEX agents
  (Apr 3 – Apr 12). File-ownership declarations at top reference
  obsolete `src_v2`/`test_v2` paths. Preserved for provenance.
