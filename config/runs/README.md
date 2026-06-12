# Run Configs

Runtime TOMLs are consumed by the canonical runner:

```bash
julia --project=. scripts/run_transport.jl <config.toml>
```

Use `scripts/run_transport.jl --help` to list the quickstart configs.

| Folder | Purpose | Start Here |
|---|---|---|
| `quickstart/` | Three-day downloaded example-data runs. | `quickstart/ll72x37_advonly.toml` |
| `advresln/` | Short advection-resolution experiments. | `advresln/ll72x37_advonly.toml` |
| `validation/` | Synthetic and reduced-size validation runs. | Use the matching test or validation note. |
| `catrine5d/` | CATRINE five-day campaign matrix. | Pick grid, precision, and physics suffix explicitly. |
| `campaign_winter2021/` | **Maintained** Dec 2021 4-tracer production campaign (GEOS-IT omega + ERA5). | `campaign_winter2021/geosit_omega_4tracer_val_dec1-2.toml` |
| `binary_format_ab/` | Reference A/B configs proving per-substep vs full-window flux storage equivalence (`flux_kind`). | `binary_format_ab/c45_new_format.toml` |
| `anomaly_ref_gate/` | Reference-state (anomaly) transport gate configs (`stage0_linrood_1day.toml` is the bit-identical regression anchor). | `anomaly_ref_gate/stage0_linrood_1day.toml` |
| `completed_experiments/` | Preserved historical baselines and comparison runs. | Read `completed_experiments/README.md`. |
| `likely_legacy/` | Older configs that may reference moved data or old schemas. Effective archive — avoid for new work. | Avoid for new work. |

## Classification

- **Maintained**: `campaign_winter2021/`, `quickstart/`, `validation/` — kept current with the schema and data layout.
- **Reference**: `anomaly_ref_gate/`, `binary_format_ab/` — regression/equivalence anchors; change only with a matching gate update.
- **Experimental**: `advresln/`, `catrine5d/`, `catrine_*`, `*_compare_*`, `_diag_*` — one-off study configs; may go stale.
- **Archived**: `likely_legacy/`, `completed_experiments/` — preserved for provenance; not expected to run as-is.

## Data Roots

Most production configs use `$ATMOSTRANSPORT_DATA_ROOT/...`, which defaults to
`~/data/AtmosTransport` when unset. Quickstart configs use
`$ATMOSTRANSPORT_DATA_ROOT_quickstart/...`, which defaults to
`~/data/AtmosTransport_quickstart`.

For custom storage locations:

```bash
export ATMOSTRANSPORT_DATA_ROOT=/scratch/$USER/AtmosTransport
export ATMOSTRANSPORT_DATA_ROOT_quickstart=/scratch/$USER/AtmosTransport_quickstart
```

## Custom Runs

For a new run, copy `config/examples/minimal_template.toml` or one of the
four `quickstart/` files, then change `[input]`, `[tracers]`, and `[output]`.
The runtime auto-detects grid topology from the first transport binary header;
run configs do not need a `[grid]` section.
