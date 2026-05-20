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
| `completed_experiments/` | Preserved historical baselines and comparison runs. | Read `completed_experiments/README.md`. |
| `likely_legacy/` | Older configs that may reference moved data or old schemas. | Avoid for new work. |

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
