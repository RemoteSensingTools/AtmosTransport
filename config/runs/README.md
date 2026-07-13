# Run Configs

Runtime TOMLs are consumed by the canonical runner:

```bash
julia --project=. scripts/run_transport.jl <config.toml>
```

For a first run without external data, use the maintained synthetic example:

```bash
julia --project=. examples/generate_synthetic_quickstart.jl
julia --project=. scripts/run_transport.jl config/examples/minimal_template.toml
```

| Folder | Purpose | Start Here |
|---|---|---|
| `advresln/` | Short advection-resolution experiments. | `advresln/ll72x37_advonly.toml` |
| `validation/` | Synthetic and reduced-size validation runs. | Use the matching test or validation note. |
| `catrine5d/` | CATRINE five-day campaign matrix. | Pick grid, precision, and physics suffix explicitly. |
| `completed_experiments/` | Preserved historical baselines and comparison runs. | Read `completed_experiments/README.md`. |
| `likely_legacy/` | Older configs that may reference moved data or old schemas. | Avoid for new work. |

## Data Roots

Most production configs use `$ATMOSTRANSPORT_DATA_ROOT/...`, which defaults to
`~/data/AtmosTransport` when unset. The synthetic quickstart writes beneath the
repository's ignored `data/quickstart/` directory.

For custom storage locations:

```bash
export ATMOSTRANSPORT_DATA_ROOT=/scratch/$USER/AtmosTransport
```

## Custom Runs

For a new run, copy `config/examples/minimal_template.toml`, then change
`[input]`, `[tracers]`, and `[output]`.
The runtime auto-detects grid topology from the first transport binary header;
run configs do not need a `[grid]` section.
