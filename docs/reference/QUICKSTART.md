# Quickstart

This legacy reference page intentionally redirects to the rendered
getting-started guide:

- Source: `docs/src/getting_started/quickstart.md`
- Rendered docs: `https://RemoteSensingTools.github.io/AtmosTransport/dev/getting_started/quickstart`

The current quickstart uses the `data-quickstart-v2` bundle, the canonical
runtime runner, and the quickstart data root:

```bash
bash scripts/download_quickstart_data.sh ll
julia --project=. scripts/run_transport.jl config/runs/quickstart/ll72x37_advonly.toml
```

Set `ATMOSTRANSPORT_DATA_ROOT_quickstart` before downloading if the bundle
should live outside `~/data/AtmosTransport_quickstart`.
