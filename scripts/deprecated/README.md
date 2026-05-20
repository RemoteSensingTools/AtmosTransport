# Deprecated Runner Shims

These files preserve the old runtime script names for reference:

- `run_transport_binary.jl`
- `run_cs_driven.jl`
- `run_cs_transport.jl`

New runs should use the single canonical runner:

```bash
julia --project=. scripts/run_transport.jl <config.toml>
```

The low-level cubed-sphere benchmark remains at
`scripts/benchmarks/run_cs_transport.jl`.
