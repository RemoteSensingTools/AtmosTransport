# About these docs

The documentation is being overhauled. This page tracks where we are.

## Status (2026-04-25)

**Phase 1 — Infrastructure** (shipped). Documenter + Literate.jl scaffolding;
CI workflow wired green; landing page + this status page in place.

**Phase 2 — Getting Started onramp (current).** Four pages cover
installation, a downloadable-bundle **quickstart**, the general
"first run" walkthrough, and inspecting output. All four are
written against the current `scripts/run_transport.jl` invocation,
the current TOML schema (`[input]` + `[architecture]` + `[numerics]`
+ `[run]` + `[tracers.<name>.init]` + `[output]`), and the current
NetCDF output schema (`<tracer>_column_mean`,
`<tracer>_column_mass_per_area`, `air_mass_column`).

The quickstart bundle ships 3 days of preprocessed ERA5 transport
binaries at four grid configurations (LL 72×37, LL 144×73, CS C24,
CS C90, all F32) with matching ready-to-run configs in
`config/runs/quickstart/`. After upload, the convenience script
`scripts/download_quickstart_data.sh` fetches the tarball, verifies
its SHA-256, and extracts it under `~/data/AtmosTransport_quickstart/`.

**Caveats:**

- The `[gpu verified]` runtime line and the per-day output figures in
  these pages were captured from a real LL 72×37 F32 GPU run on this
  machine; CPU and other hardware will print equivalent but differently
  formatted lines.
- The two repository Python diagnostic scripts
  (`scripts/diagnostics/verify_snapshot_netcdf.py` and `quick_viz.py`)
  were written against an older snapshot variable schema; this is
  flagged in `inspecting_output.md` and tracked as a Phase-2 follow-up.
- The Dropbox URL and SHA-256 in `download_quickstart_data.sh` are
  TODO placeholders until the bundle is uploaded.

The scattered legacy docs at `docs/reference/`, `docs/memos/`, and the
top-level numbered files (`docs/00_*.md`, `docs/30_*.md`, etc.) are
**not yet integrated**. They remain in the repository (still browsable
via the GitHub web UI and any prior bookmarks) but are **not built
into this Documenter site** — the search index and cross-references
here only cover pages under `docs/src/`. Subsequent phases will fold
each legacy file into this tree or archive it.

## Roadmap

| Phase | Scope | State |
|------:|-------|-------|
| 1 | Documenter + Literate scaffolding, CI green | shipped |
| 2 | Getting-Started onramp (install, first run, inspecting output) | **in progress** |
| 3 | Concepts (grids, state, operators, binary) | pending |
| 4 | Tutorials (Literate.jl per topology) | pending |
| 5 | Preprocessing guide (spectral, GEOS native, regridding) | pending |
| 6 | Theory & Verification (mass conservation, schemes, validation) | pending |
| 7 | Configuration & Runtime (TOML, sample data, output) | pending |
| 8 | API Reference (`@autodocs` per module) | pending |
| 9 | Developer internals + final README polish | pending |

## Content owners

The docs aim to satisfy two audiences in one site:

- **Newcomers** — gentle onramp through Getting Started, Concepts, and
  one runnable Tutorial per topology.
- **Atmospheric-transport practitioners** — Theory & Verification with
  scheme derivations, mass-conservation contracts, and the full API
  reference.

If the level of explanation in a given page does not serve both, file
an issue and we will rework it.

## How to build the docs locally

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The HTML output lands in `docs/build/`. CI builds the same target on
every push and PR; deployment to `gh-pages` happens on pushes to
`main`.
