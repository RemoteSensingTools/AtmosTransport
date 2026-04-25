# About these docs

The documentation is being overhauled. This page tracks where we are.

## Status (2026-04-25)

**Phase 1 — Infrastructure** (shipped). Documenter + Literate.jl scaffolding;
CI workflow wired green; landing page + this status page in place.

**Phase 2 — Getting Started onramp** (shipped). Four pages cover
installation, a downloadable-bundle **quickstart**, the general
"first run" walkthrough, and inspecting output. All four are written
against the current `scripts/run_transport.jl` invocation, the
current TOML schema, and the current NetCDF output schema.

The quickstart bundle ships 3 days of preprocessed ERA5 transport
binaries at two LL resolutions (72×37 and 144×73, F32) with matching
ready-to-run configs in `config/runs/quickstart/`. The convenience
script `scripts/download_quickstart_data.sh` fetches the tarball,
verifies its SHA-256, and extracts it under
`~/data/AtmosTransport_quickstart/`. CS bundles are deferred until
the F32 spectral-CS preprocessor fix lands.

**Phase 3A — Concepts: grids + state & basis** (shipped). Two pages
covering the three horizontal mesh types, the AtmosGrid composite,
state types, dry-basis contract, and tracer accessor API. Mermaid
hierarchy diagrams + three CairoMakie-rendered grid schematics
(LatLon, ReducedGaussian, CubedSphere with both panel conventions
side by side).

**Phase 3B — Concepts: operators + binary format** (shipped). Two
pages: the four operator families (advection schemes, diffusion,
convection, surface flux) with the `apply!` contract and the Strang
palindrome; and the v4 transport-binary header schema, payload
sections, capability surface, mass-basis contract, streaming-writer
entry points, and replay-gate contract.

**Phase 4 — Tutorials** (shipped, partial). Literate.jl wired into
the docs build (`docs/literate/*.jl` → `docs/src/tutorials/_generated/*.md`,
executed at build time so the rendered output matches what the
reader will actually see locally). First tutorial:
`synthetic_latlon.jl` — builds a tiny synthetic transport binary in
memory using only public API and steps it forward. CI-safe.
Real-data topology tutorials will land once the LL+CS quickstart
bundle is distributable as a Julia LazyArtifact (currently blocked
on the F32 spectral-CS preprocessing fix).

**Phase 5 — Preprocessing guide (current).** Five pages: a unified
overview with the source × target dispatch model and a prominent
warning that preprocessing is time-intensive (but the resulting
binaries enable optimized I/O at runtime); per-source deep-dives for
ERA5 spectral and GEOS native CS (FV3 pressure-fixer cm,
mass_flux_dt = 450, GCHP convection wiring); a regridding chapter
covering conservative weights, IdentityRegrid, and the JLD2 cache;
and a one-page conventions cheat sheet (units, replay tolerances,
level orientation, panel conventions).

**Caveats carried forward:**

- The two repository Python diagnostic scripts
  (`scripts/diagnostics/verify_snapshot_netcdf.py` and `quick_viz.py`)
  were written against an older snapshot variable schema; this is
  flagged in `inspecting_output.md` and tracked as a Phase-2
  follow-up.
- The Dropbox URL in `download_quickstart_data.sh` is still a TODO
  placeholder until the LL bundle is uploaded.

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
| 2 | Getting-Started onramp (install, quickstart, first run, inspecting output) | shipped |
| 3A | Concepts: grids, state & basis (Mermaid hierarchies + grid schematics) | shipped |
| 3B | Concepts: operators, binary format | shipped |
| 4 | Tutorials (Literate.jl) — synthetic LL shipped; real-data topology tutorials pending bundle artifact | shipped |
| 5 | Preprocessing guide (overview, spectral_era5, geos_native_cs, regridding, conventions) | **in progress** |
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
