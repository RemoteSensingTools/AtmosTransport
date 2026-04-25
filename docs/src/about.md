# About these docs

The documentation is being overhauled. This page tracks where we are.

## Status (2026-04-25)

**Phase 1 — Infrastructure (current).** The Documenter + Literate.jl
scaffolding lands in this commit. The site is intentionally minimal —
landing page, this status page, and not much else — so reviewers can
verify the build, deploy, and CI workflow are wired correctly before
content lands.

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
| 1 | Documenter + Literate scaffolding, CI green | **in progress** |
| 2 | Getting-Started onramp (install, first run, inspecting output) | pending |
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
