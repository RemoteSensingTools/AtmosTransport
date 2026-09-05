# Test layout

Tests are organized by tier. The orchestrator is [`runtests.jl`](runtests.jl);
it runs `core/` and `regridding/` by default and accepts opt-in flags to
include other tiers.

| Tier | Folder | When it runs | What's here |
|---|---|---|---|
| **Core** | [`core/`](core/) | Default (CI green-bar) | API/health gates, kernel correctness, runtime composition, adjoint identity, preprocessor contracts, mesh / regridding / vertical-transform invariants, type and aliasing safety. Everything that catches regressions of the production code paths. |
| **Real data** | [`real_data/`](real_data/) | `--all` or `--real-data` | Tests that need preprocessed transport binaries or ERA5 GRIB files on disk in `~/data/AtmosTransport/`. Skipped on CI runners. |
| **Diagnostic** | [`diagnostic/`](diagnostic/) | `--all` or `--diagnostic` | Large numerical sweeps useful for adjoint / inversion debugging (Taylor-sweep, integration footprints). Run occasionally, not every PR. |
| **Orphan** | [`orphan/`](orphan/) | `--orphan` only | Promotion candidates — tests that exist but were not in the CI roster as of 2026-05-29. Each should be reviewed and either promoted to `core/` (with a short note here) or moved to `archived/`. |
| **Archived** | [`archived/`](archived/) | Never (kept for reference) | Tests against deleted preprocessing wrappers and one-off plan-decision "studies". See [`archived/legacy_README.md`](archived/legacy_README.md). |
| **Regridding** | [`regridding/`](regridding/) | Default (CI green-bar) | Conservative-remapping geometry, conservation, direction, and persistence checks. Its `runtests.jl` is included as one isolated suite by the top-level orchestrator. |

## Usage

```bash
julia --project=. -e 'using Pkg; Pkg.test()'         # CI-equivalent default
julia --project=test test/runtests.jl                # core + regridding
julia --project=test test/runtests.jl --all          # default + real data + diagnostic
julia --project=test test/runtests.jl --diagnostic   # default + diagnostic
julia --project=test test/runtests.jl --real-data    # default + real data
julia --project=test test/runtests.jl --orphan       # default + orphan watchlist
julia --project=test test/runtests.jl --tiers=core,orphan # only listed tiers
```

## Adding a new test

1. Decide which tier the test belongs in. Default to `core/` if the test
   doesn't need external data and is part of a production code path.
2. Drop the file into the matching folder. The orchestrator picks up
   anything matching `test_*.jl` automatically — no manual roster edit.
3. Test files are included into anonymous modules, so they can `using
   .AtmosTransport` freely without polluting each other.
4. Use `joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl")` (note
   the **two** `..`) when including `AtmosTransport.jl` directly — the
   extra hop accounts for the tier subfolder.

## Promotion / retirement workflow

- An `orphan/` file becomes promotable when it loads cleanly under the
  current `src/`, asserts something the team wants to keep regressing,
  and is not redundant with an existing core test. Move it into `core/`
  in the same commit that fixes any path adjustments.
- A `core/` file becomes archivable when the code path it tests is
  deleted or has been fully subsumed by a newer test. Move it into
  `archived/` with a note in `archived/legacy_README.md`.

Core tests load the installed development package with `using AtmosTransport`
or `import AtmosTransport`. The runner isolates each file's test helpers in its
own module while reusing Julia's package cache. Avoid including the package
source separately in each core test: that repeats compilation and gives each
copy distinct type identities.
