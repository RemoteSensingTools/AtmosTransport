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
julia --project=test/atmoschemistry test/atmoschemistry/runtests.jl # CPU/CUDA chemistry extension
```

`test/atmoschemistry/` is a dedicated weak-dependency integration
environment. It makes both `AtmosChemistry` and CUDA available without adding
either package to the CPU-only default test target. The extension test must be
run in this environment; a skip in the ordinary core environment is not a
substitute for this gate.

## Adding a new test

1. Decide which tier the test belongs in. Default to `core/` if the test
   doesn't need external data and is part of a production code path.
2. Drop the file into the matching folder. The orchestrator picks up
   anything matching `test_*.jl` automatically — no manual roster edit.
3. Test files run in anonymous modules, keeping fixture names isolated.
4. Use `using AtmosTransport`; share the precompiled package instead of
   including its entire source separately in each test file.

## Promotion / retirement workflow

- An `orphan/` file becomes promotable when it loads cleanly under the
  current `src/`, asserts something the team wants to keep regressing,
  and is not redundant with an existing core test. Move it into `core/`
  in the same commit that fixes any path adjustments.
- A `core/` file becomes archivable when the code path it tests is
  deleted or has been fully subsumed by a newer test. Move it into
  `archived/` with a note in `archived/legacy_README.md`.

The NetCDF output contract suite was promoted from `orphan/` to `core/` on
2026-09-04, alongside selected-capture parity tests for LL, RG, and CS.

A dedicated A100 regression is available outside the automatic tiers:

```bash
CUDA_VISIBLE_DEVICES=0 julia --project=benchmarking test/a100/test_review_a100.jl
```

It requires the A100 and checks Float32/Float64 CPU-CUDA transport parity and
selected snapshot capture with GPU scalar indexing disabled. For a strictly
CPU local suite, set `CUDA_VISIBLE_DEVICES=''` so optional CUDA tests cannot
select another GPU through the global Julia environment.
