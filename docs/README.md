# AtmosTransport Documentation

Primary documentation for the basis-explicit transport architecture (`src/`, promoted from `src_v2/`).

## Structure

```
docs/
  AGENT_ONBOARDING.md       -- Quick-start for AI agents (read second, after CLAUDE.md)
  README.md                 -- This file
  0x_*.md                   -- Core architecture docs (numbered for reading order)
  3x_*.md                   -- Bug analyses and fixes (dated)
  reference/                -- Shared reference docs (apply to both legacy and new code)
    ARCHITECTURE.md
    BINARY_FORMAT.md
    DATA_LAYOUT.md
    QUICKSTART.md
    ...
  memos/                    -- Design memos and debugging analyses
    DESIGN_MEMO_*.md
    advection_kernel_refactor_memo_update.md
    ...

docs_legacy/                -- Legacy Documenter.jl site + session logs for src_legacy/
  make.jl                   -- Documenter build (references src_legacy/AtmosTransport.jl)
  src/                      -- Literate.jl + developer docs
  SESSION*.md               -- Historical debugging session logs
```

## Reading Order

For a new contributor or agent:

1. **CLAUDE.md** (repo root) -- Project rules, invariants, critical constraints
2. **AGENT_ONBOARDING.md** -- Directory map, healthy reference values, pitfalls
3. Core contracts:
   - [00_SCOPE_AND_STATUS.md](00_SCOPE_AND_STATUS.md)
   - [10_CORE_CONTRACTS.md](10_CORE_CONTRACTS.md)
   - [20_RUNTIME_FLOW.md](20_RUNTIME_FLOW.md)
   - [30_BINARY_AND_DRIVERS.md](30_BINARY_AND_DRIVERS.md)
4. Stability and quality:
   - [35_RUNTIME_STABILITY_AND_SUBCYCLING.md](35_RUNTIME_STABILITY_AND_SUBCYCLING.md)
   - [40_QUALITY_GATES.md](40_QUALITY_GATES.md)
5. On-demand (when hitting a specific topic):
   - [reference/](reference/) -- Shared reference docs
   - [memos/](memos/) -- Design memos and analyses

## Key Reference Docs

| Topic | File |
|-------|------|
| Data folder conventions | [reference/DATA_LAYOUT.md](reference/DATA_LAYOUT.md) |
| Binary format spec | [reference/BINARY_FORMAT.md](reference/BINARY_FORMAT.md) |
| Quick-start guide | [reference/QUICKSTART.md](reference/QUICKSTART.md) |
| Architecture overview | [reference/ARCHITECTURE.md](reference/ARCHITECTURE.md) |
| Preprocessing pipeline | [reference/PREPROCESSING_PHILOSOPHY.md](reference/PREPROCESSING_PHILOSOPHY.md) |
| Conservative regridding | [reference/CONSERVATIVE_REGRIDDING.md](reference/CONSERVATIVE_REGRIDDING.md) |
| Transport comparison (TM5/GCHP) | [reference/TRANSPORT_COMPARISON.md](reference/TRANSPORT_COMPARISON.md) |

## Key Design Memos

| Topic | File |
|-------|------|
| Basis-explicit transport design | [memos/DESIGN_MEMO_BASIS_EXPLICIT_TRANSPORT.md](memos/DESIGN_MEMO_BASIS_EXPLICIT_TRANSPORT.md) |
| Advection kernel refactor | [memos/advection_kernel_refactor_memo_update.md](memos/advection_kernel_refactor_memo_update.md) |
| Global mean ps fix | [memos/GLOBAL_MEAN_PS_FIX.md](memos/GLOBAL_MEAN_PS_FIX.md) |
| RG instability root cause | [memos/MEMO_REDUCED_GAUSSIAN_INSTABILITY_2026-04-10.md](memos/MEMO_REDUCED_GAUSSIAN_INSTABILITY_2026-04-10.md) |
| Cross-grid validation plan | [memos/PLAN_24H_CROSS_GRID_VALIDATION.md](memos/PLAN_24H_CROSS_GRID_VALIDATION.md) |

## Quick Inspection

```bash
julia --project=. scripts/diagnostics/inspect_transport_binary.jl path/to/file.bin
```
