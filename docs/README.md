# AtmosTransport Documentation

Primary documentation for the basis-explicit transport architecture in
[`../src/`](../src/).

## Structure

```
docs/
  README.md                  -- This file
  AGENT_ONBOARDING.md        -- Quick-start for AI agents (read second)
  00_SCOPE_AND_STATUS.md     -- What src/ is, what ships today
  10_CORE_CONTRACTS.md       -- State / flux / driver contracts
  20_RUNTIME_FLOW.md         -- End-to-end step! walkthrough
  30_BINARY_AND_DRIVERS.md   -- Transport binary format + driver API
  35_RUNTIME_STABILITY_AND_SUBCYCLING.md
  36_POISSON_BALANCE_TARGET_BUG_2026-04-09.md
  37_WINDOW_CONSTANT_FLUX_INTERPRETATION_BUG_2026-04-09.md
  38_REDUCED_GAUSSIAN_THREADED_PREPROCESS_BUG_2026-04-09.md
  40_QUALITY_GATES.md
  plans/                     -- Plan history + reference design docs
    PLAN_HISTORY.md            -- Canonical manifest (what shipped when)
    OPERATOR_COMPOSITION.md    -- Operator block-ordering contract
    TIME_VARYING_FIELD_MODEL.md
    ARCHITECTURAL_SKETCH_v3.md
  reference/                 -- Shared reference docs (data layouts, APIs)
  memos/                     -- Design memos and debugging analyses
  resources/                 -- Archival material (bug archive, dev notes)
```

## Reading Order

For a new contributor or agent:

1. **`CLAUDE.md`** (repo root) — Project rules, invariants, critical constraints
2. **[`AGENT_ONBOARDING.md`](AGENT_ONBOARDING.md)** — Directory map, type hierarchy, healthy reference values
3. **[`../src/README.md`](../src/README.md)** — Runtime entry points
4. **[`../src/Operators/TOPOLOGY_SUPPORT.md`](../src/Operators/TOPOLOGY_SUPPORT.md)** — Canonical operator × topology matrix
5. **Core contracts:**
   - [`00_SCOPE_AND_STATUS.md`](00_SCOPE_AND_STATUS.md)
   - [`10_CORE_CONTRACTS.md`](10_CORE_CONTRACTS.md)
   - [`20_RUNTIME_FLOW.md`](20_RUNTIME_FLOW.md)
   - [`30_BINARY_AND_DRIVERS.md`](30_BINARY_AND_DRIVERS.md)
6. **Plan history:** [`plans/PLAN_HISTORY.md`](plans/PLAN_HISTORY.md) — what shipped, what's in flight
7. **Stability and quality:**
   - [`35_RUNTIME_STABILITY_AND_SUBCYCLING.md`](35_RUNTIME_STABILITY_AND_SUBCYCLING.md)
   - [`40_QUALITY_GATES.md`](40_QUALITY_GATES.md)
8. **On-demand (when hitting a specific topic):**
   - [`reference/`](reference/) — Shared reference docs
   - [`memos/`](memos/) — Design memos and analyses

## Key Reference Docs

| Topic | File |
|-------|------|
| Data folder conventions | [`reference/DATA_LAYOUT.md`](reference/DATA_LAYOUT.md) |
| Binary format spec | [`reference/BINARY_FORMAT.md`](reference/BINARY_FORMAT.md) |
| Quick-start guide | [`reference/QUICKSTART.md`](reference/QUICKSTART.md) |
| Architecture overview | [`reference/ARCHITECTURE.md`](reference/ARCHITECTURE.md) |
| Preprocessing pipeline | [`reference/PREPROCESSING_PHILOSOPHY.md`](reference/PREPROCESSING_PHILOSOPHY.md) |
| Diagnostic NetCDF output | [`reference/OUTPUT.md`](reference/OUTPUT.md) |
| Snapshot visualization | [`reference/VISUALIZATION.md`](reference/VISUALIZATION.md) |
| Conservative regridding | [`reference/CONSERVATIVE_REGRIDDING.md`](reference/CONSERVATIVE_REGRIDDING.md) |
| Transport comparison (TM5/GCHP) | [`reference/TRANSPORT_COMPARISON.md`](reference/TRANSPORT_COMPARISON.md) |

## Key Design Memos

| Topic | File |
|-------|------|
| Basis-explicit transport design | [`memos/DESIGN_MEMO_BASIS_EXPLICIT_TRANSPORT.md`](memos/DESIGN_MEMO_BASIS_EXPLICIT_TRANSPORT.md) |
| Advection kernel refactor | [`memos/advection_kernel_refactor_memo_update.md`](memos/advection_kernel_refactor_memo_update.md) |
| Global mean ps fix | [`memos/GLOBAL_MEAN_PS_FIX.md`](memos/GLOBAL_MEAN_PS_FIX.md) |
| RG instability root cause | [`memos/MEMO_REDUCED_GAUSSIAN_INSTABILITY_2026-04-10.md`](memos/MEMO_REDUCED_GAUSSIAN_INSTABILITY_2026-04-10.md) |
| Cross-grid validation plan | [`memos/PLAN_24H_CROSS_GRID_VALIDATION.md`](memos/PLAN_24H_CROSS_GRID_VALIDATION.md) |

## Quick Inspection

```bash
julia --project=. scripts/diagnostics/inspect_transport_binary.jl path/to/file.bin
```
