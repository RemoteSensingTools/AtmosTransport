# Vertical Transport Debug Path (Single Met File, Sub-Step Tracking)

## Goal

Establish a deterministic, reproducible validation workflow for cubed-sphere transport that:

1. Uses only the first meteorology file.
2. Tracks behavior at each advection sub-step.
3. Starts from synthetic dry VMR initial conditions with no emissions.
4. Detects where unphysical vertical mixing is introduced (especially rapid transport toward ~200 hPa with horizontal extrema).

## Critical setup rules

1. Use one met file, but keep at least 2 windows for vertical-remap tests.
   - In `advection_phase!`, the remap path uses `has_next` to build target pressure from next-window DELP.
   - If only 1 window is present, remap falls back to identity (`target = source`) and will not exercise the true remap transition.
2. Disable all non-transport processes.
   - `emission = "none"`, `chemistry.type = "none"`, no convection, no diffusion.
3. Use a single tracer for debugging.
   - Start with `co2` only, then add additional tracer cases once baseline behavior is understood.
4. Keep dry-basis consistency.
   - Use `dry_correction = true` for dry tests unless explicitly testing wet-vs-dry differences.

## Phase 1: Minimal deterministic cases

Create dedicated debug configs (copy from existing C180 configs) so runs are short and controlled.

Suggested base settings:

- Grid/met: GEOS-IT C180 (`product = "geosit_c180"`), same `dt = 300`, `met_interval = 3600`, `mass_flux_dt = 450`.
- Data span: one day (`start_date = end_date = first available date`) to keep to first file.
- Transport scheme: `ppm_order = 7`.
- Outputs: high frequency for debugging (`interval = 300` or one output per sub-step window stage if practical).

Run matrix for the same IC and same met inputs:

1. `vertical_remap = false`, `mass_fixer = false`, `dry_correction = true` (baseline Strang path).
2. `vertical_remap = true`, `remap_pressure_fix = false`, `mass_fixer = false`, `dry_correction = true`.
3. `vertical_remap = true`, `remap_pressure_fix = true`, `mass_fixer = false`, `dry_correction = true`.
4. Optional sensitivity: repeat (2) and (3) with `dry_correction = false`.

This isolates:

- effect of remap itself,
- effect of `fix_target_bottom_pe!`,
- dry/wet basis mismatch effects.

## Phase 2: Initial-condition test suite

### Case A: Uniform dry VMR invariant test

- IC: uniform `co2.uniform_value = 400e-6`.
- Expectation: field remains spatially uniform at all times.
- Any growth in horizontal range or vertical structure indicates numerical inconsistency/bug.

### Case B: Surface-enhanced tracer

- IC: synthetic file with elevated VMR in bottom 1-2 model layers only.
- Expectation: physically plausible upward transport rate; no immediate jump to upper troposphere in a few sub-steps.

### Case C: TOA-enhanced tracer

- IC: synthetic file with elevated VMR in top 1-2 layers only.
- Expectation: mirror behavior of Case B (downward transport limited by resolved dynamics and scheme).

## Phase 3: Sub-step instrumentation (required)

Instrument diagnostics inside the CS `advection_phase!` loop and around remap boundaries.

Capture stats at:

1. Start of each sub-step.
2. After first Lin-Rood half-step.
3. After second Lin-Rood half-step.
4. Before vertical remap call.
5. After vertical remap call.
6. After `update_air_mass_from_target!`.

For non-remap path, capture:

1. Start of each sub-step.
2. After `_apply_advection_cs!`.
3. After optional mass fixer.

Per snapshot, compute and log:

- global tracer mass,
- global VMR min/max/mean,
- per-level VMR min/max/mean,
- per-level tracer mass,
- fraction of tracer mass above 200 hPa,
- horizontal standard deviation at selected levels (near surface, mid-trop, upper trop),
- count of negative mass or negative VMR cells (should be zero).

Write to compact CSV/NetCDF diagnostics file keyed by:

- `window`, `substep`, `stage`, `k` (level), and metric columns.

## Phase 4: Acceptance thresholds

### Uniform case (Case A)

- `max(VMR) - min(VMR)` stays near machine noise and does not grow systematically.
- no level-dependent bias appears over sub-steps.
- tracer mass conserved within tight floating-point tolerance.

### Gradient cases (B/C)

- vertical center-of-mass evolves smoothly, not in large single-step jumps.
- no spurious new horizontal extrema at remote levels.
- mass above 200 hPa (for surface-enhanced case) increases gradually, not explosively within one window.

Use baseline run (no remap) as reference and flag remap runs if error metrics exceed baseline by large factors.

## Phase 5: Debug decision tree

1. If Case A fails with `vertical_remap = false`:
   - bug is outside remap (horizontal transport, air-mass handling, or diagnostics).
2. If Case A passes without remap but fails with remap:
   - focus on target pressure construction and remap kernels.
3. If remap fails only when `remap_pressure_fix = true`:
   - focus on `fix_target_bottom_pe!` scaling path.
4. If remap fails even with `remap_pressure_fix = false`:
   - focus on dry target PE construction (`next DELP` + QV basis), sub-step mass reset logic, and remap integration details.

## Implementation notes for Claude

1. Start from existing uniform/no-emission configs and add dedicated debug variants rather than modifying production configs.
2. Add instrumentation behind a debug flag in config metadata to avoid runtime cost in normal runs.
3. Keep diagnostics GPU-side and transfer only reduced statistics to CPU to avoid heavy I/O.
4. After Case A is stable, run Cases B/C to evaluate physical plausibility of vertical transport rates.

## Immediate first run recommendation

1. Run Case A with matrix items (1)-(3) above.
2. Analyze first two windows only (enough to trigger true remap with `has_next = true`).
3. Compare per-substep diagnostics, especially:
   - first point where uniformity breaks,
   - first point where above-200 hPa mass deviates.

This will localize whether the remaining over-mixing comes from:

- remap pressure target construction,
- pressure-fix scaling,
- or non-remap transport logic.
