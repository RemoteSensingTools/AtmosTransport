# Unified Transport-Binary Generation

This note is the production contract for every preprocessor that writes
transport binaries. The source may be ERA5 spectral fields, native GEOS/GCHP
fields, or a future met reader; the target may be lat-lon, reduced Gaussian, or
cubed sphere. Those choices change only the dispatched methods. They do not
change the lifecycle.

## One Lifecycle

Every binary-producing path must follow this sequence:

1. Build a typed source reader and a typed target geometry.
2. Allocate a target workspace for one day of windows.
3. Ingest each source met window into target air mass and flux buffers.
4. Choose the integer `steps_per_window[win]` from the shared
   `SubstepSchedulePolicy`.
5. Rescale stored per-substep flux amounts when the selected step count changes.
6. Balance the horizontal and vertical mass closure against the explicit window
   endpoint using that selected step count.
7. Verify the topology contract: replay continuity plus per-substep positivity.
8. Write the payload and patch the header schedule before close.
9. Load-time readers reject any binary whose header schedule, Poisson scale, and
   payload semantics disagree.

The generic lifecycle is `run_unified_preprocessor_day!`. A topology supplies
only the methods that are physically topology-specific:

```julia
driver_windows_per_day(reader, context)
driver_ingest_window!(workspace, reader, win, context)
driver_drain_ready_windows!(workspace, contract, win, context)
driver_flush_final_windows!(workspace, reader, contract, context)
driver_before_close_writer!(workspace, reader, contract, writer, context)
driver_after_write_window!(workspace, reader, ready, context)
```

If a new source or topology cannot implement one of these methods, the missing
method should fail at dispatch or construction time. It should not silently use
a parallel script with different header semantics.

## Substep Schedule Semantics

`format_version = 3` stores fluxes as per-substep mass amounts:

```text
flux_kind = "substep_mass_amount"
m_next - m_cur = 2 * steps_per_window[win] * div(stored_half_sweep_fluxes)
```

Therefore changing `steps_per_window[win]` requires changing the payload:

```text
new_flux = old_flux * old_steps / new_steps
cm       = recomputed from (m_next - m_cur) / (2 * new_steps)
```

Header-only schedule edits are invalid for v3. A future binary version may store
full-window mass amounts or rates and divide by the schedule at runtime, but that
would be a different on-disk contract and must bump `format_version`.

## Shared Policy

All paths use `SubstepSchedulePolicy`:

```julia
SubstepSchedulePolicy(
    adaptive_substeps = true,
    substep_cfl_target = 0.95,
    min_steps_per_window = 1,
    max_steps_per_window = 64,
)
```

The selected step count is the smallest integer that brings the measured
positivity ratio under `substep_cfl_target`, clamped to the configured bounds.
The schedule is not restricted to powers of two.

The default TOML behavior is adaptive CFL scheduling. A fixed schedule is an
explicit opt-out:

```toml
[numerics]
substep_schedule = "fixed"
```

Fixed schedules still run the final positivity contract. If they fail, the
binary is quarantined unless the diagnostic escape hatch is explicitly enabled.

## Topology Responsibilities

Lat-lon stores directional structured fluxes. It chooses a per-window schedule,
rescales `am` and `bm`, recomputes `cm`, and rewrites the deferred header before
the first payload byte is committed.

Reduced Gaussian stores face-indexed horizontal fluxes. It chooses a per-window
schedule in the sliding-buffer drain step, rescales `hflux`, recomputes `cm`,
and patches the streaming header before close.

Cubed sphere stores per-panel directional fluxes. Native GEOS and ERA5-spectral
CS both choose a per-window schedule, rescale `am` and `bm`, recompute `cm`, and
write `runtime_substep_contract = "binary_schedule"` so the runtime honors the
baked schedule instead of re-piloting CS CFL.

## Contract Failure Modes

These are hard errors:

- Missing or obsolete `format_version`.
- Missing `steps_per_window_by_window`.
- `steps_per_window != maximum(steps_per_window_by_window)`.
- `poisson_balance_target_scale_by_window[win] != 1 / (2 * steps_per_window[win])`.
- Header declares a binary schedule that the runtime cannot honor.
- Replay continuity failure after balancing.
- Per-substep positivity failure after adaptive refinement reaches its ceiling.

These rules make old binaries obsolete and make partial regeneration failures
obvious at preprocessing time, not during a transport campaign.
