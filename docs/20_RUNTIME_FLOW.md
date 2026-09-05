# Runtime Flow

This walkthrough follows the current driven runtime from a meteorology window
through one transport step. Start with
[`DrivenSimulation.jl`](../src/Models/DrivenSimulation.jl) for forcing and timing,
and [`TransportModel.jl`](../src/Models/TransportModel.jl) for operator ordering.

## Ownership

| Quantity or resource | Owner |
| --- | --- |
| Input reader, grid metadata, window duration and step schedule | `sim.driver`, a `TransportBinaryDriver` or `CubedSphereTransportDriver` |
| Current forcing and next-window prefetch buffer | `sim.window` and `sim.prefetch_window` |
| Prognostic air and tracer masses, operators and numerical workspaces | `sim.model` |
| Time, window index, iteration count and callbacks | `sim`, the `DrivenSimulation` |
| Input/output task cleanup and optional file staging | `run_driven_simulation` and its resource scopes |

Driver windows contain meteorology, not prognostic tracer state. The model
copies their forcing into the buffers used by its operators. Tracer storage
and its mass basis are described in [the core contracts](10_CORE_CONTRACTS.md).

## Window loading and prefetch

The constructor validates grid geometry, mass basis, forcing capabilities and
the requested window range. It reads the first window once. On a GPU, with
multiple Julia threads and another window to process, it creates two independent
device windows from that payload. A custom driver that returns device arrays
gets an explicit copy for the second buffer. CPU and single-window runs share
the initial window reference because they do not start a prefetch task.

The background task loads the next window into the spare device buffer while
transport uses the current one. At the boundary, `_take_prefetched_window!`
waits for that task and exchanges the two buffers. Without prefetch, the same
boundary loads the window synchronously. `ATMOSTR_DISABLE_PREFETCH=1` disables
prefetch for diagnosis or comparison.

Window advancement updates the step count and `Δt`, applies the configured
air-mass reset policy, refreshes diffusion geometry/forcing, invalidates
convection caches, and schedules the following prefetch. Installing the first
window of a new driver also invalidates convection caches. The multi-file
runner retains model state and numerical workspaces across compatible input
files while advancing the run clock continuously.

The runner drains prefetch before closing its input reader, including on
failure. A prefetch failure already observed at the window boundary is consumed
once, so cleanup does not report the same task error again. Users who construct a `DrivenSimulation` directly must keep its driver
open while it is running.

## One simulation step

`step!(sim)` performs the following sequence:

1. Advance to the next meteorology window if the previous window is complete.
2. Determine the substep within that window and refresh model forcing.
3. Apply the transport block with `dt = sim.Δt`. For a driver without the binary
   scheduling contract, apply convection and chemistry at this cadence too.
4. Advance `sim.time` and `sim.iteration`.
5. For a binary-scheduled driver, at the window boundary, apply the configured
   endpoint air-mass reset and then convection and chemistry with
   `dt = sim.window_dt`.
6. Call each callback with the updated simulation.

`run_window!(sim)` repeats this sequence to a meteorology-window boundary;
`run!(sim)` repeats it through the requested final window. Window-specific step
counts come from the driver, so `Δt` can change between windows. The stored
substep schedule controls transport resolution, not the number of convection
applications per hour.

`ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS=1` is a diagnostic override that applies
convection and chemistry at each transport substep even for binary-scheduled
drivers. It is off by default and changes the physical splitting cadence.

## Forcing refresh and time

`_refresh_forcing!(sim, substep)` prepares the arrays consumed by the operators:

- Mass fluxes are copied for window-constant forcing, or interpolated when the
  driver's contract requests interpolation. Full-window stored mass amounts
  are scaled by `1 / (2 * steps_per_window)` for the transport palindrome.
- Expected air mass is evaluated from the window and its optional
  mass deltas. Available humidity endpoints update `sim.qv_buffer`.
- Active convection receives a copy of the window's CMFMC/DTRAIN or TM5 fields.
  These fields are window forcing; they are not automatically interpolated
  between neighboring windows.

For more than one substep, the default interpolation fraction is
`(substep - 0.5) / steps_per_window`; disabling midpoint forcing uses
`(substep - 1) / steps_per_window`. A one-substep window uses fraction zero.
This interpolation fraction is distinct from the run clock used by time-varying
surface fluxes and diffusivity fields.

Operators receive `meteo = sim`. `current_time(sim)` returns seconds since the
run start, including preceding input files. A bare driver has no evolving clock;
its compatibility method `current_time(driver)` returns zero. Callbacks see time
after the step and any due window-boundary physics have completed.

## Operator blocks

`transport_step!` dispatches advection on the model's mesh and scheme, passing
its diffusion and surface-emission operators and their workspaces into the
transport splitting. The main implementations are
[`StrangSplitting.jl`](../src/Operators/Advection/StrangSplitting.jl) for
latitude–longitude and reduced-Gaussian meshes, and
[`CubedSphereStrang.jl`](../src/Operators/Advection/CubedSphereStrang.jl) for
panel-based transport. Horizontal CS sweeps exchange panel halos. Diffusion
and emissions participate in this transport block, so their cadence follows
transport substeps.

`convection_chemistry_step!` applies convection and then chemistry. Supported
convection families are CMFMC, TM5 matrix convection, and CMFMC-derived matrix
convection. Their forcing and solver contracts are documented in
[the convection guide](../src/Operators/Convection/README.md). The collaborative
matrix solver's six shared-memory tracer slots are reused across batches;
they do not impose a six-tracer limit.

Chemistry dispatches on the configured operator, including exponential decay
and composite operators. No-op operator types disable their corresponding
blocks. See [the topology support table](../src/Operators/TOPOLOGY_SUPPORT.md)
for supported mesh/scheme combinations and
[the chemistry guide](../src/Operators/Chemistry/README.md) for its contract.

## Mass basis and continuity

Preprocessing produces carrier-mass and flux fields with an explicit mass
basis and continuity contract. Dry basis is the default; the runtime checks
that driver, state and flux bases agree. It does not repair an inconsistent
binary by redoing preprocessing continuity closure. Runtime endpoint resets
and flux-storage scaling serve the declared transport contract and do not
replace that preprocessing step.

See [binary and driver contracts](30_BINARY_AND_DRIVERS.md),
[runtime stability and subcycling](35_RUNTIME_STABILITY_AND_SUBCYCLING.md), and
[the architecture reference](reference/ARCHITECTURE.md).

## Optional run instrumentation

`ATMOSTR_TIMERS=1` enables host section timings; `ATMOSTR_ALLOC_TIMERS=1`
adds host allocation counts when timing is enabled. `ATMOSTR_NVTX=1` enables
GPU profiling ranges independently when the NVTX extension is loaded.
Instrumentation enabled by the runner stops on every exit, including input
inspection, setup, stepping, and cleanup failures. Completed samples remain
available through `AtmosTransport.SectionTimer.report()` after an error.
Successful runs print the report and write a `.timings.csv` beside configured
snapshot output. Failed runs propagate the error without writing a timing CSV.
