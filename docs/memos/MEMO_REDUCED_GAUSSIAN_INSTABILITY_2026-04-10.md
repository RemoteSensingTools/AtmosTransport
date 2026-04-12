# Reduced-Gaussian Instability Memo

Date: 2026-04-10

## Goal

Debug the `src` reduced-Gaussian transport run produced by:

- [config/runs/era5_reduced_gaussian_transport_v2_dec2021.toml](/home/cfranken/code/gitHub/AtmosTransportModel/config/runs/era5_reduced_gaussian_transport_v2_dec2021.toml)
- [scripts/run_transport_binary_v2.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/run_transport_binary_v2.jl)
- [scripts/diagnostics/export_transport_v2_snapshots.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/diagnostics/export_transport_v2_snapshots.jl)

The symptom was that the reduced-grid snapshot file looked bad in the panel PNG and then went catastrophically unstable by `t = 12 h`.

## High-signal findings

1. The reduced-grid initial condition is not the problem.
   The lat-lon and reduced configs use the same `gaussian_blob` parameters, and the reduced `t = 0 h` file has the expected background and peak.

2. The original reduced run really is unstable, not just visually rough.
   Existing snapshot diagnostics showed:
   - `t = 0 h`: `min = 4.000e-04`, `max = 7.999e-04`
   - `t = 6 h`: `min = 2.345e-04`, `max = 7.994e-04`
   - `t = 12 h`: `min = -2.629e+10`, `max = 2.246e+12`

3. The first reduced-vs-latlon spatial mismatch appears well before the blow-up.
   Using the diagnostic script below on the existing snapshot files:
   - `t = 0 h`: equatorial / meridional cut mismatch only a few `1e-6`
   - `t = 6 h`: equatorial mismatch already `4.790e-01`, meridional mismatch already `1.133e+03`

4. The face-indexed reduced-grid sweeps were over-draining cells from the start.
   Diagnostic output for the real reduced binary, window 1:
   - combined horizontal outgoing ratio: `2.856097`
   - zonal-only outgoing ratio: `2.233619`
   - meridional-only outgoing ratio: `1.169701`
   - vertical outgoing ratio: `0.768562`

5. After adding face-indexed subcycling, the within-window reduced run stayed monotone through window 1.
   With the patch in [src/Operators/Advection/StrangSplitting.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src/Operators/Advection/StrangSplitting.jl), the reduced run stayed at `qmin = 4.000e-04` through `t = 1.00 h`.

6. The first strong undershoot was then traced to the per-window air-mass hard reset.
   With the patched subcycling path:
   - `qmin` after window 1 transport: `3.9999999999996283e-4`
   - `qmin` immediately after copying in window 2 air mass, before any window 2 advection: `3.1316999766128087e-4`

7. Disabling the per-window air-mass reset removes the early reduced undershoot.
   With `reset_air_mass_each_window = false`, the reduced step trace stayed at:
   - `t = 1.25 h`: `qmin = 4.000e-04`
   - `t = 1.50 h`: `qmin = 4.000e-04`

## Code changes already made

1. Added face-indexed CFL subcycling to the reduced-grid sweeps in:
   - [src/Operators/Advection/StrangSplitting.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src/Operators/Advection/StrangSplitting.jl)

2. Added a reusable diagnostic script:
   - [scripts/diagnostics/debug_reduced_gaussian_instability.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/diagnostics/debug_reduced_gaussian_instability.jl)

3. Changed the `transport_binary_v2` harnesses to default to `reset_air_mass_each_window = false` unless the config overrides it:
   - [scripts/run_transport_binary_v2.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/run_transport_binary_v2.jl)
   - [scripts/diagnostics/export_transport_v2_snapshots.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/diagnostics/export_transport_v2_snapshots.jl)

4. Made the reduced and lat-lon comparison configs explicit about no-reset:
   - [config/runs/era5_reduced_gaussian_transport_v2_dec2021.toml](/home/cfranken/code/gitHub/AtmosTransportModel/config/runs/era5_reduced_gaussian_transport_v2_dec2021.toml)
   - [config/runs/era5_latlon_transport_v2_dec2021.toml](/home/cfranken/code/gitHub/AtmosTransportModel/config/runs/era5_latlon_transport_v2_dec2021.toml)

5. Added regression coverage:
   - [test/test_basis_explicit_core.jl](/home/cfranken/code/gitHub/AtmosTransportModel/test/test_basis_explicit_core.jl)
   - [test/test_driven_simulation.jl](/home/cfranken/code/gitHub/AtmosTransportModel/test/test_driven_simulation.jl)

6. Added file-based IC loading to the `src` transport-binary runner, including the CATRINE `startCO2` shortcut:
   - [scripts/run_transport_binary_v2.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/run_transport_binary_v2.jl)
   - Supports `init.kind = "file"` with `init.file` + `init.variable`
   - Supports `init.kind = "catrine_co2"` as shorthand for `~/data/AtmosTransport/catrine/InitialConditions/startCO2_202112010000.nc`
   - Uses the same legacy coordinate normalization as `src/IO/initial_conditions.jl`
   - Uses pressure-based vertical interpolation when the source file carries `ap/bp/Psurf`

7. Made the `TransportBinaryDriver` accept legacy delta-bearing lat-lon binaries that are missing Poisson metadata but otherwise match the expected runtime semantics:
   - [src/MetDrivers/TransportBinaryDriver.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src/MetDrivers/TransportBinaryDriver.jl)
   - The current December 2021 lat-lon binary is `format_version = 1`, `flux_sampling = :window_start_endpoint`, `delta_semantics = :forward_window_endpoint_difference`, `steps_per_window = 4`, but `poisson_balance_target_scale = NaN`
   - The driver now infers the expected scale `1 / (2 * steps_per_window)` for this narrow legacy case and emits a warning instead of throwing

8. Added explicit CATRINE `t = 0` snapshot configs:
   - [config/runs/era5_latlon_transport_v2_catrine_t0_dec2021.toml](/home/cfranken/code/gitHub/AtmosTransportModel/config/runs/era5_latlon_transport_v2_catrine_t0_dec2021.toml)
   - [config/runs/era5_reduced_transport_v2_catrine_t0_dec2021.toml](/home/cfranken/code/gitHub/AtmosTransportModel/config/runs/era5_reduced_transport_v2_catrine_t0_dec2021.toml)

9. Added optional CUDA backend support to the standalone `src` transport-binary runner/export path:
   - [scripts/run_transport_binary_v2.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/run_transport_binary_v2.jl)
   - [scripts/diagnostics/export_transport_v2_snapshots.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/diagnostics/export_transport_v2_snapshots.jl)
   - [src/Models/DrivenSimulation.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/DrivenSimulation.jl)
   - New behavior:
     - honor `[architecture] use_gpu = true`
     - lazily load CUDA on Linux hosts
     - adapt the `TransportModel` state/flux/workspace to `CuArray`
     - adapt each loaded transport window onto the same backend as the model
     - log backend label and elapsed runtime
   - Verified on this host:
     - `CUDA.functional() == true`
     - device name: `NVIDIA L40S`
     - with `ensure_gpu_runtime!` called before model construction, the reduced-grid model and window arrays are all `CuArray{Float64,2,CUDA.DeviceMemory}`

## Tests already passing

- `julia --project=. test/test_driven_simulation.jl`
- `julia --project=. test/test_run_transport_binary_v2.jl`
- `julia --project=. test/test_basis_explicit_core.jl`
- `julia --project=. test/test_transport_binary_reader.jl`

## Useful commands

Step-by-step reduced trace with default config behavior:

```bash
julia --project=. scripts/diagnostics/debug_reduced_gaussian_instability.jl \
  step-trace config/runs/era5_reduced_gaussian_transport_v2_dec2021.toml 12
```

Step-by-step reduced trace with no per-window air-mass reset:

```bash
AT_DEBUG_RESET_AIR_MASS_EACH_WINDOW=false \
  julia --project=. scripts/diagnostics/debug_reduced_gaussian_instability.jl \
  step-trace config/runs/era5_reduced_gaussian_transport_v2_dec2021.toml 12
```

Outgoing-mass ratios for the real reduced binary:

```bash
julia --project=. scripts/diagnostics/debug_reduced_gaussian_instability.jl \
  outgoing-cfl config/runs/era5_reduced_gaussian_transport_v2_dec2021.toml 6
```

Lat-lon vs reduced snapshot cut comparison:

```bash
julia --project=. scripts/diagnostics/debug_reduced_gaussian_instability.jl \
  compare-snapshots \
  ~/data/AtmosTransport/output/src_v2_snapshots/era5_latlon_transport_v2_dec2021.nc \
  ~/data/AtmosTransport/output/src_v2_snapshots/era5_reduced_gaussian_transport_v2_dec2021.nc
```

Create the CATRINE `t = 0` snapshot exports:

```bash
julia --project=. scripts/diagnostics/export_transport_v2_snapshots.jl \
  config/runs/era5_latlon_transport_v2_catrine_t0_dec2021.toml

julia --project=. scripts/diagnostics/export_transport_v2_snapshots.jl \
  config/runs/era5_reduced_transport_v2_catrine_t0_dec2021.toml
```

Compare the two CATRINE `t = 0` snapshot files:

```bash
julia --project=. scripts/diagnostics/debug_reduced_gaussian_instability.jl \
  compare-snapshots \
  ~/data/AtmosTransport/output/src_v2_snapshots/era5_latlon_transport_v2_catrine_t0_dec2021.nc \
  ~/data/AtmosTransport/output/src_v2_snapshots/era5_reduced_transport_v2_catrine_t0_dec2021.nc
```

Regenerate the Gaussian `0 h / 6 h` lat-lon snapshot file:

```bash
julia --project=. -e '
using TOML
include("scripts/diagnostics/export_transport_v2_snapshots.jl")
cfg = TOML.parsefile("config/runs/era5_latlon_transport_v2_dec2021.toml")
cfg["run"]["stop_window"] = 6
cfg["output"] = Dict(
    "snapshot_hours" => Any[0, 6],
    "snapshot_file" => expanduser("~/data/AtmosTransport/output/src_v2_snapshots/era5_latlon_transport_v2_gaussian_6h_dec2021.nc"),
)
binary_paths = [expanduser(String(p)) for p in cfg["input"]["binary_paths"]]
println(export_snapshots(binary_paths, cfg))
'
```

Regenerate the Gaussian `0 h / 6 h` reduced snapshot file:

```bash
julia --project=. -e '
using TOML
include("scripts/diagnostics/export_transport_v2_snapshots.jl")
cfg = TOML.parsefile("config/runs/era5_reduced_gaussian_transport_v2_dec2021.toml")
cfg["run"]["stop_window"] = 6
cfg["output"] = Dict(
    "snapshot_hours" => Any[0, 6],
    "snapshot_file" => expanduser("~/data/AtmosTransport/output/src_v2_snapshots/era5_reduced_gaussian_transport_v2_6h_dec2021.nc"),
)
binary_paths = [expanduser(String(p)) for p in cfg["input"]["binary_paths"]]
println(export_snapshots(binary_paths, cfg))
'
```

Try the reduced Gaussian snapshot export on GPU:

```bash
julia --project=. -e '
using TOML
include("scripts/diagnostics/export_transport_v2_snapshots.jl")
cfg = TOML.parsefile("config/runs/era5_reduced_gaussian_transport_v2_dec2021.toml")
cfg["architecture"] = Dict("use_gpu" => true)
cfg["run"]["stop_window"] = 6
cfg["output"] = Dict(
    "snapshot_hours" => Any[0, 6],
    "snapshot_file" => expanduser("~/data/AtmosTransport/output/src_v2_snapshots/era5_reduced_gaussian_transport_v2_6h_gpu_dec2021.nc"),
)
binary_paths = [expanduser(String(p)) for p in cfg["input"]["binary_paths"]]
println(export_snapshots(binary_paths, cfg))
'
```

## Most likely current interpretation

There were at least two bugs chained together:

1. The reduced face-indexed sweeps needed CFL-aware subcycling.
2. Resetting the air mass to the next window endpoint at every window boundary was turning local air-mass mismatch into immediate mixing-ratio error.

The second effect is directly observed and is large enough by itself to explain the first early reduced undershoot after `t = 1 h`.

The file-based CATRINE comparison supports the same broader picture:

1. The reduced grid does introduce visible horizontal sampling bias at `t = 0`, but it is not an immediate blow-up.
2. The first CATRINE cut comparison gives:
   - equatorial max absolute difference `1.627e-05` (`16.27 ppm`) at `lon = 31.25°`
   - meridional max absolute difference `3.081e-06` (`3.08 ppm`) at `lat = 9.47°`
3. A direct global comparison at the reduced cell centers gives:
   - surface RMS bias `13.99 ppm`
   - surface max bias `327.98 ppm` near `(82.35°E, 24.59°N)`
   - column-mean RMS bias `1.393 ppm`
   - column-mean max bias `25.61 ppm` near `(32.77°E, 1.55°N)`
4. That makes the CATRINE `t = 0` field useful for geographic bias checks, but it does not look like the immediate reduced-run instability is coming from the IC loader itself.

Current GPU interpretation:

1. The standalone v2 transport-binary scripts can now place the real reduced-grid model state and transport windows on `CuArray`.
2. However, a real reduced `1 h` export on GPU was not fast enough to finish interactively during this turn.
3. Snapshot samples of `nvidia-smi pmon` during the first GPU attempts showed the Julia process attached to the GPU but with low instantaneous SM utilization, so the current path may still be dominated by host-side setup / compilation / transfers.
4. GPU correctness for the real transport path still needs a completed reduced run and comparison against the CPU result.

## Remaining work

1. Run a longer reduced validation with the new code and `reset_air_mass_each_window = false`.
   The immediate next check is whether the reduced run stays well-behaved through at least `t = 6 h`.

2. Regenerate reduced snapshot output with the patched code and compare it to the lat-lon reference near the blob edge.
   The key question is whether the `t = 6 h` mismatch collapses back toward the `t = 0 h` level.

   Status as of this memo update:
   - [x] `~/data/AtmosTransport/output/src_v2_snapshots/era5_latlon_transport_v2_gaussian_6h_dec2021.nc`
   - [ ] `~/data/AtmosTransport/output/src_v2_snapshots/era5_reduced_gaussian_transport_v2_6h_dec2021.nc`
   - There is a long-running reduced export in flight using the exact command above

3. If the no-reset + subcycling path is still materially off relative to lat-lon, inspect whether the face-indexed `H → V → V → H` composition should be split further by face family.
   The current reduced path is not a direct analog of the structured `X → Y → Z → Z → Y → X` path.
