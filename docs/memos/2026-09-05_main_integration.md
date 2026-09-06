# Revamp integration with current main

The first review continued on `fix/era-tm5-physics-equivalence`, whose common
ancestor with main was `65676b82` (2026-07-11). After fetching origin on September
5, main was `a0698dde` (v0.3.0): 47 commits were unique to main and 34 to the
review branch. Main uses format-4 transport binaries; the review branch still
used format 3. The available ERA5 and GEOS archives therefore could not be
profiled directly on the original review branch.

Integration proceeds on `revamp/current-main`, based on `a0698dde`, in a separate
worktree. The original branch and unrelated uncommitted chemistry/inventory
work remain untouched. Main's typed backend API, unified binary readers,
signed-tracer support, and independent diffusion workspace remain authoritative.
Review changes that main already supersedes are not reinstated.

## Matrix convection

The tracer batching, deferred scratch, structured LU, specialized RHS solve,
and parallel matrix assembly changes have been ported. Main removed unused
persistent TM5 LU-cache fields; the port keeps that removal and adds only
`scratch_columns` and `defer_scratch` to the workspace. The CMFMC-derived forcing
cache remains intact and receives its existing adaptation checks.

Focused CPU tests on the main-based package pass 3,154 checks, covering host
batch selection, partial/pivoted batches, forward/adjoint structure selection,
and deferred workspace adaptation. The main-based package also passes 727 checks on tofu GPU 0 (V100, CUDA
runtime 12.6): 7 workspace adaptation checks, 480 positive-tracer checks, and
240 signed-tracer checks. These compare against dense CPU LU, verify
conservation and untouched halos/cloud-free regions, and compare each batch
against independent launches. Historical timing results remain labeled by the
original measured commits; full runtime timings are separate from kernel tests.

## Readability and test loading

The runner helpers and initial-condition/surface-inventory helpers are extracted
from main without changes to their implementations. Reassembling the extracted
files reproduces the original source apart from file-boundary whitespace. In particular, main's signed native initialization, physical
inventory-to-storage conversion, and time-varying sources remain intact. Core
tests share the cached package while isolating their test helpers per module.

The large original revamp commit was not cherry-picked wholesale: it combined
these changes with older workspace and backend APIs. Integration uses smaller
changes appropriate to current main.

## Real-input baseline

The current reader now runs the existing C90 L66 ERA5 format-4 archive directly.
On tofu's V100, two measured repetitions after warmup take a median 4.149 s
with six tracers and 16.396 s with 32 tracers for two model hours. The 32-tracer
convection section is 0.306 s, while cumulative host allocation is 8.855 GB.
This is a warm-cache baseline with experimental archived forcing, not a
cold-I/O measurement or a before/after speedup claim. All 462 output checks
pass; maximum relative total-storage drift is 2.572e-6. See the complete
[method and artifacts](../../scripts/benchmarks/results/main_real_input_v100_20260905/README.md).

## Main-based validation

The complete Julia 1.12.6 CPU baseline (`julia --startup-file=no --project=test test/runtests.jl`, four Julia threads,
GPUs hidden) passes 83,556 checks across
117 core files plus the regridding runner after output, input, staging, device
workspace, pressure-layer typing, and direct tracer-packing integration. There are 22 existing skips or
expected-broken checks and no failures. Aqua passes. The final focused JET
check reports 142 findings against main's unchanged 144-report allowance after
restricting the new initialization helper to CS grids and six-panel arrays. The run includes signed
advection, signed native initialization, surface-inventory conversions, model
construction, and the extracted runner's CLI workflow.

## Output integration

Selected capture, single-file streaming, and background output ownership are
ported, retaining main's signed-total and ATMSNAP contracts. The
[output integration report](2026-09-05_main_output_port.md) records validation
and the real-input before/after measurements. Remaining whole-run allocation
is predominantly outside snapshot capture.

## Input integration

Input ownership, startup loading, cache handoff, and cubed-sphere workspace
reuse are ported to main's APIs. The [input integration report](2026-09-05_main_input_port.md)
records the lifetime and continuous/split-file regression checks.

Staging ownership and source-identity checks are now integrated too; see the
[staging report](2026-09-05_main_staging_port.md). Optional preprocessing Git
provenance probes preserve fallback values without printing fatal Git messages
from exported source trees.

CS workspace construction now follows the device state directly; see the
[allocation measurements](2026-09-05_main_device_workspace.md). It reduces host
allocation and isolated construction time without a claimed additional
whole-run speedup.

## Startup and package loading

[Pressure-layer configuration typing](2026-09-05_main_pressure_initialization.md)
removes 4.98 GB of temporary scalar allocation from 32-tracer initialization.
The corresponding real V100 workload now takes 5.675 s versus 15.556 s at the
preceding checkpoint; all 196 output arrays remain exactly equal. Focused
scientific initialization and Aqua/JET checks pass; these changes are included
in the complete CPU suite above.

The maintained preprocessing, regridding, inversion, and snapshot-converter
entry points now [reuse the compiled package](2026-09-05_main_package_loading.md).
Their existing CLI/regridding/inversion suites and a signed converter smoke pass.

[Direct state packing](2026-09-05_main_direct_packing.md) removes the next
467 MB of temporary 32-tracer storage. The same real V100 workload now takes
5.030 s with 1.808 GB cumulative host allocation, versus 5.675 s and 2.275 GB
at the pressure-typing checkpoint. All 196 output arrays remain exactly equal;
75 GPU file-handoff physics checks also pass.

## Remaining integration work

- Profile the remaining initial-VMR construction and longer steady-state
  workloads before further changes. The sweep/halo timing ambiguity is resolved
  by the CUDA PPM checkpoint below.
- Retain current signed-tracer conservation and output-total contracts.
- Reconcile scientific documentation and package-loading improvements.
- Repeat CPU and GPU runtime validation after the remaining I/O ports. Compare
  against the real-input baseline above before promoting an adaptive tracer
  batch size. The broader revamp is not yet fully integrated.

## CUDA PPM launch layout

[Synchronized sweep profiling and tile selection](2026-09-05_main_ppm_launch_tiles.md)
reduce padding in the packed Float32 CUDA PPM launch without changing kernel
arithmetic. A 32×2 tile reduces the real V100 32-tracer workload from 5.030 to
4.466 s median, with essentially unchanged host allocation. All 196 output
arrays remain exactly equal. The new GPU diagnostic passes 540 checks through
65 tracers; focused CPU advection/Aqua/JET checks pass 619 assertions. The full
CPU suite above predates this launch-only change. Other GPU architectures and
longer runs remain unbenchmarked.
