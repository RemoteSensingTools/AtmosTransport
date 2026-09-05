# Input lifetime and workspace reuse on current main

This follows the selected-output checkpoint `36c23f8a` on
`revamp/current-main`. It ports the remaining runner lifetime, startup, and
file-handoff fixes using main's unified format-4 driver and independent
`DiffusionWorkspace`.

## Changes

- The runner owns each open driver from construction through setup and
  stepping. It drains any scheduled window prefetch before closing that driver
  and advising release of mapped payload pages, on success and on failure.
  Main already closed readers after rejected driver construction; that code
  remains unchanged.
- GPU startup reads the first forcing window once and adapts its host payload
  into two independent device buffers. Custom drivers returning device arrays
  get an explicit copy for the spare buffer. Failed completed prefetch tasks
  are consumed once; a failed fetch never swaps the active window.
- Cubed-sphere file handoff retains state, flux arrays, and numerical
  workspaces. Simulation setup refreshes forcing and diffusion layer thickness
  and invalidates CMFMC-derived caches. Main's removed TM5 LU cache is not
  restored.
- Partial window ranges across input files are rejected because they would
  skip forcing while carrying tracer state forward. Cubed-sphere runs reject
  an unsupported `start_window` instead of silently ignoring it. Single-file
  structured partial runs remain supported.
- Run instrumentation is disabled in `finally`, including when input
  inspection or resource construction fails before stepping begins.

## Validation

The focused CPU suite passes 337 checks: resource lifetime, constructor and
setup failures, partial-window guards, timer cleanup, startup reads,
continuous/split-file cubed-sphere physics, CMFMC cache handoff, asynchronous
output, and existing driven-simulation behavior. Split-file comparisons cover
Upwind/CMFMC, PPM/matrix CMFMC, and LinRood/TM5, each with diffusion and tracer
decay. Aqua passes 10 checks; JET retains 141 findings against the unchanged
144-report allowance.

The preceding output checkpoint passed the full CPU suite (81,851 checks,
22 existing skips or expected failures). The input changes are validated by
the focused suites above and the separate V100 checks recorded below.

On tofu GPU 0 (V100, CUDA runtime 12.6), all 223 checks pass: 140 startup and
buffer independence checks, 8 failed-prefetch cleanup checks, and 75 split-file
physics / CPU-reference checks. See [raw results and reproduction notes](../../scripts/benchmarks/results/main_input_v100_20260905/README.md).
