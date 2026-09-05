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
and deferred workspace adaptation. Historical V100 results remain labeled by
the original measured commits; main-based GPU and runtime validation is tracked
separately as integration progresses.

## Remaining integration work

- Validate matrix kernels on the main-based V100 package.
- Port selected/streaming output, run resource ownership, staging, and cache
  handoff improvements onto the current reader/model APIs.
- Retain current signed-tracer conservation and output-total contracts.
- Reconcile scientific documentation and package-loading improvements.
- Run the full main-based CPU suite and focused GPU runtime checks before
  treating integration as complete, then resume real-input profiling and
  adaptive batching measurements.
