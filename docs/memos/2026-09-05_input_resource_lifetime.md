# Driver and prefetch lifetime in driven runs

Previously, daily driver handles closed only after successful setup and stepping.
If either failed, the driver could stay open until garbage collection and a GPU
window-prefetch task could outlive the run. The path-based driver constructors
also left their newly opened readers to GC when later validation failed.

The runner now owns its current driver and `DrivenSimulation` explicitly. At
normal file completion and exceptional run exit it drains a scheduled prefetch,
then closes the driver. Cubed-sphere cleanup retains its existing mapped-payload
release hint after prefetch completes. Ownership references are cleared and
repeated cleanup is harmless. Both run and cleanup failures remain visible.
Path-based driver constructors close their own reader if validation or grid
construction fails; the overload accepting a caller-owned reader retains its
existing ownership contract.

The existing output exception wrapper is now the shared `_with_run_resource`
helper in `runner/resources.jl`. Output behavior is otherwise unchanged.

Validation:

- 24 input-lifetime assertions pass on Julia 1.12.6 and 1.10.12, including a
  channel-controlled background task that reads a real binary while cleanup
  waits, failed reads, early setup rejection, and Linux file-descriptor checks
  after rejected LL and CS driver construction.
- The 28 asynchronous-output assertions still pass. Existing driven-window
  tests and 120 CS builder assertions pass from a clean export.
- Aqua passes 10 checks; JET reports 180 against the unchanged 181 threshold.
- Tofu V100 GPU 0 completes 12 binary-reader → transport → NetCDF cases with
  LL/RG/CS, one/four tracers, full/column-only output, two input files and five
  output times. Every tracer and output time passes the public-reader checks.
  These synthetic warm-cache cases exercise successful GPU prefetch handoffs;
  exceptional cleanup is exercised by the deterministic host tests.
