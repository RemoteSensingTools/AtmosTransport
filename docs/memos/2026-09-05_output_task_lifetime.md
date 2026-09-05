# Daily output task ownership

The runner previously stored its background daily writer in a local variable
inside each topology loop. Successful runs waited for that task, but an error
in a later file or transport step bypassed the wait. The outer cleanup owned
only the single-file NetCDF stream. A failed run could therefore return while
the previous day's task was still writing and retaining its snapshot arrays.

`RunSnapshotOutput` now owns both the stream and the pending daily task. A
shared helper starts daily writes for LL/RG and CS, drains the previous task
before accepting new frames, and transfers ownership of a copied frame list.
Run cleanup drains the task on normal and exceptional exits, clears the task
reference even after a write failure, and closes any stream. Concurrent
transport and writer failures are reported together in a `CompositeException`.
Input staging cleanup still executes after output cleanup.

This keeps the existing overlap and one-write memory bound. It changes failure
lifetime, not output values, filenames, frame selection, or normal cadence.

Validation:

- 28 focused assertions pass on Julia 1.12.6 and 1.10.12. Channel-controlled
  tasks prove that exceptional run exit waits for a pending writer. Tests also
  check successful return, writer-only failure, dual failure, idempotent cleanup,
  real daily NetCDF values, and stream closure after a background I/O error.
- 1,302 assertions pass across the new tests, existing output integration
  (including the canonical daily runner), and streaming NetCDF equivalence.
- All tests run on CPU with CUDA devices hidden.

The resource boundary lives in
[`runner/output.jl`](../../src/Models/runner/output.jl); both topology loops use
it through [`DrivenRunner.jl`](../../src/Models/DrivenRunner.jl).
