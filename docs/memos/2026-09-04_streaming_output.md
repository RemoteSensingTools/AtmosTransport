# Incremental single-file NetCDF output

Follow-up to the [September 4 review fixes](2026-09-04_review_followup.md),
published as `0a454a9b` on `fix/era-tm5-physics-equivalence`.

Single-file NetCDF runs now append each captured snapshot to an unlimited time
dimension instead of retaining every frame until completion. The outer run owns
the file handle, closes it on success or exception, and retains only schema
metadata and the reduced-Gaussian plotting map between snapshots. Each record
is flushed before publishing the `completed_snapshots` attribute. A failed
write aborts the stream; a partial final record can remain beyond that count.
This output is not a restart checkpoint, and a new run replaces its output on
the first successful capture.

The batch writer's names, dimensions, variable attributes and values are
preserved, apart from the streaming file's unlimited time dimension and added
completion attribute. Daily partitioning and binary snapshot retention keep
their existing behavior. Reserved or duplicate output variable names still
raise an error during initial schema construction.

The public CS reader now discovers and uses stored column means, including
column-only files. Selected-level lookup uses original model indices; asking
for a surface layer that was not written fails clearly. The reader also reports
the correct original number of model levels.

Tests compare four changing records across LL/RG/CS, Float32/Float64, and
full/selected/column-only output. They check exact batch/stream values and
variable attributes, visibility after each flush, unchanged retained Julia
writer size across records, public-reader results, and invalid-frame/failed-I/O
cleanup. This retained-size check is not a peak-RSS measurement.

CPU pipeline coverage includes all 12 topology/tracer-count/output combinations,
with every output time and tracer checked through the public reader. The
[raw measurements](../../scripts/benchmarks/results/streaming_output_20260904/)
record the overhead of incremental flushing: the small four-tracer CS fixture
took about 59 ms with full output and 46 ms with column-only output, compared
with earlier batch timings of 47 and 41 ms. The memory improvement removes
retention proportional to the number of snapshots; it is not a throughput claim.

Final validation is CPU-only following the user's instruction to stop GPU use.
The active GPU job was terminated when that instruction arrived. Preliminary
GPU runs are not evidence for the final implementation.

Validation:

- Julia 1.12.6 and 1.10.12: 1,179 streaming/reader/cleanup assertions passed
  (1,162 equivalence and reader checks, 17 rejection/cleanup checks).
- Existing output, selected-capture, visualization, builder and input-staging
  regression checks passed. The final visualization suite passed 15 checks.
- Aqua passed all 10 checks. JET remained at 181 reports against the unchanged
  181 baseline in the working tree.
- All 12 CPU pipeline cases completed, checking every time and tracer through
  the public reader outside the timed interval.
- The documentation build completed with deployment disabled; the only warning
  concerned detection of a local deployment environment.
- The isolated staged snapshot passed all 1,179 streaming checks, 120 CS
  builder checks, 42 input-staging checks and 21 public-contract checks.
  Pre-existing chemistry, inventory and coarsening changes remain local.
