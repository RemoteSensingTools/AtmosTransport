# Consume a failed prefetch once

When `_take_prefetched_window!` fetched a failed background task, it threw before
clearing `prefetch_task` and `prefetch_window_index`. The runner's input cleanup
then waited on the same failed task, producing a second `TaskFailedException`
and wrapping the original failure in a redundant `CompositeException`.

The fetch now clears ownership of a completed task in `finally`, before either
swapping a successful window or propagating the failure. An interrupted wait
retains ownership if the background task is still running, so input cleanup can
still drain that task before closing the reader. Active forcing is not swapped
when the prefetch fails.

The initial V100 probe passed three assertions, failed two ownership assertions,
and errored when cleanup observed the failure again. The regression now exercises
the actual `RunInputResources` scope and checks that the propagated exception
refers to the original failed task, input ownership is released, active forcing
is unchanged, and a subsequent drain is a no-op.

This changes exceptional-exit bookkeeping only. It does not change window
loading, numerical kernels, or the successful window-buffer exchange.

The final V100 run passes all eight failure-scope checks and repeats all 140
startup/buffer checks successfully on the same authorized GPU 0.
[Raw test output](../../scripts/benchmarks/results/prefetch_startup_v100_20260905/failure_after.txt)
is retained beside the earlier probe and startup benchmark results.
The final source also passes the 58-check input-resource suite and JET
(180 reports against the unchanged 181 threshold) in a clean Julia 1.12.6
export. All staged Julia source files match that verification export.
