# Run instrumentation lifetime

`run_driven_simulation` previously disabled section timing only after a
successful run. An invalid transport header or unsupported physics request
left timing/allocation/NVTX flags enabled in the process. Subsequent runs could
therefore inherit instrumentation after their environment settings were off.

The public entry point now owns instrumentation around an internal input/run
helper, with `disable!` in `finally`. This covers input inspection and resource
construction as well as stepping and cleanup. Success reporting is unchanged;
on failure the original exception propagates and completed samples remain
available for an explicit `SectionTimer.report()`.

The regression exercises real public runs with a malformed binary and a valid
binary missing requested TM5 convection forcing, in both timing and NVTX-only
modes. It also checks a following untimed run and a successful timed run.
Before the fix, 12 of 28 new assertions failed; afterward all 28 passed on
Julia 1.10.12 and 1.12.6. The complete input-resource test file passed 58
assertions on both versions.
The process-global instrumentation API and manually enabled instrumentation
when runner environment switches are off retain their existing semantics.
