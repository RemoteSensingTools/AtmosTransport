# Deferred legacy scratch for collaborative matrix convection

Follow-up to [tracer batching](2026-09-04_matrix_convection_batching.md),
implemented 2026-09-04 without GPU execution.

## Allocation lifetime

Runtime factories for TM5 and CMFMC matrix convection now set
`defer_scratch=true` when collaborative LU is requested. The workspace keeps
zero-column matrix, pivot, cloud-index, and flux-vector buffers plus the
configured legacy tile-column count. Cell metrics, CMFMC derived-rate arrays,
and any explicitly requested persistent-cache arrays retain their normal
allocation and lifetime.

The collaborative kernels need no legacy scratch, so those buffers stay empty.
If a CPU/Float64 fallback or supported legacy adjoint entry point needs them,
`_ensure_tm5_scratch!` allocates the full configured tile once. Later calls reuse
the same buffers. All allocations complete before the new arrays are installed.
`f_scratch` continues to alias `conv1`. TM5Workspace is now mutable to support
this one-time installation; its array types remain concrete.

Backend adaptation preserves the deferral policy. Even if a CPU fallback has
already populated a workspace, adapting the model starts the destination with
empty scratch instead of transferring an unused matrix tile. Persistent data
and cache sentinels retain their existing adaptation semantics. This matters
because structured model setup constructs a CPU model before adapting it.

The default constructor remains eager unless deferral is explicitly requested.
CPU/Float64 fallback retains the configured tile capacity; this change does not
replace it with slow single-column launches. Existing restrictions on the
operators supported by the adjoint remain in force.

## Memory effect

For 85 levels and 32,400 tile columns, the legacy payload is

```
[4*(85^2 + 2*86) + 8*(85+3)] * 32,400
= 981,460,800 bytes = 0.914 GiB
```

Deferred buffers contain zero payload bytes until a legacy solve requests them.
Small array headers and workspace metadata remain. The number above uses a
C180 interior-sized panel; the current CS constructor sizes from the supplied
panel dimensions, so halo-padded arrays can make the avoided allocation larger,
up to approximately the configured 1 GiB budget. This refines the review's C180
estimate. Larger vertical grids generally reach that budget.

This is a verified allocation-shape result, not a measurement of total device
peak memory. CMFMC rates, state arrays, metrics, and aggregation temporaries still
consume their usual memory. Optional persistent LU-cache scaffolding is separate.

## CPU validation

- 101 workspace-lifetime assertions on Julia 1.12.6 and 1.10.12: both matrix-operator
  factories on all three topologies, eager/deferred selection, allocation and
  reuse, aliasing, adaptation, persistent data/sentinel ownership, validation of
  invalid tile settings, and the C180 scratch payload calculation.
- 483 assertions across existing TM5, CMFMC matrix, alias/tile equality, and
  transport-model runtime tests. Deferred CPU fallback is compared directly
  with the eager solver on LL/RG/CS, and both CMFMC matrix adjoint directions
  exercise deferred workspaces. Six CUDA testsets were skipped.
- The opt-in A100 batching regression now also checks deferred workspace
  adaptation and legacy allocation on the device. It has not been run.
- A clean export of the staged files passed Aqua (10 assertions), the JET
  inference gate (179 reports against the existing 181-report baseline), and
  all 101 workspace-lifetime assertions. Unrelated working-tree edits were
  excluded from this export.

The production batching kernel still needs A100 compilation, race/correctness
checks, and performance measurement when GPU use resumes. The CMFMC-specific
Hessenberg factorization remains a CPU prototype; it is not enabled by this
workspace change.
