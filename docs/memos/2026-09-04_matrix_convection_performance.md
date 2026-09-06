# Matrix convection: shared memory, tracer scaling, and exact solver opportunities

Reviewed 2026-09-04 against `a89cd6af`, with unrelated local work present.
No GPU was used. Production dispatch and kernels are unchanged. The accompanying
CPU prototype is an investigation artifact, not a deployed optimization.

## Findings and recommended order

1. **Remove the total-tracer ceiling by batching RHS vectors inside one kernel.**
   Build and factor each column once, then load, solve, and store small tracer
   batches against the same factors. This applies to both TM5 and CMFMC matrix
   convection. Total tracer count need not determine shared-memory allocation.
2. **Specialize CMFMC factorization for its upper-Hessenberg matrix.** It has no
   downdrafts, so only the first subdiagonal can be nonzero. Partial-pivot LU can
   preserve the equations with quadratic rather than cubic factorization work.
   The CPU prototype matched the existing solver, including pivot stress cases.
3. **Stop allocating unused legacy matrix scratch for collaborative solves.**
   This can recover approximately 0.91 GiB for C180 with 85 total levels, or near
   the configured 1 GiB budget for larger columns.
4. **Tune CUDA tile width, workgroup size, and matrix construction on A100 when
   GPU testing resumes.** Larger shared-memory allocations alone are not a speed
   guarantee. Kernel throughput and occupancy must decide the configuration.

## What the six-tracer limit actually is

The fast storage is **workgroup shared memory** (`@localmem`, also called
threadgroup memory on Metal). The implementation is the collaborative LU path:
one 32-thread workgroup handles one atmospheric column.

[`tm5_kernels.jl`](../../src/Operators/Convection/tm5_kernels.jl) sets
`_TM5_COLLAB_NT_MAX = 6` and `_tm5_collab_supports` requires both `L <= 85` and
`Nt <= 6`. Here `L` is the effective matrix dimension after any configured
vertical aggregation, not necessarily the model's full vertical dimension.
[`TM5Convection.jl`](../../src/Operators/Convection/TM5Convection.jl) enforces
this envelope: a Float32 GPU run requesting collaborative LU throws outside it.
Float64 and CPU requests warn and use the legacy path. Some kernel comments
still incorrectly describe an eight-tracer limit or silent fallback.

For Float32, declared shared-memory arrays require

```
bytes = 4*L^2 + (4*tracer_slots + 16)*L + 16
```

This includes the matrix, tracer RHS slots, pivots, layer masses, two flux
vectors, and cloud indices. It excludes compiler padding and device allocation
granularity.

| Matrix levels | RHS slots | Declared shared bytes |
| --- | --- | --- |
| 85 | 6 | 32,316 |
| 85 | 7 | 32,656 |
| 85 | 8 | 32,996 |
| 85 | 16 | 35,716 |
| 85 | 32 | 41,156 |
| 137 | 6 | 80,572 |

The conservative six-slot setting targets a 32 KiB cross-backend budget. Its
comment that six is the mathematical maximum is inaccurate: the declared-array
formula also fits seven, before any compiler padding. Raising six to seven would
still leave the underlying scaling problem.

An A100 supports up to 163 KiB shared memory per block; static allocation remains
limited to 48 KiB, with explicit opt-in and dynamic allocation needed above that.
Thus 85 levels with 32 RHS slots fits the documented static capacity by declared
size. Full 137-level dense storage needs a different allocation strategy or
representation, even with only six slots. This is a hardware-capacity observation,
not a verified KernelAbstractions launch configuration.
[NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#unified-shared-memory-l1-texture-cache).

## Tracer batching without repeated LU

The existing kernels already factor once for all tracers in a call. Preserve
that property when removing the ceiling:

```
build A from this column's forcing, air mass, area, and timestep
factor A once; retain A and pivots in shared memory
for first_tracer = 1:batch_width:total_tracers
    cooperatively load this batch into q_shared
    apply pivots and solve each RHS against the retained factors
    cooperatively store the batch
    synchronize before reusing q_shared
end
```

Handle the final partial batch and use uniform workgroup control flow around
barriers. Implement the same behavior for lat-lon, reduced Gaussian, and all
cubed-sphere panels. Batch width is an implementation choice, separate from the
scientific tracer count. Start with six slots for the existing memory envelope;
measure 8, 16, and 32 slots on CUDA. Do not split tracers into separate convection
calls that rebuild and refactor the same matrix.

Today pivot application is serial in thread 1; triangular solves assign one
tracer to each active thread. With six tracers, only six of 32 threads work during
those solves. Independent tracer permutations can move to their owning threads.
Wider batches may improve lane utilization, but also consume more shared memory
and reduce resident workgroups. The CPU probe validates arithmetic equivalence,
not GPU synchronization or the best batch size.

## CMFMC-specific quadratic factorization

[`CMFMCMatrixConvection.jl`](../../src/Operators/Convection/CMFMCMatrixConvection.jl)
derives updraft entrainment/detrainment from CMFMC and passes zero downdraft rates
to TM5. In the matrix builder, the updraft creates upper-triangular entries,
subsidence adds the first subdiagonal, and the final flux-divergence assembly
does not introduce entries below that subdiagonal. The result is an
upper-Hessenberg matrix.

At elimination step `k`, only rows `k` and `k+1` can provide a nonzero pivot
candidate. Only row `k+1` needs a trailing update. Keep adjacent row pivoting,
swap previous L multipliers as well, and retain the existing general triangular
solve. Earlier pivots can move stored L entries below the first subdiagonal;
simplifying the forward solve to a bidiagonal recurrence would require a
different factor representation and proof.

This specialization is appropriate for the explicitly zero-downdraft CMFMC
operator. General TM5 columns with downdrafts must retain a general factorization
unless a separate structural specialization is established. Do not use a small
downdraft threshold to change the equations. The current adjoint consumes LU and
pivots, so a production change also needs forward/transpose identity tests.

The prototype still stores a dense matrix; it reduces arithmetic, not shared
matrix storage. Further compression or a direct plume recurrence is a separate
investigation. Once LU is cheaper, serial matrix construction and triangular
solves become more important; the old comment dismissing matrix construction
based on its flop fraction is not sufficient evidence about GPU runtime.

## CPU evidence

Run the reproducible probe with GPUs hidden:

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 julia --project=. \
  scripts/benchmarks/probe_matrix_convection_cpu.jl \
  scripts/benchmarks/results/matrix_convection_cpu_20260904.toml
```

[`probe_matrix_convection_cpu.jl`](../../scripts/benchmarks/probe_matrix_convection_cpu.jl)
passed **2,489 assertions** on Julia 1.12.6, Intel Xeon Platinum 8462Y+:

- Actual production CPU rate derivation and matrix construction for synthetic
  CMFMC columns, Float32/Float64, 25/60/85/137 levels, two cloud tops, three seeds
  per configuration. All matrices had the predicted exact structural zeros.
- Specialized and dense LU agreed in values and pivot sequences. Stress matrices
  exercised 1,958 row swaps.
- Tracer counts 1/6/7/12/32/65 with batch widths 4/6/16/32 produced bit-identical
  solved values. Additional fully dense matrices checked batching independently
  of the CMFMC structure.
- Physical fixtures passed positivity, tracer mass conservation, and linear
  residual checks. Pivot stress fixtures passed normalized residual checks.

Timings below are medians of nine samples of 30 repeated **serial CPU column
operations**, after warmup. Both paths use six-wide RHS batches and preallocated
buffers. Complete column timings include matrix construction, factorization,
RHS copy, and solve; factor timings also include copying the input matrix.
They exclude GPU launches, global-memory traffic across a grid, forcing I/O,
and met-window setup. The process had 16 Julia threads available, but these
column operations are serial. Synthetic inputs are not a real-forcing benchmark.

| Levels | Tracers | Dense column, µs | Hessenberg column, µs | CPU speedup |
| --- | --- | --- | --- | --- |
| 60 | 6 | 45.8 | 17.6 | 2.60× |
| 85 | 6 | 107.7 | 33.9 | 3.18× |
| 85 | 32 | 215.5 | 136.1 | 1.58× |
| 85 | 65 | 353.0 | 267.7 | 1.32× |
| 137 | 6 | 385.3 | 85.7 | 4.49× |

At 85 levels, factorization plus matrix copy fell from about 74.5 to 2.1 µs.
The smaller complete-column gain at larger tracer counts shows why optimizing
the RHS solve matters too. These measurements do **not** predict A100 speedups.
Full results: [TOML](../../scripts/benchmarks/results/matrix_convection_cpu_20260904.toml).

## Global scratch and persistent caches

The runtime factories in
[`TransportModel.jl`](../../src/Models/TransportModel.jl) always construct a
legacy `TM5Workspace`. Its matrix, pivot, cloud, and flux scratch allocations
are unused by the unmerged collaborative kernels. At C180 with 85 total levels,
the per-panel tile workspace contains about 0.914 GiB. It is reused across the
six panels, so this is not six times that allocation. With 137 total levels,
the default tile budget limits this scratch to approximately 1 GiB. A minimal
collaborative workspace must preserve cell metrics and account for CPU/Float64
fallbacks, vertical aggregation, and adjoint workspace requirements.

The `cache_A`, `cache_pivots`, and `cache_valid` fields are scaffold in the current
production solver: factories leave them disabled, and no production apply kernel
reads cached factors. This is distinct from the working CMFMC derived-rate cache.
Historical LU-cache benchmark scripts should not be mistaken for enabled runtime
behavior.

A full C180 Float32 LU cache with Int32 pivots would occupy about **5.294 GiB at
85 levels**, or **13.692 GiB at 137 levels**, before other model arrays. The current
constructor sizes it from total `Nz`, not the smaller convection limit.

Across-call factor reuse is also not the first priority for current driven binary
runs: [`DrivenSimulation.jl`](../../src/Models/DrivenSimulation.jl) applies
convection once at the end of each met window unless the per-substep override is
enabled. Window advancement invalidates the cache. Even in repeated-call
workflows, factors depend on air mass and timestep as well as forcing; a met-window
key alone is insufficient when air mass changes. In-kernel tracer batching reuses
the exact same matrix without this lifetime problem.

## A100 validation when GPU use resumes

Measure baseline and batched kernels at tracer counts 1, 6, 7, 12, 32, and 65,
including nonmultiples of batch width, no-convection columns, shallow/deep clouds,
and representative real forcing on all topologies. Check mass, positivity,
reference solves, and adjoint identities before accepting throughput results.
Compare CMFMC Hessenberg against general LU separately from tracer batching.

Profile matrix build, factorization, RHS solve, barriers, register count, shared
allocation, and resident warps. One 32-thread workgroup consumes roughly 32 KiB
at 85 levels, so shared memory permits only a few resident warps per SM; inspect
larger workgroups as well as wider tracer batches. Test leading-dimension padding
where strided shared accesses conflict, rather than adding padding blindly.
Finally measure full-window convection time and peak device memory. No GPU
compilation, race checking, occupancy measurement, or production performance
validation was performed in this investigation.
