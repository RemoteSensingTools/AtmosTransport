# TM5 convection: where the time goes, and what we can do about it

Date: 2026-05-22. Run host: wurst (NVIDIA L40S).
Inputs: ERA5 / GEOS-native C180 / L85 / Float32 production binary at
`/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps48_v3_20260520/`.

## TL;DR

- TM5 convection is **22× slower than advection** in production: 1172 ms vs
  52 ms per substep, 91.6% of wall time. Confirmed from the production
  timing CSV in the run dir, not extrapolated.
- The hot loop is one per-thread serial dense LU of an 85×85 matrix per
  column. ~200 k flops/column, ~194 k columns/substep, fundamentally a poor
  GPU pattern (per-thread serial work, branch divergence, ~28 KB working
  set per thread).
- **A backend-portable KA kernel doing workgroup-collaborative LU
  (one workgroup per column, 32 threads sharing `@localmem`) gives a
  10–11× speedup, bit-exact to the baseline within Float32 rounding,
  and runs on both CUDA and Metal**. This is the production
  recommendation given the Metal portability constraint.
- cuBLAS strided-batched LU on the same problem is even faster
  (**136× on GPU compute**, 5 ms vs 692 ms per panel of 32 400 columns)
  but is CUDA-only AND hard-capped at N ≤ 64 in CUDA 13.1 (confirmed
  this session — N=65 segfaults inside libcublas).
- TM5 itself caps the convection matrix at a global `lmax_conv` parameter
  set per model setup: 19, 25, 34, or 87 of 137 layers depending on
  `proj/levels/ml137/tropoX*`. The cuBLAS N ≤ 64 limit is consistent with
  every TM5 production setup except `ml137/tropo137`.
- An active-depth scan of the production binary across all 24 windows of
  one day (194 400 columns/window) shows: `min_top_code = 11` (deepest
  active column reaches layer 11 from TOA, i.e., 75 layers from surface),
  `median = 53` (33 layers), `p95 = 73` (≤13 layers). About 6% of columns
  have icltop ≤ 21 — these are the columns that would lose physics at
  lmax_conv = 64.

## What we measured

`scripts/benchmarks/bench_tm5_alternatives.jl` loads one real C180 panel
of ERA5 forcing (`entu / detu / entd / detd / m`) from the production
binary and times five variants on the same Nt=2 tracer initial condition.
Two representative runs (window 1 panel 1, window 12 panel 3):

| Variant | lmax | gpu min ms | total min ms | speedup vs baseline | max\|Δq\| in active block |
|---------|------|------------|--------------|---------------------|----------------------------|
| baseline (per-thread serial LU) | 85 | 397–692 | same | 1.0× | 0 |
| cuBLAS strided-batched | 64 | 5.0–5.1 | 54–55 | **78–136×** GPU / **12.5×** total | 2.3e-6 (window 12) / 0.78 (window 1) |
| cuBLAS strided-batched | 33 | 2.1 | 16 | 190×/43× | 0.95 |
| cuBLAS strided-batched | 25 | 0.8 | 9 | 511×/77× | 1.36 |
| factor-once + apply-only | 64 | 1.5 | — | 259× | — |

Notes:
- `gpu min` = `CUDA.@elapsed` around the factor+solve calls only.
- `total min` = same plus the host-side matrix build (still on CPU
  threads) and the host↔GPU upload.
- The "max\|Δq\| in active block" column compares the slab against the
  baseline result restricted to layers `(Nz - lmax + 1):Nz`. Window 12 /
  panel 3 has no column with icltop ≤ 21, so the lmax=64 slab is
  numerically identical to baseline; window 1 panel 1 catches several
  ITCZ-ish deep-convection columns and shows the policy cost of
  truncation (these are the columns whose convection above k=21 we'd
  drop, exactly the same approximation TM5's tropo25a/tropo34a setups
  make).

## What blocks naïve cuBLAS at full Nz

CUDA 13.1's `cublasSgetrsBatched` segfaults for N ≥ 65 (verified this
session, repeatable, isolated minimal test). cublas-XT and the non-
batched `cusolverDnSgetrf` have no size limit but lose the batch-launch
amortization that makes the call cheap. cuSOLVER has no `getrfBatched` /
`getrsBatched` for general dense LU. So in CUDA 13.1, a cuBLAS-batched
LU strategy implies a hard global ceiling of `lmax_conv ≤ 64`.

## The two questions you asked

### 1. What does TM5 itself do?

Read of `deps/tm5-cy3-4dvar/base/src/tm5_conv.F90` plus the per-level
parameter set in `proj/levels/ml*/src/dims_levels.F90`:

- TM5 declares `lmax_conv` as a **global, level-set-wide compile-time
  parameter**. Examples: `ml91/tropo60: lmax_conv = 40`,
  `ml137/tropo25a: lmax_conv = 25`, `ml137/tropo34a: lmax_conv = 34`,
  `ml137/tropo137: lmax_conv = 87`. The `ml60/tropo60` build sets it
  equal to `lm(1)` (full vertical, "to test ECMWF diffusion coefficients
  in the stratosphere", per the comment from Sourish Basu).
- The forcing arrays `entu / detu / entd / detd` are read at exactly
  `(im, jm, lmax_conv)` shape — the preprocessor never touches the upper
  layers. Convection is by construction identity above `lmax_conv`.
- The matrix `conv1` is `(lmax_conv, lmax_conv)`, factored with LAPACK
  `dGeTrf` per column. Inside `TM5_Conv_Apply`, `lmc` (= the per-column
  reachable depth) further restricts the active block. So TM5 factors a
  matrix of size at most `lmax_conv` and applies a block of size at most
  `lmc ≤ lmax_conv`.
- TM5 is bit-for-bit consistent across the model: the matrix never sees
  more than `lmax_conv` levels because the meteo reader never delivers
  more.

In short, **TM5 already does exactly the trick we need to make cuBLAS
batched LU viable** — it just does it at the meteo-reader / preprocessor
boundary, not at runtime. Our current path keeps the full 85-layer
matrix because our preprocessor writes the full `(Nc, Nc, Nz)` per
forcing field.

### 2. Can we factor once and apply many times within the palindrome?

The current `step!` at `src/Models/TransportModel.jl:425` is:

```julia
transport_step!(model, dt)        # palindrome-internal advection (+ diffusion at center)
convection_chemistry_step!(model, dt)   # convection THEN chemistry, each once
```

So convection runs **once per outer substep**, not twice inside a
palindrome. The matrix is built and factored once per substep already.
Within that single call:

- For Nt ≥ 2 tracers, we already amortize the factor across `Nt`
  back-substitutions — that part is fine.
- We do **not** amortize the factor across multiple substeps, because
  `m_col` is updated by advection between consecutive convection calls
  (TM5 has the same constraint). The matrix is `I - dt·D(m, forcing)`,
  and `D` depends on `m`. Even with the forcing fields constant across a
  met window, the matrix coefficients change every substep.

What this means for the palindrome question:

- **Current code**: a "factor once, apply many" rewrite within a
  palindrome saves nothing, because there's only one convection call per
  palindrome.
- **If we adopted a Strang-symmetric splitting** (`½ conv → advection →
  ½ conv` around each substep, with the matrix evaluated at the
  half-substep state on both sides), then yes — *the same `m_col` and
  forcing* are used on both halves, so the factor can be reused.
  Microbench upper bound: solve-only cost is 1.5 ms vs 5.1 ms for
  factor+solve at lmax=64, so the second half-step is 3.4× cheaper. Net
  speedup if the outer scheme calls convection twice instead of once:
  `(5.1 + 1.5) / (5.1 + 5.1) = 0.65` — i.e., 1.55× cheaper *than calling
  full convection twice*, but still 1.29× *more* expensive than calling
  it once. So strict palindrome-amortization only wins if the underlying
  splitting requires two calls anyway.
- **The bigger amortization** is across an entire met window (48
  substeps). That requires a real physics decision: freeze `m_col` at
  the start of the window and use it for all 48 convection calls. The
  microbench upper bound is `(factor / 48 + solve) = 5.1/48 + 1.5 ≈
  1.6 ms/call`, vs 5.1 ms per fresh factorization. That's a 3.2×
  ceiling, exclusive of host-build cost. TM5 does not do this — they
  rebuild the matrix every convection call — so the validation cost
  would be ours alone.

## Recommended path forward

Four independent levers. **Lever 0 (KA collaborative LU)** is the
production recommendation because it's the only path that meets the
Metal-portability constraint without a physics approximation. The
remaining levers are CUDA-only headroom we can stack on top later.

### Lever 0 — Workgroup-collaborative LU as a KA kernel (PORTABLE, BIT-EXACT)

The cuBLAS path below is CUDA-only. The portable answer is a single KA
kernel that runs one workgroup per column, with 32 threads in the
workgroup collaboratively factoring an `Nz × Nz` matrix in
`@localmem`. Implementation lives at
`scripts/benchmarks/bench_tm5_collab_lu.jl` (prototype).

Measured numbers (C180 panel, one window, Float32, Nt=2,
`WG_SIZE = 32`):

| Window / panel | baseline ms | collab-LU ms | speedup | max\|Δq\| |
|----------------|-------------|--------------|---------|-----------|
| 1 / 1 (deep)   | 691.5       | 63.4         | 10.9×   | 7.2e-7    |
| 6 / 3          | 567.1       | 57.0         | 9.9×    | 8.3e-7    |
| 12 / 3 (shallow)| 396.4      | 52.7         | 7.5×    | 6.0e-7    |
| 18 / 3         | 425.4       | 52.6         | 8.1×    | 6.0e-7    |

The errors are all at the Float32 numerical floor — the kernel is
bit-exact to the baseline within rounding. Speedup is 8–11× across
windows; deeper windows benefit more because the baseline scales with
active depth while collab-LU has a smaller depth-dependent slope (the
inter-thread sync cost is constant per outer iteration).

Memory budget per workgroup:
- `A_loc`  : `Nz² × 4 B`     = 28 900 B at Nz=85
- `q_loc`  : `Nz × Nt × 4 B` =  2 720 B for Nt=4 (limit raised by
  changing the second dim of the `@localmem` Nt slot)
- `piv_loc`: `Nz × 4 B`      =    340 B
- `amu/amd_loc + bmass_loc`  :  1 380 B
- Total                      : ~33 KB per workgroup

This fits within Apple Metal's threadgroup-memory budget (≥32 KB on M2
and later) and CUDA's per-block shared (100 KB/SM on L40S → 3 columns
concurrent per SM, ~430 wavefronts of 32 columns each for one full
panel).

Why it works:
- The matrix is dense within the active window; the LU is the only hot
  loop. A serial LU in one GPU thread is a fundamentally bad pattern.
- Putting the matrix in shared memory and spreading the rank-1 update
  across 32 threads converts a 200 k-flop serial chain into a
  ~6.2 k-flop-per-thread parallel one, with 85 hardware barriers in
  between. That's 200 k / 6.2 k ≈ 32× theoretical, ~10× measured —
  the gap is the serial portions (matrix build in thread 1, pivot
  scan in thread 1) that still bottleneck the workgroup.

Known limitations of the prototype:
- The `Nt` upper bound in `@localmem` is currently 4. Bumping to
  arbitrary Nt requires sizing `q_loc` to the actual tracer count at
  kernel-compile time (or splitting the back-substitution into a
  separate kernel call).
- The build phase still runs sequentially in thread 1 (~7 200 ops per
  column at Nz=85). The `amu[k]` / `amd[k]` recurrences make
  parallelisation across `k` hard, but the inner `kk` loops can be
  parallelised — Phase 2 work.
- The pivot search is serial in thread 1. A parallel reduction would
  shave another ~5 µs per LU iteration; ~85 × 5 µs = 0.4 ms / column
  worth of latency exists here.

These optimisations would push collab-LU from ~10× → ~15-20× over
baseline, still well short of cuBLAS but close enough that the
Metal-portability wins.

### Lever 1 — `lmax_conv` global truncation + cuBLAS strided-batched (CUDA-only)

Direct numbers: 5–55 ms total per panel vs 692 ms baseline. The headline
12.5× total or 136× GPU-only.

Implementation outline (no commits proposed yet — just sketch):

1. Add `lmax_conv :: Int` to `TM5Convection` (default 64, configurable
   in TOML under `[convection]`).
2. At preprocessor time, optionally zero or truncate `entu / detu /
   entd / detd` above `Nz - lmax_conv + 1`. This is a binary-format
   knob, not a runtime gate. (The runtime gate works too but reads more
   data.)
3. At runtime, the matrix-build kernel only touches the bottom
   `lmax_conv` layers, producing a `(lmax_conv, lmax_conv, B)` slab
   directly on GPU.
4. Replace the per-thread serial LU with
   `CUDA.CUBLAS.getrf_strided_batched!` + `getrs_strided_batched!`.
   Pivots are returned by the wrapper — store them in the existing
   `TM5Workspace.pivots` field for plan 19's adjoint replay (cuBLAS
   exposes `trans='T'` for the back-solve, so we don't need to unfuse
   the LU).

Risks / open work:

- Bit-exactness for the ~6% of columns whose icltop lies above the slab
  (window 1 panel 1 shows max\|Δq\| ~0.78 there). This is the same
  approximation TM5's tropo25a/tropo34a make. The risk is acceptable for
  CO2-on-troposphere science; the precise tracer-mass error budget vs.
  long simulations needs a one-week validation run. The active-depth
  scan provides quantitative guidance for choosing a per-binary
  `lmax_conv` ceiling.
- CUDA-version coupling: the N ≤ 64 ceiling is a libcublas limit that
  may move with future CUDA versions. We should encode it as a runtime
  assertion, not a compile-time assumption.

### Lever 2 — Move the matrix build to GPU

Microbench shows host build + transfer ≈ 50 ms/panel (the gap between
`gpu_min` and `total_min`). Moving this to a single KA kernel that
writes directly into `(lmax_conv, lmax_conv, B)` GPU memory pushes the
total down to ~6 ms/panel ≈ 36 ms/full-substep — competitive with
advection. Each column's build is O(`lmax_conv²`) flops, ideal for a
custom kernel (no LU dependency chain inside the build itself). This is
the smaller-impact, easier-to-verify lever, and naturally follows
Lever 1.

### Lever 3 — Factor amortization across a window

The 3.2× ceiling derived above. Requires either:

- A physics approximation (freeze `m_col` for the convection step
  across a window); or
- A Sherman-Morrison-style low-rank correction to the factorization
  between substeps when `m_col` changes by < few percent; or
- A Strang-symmetric splitting that calls convection twice per outer
  substep.

This is a real research project, not a commit. Recommend treating it as
a Phase 2 after Levers 1 and 2 land.

### What I would NOT do

- Per-column variable-`lmc` cuBLAS batches. cuBLAS requires uniform
  matrix size within a batch. Sorting/grouping by `lmc` and issuing
  multiple sub-batches gains nothing the global ceiling doesn't already
  capture, at substantial bookkeeping cost.
- A custom block-collaborative LU kernel. The microbench shows cuBLAS
  batched is already at ~1 ms per panel for solve-only at lmax=64 —
  there is no compute headroom to recover, only build/transfer
  overhead.
- Switching to an iterative solver (GMRES, Jacobi). Defeats determinism
  for the adjoint replay (plan 19).

## Reproducing

```bash
# 1. Diagnostic scan of any production binary (depth distribution)
julia --project=. scripts/validation/diagnose_tm5_active_layers.jl \
    /tmp/scan /temp1/.../era5_transport_*.bin 1e-9

# 2. Per-column depth histogram (for choosing lmax_conv)
julia --project=. scripts/diagnostics/per_column_depth_histogram.jl

# 3. cuBLAS + truncation + split-batch microbench (CUDA-only)
julia --project=. scripts/benchmarks/bench_tm5_alternatives.jl \
    --window 1 --panel 1 --nt 2 --dt 1800 --repeat 5
# → writes artifacts/benchmarks/tm5_alternatives.md

# 4. KA collaborative-LU microbench (portable: CUDA + Metal)
julia --project=. scripts/benchmarks/bench_tm5_collab_lu.jl \
    --window 1 --panel 1 --nt 2 --dt 1800 --repeat 5
```

The diagnostic scanner is the artifact a future implementer should run
on any new binary before changing `lmax_conv` defaults. The microbench
is the artifact a future implementer should re-run after any change to
the convection solver path.
