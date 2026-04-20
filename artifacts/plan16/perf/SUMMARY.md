# Plan 16b Commit 6 — diffusion overhead benchmarks

Host: wurst.gps.caltech.edu (NVIDIA L40S, F32-only path). Configs via
`scripts/benchmarks/bench_diffusion_overhead.jl`.

## Wall-clock per-step time (GPU, Float32, cfl = 0.4)

| Size   | Grid          | Scheme  | Nt |  Baseline (ms) |  Constant (ms) |  Profile (ms) |  PreComp (ms) | max Δ% |
|--------|---------------|---------|---:|---------------:|---------------:|--------------:|--------------:|-------:|
| small  | 72×36×4       | upwind  |  5 |          0.221 |          0.229 |         0.235 |         0.234 |  6.12% |
| small  | 72×36×4       | upwind  | 10 |          0.235 |          0.246 |         0.250 |         0.251 |  6.70% |
| small  | 72×36×4       | slopes  |  5 |          0.238 |          0.250 |         0.253 |         0.250 |  6.05% |
| small  | 72×36×4       | slopes  | 10 |          0.273 |          0.287 |         0.287 |         0.287 |  5.11% |
| medium | 288×144×32    | upwind  |  5 |          0.671 |          0.715 |         0.714 |         0.748 | 11.41% |
| medium | 288×144×32    | upwind  | 10 |          1.322 |          1.537 |         1.536 |         1.602 | 21.23% |
| medium | 288×144×32    | slopes  |  5 |          1.277 |          1.354 |         1.356 |         1.391 |  8.93% |
| medium | 288×144×32    | slopes  | 10 |          2.276 |          2.506 |         2.509 |         2.550 | 12.03% |
| large  | 576×288×72    | upwind  |  5 |          7.022 |         10.486 |        10.480 |        10.938 | 55.77% |
| large  | 576×288×72    | upwind  | 10 |         11.417 |         18.727 |        18.759 |        20.060 | 75.70% |
| large  | 576×288×72    | slopes  |  5 |         10.246 |         13.657 |        13.672 |        14.105 | 37.66% |
| large  | 576×288×72    | slopes  | 10 |         18.329 |         25.063 |        25.039 |        26.304 | 43.52% |

`baseline` = `NoDiffusion` (dead branch).
`Constant`  = `ImplicitVerticalDiffusion{ConstantField{Float32, 3}}`.
`Profile`   = `ImplicitVerticalDiffusion{ProfileKzField{Float32, CuArray{Float32, 1}}}`.
`PreComp`   = `ImplicitVerticalDiffusion{PreComputedKzField{Float32, CuArray{Float32, 3}}}`.

## Observations

1. **All three Kz-field backings cost about the same** at a given grid
   size + Nt. `ConstantField` (scalar) and `ProfileKzField` (length-Nz
   vector, device-side via Adapt) are indistinguishable; `PreComputedKzField`
   (full 3D) is 2–6% more expensive, consistent with the extra memory
   traffic. So field-type choice does not materially move the needle —
   the cost comes from the Thomas solve, not from `field_value`.

2. **Overhead scales with Nz.** Small (Nz=4) is 5–7%. Medium (Nz=32)
   is 9–21%. Large (Nz=72) is 38–76%. Per-column Thomas is sequential
   across k — at Nz=72 each thread does ~144 fp ops serially before
   back-substitution. More columns (larger Nx·Ny·Nt) gives more
   parallelism but the column depth is the critical path.

3. **Upwind is the stress case.** Slopes advection already costs more
   per sweep, so the diffusion absolute cost (~8 ms at large Nt=10) is
   a smaller fraction of the total. Upwind large Nt=10 is the worst
   case: 76% overhead on top of a fast baseline.

4. **Plan target comparison.** Plan 16b §4.6 targets ≤30% overhead on
   GPU at Nt=10, cfl=0.4. Met for small + medium grids. Exceeded on
   large grid (upwind 76%, slopes 44%). This is a **soft** target per
   §4.6 ("Performance (soft)"), so the result is not a shipping blocker
   — but future work should address the large-Nz case.

## ProfileKzField GPU dispatch (plan pitfall 11 resolved)

The three options from the plan:
1. NTuple{Nz, FT} materialization (compile-time bound Nz)
2. Adapt.jl pattern
3. 3D-cache materialization (wasteful)

Shipped: **Option 2**. `ProfileKzField{FT, V}` is parametric on the
vector type `V <: AbstractVector{FT}`. An `Adapt.adapt_structure`
method moves the backing vector to the device when the Kernel
Abstractions infrastructure walks the kernel's arguments. Same
pattern extended to `PreComputedKzField{FT, A}`.

Result: no observable performance difference between `ConstantField`
(scalar, trivially bits) and `ProfileKzField` (Nz-length
`CuArray{FT, 1}`). Option 2 is the right choice; options 1 and 3
are not needed.

## Scope for future optimization (out of plan 16b)

Large-grid overhead is dominated by column-serial Thomas. Options:
- **Multi-tracer fusion inside the kernel**: current ndrange is
  `(Nx, Ny, Nt)`. A `(Nx, Ny)` ndrange with a Nt-loop inside the
  thread would reuse the Thomas coefficients across tracers — one
  column build + one factorization + Nt back-substitutions — saving
  most of the arithmetic at Nt > 1.
- **Shared-memory Thomas**: for moderate Nz, stage `w[k]` in shared
  memory instead of global `w_scratch`, reducing DRAM round-trips.
- **Persistent w_scratch across timesteps** when Kz and dz are
  time-constant — reuse the factorization.

None of these change the operator's interface; all are within-kernel
optimizations. The current implementation is clean and adjoint-ready;
optimization can be layered on in a dedicated plan once operational
needs motivate it.
