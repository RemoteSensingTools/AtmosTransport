# Ping-Pong Buffer Refactor — Implementation Plan for Plan Agent

**Status:** Ready for execution.
**Target branch:** `restructure/dry-flux-interface`
**Estimated effort:** 1-2 weeks, single engineer / agent.
**Primary goal:** Eliminate `copyto!` after each advection sweep by
alternating source/destination buffers across the Strang palindrome.

This document is self-contained. Read it top-to-bottom before touching
any code. It takes ~30-45 minutes to read and will save you days of
wrong turns.

---

# Part 1 — Orientation

## 1.1 What this repo is

AtmosTransport.jl is an offline atmospheric transport model. "Offline"
means: meteorological fluxes (mass fluxes, winds) are precomputed from
ERA5 reanalysis and stored in binary files; the model reads them back
and advects tracers through the prepared flow. No online met, no PDE
solver — pure tracer transport on given fluxes.

The model is Julia-first and uses KernelAbstractions.jl for CPU/GPU
portability. Kernels run on CPU (Julia threads) or NVIDIA GPU
(CUDA.jl backend) from the same source code.

Three horizontal grid topologies are supported:
- **LatLonMesh** — regular longitude-latitude, structured `(Nx, Ny)`
- **ReducedGaussianMesh** — Gaussian latitudes with variable
  longitudinal resolution; face-indexed (cells × faces tables)
- **CubedSphereMesh** — six panels, structured within each panel,
  with halo exchange across panel boundaries

Vertical grid: hybrid sigma-pressure levels (Nz cells).

## 1.2 Transport core — directory layout

All advection code lives under:

```
src/Operators/Advection/
├── schemes.jl              — scheme type hierarchy (UpwindScheme, SlopesScheme, PPMScheme)
├── limiters.jl             — branchless slope/moment limiters
├── reconstruction.jl       — @inline face-flux functions dispatched on scheme
├── ppm_subgrid_distributions.jl — PPM ORD=4,5,6,7 edge-value variants
├── structured_kernels.jl   — thin KA kernel shells for LatLon
├── multitracer_kernels.jl  — fused multi-tracer kernel shells (TracerView)
├── MassCFLPilot.jl         — max_cfl_x/y/z utilities
├── StrangSplitting.jl      — orchestrator: sweep functions, subcycling, strang_split!
├── LinRood.jl              — cubed-sphere cross-term (FV3-style) path
└── CubedSphereStrang.jl    — cubed-sphere Strang dispatch
```

**What to read deeply (you'll touch these):**
- `StrangSplitting.jl` — this is the primary file for this work
- `LinRood.jl` — briefly; it reaches into the workspace you'll be
  changing (lines 71-74, 705-711)
- `CubedSphereStrang.jl` — briefly; same (lines 460, 553)

**What to skim (understand the shape):**
- `structured_kernels.jl` — shows the kernel contract
- `multitracer_kernels.jl` — shows the fused 4D pattern

**What NOT to touch:**
- `reconstruction.jl`, `limiters.jl`, `schemes.jl`,
  `ppm_subgrid_distributions.jl`, `MassCFLPilot.jl` — these contain
  the actual advection physics and are already correct for GPU. See
  §2.2 (Out of Scope).

## 1.3 Key data structures

Three types you must know:

**`AdvectionWorkspace`** (`StrangSplitting.jl` lines 62-114).
Pre-allocated buffers for advection. Current fields:
```julia
struct AdvectionWorkspace{FT, A, V1, A4}
    rm_buf        :: A       # 3D tracer-mass double buffer
    m_buf         :: A       # 3D air-mass double buffer
    m_save        :: A       # backup of m for multi-tracer path
    cluster_sizes :: V1      # reduced-grid clustering factors
    face_left     :: V1      # face connectivity (reduced Gaussian)
    face_right    :: V1      # face connectivity (reduced Gaussian)
    rm_4d_buf     :: A4      # 4D tracer double buffer (Nx×Ny×Nz×Nt)
end
```
Constructed once per run (`src/Models/TransportModel.jl` lines 18, 27,
36). Survives the entire simulation. Allocation for 30-tracer C180 at
Float64: ~3.7 GB.

**`CellState{B}`** — wrapper holding:
- `air_mass::AbstractArray{FT,3}` (the `m` you'll see everywhere)
- `tracers::NamedTuple` of tracer-mass arrays (the `rm`s)

**`StructuredFaceFluxState{B}`** — wrapper holding:
- `am::AbstractArray{FT,3}` of size `(Nx+1, Ny, Nz)` — x-face mass fluxes
- `bm::AbstractArray{FT,3}` of size `(Nx, Ny+1, Nz)` — y-face mass fluxes
- `cm::AbstractArray{FT,3}` of size `(Nx, Ny, Nz+1)` — z-face mass fluxes

## 1.4 The Strang palindrome

`strang_split!` (line 1203) runs the sweep sequence:

```
X → Y → Z → Z → Y → X
```

Six sweeps per step. Second-order accurate in splitting error. Each
sweep updates `m` AND one `rm` using the divergence of the face
fluxes.

For multi-tracer runs, the OUTER loop is over tracers, and EACH tracer
gets a full palindrome. Between tracers, `m` is restored from `m_save`
so every tracer sees the same initial air mass. This is the
"per-tracer path."

There's also a multi-tracer fused path (`strang_split_mt!`, line 1386)
that operates on a 4D `rm_4d` array and does ONE palindrome total,
computing the mass update once and looping over tracers inside the
kernel. Both paths exist; this refactor touches both.

## 1.5 The current sweep contract

A sweep function (e.g., `sweep_x!` at line 168):

```julia
function sweep_x!(rm, m, flux, scheme, ws)
    backend = get_backend(m)
    kernel! = _xsweep_kernel!(backend, 256)
    kernel!(ws.rm_buf, rm, ws.m_buf, m, flux, scheme, Int32(size(m, 1)), one(FT);
            ndrange=size(m))
    synchronize(backend)
    copyto!(rm, ws.rm_buf)     # <-- this is what we're eliminating
    copyto!(m,  ws.m_buf)      # <-- and this
    return nothing
end
```

The kernel writes results to `ws.rm_buf` and `ws.m_buf`. Then the
sweep copies them back to `rm` and `m`. This is the double-buffer
contract — the kernel cannot write in-place because its stencil reads
neighbors (`rm[i±1]`, etc.) that need pre-update values.

The inefficiency: those two `copyto!` calls are a full-volume device
memcpy after EVERY sweep. Six sweeps per step. **This is what we're
fixing.**

## 1.6 Key invariants (from CLAUDE.md)

1. **Mass conservation is bit-identical** — total `sum(m)` and
   `sum(rm)` must be preserved to machine precision per step. Tests
   enforce `1e-12` relative for Float64.

2. **GPU kernels are branchless** — `ifelse`, not `if/else`. This is
   already how the existing kernels are written. Do not introduce
   new branches.

3. **CPU and GPU must agree to ≤4 ULP (1-step), ≤16 ULP (4-step)** —
   Float64 and Float32 both. Tests enforce this.

4. **Multi-tracer path must be BIT-IDENTICAL to per-tracer path** —
   `maximum(abs.(rm_4d[:,:,:,t] - rm_ref[t])) == zero(FT)`. Tests
   enforce this. Any arithmetic reordering breaks this test.

5. **In-place kernel updates are forbidden** — they break conservation
   by ~10% per step. The double-buffer contract exists because of
   this. DO NOT attempt to write `rm_new` and `rm` to the same array
   to "save a copyto."

## 1.7 Test suite you'll use

Location: `test/`

**Core tests (no external data, run in <2 minutes on CPU):**
```
julia --project=. test/runtests.jl
```
Runs: `test_basis_explicit_core.jl`, `test_advection_kernels.jl`,
`test_structured_mesh_metadata.jl`, `test_reduced_gaussian_mesh.jl`,
`test_driven_simulation.jl`, `test_cubed_sphere_advection.jl`,
`test_poisson_balance.jl`.

**Real-data tests (require ERA5 binaries at `~/data/AtmosTransport/`):**
```
julia --project=. test/runtests.jl --all
```
Adds: `test_era5_latlon_e2e.jl`, `test_dry_flux_interface.jl`, etc.
You probably don't have the data. Skip unless explicitly told otherwise.

**Most critical test file for this work:**
- `test/test_advection_kernels.jl` — contains ULP-tolerance CPU/GPU
  tests, conservation tests, and the CRITICAL bit-identical MT ≡
  per-tracer test (lines 569-606).

**GPU tests** auto-skip if CUDA isn't available. You probably are
working without a GPU in the dev environment; that's OK — commits
4-5 specifically require a GPU-equipped machine to validate, which
is called out in the commit sequence.

---

# Part 2 — Why This Specific Change

## 2.1 The performance problem

Look at any sweep function. Take `sweep_x!` from line 168:

```julia
kernel!(ws.rm_buf, rm, ws.m_buf, m, flux, ...)
synchronize(backend)
copyto!(rm, ws.rm_buf)
copyto!(m,  ws.m_buf)
```

Each `copyto!` is a full device-side memcpy.

For a representative run size (C180 × 72 levels, Float64):
- `rm`: 180² × 6 panels × 72 × 8 bytes ≈ 112 MB per copy
- `m`: same ≈ 112 MB per copy
- Per sweep: ~224 MB of pure bookkeeping
- Per Strang step (6 sweeps): ~1.35 GB

On an NVIDIA A100 with 1.5 TB/s HBM2e bandwidth, that's ~0.9 ms per
step of pure write-amplification. For comparison, the actual advection
kernel itself for a similar problem size is ~0.4-0.6 ms. **The
double-buffer copy is currently larger than the useful work.**

## 2.2 The fix

Instead of "kernel writes buf, then copy buf to current":

```
Current: kernel(BUF ← RM, RM);   copyto(RM, BUF)
         kernel(BUF ← RM, RM);   copyto(RM, BUF)
         ...
```

Ping-pong: alternate "source" and "destination" across sweeps:

```
Ping-pong: kernel(B ← A, A)  # write to B, read from A
           kernel(A ← B, B)  # write to A, read from B
           kernel(B ← A, A)
           kernel(A ← B, B)
           ...
```

No `copyto!` at all. The kernel's output IS the next kernel's input.

The Strang palindrome has 6 sweeps, which is **even** — so after six
alternations, the result lands back in the array we started with.
This means the caller's `rm` and `m` receive the final result
naturally, no final copy needed. (If this math is ever wrong — e.g.,
if someone adds a 7th sweep — the final result will be in the "wrong"
buffer and tests will fail loudly.)

## 2.3 Why this doesn't break anything

The fundamental correctness question: **does ping-pong change the
arithmetic?**

No. The kernel does exactly the same floating-point operations in
exactly the same order. The only difference is WHICH array those
results get written to. Since the kernel is a pure read-from-source,
write-to-destination operation, and the arithmetic inside is
deterministic, identical inputs produce identical outputs regardless
of which physical array "destination" is.

The ULP-tolerance tests should pass bit-identically after this
refactor. The bit-identical MT ≡ per-tracer test (line 603) should
also pass bit-identically.

## 2.4 Expected speedup

On GPU:
- Write traffic per step: drops from 2× useful to 1× useful (~30-40%
  savings in HBM write traffic)
- Per-step wall time: 20-35% improvement expected for a memory-bound
  workload like this one

On CPU:
- The `copyto!` is still expensive (memcpy) but CPU memory bandwidth
  is much higher relative to CPU compute. Expected improvement:
  5-15% per step.

These are predictions. Benchmark acceptance criteria are in §4.7.

---

# Part 3 — Out of Scope

The agent WILL be tempted to fix these adjacent problems. Don't. Each
is real; each is also a separate project with its own plan.

## 3.1 Do NOT touch

**The limiters** (`_ppm_limit_profile`, `_minmod3`, `_limited_slope`,
`_limited_moment`). They are already branchless via `ifelse` and work
correctly on GPU. An earlier strategic document proposed a "blended"
limiter; that proposal is NOT part of this work and may never be.

**The CFL pilot's dual CPU/GPU algorithm.** `_x_subcycling_pass_count`
(line 957) has a GPU fast-path (static estimate via broadcast) and a
CPU path (evolving-mass iterative). They give different `n_sub`
values. This is a real issue that causes CPU/GPU test drift, but it
is tier-2 work and NOT in this plan. The only interaction with this
plan: the CPU pilot reuses `ws.m_buf`/`ws.rm_buf` as scratch space
(lines 821-822, 885-886, 973-974, 1021-1022, 1069-1070), which we
MUST accommodate in the ping-pong refactor. See §4.4 Commit 2.

**The per-tracer Strang loop vs. multi-tracer default.**
`strang_split!` calls the per-tracer loop by default (line 1219);
`strang_split_mt!` exists but isn't the default. Switching the
default is tier-1 item #3 — a separate plan, not this one.

**The `@atomic` reduced-Gaussian kernel** (line 238). Real
performance issue but separate work.

**`reconstruction.jl`, `limiters.jl`, `schemes.jl`,
`ppm_subgrid_distributions.jl`, `MassCFLPilot.jl` entirely.** Not
touched by this refactor. If you find yourself opening these files,
stop and check why.

## 3.2 Do NOT "clean up" adjacent code

The codebase has some rough edges near what you'll touch:
- Some `@eval` loops generate sweep variants with slightly different
  signatures. Don't refactor them into a cleaner form.
- `AdvectionWorkspace` has a conditional 4D buffer allocation
  (`n_tracers > 0 ? similar(m, ...) : similar(m, 0, 0, 0, 0)`).
  Leave this pattern alone.
- The face-indexed 2D workspace has some `copyto!` calls (lines
  299-300) that initialize buffers; leave those unless they are
  directly in the sweep contract.
- Rename temptation: `rm_buf` is an awkward name, but renaming breaks
  every test that checks `ws.rm_buf`. Leave names alone unless a
  commit explicitly requires renaming.

## 3.3 DO flag, but do NOT fix

If you notice something genuinely broken (not just ugly) — e.g., a
missing synchronize that causes a race, an off-by-one in a stencil,
a test that doesn't actually test what it claims — write it in
`NOTES.md` under "Deferred observations." Do not try to fix it in
this refactor. The scope is ping-pong only.

---

# Part 4 — Implementation Plan

## 4.1 Precondition verification

Before writing any code, run this checklist:

```
# 1. Verify branch
cd $REPO && git branch --show-current
# Expected: restructure/dry-flux-interface

# 2. Verify clean working tree
git status
# Expected: "nothing to commit, working tree clean"

# 3. Run core tests and capture which pass (there may be
# pre-existing failures unrelated to this work — note them)
julia --project=. test/runtests.jl 2>&1 | tee artifacts/baseline_tests.log
# Expected: most tests pass. Any failures should be NOTED in NOTES.md
# as "pre-existing" before you start. If you forget, you'll chase
# them later thinking you caused them.

# 4. Run the benchmark (see §4.6 for the benchmark script). Capture
# baseline timings.
julia --project=. scripts/benchmarks/bench_strang_sweep.jl | tee artifacts/perf/pingpong_before.log
# Expected output format: "Per-step: XX.X ms (median of N)"

# 5. Record git commit hash
git rev-parse HEAD > artifacts/baseline_commit.txt
```

If any of these fail or produce unexpected output, STOP and ask a
human before proceeding.

## 4.2 Change scope — the exact file list

Every file that gets edited, with the function(s) in each:

**`src/Operators/Advection/StrangSplitting.jl`** (the primary file):
- `struct AdvectionWorkspace` (lines 62-70) — add buffer pair B
- `AdvectionWorkspace(m::AbstractArray{FT,3}; ...)` constructor (lines 83-95) — allocate pair B
- `AdvectionWorkspace(m::AbstractArray{FT,2}; ...)` constructor (lines 97-114) — allocate pair B for face-indexed
- `Adapt.adapt_structure` (lines 116-126) — adapt pair B
- Generated `sweep_x!`/`sweep_y!`/`sweep_z!` (lines 147-182) — rewrite to take source/dest buffer pair
- Legacy per-scheme sweeps (lines 195-217) — same
- `_sweep_horizontal_face_gpu!` (lines 291-308) — same pattern
- `_sweep_vertical_face_gpu!` (lines 310-323) — same pattern
- Subcycled sweep overloads (lines 328-347, 409-432) — same
- Face-indexed sweep helpers via `@eval` (lines 591-653) — same
- CFL pilot functions (lines 797-920, 957-1101) — CFL scratch role
  must be disentangled from ping-pong role. See §4.4 Commit 2.
- `strang_split!` for structured (lines 1203-1233) — drives the
  ping-pong state across the palindrome
- Multi-tracer fused sweep functions (lines 1328-1361) — ping-pong
  for 4D buffers
- `strang_split_mt!` (lines 1386-1410) — drives MT ping-pong state
- Multi-tracer apply! via `@eval` (lines 1271-1303) — same pattern

**`src/Operators/Advection/LinRood.jl`**:
- `apply_divergence_damping_cs!` (lines 59-78) — uses `ws.rm_buf`
  directly at lines 71, 74
- `_apply_lin_rood_update_cs!` (lines ~700-715) — uses `ws.rm_buf`
  and `ws.m_buf` directly at lines 705, 710, 711

These call patterns predate ping-pong. You have two choices:
- (a) Preserve the old `ws.rm_buf`/`ws.m_buf` names for these callers
  as aliases that point to pair A's buffers
- (b) Rewrite these callers to use the new API

Choice (a) is lower risk; see §4.3 Decision 4.

**`src/Operators/Advection/CubedSphereStrang.jl`**:
- Lines 460, 553 — same direct buffer access; same treatment as LinRood

**`src/Models/TransportModel.jl`**:
- Lines 18, 27, 36 — constructor call sites. Signatures must remain
  backward-compatible (`AdvectionWorkspace(state.air_mass)` should
  still work).

**`test/test_basis_explicit_core.jl`**:
- Lines 326, 333 — type-check tests (`@test model.workspace.rm_buf
  isa Array{FT,3}`). Update field names if we rename, otherwise no
  change.

## 4.3 Design decisions (pre-answered)

You'll encounter these judgment calls during implementation. Here are
the decisions; do not re-litigate them.

**Decision 1: Two buffer pairs (A and B), not ping-pong by swapping.**
The workspace holds TWO full buffer sets: `(rm_A, m_A)` and
`(rm_B, m_B)`. We could alternatively swap pointers within a single
pair per sweep, but Julia doesn't have mutable struct fields that are
trivially swappable without type-instability concerns. Two named
pairs are boring and correct.

**Decision 2: The "home" buffer is always the one the caller passes in.**
When `sweep_x!` is called with `rm` and `m`, those ARE one of the
buffer pairs (logically pair A). The other pair (pair B) is in
`ws.rm_A_alt`/`ws.m_A_alt` (the buffer-B fields on the workspace).
The sweep must detect which pair is which — see Decision 3.

**Decision 3: Pass source/destination explicitly via a ping-pong
counter, not auto-detected.** Adding a `parity::Bool` argument to
each sweep (or a `pingpong_state::Int` parameter on the workspace)
makes the alternation explicit. `strang_split!` is the single owner
of this state — it tracks which parity each sweep is.

Avoid auto-detection tricks like "whichever array the caller passed
is source, the other is dest" — this couples sweep behavior to
object identity, which is fragile.

Proposed API: add `swap::Bool` or `parity::Int` arg to each sweep
function. When `swap=false`, the sweep does:
```
kernel!(ws.rm_alt, rm, ws.m_alt, m, flux, ...)
# and then swaps rm ↔ ws.rm_alt (array reassignment, not content copy)
```
When `swap=true`, sources and destinations flip.

Concrete mechanism (see §4.4 Commit 3 for the exact code):
```julia
# strang_split! chooses which pair is "live" this sweep
# by passing current & next arrays explicitly.
function sweep_x!(rm_in, rm_out, m_in, m_out, flux, scheme, ws)
    backend = get_backend(m_in)
    kernel! = _xsweep_kernel!(backend, 256)
    kernel!(rm_out, rm_in, m_out, m_in, flux, scheme, Int32(size(m_in, 1)), one(FT);
            ndrange=size(m_in))
    # NO synchronize needed — KA stream ordering is preserved
    # NO copyto! needed — result is already in rm_out/m_out
    return nothing
end
```

`strang_split!` orchestrates by passing the right arrays each call.

**Decision 4: Preserve `ws.rm_buf` and `ws.m_buf` names for
backward-compatibility callers (LinRood, CubedSphereStrang, tests).**

The workspace struct gains new fields but keeps the old ones as
aliases:

```julia
struct AdvectionWorkspace{FT, A, V1, A4}
    rm_A    :: A   # primary tracer-mass buffer (= old rm_buf)
    m_A     :: A   # primary air-mass buffer (= old m_buf)
    rm_B    :: A   # alternate tracer-mass buffer (NEW)
    m_B     :: A   # alternate air-mass buffer (NEW)
    m_save  :: A   # unchanged
    cluster_sizes :: V1
    face_left :: V1
    face_right :: V1
    rm_4d_A :: A4  # primary 4D buffer (= old rm_4d_buf)
    rm_4d_B :: A4  # alternate 4D buffer (NEW)
end

# Backward-compatible aliases via getproperty:
Base.getproperty(ws::AdvectionWorkspace, name::Symbol) =
    name === :rm_buf    ? getfield(ws, :rm_A) :
    name === :m_buf     ? getfield(ws, :m_A) :
    name === :rm_4d_buf ? getfield(ws, :rm_4d_A) :
    getfield(ws, name)
```

This means LinRood.jl, CubedSphereStrang.jl, and
test_basis_explicit_core.jl continue to work unchanged. If the tests
check `ws.rm_buf` they still get pair A's buffer.

**Decision 5: `m_save` does NOT participate in ping-pong.**
It's used by the multi-tracer per-tracer protocol (copy `m` before
each tracer, restore between tracers). Leave it alone.

**Decision 6: Do not change the kernel signatures.**
`_xsweep_kernel!` etc. continue to take `rm_new, rm, m_new, m, ...`
arguments in the same order. Ping-pong is a caller-side concern only.

**Decision 7: Do not remove `synchronize` calls yet.**
The revised GPU optimization doc (which you haven't seen) identifies
`synchronize(backend)` between sweeps as another inefficiency. But
removing them is a separate change with its own risk profile. For
THIS plan, leave synchronize calls in place — just eliminate the
`copyto!` that follow them. Tier-1 item #2 addresses sync removal.

## 4.4 Atomic commit sequence

Each commit is compilable, testable, and independently reversible.
The full test suite must pass after each commit.

### Commit 0: Add benchmark infrastructure

**File:** `scripts/benchmarks/bench_strang_sweep.jl` (NEW)

See §4.6 for the exact script content. This gives you a reproducible
"before" baseline you can compare against.

**Test:**
```
julia --project=. scripts/benchmarks/bench_strang_sweep.jl
```
Expected: runs without error, prints per-step ms. Save output to
`artifacts/perf/pingpong_before.log`.

### Commit 1: Add alternate buffer fields to AdvectionWorkspace

Add `rm_B`, `m_B`, `rm_4d_B` fields. Allocate in both constructors.
Update Adapt. Add `getproperty` backward-compat shim for `rm_buf`,
`m_buf`, `rm_4d_buf`.

Nothing else changes. All existing code still works — new buffers are
allocated but unused.

**Test after commit 1:**
```
julia --project=. test/runtests.jl
```
Expected: ALL tests pass unchanged (the alternate buffers exist but
aren't written to yet).

Memory increase: ~112 MB for 3D buffers (C180), plus Nt × 112 MB for
4D buffer. Check this with a `@time AdvectionWorkspace(m; n_tracers=30)`
from a REPL if you want to see actual numbers.

### Commit 2: Disentangle CFL scratch from ping-pong roles

The CFL pilot functions (lines 821-822, 885-886, 973-974, 1021-1022,
1069-1070) currently do:
```julia
mx = ws.m_buf       # as CPU-path scratch
mx_next = ws.rm_buf # as CPU-path scratch
```

This works today because the pilot runs before the sweep, so the sweep
overwrites the buffers anyway. BUT after ping-pong, the buffers are
carrying REAL DATA between sweeps — the CFL pilot can no longer trash
them.

**Fix:** Add two dedicated scratch fields to AdvectionWorkspace:
- `cfl_scratch_m :: A`  (same size as m)
- `cfl_scratch_rm :: A` (same size as rm)

Allocate in constructors. Update the five CFL pilot functions to use
these new fields instead of `rm_buf`/`m_buf`.

Note: this adds ~224 MB of memory for C180, so total workspace growth
after commits 1+2 is ~450 MB. This is the memory cost of correctness;
it's unavoidable if ping-pong and CFL pilot coexist.

**Test after commit 2:**
```
julia --project=. test/runtests.jl
```
Expected: all tests pass (same as before; CFL pilot still works,
just using different scratch arrays).

### Commit 3: Refactor 3D sweeps to ping-pong

**This is the main commit.** All other commits are setup.

Rewrite `sweep_x!`, `sweep_y!`, `sweep_z!` (the generated ones at
lines 147-182) to accept separate `(rm_in, rm_out, m_in, m_out)`:

```julia
for (sweep_fn, kernel_fn, dim) in (
    (:sweep_x!, :_xsweep_kernel!, 1),
    (:sweep_y!, :_ysweep_kernel!, 2),
    (:sweep_z!, :_zsweep_kernel!, 3),
)
    @eval begin
        function $sweep_fn(rm_in::AbstractArray{FT,3}, rm_out::AbstractArray{FT,3},
                           m_in::AbstractArray{FT,3}, m_out::AbstractArray{FT,3},
                           flux::AbstractArray{FT,3},
                           scheme::AbstractAdvectionScheme,
                           ws::AdvectionWorkspace{FT}) where FT
            backend = get_backend(m_in)
            kernel! = $kernel_fn(backend, 256)
            kernel!(rm_out, rm_in, m_out, m_in, flux, scheme,
                    Int32(size(m_in, $dim)), one(FT);
                    ndrange=size(m_in))
            synchronize(backend)
            # NO copyto! — result is already in rm_out/m_out
            return nothing
        end
    end
end
```

Add flux_scale variants (lines 328-347) with the same pattern:
```julia
function $sweep_fn(rm_in, rm_out, m_in, m_out, flux, scheme, ws, flux_scale)
    ...same but pass flux_scale...
end
```

Keep the OLD sweep_x!(rm, m, flux, scheme, ws) signatures as
backward-compat wrappers that do the ping-pong internally (for
callers like LinRood, CubedSphereStrang, and any tests that call
sweep_x! directly):

```julia
# Backward-compat: write to alternate buffer, then swap (via copyto)
function sweep_x!(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                  flux, scheme, ws::AdvectionWorkspace{FT}) where FT
    sweep_x!(rm, ws.rm_B, m, ws.m_B, flux, scheme, ws)
    copyto!(rm, ws.rm_B)  # intentional: preserves old API semantics
    copyto!(m,  ws.m_B)
    return nothing
end
```

This keeps existing tests green while enabling the ping-pong API for
`strang_split!` to use.

Now rewrite `strang_split!` (line 1203) to use the ping-pong API:

```julia
function strang_split!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
                       grid::AtmosGrid{<:LatLonMesh},
                       scheme::Union{AbstractAdvection, AbstractAdvectionScheme};
                       workspace::AdvectionWorkspace,
                       cfl_limit::Real = one(eltype(state.air_mass))) where {B <: AbstractMassBasis}
    m = state.air_mass
    am, bm, cm = fluxes.am, fluxes.bm, fluxes.cm
    cfl_limit_ft = convert(eltype(m), cfl_limit)
    n_tr = length(state.tracers)
    m_save = workspace.m_save
    if n_tr > 1
        copyto!(m_save, m)
    end

    for (idx, (name, rm_tracer)) in enumerate(pairs(state.tracers))
        if idx > 1
            copyto!(m, m_save)
        end

        # Six-sweep palindrome with ping-pong.
        # Buffer A = caller's (rm_tracer, m).
        # Buffer B = (ws.rm_B, ws.m_B).
        # After 6 swaps, result lands back in A (caller's buffers).
        #
        # Swap plan: A→B, B→A, A→B, B→A, A→B, B→A  (6 alternations)
        # Note: if subcycling makes the sweep count per direction vary,
        # this gets more complex. See _sweep_x_subcycled_pingpong! below.

        _sweep_x_subcycled_pingpong!(rm_tracer, workspace.rm_B, m, workspace.m_B,
                                      am, scheme, workspace, cfl_limit_ft)
        # After X: final result is in (rm_tracer, m) if n_x is even,
        # else in (ws.rm_B, ws.m_B). The subcycled wrapper handles this.
        _sweep_y_subcycled_pingpong!(rm_tracer, workspace.rm_B, m, workspace.m_B,
                                      bm, scheme, workspace, cfl_limit_ft)
        _sweep_z_subcycled_pingpong!(rm_tracer, workspace.rm_B, m, workspace.m_B,
                                      cm, scheme, workspace, cfl_limit_ft)
        _sweep_z_subcycled_pingpong!(rm_tracer, workspace.rm_B, m, workspace.m_B,
                                      cm, scheme, workspace, cfl_limit_ft)
        _sweep_y_subcycled_pingpong!(rm_tracer, workspace.rm_B, m, workspace.m_B,
                                      bm, scheme, workspace, cfl_limit_ft)
        _sweep_x_subcycled_pingpong!(rm_tracer, workspace.rm_B, m, workspace.m_B,
                                      am, scheme, workspace, cfl_limit_ft)
    end

    return nothing
end
```

Each `_sweep_*_subcycled_pingpong!` is a new helper (replacing the
current `_sweep_x_subcycled!`) that:
1. Computes n_sub via existing CFL pilot
2. Runs n_sub ping-pong kernel launches, alternating source/dest
3. Ensures final result lands in the caller's A-arrays (use an extra
   swap if n_sub is odd)

Reference implementation:

```julia
@inline function _sweep_x_subcycled_pingpong!(rm_A, rm_B, m_A, m_B,
                                               am, scheme, ws, cfl_limit)
    n_sub = _x_subcycling_pass_count(am, m_A, ws, cfl_limit)
    if n_sub == 1
        sweep_x!(rm_A, rm_B, m_A, m_B, am, scheme, ws)
        # Result in rm_B / m_B. Need it back in A for next sweep.
        copyto!(rm_A, rm_B)
        copyto!(m_A,  m_B)
        return 1
    end
    flux_scale = inv(eltype(m_A)(n_sub))
    # Alternate directions. Start with A→B.
    for pass in 1:n_sub
        if isodd(pass)
            sweep_x!(rm_A, rm_B, m_A, m_B, am, scheme, ws, flux_scale)
        else
            sweep_x!(rm_B, rm_A, m_B, m_A, am, scheme, ws, flux_scale)
        end
    end
    # If n_sub is odd, final result is in B; copy back to A.
    if isodd(n_sub)
        copyto!(rm_A, rm_B)
        copyto!(m_A,  m_B)
    end
    return n_sub
end
```

**CAUTION:** Note this still has `copyto!` calls inside each
subcycled sweep function. The `copyto!` moves OUT OF the inter-sweep
interface and INTO the subcycle boundary (only when n_sub is odd or
n_sub == 1). For n_sub == 1 (the common case for most runs), we
UNAVOIDABLY have one copyto per sweep, which... gives us no win?

Wait, let me reconsider. For n_sub == 1, the sweep is:
```
sweep_x!(A→B)
copyto(A ← B)   # this is what we're trying to avoid!
```

The expected case (most sweeps have n_sub == 1) still has a copyto.
We haven't saved anything.

**Revised approach:** push the ping-pong up a level. The palindrome
orchestrator tracks WHICH pair is current, and only does one final
copy if needed. Concretely:

```julia
# At start: caller's (rm, m) are in pair A.
current_is_A = true

# X sweep (n_sub_x subcycles). Each subcycle alternates.
# After n_sub_x subcycles, current_is_A flips iff n_sub_x is odd.
current_is_A = run_subcycled_x!(ws, am, ...)  # returns new parity

# Same for Y, Z, Z, Y, X.

# After 6 full-direction sweeps, if current_is_A is false,
# we need to copy alt → A before returning.
if !current_is_A
    copyto!(m, ws.m_B)
    for rm in state.tracers
        copyto!(rm, ws.rm_B)  # wait, this is wrong — each tracer
                              # goes through its own palindrome
    end
end
```

For a typical run with n_sub == 1 for all six directions:
- Parity flips 6 times (even) → final is in A
- ZERO `copyto!` calls
- This is the win

For a run with n_sub == 2 on all six directions:
- Parity flips 12 times (even) → final is in A
- Zero copyto calls inside the palindrome
- Win again

For odd subcycle counts in some direction:
- An interior copy may be needed OR a final copy is needed
- But this is 1 copy per palindrome at worst, not 6

**Net savings:** in the common case (n_sub == 1 everywhere), 6
copyto operations eliminated per Strang step per tracer.

The simpler implementation that achieves this:

```julia
function strang_split!(state, fluxes, grid, scheme; workspace, cfl_limit=...)
    m = state.air_mass
    # ...
    for (idx, (_, rm_tracer)) in enumerate(pairs(state.tracers))
        if idx > 1
            copyto!(m, m_save)
        end

        # Track which of (A, B) holds the current state.
        # A = (rm_tracer, m) -- caller-provided
        # B = (workspace.rm_B, workspace.m_B)
        # We track parity implicitly by swapping variable bindings.

        rm_cur, m_cur = rm_tracer, m
        rm_alt, m_alt = workspace.rm_B, workspace.m_B

        n_x1 = _sweep_x_pp!(rm_cur, rm_alt, m_cur, m_alt, am, scheme, workspace, cfl_limit)
        if isodd(n_x1); rm_cur, rm_alt = rm_alt, rm_cur; m_cur, m_alt = m_alt, m_cur; end

        n_y1 = _sweep_y_pp!(rm_cur, rm_alt, m_cur, m_alt, bm, scheme, workspace, cfl_limit)
        if isodd(n_y1); rm_cur, rm_alt = rm_alt, rm_cur; m_cur, m_alt = m_alt, m_cur; end

        n_z1 = _sweep_z_pp!(rm_cur, rm_alt, m_cur, m_alt, cm, scheme, workspace, cfl_limit)
        if isodd(n_z1); rm_cur, rm_alt = rm_alt, rm_cur; m_cur, m_alt = m_alt, m_cur; end

        n_z2 = _sweep_z_pp!(rm_cur, rm_alt, m_cur, m_alt, cm, scheme, workspace, cfl_limit)
        if isodd(n_z2); rm_cur, rm_alt = rm_alt, rm_cur; m_cur, m_alt = m_alt, m_cur; end

        n_y2 = _sweep_y_pp!(rm_cur, rm_alt, m_cur, m_alt, bm, scheme, workspace, cfl_limit)
        if isodd(n_y2); rm_cur, rm_alt = rm_alt, rm_cur; m_cur, m_alt = m_alt, m_cur; end

        n_x2 = _sweep_x_pp!(rm_cur, rm_alt, m_cur, m_alt, am, scheme, workspace, cfl_limit)
        if isodd(n_x2); rm_cur, rm_alt = rm_alt, rm_cur; m_cur, m_alt = m_alt, m_cur; end

        # If rm_cur is not the caller's rm_tracer, copy back.
        if rm_cur !== rm_tracer
            copyto!(rm_tracer, rm_cur)
            copyto!(m, m_cur)
        end
    end

    return nothing
end
```

Where `_sweep_x_pp!(rm_in, rm_out, m_in, m_out, ...)` alternates
inside the subcycle loop and returns n_sub.

**This is the correct pattern.** It cleanly handles any n_sub per
direction and puts at most one copyto per Strang step (only if the
total parity is odd).

**Type stability note:** `rm_cur` and `rm_alt` both have the same
concrete type (they're both `A` from the workspace or the caller's
`rm_tracer`, which has the same underlying array type). The
reassignment should not cause type instability. Verify with
`@code_warntype strang_split!(...)` after this commit.

**Test after commit 3:**
```
julia --project=. test/runtests.jl
```
Expected: ALL tests pass, ULP tolerances unchanged, mass
conservation unchanged, bit-identical MT ≡ per-tracer test still
passes.

Run benchmark:
```
julia --project=. scripts/benchmarks/bench_strang_sweep.jl > artifacts/perf/pingpong_after_commit3.log
```
Expected: per-step time drops by 15-25% on CPU (memory-bound);
GPU verification requires a GPU-equipped machine.

### Commit 4: Refactor multi-tracer fused sweeps to ping-pong

Same pattern as Commit 3, but for `sweep_x_mt!`, `sweep_y_mt!`,
`sweep_z_mt!` and `strang_split_mt!`.

Key difference: the 4D tracer buffer (`ws.rm_4d_A` and `ws.rm_4d_B`)
is on the workspace; the caller passes `rm_4d` which logically is
pair A.

```julia
function strang_split_mt!(rm_4d, m, am, bm, cm, scheme, ws; cfl_limit=...)
    rm_cur, rm_alt = rm_4d, ws.rm_4d_B
    m_cur,  m_alt  = m, ws.m_B

    n_x = _x_subcycling_pass_count(am, m_cur, ws, cfl_limit)
    n_y = _y_subcycling_pass_count(bm, m_cur, ws, cfl_limit)
    n_z = _z_subcycling_pass_count(cm, m_cur, ws, cfl_limit)

    for _ in 1:n_x
        sweep_x_mt!(rm_cur, rm_alt, m_cur, m_alt, am, scheme, ws, inv(FT(n_x)))
        rm_cur, rm_alt = rm_alt, rm_cur; m_cur, m_alt = m_alt, m_cur
    end
    # ...(same for y, z, z, y, x)...

    if rm_cur !== rm_4d
        copyto!(rm_4d, rm_cur)
        copyto!(m,     m_cur)
    end
    return nothing
end
```

**Test after commit 4:** bit-identical MT test (line 603) must
still produce `max_diff == zero(FT)`. This is the test that WILL
fail if the ping-pong reordering accidentally changes arithmetic.

### Commit 5: Update Lin-Rood and Cubed-Sphere paths

These paths access `ws.rm_buf`/`ws.m_buf` directly. Thanks to the
getproperty shim from Commit 1, these still work (aliasing to
`rm_A`/`m_A`). But we should audit:

- `LinRood.jl` line 71: writes to `ws.rm_buf` (= ws.rm_A) then
  copies interior back. Ping-pong isn't used here — this path runs
  its own sequence. Fine as-is.
- `LinRood.jl` lines 705-711: same — single-shot write + interior
  copy. Not ping-ponging. Fine.
- `CubedSphereStrang.jl` line 460, 553: direct buffer access; not
  ping-ponging. Fine.

**Test after commit 5:**
```
julia --project=. test/test_cubed_sphere_advection.jl
```
Expected: all tests pass. The Lin-Rood code path is untouched except
for the aliased field access.

### Commit 6: Finalize benchmarks and documentation

- Run `scripts/benchmarks/bench_strang_sweep.jl` on CPU and (if
  available) GPU
- Save results to `artifacts/perf/pingpong_after/`
- Write a brief markdown summary (`artifacts/perf/pingpong_summary.md`):
  - Hardware used
  - Per-step time before/after for each test size
  - Percentage improvement
  - Any regressions noticed

- Update any comments in `StrangSplitting.jl` that reference the
  old `copyto!` pattern
- Update the CLAUDE.md double-buffer invariant to reflect ping-pong

## 4.5 Test plan per commit

After EACH commit above, run this exact sequence:

```bash
# 1. Compile check
julia --project=. -e 'using AtmosTransport'
# Expected: silent success

# 2. Core test suite
julia --project=. test/runtests.jl
# Expected: every test that passed in baseline still passes

# 3. Specifically the advection kernel tests
julia --project=. test/test_advection_kernels.jl
# Expected: ULP tolerances unchanged, conservation tolerances
# unchanged, bit-identical MT ≡ per-tracer test passes

# 4. Specifically the type-check tests
julia --project=. test/test_basis_explicit_core.jl
# Expected: passes (workspace still exposes rm_buf via getproperty)
```

If a GPU is available:
```bash
# 5. Run with CUDA enabled (if available)
HAS_GPU=true julia --project=. test/test_advection_kernels.jl
# Expected: GPU tests pass, ULP tolerances unchanged
```

**Stop conditions:**
- Any test that passed in baseline now fails → STOP, revert, investigate
- ULP tolerance exceeds thresholds → STOP, revert; the arithmetic
  changed somewhere
- Mass conservation regresses → STOP, revert; buffer role got
  confused (something is being read before written, or vice versa)

## 4.6 The benchmark script

**File:** `scripts/benchmarks/bench_strang_sweep.jl`

```julia
#!/usr/bin/env julia
# ===========================================================================
# Strang split per-step benchmark
#
# Measures per-step wall time for strang_split! on a synthetic LatLon
# problem. Use this to compare ping-pong before/after.
#
# Usage:
#   julia --project=. scripts/benchmarks/bench_strang_sweep.jl [size]
# where size ∈ {small, medium, large}
# ===========================================================================

using AtmosTransport
using AtmosTransport.Operators.Advection: AdvectionWorkspace, strang_split!,
    UpwindScheme, SlopesScheme, PPMScheme, MonotoneLimiter
using AtmosTransport.Grids: LatLonMesh, AtmosGrid, HybridSigmaPressure,
    cell_areas_by_latitude, gravity, reference_pressure, level_thickness
using AtmosTransport.States: CellState, StructuredFaceFluxState
using Statistics
using Printf

const SIZE = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :medium

const DIMS = Dict(
    :small  => (Nx=72,  Ny=36,  Nz=4,  Nt=1,  n_steps=50),
    :medium => (Nx=288, Ny=144, Nz=32, Nt=5,  n_steps=20),
    :large  => (Nx=576, Ny=288, Nz=72, Nt=10, n_steps=10),
)[SIZE]

function build_problem(FT, cfg)
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
    A_ifc = FT.(vcat([0], range(100, 50000, length=Nz - 1), [0]))
    B_ifc = FT.(vcat([0], range(0.0, 1.0, length=Nz)))
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    g  = gravity(grid)
    ps = reference_pressure(grid)
    areas = cell_areas_by_latitude(mesh)

    m = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        m[i, j, k] = FT(level_thickness(vc, k, ps)) * areas[j] / g
    end

    # Multi-tracer: each tracer gets a different gradient.
    rms = [similar(m) for _ in 1:cfg.Nt]
    for t in 1:cfg.Nt
        χ0 = FT(100e-6) * t
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            lat_frac = FT(j - 1) / FT(Ny - 1)
            rms[t][i, j, k] = m[i, j, k] * (χ0 + FT(200e-6) * lat_frac)
        end
    end

    # Synthetic fluxes with target CFL ~ 0.3
    m_min = minimum(m)
    cfl = FT(0.3)
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:(Nx + 1)
        lat = FT(-90) + FT(j - FT(0.5)) * (FT(180) / FT(Ny))
        cos_lat = cos(lat * FT(π) / FT(180))
        am[i, j, k] = cfl * m_min * cos_lat
    end
    am[:, 1, :]  .= zero(FT)
    am[:, Ny, :] .= zero(FT)
    for k in 1:Nz, j in 2:Ny, i in 1:Nx
        lat = FT(-90) + FT(j - 1) * (FT(180) / FT(Ny))
        bm[i, j, k] = cfl * m_min * FT(0.1) * sin(FT(2) * lat * FT(π) / FT(180))
    end
    cm = zeros(FT, Nx, Ny, Nz + 1)

    return grid, m, rms, am, bm, cm
end

function run_benchmark(FT, scheme, cfg)
    grid, m, rms, am, bm, cm = build_problem(FT, cfg)

    tracers = NamedTuple{Tuple(Symbol("tr$t") for t in 1:cfg.Nt)}(
        Tuple(copy(rm) for rm in rms))
    state = CellState(copy(m); tracers=tracers)
    fluxes = StructuredFaceFluxState(copy(am), copy(bm), copy(cm))
    ws = AdvectionWorkspace(state.air_mass)

    # Warmup
    for _ in 1:3
        strang_split!(state, fluxes, grid, scheme; workspace=ws)
    end

    # Timed
    times = Float64[]
    for _ in 1:cfg.n_steps
        t = @elapsed strang_split!(state, fluxes, grid, scheme; workspace=ws)
        push!(times, t)
    end

    return median(times) * 1e3, mean(times) * 1e3, std(times) * 1e3
end

function main()
    @printf("Strang sweep benchmark (%s: Nx=%d Ny=%d Nz=%d Nt=%d)\n",
            SIZE, DIMS.Nx, DIMS.Ny, DIMS.Nz, DIMS.Nt)
    @printf("%-10s %-10s %10s %10s %10s\n",
            "FT", "Scheme", "median(ms)", "mean(ms)", "std(ms)")
    println("─" ^ 60)

    for FT in (Float64, Float32)
        for (scheme, tag) in (
            (UpwindScheme(), "Upwind"),
            (SlopesScheme(MonotoneLimiter()), "Slopes"),
            (PPMScheme(MonotoneLimiter()), "PPM"),
        )
            med, avg, sd = run_benchmark(FT, scheme, DIMS)
            @printf("%-10s %-10s %10.3f %10.3f %10.3f\n",
                    FT, tag, med, avg, sd)
        end
    end
end

main()
```

This script:
- Builds a purely synthetic problem (no data dependencies)
- Uses realistic hybrid-sigma vertical coordinates
- Scales by flag to small/medium/large
- Runs Upwind/Slopes/PPM in both Float64/Float32
- Reports median, mean, std to detect variance

Put it at `scripts/benchmarks/bench_strang_sweep.jl` and commit as
part of Commit 0.

## 4.7 Acceptance criteria

Ship when ALL of:

**Correctness (hard requirements):**
- Every test that passed pre-refactor passes post-refactor, unchanged
- ULP tolerances match baseline:
  - Float64 1-step CPU-GPU: ≤ 4 ULP (matches current limit)
  - Float64 4-step CPU-GPU: ≤ 16 ULP
  - Float32 same tolerances
- Mass conservation matches baseline:
  - Float64: `abs(sum - sum_init) / sum_init < 1e-12` (4-step)
  - Float32: `< 5e-5`
- Bit-identical MT ≡ per-tracer test passes with `max_diff ==
  zero(FT)` (this is the strictest constraint)

**Performance (target):**
- Per-step wall time on CPU medium config: ≥10% improvement over
  baseline (CPU is less memory-bound, so smaller win expected)
- Per-step wall time on GPU medium config: ≥20% improvement
  (larger win expected on GPU; if GPU not available, CPU result
  is sufficient for this plan)
- If improvement is <5% on either CPU or GPU, something is wrong
  (either the change didn't take effect, or there's a subtle
  inefficiency elsewhere masking the savings). Investigate
  before shipping.

**Memory (soft requirement):**
- Workspace allocation at C180/30-tracer: ≤1.5× baseline
  (pair A + pair B + CFL scratch is additive, so ~2x is the
  expected upper bound at full 4D scale; ≤1.5× means something
  failed to allocate and should be investigated)

**Code quality:**
- No new compile-time warnings
- `@code_warntype strang_split!(...)` shows no new instabilities
- `using AtmosTransport` still works without errors

## 4.8 Rollback plan

If commit N causes a test regression that isn't fixable in ≤30 min:

1. `git reset --hard HEAD~1` (or whatever commit was last good)
2. `julia --project=. test/runtests.jl` to confirm you're back to
   a known-good state
3. Write the failure in `NOTES.md`:
   - What test failed
   - Exact error message
   - What change you think caused it
   - What you'll try differently next time
4. STOP. Do not immediately retry. Take a break or ask a human.

Do NOT "fix forward" — do not add more commits to try to fix a
failing commit. Revert first, understand, then retry.

## 4.9 Known pitfalls

Things the agent WILL be tempted to do, with reasons not to:

1. **"I'll just remove the `synchronize(backend)` while I'm here."**
   NO. That's tier-1 item #2, a separate plan. Leave synchronize.

2. **"I'll rename `rm_buf` to `primary_tracer_buf` for clarity."**
   NO. Keeps the diff narrow and the revert clean.

3. **"I'll merge the per-tracer Strang loop with the multi-tracer
   fused path since I'm refactoring the sweeps anyway."**
   NO. That's tier-1 item #3.

4. **"This CFL pilot code is ugly; I'll clean it up."**
   NO. Only change what's required for ping-pong to work.

5. **"I'll make the `rm_B` field Optional to save memory when not
   needed."** NO. Type instability + special-case code paths.
   Just allocate.

6. **"I'll use mutable struct for AdvectionWorkspace so I can swap
   fields in place."** NO. Immutable is fine; swap variable
   bindings locally in `strang_split!`.

7. **"This `@eval` loop is confusing; I'll expand it."** NO. It
   generates the per-direction sweep variants. Leave the generator.

8. **"I noticed the cubed-sphere path is not implemented for new
   schemes — I'll add that."** NO. Out of scope.

9. **"The `?:` in `_vertical_face_kernel!` should be `ifelse`."**
   NO. There's a comment explaining it — `?:` intentionally avoids
   OOB evaluation.

10. **"I'll sneak in a quick fix for [anything]."** NO. Scope
    discipline matters more than being clever.

---

# Part 5 — How to Work

## 5.1 Start of each session

1. `cd $REPO && git status` — know what's pending
2. `git log --oneline -10` — remember what's been done
3. Open `NOTES.md` — what was I thinking last time?
4. Read §4.4 to find the next commit in the sequence
5. Run the test suite once to confirm baseline: `julia --project=.
   test/runtests.jl` — catches environmental regressions (Julia
   update, dependency change, etc.)

## 5.2 During work

- **One commit at a time.** Do not start commit 4 while commit 3
  is in progress or unverified.
- **Test after every non-trivial edit.** Running tests every 10
  minutes catches mistakes while the change is fresh.
- **Write down decisions.** If you made a judgment call that §4.3
  didn't cover, write it in `NOTES.md` as "Decision N: …" so
  future-you and reviewers can find it.
- **Write down surprises.** If something was not as §1 described
  (e.g., "I found a third place that touches `rm_buf`"), write it
  in `NOTES.md`. This is how we improve the plan for future
  refactors.

## 5.3 End of each session

1. Commit whatever's done. No uncommitted changes overnight.
   (Exception: you can stash uncommitted experimental work in a
   branch, but tag it clearly as "WIP, not part of plan.")
2. Update `NOTES.md` with a "Next session: …" line. Tell future-you
   exactly where to pick up.
3. If there's an open question, write it down clearly in `NOTES.md`.

## 5.4 When to stop and ask a human

ANY of these:
- A test that passed before now fails and you can't explain why
  in 30 minutes
- Your diff is growing to touch files not in §4.2
- Benchmark shows a regression or no improvement
- You discover that §1 (orientation) is factually wrong about
  something material
- You want to deviate from §4.3 (decisions) — write the alternative
  and rationale in `NOTES.md`, then ask
- The scope is expanding — you think you need to do "one more
  thing" beyond §4.4

Do not just silently expand scope. The plan works because it's
bounded.

## 5.5 NOTES.md template

Keep this file in the working directory. Format:

```markdown
# Ping-Pong Refactor — Working Notes

## Last session: [date]
[One paragraph on what was done, what state things are in]

## Next session: start here
[Specific file/function/commit to begin with]

## Decisions made beyond the plan
1. [date] Decision: [what you decided and why]

## Deferred observations
1. [thing you noticed was broken/suboptimal but didn't fix]

## Open questions
1. [thing you're unsure about — review with a human]

## Benchmark results
[Copy-paste from each run]

## Test anomalies
[Any test that behaved oddly, even if it passed]
```

Update this file DURING work, not at the end. If you think "I should
remember this" — write it down right then.

---

# End of Plan

You now have everything you need to execute the ping-pong refactor.
Read the whole document once before starting. Read §4.3 (decisions)
and §4.9 (pitfalls) every time you start a session.

Good luck. Ask questions before acting if anything is unclear.
