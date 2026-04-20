# TM5 Convection — Upstream Fortran Reference Notes

**Source:** TM5-4DVAR source code (tm5-4dvar.zip, Zenodo record 11243627),
`toupload/base/src/`.

**Companion files in repo:**
- `tm5_conv.F90` (344 lines) — matrix builder + LU solver interface
- `convection.F90` (750 lines) — driver, met-field handling, OpenMP loop
- `tmphys_convec.F90` (138 lines) — cloud dimensions (ConvCloudDim)
- `tm5_convdiff.F90` (492 lines) — combined convection+diffusion variant
  (NOT used by AtmosTransport plan 18 — we ship convection separately
  from diffusion per plan 16b)

**Purpose of these notes:**
Comparison reference for plan 18 port. Per CLAUDE_additions_v2
§ "Validation discipline for physics ports", legacy Julia is
starting point, not ground truth. This document captures the
Fortran reference for independent validation.

---

## 1. Algorithm summary

**Scheme:** Tiedtke 1989 mass-flux convection with updraft +
downdraft mass-flux matrix.

**Inputs (per column):**
- `entu(lmx)` — updraft entrainment [kg/m²/s]
- `detu(lmx)` — updraft detrainment [kg/m²/s]
- `entd(lmx)` — downdraft entrainment [kg/m²/s]
- `detd(lmx)` — downdraft detrainment [kg/m²/s]
- `m(lmx)` — air mass per layer [kg/m²]
- `ld` — level of free sinking (start of cumulus downdraft)
- `li` — cloud top (highest level with detu > 0)
- `dt` — timestep [s]

**Output:** `conv1(lmx, lmx)` — implicit-solve matrix such that
`conv1 * rm_new = rm_old`, where `conv1 = I - dt * D` and D is
the flux-divergence operator.

**Application:** LU factorize `conv1` (LAPACK dgetrf), then
`dgetrs(trans='N', ...)` for forward integration. `trans='T'`
for reverse (adjoint) integration — so adjoint is literally the
LU transpose.

## 2. Level convention

**TM5 internal:** k=1 = surface (bottom), k=lmx = TOA (top).

This is the OPPOSITE of AtmosTransport's convention
(k=1=TOA, k=Nz=surface, CLAUDE.md Invariant 2).

**Port must reverse the column at the TM5/AtmosTransport
interface.** Legacy Julia does this at
`tm5_matrix_convection.jl:17`. Plan 18 preserves.

## 3. Matrix builder (TM5_Conv_Matrix, tm5_conv.F90:37-191)

### 3.1 Updraft section (lines 97-118)

Fortran with 0-based `amu(0:lmx)`, `fu(0:lmx, 1:lmx)`:

```fortran
amu(0) = 0.0
fu(0, :) = 0.0
do k = 1, li
    amu(k) = amu(k-1) + entu(k) - detu(k)
    if (amu(k) > 0.0) then
        zxi = max(0.0, 1.0 - detu(k)/(amu(k-1) + entu(k)))
    else
        amu(k) = 0.0
        zxi = 0.0
    end if
    do kk = 1, k-1
        fu(k, kk) = fu(k-1, kk) * zxi
    end do
    fu(k, k) = entu(k) / m(k) * zxi
end do
```

### 3.2 Downdraft section (lines 122-135)

```fortran
amd(lmx) = 0.0
f(:, :) = 0.0
do k = ld, 2, -1
    amd(k-1) = amd(k) - entd(k) + detd(k)
    if (amd(k-1) < 0.0) then
        zxi = max(0.0, 1.0 + detd(k)/(amd(k) - entd(k)))
    else
        amd(k-1) = 0.0
        zxi = 0.0
    end if
    do kk = k+1, ld
        f(k-1, kk) = f(k, kk) * zxi
    end do
    f(k-1, k) = -entd(k) / m(k) * zxi
end do
```

### 3.3 Assembly — combine updraft + downdraft + subsidence (lines 140-149)

```fortran
do k = 1, lmx-1
    do kk = 1, lmx
        f(k, kk) = fu(k, kk) + f(k, kk)
    end do
    ! Subsidence contribution:
    f(k, k+1) = f(k, k+1) - amu(k) / m(k+1)
    f(k, k  ) = f(k, k  ) - amd(k) / m(k  )
end do
```

**NOTE from original author (CMK SH):** line 147 was corrected by
Sander Houweling — originally the subsidence term combined `amu`
AND `amd` into `f(k, k+1)`, but the correct form separates them
by column index (amu-related into f(k,k+1), amd-related into
f(k,k)).

### 3.4 Final matrix assembly (lines 167-183)

```fortran
lmc = 0  ! convection top level
conv1(:, :) = 0.0
do k = 1, lmx
    do kk = 1, lmx
        if (f(k-1, kk) /= f(k, kk)) then
            conv1(k, kk) = -dt * (f(k-1, kk) - f(k, kk))
            lmc = max(lmc, max(k, kk))
        end if
    end do
    conv1(k, k) = conv1(k, k) + 1.0
end do
```

The `conv1` matrix is `I - dt * D` where D is the flux-
divergence operator. Backward Euler: `conv1 * rm_new = rm_old`.

## 4. Solver (TM5_Conv_Apply, tm5_conv.F90:197-341)

```fortran
! LU decomposition — general matrix factorization, not tridiagonal
call dgetrf(lmx, lmx, conv1, lmx, ipiv, status)

! Solve system. trans='N' for forward, 'T' for reverse (adjoint)
call dgetrs(trans, lmx, ntr, conv1, lmx, ipiv, rm, lmx, status)
```

One factorization, multiple tracers solve. LU with pivoting
(`ipiv`). The adjoint is literally `dgetrs('T', ...)` with the
SAME LU decomposition — no separate factorization needed.

**For GPU port:** LAPACK isn't directly available in KA kernels.
Options:
1. **KA kernel writes Gaussian elimination in-kernel** — legacy
   approach. Works on all backends. Matrix is diagonal-dominant
   (guaranteed by construction) so no pivoting needed.
2. **Batched LAPACK on GPU** — cuBLAS/hipBLAS/etc. Faster for
   large grids but backend-specific.

Plan 18 ships option 1 (per plan design §4.3 Decision 7, two
kernel split: build+factorize, solve). Batched BLAS is a future
perf plan.

## 5. Cloud dimensions (ConvCloudDim, tmphys_convec.F90:39-132)

```fortran
! Cloud top: highest level where detu > 0
icltop = 0
do l = ltop, lbot, -ldir
    if (detu(l) > 0.0) then
        icltop = l
        exit
    end if
end do

! Cloud base: lowest level where detu > 0
iclbas = 0
do l = lbot, ltop, ldir
    if (detu(l) > 0.0) then
        iclbas = l
        exit
    end if
end do

! Level of free sinking: highest level where entd > 0
icllfs = 0
do l = ltop, lbot, -ldir
    if (entd(l) > 0.0) then
        icllfs = l
        exit
    end if
end do
```

Plan 18's port (`_conv_cloud_dim` or similar) uses our
top-down convention — iterate k=1..Nz for "top" and k=Nz..1 for
"bottom" — with signs reversed.

## 6. Driver (convection.F90)

Lines 487-495: matrix build invocation:

```fortran
call TM5_Conv_Matrix(dt, lmax_conv, m(i,j,1:lmax_conv), &
                     entu(i,j,:), detu(i,j,:), &
                     entd(i,j,:), detd(i,j,:), &
                     cloud_lfs(i,j), cloud_top(i,j), &
                     conv1, lbdcv, lmc, status)
```

Lines 508-509: solve invocation:

```fortran
call TM5_Conv_Apply(trans, &
                    lmc, ntracetloc, conv1(1:lmc,1:lmc), &
                    rm(i,j,1:lmc,:), ...)
```

**Key observations from driver:**

1. **Per-column serial:** one `(i,j)` at a time, OpenMP over
   `(i,j)` pairs (line 429: `do counter = 1, region_ij_counter(region)%n_iter`).
2. **`lmax_conv` clamps the matrix size** — not every column
   uses full `Nz`. Typically convection top levels (~25-30 of
   72 levels); levels above are pass-through.
3. **`lmc` = actual convection top returned by matrix builder**.
   If `lmc > 0`, solve is applied to levels `1..lmc`. If
   `lmc == 0`, no convection in this column (matrix is identity).
4. **`trans` character arg** — `'N'` forward, `'T'` reverse.
   Passed from sim orchestration (step!(), reverse-mode sims).
5. **All tracers solved with one factorization** — `rm(i,j,1:lmc,:)`
   is `(lmc, ntracetloc)`.

## 6.5 Field level placement (full levels vs half levels)

**IMPORTANT clarification from `phys_convec_ec2tm.F90`:**

- **ECMWF input:** `mflu_ec`, `mfld_ec` (total mass fluxes) are at
  **half levels** (interfaces). `detu_ec`, `detd_ec` (detrainment
  rates) are at **full levels** (layer centers). See lines 82-84
  and type declarations lines 100-105.

- **After ec2tm conversion:** `entu`, `detu`, `entd`, `detd` are
  all at **full levels** (layer centers). Lines 104-105 confirm:
  `(full lev)` annotation. Units: kg/m²/s.

Plan 18's `TM5Convection{FT, EU, DU, ED, DD}` expects all four
fields at layer centers. Shape: `(Nx, Ny, Nz)`. No interface
handling needed inside the operator.

**Parallel for CMFMCConvection:**
- `CMFMC` is at interfaces, shape `(Nx, Ny, Nz+1)`
- `DTRAIN` is at layer centers, shape `(Nx, Ny, Nz)`

These shapes are DIFFERENT, and `CMFMCConvection`'s struct must
capture that. Use different type parameters if needed:

```julia
struct CMFMCConvection{FT, MF, DT} <: AbstractConvectionOperator
    cmfmc::MF   # ::TimeVaryingField{FT, 3}, shape (Nx, Ny, Nz+1)
    dtrain::DT  # ::Union{TimeVaryingField{FT, 3}, Nothing}
                # shape (Nx, Ny, Nz) when present
end
```

vs TM5Convection where all four inputs are (Nx, Ny, Nz).

## 7. Met field sign conventions (from phys_convec_ec2tm.F90:7-14)

Original ECMWF output:
- `mfup` (updraft mass flux): positive
- `mfdo` (downdraft mass flux): negative
- `dtup` (updraft detrainment): positive (very small negative values
   possible, "contaminated" ~-0.99e-19)
- `dtdo` (downdraft detrainment): positive (same contamination)

TM5 uses:
- `entu` (updraft entrainment): positive (diagnosed from mass balance)
- `detu` (updraft detrainment): positive
- `entd` (downdraft entrainment): positive (SIGN FLIPPED from ECMWF mfdo)
- `detd` (downdraft detrainment): positive

The sign flip on downdraft is why in the Fortran matrix builder:
- `amd(k-1) = amd(k) - entd(k) + detd(k)` (looks like subtracting
  a positive, but that's because entd is defined positive here)
- `f(k-1, k) = -entd(k) / m(k) * zxi` (negative because amd is
  conceptually negative-going but entd is stored positive)

This is **subtle** — any port needs to get this sign convention
right. Legacy Julia does.

## 8. Comparison against legacy Julia port

I did a spot-check comparison (see plan 18 Commit 0 session notes):

**Index mapping:**
- Fortran uses `amu(0:lmx)`, `f(0:lmx, 1:lmx)`, `fu(0:lmx, 1:lmx)` — zero-based
- Julia uses `amu[1:lmx+1]`, `f[1:lmx+1, 1:lmx]`, `fu[1:lmx+1, 1:lmx]` — shifted by +1
- Mapping: Fortran index `p` ↔ Julia index `p+1`

**Verified faithfully ported:**
- Updraft loop (Fortran 97-118 ↔ Julia 71-83)
- Downdraft loop (Fortran 122-135 ↔ Julia 85-97)
- Assembly with subsidence (Fortran 140-149 ↔ Julia 99-105)
  — including the Sander Houweling correction
- Final matrix assembly (Fortran 167-183 ↔ Julia 107-119)

**Not yet verified (check during plan 18 Commit 4 port):**
- Column reversal at TM5/AtmosTransport interface
- Cloud dimension diagnostic (`_conv_cloud_dim` vs `ConvCloudDim`)
- KA kernel that invokes the matrix builder
- LU solver path (in-kernel Gaussian elimination)

**Legacy may have bugs elsewhere** — e.g., in driver logic,
kernel boundary handling, Adapt patterns. Plan 18 Commit 4 tests
(Tier A, B, C) will catch these.

## 9. What TM5 does NOT do

- **No sub-stepping for CFL.** The implicit solve is
  unconditionally stable by construction (diagonal-dominant
  `I - dt·D`), so no timestep constraint from convection.
- **No mass flux limiter.** The matrix builder clamps
  `amu(k) >= 0` and `amd(k) <= 0` (via `zxi = 0` branches when
  fluxes would become unphysical), but there's no separate
  CFL-like constraint.
- **No Grell-Freitas-specific handling.** TM5 uses Tiedtke-
  derived fields throughout; the Grell-Freitas scheme (used by
  post-June-2020 GEOS-FP and GEOS-IT) would need a different
  treatment in principle, but from the tracer-transport
  perspective, the CMFMC + DTRAIN fields have the same meaning
  regardless of which scheme generated them.

## 10. Plan 18 port decisions based on this reference

1. **`TM5Convection` matrix builder:** port lines 97-183 of
   `tm5_conv.F90` directly. Legacy Julia's port appears faithful;
   light cleanup only (naming, type stability).
2. **LU solve:** in-kernel Gaussian elimination (legacy pattern),
   not LAPACK. Matrix is diagonal-dominant; no pivoting needed.
3. **Column reversal:** at the `apply!` boundary, not in the
   kernel. Kernel operates in TM5 convention (k=1=surface).
4. **`lmax_conv`:** preserve as struct parameter;
   `lmax_conv = 0` sentinel for "use full Nz" per legacy.
5. **Adjoint structure:** natural — the forward LU decomposition
   transposes directly (`dgetrs(trans='T', ...)` reuses the same
   factorization). Document but don't ship adjoint in plan 18.
6. **No wet deposition** — Fortran has `#ifndef without_wet_deposition`
   branches for `cp`, `sceffdeep`, `lbdcv`, `cvsfac`. Plan 18
   skips all of these per §4.3 Decision 16. Add `# TODO:
   scavenging hook here` at line 157-165's equivalent position
   in the Julia kernel.
7. **No combined convdiff** — `tm5_convdiff.F90` exists but
   AtmosTransport runs convection separately from diffusion per
   plan 16b's operator boundaries.

---

**End of notes.**

Plan 18 Commit 0 saves this (or an updated version) to
`artifacts/plan18/upstream_fortran_notes.md` for future
reference.
