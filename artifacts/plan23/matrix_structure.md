# TM5 `conv1 = I - dt·D` Matrix Structure (Commit 0 Survey)

**Purpose:** Characterize the sparsity / structure of the TM5
convection matrix so Commit 2's column solver ships at the correct
complexity class once (principle 2).

**References:**

- [`deps/tm5/base/src/tm5_conv.F90:32–191`](../../deps/tm5/base/src/tm5_conv.F90) — matrix builder `TM5_Conv_Matrix`.
- [`artifacts/plan18/upstream_fortran_notes.md`](../plan18/upstream_fortran_notes.md) §3 — line-by-line walk-through.

## 1. Construction sketch

The matrix is assembled from three flux contributions into two
intermediate arrays `f(0:lmx, 1:lmx)` and `fu(0:lmx, 1:lmx)`:

1. **Updraft** (`tm5_conv.F90:92–113`):
   - Loop `k = 1, li` (cloud-top index).
   - For each `k`, fills `fu(k, kk)` for `kk = 1, k-1` (strict
     subdiagonal) plus `fu(k, k)` (diagonal).
   - **Pattern:** column-by-column accumulation; `fu` is lower
     triangular in index space up to row `li`.

2. **Downdraft** (`tm5_conv.F90:118–130`):
   - Loop `k = ld, 2, -1` (level-of-free-sinking down to 2).
   - For each `k`, fills `f(k-1, kk)` for `kk = k+1, ld` (strict
     superdiagonal) plus `f(k-1, k)` (subdiagonal, negative).
   - **Pattern:** upper triangular in index space within
     `[2, ld]`.

3. **Subsidence** (`tm5_conv.F90:140–144`):
   - Loop `k = 1, lmx-1`.
   - Adjusts `f(k, k+1)` (super-diagonal, updraft subsidence) and
     `f(k, k)` (diagonal, downdraft subsidence).

4. **Assembly** (`tm5_conv.F90:165–177`):
   - `conv1(k, kk) = -dt · (f(k-1, kk) - f(k, kk))` when fluxes
     differ between interfaces.
   - `conv1(k, k) += 1.0`.

## 2. Resulting structure

Within the cloud window (k = 1 … max(li, ld)):

- **Updraft subplane:** `conv1` has non-zeros at `(k, kk)` for every
  `kk ≤ k ≤ li`. Strictly lower triangular band extending all the
  way from diagonal to column 1.
- **Downdraft subplane:** `conv1` has non-zeros at `(k-1, kk)` for
  every `k+1 ≤ kk ≤ ld`. Strictly upper triangular band.
- **Subsidence:** adds super- / sub-diagonal entries.
- **Identity baseline:** diagonal has +1 from line 176.

Outside the cloud window (levels above `max(li, ld)`), `conv1` is
pure identity (matrix builder leaves those rows as identity after
the `conv1 = 0; conv1(k,k) += 1` pattern).

**Net:** the active block is **dense lower + dense upper
triangular** within `[1, max(li, ld)]`. Not banded. Not
block-bidiagonal. Not lower-Hessenberg. Outside the active block
it's identity.

## 3. Diagonal dominance

TM5 upstream comment: "Matrix is diagonal-dominant (guaranteed by
construction) so no pivoting needed"
([`upstream_fortran_notes.md:157`](../plan18/upstream_fortran_notes.md#L157)).

The upstream Fortran `TM5_Conv_Apply` still calls `dgetrf` with
pivoting for numerical safety + to keep the same factored matrix
usable for forward and adjoint (`dgetrs(trans='T', ...)`).

## 4. Recommended solver class for Commit 2

**Decision: full partial-pivot Gaussian elimination on the active
`lmc × lmc` sub-block.** Reasoning:

- **Not banded.** The updraft contribution reaches from any
  `kk ≤ k` into `k`, so the effective bandwidth is `li - 1` — which
  at typical convective configurations (li ≈ 25–35 out of 72 levels)
  means a "banded" solver would still need a band of 25–35. Thomas
  is only competitive at bandwidth 1 (tridiagonal).
- **`lmc` optimization matters.** Upstream Fortran only factorizes
  and solves the `1..lmc` sub-block returned by `TM5_Conv_Matrix`.
  For columns with shallow or no convection, `lmc` is small (or 0).
  Ship this optimization from Commit 2 — work scales with actual
  cloud extent, not nominal Nz.
- **Diagonal dominance → pivoting optional in theory, kept in
  practice.** Store `pivots_buf` in `TM5Workspace` per principle 3
  (adjoint preservation, type stability across columns with and
  without surprises). The adjoint path in plan 19 reuses the same
  factorization with `'T'` — permutation is trivial to replay.

**Complexity budget:** `O(lmc³ + Nt · lmc²)` per column.
- Factorization: `(2/3) · lmc³` flops.
- Solve: `2 · lmc² · Nt` flops (Nt = tracer count).
- At `lmc = 35`, `Nt = 30`: ~29k factor flops + ~73k solve flops
  per column. At `lmc = 72` (worst case, deep convection reaches
  TOA): ~249k + ~311k.

On wurst L40S F32 with ~144 TFLOPS theoretical, this is trivial
per column; the bottleneck is memory traffic + launch config, not
arithmetic. Budget: expect parity with CMFMC's sub-cycled cost at
representative `lmc`.

## 5. What NOT to ship

- **Not** a banded Thomas sweep. Structure doesn't support it
  efficiently.
- **Not** a full `Nz × Nz` GE. The `lmc` sub-block optimization is
  free mass transport wrt correctness (identity rows above the
  cloud are a no-op) and saves significant work per column.
- **Not** LAPACK's `lu!` on the host. The column solve runs inside
  a KA kernel; no GPU LAPACK call (upstream_fortran_notes §4
  documents this option but defers to plan 18 Decision 7).
- **Not** LU without stored permutation. Keep `pivots_buf` for
  adjoint compatibility per principle 3.

## 6. Commit 2 implementation lock

```
@inline function _tm5_solve_column!(
    rm_col,         # (Nz, Nt)
    m_col,          # (Nz,)
    (entu_col, detu_col, entd_col, detd_col),  # (Nz,) each
    conv1_buf,      # (Nz, Nz) workspace — use 1:lmc sub-block
    pivots_buf,     # (Nz,) permutation storage
    cloud_dims,     # (3,) — (icltop, iclbas, icllfs)
    dt,
)
    # 1) Diagnose (icltop, iclbas, icllfs) and compute lmc.
    # 2) If lmc == 0: return rm_col unchanged (identity).
    # 3) Build conv1[1:lmc, 1:lmc] per F90:92–177 (assembly above).
    # 4) Factorize conv1[1:lmc, 1:lmc] with partial pivoting; store
    #    pivots_buf[1:lmc].
    # 5) For each tracer t in 1:Nt, back-substitute rm_col[1:lmc, t]
    #    using the factored matrix + permutation.
    # 6) rm_col[lmc+1:Nz, :] untouched (identity rows above lmc).
end
```

No runtime orientation flip (principle 1). Preprocessor delivers
all four entrainment/detrainment fields in AtmosTransport
orientation (k=1=TOA, k=Nz=surface); the solver reads them
directly.

## Status

Commit 0 survey complete. Commit 2 is locked to partial-pivot GE
on the `lmc × lmc` active sub-block.
