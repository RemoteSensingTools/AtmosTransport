# Plan 23 Commit 7 — TM5Convection bench

Measurements on this host (L40S available if GPU column nonempty).
Latency is min of 5 runs after warm-up.

| Grid | Nt | CPU (ms) | GPU (ms) | GPU speedup |
|------|----|----------|----------|-------------|
| small  (72×37×10) | 1 | 2.177 | 0.637 | 3.4x |
| small  (72×37×10) | 10 | 5.511 | 0.832 | 6.6x |
| small  (72×37×10) | 30 | 12.932 | 1.263 | 10.2x |
| medium (144×73×20) | 1 | 36.178 | 3.831 | 9.4x |
| medium (144×73×20) | 10 | 116.694 | 4.771 | 24.5x |
| medium (144×73×20) | 30 | 296.574 | 6.875 | 43.1x |
| large  (288×145×34) | 1 | 571.688 | 307.555 | 1.9x |
| large  (288×145×34) | 10 | 1995.755 | 330.581 | 6.0x |
| large  (288×145×34) | 30 | 5261.935 | 390.811 | 13.5x |

## Observations

- TM5 solver is O(lmc³) per column + O(lmc²·Nt) back-substitution.
- CPU scales primarily with Nt at fixed grid (back-sub per tracer).
- GPU launches amortize the matrix build cost across all columns.
- Workspace memory overhead: ~63 KB per column (f/fu/amu/amd
  scratches added in Commit 4 for allocation-free kernel use).
