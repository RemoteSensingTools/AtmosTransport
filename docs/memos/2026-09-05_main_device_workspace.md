# Allocate cubed-sphere scratch on the selected backend

The CS runner now adapts initialized state and fluxes before constructing
`TransportModel`. Existing workspace constructors use `similar` on state, so
scratch is allocated directly on the selected backend. Final model adaptation
retains those device arrays and transfers remaining operator/geometry metadata.
CPU construction and numerical kernels are unchanged.

On the V100, five alternating warm construction repetitions reduce the
32-tracer C90 L66 median from 165.46 to 104.70 ms and cumulative host allocation
from 596.61 to 0.23 MB, excluding initial host-state construction. The whole-run
ERA5 profile saves another 596 MB of host allocation (7.848 to 7.252 GB), but
two repetitions show a slightly slower whole-run median, 15.148 to 15.556 s.
This is retained for the isolated construction and memory-allocation gains;
no additional whole-run speedup is claimed.

All 196 real-input output arrays remain exactly equal (280 checks), and all
75 V100 continuous/split-file physics and CPU-reference checks pass. See the
[method, raw results, and limits](../../scripts/benchmarks/results/main_device_workspace_v100_20260905/README.md).

The final Julia 1.12.6 CPU suite passes 82,059 checks across 115 core files
plus the regridding runner, with 22 existing skips or expected failures. Aqua
passes; JET reports 142 findings against the unchanged allowance of 144.
