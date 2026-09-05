# Retain cubed-sphere numerical workspaces across input files

The CS runner previously allocated new flux arrays and reconstructed/adapted a
`TransportModel` for every daily binary after the first. State and physics
choices were already carried over, and `DrivenSimulation` refreshes window
forcing and diffusion geometry when opening each new driver.

The runner now retains its model, flux arrays, and numerical workspaces across
files, matching the structured runner's ownership pattern. The preceding
[convection handoff fix](2026-09-05_convection_driver_handoff.md) invalidates
CMFMC subcycle counts and matrix-derived rates when installing the new forcing.
The new simulation still allocates its window buffers; this change does not
remove all setup allocations.

## Numerical checks

- A new CPU regression compares four continuous windows with the same forcing
  split across two input files. The forcing increases eightfold at the file
  boundary. It exercises upwind/CMFMC, PPM/CMFMC matrix, and Lin–Rood/TM5,
  each with diffusion, two distinct vertical tracer profiles, and decay.
  All 54 numerical assertions pass on Julia 1.12.6 and 1.10.12.
- The corresponding V100 test passes 75 assertions, including CPU reference
  agreement with scalar GPU indexing disabled. All cases retain finite values
  and identical air mass across file boundaries.
- Bitwise equality between continuous and split GPU runs was too strict even
  for the preceding runner. Its maximum observed file-boundary difference was
  1.74e-7 relative. The regression bounds this difference by four Float32 epsilons
  and separately checks the CPU reference at rtol=5e-5.
- Direct comparison of 36 saved GPU arrays from the old and new runners gives
  21 bitwise-equal arrays and maximum relative difference 1.74e-7. All 36 pass
  the same four-epsilon bound. These are end-to-end Float32 results, not a claim
  of bitwise equality for the complete runtime.

The [opt-in GPU test](../../test/diagnostic/test_cs_multifile_gpu.jl) accepts
`ATMOSTR_MULTIFILE_GPU_OUTPUT` to save arrays for an independent comparison.
All GPU checks ran on tofu's V100 GPU 0 with CUDA runtime 12.6.

## V100 pipeline measurements

C48/L40, three repeated input files (six one-hour windows), seven output times,
Float32 upwind transport, three warmed samples. These fixtures isolate runtime
setup/transport/output behavior; convection kernel timings are in the separate
matrix reports. Times include model setup, transfers, capture, and NetCDF write.

| Tracers | Output | Before, s | Reused workspace, s | Host allocation reduction |
| ---: | --- | ---: | ---: | ---: |
| 1 | Full layers | 0.731 | 0.760 | 29.5 MB |
| 1 | Column only | 0.253 | 0.235 | 29.5 MB |
| 65 | Full layers | 11.962 | 11.589 | 29.6 MB |
| 65 | Column only | 3.235 | 2.988 | 29.5 MB |

The allocation reduction is consistent; wall-time changes are modest and noisy,
including a small regression in the one-tracer/full-layer case. This does not
establish a universal throughput improvement from workspace reuse.

Output selection is the larger effect in this fixture: with 65 tracers, full
layers write 1.09 GB and take 11.59 s, while column-only output writes 52.2 MB and
takes 2.99 s. Host allocations fall from 14.05 GB to 1.53 GB over the whole run.
Those are cumulative allocations, not peak live memory. Single-file NetCDF
streams frames throughout the run.

[Raw measurements and comparison summary](../../scripts/benchmarks/results/pipeline_v100_20260905/)
retain every timing sample, sizes, hardware/software versions, and temporary
revision labels. Both exports use the input-resource cleanup; the earlier one
reconstructs the CS model each file, while the later one retains its workspace
and includes first-window convection-cache invalidation. No convection is
configured in these pipeline timings. Reproduce the larger shape with:

```bash
CUDA_VISIBLE_DEVICES=0 ATMOSTR_BENCH_GPU_NAME=V100 \
ATMOSTR_BENCH_FILES=3 ATMOSTR_BENCH_TOPOLOGIES=cs \
ATMOSTR_BENCH_NC=48 ATMOSTR_BENCH_NZ=40 ATMOSTR_BENCH_TRACERS=1,65 \
julia --project=benchmarking benchmarking/run_pipeline_benchmarks.jl cuda pipeline.json
```

Use the CUDA 12.6 benchmark environment described in the
[benchmark guide](../../benchmarking/README.md). These are warm-cache synthetic
measurements, not cold-NAS or campaign-throughput measurements.
