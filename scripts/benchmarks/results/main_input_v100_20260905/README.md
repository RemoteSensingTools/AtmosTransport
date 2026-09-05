# Input and file-handoff validation on current main

Julia 1.12.6, four threads; CUDA.jl 5.11.3 / runtime 12.6 on tofu GPU 0,
Tesla V100-PCIE-16GB. Only the explicitly selected V100 was used.

- CPU: 337 focused checks pass, plus Aqua (10) and JET (141 findings,
  unchanged allowance 144).
- GPU: 223 checks pass (140 startup, 8 failure consumption, 75 split-file
  physics / CPU reference). See the raw log and summaries.
- File-handoff physics uses Upwind/CMFMC, PPM/matrix CMFMC, and LinRood/TM5,
  each with vertical diffusion and tracer decay. GPU comparisons use Float32;
  CPU continuous/split-file checks use Float64. The small fixture tests
  correctness, not representative throughput.

Reproduce the GPU checks with `test/diagnostic/test_window_prefetch_gpu.jl`,
`test_window_prefetch_failure_gpu.jl`, and `test_cs_multifile_gpu.jl`. Their
headers list the opt-in variables. Select the authorized device with
`CUDA_VISIBLE_DEVICES` and set both expected-device variables to `V100`.
Do not use CUDA runtime 13 for this Volta GPU.
