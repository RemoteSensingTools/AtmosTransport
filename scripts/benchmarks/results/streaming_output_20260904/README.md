# Incremental NetCDF pipeline — September 4, 2026

`pipeline_cpu.json` contains all 12 synthetic binary-reader → runtime → NetCDF
cases with incremental single-file output. Each case records three warm samples
after compilation/setup warmup, covering LL/RG/CS, one/four tracers, and
full/column-only output. The public visualization reader checks all three output
times and every tracer's column mean outside the timed interval.

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' julia --project=benchmarking benchmarking/run_pipeline_benchmarks.jl cpu /tmp/pipeline_cpu.json
```

The four-tracer CS fixture measured 59.24 ms with full output and 45.51 ms with
column-only output. Earlier batch-output measurements were 47.41 and 40.92 ms.
Incremental flushing adds I/O overhead on this small fixture while eliminating
snapshot retention across the run. These separate warm-cache measurements are
not a controlled throughput comparison or a cold-NAS benchmark. Host allocated
bytes count cumulative allocation, not retained memory or peak RSS.

Final validation uses CPU only, per the user's updated instruction. No GPU
performance claim is made for this follow-up.
