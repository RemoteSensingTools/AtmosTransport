#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb
export JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 ATMOSTR_TIMERS=1
weekly_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case_label="$1"
precision="$2"
tracer_count="$3"
export_dir="$4"
run_label="${case_label}-${precision}-${tracer_count}"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv --loop-ms=500 > "/tmp/atmos-weekly-${run_label}-device.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT
cd "$export_dir"
/usr/bin/time -v -o "/tmp/atmos-weekly-${run_label}-resources.txt" /home/cfranken/.juliaup/bin/julia --startup-file=no --project=. "${weekly_script_dir}/run.jl" "$case_label" "$precision" "$tracer_count" > "/tmp/atmos-weekly-${run_label}.log" 2>&1
