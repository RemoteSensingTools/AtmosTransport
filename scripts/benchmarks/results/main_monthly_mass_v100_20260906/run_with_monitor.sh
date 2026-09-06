#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb
export JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 ATMOSTR_TIMERS=1
monthly_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case_label="$1"
precision="$2"
tracer_count="$3"
export_dir="$4"
run_label="${case_label}-${precision}-${tracer_count}"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv --loop-ms=500 > "/tmp/atmos-monthly-${run_label}-device.csv" &
device_monitor_pid=$!
cpu_monitor_pid=""
cleanup() {
    kill "$device_monitor_pid" 2>/dev/null || true
    wait "$device_monitor_pid" 2>/dev/null || true
    if [[ -n "$cpu_monitor_pid" ]]; then
        kill "$cpu_monitor_pid" 2>/dev/null || true
        wait "$cpu_monitor_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
cd "$export_dir"
/usr/bin/time -v -o "/tmp/atmos-monthly-${run_label}-resources.txt" /home/cfranken/.juliaup/bin/julia --startup-file=no --project=. "${monthly_script_dir}/run.jl" "$case_label" "$precision" "$tracer_count" > "/tmp/atmos-monthly-${run_label}.log" 2>&1 &
run_pid=$!
# GNU time owns the Julia process. Sample only its direct child, avoiding
# unrelated Julia jobs on this shared host. RSS includes resident mmap pages.
(
    printf 'unix_seconds\tpid\trss_kib\tvsz_kib\n'
    while kill -0 "$run_pid" 2>/dev/null; do
        sample_time="${EPOCHREALTIME}"
        ps --ppid "$run_pid" -o pid=,rss=,vsz= | awk -v now="$sample_time" '{print now "\t" $1 "\t" $2 "\t" $3}' || true
        sleep 1
    done
) > "/tmp/atmos-monthly-${run_label}-host.tsv" &
cpu_monitor_pid=$!
run_status=0
wait "$run_pid" || run_status=$?
exit "$run_status"
