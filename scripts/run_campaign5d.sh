#!/bin/bash
# ---------------------------------------------------------------------------
# 5-day Catrine campaign — run a chosen subset of the 36 generated configs.
#
# Usage:
#   scripts/run_campaign5d.sh f32-gpu          # 12 F32 GPU runs (wurst)
#   scripts/run_campaign5d.sh f64-gpu          # 12 F64 GPU runs (curry)
#   scripts/run_campaign5d.sh f32-cpu          # 6  F32 CPU runs (coarse only)
#   scripts/run_campaign5d.sh f64-cpu          # 6  F64 CPU runs (coarse only)
#
# GPU runs go sequentially (single GPU device). CPU runs go sequentially as
# well to avoid mmap thrashing — Julia's @threads is already wide.
#
# Logs land in /tmp/catrine5d/<host>/<run-name>.log on the local server.
# ---------------------------------------------------------------------------
set -u
GROUP="${1:-}"
if [[ -z "$GROUP" ]]; then
    echo "usage: $0 {f32-gpu|f64-gpu|f32-cpu|f64-cpu}"
    exit 2
fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
CFG_DIR="$REPO/config/runs/catrine5d"
HOST=$(hostname -s)
LOG_DIR="/tmp/catrine5d/$HOST"
mkdir -p "$LOG_DIR"

case "$GROUP" in
    f32-gpu) PATTERN="*_f32_*_gpu.toml"  ;;
    f64-gpu) PATTERN="*_f64_*_gpu.toml"  ;;
    f32-cpu) PATTERN="*_f32_*_cpu.toml"  ;;
    f64-cpu) PATTERN="*_f64_*_cpu.toml"  ;;
    *) echo "unknown group $GROUP"; exit 2 ;;
esac

shopt -s nullglob
CONFIGS=( "$CFG_DIR"/$PATTERN )
echo "[$HOST] running $GROUP — ${#CONFIGS[@]} configs"

failed=()
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .toml)
    log="$LOG_DIR/$name.log"
    echo "[$(date +%H:%M:%S)] start $name → $log"
    if julia -t8 --project="$REPO" "$REPO/scripts/run_transport.jl" "$cfg" >"$log" 2>&1; then
        echo "[$(date +%H:%M:%S)] DONE  $name"
    else
        echo "[$(date +%H:%M:%S)] FAIL  $name (see $log)"
        failed+=( "$name" )
    fi
done

echo
echo "[$HOST] $GROUP complete. ${#failed[@]} failures: ${failed[*]:-none}"
exit ${#failed[@]}
