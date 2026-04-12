#!/bin/bash
# Run full CATRINE GCHP v4 7-day test, then generate comparison animation vs GEOS-Chem
set -e

cd /home/cfranken/code/gitHub/AtmosTransportModel

echo "=== Step 1: Running CATRINE GCHP v4 (7 days, 4 tracers) ==="
julia --threads=2 --project=. scripts/run.jl config/runs/catrine_geosit_c180_gchp_v4_7d.toml 2>&1 | tee /tmp/gchp_v4_7d_run.log

echo ""
echo "=== Step 2: Generating animation vs GEOS-Chem ==="
AT_PATTERN="catrine_gchp_v4_7d" \
OUT_GIF="/temp1/catrine/output/gchp_v4_vs_geoschem_co2_7d.gif" \
julia --project=. scripts/visualization/animate_gchp_vs_geoschem.jl 2>&1 | tee /tmp/gchp_v4_animation.log

echo ""
echo "=== Done ==="
echo "Run log:       /tmp/gchp_v4_7d_run.log"
echo "Animation log: /tmp/gchp_v4_animation.log"
echo "Animation:     /temp1/catrine/output/gchp_v4_vs_geoschem_co2_7d.gif"
