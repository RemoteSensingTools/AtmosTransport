#!/bin/bash
# Run CATRINE GCHP v4 with full physics (RAS + PBL), then animate vs GEOS-Chem
set -e
cd /home/cfranken/code/gitHub/AtmosTransportModel

echo "=== Running CATRINE GCHP v4 full-physics (7d, 4 tracers, RAS + PBL) ==="
julia --threads=2 --project=. scripts/run.jl config/runs/catrine_geosit_c180_gchp_v4_7d_fullphys.toml 2>&1 | tee /tmp/gchp_v4_fullphys_run.log

echo ""
echo "=== Generating animation vs GEOS-Chem ==="
julia --project=. scripts/visualization/animate_gchp_v4_fullphys.jl 2>&1 | tee /tmp/gchp_v4_fullphys_anim.log

echo ""
echo "=== Done ==="
echo "Run log:   /tmp/gchp_v4_fullphys_run.log"
echo "Animation: /temp1/catrine/output/gchp_v4_fullphys_vs_geoschem.gif"
