#!/bin/bash
# Run all 8 Catrine 2-day configurations and generate snapshots.
#
# Usage: bash scripts/run_catrine_8way.sh
#
# LL and RG configs use run_transport_binary.jl (TransportBinaryDriver).
# CS configs use run_cs_transport.jl (CubedSphereBinaryReader).
#
# Output: ~/data/AtmosTransport/output/catrine_2day/*.nc

set -e

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

mkdir -p ~/data/AtmosTransport/output/catrine_2day

echo "================================================================"
echo "Running 8 Catrine 2-day configurations"
echo "================================================================"

# LatLon runs (3 schemes)
for scheme in upwind slopes ppm; do
    echo ""
    echo "--- LL $scheme ---"
    julia --project=. scripts/run_transport_binary.jl \
        config/runs/catrine_2day_ll_${scheme}.toml
done

# Reduced Gaussian (upwind only)
echo ""
echo "--- RG upwind ---"
julia --project=. scripts/run_transport_binary.jl \
    config/runs/catrine_2day_rg_upwind.toml

# Cubed Sphere (4 schemes)
for scheme in upwind slopes ppm linrood; do
    echo ""
    echo "--- CS $scheme ---"
    julia --project=. scripts/run_cs_transport.jl \
        config/runs/catrine_2day_cs_${scheme}.toml
done

echo ""
echo "================================================================"
echo "All 8 runs complete. Snapshots in ~/data/AtmosTransport/output/catrine_2day/"
echo "================================================================"
