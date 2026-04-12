#!/bin/bash
# Sync GCHP flat IC test data (Dec 1-4) to curry following DATA_LAYOUT.md
#
# Usage: bash scripts/sync_gchp_flat_to_curry.sh

set -euo pipefail

REMOTE="curry.gps.caltech.edu"
DATA_ROOT="~/data/AtmosTransport"

echo "=== Creating directory structure on $REMOTE ==="
ssh $REMOTE "mkdir -p $DATA_ROOT/met/geosit_c180/preprocessed/massflux \
                       $DATA_ROOT/met/geosit_c180/preprocessed/physics \
                       $DATA_ROOT/grids \
                       $DATA_ROOT/output/gchp_flat"

echo ""
echo "=== Syncing mass flux binaries (4 days × 8.9 GB = ~36 GB) ==="
for d in 01 02 03 04; do
    echo "  Dec $d..."
    rsync -avP --progress \
        /temp1/catrine/met/geosit_c180/massflux_v3/geosfp_cs_202112${d}_float32.bin \
        $REMOTE:$DATA_ROOT/met/geosit_c180/preprocessed/massflux/
done

echo ""
echo "=== Syncing physics binaries (A1 + CTM_I1, ~5.6 GB) ==="
for d in 01 02 03 04; do
    echo "  Dec $d..."
    rsync -avP --progress \
        /temp1/catrine/met/geosit_c180/surface_bin/GEOSFP_CS180.202112${d}.A1.bin \
        /temp1/catrine/met/geosit_c180/surface_bin/GEOSFP_CS180.202112${d}.CTM_I1.bin \
        $REMOTE:$DATA_ROOT/met/geosit_c180/preprocessed/physics/
done

echo ""
echo "=== Syncing grid specification ==="
rsync -avP \
    /home/cfranken/code/gitHub/AtmosTransportModel/data/grids/cs_c180_gridspec.nc \
    $REMOTE:$DATA_ROOT/grids/

echo ""
echo "=== Done! Total ~41 GB transferred ==="
echo ""
echo "On curry, update config paths to:"
echo "  preprocessed_dir     = \"~/data/AtmosTransport/met/geosit_c180/preprocessed/massflux\""
echo "  surface_data_bin_dir = \"~/data/AtmosTransport/met/geosit_c180/preprocessed/physics\""
echo "  coord_file           = \"~/data/AtmosTransport/grids/cs_c180_gridspec.nc\""
echo "  output filename      = \"~/data/AtmosTransport/output/gchp_flat/catrine_gchp_flat\""
