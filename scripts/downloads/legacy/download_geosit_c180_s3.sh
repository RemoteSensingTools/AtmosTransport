#!/bin/bash
# ---------------------------------------------------------------------------
# Download GEOS-IT C180 data from AWS S3 (no auth required)
#
# Downloads CTM_A1, A3mstE, A1, A3dyn collections into per-day directories
# matching our local layout: $OUT_DIR/YYYYMMDD/GEOSIT.YYYYMMDD.COLL.C180.nc
#
# Skips files that already exist. Runs N_PARALLEL concurrent downloads.
#
# Usage:
#   bash scripts/download/download_geosit_c180_s3.sh
#
# Environment:
#   OUT_DIR      — output directory (default: ~/data/geosit_c180_catrine)
#   START_DATE   — first date YYYY-MM-DD (default: 2022-02-01)
#   END_DATE     — last date YYYY-MM-DD  (default: 2023-12-31)
#   COLLECTIONS  — space-separated list  (default: "CTM_A1 A3mstE A1 A3dyn")
#   N_PARALLEL   — concurrent downloads  (default: 4)
# ---------------------------------------------------------------------------

OUT_DIR="${OUT_DIR:-$HOME/data/geosit_c180_catrine}"
START_DATE="${START_DATE:-2022-02-01}"
END_DATE="${END_DATE:-2023-12-31}"
COLLECTIONS="${COLLECTIONS:-CTM_A1 A3mstE A1 A3dyn}"
N_PARALLEL="${N_PARALLEL:-4}"

S3_BASE="s3://geos-chem/GEOS_C180/GEOS_IT"

# Find aws binary (may be aliased, not in PATH for subshells)
AWS_BIN="${AWS_BIN:-}"
for candidate in "$(type -P aws 2>/dev/null)" "$(which aws 2>/dev/null)" \
                 "/kiwi-data/software/AWS-CLI/install/v2/current/bin/aws" \
                 "/usr/local/bin/aws" "/usr/bin/aws"; do
    [[ -n "$candidate" && -x "$candidate" ]] && { AWS_BIN="$candidate"; break; }
done
[[ -x "$AWS_BIN" ]] || { echo "ERROR: aws CLI not found. Set AWS_BIN=/path/to/aws"; exit 1; }
echo "Using aws: $AWS_BIN"

echo "============================================="
echo "GEOS-IT C180 S3 Download"
echo "============================================="
echo "Period:      $START_DATE → $END_DATE"
echo "Collections: $COLLECTIONS"
echo "Parallel:    $N_PARALLEL"
echo "Output:      $OUT_DIR"
echo "============================================="

# Build task list (skip existing files)
TASK_FILE=$(mktemp /tmp/geosit_tasks.XXXXXX)
n_total=0
n_skip=0

d="$START_DATE"
while [[ "$d" < "$END_DATE" ]] || [[ "$d" == "$END_DATE" ]]; do
    ymd=$(echo "$d" | tr -d '-')
    year=${d:0:4}
    month=${d:5:2}

    for coll in $COLLECTIONS; do
        fname="GEOSIT.${ymd}.${coll}.C180.nc"
        dest="$OUT_DIR/$ymd/$fname"

        if [[ -f "$dest" ]]; then
            n_skip=$((n_skip + 1))
        else
            echo "$S3_BASE/$year/$month/$fname|$dest" >> "$TASK_FILE"
            n_total=$((n_total + 1))
        fi
    done
    d=$(date -d "$d + 1 day" +%Y-%m-%d)
done

echo "Files to download: $n_total (skipping $n_skip existing)"
[[ $n_total -eq 0 ]] && { echo "Nothing to download!"; rm -f "$TASK_FILE"; exit 0; }

# Write a small helper script for xargs
HELPER=$(mktemp /tmp/geosit_dl.XXXXXX.sh)
cat > "$HELPER" << DLEOF
#!/bin/bash
line="\$1"
s3_path="\${line%%|*}"
dest="\${line##*|}"
fname=\$(basename "\$dest")
dir=\$(dirname "\$dest")
mkdir -p "\$dir"

for attempt in 1 2 3; do
    if $AWS_BIN s3 cp --no-sign-request "\$s3_path" "\$dest" >/dev/null 2>&1; then
        sz=\$(stat --printf='%s' "\$dest" 2>/dev/null || echo 0)
        echo "  OK: \$fname (\$(( sz / 1048576 )) MB)"
        exit 0
    fi
    echo "  RETRY (\$attempt/3): \$fname"
    rm -f "\$dest"
    sleep 5
done
echo "  FAILED: \$fname"
exit 1
DLEOF
chmod +x "$HELPER"

echo ""
echo "Starting $N_PARALLEL parallel downloads..."
echo "$(date '+%Y-%m-%d %H:%M:%S'): BEGIN"
echo ""

# Run parallel downloads, continue on individual failures
cat "$TASK_FILE" | xargs -P "$N_PARALLEL" -I{} bash "$HELPER" "{}" || true

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S'): DONE"

# Summary
n_present=0
n_missing=0
d="$START_DATE"
while [[ "$d" < "$END_DATE" ]] || [[ "$d" == "$END_DATE" ]]; do
    ymd=$(echo "$d" | tr -d '-')
    for coll in $COLLECTIONS; do
        dest="$OUT_DIR/$ymd/GEOSIT.${ymd}.${coll}.C180.nc"
        if [[ -f "$dest" ]]; then
            n_present=$((n_present + 1))
        else
            n_missing=$((n_missing + 1))
        fi
    done
    d=$(date -d "$d + 1 day" +%Y-%m-%d)
done

echo "============================================="
echo "Summary"
echo "============================================="
echo "Present: $n_present / $((n_present + n_missing))"
echo "Missing: $n_missing"
echo "============================================="

rm -f "$TASK_FILE" "$HELPER"
