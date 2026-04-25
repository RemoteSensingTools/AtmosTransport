#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Download and verify the AtmosTransport quickstart bundle.
#
# The bundle contains 3 days of preprocessed ERA5 transport binaries
# (Dec 1-3, 2021) at two regular lat-lon resolutions:
#
#   - era5_ll72x37_dec2021_f32   (5° lat-lon, F32)
#   - era5_ll144x73_dec2021_f32  (2.5° lat-lon, F32)
#
# Cubed-sphere bundles are deferred until the F32 spectral-CS
# preprocessing path is fixed — see docs/src/getting_started/quickstart.md
# for context.
#
# Pairs with config/runs/quickstart/*.toml — see
# docs/src/getting_started/quickstart.md for the runnable walkthrough.
# ---------------------------------------------------------------------------
set -euo pipefail

# ── Bundle metadata (fill these in once the upload is in place) ────────────
BUNDLE_URL="${ATMOSTR_QUICKSTART_URL:-TODO_PASTE_DROPBOX_DIRECT_DOWNLOAD_URL}"
BUNDLE_SHA256="${ATMOSTR_QUICKSTART_SHA256:-42c63d300c5da7e776de9b25cc00884c28e3c37abf9d421df9151793a4c85f88}"
BUNDLE_NAME="atmos_transport_quickstart_v1.tar.gz"

# ── Destination ────────────────────────────────────────────────────────────
DEST_DIR="${ATMOSTR_QUICKSTART_DIR:-$HOME/data/AtmosTransport_quickstart}"

# ── Helpers ────────────────────────────────────────────────────────────────
err() { echo "ERROR: $*" >&2; exit 1; }
note() { echo "[quickstart] $*"; }

if [[ "$BUNDLE_URL" == TODO_* ]]; then
    err "Bundle URL not configured. Set ATMOSTR_QUICKSTART_URL or update this script."
fi
if [[ "$BUNDLE_SHA256" == TODO_* ]]; then
    err "Bundle SHA-256 not configured. Set ATMOSTR_QUICKSTART_SHA256 or update this script."
fi

mkdir -p "$DEST_DIR"
TARBALL_PATH="$DEST_DIR/$BUNDLE_NAME"

# ── Download ───────────────────────────────────────────────────────────────
if [[ -f "$TARBALL_PATH" ]]; then
    note "Tarball already present at $TARBALL_PATH; skipping download."
else
    note "Downloading $BUNDLE_URL → $TARBALL_PATH"
    if command -v curl >/dev/null; then
        curl --fail --location --output "$TARBALL_PATH" "$BUNDLE_URL"
    elif command -v wget >/dev/null; then
        wget --output-document="$TARBALL_PATH" "$BUNDLE_URL"
    else
        err "Need curl or wget on \$PATH to download the bundle."
    fi
fi

# ── Verify SHA-256 ─────────────────────────────────────────────────────────
note "Verifying SHA-256 …"
if command -v sha256sum >/dev/null; then
    ACTUAL_SHA="$(sha256sum "$TARBALL_PATH" | cut -d' ' -f1)"
elif command -v shasum >/dev/null; then
    ACTUAL_SHA="$(shasum -a 256 "$TARBALL_PATH" | cut -d' ' -f1)"
else
    err "Need sha256sum (Linux) or shasum (macOS) on \$PATH to verify the bundle."
fi
if [[ "$ACTUAL_SHA" != "$BUNDLE_SHA256" ]]; then
    err "SHA-256 mismatch: expected $BUNDLE_SHA256, got $ACTUAL_SHA"
fi
note "SHA-256 OK ($ACTUAL_SHA)"

# ── Validate tar contents before extracting (no absolute / parent paths) ───
note "Validating tarball member paths …"
BAD_PATHS="$(tar --list --gzip --file="$TARBALL_PATH" |
             grep -E '^/|(^|/)\.\.(/|$)' || true)"
if [[ -n "$BAD_PATHS" ]]; then
    err "Tarball contains absolute or parent-traversing paths; refusing to extract:"$'\n'"$BAD_PATHS"
fi

# ── Extract ────────────────────────────────────────────────────────────────
note "Extracting under $DEST_DIR …"
tar --extract --gzip --file="$TARBALL_PATH" --directory="$DEST_DIR"

# ── Sanity-check layout ────────────────────────────────────────────────────
expected_subdirs=(
    "met/era5_ll72x37_dec2021_f32"
    "met/era5_ll144x73_dec2021_f32"
)
for sub in "${expected_subdirs[@]}"; do
    [[ -d "$DEST_DIR/$sub" ]] || err "Expected directory missing after extraction: $DEST_DIR/$sub"
done

note "Quickstart bundle ready at $DEST_DIR"
note ""
note "Next: run one of"
note "  julia --project=. scripts/run_transport.jl config/runs/quickstart/ll72x37_advonly.toml"
note "  julia --project=. scripts/run_transport.jl config/runs/quickstart/ll144x73_advonly.toml"
