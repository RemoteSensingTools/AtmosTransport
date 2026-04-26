#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Download and verify the AtmosTransport quickstart bundles.
#
# Two tarballs are hosted as assets on a GitHub Release tag, both built
# from raw ERA5 spectral input for Dec 1-3, 2021:
#
#   quickstart_ll_dec2021_v1.tar.gz
#       - era5_ll72x37_dec2021_f32   (5° lat-lon, F32)
#       - era5_ll144x73_dec2021_f32  (2.5° lat-lon, F32)
#
#   quickstart_cs_dec2021_v1.tar.gz
#       - era5_cs_c24_dec2021_f32    (cubed-sphere C24, F32)
#       - era5_cs_c90_dec2021_f32    (cubed-sphere C90 ~1°, F32)
#
# Usage:
#   bash scripts/download_quickstart_data.sh         # both bundles (default)
#   bash scripts/download_quickstart_data.sh ll      # LL only — newcomer path
#   bash scripts/download_quickstart_data.sh cs      # CS only
#   bash scripts/download_quickstart_data.sh all     # both (alias for default)
#
# Env overrides:
#   ATMOSTR_QUICKSTART_DIR      — destination root (default ~/data/AtmosTransport_quickstart)
#   ATMOSTR_QUICKSTART_LL_URL   — override the LL tarball URL
#   ATMOSTR_QUICKSTART_LL_SHA   — override the LL tarball SHA-256
#   ATMOSTR_QUICKSTART_CS_URL   — override the CS tarball URL
#   ATMOSTR_QUICKSTART_CS_SHA   — override the CS tarball SHA-256
#
# Pairs with config/runs/quickstart/*.toml — see
# docs/src/getting_started/quickstart.md for the runnable walkthrough.
# ---------------------------------------------------------------------------
set -euo pipefail

# ── Bundle metadata (one line each — update on a new release) ──────────────
RELEASE_TAG="data-quickstart-v1"
RELEASE_BASE="https://github.com/RemoteSensingTools/AtmosTransport/releases/download/${RELEASE_TAG}"

LL_NAME="quickstart_ll_dec2021_v1.tar.gz"
LL_URL_DEFAULT="${RELEASE_BASE}/${LL_NAME}"
LL_SHA_DEFAULT="1d9928c3f43084f8397af14399f8c438a6c4bfeadabe37f0000fad3fa1ef76d7"

CS_NAME="quickstart_cs_dec2021_v1.tar.gz"
CS_URL_DEFAULT="${RELEASE_BASE}/${CS_NAME}"
CS_SHA_DEFAULT="ada76e875cf2852d23f544f9aeb41456e6f13c502d4d6227fac676dcca554b94"

# ── Selection ──────────────────────────────────────────────────────────────
SELECT="${1:-all}"
case "$SELECT" in
    ll|cs|all) ;;
    *) echo "Usage: $0 [ll|cs|all]" >&2; exit 2;;
esac

DEST_DIR="${ATMOSTR_QUICKSTART_DIR:-$HOME/data/AtmosTransport_quickstart}"
mkdir -p "$DEST_DIR"

# ── Helpers ────────────────────────────────────────────────────────────────
err()  { echo "ERROR: $*" >&2; exit 1; }
note() { echo "[quickstart] $*"; }

_sha256() {
    if   command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum    >/dev/null; then shasum -a 256 "$1" | cut -d' ' -f1
    else err "Need sha256sum (Linux) or shasum (macOS) on \$PATH"; fi
}

_download() {
    local url="$1" out="$2"
    if   command -v curl >/dev/null; then curl --fail --location --output "$out" "$url"
    elif command -v wget >/dev/null; then wget --output-document="$out" "$url"
    else err "Need curl or wget on \$PATH to download the bundle."; fi
}

# Fetch + verify + extract one tarball.
# Args: <bundle name>  <url default>  <sha default>  <env-url var>  <env-sha var>
_handle_bundle() {
    local name="$1" url_default="$2" sha_default="$3" env_url="$4" env_sha="$5"
    local url sha tarball
    url="${!env_url:-$url_default}"
    sha="${!env_sha:-$sha_default}"
    tarball="$DEST_DIR/$name"

    [[ "$url" == __* ]] && err "Bundle URL not configured for $name (set $env_url or update this script)"
    [[ "$sha" == __* ]] && err "Bundle SHA-256 not configured for $name (set $env_sha or update this script)"

    if [[ -f "$tarball" ]]; then
        note "$name already present at $tarball; skipping download."
    else
        note "Downloading $name from $url"
        _download "$url" "$tarball"
    fi

    note "Verifying SHA-256 for $name …"
    local actual; actual=$(_sha256 "$tarball")
    [[ "$actual" == "$sha" ]] || err "SHA-256 mismatch for $name: expected $sha, got $actual"
    note "SHA-256 OK for $name ($actual)"

    note "Validating tarball member paths …"
    local bad
    bad=$(tar --list --gzip --file="$tarball" | grep -E '^/|(^|/)\.\.(/|$)' || true)
    [[ -z "$bad" ]] || err "Tarball $name contains absolute or parent-traversing paths; refusing to extract:"$'\n'"$bad"

    note "Extracting $name under $DEST_DIR …"
    tar --extract --gzip --file="$tarball" --directory="$DEST_DIR"
}

# ── Fetch the requested set ────────────────────────────────────────────────
if [[ "$SELECT" == "ll" || "$SELECT" == "all" ]]; then
    _handle_bundle "$LL_NAME" "$LL_URL_DEFAULT" "$LL_SHA_DEFAULT" \
                   ATMOSTR_QUICKSTART_LL_URL ATMOSTR_QUICKSTART_LL_SHA
fi

if [[ "$SELECT" == "cs" || "$SELECT" == "all" ]]; then
    _handle_bundle "$CS_NAME" "$CS_URL_DEFAULT" "$CS_SHA_DEFAULT" \
                   ATMOSTR_QUICKSTART_CS_URL ATMOSTR_QUICKSTART_CS_SHA
fi

# ── Sanity-check expected layout ───────────────────────────────────────────
expected_ll=( "met/era5_ll72x37_dec2021_f32" "met/era5_ll144x73_dec2021_f32" )
expected_cs=( "met/era5_cs_c24_dec2021_f32"  "met/era5_cs_c90_dec2021_f32"  )

if [[ "$SELECT" == "ll" || "$SELECT" == "all" ]]; then
    for sub in "${expected_ll[@]}"; do
        [[ -d "$DEST_DIR/$sub" ]] || err "Expected directory missing after extraction: $DEST_DIR/$sub"
    done
fi
if [[ "$SELECT" == "cs" || "$SELECT" == "all" ]]; then
    for sub in "${expected_cs[@]}"; do
        [[ -d "$DEST_DIR/$sub" ]] || err "Expected directory missing after extraction: $DEST_DIR/$sub"
    done
fi

note "Quickstart bundle ready at $DEST_DIR"
note ""
note "Next: run one of the bundled configs"
[[ "$SELECT" == "ll" || "$SELECT" == "all" ]] && {
    note "  julia --project=. scripts/run_transport.jl config/runs/quickstart/ll72x37_advonly.toml"
    note "  julia --project=. scripts/run_transport.jl config/runs/quickstart/ll144x73_advonly.toml"
}
[[ "$SELECT" == "cs" || "$SELECT" == "all" ]] && {
    note "  julia --project=. scripts/run_transport.jl config/runs/quickstart/cs_c24_advonly.toml"
    note "  julia --project=. scripts/run_transport.jl config/runs/quickstart/cs_c90_advonly.toml"
}
