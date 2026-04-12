#!/usr/bin/env bash
# Clone TM5 source into deps/tm5 (or TM5_ROOT). Run from project root.
# Requires: Mercurial (hg). Install with:  sudo apt install mercurial  or  brew install mercurial

set -e
ROOT="${TM5_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/deps/tm5}"
mkdir -p "$(dirname "$ROOT")"
if ! command -v hg &>/dev/null; then
  echo "Mercurial (hg) is not installed. Install it first:"
  echo "  Red Hat/RHEL/CentOS/Fedora: sudo dnf install mercurial  (or sudo yum install mercurial)"
  echo "  Debian/Ubuntu:              sudo apt install mercurial"
  echo "  macOS:                      brew install mercurial"
  echo "Then re-run this script."
  exit 1
fi
if [[ -d "$ROOT/.hg" ]]; then
  echo "TM5 already cloned at $ROOT"
  (cd "$ROOT" && hg pull -u || true)
else
  echo "Cloning TM5 to $ROOT (anonymous read-only)"
  # Prefer anonymous HTTP URL to avoid SourceForge prompting for a user/password
  if hg clone "http://tm5.hg.sourceforge.net:8000/hgroot/tm5/code" "$ROOT" 2>/dev/null; then
    echo "Cloned via anonymous HTTP."
  else
    echo "Anonymous clone failed. Trying HTTPS..."
    if ! hg clone "https://hg.code.sf.net/p/tm5/code" "$ROOT"; then
      echo ""
      echo "Clone failed (e.g. asked for username). Options:"
      echo "  1. Create a free account: https://sourceforge.net/user/registration/"
      echo "     Then re-run this script and use that username when hg prompts."
      echo "  2. When hg asks for username, try leaving it blank and pressing Enter."
      echo "  3. See docs/TM5_LOCAL_SETUP.md for more."
      exit 1
    fi
  fi
fi
echo "Next: follow docs/TM5_LOCAL_SETUP.md to configure and run TM5."
