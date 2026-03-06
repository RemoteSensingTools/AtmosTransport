#!/usr/bin/env bash
# Run TM5 compile/setup from deps/tm5 with correct Python path.
# Usage: ./scripts/run_tm5_setup.sh [rcfile]
#   rcfile: path to rc file (default: rc/nam1x1-dummy_tr.rc), relative to TM5 root.
# Run from project root. Requires Python 3.5+ and (for full build) Fortran compiler + libs.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TM5_ROOT="${TM5_ROOT:-$PROJECT_ROOT/deps/tm5}"

# First arg is rc file if it doesn't look like an option; otherwise default rc file
if [[ -n "${1:-}" && "$1" != -* ]]; then
  RCFILE="$1"
  shift
else
  RCFILE="rc/nam1x1-dummy_tr.rc"
fi

if [[ ! -d "$TM5_ROOT" ]]; then
  echo "TM5 root not found: $TM5_ROOT"
  echo "Obtain TM5 source first (see docs/TM5_LOCAL_SETUP.md)."
  exit 1
fi

# Load Intel oneAPI (ifx, MKL) and OpenMPI
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
fi
module load mpi/openmpi-x86_64 2>/dev/null || true

# makedepf90 may be in user-local bin
export PATH="/home/cfranken/.local/bin:$PATH"

export PYTHONPATH="$TM5_ROOT:$TM5_ROOT/base/py:$TM5_ROOT/base/py/helper"
cd "$TM5_ROOT"
python3 base/bin/pycasso_setup_tm5 "$@" "$RCFILE"
