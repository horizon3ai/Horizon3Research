#!/usr/bin/env bash
# Set up a virtual environment with the analysis dependencies.
# Usage:  ./install.sh   then   ./reproduce.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

python3 -m venv .venv
./.venv/bin/pip install --upgrade pip >/dev/null
./.venv/bin/pip install -r requirements.txt

echo
echo "Done. Reproduce the paper numbers with:  ./reproduce.sh"
echo "(or activate the env first:  source .venv/bin/activate)"
