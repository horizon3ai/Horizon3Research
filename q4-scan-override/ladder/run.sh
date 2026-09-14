#!/usr/bin/env bash
# Q4 decomposition ladder (h4-ladder-decomp): which planted-finding cue
# (VULNERABLE label, named CVE, demonstrated exploit) redirects model selection.
# The config is in configs/hyp/. The full run is 12 sets x 7 models x 50 rounds = 4,200 trials.
# Pass --dry-run to preview, or run bare for the full experiment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (see .env.example)}"
export OPENROUTER_API_KEY
cd "$ROOT"
python3 engine/run_experiment.py --config configs/hyp/h4-ladder-decomp.yaml "$@"
