#!/usr/bin/env bash
# Q4: Scan-flag override (H4): can a VULNERABLE flag beat prominence+severity?
# Configs are in configs/hyp/. Run from anywhere; the script cd's to the repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (see .env.example)}"
export OPENROUTER_API_KEY
cd "$ROOT"
python3 engine/run_experiment.py --config configs/hyp/h4-scan-vs-severity.yaml "$@"
