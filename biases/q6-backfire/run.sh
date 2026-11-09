#!/usr/bin/env bash
# Q6: Backfire (H5): can a deception warning steer models off the bait?
# Configs are in configs/hyp/. Run from anywhere; the script cd's to the repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (see .env.example)}"
export OPENROUTER_API_KEY
cd "$ROOT"
python3 engine/run_experiment.py --config configs/hyp/h5-backfire.yaml "$@"  # all 3 sets
