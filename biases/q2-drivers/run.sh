#!/usr/bin/env bash
# Q2: Attribute drivers (E4/H2): prominence vs severity vs recency
# Configs are in configs/hyp/. Run from anywhere; the script cd's to the repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (see .env.example)}"
export OPENROUTER_API_KEY
cd "$ROOT"
python3 engine/run_experiment.py --config configs/hyp/e4b-fame-vs-severity.yaml "$@"  # all 3 sets
python3 engine/run_experiment.py --config configs/hyp/h2-recency.yaml "$@"            # all 3 sets
