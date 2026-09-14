#!/usr/bin/env bash
# Q1 Recognition (H1): do models prefer CVEs they recognize? Runs all 3 sets.
# Configs in configs/hyp/; run from anywhere (cd's to the repo root).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (see .env.example)}"
export OPENROUTER_API_KEY
cd "$ROOT"
python3 engine/run_experiment.py --config configs/hyp/h1-recognition.yaml "$@"
