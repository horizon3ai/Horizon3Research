#!/usr/bin/env bash
# Q5: Contradictory/fabricated scan (H8): do models notice a fake flag?
# Configs are in configs/hyp/. Run from anywhere; the script cd's to the repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (see .env.example)}"
export OPENROUTER_API_KEY
cd "$ROOT"
# Canonical Q5 result: the controlled contradiction-vs-consistent pair. This is
# what the paper reports and what panel_glmm.py sources (data/h8-contradiction-full.csv,
# data/h8-contradiction-control-full.csv).
python3 engine/run_experiment.py --config configs/hyp/h8-contradiction-full.yaml "$@"
python3 engine/run_experiment.py --config configs/hyp/h8-contradiction-control-full.yaml "$@"
# Supplementary scan-only fabrication arm (data/h8-fab-scanonly.csv); not the headline Q5 number.
python3 engine/run_experiment.py --config configs/hyp/h8-fab-scanonly.yaml "$@"
