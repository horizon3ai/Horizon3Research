#!/usr/bin/env bash
# Validity robustness: recognition gap (Table 8) and scan-flag pull (Table 9)
# across prompt framings and temperatures. Configs are in configs/hyp/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (see .env.example)}"
export OPENROUTER_API_KEY
cd "$ROOT"
# recognition-menu robustness (Table 8): prominent vs obscure gap
python3 engine/run_experiment.py --config configs/hyp/recognition-robust-framing.yaml "$@"
python3 engine/run_experiment.py --config configs/hyp/recognition-robust-temp.yaml "$@"
# champions-menu scan-flag robustness (Table 9): flagged-pick rate
python3 engine/run_experiment.py --config configs/hyp/h6-robustness.yaml "$@"
python3 engine/run_experiment.py --config configs/hyp/h7-temperature.yaml "$@"
