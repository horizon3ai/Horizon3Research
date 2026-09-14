#!/usr/bin/env bash
# Q5 abstain arm: the contradiction arm + its matched consistent control, both
# with a NEUTRAL opt-out (prompt blind_abstain). Detection = the DIFFERENCE in opt-out rate between the
# two arms at L3_nmap. Configs are in configs/hyp/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (see .env.example)}"
export OPENROUTER_API_KEY
cd "$ROOT"
# Full run: 2 arms x 2 menus (nexus, grafana) x rungs x 7 models x 50.
#   contradiction: L2_cve + L3_nmap  -> 1,400 trials
#   control:       L3_nmap only      ->   700 trials
python3 engine/run_experiment.py --config configs/hyp/h8-contradiction-abstain.yaml "$@"
python3 engine/run_experiment.py --config configs/hyp/h8-contradiction-control-abstain.yaml "$@"
# After the run, copy the CSVs into this folder's data/ then analyze:
#   cp "$ROOT"/out/h8-contradiction-abstain.csv           data/
#   cp "$ROOT"/out/h8-contradiction-control-abstain.csv   data/
#   python3 abstain_analysis.py
