#!/usr/bin/env bash
# Q6 caution-placebo control. Runs ONLY the `caution` prompt arm
# across all 3 sets (2,100 trials). blind/warned/told_all_real already exist in the merged
# Q6 data (../data/h5-backfire.csv) and are not run here.
# The config is in configs/hyp/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (see .env.example)}"
export OPENROUTER_API_KEY
cd "$ROOT"

# 1) full caution arm (3 sets x 1 prompt x 2 rungs x 7 models x 50 = 2,100 trials)
python3 engine/run_experiment.py --config configs/hyp/h5-backfire-caution.yaml "$@"

# 2) freeze the engine output into the paper folder
cp "$ROOT/out/h5-backfire-caution.csv" "$HERE/data/h5-backfire-caution.csv"

# 3) analysis (four conditions side by side + contrasts + GLMM)
python3 "$HERE/caution_analysis.py"
