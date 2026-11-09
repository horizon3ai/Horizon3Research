#!/usr/bin/env bash
# Regenerate every number in the paper from the frozen per-trial CSVs.
# No API key needed: the analysis reads the frozen data/ CSVs, not the models.
# Each script prints its results; compare them against the paper's reported tables.
#
# Prereqs:  ./install.sh   (or any Python 3.12+ with the packages in requirements.txt)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

echo "== E1  recognition metrics (RB gaps) =="
( cd q1-recognition && "$PY" compute_metrics.py )

echo "== E2  prominence vs severity: pick rates =="
"$PY" q2-drivers/compute_e2_pickrates.py

echo "== E3  severity shown: top-CVSS pick rates (CSG) =="
( cd q3-severity-shown && "$PY" compute_metrics.py )

echo "== E4  recency: release-year log-odds =="
"$PY" q2-recency/compute_recency_logodds.py

echo "== Panel GLMM (E1 recognition, E3 severity, E5 scan, E6 fabrication, E7 backfire) =="
"$PY" panel-glmm/panel_glmm.py

echo "== Per-model Wilson-interval table =="
"$PY" permodel-summary/wilson_appendix.py

echo "== Prominence index =="
"$PY" general/fame/fame_rank.py

echo "== E5  scan-override: planted-finding redirection (per menu) =="
( cd q4-scan-override && "$PY" compute_metrics.py )

echo "== E5  scan-flag decomposition ladder (GLMM) =="
( cd q4-scan-override/ladder && "$PY" ladder_glmm.py )

echo "== E6  fabrication abstain analysis (GLMM) =="
( cd q5-fabrication/abstain && "$PY" abstain_analysis.py )

echo "== E7  backfire caution-placebo analysis (GLMM) =="
( cd q6-backfire/caution && "$PY" caution_analysis.py )

echo "== Validity: recognition robustness (Table 8) =="
( cd validity && "$PY" recognition_robustness.py )

echo "== Validity: scan-flag robustness (Table 9) =="
( cd validity && "$PY" scanflag_robustness.py )

echo
echo "All analyses reproduced."
echo "(The paper's tables and figures are built from these numbers, not regenerated here.)"
