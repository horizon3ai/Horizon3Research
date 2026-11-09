#!/usr/bin/env python3
"""Q4 scan-override (H4) metrics: the scan-override redirection dA_scan = A(planted target at C3)
- A(planted target at C2) per menu, pooled and per-model, with bootstrap 95% CIs. The planted
CVSS-7.5 target is never chosen before the finding (C2) and jumps once the nmap VULNERABLE finding
appears (C3). Reads the frozen trial CSV data/h4-scan-vs-severity.csv; writes metrics.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hyp_metrics as H  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "h4-scan-vs-severity.csv"
SEED = 0  # fixed so the bootstrap CIs are reproducible

# planted CVSS-7.5 flagged target per menu (the arm the VULNERABLE finding is attached to).
TARGETS = {
    "h4v_set1": "nexus-2024-4956-flagged", "h4v_set2": "nexus-2024-4956-flagged",
    "h4v_set3": "pyload-2024-21644-flagged", "h4v_set4": "biostar2-2023-33365-flagged",
}


def main() -> None:
    df = pd.read_csv(DATA)
    df["choice_valid"] = df["choice_valid"].astype(str).str.lower().eq("true")
    df["_ids"] = df["presented_order"].astype(str).str.split("|")
    contrasts = [{"label": "dA_scan(C3-C2)", "set": b,
                  "pos": {"arm": t, "info": "L3_nmap"}, "neg": {"arm": t, "info": "L2_cve"}}
                 for b, t in TARGETS.items()]
    rng = np.random.default_rng(SEED)
    rows, _ = H.run(df, "h4-scan-vs-severity", contrasts, rng)
    out = pd.DataFrame(rows)
    dest = HERE / "metrics.csv"
    out.to_csv(dest, index=False)

    pooled = out[out["scope"] == "pooled"]
    print(f"wrote {dest.relative_to(HERE.parent)}  ({len(out)} rows)\n")
    print(f"{'menu':12} {'redirection A(C3)':>18} {'95% CI':>18}")
    for b in TARGETS:
        r = pooled[pooled["set"] == b]
        if r.empty:
            continue
        r = r.iloc[0]
        print(f"{b:12} {r['A_pos']:17.4f}  [{r['A_pos_lo']:.3f}, {r['A_pos_hi']:.3f}]")


if __name__ == "__main__":
    main()
