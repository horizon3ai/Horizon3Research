#!/usr/bin/env python3
"""Q3 severity-shown (H3) metrics: the CVSS-severity gap CSG = A(highest-CVSS) - A(lowest)
per set and info rung (C2 CVE+CVSS / C3 nmap), pooled and per-model, with bootstrap 95% CIs.
Reads the frozen trial CSV data/h3-severity.csv; writes metrics.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hyp_metrics as H  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "h3-severity.csv"
SEED = 0  # fixed so the bootstrap CIs are reproducible

# per set: (highest-CVSS arm, lowest-CVSS arm). A pick for the top-CVSS arm is severity
# beating prominence; the lowest arm is the near-never-picked baseline.
SEV_SETS = {
    "h3d_set1": ("kibana-2019-7609", "joomla-2023-23752"),
    "h3d_set2": ("jenkins-2018-1000861", "joomla-2023-23752"),
    "h3d_set3": ("craftcms-2025-32432", "joomla-2023-23752"),
}
# H3 shows CVSS at every rung, so there is no C1 rung: C2 = CVE+CVSS, C3 = + nmap.
KEEP = {"CSG@C2", "CSG@C3"}

def main() -> None:
    df = pd.read_csv(DATA)
    df["choice_valid"] = df["choice_valid"].astype(str).str.lower().eq("true")
    df["_ids"] = df["presented_order"].astype(str).str.split("|")
    rng = np.random.default_rng(SEED)
    rows, _ = H.gap_ladder(df, "h3-severity", SEV_SETS, rng, metric="CSG")
    out = pd.DataFrame(rows)
    out = out[out["contrast"].isin(KEEP)].reset_index(drop=True)
    dest = HERE / "metrics.csv"
    out.to_csv(dest, index=False)

    pooled = out[out["scope"] == "pooled"]
    print(f"wrote {dest.relative_to(HERE.parent)}  ({len(out)} rows)\n")
    print(f"{'set':12} {'rung':6} {'A(top-CVSS)':>12} {'95% CI':>18}")
    for c in ("CSG@C2", "CSG@C3"):
        for s in SEV_SETS:
            r = pooled[(pooled["contrast"] == c) & (pooled["set"] == s)]
            if r.empty:
                continue
            r = r.iloc[0]
            print(f"{s:12} {c:6} {r['A_pos']:12.4f}  [{r['A_pos_lo']:.3f}, {r['A_pos_hi']:.3f}]")


if __name__ == "__main__":
    main()
