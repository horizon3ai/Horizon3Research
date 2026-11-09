#!/usr/bin/env python3
"""Q1 recognition (H1) metrics: the recognition gap RB = A(prominent) - A(obscure) per set and
info rung (C1 version / C2 CVE / C3 nmap), pooled and per-model, with bootstrap 95% CIs.
Reads the frozen trial CSV data/h1-recognition-sets123.csv; writes metrics.csv.

Usage:  python compute_metrics.py [--refresh]   # --refresh regenerates data/ from out/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hyp_metrics as H  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "h1-recognition-sets123.csv"
SEED = 0  # fixed so the bootstrap CIs are reproducible


def freeze_data() -> pd.DataFrame:
    """Regenerate data/ from the raw out/h1-recognition.csv (all 3 sets)."""
    out_dir = HERE.parents[1] / "out"
    df = pd.read_csv(out_dir / "h1-recognition.csv")
    DATA.parent.mkdir(exist_ok=True)
    df.to_csv(DATA, index=False)
    return df


def load_data(refresh: bool) -> pd.DataFrame:
    df = freeze_data() if (refresh or not DATA.exists()) else pd.read_csv(DATA)
    # choice_valid + presented_order parsing
    df["choice_valid"] = df["choice_valid"].astype(str).str.lower().eq("true")
    if "temperature" in df.columns:
        df["temperature"] = df["temperature"].astype(float)
    df["_ids"] = df["presented_order"].astype(str).str.split("|")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="re-freeze data/ from the shared out/ dir before computing")
    args = ap.parse_args()
    df = load_data(args.refresh)
    rng = np.random.default_rng(SEED)
    rows, verdict = H.gap_ladder(df, "h1-recognition", H.FAM_OBS, rng, metric="RB")
    out = pd.DataFrame(rows)
    dest = HERE / "metrics.csv"
    out.to_csv(dest, index=False)

    sets = list(H.FAM_OBS)
    labels = {"h1_set1_ftp": "set1 ftp", "h1_set2_spring": "set2 spring",
              "h1_set3_enterprise": "set3 ent"}

    print(f"\nwrote {dest.relative_to(HERE.parent)}  "
          f"({len(out)} rows)  |  H1 supported={verdict['supported']} "
          f"(sets CI∌0 at C2: {verdict['sets_supported_C2']})\n")

    print(f"{'set':12} {'rung':5} {'A(prom)':>7} {'A(obs)':>7} {'RB gap':>8} "
          f"{'95% CI':>16} {'CI∌0':>5}")
    pooled = out[out["scope"] == "pooled"]
    for s in sets:
        for rung, c in [("C1", "RB@C1"), ("C2", "RB@C2"), ("C3", "RB@C3")]:
            r = pooled[(pooled["set"] == s) & (pooled["contrast"] == c)]
            if r.empty:
                continue
            r = r.iloc[0]
            ci = f"[{r['gap_lo']:+.2f}, {r['gap_hi']:+.2f}]"
            print(f"{labels[s]:12} {rung:5} {r['A_pos']:7.3f} {r['A_neg']:7.3f} "
                  f"{r['gap']:+8.3f} {ci:>16} {str(r['ci_excl_0']):>5}")
        # recognition jump
        d = pooled[(pooled["set"] == s) & (pooled["contrast"] == "dA_rec(C2-C1)")]
        if not d.empty:
            d = d.iloc[0]
            print(f"{labels[s]:12} {'ΔA':5} {'':7} {'':7} {d['gap']:+8.3f} "
                  f"{'['+format(d['gap_lo'],'+.2f')+', '+format(d['gap_hi'],'+.2f')+']':>16} "
                  f"{str(d['ci_excl_0']):>5}")
        print()

    # Per-model RB@C2 and the clean recognition column: mean RB@C2 over all valid
    # sets. Claude drops set1 (its set1 RB is filler-capture, not recognition);
    # every other model averages all three sets.
    model = out[(out["scope"] == "model") & (out["contrast"] == "RB@C2")]
    print("per-model RB@C2:")
    print(f"  {'model':22} {'set1':>6} {'set2':>6} {'set3':>6} {'clean':>7}")
    for m in sorted(model["model"].unique()):
        vals = {}
        for s in sets:
            r = model[(model["model"] == m) & (model["set"] == s)]
            vals[s] = float(r.iloc[0]["gap"]) if not r.empty else None
        drop = {"h1_set1_ftp"} if m.startswith("claude") else set()
        parts = [v for s, v in vals.items() if v is not None and s not in drop]
        clean = sum(parts) / len(parts) if parts else float("nan")
        f = lambda x: f"{x:+.2f}" if x is not None else "  -"
        tag = "  (excl set1)" if m.startswith("claude") else ""
        print(f"  {m:22} {f(vals['h1_set1_ftp']):>6} {f(vals['h1_set2_spring']):>6} "
              f"{f(vals['h1_set3_enterprise']):>6} {clean:>+7.2f}{tag}")


if __name__ == "__main__":
    main()
