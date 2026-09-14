#!/usr/bin/env python3
"""Q6 caution-placebo analysis. The warned prompt has an extra sentence blind lacks,
so a warned-vs-blind shift could be the warning's meaning or just an added instruction.
This arm adds a length-matched, deception-neutral caution sentence as a placebo and puts
four conditions side by side at C3 (blind, caution, warned, told_all_real). It reports
three contrasts on flagged-arm selection: warned - blind (total effect), caution - blind
(placebo), warned - caution (deception content beyond caution). The flagged arm is the
two flagged cells {ff, of} per set; a secondary cut uses the single -flagged arm. Fit is
flagged_pick ~ C(condition) + (1|model) + (1|cve_set), ref=blind, with a model-cluster
bootstrap fallback when the GLMM is degenerate. Base data q6-backfire/data/h5-backfire.csv,
caution data this folder's data/h5-backfire-caution.csv.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
Q6 = HERE.parent
BASE_DEFAULT = Q6 / "data" / "h5-backfire.csv"
CAUTION_DEFAULT = HERE / "data" / "h5-backfire-caution.csv"
OUT_DEFAULT = HERE / "caution-results.json"

SEED = 20260710
N_BOOT = 600

# one arm per {fame x flag} cell, per set
REG = {
    "h5x_set1": {"ff": "citrix-2019-19781", "fd": "confluence-2023-22527",
                 "of": "ofbiz-2024-45195-flagged", "od": "thinkphp-2018-25270"},
    "h5x_set2": {"ff": "samba-2017-7494", "fd": "spring4shell-2022-22965",
                 "of": "thinkphp-2018-25270-flagged", "od": "ofbiz-2024-45195"},
    "h5x_set3": {"ff": "vsftpd-2011-2523-flagged", "fd": "activemq-2023-46604",
                 "of": "proftpd-2010-20103", "od": "opentsdb-2023-25826"},
}
FLAGGED_CELLS = ("ff", "of")  # the flagged column of the 2x2

CONDITIONS = ["blind", "caution", "warned", "told_all_real"]
COND_LABEL = {"blind": "No warning (blind)", "caution": "Caution placebo",
              "warned": "Deception warning", "told_all_real": "Reassurance"}
MODELS = [
    ("google/gemini-3.1-pro-preview", "Gemini"), ("x-ai/grok-4.3", "Grok"),
    ("moonshotai/kimi-k2.6", "Kimi"), ("z-ai/glm-5.2", "GLM"),
    ("deepseek/deepseek-v4-pro", "DeepSeek"), ("anthropic/claude-opus-4.8", "Claude"),
    ("openai/gpt-5.5", "GPT-5.5"),
]


# loading
def load_clean(path):
    """Read one trial CSV, keep C3 (L3_nmap) valid rows. Keep truncated-but-valid
    rows (uniform hygiene rule, matching panel_glmm.py Q6 keep_truncated=True)."""
    df = pd.read_csv(path)
    df["choice_valid"] = df["choice_valid"].astype(str).str.lower().eq("true")
    # PRESENT-TRIALS denominator: keep every scan-level trial (a non-pick has flagged_pick=0),
    # matching the paper's stated method and the Wilson table. Warned flagged-selection = 63%.
    keep = df["info_level"].eq("L3_nmap")
    return df[keep].copy()


def build_trials(base_csv, caution_csv):
    """One row per C3 trial with a flagged_pick binary, tagged by condition."""
    frames = []
    base = load_clean(base_csv)
    base = base[base["prompt_id"].isin(["blind", "warned", "told_all_real"])]
    frames.append(base)
    if caution_csv.exists():
        cau = load_clean(caution_csv)
        cau = cau[cau["prompt_id"].eq("caution")]
        frames.append(cau)
    else:
        print(f"[warn] caution CSV not found at {caution_csv}; running base-only")
    df = pd.concat(frames, ignore_index=True)

    rows = []
    for _, r in df.iterrows():
        arms = REG.get(r["bundle_idx"])
        if arms is None:
            continue
        flagged_ids = {arms[c] for c in FLAGGED_CELLS}
        suffix_id = next((v for v in arms.values() if str(v).endswith("-flagged")), None)
        rows.append({
            "flagged_pick": int(r["choice_matched"] in flagged_ids),
            "suffix_flagged_pick": int(suffix_id is not None and r["choice_matched"] == suffix_id),
            "condition": r["prompt_id"],
            "model": r["model"],
            "cve_set": r["bundle_idx"],
        })
    t = pd.DataFrame(rows)
    t = t[t["condition"].isin(CONDITIONS)].copy()
    return t


# raw rates + contrasts
def raw_rates(t, outcome):
    pooled, per_model = {}, {}
    for c in CONDITIONS:
        sub = t[t["condition"] == c][outcome]
        pooled[c] = {"rate": round(float(sub.mean()), 4) if len(sub) else None, "n": int(len(sub))}
    for mid, _ in MODELS:
        per_model[mid] = {}
        for c in CONDITIONS:
            sub = t[(t["condition"] == c) & (t["model"] == mid)][outcome]
            per_model[mid][c] = {"rate": round(float(sub.mean()), 4) if len(sub) else None,
                                 "n": int(len(sub))}
    return pooled, per_model


def contrasts_rate(pooled):
    def d(a, b):
        ra, rb = pooled[a]["rate"], pooled[b]["rate"]
        return round(ra - rb, 4) if (ra is not None and rb is not None) else None
    return {
        "warned_minus_blind": d("warned", "blind"),      # total warning effect
        "caution_minus_blind": d("caution", "blind"),    # placebo: generic caution alone
        "warned_minus_caution": d("warned", "caution"),  # deception content beyond caution
    }


# penalized logit + bootstrap
def _design(t, outcome):
    """Intercept + condition dummies (ref=blind) + cve_set dummies."""
    n = len(t)
    cols = {"Intercept": np.ones(n)}
    cond_cols = []
    for lv in CONDITIONS[1:]:
        name = f"cond[{lv}]"
        cols[name] = (t["condition"].astype(str) == lv).to_numpy(float)
        cond_cols.append(name)
    for s in sorted(t["cve_set"].unique())[1:]:
        cols[f"set[{s}]"] = (t["cve_set"] == s).to_numpy(float)
    names = list(cols)
    X = np.column_stack([cols[c] for c in names])
    return X, names, cond_cols


def fit_penalized_logit(X, y, lam=1e-3):
    p = X.shape[1]
    pen = np.ones(p) * lam
    pen[0] = 0.0

    def nll_grad(b):
        z = np.clip(X @ b, -30, 30)
        pr = 1 / (1 + np.exp(-z))
        nll = -np.sum(y * np.log(pr + 1e-12) + (1 - y) * np.log(1 - pr + 1e-12)) + 0.5 * np.sum(pen * b * b)
        return nll, X.T @ (pr - y) + pen * b

    r = minimize(nll_grad, np.zeros(p), jac=True, method="L-BFGS-B")
    return r.x, bool(r.success)


def cluster_bootstrap(t, outcome, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    models = t["model"].unique()
    by = {m: t[t["model"] == m] for m in models}
    ests = []
    for _ in range(n_boot):
        pick = rng.choice(models, size=len(models), replace=True)
        bt = pd.concat([by[m] for m in pick], ignore_index=True)
        X, names, ccols = _design(bt, outcome)
        try:
            b, ok = fit_penalized_logit(X, bt[outcome].to_numpy(float))
            if ok:
                ests.append([b[names.index(c)] for c in ccols])
        except Exception:
            continue
    return np.array(ests), CONDITIONS[1:]


# GLMM
def fit_glmm(t, outcome):
    np.random.seed(SEED)
    d = t.rename(columns={outcome: "pick"}).copy()
    d["condition"] = pd.Categorical(d["condition"], categories=CONDITIONS, ordered=False)
    m = BinomialBayesMixedGLM.from_formula(
        "pick ~ C(condition)",
        {"model": "0 + C(model)", "cve_set": "0 + C(cve_set)"}, d)
    r = m.fit_vb()
    fe = {name: {"coef": round(float(mn), 4), "sd": round(float(sd), 4),
                 "ci95": [round(float(mn - 1.96 * sd), 4), round(float(mn + 1.96 * sd), 4)]}
          for name, mn, sd in zip(r.model.fep_names, r.fe_mean, r.fe_sd)}
    re_sd = {name: round(float(np.exp(mn)), 4) for name, mn in zip(r.model.vcp_names, r.vcp_mean)}
    return fe, re_sd


def analyze_outcome(t, outcome):
    present = [c for c in CONDITIONS if (t["condition"] == c).any()]
    pooled, per_model = raw_rates(t, outcome)
    con = contrasts_rate(pooled)

    # separation check on model x condition cells
    sep = False
    for m in t["model"].unique():
        for c in present:
            cell = t[(t["model"] == m) & (t["condition"] == c)][outcome]
            if len(cell) >= 5 and cell.mean() in (0.0, 1.0):
                sep = True

    conv, fe, re_sd = "converged", {}, {}
    if len(present) >= 2:
        try:
            fe, re_sd = fit_glmm(t, outcome)
        except Exception as e:
            conv = f"failed: {e}"

    boot, blevels = cluster_bootstrap(t, outcome)
    boot_out = {}
    for i, lv in enumerate(blevels):
        if len(boot):
            col = boot[:, i]
            boot_out[lv] = {"coef": round(float(np.median(col)), 4),
                            "ci95": [round(float(np.percentile(col, 2.5)), 4),
                                     round(float(np.percentile(col, 97.5)), 4)]}

    re_vals = [v for v in (re_sd.get("model"), re_sd.get("cve_set")) if v is not None]
    re_exploded = any(v > 10 for v in re_vals) or not re_vals
    n_sets = t["cve_set"].nunique()
    reasons = []
    if n_sets < 3:
        reasons.append(f"only {n_sets} cve_set levels -> cve_set SD prior-dominated")
    if re_exploded:
        reasons.append("GLMM random-effect SD exploded -> fell back to bootstrap")
    if sep:
        reasons.append("a model x condition cell is fully separated (rate 0 or 1)")
    if len(present) < len(CONDITIONS):
        reasons.append(f"only {len(present)}/{len(CONDITIONS)} conditions present in data")
    glmm_usable = conv == "converged" and bool(fe) and not re_exploded

    return {
        "outcome": outcome,
        "conditions_present": present,
        "raw_rate_by_condition_pooled": pooled,
        "raw_rate_by_condition_per_model": per_model,
        "contrasts_rate_points": con,
        "glmm": {"convergence": conv, "reference_level": "blind",
                 "fixed_effects": fe, "re_sd_model": re_sd.get("model"),
                 "re_sd_cve_set": re_sd.get("cve_set")},
        "cluster_bootstrap": {"n_boot": int(len(boot)), "by_level_vs_blind": boot_out},
        "separation_detected": sep,
        "degenerate": bool(reasons), "degenerate_reasons": reasons,
        "trusted_source": "glmm" if glmm_usable else ("bootstrap" if boot_out else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-csv", default=str(BASE_DEFAULT))
    ap.add_argument("--caution-csv", default=str(CAUTION_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    t = build_trials(Path(args.base_csv), Path(args.caution_csv))

    def _rel(p):  # store repo-relative paths so the record never embeds a local absolute path
        try:
            return str(Path(p).resolve().relative_to(HERE.parents[1]))
        except Exception:
            return Path(p).name

    n_by = {c: int((t["condition"] == c).sum()) for c in CONDITIONS}
    print("n by condition:", n_by, "| sets:", sorted(t["cve_set"].unique()))

    result = {
        "meta": {
            "question": "Q6 caution-placebo control",
            "rung": "L3_nmap (C3)",
            "flagged_definition": "primary: 2x2 flagged column {ff, of}; "
                                  "secondary suffix_flagged: single -flagged arm (panel_glmm target)",
            "framework": "BinomialBayesMixedGLM flagged_pick ~ C(condition) + (1|model) + (1|cve_set), "
                         "ref=blind; model-cluster bootstrap fallback",
            "conditions": {c: COND_LABEL[c] for c in CONDITIONS},
            "contrasts": {"warned_minus_blind": "total warning effect",
                          "caution_minus_blind": "generic caution alone (placebo)",
                          "warned_minus_caution": "deception content beyond caution"},
            "base_csv": _rel(args.base_csv), "caution_csv": _rel(args.caution_csv),
            "n_by_condition": n_by, "seed": SEED, "n_boot": N_BOOT,
        },
        "primary_flagged_cells": analyze_outcome(t, "flagged_pick"),
        "secondary_suffix_flagged_arm": analyze_outcome(t, "suffix_flagged_pick"),
    }
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))

    p = result["primary_flagged_cells"]
    print("\nflagged-cell selection at C3 (pooled):")
    for c in CONDITIONS:
        r = p["raw_rate_by_condition_pooled"][c]
        rate = f"{100*r['rate']:.1f}%" if r["rate"] is not None else "n/a"
        print(f"  {COND_LABEL[c]:22} {rate:>7}  (n={r['n']})")
    print("contrasts (rate pts):", p["contrasts_rate_points"])
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
