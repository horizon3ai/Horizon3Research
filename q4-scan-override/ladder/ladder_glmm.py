#!/usr/bin/env python3
"""Q4 decomposition ladder: which cue (VULNERABLE label, named CVE, demonstrated
exploit) redirects selection. Four rungs of one CVSS-7.5 decoy (nexus-2024-4956) add
the cues one at a time (l1normal, l2vuln, l3cve, l4exploit). Reports the decoy pick
rate per rung, the marginal jump vs the previous rung, and a mixed-effects fit
decoy_pick ~ C(rung, ref=l1normal) + (1|model) + (1|menu). Reads data/h4-ladder-decomp.csv.
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "h4-ladder-decomp.csv"
RUNGS = ["l1normal", "l2vuln", "l3cve", "l4exploit"]
CUE_ADDED = {"l1normal": "(reference: service + version only)",
             "l2vuln": "+ VULNERABLE label", "l3cve": "+ named CVE",
             "l4exploit": "+ demonstrated exploit"}
SEED = 20260814

_SET = re.compile(r"ladder_(m\d+)_(l\w+)")


def load_clean():
    df = pd.read_csv(DATA)
    df["choice_valid"] = df["choice_valid"].astype(str).str.lower().eq("true")
    df["content_was_null"] = df["content_was_null"].astype(str).str.lower().eq("true")
    # Attraction convention, matching the paper
    # tables/prose: denominator = every present trial at a rung (a refusal or
    # unparseable response counts as a non-pick of the decoy), numerator = valid
    # decoy picks. This also keeps truncated-but-valid rows (uniform hygiene rule,
    # same as panel_glmm.py). No rows are dropped.
    dropped = 0
    return df.copy(), dropped


def decoy_id(row):
    """The ladder decoy arm present in this menu = the presented arm whose id starts with nexus-2024-4956-l."""
    for a in str(row["presented_order"]).split("|"):
        if a.startswith("nexus-2024-4956-l"):
            return a
    return None


def build_trials(df):
    rows = []
    for _, r in df.iterrows():
        m = _SET.match(str(r["bundle_idx"]))
        if not m:
            continue
        menu, rung = m.group(1), m.group(2)
        d = decoy_id(r)
        if d is None or rung not in RUNGS:
            continue
        rows.append({"pick": int(bool(r["choice_valid"]) and r["choice_matched"] == d),
                     "rung": rung, "model": r["model"], "menu": menu})
    t = pd.DataFrame(rows)
    t["rung"] = pd.Categorical(t["rung"], categories=RUNGS, ordered=False)
    return t


def rates(t):
    """Pooled + per-model decoy pick rate by rung, plus per-rung marginal jump vs previous rung."""
    pooled = {}
    for rung in RUNGS:
        sub = t[t["rung"].astype(str) == rung]
        pooled[rung] = {"rate": round(float(sub["pick"].mean()), 4) if len(sub) else None,
                        "n": int(len(sub)), "picks": int(sub["pick"].sum())}
    marg = {}
    for i, rung in enumerate(RUNGS):
        if i == 0:
            marg[rung] = None
        else:
            prev, cur = pooled[RUNGS[i - 1]]["rate"], pooled[rung]["rate"]
            marg[rung] = round(cur - prev, 4) if (prev is not None and cur is not None) else None
    per_model = {}
    for mdl in sorted(t["model"].unique()):
        sm = t[t["model"] == mdl]
        per_model[mdl] = {rung: (round(float(sm[sm["rung"].astype(str) == rung]["pick"].mean()), 4)
                                 if len(sm[sm["rung"].astype(str) == rung]) else None)
                          for rung in RUNGS}
    return pooled, marg, per_model


def fit_glmm(t):
    np.random.seed(SEED)
    m = BinomialBayesMixedGLM.from_formula(
        "pick ~ C(rung)", {"model": "0 + C(model)", "menu": "0 + C(menu)"}, t)
    r = m.fit_vb()
    fe = {name: {"coef": round(float(mn), 4), "sd": round(float(sd), 4),
                 "ci95": [round(float(mn - 1.96 * sd), 4), round(float(mn + 1.96 * sd), 4)]}
          for name, mn, sd in zip(r.model.fep_names, r.fe_mean, r.fe_sd)}
    re_sd = {name: round(float(np.exp(mn)), 4) for name, mn in zip(r.model.vcp_names, r.vcp_mean)}
    return fe, re_sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "ladder-glmm.json"))
    args = ap.parse_args()
    if not DATA.exists():
        raise SystemExit(f"missing {DATA} — copy the frozen full-run CSV here first "
                         f"(out/h4-ladder-decomp.csv after the run).")

    df, dropped = load_clean()
    t = build_trials(df)
    pooled, marg, per_model = rates(t)

    conv = "converged"
    try:
        fe, re_sd = fit_glmm(t)
    except Exception as e:
        fe, re_sd, conv = {}, {}, f"failed: {e}"

    print(f"{'rung':<11}{'cue added':<24}{'decoy rate':>11}{'marginal':>10}   n")
    for rung in RUNGS:
        p = pooled[rung]
        mj = marg[rung]
        rate_s = f"{100*p['rate']:.1f}%" if p["rate"] is not None else "n/a"
        mj_s = f"{100*mj:+.1f}pt" if mj is not None else "  --"
        print(f"{rung:<11}{CUE_ADDED[rung]:<24}{rate_s:>11}{mj_s:>10}{p['n']:>5}")
    print(f"\nGLMM ({conv}): fixed effects vs reference l1normal")
    for name, c in fe.items():
        if name.startswith("C(rung)"):
            print(f"  {name:<24} coef={c['coef']:+.3f}  CI{c['ci95']}")
    print(f"  RE sd: model={re_sd.get('model')}  menu={re_sd.get('menu')}")

    out = {
        "experiment_id": "h4-ladder-decomp",
        "question": "Q4 decomposition ladder: which planted-finding cue redirects selection",
        "rungs": RUNGS, "cue_added": CUE_ADDED,
        "n_trials": int(len(t)), "truncated_rows_dropped": dropped,
        "n_models": int(t["model"].nunique()), "n_menus": int(t["menu"].nunique()),
        "decoy_rate_pooled": pooled,
        "marginal_jump_vs_prev_rung": marg,
        "decoy_rate_per_model": per_model,
        "glmm": {"formula": "pick ~ C(rung) + (1|model) + (1|menu)", "reference_rung": "l1normal",
                 "convergence": conv, "fixed_effects": fe,
                 "re_sd_model": re_sd.get("model"), "re_sd_menu": re_sd.get("menu")},
        "seed": SEED,
        "note": "Marginal jump is the effect of adding ONE cue: l2vuln-l1normal isolates the VULNERABLE "
                "label, l3cve-l2vuln the named CVE, l4exploit-l3cve the demonstrated exploit.",
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
