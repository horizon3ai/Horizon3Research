"""Per-hypothesis descriptive metrics for the H-suite runs, with CIs and spec support verdicts.

For each hypothesis this computes the metric its spec's Evaluation section names: attraction
A(c,v)=picks/exposures with Wilson 95% CIs, and the hypothesis GAP (RB, ΔA_rec, CSG, ΔA_scan,
LE/ΔLE/Recovery, δ_rec/δ_scan, FA, RS + factorial) with a trial-level bootstrap 95% CI: per model
and pooled, then applies each spec's "≥2/3 sets, ≥5/7 models, CI excludes 0 in the predicted
direction" rule. Descriptive companion layer (METHODOLOGY §6.0); the primary random-utility model is
analysis/e4_rum.py.

Writes out/<id>-metrics.csv (one row per contrast × scope) and (with --all) analysis/hyp-verdicts.json.
Usage: python analysis/hyp_metrics.py --id h1-recognition   |   --all
"""



import math

from pathlib import Path

import numpy as np

import pandas as pd

HERE = Path(__file__).resolve().parent

ROOT = HERE.parent

N_BOOT_POOLED = 10000

N_BOOT_MODEL = 2500

SEED = 20260710

MODELS = ["anthropic/claude-opus-4.8", "deepseek/deepseek-v4-pro", "google/gemini-3.1-pro-preview",
          "moonshotai/kimi-k2.6", "openai/gpt-5.5", "x-ai/grok-4.3", "z-ai/glm-5.2"]

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(p, 4), round((c - h) / d, 4), round((c + h) / d, 4))

def cell(df, *, bundle=None, model=None, prompt=None, info=None, temp=None):
    m = pd.Series(True, index=df.index)
    if bundle is not None:
        m &= df["bundle_idx"] == bundle
    if model is not None:
        m &= df["model"] == model
    if prompt is not None:
        m &= df["prompt_id"] == prompt
    if info is not None:
        m &= df["info_level"] == info
    if temp is not None:
        m &= np.isclose(df["temperature"], temp)
    return df[m]

def _picked(sub, arm):
    return ((sub["choice_matched"] == arm) & sub["choice_valid"]).to_numpy(dtype=float)

def _exp(sub, arm):
    return sub["_ids"].apply(lambda ids: arm in ids).to_numpy(dtype=bool)

def attr(sub, arm):
    exp = int(_exp(sub, arm).sum())
    pk = int((_picked(sub, arm).astype(bool) & _exp(sub, arm)).sum())
    return (pk, exp) + wilson(pk, exp)

def _pctl(a):
    return (round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4))

def gap(df, pos, neg, *, bundle_pos=None, bundle_neg=None, model=None, rng=None, n_boot=N_BOOT_POOLED):
    """A(pos.arm|pos.cell) - A(neg.arm|neg.cell) with bootstrap CI. cell keys: prompt/info/temp."""
    bundle_neg = bundle_neg if bundle_neg is not None else bundle_pos
    sp = cell(df, bundle=bundle_pos, model=model, prompt=pos.get("prompt"), info=pos.get("info"), temp=pos.get("temp"))
    sn = cell(df, bundle=bundle_neg, model=model, prompt=neg.get("prompt"), info=neg.get("info"), temp=neg.get("temp"))
    pk_p, ex_p, Ap, Ap_lo, Ap_hi = attr(sp, pos["arm"])
    pk_n, ex_n, An, An_lo, An_hi = attr(sn, neg["arm"])
    same = (bundle_pos == bundle_neg and pos.get("prompt") == neg.get("prompt")
            and pos.get("info") == neg.get("info") and pos.get("temp") == neg.get("temp"))
    if same and len(sp):
        ip, ineg = _picked(sp, pos["arm"]), _picked(sp, neg["arm"])
        idx = rng.integers(0, len(sp), size=(n_boot, len(sp)))
        g = ip[idx].mean(1) - ineg[idx].mean(1)
        glo, ghi = _pctl(g)
    elif len(sp) and len(sn):
        ip, ineg = _picked(sp, pos["arm"]), _picked(sn, neg["arm"])
        a = ip[rng.integers(0, len(ip), size=(n_boot, len(ip)))].mean(1)
        b = ineg[rng.integers(0, len(ineg), size=(n_boot, len(ineg)))].mean(1)
        glo, ghi = _pctl(a - b)
    else:
        glo, ghi = (0.0, 0.0)
    return dict(A_pos=Ap, A_pos_lo=Ap_lo, A_pos_hi=Ap_hi, n_pos=ex_p,
                A_neg=An, A_neg_lo=An_lo, A_neg_hi=An_hi, n_neg=ex_n,
                gap=round(Ap - An, 4), gap_lo=glo, gap_hi=ghi)

def excl0(g, sign=1):
    return bool(g["gap_lo"] > 0) if sign > 0 else bool(g["gap_hi"] < 0)

def run(df, eid, contrasts, rng):
    """contrasts: [{label,set,pos,neg,[bundle_neg],[sign]}]. Returns (rows, summaries[list])."""
    rows, summaries = [], []
    for c in contrasts:
        b, bn, sign = c.get("set"), c.get("bundle_neg", c.get("set")), c.get("sign", 1)
        gp = gap(df, c["pos"], c["neg"], bundle_pos=b, bundle_neg=bn, rng=rng, n_boot=N_BOOT_POOLED)
        rows.append({"hyp": eid, "contrast": c["label"], "set": b or "-", "scope": "pooled", "model": "-",
                     "sign": sign, **gp, "ci_excl_0": excl0(gp, sign)})
        n_ok = 0
        for m in MODELS:
            gm = gap(df, c["pos"], c["neg"], bundle_pos=b, bundle_neg=bn, model=m, rng=rng, n_boot=N_BOOT_MODEL)
            ok = excl0(gm, sign)
            n_ok += int(ok)
            rows.append({"hyp": eid, "contrast": c["label"], "set": b or "-", "scope": "model",
                         "model": m.split("/")[-1], "sign": sign, **gm, "ci_excl_0": ok})
        summaries.append({"label": c["label"], "set": b, "sign": sign, "pooled_gap": gp["gap"],
                          "pooled_ci": [gp["gap_lo"], gp["gap_hi"]], "pooled_supported": excl0(gp, sign),
                          "models_supporting": n_ok})
    return rows, summaries

def find(summaries, label, bundle=None):
    return [s for s in summaries if s["label"] == label and (bundle is None or s["set"] == bundle)]


FAM_OBS = {"h1_set1_ftp": ("vsftpd-2011-2523-flagged", "proftpd-2010-20103"),
           "h1_set2_spring": ("spring4shell-2022-22965", "opentsdb-2023-25826"),
           "h1_set3_enterprise": ("confluence-2023-22527", "ofbiz-2024-45195")}

def gap_ladder(df, eid, sets, rng, conds=("L1_version", "L2_cve", "L3_nmap"), metric="RB"):
    c1, c2, c3 = conds
    contrasts = []
    for b, (pos, neg) in sets.items():
        for c, lbl in [(c1, f"{metric}@C1"), (c2, f"{metric}@C2"), (c3, f"{metric}@C3")]:
            contrasts.append({"label": lbl, "set": b, "pos": {"arm": pos, "info": c}, "neg": {"arm": neg, "info": c}})
        contrasts.append({"label": "dA_rec(C2-C1)", "set": b, "pos": {"arm": pos, "info": c2}, "neg": {"arm": pos, "info": c1}})
    rows, summ = run(df, eid, contrasts, rng)
    sets_ok = [b for b in sets if find(summ, f"{metric}@C2", b)[0]["models_supporting"] >= 5]
    verdict = {"metric": metric, "rule": ">=2/3 sets with >=5/7 models CI∌0 at C2",
               "sets_supported_C2": sets_ok, "supported": len(sets_ok) >= 2,
               "detail": {f"{s['label']}|{s['set']}": {"gap": s["pooled_gap"], "ci": s["pooled_ci"],
                          "models": s["models_supporting"]} for s in summ}}
    return rows, verdict
