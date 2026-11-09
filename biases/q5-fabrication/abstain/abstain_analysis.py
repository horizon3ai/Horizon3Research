#!/usr/bin/env python3
"""Q5 abstain arm: does a neutral opt-out reveal hidden detection? Detection is the
abstain-rate difference between the contradiction arm and its consistent control at
L3_nmap, with a mixed-effects fit abstain ~ C(condition) + (1|model) + (1|menu). Opt-out
is rare, so a degenerate fit falls back to a model-cluster bootstrap CI. Reads data/.
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
DATA = HERE / "data"
CONTRA = "h8-contradiction-abstain.csv"          # nexus + grafana, contradiction arm
CONTROL = "h8-contradiction-control-abstain.csv"  # nexus + grafana, consistent arm
# Elasticsearch is a scan-only contradiction (genuine at L2, injected only in the L3 banner).
# Both conditions live in ONE file, split by arm: h8so_test = contradiction, h8so_control = consistent.
SCANONLY = "h8-fab-scanonly-abstain.csv"
SCANONLY_ARM = {"h8so_test": "contradiction", "h8so_control": "consistent"}
COND_LEVELS = ["consistent", "contradiction"]   # consistent = reference
RUNG = "L3_nmap"
SEED = 20260710
N_BOOT = 600


# load + hygiene
def load_clean(rel):
    """Read one paper-folder CSV; drop choice_valid=False and truncated (content_was_null or
    finish_reason=='length'). Abstain rows are choice_valid=True, so they are KEPT."""
    df = pd.read_csv(DATA / rel)
    df["choice_valid"] = df["choice_valid"].astype(str).str.lower().eq("true")
    df["content_was_null"] = df["content_was_null"].astype(str).str.lower().eq("true")
    fr = df["finish_reason"].astype(str).str.lower() if "finish_reason" in df else pd.Series([""] * len(df))
    truncated = df["content_was_null"] | fr.eq("length")
    # Keep every trial with a validly parsed choice, including truncated-but-valid ones
    # (uniform hygiene, matching panel_glmm.py); only unparseable picks are excluded.
    keep = df["choice_valid"]
    dropped = 0
    return df[keep].copy(), dropped


def _menu(bundle):
    for fam in ("grafana", "nexus"):
        if fam in str(bundle):
            return fam
    if "h8so" in str(bundle):
        return "elasticsearch"
    return str(bundle)


def _abstain(row):
    """Robust abstain read: the engine `abstain` column if present, else matched_by / sentinel id."""
    if "abstain" in row and pd.notna(row["abstain"]):
        return 1 if str(row["abstain"]).lower() == "true" else 0
    if str(row.get("matched_by", "")).lower() == "abstain":
        return 1
    return 1 if str(row.get("choice_matched", "")) == "__ABSTAIN__" else 0


def build():
    frames, dropped = [], 0
    for rel, cond in ((CONTRA, "contradiction"), (CONTROL, "consistent")):
        df, d = load_clean(rel)
        dropped += d
        df["_condition"] = cond
        frames.append(df)
    # Elasticsearch: one file, condition assigned by arm (h8so_test / h8so_control).
    es, d = load_clean(SCANONLY)
    dropped += d
    es["_condition"] = es["bundle_idx"].astype(str).map(SCANONLY_ARM)
    es = es[es["_condition"].notna()]
    frames.append(es)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["info_level"].astype(str) == RUNG].copy()
    df["_menu"] = df["bundle_idx"].map(_menu)
    df["abstain"] = df.apply(_abstain, axis=1)
    t = pd.DataFrame({
        "abstain": df["abstain"].astype(int),
        "condition": pd.Categorical(df["_condition"], categories=COND_LEVELS, ordered=False),
        "model": df["model"].astype(str),
        "menu": df["_menu"].astype(str),
    })
    return t, dropped


# rates
def rates(t):
    """Pooled + per-model abstain rate by condition and the detection difference."""
    def rate(sub, cond):
        s = sub[sub["condition"].astype(str) == cond]["abstain"]
        return (round(float(s.mean()), 4) if len(s) else None), int(len(s))

    pooled = {}
    for cond in COND_LEVELS:
        r, n = rate(t, cond)
        pooled[cond] = {"rate": r, "n": n}
    det = (None if pooled["contradiction"]["rate"] is None or pooled["consistent"]["rate"] is None
           else round(pooled["contradiction"]["rate"] - pooled["consistent"]["rate"], 4))
    per_model = {}
    for m in sorted(t["model"].unique()):
        sub = t[t["model"] == m]
        row = {}
        for cond in COND_LEVELS:
            r, n = rate(sub, cond)
            row[cond] = {"rate": r, "n": n}
        row["detection_diff"] = (None if row["contradiction"]["rate"] is None
                                 or row["consistent"]["rate"] is None
                                 else round(row["contradiction"]["rate"] - row["consistent"]["rate"], 4))
        per_model[m] = row
    return {"pooled": {**pooled, "detection_diff": det}, "per_model": per_model}


# penalized logit + bootstrap
def design(t, with_menu_dummies=True):
    n = len(t)
    cols = {"Intercept": np.ones(n)}
    cols["cond[contradiction]"] = (t["condition"].astype(str) == "contradiction").to_numpy(float)
    if with_menu_dummies:
        menus = sorted(t["menu"].unique())
        for s in menus[1:]:
            cols[f"menu[{s}]"] = (t["menu"] == s).to_numpy(float)
    names = list(cols)
    X = np.column_stack([cols[c] for c in names])
    return X, names


def fit_penalized_logit(X, y, lam=1e-3):
    p = X.shape[1]
    pen = np.ones(p) * lam
    pen[0] = 0.0

    def nll_grad(b):
        z = np.clip(X @ b, -30, 30)
        pr = 1 / (1 + np.exp(-z))
        nll = -np.sum(y * np.log(pr + 1e-12) + (1 - y) * np.log(1 - pr + 1e-12)) + 0.5 * np.sum(pen * b * b)
        grad = X.T @ (pr - y) + pen * b
        return nll, grad

    r = minimize(nll_grad, np.zeros(p), jac=True, method="L-BFGS-B")
    return r.x, bool(r.success)


def cluster_bootstrap(t, n_boot=N_BOOT, seed=SEED):
    """Model-cluster bootstrap (resample the models) on the contradiction coef of a penalized logit."""
    rng = np.random.default_rng(seed)
    models = t["model"].unique()
    by = {m: t[t["model"] == m] for m in models}
    ests = []
    for _ in range(n_boot):
        pick = rng.choice(models, size=len(models), replace=True)
        bt = pd.concat([by[m] for m in pick], ignore_index=True)
        X, names = design(bt)
        try:
            b, ok = fit_penalized_logit(X, bt["abstain"].to_numpy(float))
            if ok:
                ests.append(b[names.index("cond[contradiction]")])
        except Exception:
            continue
    return np.array(ests)


# GLMM
def fit_glmm(t):
    np.random.seed(SEED)
    m = BinomialBayesMixedGLM.from_formula(
        "abstain ~ C(condition)",
        {"model": "0 + C(model)", "menu": "0 + C(menu)"}, t)
    r = m.fit_vb()
    fe = {name: {"coef": round(float(mn), 4), "sd": round(float(sd), 4),
                 "ci95": [round(float(mn - 1.96 * sd), 4), round(float(mn + 1.96 * sd), 4)]}
          for name, mn, sd in zip(r.model.fep_names, r.fe_mean, r.fe_sd)}
    re_sd = {name: round(float(np.exp(mn)), 4) for name, mn in zip(r.model.vcp_names, r.vcp_mean)}
    return fe, re_sd


# run
def analyze():
    t, dropped = build()
    n_menus = t["menu"].nunique()
    n_models = t["model"].nunique()
    rate_summary = rates(t)

    # separation / near-ceiling scan
    sep = False
    for m in t["model"].unique():
        for cond in COND_LEVELS:
            c = t[(t["model"] == m) & (t["condition"].astype(str) == cond)]["abstain"]
            if len(c) >= 5 and (c.mean() == 0.0 or c.mean() == 1.0):
                sep = True
    all_zero = bool(t["abstain"].sum() == 0)

    conv = "converged"
    fe, re_sd = {}, {}
    try:
        fe, re_sd = fit_glmm(t)
    except Exception as e:
        conv = f"failed: {e}"
    primary_fe = fe.get("C(condition)[T.contradiction]")

    boot = cluster_bootstrap(t)
    primary_boot = None
    if len(boot):
        primary_boot = {"coef": round(float(np.median(boot)), 4),
                        "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                                 round(float(np.percentile(boot, 97.5)), 4)]}

    re_vals = [v for v in (re_sd.get("model"), re_sd.get("menu")) if v is not None]
    re_exploded = any(v > 10 for v in re_vals) or not re_vals
    glmm_usable = conv == "converged" and primary_fe is not None and not re_exploded and not all_zero

    reasons = []
    if all_zero:
        reasons.append("abstain outcome is all-zero (no model opted out); GLMM degenerate, report raw "
                       "rate difference + model-cluster bootstrap CI")
    if n_menus < 3:
        reasons.append(f"only {n_menus} menu levels -> menu random-effect SD is prior-dominated")
    if re_exploded and not all_zero:
        reasons.append(f"GLMM random-effect SD exploded (model={re_sd.get('model')}, "
                       f"menu={re_sd.get('menu')}) -> variance unidentified, fell back to bootstrap")
    if sep:
        reasons.append("a model x condition cell is fully separated (abstain rate 0 or 1); "
                       "GLMM prior shrinks it, penalized-logit bootstrap inflates it")
    degenerate = bool(reasons)

    if glmm_usable:
        trust, src = primary_fe, "glmm"
    elif primary_boot and not all_zero:
        trust, src = primary_boot, "bootstrap"
    else:
        trust, src = None, "raw_difference"

    det = rate_summary["pooled"]["detection_diff"]
    if src == "raw_difference":
        note = (f"detection signal = raw abstain-rate difference = {det} "
                f"(contradiction {rate_summary['pooled']['contradiction']['rate']} - consistent "
                f"{rate_summary['pooled']['consistent']['rate']}); model too degenerate for a coef CI "
                f"({'; '.join(reasons)})")
    else:
        ci = trust["ci95"]
        excl = bool(ci[0] > 0 or ci[1] < 0)
        note = (f"[{src}] C(condition)[T.contradiction] coef {trust['coef']} CI {ci}; "
                f"raw abstain-rate difference = {det}. "
                f"{'CI excludes 0 -> contradiction shifts opt-out (detection)' if excl else 'CI includes 0 -> no reliable opt-out shift (forced-choice ceiling was not hiding detection)'}")

    return {
        "question": "Q5_abstain",
        "design_note": "forced choice can hide detection; add a neutral opt-out and read "
                          "detection as the abstain-rate difference (contradiction - consistent)",
        "rung": RUNG,
        "condition_levels_ref_first": COND_LEVELS,
        "data_source": [f"data/{CONTRA}", f"data/{CONTROL}", f"data/{SCANONLY}"],
        "n_trials": int(len(t)), "n_models": n_models, "n_menus": n_menus,
        "truncated_rows_dropped": dropped,
        "abstain_rate": rate_summary,
        "detection_signal_pooled": det,
        "separation_detected": sep, "abstain_all_zero": all_zero,
        "glmm": {"convergence": conv, "fixed_effects": fe,
                 "re_sd_model": re_sd.get("model"), "re_sd_menu": re_sd.get("menu"),
                 "primary_coef": primary_fe},
        "cluster_bootstrap": {"n_boot": len(boot), "primary_coef": primary_boot},
        "degenerate": degenerate, "degenerate_reasons": reasons,
        "trusted_source": src, "interpretation": note,
        "glmm_routine": "statsmodels BinomialBayesMixedGLM (variational Bayes), crossed (1|model)+(1|menu)",
        "fallback": "model-cluster bootstrap on penalized logit (contradiction coef + menu dummies)",
        "n_boot": N_BOOT, "seed": SEED,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "abstain-analysis.json"))
    args = ap.parse_args()
    for rel in (CONTRA, CONTROL, SCANONLY):
        if not (DATA / rel).exists():
            raise SystemExit(f"missing {DATA / rel}; run run.sh (full) then copy CSVs into data/")
    r = analyze()
    Path(args.out).write_text(json.dumps(r, indent=2, default=str))
    p = r["abstain_rate"]["pooled"]
    print(f"n={r['n_trials']} models={r['n_models']} menus={r['n_menus']}  rung={r['rung']}")
    print(f"abstain: contradiction={p['contradiction']['rate']} (n={p['contradiction']['n']})  "
          f"consistent={p['consistent']['rate']} (n={p['consistent']['n']})")
    print(f"DETECTION signal (contradiction - consistent) = {r['detection_signal_pooled']}")
    print(f"trusted_source={r['trusted_source']}  degenerate={r['degenerate']}")
    print(r["interpretation"])
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
