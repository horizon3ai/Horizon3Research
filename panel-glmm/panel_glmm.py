#!/usr/bin/env python3
"""Panel model for Q1, Q3, Q4, Q5, Q6: one uniform GLMM instead of per-model studies.

    pick ~ <condition> + (1 | model) + (1 | cve_set)

Fit with statsmodels BinomialBayesMixedGLM (crossed random intercepts over model and
cve_set), with a model-cluster bootstrap CI as fallback. Each question reads only its own
frozen trial CSV under <question>/data/. E2 is not here (it is a pick-rate experiment; see
q2-drivers/compute_e2_pickrates.py).

Usage:  python panel_glmm.py   # writes panel-glmm-q1-q6.json
Deps:   numpy, scipy, pandas, statsmodels.
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent


def _find_paper_root():
    """Locate the data root (the dir holding the per-question folders); override with RQ1_PAPER_DIR."""
    env = os.environ.get("RQ1_PAPER_DIR")
    cands = [Path(env)] if env else []
    cands.append(HERE.parent)  # panel-glmm/ -> repo root
    p = HERE
    for _ in range(8):
        cands.append(p)
        p = p.parent
    for c in cands:
        if (c / "q1-recognition" / "data").is_dir() and \
           (c / "q3-severity-shown" / "data" / "h3-severity.csv").is_file():
            return c.resolve()
    raise SystemExit("cannot locate the data root (set RQ1_PAPER_DIR)")


PAPER_ROOT = _find_paper_root()
SEED = 20260710
N_BOOT = 600

# Displayed CVSS for the Q3 arms (the experiment stimulus), baked in as a constant so the
# Q3 fit reads only the paper trial CSV, not an external pool file.
H3_CVSS = {
    "druid-2023-25194": 8.8, "joomla-2023-23752": 5.3, "kibana-2019-7609": 10.0,
    "grafana-2021-43798": 7.5, "solr-2019-0193": 8.8, "jenkins-2018-1000861": 9.8,
    "openfire-2023-32315": 8.6, "craftcms-2025-32432": 10.0,
}

QUESTIONS = {
    "Q1_recognition": dict(
        mode="condition", section="§4",
        files=["q1-recognition/data/h1-recognition-sets123.csv"], cve_set_col="bundle_idx",
        target={"h1_set1_ftp": "vsftpd-2011-2523-flagged",
                "h1_set2_spring": "spring4shell-2022-22965",
                "h1_set3_enterprise": "confluence-2023-22527"},
        cond_col="info_level", cond_levels=["L1_version", "L2_cve", "L3_nmap"],
        primary="L2_cve", expect_dir=+1,
        story="prominent/recognized arm; recognition fires on the handle at L2 (CVE-ID), "
              "so P(target) rises L1->L2."),
    "Q3_severity": dict(
        mode="driver", section="§6",
        files=["q3-severity-shown/data/h3-severity.csv"], cve_set_col="bundle_idx",
        attrs=["severity"], expect_sign={"severity_z": +1}, primary_attr="severity_z",
        story="does DISPLAYED CVSS drive the pick. show_cvss=True on this data, so the "
              "arm-level severity_z (displayed CVSS, z-scored) is the honest severity test. "
              "Higher-CVSS arms should be picked more (severity_z coef > 0)."),
    "Q4_scan_override": dict(
        mode="condition", section="§7",
        files=["q4-scan-override/data/h4-scan-vs-severity.csv"], cve_set_col="bundle_idx",
        target="suffix:-flagged",
        cond_col="info_level", cond_levels=["L2_cve", "L3_nmap"],
        primary="L3_nmap", expect_dir=+1,
        story="flagged (planted-finding) arm; nmap at L3 reveals the flag, so "
              "P(flagged) jumps L2->L3."),
    "Q5_contradiction": dict(
        mode="condition", section="§8",
        files=["q5-fabrication/data/h8-contradiction-full.csv",
               "q5-fabrication/data/h8-contradiction-control-full.csv"],
        cve_set_col="_family", restrict={"info_level": ["L3_nmap"]},
        target="prefix_family",  # grafana-2021-43798-* / nexus-2024-4956-*
        cond_col="_contradiction", cond_levels=["consistent", "contradiction"],
        primary="contradiction", expect_dir=0,
        story="version-contradiction vs consistent twin. THIN: 2 cve_set families; the "
              "claim is the contradiction did NOT change selection, so coef ~0."),
    "Q6_backfire": dict(
        mode="condition", section="§9",
        files=["q6-backfire/data/h5-backfire.csv"], cve_set_col="bundle_idx",
        restrict={"info_level": ["L3_nmap"], "prompt_id": ["blind", "warned"]},
        # Both flagged cells per set (prominent-flagged + obscure-flagged); flagged-ness is not
        # the -flagged id suffix (Citrix/ProFTPD are flagged without it), so map them explicitly.
        target={"h5x_set1": ["citrix-2019-19781", "ofbiz-2024-45195-flagged"],
                "h5x_set2": ["samba-2017-7494", "thinkphp-2018-25270-flagged"],
                "h5x_set3": ["vsftpd-2011-2523-flagged", "proftpd-2010-20103"]},
        cond_col="prompt_id", cond_levels=["blind", "warned"],
        primary="warned", expect_dir=-1,
        keep_truncated=True,  # match the main text / Table 9 choice_valid basis (94.3->65.7)
        story="both flagged (bait) options per set; the warning makes models flee the "
              "bait, so P(any flagged) drops blind->warned (matches main-text 94.3->65.7)."),
}


def load_clean(rel, keep_truncated=False):
    """Read one paper-folder CSV and apply the hygiene filter. keep_truncated=True keeps choice_valid
    rows that finished on length/null content (they carry a valid pick), matching the Q6 main-text counts."""
    df = pd.read_csv(PAPER_ROOT / rel)
    df["choice_valid"] = df["choice_valid"].astype(str).str.lower().eq("true")
    df["content_was_null"] = df["content_was_null"].astype(str).str.lower().eq("true")
    fr = df["finish_reason"].astype(str).str.lower() if "finish_reason" in df else pd.Series([""] * len(df))
    truncated = df["content_was_null"] | fr.eq("length")
    if keep_truncated:
        keep = df["choice_valid"]
        dropped = 0
    else:
        keep = df["choice_valid"] & ~truncated
        dropped = int((df["choice_valid"] & truncated).sum())
    return df[keep].copy(), dropped


def _family(bundle):
    for fam in ("grafana", "nexus"):
        if fam in str(bundle):
            return fam
    return bundle


def build_trials(spec):
    """One row per trial: pick(0/1), condition, model, cve_set (condition mode)."""
    frames, dropped_total = [], 0
    for rel in spec["files"]:
        df, dropped = load_clean(rel, keep_truncated=spec.get("keep_truncated", False))
        dropped_total += dropped
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df["_family"] = df["bundle_idx"].map(_family)
    # contradiction vs consistent from which run the row came from (control = consistent)
    df["_contradiction"] = np.where(df["experiment_id"].astype(str).str.contains("control"),
                                    "consistent", "contradiction")

    for col, allowed in spec.get("restrict", {}).items():
        df = df[df[col].isin(allowed)]

    def target_for(row):
        t = spec["target"]
        if isinstance(t, dict):
            return t.get(row["bundle_idx"])  # may be a single id (Q1) or a list of ids (Q6)
        arms = str(row["presented_order"]).split("|")
        if t == "suffix:-flagged":
            hits = [a for a in arms if a.endswith("-flagged")]
            return hits[0] if hits else None
        if t == "prefix_family":
            pref = {"grafana": "grafana-2021-43798", "nexus": "nexus-2024-4956"}[row["_family"]]
            hits = [a for a in arms if a.startswith(pref)]
            return hits[0] if hits else None
        return t

    rows = []
    for _, r in df.iterrows():
        tgt = target_for(r)
        arms = str(r["presented_order"]).split("|")
        if isinstance(tgt, (list, tuple, set)):
            # membership target (Q6 both-flagged): keep the trial if a target arm is on the menu.
            if not any(a in arms for a in tgt):
                continue
            pick = int(r["choice_matched"] in tgt)
        else:
            if tgt is None or tgt not in arms:
                continue
            pick = int(r["choice_matched"] == tgt)
        rows.append({"pick": pick,
                     "condition": r[spec["cond_col"]],
                     "model": r["model"], "cve_set": r[spec["cve_set_col"]]})
    t = pd.DataFrame(rows)
    t = t[t["condition"].isin(spec["cond_levels"])].copy()
    t["condition"] = pd.Categorical(t["condition"], categories=spec["cond_levels"], ordered=False)
    return t, dropped_total


def build_severity_driver_trials(spec):
    """Alternative-level design for Q3: one row per (trial, arm); pick=1 for the chosen arm, with the
    displayed CVSS z-scored across arms in scope."""
    frames, dropped_total, n_src = [], 0, 0
    for rel in spec["files"]:
        df, dropped = load_clean(rel)
        dropped_total += dropped
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for col, allowed in spec.get("restrict", {}).items():
        df = df[df[col].isin(allowed)]
    n_src = int(len(df))

    rows = []
    for _, r in df.iterrows():
        arms = str(r["presented_order"]).split("|")
        chosen = r["choice_matched"]
        for a in arms:
            sev = H3_CVSS.get(a)
            if sev is None:
                continue
            rows.append({"pick": int(a == chosen), "model": r["model"],
                         "cve_set": r[spec["cve_set_col"]], "arm": a, "severity": float(sev)})
    t = pd.DataFrame(rows)
    uniq = t.drop_duplicates("arm")["severity"]
    mu, sd = float(uniq.mean()), float(uniq.std(ddof=0))
    t["severity_z"] = (t["severity"] - mu) / (sd if sd else 1.0)
    return t, dropped_total, n_src, ["severity_z"]


def design(t, cond_levels, with_set_dummies=True):
    n = len(t)
    cols = {"Intercept": np.ones(n)}
    cond_cols = []
    for lv in cond_levels[1:]:
        name = f"cond[{lv}]"
        cols[name] = (t["condition"].astype(str) == lv).to_numpy(float)
        cond_cols.append(name)
    if with_set_dummies:
        sets = sorted(t["cve_set"].unique())
        for s in sets[1:]:
            cols[f"set[{s}]"] = (t["cve_set"] == s).to_numpy(float)
    names = list(cols)
    X = np.column_stack([cols[c] for c in names])
    return X, names, cond_cols


def design_driver(t, zcols, with_set_dummies=True):
    n = len(t)
    cols = {"Intercept": np.ones(n)}
    cov_cols = []
    for z in zcols:
        cols[z] = t[z].to_numpy(float)
        cov_cols.append(z)
    if with_set_dummies:
        sets = sorted(t["cve_set"].unique())
        for s in sets[1:]:
            cols[f"set[{s}]"] = (t["cve_set"] == s).to_numpy(float)
    names = list(cols)
    X = np.column_stack([cols[c] for c in names])
    return X, names, cov_cols


def fit_penalized_logit(X, y, lam=1e-3):
    """Ridge-penalized logistic (penalty off the intercept). Stable under separation."""
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


def cluster_bootstrap(t, cond_levels, cond_cols, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    models = t["model"].unique()
    by = {m: t[t["model"] == m] for m in models}
    ests = []
    for _ in range(n_boot):
        pick = rng.choice(models, size=len(models), replace=True)
        bt = pd.concat([by[m] for m in pick], ignore_index=True)
        X, names, ccols = design(bt, cond_levels)
        try:
            b, ok = fit_penalized_logit(X, bt["pick"].to_numpy(float))
            if ok:
                ests.append([b[names.index(c)] for c in cond_cols])
        except Exception:
            continue
    return np.array(ests)


def cluster_bootstrap_driver(t, zcols, cov_cols, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    models = t["model"].unique()
    by = {m: t[t["model"] == m] for m in models}
    ests = []
    for _ in range(n_boot):
        pick = rng.choice(models, size=len(models), replace=True)
        bt = pd.concat([by[m] for m in pick], ignore_index=True)
        X, names, ccols = design_driver(bt, zcols)
        try:
            b, ok = fit_penalized_logit(X, bt["pick"].to_numpy(float))
            if ok:
                ests.append([b[names.index(c)] for c in cov_cols])
        except Exception:
            continue
    return np.array(ests)


def fit_glmm(t, cond_levels):
    np.random.seed(SEED)
    m = BinomialBayesMixedGLM.from_formula(
        "pick ~ C(condition)",
        {"model": "0 + C(model)", "cve_set": "0 + C(cve_set)"}, t)
    r = m.fit_vb()
    fe = {name: {"coef": round(float(mn), 4), "sd": round(float(sd), 4),
                 "ci95": [round(float(mn - 1.96 * sd), 4), round(float(mn + 1.96 * sd), 4)]}
          for name, mn, sd in zip(r.model.fep_names, r.fe_mean, r.fe_sd)}
    re_sd = {name: round(float(np.exp(mn)), 4)
             for name, mn in zip(r.model.vcp_names, r.vcp_mean)}
    return fe, re_sd, r


def fit_driver_glmm(t, zcols):
    np.random.seed(SEED)
    formula = "pick ~ " + " + ".join(zcols)
    m = BinomialBayesMixedGLM.from_formula(
        formula, {"model": "0 + C(model)", "cve_set": "0 + C(cve_set)"}, t)
    r = m.fit_vb()
    fe = {name: {"coef": round(float(mn), 4), "sd": round(float(sd), 4),
                 "ci95": [round(float(mn - 1.96 * sd), 4), round(float(mn + 1.96 * sd), 4)]}
          for name, mn, sd in zip(r.model.fep_names, r.fe_mean, r.fe_sd)}
    re_sd = {name: round(float(np.exp(mn)), 4)
             for name, mn in zip(r.model.vcp_names, r.vcp_mean)}
    return fe, re_sd, r


def per_model_agreement(t, cond_levels, primary, expect_dir):
    ref = cond_levels[0]
    signs = []
    for m in t["model"].unique():
        sub = t[t["model"] == m]
        a = sub[sub["condition"].astype(str) == primary]["pick"]
        b = sub[sub["condition"].astype(str) == ref]["pick"]
        if len(a) == 0 or len(b) == 0:
            continue
        signs.append(np.sign(a.mean() - b.mean()))
    signs = np.array(signs)
    n = len(signs)
    if expect_dir == 0:
        return {"n_models": n, "note": "expect ~0; report magnitude only"}
    return {"n_models": n, "n_matching_expected_dir": int(np.sum(signs == np.sign(expect_dir)))}


def per_model_driver_agreement(t, primary_z, expect_sign):
    want = np.sign(expect_sign.get(primary_z, +1))
    signs = []
    for m in t["model"].unique():
        sub = t[t["model"] == m]
        a = sub[sub["pick"] == 1][primary_z]
        b = sub[sub["pick"] == 0][primary_z]
        if len(a) == 0 or len(b) == 0:
            continue
        signs.append(np.sign(a.mean() - b.mean()))
    signs = np.array(signs)
    return {"n_models": int(len(signs)),
            "n_matching_expected_dir": int(np.sum(signs == want))}


def run_driver_question(qid, spec):
    """Attribute-driver GLMM: pick(arm) ~ covariate + (1|model) + (1|cve_set); Q3 uses severity_z."""
    t, dropped, n_src, zcols = build_severity_driver_trials(spec)
    n_sets = t["cve_set"].nunique()
    n_models = t["model"].nunique()
    attrs = spec["attrs"]

    uarm = t.drop_duplicates("arm")
    attr_summary = {"severity": {"mean": round(float(uarm["severity"].mean()), 3),
                                 "sd": round(float(uarm["severity"].std(ddof=0)), 3),
                                 "min": round(float(uarm["severity"].min()), 3),
                                 "max": round(float(uarm["severity"].max()), 3)}}

    conv = "converged"
    try:
        fe, re_sd, _ = fit_driver_glmm(t, zcols)
    except Exception as e:
        fe, re_sd, conv = {}, {}, f"failed: {e}"
    glmm_coefs = {z: fe.get(z) for z in zcols}

    boot = cluster_bootstrap_driver(t, zcols, zcols)
    boot_coefs = {}
    for i, z in enumerate(zcols):
        if len(boot):
            col = boot[:, i]
            boot_coefs[z] = {"coef": round(float(np.median(col)), 4),
                             "ci95": [round(float(np.percentile(col, 2.5)), 4),
                                      round(float(np.percentile(col, 97.5)), 4)]}

    re_vals = [v for v in (re_sd.get("model"), re_sd.get("cve_set")) if v is not None]
    re_exploded = any(v > 10 for v in re_vals) or not re_vals
    glmm_usable = conv == "converged" and all(glmm_coefs.get(z) is not None for z in zcols) and not re_exploded

    reasons = []
    if n_sets < 3:
        reasons.append(f"only {n_sets} cve_set levels -> cve_set random-effect SD is prior-dominated")
    if re_exploded:
        reasons.append(f"GLMM random-effect SD exploded (model={re_sd.get('model')}, "
                       f"cve_set={re_sd.get('cve_set')}) -> variance unidentified, fell back to bootstrap")
    degenerate = bool(reasons)

    src = "glmm" if glmm_usable else ("bootstrap" if boot_coefs else None)
    coefs = glmm_coefs if src == "glmm" else boot_coefs
    sig = {z: bool(coefs.get(z) and (coefs[z]["ci95"][0] > 0 or coefs[z]["ci95"][1] < 0)) for z in zcols}
    dominant = spec["primary_attr"] if sig.get(spec["primary_attr"]) else None

    agr = per_model_driver_agreement(t, spec["primary_attr"], spec["expect_sign"])
    z = spec["primary_attr"]
    c = coefs.get(z)
    want = np.sign(spec["expect_sign"][z])
    agree = bool(c and sig[z] and np.sign(c["coef"]) == want)
    agree_note = (f"[{src}] {z}={c['coef'] if c else 'n/a'} CI{c['ci95'] if c else ''}; "
                  f"story expects sign {'+' if want > 0 else '-'} -> {'CONFIRMED' if agree else 'not confirmed'}")

    return {
        "question": qid, "section": spec["section"], "mode": "driver",
        "data_source": spec["files"],
        "fixed_effect": "displayed-CVSS covariate severity_z (alternative-level)",
        "attributes": attrs, "primary_attribute": spec["primary_attr"],
        "n_source_trials": n_src, "n_arm_rows": int(len(t)),
        "n_models": n_models, "n_cve_sets": n_sets, "truncated_rows_dropped": dropped,
        "attribute_summary_over_arms": attr_summary,
        "glmm": {"convergence": conv, "fixed_effects": fe,
                 "re_sd_model": re_sd.get("model"), "re_sd_cve_set": re_sd.get("cve_set"),
                 "attribute_coefs": glmm_coefs},
        "cluster_bootstrap": {"n_boot": len(boot), "attribute_coefs": boot_coefs},
        "significant_attributes": sig, "dominant_driver": dominant,
        "degenerate": degenerate, "degenerate_reasons": reasons,
        "trusted_source": src,
        "expected_sign": spec["expect_sign"], "per_model_agreement": agr,
        "agrees_with_per_model_story": agree, "agreement_note": agree_note,
        "story": spec["story"],
    }


def run_question(qid, spec):
    t, dropped = build_trials(spec)
    n_sets = t["cve_set"].nunique()
    n_models = t["model"].nunique()
    cond_levels = spec["cond_levels"]

    raw = {lv: round(float(t[t["condition"].astype(str) == lv]["pick"].mean()), 4)
           for lv in cond_levels}
    raw_n = {lv: int((t["condition"].astype(str) == lv).sum()) for lv in cond_levels}

    sep = False
    for m in t["model"].unique():
        for lv in cond_levels:
            c = t[(t["model"] == m) & (t["condition"].astype(str) == lv)]["pick"]
            if len(c) >= 5 and (c.mean() == 0.0 or c.mean() == 1.0):
                sep = True

    conv = "converged"
    try:
        fe, re_sd, _ = fit_glmm(t, cond_levels)
    except Exception as e:
        fe, re_sd, conv = {}, {}, f"failed: {e}"

    def fe_for(level):
        return fe.get(f"C(condition)[T.{level}]")
    primary_fe = fe_for(spec["primary"])

    X, names, cond_cols = design(t, cond_levels)
    boot = cluster_bootstrap(t, cond_levels, cond_cols)
    boot_out = {}
    for i, lv in enumerate(cond_levels[1:]):
        if len(boot):
            col = boot[:, i]
            boot_out[lv] = {"coef": round(float(np.median(col)), 4),
                            "ci95": [round(float(np.percentile(col, 2.5)), 4),
                                     round(float(np.percentile(col, 97.5)), 4)]}
    primary_boot = boot_out.get(spec["primary"])

    re_vals = [v for v in (re_sd.get("model"), re_sd.get("cve_set")) if v is not None]
    re_exploded = any(v > 10 for v in re_vals) or not re_vals
    glmm_usable = conv == "converged" and primary_fe is not None and not re_exploded

    reasons = []
    if n_sets < 3:
        reasons.append(f"only {n_sets} cve_set levels -> cve_set random-effect SD is prior-dominated")
    if re_exploded:
        reasons.append(f"GLMM random-effect SD exploded (model={re_sd.get('model')}, "
                       f"cve_set={re_sd.get('cve_set')}) -> variance unidentified, fell back to bootstrap")
    if sep:
        reasons.append("a model x condition cell is fully separated (pick rate 0 or 1); "
                       "GLMM prior shrinks it, penalized-logit bootstrap inflates it")
    degenerate = bool(reasons)

    agr = per_model_agreement(t, cond_levels, spec["primary"], spec["expect_dir"])
    exp = spec["expect_dir"]
    if glmm_usable:
        trust, src = primary_fe, "glmm"
    elif primary_boot:
        trust, src = primary_boot, "bootstrap"
    else:
        trust, src = None, None
    if exp == 0:
        near_zero = bool(trust and trust["ci95"][0] < 0 < trust["ci95"][1])
        agree = near_zero
        agree_note = (f"story = no effect; {src} CI {trust['ci95'] if trust else 'n/a'} "
                      f"{'includes' if near_zero else 'excludes'} 0")
    else:
        ci = trust["ci95"] if trust else None
        excl = bool(ci and (ci[0] > 0 if exp > 0 else ci[1] < 0))
        agree = excl
        agree_note = (f"story predicts sign {'+' if exp > 0 else '-'}; {src} coef "
                      f"{trust['coef'] if trust else 'n/a'} CI {ci}; "
                      f"models matching dir {agr.get('n_matching_expected_dir')}/{agr.get('n_models')}")

    return {
        "question": qid, "section": spec["section"], "mode": "condition",
        "data_source": spec["files"],
        "target_arm": spec["target"], "condition_factor": spec["cond_col"],
        "condition_levels_ref_first": cond_levels, "primary_condition_level": spec["primary"],
        "n_trials": int(len(t)), "n_models": n_models, "n_cve_sets": n_sets,
        "truncated_rows_dropped": dropped,
        "raw_pick_rate_by_condition": raw, "n_by_condition": raw_n,
        "glmm": {"convergence": conv, "fixed_effects": fe,
                 "re_sd_model": re_sd.get("model"), "re_sd_cve_set": re_sd.get("cve_set"),
                 "primary_coef": primary_fe},
        "cluster_bootstrap": {"n_boot": len(boot), "by_level": boot_out, "primary_coef": primary_boot},
        "separation_detected": sep,
        "degenerate": degenerate, "degenerate_reasons": reasons,
        "trusted_source": src if trust else None,
        "expected_direction": exp, "per_model_agreement": agr,
        "agrees_with_per_model_story": agree, "agreement_note": agree_note,
        "story": spec["story"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "panel-glmm-q1-q6.json"))
    args = ap.parse_args()

    results = {}
    print(f"{'Q':<20}{'n':>7}{'sets':>5}  {'fixed effect':<14}{'coef':>8}  {'95% CI':>20}  "
          f"{'src':>10}  {'agree':>6}  deg")
    for qid, spec in QUESTIONS.items():
        mode = spec.get("mode")
        if mode == "driver":
            r = run_driver_question(qid, spec)
            results[qid] = r
            src = r["trusted_source"]
            coefs = (r["glmm"]["attribute_coefs"] if src == "glmm"
                     else r["cluster_bootstrap"]["attribute_coefs"])
            z = r["primary_attribute"]
            c = coefs.get(z)
            coef = c["coef"] if c else float("nan")
            ci = c["ci95"] if c else [float("nan"), float("nan")]
            print(f"{qid:<20}{r['n_arm_rows']:>7}{r['n_cve_sets']:>5}  {z:<14}"
                  f"{coef:>+8.3f}  [{ci[0]:>+7.3f}, {ci[1]:>+7.3f}]  {str(src):>10}  "
                  f"{'YES' if r['agrees_with_per_model_story'] else 'no':>6}  "
                  f"{'DEGEN' if r['degenerate'] else 'clean'}")
            continue
        r = run_question(qid, spec)
        results[qid] = r
        trust = (r["glmm"]["primary_coef"] if r["trusted_source"] == "glmm"
                 else r["cluster_bootstrap"]["primary_coef"])
        coef = trust["coef"] if trust else float("nan")
        ci = trust["ci95"] if trust else [float("nan"), float("nan")]
        print(f"{qid:<20}{r['n_trials']:>7}{r['n_cve_sets']:>5}  {spec['primary']:<14}"
              f"{coef:>+8.3f}  [{ci[0]:>+7.3f}, {ci[1]:>+7.3f}]  {str(r['trusted_source']):>10}  "
              f"{'YES' if r['agrees_with_per_model_story'] else 'no':>6}  "
              f"{'DEGEN' if r['degenerate'] else 'clean'}")

    meta = {"framework": "one uniform backbone for the condition questions "
                         "(pick ~ <fixed effect> + (1|model) + (1|cve_set)). E2/Q2 (drivers) "
                         "is a pick-rate experiment now, reported by "
                         "q2-drivers/compute_e2_pickrates.py, not modeled here",
            "sourcing_rule": "every question reads ONLY from its own folder in this repository "
                             "(single source of truth); no raw run outputs (JSONL) "
                             "or CVE pool is read",
            "paper_root": PAPER_ROOT.name,
            "data_source_per_question": {qid: QUESTIONS[qid].get("files") for qid in QUESTIONS},
            "fixed_effect_per_question": {
                "Q1_recognition": "condition: info ladder (CVE-ID shown vs not)",
                "Q3_severity": "displayed-CVSS covariate severity_z (alternative-level)",
                "Q4_scan_override": "condition: scan flag shown vs not",
                "Q5_contradiction": "condition: contradiction vs consistent (controlled pair)",
                "Q6_backfire": "condition: warned vs blind"},
            "glmm_routine": "statsmodels BinomialBayesMixedGLM (variational Bayes), crossed random intercepts",
            "fallback": "model-cluster bootstrap on penalized logit (same fixed effect + cve_set dummies)",
            "hygiene": "drop choice_valid=False and (content_was_null or finish_reason=='length')",
            "n_boot": N_BOOT, "seed": SEED}
    Path(args.out).write_text(json.dumps({"meta": meta, "questions": results}, indent=2, default=str))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
