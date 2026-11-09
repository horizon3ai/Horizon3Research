#!/usr/bin/env python3
"""Recency effect for E2 as a log-odds coefficient. The recency menus are 4-arm
choices spanning release years; selection is modeled with a conditional logit on
release year, P(choose a) = exp(beta*year_a) / sum_b exp(beta*year_b). beta is the
change in log-odds of selection per one-year increase (positive means newer CVEs are
chosen more), reported per year and per decade with a model-cluster bootstrap 95% CI.
Reads the recency trials (L2_cve, valid) from data/raw/h2-recency.csv. Stdlib only.
"""
import csv
import math
import random
import re
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

MODEL_IDS = [
    "google/gemini-3.1-pro-preview", "x-ai/grok-4.3", "moonshotai/kimi-k2.6",
    "z-ai/glm-5.2", "deepseek/deepseek-v4-pro", "anthropic/claude-opus-4.8",
    "openai/gpt-5.5",
]

# same recency menus as compute_e2_pickrates.py
RECENCY = {
    "S1": ("raw/h2-recency.csv", "h2_set1_ci_web"),
    "S2": ("raw/h2-recency.csv", "h2_set2_data"),
    "S3": ("raw/h2-recency.csv", "h2_set3_enterprise"),
}
INFO = "L2_cve"

_CACHE = {}


def read_csv(rel):
    if rel not in _CACHE:
        _CACHE[rel] = list(csv.DictReader(open(DATA / rel)))
    return _CACHE[rel]


def _year(arm):
    if "ms17" in str(arm).lower():
        return 2017
    m = re.search(r"(19|20)\d\d", str(arm))
    return int(m.group(0)) if m else -1


def load_trials():
    """Each trial -> (model, [years of presented arms], index of chosen arm)."""
    trials = []
    for rel, bundle in RECENCY.values():
        for r in read_csv(rel):
            if r["bundle_idx"] != bundle or r["info_level"] != INFO:
                continue
            if r["model"] not in MODEL_IDS:
                continue
            if str(r["choice_valid"]).lower() != "true":
                continue
            arms = str(r["presented_order"]).split("|")
            chosen = r["choice_matched"]
            if chosen not in arms:
                continue
            years = [_year(a) for a in arms]
            if any(y < 0 for y in years):
                continue
            trials.append((r["model"], years, arms.index(chosen)))
    return trials


def neg_ll(beta, trials, center):
    """Negative conditional-logit log-likelihood for scalar beta."""
    nll = 0.0
    for _, years, ci in trials:
        us = [beta * (y - center) for y in years]
        mx = max(us)
        denom = mx + math.log(sum(math.exp(u - mx) for u in us))
        nll -= (us[ci] - denom)
    return nll


def fit_beta(trials, center):
    """1-D Newton on the conditional-logit score. Returns beta (per year)."""
    beta = 0.0
    for _ in range(100):
        g = 0.0   # gradient of log-lik
        h = 0.0   # hessian
        for _, years, ci in trials:
            xs = [y - center for y in years]
            us = [beta * x for x in xs]
            mx = max(us)
            ws = [math.exp(u - mx) for u in us]
            s = sum(ws)
            ps = [w / s for w in ws]
            exp_x = sum(p * x for p, x in zip(ps, xs))
            g += xs[ci] - exp_x
            exp_x2 = sum(p * x * x for p, x in zip(ps, xs))
            h -= (exp_x2 - exp_x * exp_x)
        if abs(h) < 1e-12:
            break
        step = g / h
        beta -= step
        if abs(step) < 1e-9:
            break
    return beta


def main():
    trials = load_trials()
    years_all = sorted({y for _, ys, _ in trials for y in ys})
    center = sum(years_all) / len(years_all)
    n_by_model = {m: 0 for m in MODEL_IDS}
    for m, _, _ in trials:
        n_by_model[m] += 1

    beta = fit_beta(trials, center)
    print(f"trials: {len(trials)}  (models: "
          f"{', '.join(str(n_by_model[m]) for m in MODEL_IDS)})")
    print(f"distinct release years present: {years_all}")
    print(f"beta (log-odds per year)   = {beta:+.4f}")
    print(f"beta (log-odds per decade) = {beta*10:+.4f}")

    # model-cluster bootstrap
    by_model = {m: [] for m in MODEL_IDS}
    for t in trials:
        by_model[t[0]].append(t)
    rng = random.Random(20260903)
    boot = []
    for _ in range(600):
        samp = []
        for _ in MODEL_IDS:
            m = rng.choice(MODEL_IDS)
            samp.extend(by_model[m])
        boot.append(fit_beta(samp, center))
    boot.sort()
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    med = boot[len(boot) // 2]
    print(f"bootstrap median (per year) = {med:+.4f}")
    print(f"bootstrap 95% CI (per year) = [{lo:+.4f}, {hi:+.4f}]  "
          f"{'crosses 0' if lo < 0 < hi else 'excludes 0'}")
    print(f"bootstrap 95% CI (per decade) = [{lo*10:+.3f}, {hi*10:+.3f}]")

    # asymptotic SE from the fitted hessian, for a sanity cross-check
    h = 0.0
    for _, years, ci in trials:
        xs = [y - center for y in years]
        us = [beta * x for x in xs]
        mx = max(us)
        ws = [math.exp(u - mx) for u in us]
        s = sum(ws)
        ps = [w / s for w in ws]
        ex = sum(p * x for p, x in zip(ps, xs))
        ex2 = sum(p * x * x for p, x in zip(ps, xs))
        h += (ex2 - ex * ex)
    se = 1 / math.sqrt(h) if h > 0 else float("nan")
    z = NormalDist()
    print(f"asymptotic SE (per year) = {se:.4f}  "
          f"Wald 95% CI [{beta-1.96*se:+.4f}, {beta+1.96*se:+.4f}]  "
          f"p={2*(1-z.cdf(abs(beta/se))):.3f}")


if __name__ == "__main__":
    main()
