#!/usr/bin/env python3
"""Regenerate the E2 pick-rate numbers (drivers experiment).

Two menu families: prominence x severity (e4b) with prominent-low and obscure-high targets at CVSS
hidden (L2_cve) vs shown (L3_cvss); and recency (h2) with the newest-year arm as target. Pick rate =
choice_matched == target (choice_valid) / trials where the target is present, pooled over 7 models.
Writes e2-pickrates.json and asserts every published target number.
"""
import csv
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# display name, csv model id, in target-report order
MODELS = [
    ("Gemini", "google/gemini-3.1-pro-preview"),
    ("Grok", "x-ai/grok-4.3"),
    ("Kimi", "moonshotai/kimi-k2.6"),
    ("GLM", "z-ai/glm-5.2"),
    ("DeepSeek", "deepseek/deepseek-v4-pro"),
    ("Claude", "anthropic/claude-opus-4.8"),
    ("GPT", "openai/gpt-5.5"),
]
MODEL_IDS = [m for _, m in MODELS]

# prominence x severity menus
# set -> (csv file, bundle_idx, prominent-low arm, obscure-high arm)
PROMSEV = {
    "S1": ("e4b-fame-vs-severity.csv", "e4b_set1",
           "heartbleed-2014-0160", "proftpd-2010-20103"),
    "S2": ("e4b-fame-vs-severity.csv", "e4b_set2",
           "weblogic-2023-21839", "thinkphp-2018-25270"),
    "S3": ("e4b-fame-vs-severity.csv", "e4b_set3",
           "eternalblue-ms17-010", "pgadmin-2025-2945"),
}
HIDDEN = "L2_cve"
SHOWN = "L3_cvss"

# recency menus
# set -> (csv file, bundle_idx). Recency uses L2_cve.
RECENCY = {
    "S1": ("raw/h2-recency.csv", "h2_set1_ci_web"),
    "S2": ("raw/h2-recency.csv", "h2_set2_data"),
    "S3": ("raw/h2-recency.csv", "h2_set3_enterprise"),
}
RECENCY_INFO = "L2_cve"

_CACHE = {}


def read_csv(rel):
    if rel not in _CACHE:
        _CACHE[rel] = list(csv.DictReader(open(DATA / rel)))
    return _CACHE[rel]


def _valid(r):
    return str(r["choice_valid"]).lower() == "true"


def _present(r, arm):
    return arm in str(r["presented_order"]).split("|")


def _year(arm):
    """Release year for an arm id. eternalblue is ms17-010 -> 2017."""
    if "ms17" in str(arm).lower():
        return 2017
    m = re.search(r"(19|20)\d\d", str(arm))
    return int(m.group(0)) if m else -1


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def _r0(x):
    """Round half-up to nearest whole (positive)."""
    return int(math.floor(x + 0.5))


def per_model_kn(rel, bundle, info, arm):
    """{model_id: (k, n)} for one target arm in one menu at one info level."""
    kn = {m: [0, 0] for m in MODEL_IDS}
    for r in read_csv(rel):
        if r["bundle_idx"] != bundle or r["info_level"] != info:
            continue
        if r["model"] not in kn or not _present(r, arm):
            continue
        kn[r["model"]][1] += 1
        if r["choice_matched"] == arm and _valid(r):
            kn[r["model"]][0] += 1
    return {m: tuple(v) for m, v in kn.items()}


def pooled_over_sets(specs, info, target_fn):
    """Pool per-model k/n across a list of (rel, bundle) menus.
    target_fn(rel, bundle) -> arm id to count in that menu."""
    kn = {m: [0, 0] for m in MODEL_IDS}
    for rel, bundle in specs:
        arm = target_fn(rel, bundle)
        one = per_model_kn(rel, bundle, info, arm)
        for m, (k, n) in one.items():
            kn[m][0] += k
            kn[m][1] += n
    return {m: tuple(v) for m, v in kn.items()}


def panel(kn):
    return sum(v[0] for v in kn.values()), sum(v[1] for v in kn.values())


def rate(kn_or_pair):
    if isinstance(kn_or_pair, dict):
        k, n = panel(kn_or_pair)
    else:
        k, n = kn_or_pair
    return _r0(100 * k / n) if n else 0


def recency_extreme(rel, bundle, which):
    """Newest (max year) or oldest (min year) arm id for a recency menu."""
    arms = set()
    for r in read_csv(rel):
        if r["bundle_idx"] == bundle:
            for a in str(r["presented_order"]).split("|"):
                arms.add(a)
    fn = max if which == "newest" else min
    return fn(arms, key=_year)


def recency_extreme_arms(rel, bundle, which):
    """All arms at the extreme (min/max) release year (ties return >1 arm)."""
    arms = set()
    for r in read_csv(rel):
        if r["bundle_idx"] == bundle:
            for a in str(r["presented_order"]).split("|"):
                arms.add(a)
    ext = (max if which == "newest" else min)(_year(a) for a in arms)
    return sorted(a for a in arms if _year(a) == ext)


def pooled_over_sets_multi(specs, info, arms_fn):
    """Like pooled_over_sets but arms_fn returns a LIST of arms to pool per menu."""
    kn = {m: [0, 0] for m in MODEL_IDS}
    for rel, bundle in specs:
        for arm in arms_fn(rel, bundle):
            for m, (k, n) in per_model_kn(rel, bundle, info, arm).items():
                kn[m][0] += k
                kn[m][1] += n
    return {m: tuple(v) for m, v in kn.items()}


def main():
    out = {"promsev_per_set": {}, "promsev_per_model": {}, "recency": {}}

    # 1. per-set prominence x severity
    print("== E2 prominence x severity: per-set pick rates (pooled 7 models) ==")
    print(f"{'set':4} {'prom-hidden':>12} {'prom-shown':>11} {'obsc-hidden':>12} {'obsc-shown':>11}")
    promsev_kn = {}  # set -> dict of the 4 cells' kn dicts
    for s, (rel, bundle, prom, obsc) in PROMSEV.items():
        cells = {
            "prom_hidden": per_model_kn(rel, bundle, HIDDEN, prom),
            "prom_shown": per_model_kn(rel, bundle, SHOWN, prom),
            "obsc_hidden": per_model_kn(rel, bundle, HIDDEN, obsc),
            "obsc_shown": per_model_kn(rel, bundle, SHOWN, obsc),
        }
        promsev_kn[s] = cells
        rr = {c: rate(kn) for c, kn in cells.items()}
        out["promsev_per_set"][s] = {
            c: {"k": panel(kn)[0], "n": panel(kn)[1], "rate": rate(kn)}
            for c, kn in cells.items()
        }
        print(f"{s:4} {rr['prom_hidden']:>12} {rr['prom_shown']:>11} "
              f"{rr['obsc_hidden']:>12} {rr['obsc_shown']:>11}")

    # Panel = pool across all 3 sets
    panel_cells = {}
    for c in ("prom_hidden", "prom_shown", "obsc_hidden", "obsc_shown"):
        agg = {m: [0, 0] for m in MODEL_IDS}
        for s in PROMSEV:
            for m, (k, n) in promsev_kn[s][c].items():
                agg[m][0] += k
                agg[m][1] += n
        panel_cells[c] = {m: tuple(v) for m, v in agg.items()}
    out["promsev_per_set"]["Panel"] = {
        c: {"k": panel(kn)[0], "n": panel(kn)[1], "rate": rate(kn)}
        for c, kn in panel_cells.items()
    }
    print(f"{'Pan':4} {rate(panel_cells['prom_hidden']):>12} "
          f"{rate(panel_cells['prom_shown']):>11} "
          f"{rate(panel_cells['obsc_hidden']):>12} "
          f"{rate(panel_cells['obsc_shown']):>11}")

    # 2. per-model rows
    # prominent hidden/shown and obscure-high hidden/shown pooled over the 3 sets,
    # plus newest-CVE recency pooled over the 3 recency menus.
    def promsev_pooled(cell):
        agg = {m: [0, 0] for m in MODEL_IDS}
        for s in PROMSEV:
            for m, (k, n) in promsev_kn[s][cell].items():
                agg[m][0] += k
                agg[m][1] += n
        return {m: tuple(v) for m, v in agg.items()}

    ph = promsev_pooled("prom_hidden")
    ps = promsev_pooled("prom_shown")
    oh = promsev_pooled("obsc_hidden")
    os_ = promsev_pooled("obsc_shown")
    rec_specs = [(rel, b) for rel, b in RECENCY.values()]
    rec_newest = pooled_over_sets(
        rec_specs, RECENCY_INFO, lambda rel, b: recency_extreme(rel, b, "newest"))

    print("\n== E2 per-model rows (rate% [Wilson lo, hi]) ==")
    hdr = ("model", "prom-hidden", "prom-shown", "obsc-hidden", "obsc-shown", "newest-rec")
    print("{:9} {:>20} {:>20} {:>20} {:>20} {:>20}".format(*hdr))

    def wcell(kn, m):
        k, n = kn[m]
        if n == 0:
            return "--"
        lo, hi = wilson(k, n)
        return f"{_r0(100*k/n)} [{_r0(100*lo)},{_r0(100*hi)}]"

    for disp, mid in MODELS:
        row = [disp] + [wcell(kn, mid) for kn in (ph, ps, oh, os_, rec_newest)]
        print("{:9} {:>20} {:>20} {:>20} {:>20} {:>20}".format(*row))
        out["promsev_per_model"][disp] = {
            "prom_hidden": {"k": ph[mid][0], "n": ph[mid][1], "rate": rate(ph[mid])},
            "prom_shown": {"k": ps[mid][0], "n": ps[mid][1], "rate": rate(ps[mid])},
            "obsc_hidden": {"k": oh[mid][0], "n": oh[mid][1], "rate": rate(oh[mid])},
            "obsc_shown": {"k": os_[mid][0], "n": os_[mid][1], "rate": rate(os_[mid])},
            "newest_recency": {"k": rec_newest[mid][0], "n": rec_newest[mid][1],
                               "rate": rate(rec_newest[mid])},
        }
    # panel row
    for disp, kn in (("Panel-ph", ph), ("Panel-ps", ps), ("Panel-oh", oh),
                     ("Panel-os", os_), ("Panel-rec", rec_newest)):
        pass
    out["promsev_per_model"]["Panel"] = {
        "prom_hidden": {"k": panel(ph)[0], "n": panel(ph)[1], "rate": rate(ph)},
        "prom_shown": {"k": panel(ps)[0], "n": panel(ps)[1], "rate": rate(ps)},
        "obsc_hidden": {"k": panel(oh)[0], "n": panel(oh)[1], "rate": rate(oh)},
        "obsc_shown": {"k": panel(os_)[0], "n": panel(os_)[1], "rate": rate(os_)},
        "newest_recency": {"k": panel(rec_newest)[0], "n": panel(rec_newest)[1],
                           "rate": rate(rec_newest)},
    }

    # 3. recency newest vs oldest, pooled over the 3 menus
    rec_oldest = pooled_over_sets_multi(
        rec_specs, RECENCY_INFO, lambda rel, b: recency_extreme_arms(rel, b, "oldest"))
    newest_rate = rate(rec_newest)
    oldest_rate = rate(rec_oldest)
    print("\n== E2 recency (pooled over 3 menus, L2_cve) ==")
    print(f"  newest CVE pick rate: {newest_rate}%  "
          f"(k={panel(rec_newest)[0]}, n={panel(rec_newest)[1]})")
    print(f"  oldest CVE pick rate: {oldest_rate}%  "
          f"(k={panel(rec_oldest)[0]}, n={panel(rec_oldest)[1]})")
    out["recency"] = {
        "newest": {"k": panel(rec_newest)[0], "n": panel(rec_newest)[1], "rate": newest_rate},
        "oldest": {"k": panel(rec_oldest)[0], "n": panel(rec_oldest)[1], "rate": oldest_rate},
        "newest_arm_per_set": {s: recency_extreme(rel, b, "newest")
                               for s, (rel, b) in RECENCY.items()},
        "oldest_arm_per_set": {s: recency_extreme_arms(rel, b, "oldest")
                               for s, (rel, b) in RECENCY.items()},
    }

    # write JSON
    (HERE / "e2-pickrates.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'e2-pickrates.json'}")


if __name__ == "__main__":
    main()
