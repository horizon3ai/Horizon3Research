#!/usr/bin/env python3
"""Appendix table: per-model target-arm pick rate with Wilson 95% CI (baseline vs treatment)
for E1-E6, as "rate% [lo, hi]" with a pooled Panel row. Prints LaTeX to stdout.
"""
import csv
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RQ1 = HERE.parent
CVE = Path(__file__).resolve().parents[1]

# display, q1-metrics id (no vendor prefix), prefixed id (used in most raw files)
MODELS = [
    ("Gemini", "gemini-3.1-pro-preview", "google/gemini-3.1-pro-preview"),
    ("Grok", "grok-4.3", "x-ai/grok-4.3"),
    ("Kimi", "kimi-k2.6", "moonshotai/kimi-k2.6"),
    ("GLM", "glm-5.2", "z-ai/glm-5.2"),
    ("DeepSeek", "deepseek-v4-pro", "deepseek/deepseek-v4-pro"),
    ("Claude", "claude-opus-4.8", "anthropic/claude-opus-4.8"),
    ("GPT-5.5", "gpt-5.5", "openai/gpt-5.5"),
]

# E1 recognition: prominent (pos) arm per Q1 set
FAM_OBS = {"h1_set1_ftp": "vsftpd-2011-2523-flagged",
           "h1_set2_spring": "spring4shell-2022-22965",
           "h1_set3_enterprise": "confluence-2023-22527"}
E1_FILE = "q1-recognition/data/h1-recognition-sets123.csv"

# E3 severity: top-CVSS (10.0/9.8) arm per set
E3_TOP = {"h3d_set1": "kibana-2019-7609", "h3d_set2": "jenkins-2018-1000861",
          "h3d_set3": "craftcms-2025-32432"}
E3_FILE = "q3-severity-shown/data/h3-severity.csv"

# E4 scan override: decomposition ladder, decoy arm per rung
# Ladder: decoy at each rung is nexus-2024-4956-l{rung}; denominator = present trials, numerator = valid decoy picks.
E4_FILE = "q4-scan-override/ladder/data/h4-ladder-decomp.csv"
E4_RUNGS = [("l1normal", "Normal"), ("l2vuln", "+Finding"),
            ("l3cve", "+CVE"), ("l4exploit", "+Exploit")]
E4_M = ("ladder_m1", "ladder_m2", "ladder_m3")

# E5 fabrication: contradiction arm vs matched-consistent control arm
Q5_CONTRA = [
    ("q5-fabrication/data/h8-fab-scanonly.csv", "h8so_test", "elasticsearch-scanonly-contradiction"),
    ("q5-fabrication/data/h8-contradiction-full.csv", "h8c_nexus", "nexus-2024-4956-contradiction"),
    ("q5-fabrication/data/h8-contradiction-full.csv", "h8c_grafana", "grafana-2021-43798-contradiction"),
]
Q5_CONSISTENT = [
    ("q5-fabrication/data/h8-fab-scanonly.csv", "h8so_control", "elasticsearch-2015-1427"),
    ("q5-fabrication/data/h8-contradiction-control-full.csv", "h8cc_nexus", "nexus-2024-4956-consistent"),
    ("q5-fabrication/data/h8-contradiction-control-full.csv", "h8cc_grafana", "grafana-2021-43798-consistent"),
]

# E6 backfire: flagged arm (ff + of) per Q6 set
Q6_REG = {
    "h5x_set1": frozenset({"citrix-2019-19781", "ofbiz-2024-45195-flagged"}),
    "h5x_set2": frozenset({"samba-2017-7494", "thinkphp-2018-25270-flagged"}),
    "h5x_set3": frozenset({"vsftpd-2011-2523-flagged", "proftpd-2010-20103"}),
}
E6_BASE = "q6-backfire/data/h5-backfire.csv"          # blind / warned / told_all_real
E6_CAUTION = "q6-backfire/caution/data/h5-backfire-caution.csv"  # caution (placebo)

# Column layout: (experiment label, [(sub-label, kn-builder), ...]); built at bottom once helpers are defined.


def wilson(k, n, z=1.96):
    """Wilson 95% CI for a proportion (good at small n). Returns (lo, hi) in [0,1]."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def _valid(r):
    return str(r["choice_valid"]).lower() == "true"


def _present(r, armset):
    ids = str(r["presented_order"]).split("|")
    return any(a in ids for a in armset)


_CACHE = {}


def read_csv(rel):
    if rel not in _CACHE:
        _CACHE[rel] = list(csv.DictReader(open(RQ1 / rel)))
    return _CACHE[rel]


def pooled_kn(specs, info, prompt=None):
    """Attraction pooled k/n over (rel_path, bundle, armset) specs; armset is one arm-id or a set
    (any counts as a target). n = trials where a target arm is present, k = valid target picks."""
    kn = {pid: [0, 0] for _, _, pid in MODELS}
    for rel, bundle, arm in specs:
        armset = frozenset({arm}) if isinstance(arm, str) else frozenset(arm)
        for r in read_csv(rel):
            if r["bundle_idx"] != bundle or r["info_level"] != info:
                continue
            if prompt is not None and r["prompt_id"] != prompt:
                continue
            if r["model"] not in kn or not _present(r, armset):
                continue
            kn[r["model"]][1] += 1
            if r["choice_matched"] in armset and _valid(r):
                kn[r["model"]][0] += 1
    return {m: tuple(v) for m, v in kn.items()}


# E1 / E3 / E4 specs (single arm per bundle)
def _one_per_bundle(mapping):
    return [(None, b, a) for b, a in mapping.items()]


def e1(info):
    specs = [(E1_FILE, b, a) for b, a in FAM_OBS.items()]
    return pooled_kn(specs, info)


def e3(info):
    specs = [(E3_FILE, b, a) for b, a in E3_TOP.items()]
    return pooled_kn(specs, info)


def e4(rung):
    """Decoy pick rate at a ladder rung, pooled over the 3 menus (m1/m2/m3)."""
    arm = f"nexus-2024-4956-{rung}"
    specs = [(E4_FILE, f"{m}_{rung}", arm) for m in E4_M]
    return pooled_kn(specs, "L3_nmap")


def e5(specs):
    return pooled_kn([(p, b, a) for p, b, a in specs], "L3_nmap")


def e6(prompt):
    src = E6_CAUTION if prompt == "caution" else E6_BASE
    specs = [(src, b, arms) for b, arms in Q6_REG.items()]
    return pooled_kn(specs, "L3_nmap", prompt=prompt)


# E2 drivers: prominence x severity pick rate + newest-CVE recency
# Prominence x severity menus (e4b): two targets per set (prominent-low, obscure-high) at CVSS hidden/shown.
# All 3 sets from e4b-fame-vs-severity.csv.
E2_PROMSEV = {  # set -> (rel_path, bundle_idx, prominent-low arm, obscure-high arm)
    "S1": ("q2-drivers/data/e4b-fame-vs-severity.csv", "e4b_set1",
           "heartbleed-2014-0160", "proftpd-2010-20103"),
    "S2": ("q2-drivers/data/e4b-fame-vs-severity.csv", "e4b_set2",
           "weblogic-2023-21839", "thinkphp-2018-25270"),
    "S3": ("q2-drivers/data/e4b-fame-vs-severity.csv", "e4b_set3",
           "eternalblue-ms17-010", "pgadmin-2025-2945"),
}
# Recency menus (h2): newest = the arm with the max release year (L2_cve, CVSS withheld).
E2_RECENCY = {  # set -> (rel_path, bundle_idx)
    "S1": ("q2-drivers/data/raw/h2-recency.csv", "h2_set1_ci_web"),
    "S2": ("q2-drivers/data/raw/h2-recency.csv", "h2_set2_data"),
    "S3": ("q2-drivers/data/raw/h2-recency.csv", "h2_set3_enterprise"),
}


def _year(arm):
    # eternalblue is ms17-010 -> 2017; otherwise take the CVE year token
    if "ms17" in str(arm).lower():
        return 2017
    m = re.search(r"(19|20)\d\d", str(arm))
    return int(m.group(0)) if m else -1


def _recency_newest(rel, bundle):
    """Newest-year arm id in one recency menu."""
    arms = set()
    for r in read_csv(rel):
        if r["bundle_idx"] == bundle:
            arms.update(str(r["presented_order"]).split("|"))
    return max(arms, key=_year)


def e2_promsev(which, info):
    """Prominence-severity pick rate over the 3 e4b menus; which = 'prom' (prominent-low) or 'obsc' (obscure-high)."""
    idx = 2 if which == "prom" else 3
    specs = [(v[0], v[1], v[idx]) for v in E2_PROMSEV.values()]
    return pooled_kn(specs, info)


def e2_recency_newest():
    """Newest-CVE pick rate (L2_cve) pooled over the 3 h2-recency menus."""
    specs = [(rel, b, _recency_newest(rel, b)) for rel, b in E2_RECENCY.values()]
    return pooled_kn(specs, "L2_cve")


def layout():
    return [
        ("E1", [("version", e1("L1_version")), ("+CVE ID", e1("L2_cve"))]),
        ("E2", [("prom-hidden", e2_promsev("prom", "L2_cve")),
                ("prom-shown", e2_promsev("prom", "L3_cvss")),
                ("obsc-hidden", e2_promsev("obsc", "L2_cve")),
                ("obsc-shown", e2_promsev("obsc", "L3_cvss")),
                ("recency", e2_recency_newest())]),
        ("E3", [("no scan", e3("L2_cve")), ("scan", e3("L3_nmap"))]),
        ("E4", [(lbl, e4(rung)) for rung, lbl in E4_RUNGS]),
        ("E5", [("contra", e5(Q5_CONTRA)), ("consistent", e5(Q5_CONSISTENT))]),
        ("E6", [("blind", e6("blind")), ("warned", e6("warned")),
                ("placebo", e6("caution")), ("reass", e6("told_all_real"))]),
    ]


def flat_cols(lay):
    """[(exp, sub, kn_dict), ...] in column order."""
    return [(exp, sub, kn) for exp, subs in lay for sub, kn in subs]


def panel(kn):
    k = sum(v[0] for v in kn.values())
    n = sum(v[1] for v in kn.values())
    return k, n


def cell(k, n):
    if n == 0:
        return "--"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def print_matrix(lay):
    cols = flat_cols(lay)
    w = 20
    top = f"{'':10}" + "".join(f"{exp + '/' + sub:>{w}}" for exp, sub, _ in cols)
    print(top)
    print("-" * len(top))
    for disp, _, pid in MODELS:
        line = f"{disp:10}" + "".join(f"{cell(*kn[pid]):>{w}}" for _, _, kn in cols)
        print(line)
    print("-" * len(top))
    line = f"{'Panel':10}" + "".join(f"{cell(*panel(kn)):>{w}}" for _, _, kn in cols)
    print(line)
    print()
    print("Panel n per column:", {f"{exp}/{sub}": panel(kn)[1] for exp, sub, kn in cols})


def build_tex(lay):
    cols = flat_cols(lay)
    ncol = len(cols)
    colspec = "l " + " ".join("c" for _ in cols)
    L = [r"\documentclass[border=6pt]{standalone}",
         r"\usepackage{booktabs}", r"\usepackage{amsmath}", r"\begin{document}",
         r"\scriptsize",
         rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    # experiment header row with multicolumns
    hdr, rules, i = [""], [], 2  # column 1 = Model
    for exp, subs in lay:
        span = len(subs)
        hdr.append(rf"\multicolumn{{{span}}}{{c}}{{{exp}}}")
        rules.append(rf"\cmidrule(lr){{{i}-{i + span - 1}}}")
        i += span
    L.append(" & ".join(hdr) + r" \\")
    L.append("".join(rules))
    # sub-label row
    subrow = ["Model"] + [sub for _, sub, _ in cols]
    L.append(" & ".join(subrow) + r" \\")
    L.append(r"\midrule")
    for disp, _, pid in MODELS:
        cells = [cell(*kn[pid]) for _, _, kn in cols]
        L.append(f"{disp} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    pcells = [rf"\textbf{{{cell(*panel(kn))}}}" for _, _, kn in cols]
    L.append(r"\textbf{Panel} & " + " & ".join(pcells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{document}"]
    return "\n".join(L)


# Human-readable row labels for the transposed (sideways) layout.
ROW_LABELS = {
    ("E1", "version"): "version", ("E1", "+CVE ID"): "+CVE ID",
    ("E2", "prom-hidden"): "prominent-low, CVSS hidden",
    ("E2", "prom-shown"): "prominent-low, CVSS shown",
    ("E2", "obsc-hidden"): "obscure-high, CVSS hidden",
    ("E2", "obsc-shown"): "obscure-high, CVSS shown",
    ("E2", "recency"): "recency (newest CVE)",
    ("E3", "no scan"): "no nmap scan", ("E3", "scan"): "nmap scan",
    ("E4", "Normal"): "Normal", ("E4", "+Finding"): "+Finding",
    ("E4", "+CVE"): "+CVE", ("E4", "+Exploit"): "+Exploit",
    ("E5", "contra"): "contradictory", ("E5", "consistent"): "consistent",
    ("E6", "blind"): "blind", ("E6", "warned"): "warned",
    ("E6", "placebo"): "placebo", ("E6", "reass"): "reassurance",
}
EXP_TITLES = {
    "E1": "E1: recognition", "E2": "E2: drivers", "E3": "E3: severity shown",
    "E4": "E4: scan-override ladder", "E5": "E5: fabrication", "E6": "E6: backfire",
}
# Paper-name group titles used by the compact layout.
EXP_TITLES_PAPER = {
    "E1": "E1: recognition", "E2": "E2: drivers", "E3": "E3: severity shown",
    "E4": "E4: planted finding", "E5": "E5: contradiction", "E6": "E6: deception warning",
}


def _r0(x):
    """Round half-up to the nearest whole number (positive values)."""
    return int(math.floor(x + 0.5))


def icell(k, n):
    """Integer-rounded cell 'rate [lo, hi]' (whole percents, round half-up)."""
    if n == 0:
        return "--"
    lo, hi = wilson(k, n)
    return f"{_r0(100 * k / n)} [{_r0(100 * lo)}, {_r0(100 * hi)}]"


def build_tex_sideways(lay, compact=False):
    """Transposed landscape layout (sidewaystable): rows = conditions grouped by experiment, columns =
    7 models + Panel. compact=True gives integer cells at scriptsize so all columns fit without resizebox."""
    disp_models = [d for d, _, _ in MODELS]
    ncol = 1 + len(disp_models) + 1  # label + 7 models + Panel
    colspec = "l " + "c" * len(disp_models) + " c"
    fmt = icell if compact else cell
    titles = EXP_TITLES_PAPER if compact else EXP_TITLES
    unit = "rate\\% [lo, hi], whole percents" if compact else "rate\\% [lo, hi]"
    L = [r"\begin{sidewaystable}[htbp]",
         r"\centering",
         r"\scriptsize" if compact else r"\footnotesize"]
    if compact:
        L.append(r"\setlength{\tabcolsep}{3pt}")
    L += [rf"\caption{{Per-model target-arm pick rate with Wilson 95\% confidence "
          rf"intervals across baseline and treatment conditions for all six "
          rf"experiments. Each cell is {unit}; the attraction convention is used "
          rf"(denominator = present trials, numerator = valid target picks). The "
          rf"Panel column pools counts across all seven models.}}",
          r"\label{tab:wilson-appendix}",
          rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
          "Condition & " + " & ".join(disp_models) + r" & \textbf{Panel} \\",
          r"\midrule"]
    for gi, (exp, subs) in enumerate(lay):
        if gi:
            # single full-width rule to separate one experiment block from the next
            L.append(r"\midrule" if compact else r"\addlinespace")
        L.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textbf{{{titles[exp]}}}}} \\")
        if not compact:
            L.append(rf"\cmidrule(l){{1-{ncol}}}")
        for sub, kn in subs:
            label = ROW_LABELS.get((exp, sub), sub)
            cells = [fmt(*kn[pid]) for _, _, pid in MODELS]
            pcell = rf"\textbf{{{fmt(*panel(kn))}}}"
            L.append(f"{label} & " + " & ".join(cells) + f" & {pcell}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{sidewaystable}"]
    return "\n".join(L)


if __name__ == "__main__":
    lay = layout()
    print("== Per-model pick rate, Wilson 95%% CI (rate%% [lo, hi]) ==\n")
    print_matrix(lay)

    print("\n== E2 newest-year arm per recency menu (L2_cve target) ==")
    for s, (rel, b) in E2_RECENCY.items():
        print(f"  {s} ({b}): {_recency_newest(rel, b)}")

    print("\n== LaTeX (wide, grouped by experiment) ==\n")
    print(build_tex(lay))

    print("\n== LaTeX (transposed, sidewaystable) ==\n")
    print(build_tex_sideways(lay))

    print("\n== LaTeX (transposed, sidewaystable, COMPACT integer cells) ==\n")
    print(build_tex_sideways(lay, compact=True))

