"""Statistical analyses for the Honeyquest replication.

Implements:

  - Aspect B1 (§3.2.2): Binomial test on whether deceptive lines are marked
    first more often than chance, per CDT and overall.

  - Aspect B2 (§3.2.3): McNemar / one-sided Binomial test on whether the
    presence of deceptive elements diverts attention away from real risks,
    per CDT and overall. Includes Relative Risk RR = (β+δ)/(γ+δ).

  - Tables 10/11: Top individual query lines ranked by exploit-mark and
    trap-mark count across the panel.

Each function returns a serializable dict so the build pipeline can dump
into metrics.json and render into HTML.

References
----------
Kahlhofer, Achleitner, Rass, Mayrhofer.
  "Honeyquest: Rapidly Measuring the Enticingness of Cyber Deception
  Techniques with Code-based Questionnaires", RAID 2024.
  Tables 5, 6, 8, 9, 10, 11.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import binomtest


# ---------- B1: anchoring on bait (Binomial test) ----------

def b1_binomial(responses, gt: dict[str, dict],
                qid_filter: set[str] | None = None) -> dict[str, Any]:
    """Aspect B1, computed the Honeyquest way (Kahlhofer et al.,
    processing.py:_get_matching_ranks_for_hack_marks).

    A deceptive-query response counts toward d_B only when it has BOTH an
    exploit mark on a deceptive line AND one on a non-deceptive line. d'_B is
    the count where the first exploit mark landed on a deceptive line. Test
    H0: p = 0.5 against alternative p > 0.5, one-sided Binomial.

    Returns {d_B, d'_B, rate, p_value, significant_05}.
    """
    d_B = 0
    d_prime_B = 0
    for r in responses:
        if r.label != "deceptive":
            continue
        if qid_filter is not None and r.qid not in qid_filter:
            continue
        dec_lines = set(gt[r.qid].get("deceptive_lines") or [])
        has_dcpt = any(m in dec_lines for m in r.exploit)
        has_non_dcpt = any(m not in dec_lines for m in r.exploit)
        if not (has_dcpt and has_non_dcpt):
            continue
        d_B += 1
        if r.exploit[0] in dec_lines:
            d_prime_B += 1
    if d_B == 0:
        return {"d_B": 0, "d_prime_B": 0, "rate": None,
                "p_value": None, "significant_05": False}
    result = binomtest(k=d_prime_B, n=d_B, p=0.5, alternative="greater")
    return {
        "d_B": int(d_B),
        "d_prime_B": int(d_prime_B),
        "rate": d_prime_B / d_B,
        "p_value": float(result.pvalue),
        "significant_05": bool(result.pvalue < 0.05),
    }


def b1_per_cdt(responses, gt: dict[str, dict]) -> dict[str, dict]:
    """B1 broken out per applied_honeywire (CDT) plus an overall row."""
    # Group qids by CDT
    qids_per_cdt: dict[str, set[str]] = defaultdict(set)
    for q in gt.values():
        if q.get("label") == "deceptive":
            cdt = q.get("applied_honeywire") or "unknown"
            qids_per_cdt[cdt].add(q["qid"])

    out = {"__overall__": b1_binomial(responses, gt)}
    for cdt, qids in qids_per_cdt.items():
        out[cdt] = b1_binomial(responses, gt, qid_filter=qids)
    return out


# ---------- B2: attention diversion (McNemar) ----------

@dataclass
class B2Pair:
    qR: str             # risky qid
    qD: str             # deceptive qid (qR + injected CDT)
    cdt: str            # the applied_honeywire id (which CDT was injected)


def load_b2_pairs(honeyquest_queries_csv: Path) -> list[B2Pair]:
    """Load all (qR, qD) pairings from the upstream honeyquest_queries.csv.

    Only deceptive queries derived from a risky original (original_risky=True)
    qualify. The risky original must also exist in the dataset."""
    df = pd.read_csv(honeyquest_queries_csv)
    risky_qids = set(df[df["query_label"] == "risky"]["qid"])
    pairs: list[B2Pair] = []
    for _, row in df[df["query_label"] == "deceptive"].iterrows():
        if not bool(row.get("original_risky", False)):
            continue
        qR = row["original_query"]
        if qR not in risky_qids:
            continue
        pairs.append(B2Pair(qR=qR, qD=row["qid"], cdt=row["applied_honeywire"]))
    return pairs


def _matches_risky(r, risky_lines: set[int]) -> bool:
    """Did this response place any exploit mark on a risky line?"""
    return bool(set(r.exploit) & risky_lines) if risky_lines else False


def b2_mcnemar(responses, gt: dict[str, dict], pairs: list[B2Pair],
               pair_filter: list[B2Pair] | None = None) -> dict[str, Any]:
    """Aspect B2 per the paper §3.2.3 + Table 8.

    Build the 2×2 contingency over (matched in qR, matched in qD) where
    "matched" = exploit mark on the risky line. Per-paper:
        α = ✗qR, ✗qD     β = ✓qR, ✗qD
        γ = ✗qR, ✓qD     δ = ✓qR, ✓qD
    H0: no effect, β = γ. One-sided alternative: β > γ (presence of
    deception reduces risk-marking → more "match in qR but not qD" than the
    reverse). McNemar via Binomial on discordant pairs.

    Relative Risk RR = Pr(match qD) / Pr(match qR) = (γ+δ)/(β+δ).
    The paper reports ΔRR = RR - 1 (negative ΔRR = deception reduces marking).

    Treats each (model, pair, trial) as one paired observation. With T trials
    per model per query, we get T paired obs per (model, pair).
    """
    pairs_to_use = pair_filter if pair_filter is not None else pairs

    # Index responses by qid → list of trial responses
    by_qid: dict[str, list] = defaultdict(list)
    for r in responses:
        by_qid[r.qid].append(r)

    alpha = beta = gamma = delta = 0
    for p in pairs_to_use:
        rR_list = by_qid.get(p.qR, [])
        rD_list = by_qid.get(p.qD, [])
        # Pair by trial index (we assume trial-aligned ordering, which is true
        # within a single submission file)
        risky_lines = set(gt[p.qR].get("risky_lines") or [])
        for rR, rD in zip(rR_list, rD_list):
            m_qR = _matches_risky(rR, risky_lines)
            m_qD = _matches_risky(rD, risky_lines)
            if not m_qR and not m_qD: alpha += 1
            elif m_qR and not m_qD:   beta  += 1
            elif not m_qR and m_qD:   gamma += 1
            else:                     delta += 1

    n_pairs = alpha + beta + gamma + delta
    if n_pairs == 0:
        return {"alpha": 0, "beta": 0, "gamma": 0, "delta": 0,
                "n_pairs": 0, "RR": None, "delta_RR": None,
                "p_value": None, "significant_05": False, "powered": False}

    discordant = beta + gamma
    pr_qR = (beta + delta) / n_pairs
    pr_qD = (gamma + delta) / n_pairs
    rr = (pr_qD / pr_qR) if pr_qR > 0 else None
    delta_rr = (rr - 1.0) if rr is not None else None

    if discordant == 0:
        p_value = 1.0
        significant = False
    else:
        # One-sided: H1 = β > γ (deception reduces marking)
        result = binomtest(k=beta, n=discordant, p=0.5, alternative="greater")
        p_value = float(result.pvalue)
        significant = p_value < 0.05

    # Per Table 8 footnote: rule-of-thumb is each expected cell ≥ 5
    expected = n_pairs / 4
    powered = (expected >= 5)

    return {
        "alpha": int(alpha), "beta": int(beta),
        "gamma": int(gamma), "delta": int(delta),
        "n_pairs": int(n_pairs),
        "pr_match_qR": pr_qR,
        "pr_match_qD": pr_qD,
        "RR": rr,
        "delta_RR_pct": (delta_rr * 100) if delta_rr is not None else None,
        "p_value": p_value,
        "significant_05": significant,
        "powered": powered,
    }


def b2_per_cdt(responses, gt: dict[str, dict],
               pairs: list[B2Pair]) -> dict[str, dict]:
    """B2 per CDT plus an overall row."""
    by_cdt: dict[str, list[B2Pair]] = defaultdict(list)
    for p in pairs:
        by_cdt[p.cdt].append(p)
    out = {"__overall__": b2_mcnemar(responses, gt, pairs)}
    for cdt, ps in by_cdt.items():
        out[cdt] = b2_mcnemar(responses, gt, pairs, pair_filter=ps)
    return out


# ---------- Appendix A: confusion-matrix recast ----------
#
# Per Honeyquest paper Appendix A, Table 12+13. Each (response, query) row
# yields a TN/FP/FN/TP label based on whether the user's marks intersect the
# annotated line set L (= L_D for deceptive, L_R for risky). The paper's
# convention (paraphrasing Equation 1 in the paper):
#
#                A = ∅            A ≠ ∅
#   q ∈ Q_N      TN               FP
#   q ∉ Q_N      FN  (L ∩ A = ∅)  TP  (L ∩ A ≠ ∅)
#
# We compute this separately for deceptive (annotated lines = L_D, exploit
# marks) and risky (annotated lines = L_R, exploit marks).


def _conf_cells(responses, gt: dict[str, dict],
                target_label: str, line_field: str) -> dict[str, int]:
    """Confusion-matrix counts under the paper's Equation-1 convention.

    The paper compares answers to NEUTRAL queries (which never carry line
    annotations) against answers to deceptive/risky queries (which do).
    For 'target_label="deceptive"': TN/FP come from neutral responses,
    FN/TP come from deceptive responses based on whether exploit marks
    intersect L_D. Mirror logic for target_label="risky".
    """
    tn = fp = fn = tp = 0
    for r in responses:
        e = set(r.exploit)
        if r.label == "neutral":
            if not e: tn += 1
            else:     fp += 1
        elif r.label == target_label:
            g = gt[r.qid]
            annot = set(g.get(line_field) or [])
            if e & annot: tp += 1
            else:         fn += 1
        # other label classes don't participate in this confusion matrix
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def _conf_metrics(c: dict[str, int]) -> dict[str, Any]:
    tn, fp, fn, tp = c["TN"], c["FP"], c["FN"], c["TP"]
    n = tn + fp + fn + tp
    if n == 0:
        return {**c, "ACC": None, "PPV": None, "TPR": None, "FPR": None}
    acc = (tn + tp) / n
    ppv = tp / (tp + fp) if (tp + fp) > 0 else None
    tpr = tp / (tp + fn) if (tp + fn) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    return {**c, "ACC": acc, "PPV": ppv, "TPR": tpr, "FPR": fpr}


def confusion_matrices(responses_by_model: dict[str, list],
                       gt: dict[str, dict]) -> dict[str, dict]:
    """Per-model confusion matrices for deceptive and risky queries
    (paper Table 13 reproduction). Plus a panel-aggregate row.

    The __panel__ aggregate EXCLUDES the __human__ alias so the panel
    statistic remains an LLM-only metric. Humans, when present, are
    surfaced under their own __human__ key.
    """
    out: dict[str, dict] = {}
    for alias, rs in responses_by_model.items():
        out[alias] = {
            "deceptive": _conf_metrics(_conf_cells(rs, gt, "deceptive", "deceptive_lines")),
            "risky":     _conf_metrics(_conf_cells(rs, gt, "risky",     "risky_lines")),
        }
    panel = [r for alias, rs in responses_by_model.items()
             if alias != "__human__" for r in rs]
    out["__panel__"] = {
        "deceptive": _conf_metrics(_conf_cells(panel, gt, "deceptive", "deceptive_lines")),
        "risky":     _conf_metrics(_conf_cells(panel, gt, "risky",     "risky_lines")),
    }
    return out


# ---------- Appendix B: five-way intersection criteria (A1-A5) ----------
#
# Per the paper, given marks A and annotation set L, classify into
#   A1: A = L exactly
#   A2: A ≠ ∅ ∧ A ⊆ L         (marked only some in L)
#   A3: A ⊈ L ∧ L ∩ A ≠ ∅     (marks overlap L, also others)
#   A4: A ≠ ∅ ∧ L ∩ A = ∅     (marked, no overlap)
#   A5: A = ∅                  (no marks)
# The paper's default match = A1 ∨ A2 ∨ A3. We also report stricter
# criteria: selective (A1 ∨ A2) and exact (A1 only).

def _a_classify(A: set[int], L: set[int]) -> str:
    if not A: return "A5"
    if A == L: return "A1"
    if A <= L: return "A2"
    if A & L:  return "A3"
    return "A4"


def strict_match_variants(responses, gt: dict[str, dict],
                          label_filter: str, line_field: str) -> dict[str, Any]:
    """Recompute fall-for-trap (or risk-found) under three matching criteria.

    Returns:
      counts: per-A-class counts (A1..A5)
      loose: # matches under A1∨A2∨A3 (paper default), as count + rate
      selective: # matches under A1∨A2, as count + rate
      exact: # matches under A1 only, as count + rate
    """
    counts: dict[str, int] = {"A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0}
    for r in responses:
        if r.label != label_filter:
            continue
        annot = set(gt[r.qid].get(line_field) or [])
        if not annot:
            continue
        e = set(r.exploit)
        counts[_a_classify(e, annot)] += 1
    total = sum(counts.values())
    if total == 0:
        return {"counts": counts, "total": 0,
                "loose": (0, None), "selective": (0, None), "exact": (0, None)}
    loose_n     = counts["A1"] + counts["A2"] + counts["A3"]
    selective_n = counts["A1"] + counts["A2"]
    exact_n     = counts["A1"]
    return {
        "counts": counts, "total": total,
        "loose":     {"matches": loose_n,     "rate": loose_n / total},
        "selective": {"matches": selective_n, "rate": selective_n / total},
        "exact":     {"matches": exact_n,     "rate": exact_n / total},
    }


def strict_match_per_model(responses_by_model: dict[str, list],
                           gt: dict[str, dict],
                           label_filter: str = "deceptive",
                           line_field: str = "deceptive_lines") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for alias, rs in responses_by_model.items():
        out[alias] = strict_match_variants(rs, gt, label_filter, line_field)
    return out


# ---------- Appendix C: canonical paper buckets ----------
#
# Per Honeyquest paper Appendix C, defines the EXACT per-query-class
# bucket counts the paper Tables 5/6/9 use.
#
#   NEUTRAL queries (q ∈ Q_N): no annotations.
#     nEx = answers with ONLY exploit marks (placed exploit on neutral lines)
#     nTr = answers with ONLY trap marks
#     n∧  = answers with both exploit AND trap marks
#     n∅  = answers with NO marks
#
#   DECEPTIVE queries (q ∈ Q_D):
#     dEx = exploit mark intersects L_D (fell for trap)
#     dTr = trap mark intersects L_D, and not dEx (trap detected)
#     d∧  = answers placed some marks, but none intersect L_D
#     d∅  = no marks
#
#   RISKY queries (q ∈ Q_R):
#     rEx = exploit mark intersects L_R (risk detected)
#     rTr = trap mark intersects L_R, and not rEx (risk mistaken for trap)
#     r∧  = answers placed some marks, but none intersect L_R
#     r∅  = no marks

def paper_buckets_per_model(responses_by_model: dict[str, list],
                            gt: dict[str, dict]) -> dict[str, dict]:
    """Canonical paper bucket counts per model."""
    out: dict[str, dict] = {}
    for alias, rs in responses_by_model.items():
        # Neutral
        nEx = nTr = nA = nO = 0
        # Deceptive
        dEx = dTr = dA = dO = 0
        # Risky
        rEx = rTr = rA = rO = 0
        for r in rs:
            e = set(r.exploit); t = set(r.trap)
            g = gt[r.qid]
            if r.label == "neutral":
                if not e and not t: nO += 1
                elif e and not t:   nEx += 1
                elif t and not e:   nTr += 1
                else:               nA += 1
            elif r.label == "deceptive":
                dec = set(g.get("deceptive_lines") or [])
                if not e and not t:    dO += 1
                elif e & dec:          dEx += 1
                elif t & dec:          dTr += 1
                else:                  dA += 1
            elif r.label == "risky":
                rsk = set(g.get("risky_lines") or [])
                if not e and not t:    rO += 1
                elif e & rsk:          rEx += 1
                elif t & rsk:          rTr += 1
                else:                  rA += 1
        out[alias] = {
            "neutral":   {"nEx": nEx, "nTr": nTr, "n_and": nA, "n_none": nO,
                          "total": nEx + nTr + nA + nO},
            "deceptive": {"dEx": dEx, "dTr": dTr, "d_and": dA, "d_none": dO,
                          "total": dEx + dTr + dA + dO},
            "risky":     {"rEx": rEx, "rTr": rTr, "r_and": rA, "r_none": rO,
                          "total": rEx + rTr + rA + rO},
        }
    return out


# ---------- HTML renderers for the appendix tables ----------

def render_confusion_table(conf: dict[str, dict],
                            cdt_display: dict[str, str] | None = None) -> str:
    """Paper Table 13 reproduction (per-model + humans + panel, deceptive + risky)."""
    import html as h

    def _row(alias: str, label: str, c: dict, row_cls: str = "") -> str:
        cls_attr = f" class='{row_cls}'" if row_cls else ""
        return (
            f"<tr{cls_attr}><th>{h.escape(label)}</th>"
            f"<td>{c['TN']}</td><td>{c['FP']}</td><td>{c['FN']}</td><td>{c['TP']}</td>"
            f"<td>{c['ACC']:.0%}</td>"
            f"<td>{(c['PPV'] or 0):.0%}</td>"
            f"<td>{(c['TPR'] or 0):.0%}</td>"
            f"<td>{(c['FPR'] or 0):.0%}</td></tr>"
        )

    rows = []
    panel = conf.get("__panel__", {})
    human = conf.get("__human__", {})

    # Panel + human + per-model, deceptive class then risky class
    for cls in ("deceptive", "risky"):
        if cls in panel:
            rows.append(_row("__panel__", f"Panel · {cls}", panel[cls], row_cls="overall"))
        if cls in human:
            rows.append(_row("__human__", f"Humans (Kahlhofer 2024) · {cls}", human[cls], row_cls="human"))
        for alias in sorted(k for k in conf if k not in ("__panel__", "__human__")):
            c = conf[alias].get(cls)
            if not c: continue
            rows.append(_row(alias, f"{alias} · {cls}", c))

    return f"""
<h3>Appendix A reproduction: confusion-matrix recast (per-model + humans + panel, deceptive + risky)</h3>
<p><em>Per Honeyquest paper Appendix A + Table 13. Equation-1 convention with the exploit-mark
interpretation: TN = neutral query, no exploit marks; FP = neutral query, any exploit marks;
FN = deceptive/risky query, exploit marks don't intersect annotated line(s);
TP = deceptive/risky query, exploit marks intersect annotated line(s) (= fell for trap if deceptive, risk-detected if risky).
ACC = accuracy; PPV = precision; TPR = recall; FPR = false-positive rate.
<strong>Humans row is the upstream Honeyquest marks dataset, re-aggregated to the same qids the agents saw.</strong></em></p>
<table class="stats">
<thead><tr><th>Subject · Query class</th>
<th>TN</th><th>FP</th><th>FN</th><th>TP</th>
<th>ACC</th><th>PPV</th><th>TPR</th><th>FPR</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
"""


def render_strict_match_table(strict_per_model: dict[str, dict]) -> str:
    """Paper Appendix B reproduction: per-model + humans fall-for-trap under
    three matching criteria (loose / selective / exact)."""
    import html as h

    def _row(alias: str, label: str, s: dict, row_cls: str = "") -> str:
        c = s["counts"]
        cls_attr = f" class='{row_cls}'" if row_cls else ""
        return (
            f"<tr{cls_attr}><th>{h.escape(label)}</th>"
            f"<td>{s['total']}</td>"
            f"<td>{c['A1']}</td><td>{c['A2']}</td><td>{c['A3']}</td><td>{c['A4']}</td><td>{c['A5']}</td>"
            f"<td>{s['loose']['rate']:.0%}</td>"
            f"<td>{s['selective']['rate']:.0%}</td>"
            f"<td>{s['exact']['rate']:.0%}</td></tr>"
        )

    rows = []
    # Humans first (reference baseline), then sorted models
    if "__human__" in strict_per_model and strict_per_model["__human__"]["total"]:
        rows.append(_row("__human__", "Humans (Kahlhofer 2024)",
                         strict_per_model["__human__"], row_cls="human"))
    for alias in sorted(k for k in strict_per_model if k != "__human__"):
        s = strict_per_model[alias]
        if s["total"] == 0:
            continue
        rows.append(_row(alias, alias, s))

    return f"""
<h3>Appendix B reproduction: five-way intersection criteria + strict-match variants (deceptive queries)</h3>
<p><em>Per Honeyquest paper Appendix B. A1 = exploit-marks equal L_D exactly; A2 = marks fully inside L_D;
A3 = marks overlap L_D AND extend beyond; A4 = marks placed, none in L_D; A5 = no marks.
Loose = A1∨A2∨A3 (paper default for fall-for-trap). Selective = A1∨A2. Exact = A1 only.
<strong>Human baseline recomputed from upstream Honeyquest marks CSV on the same qids
the agents saw, so loose ≈ paper's reported 37% fall-for-trap.</strong></em></p>
<table class="stats">
<thead><tr><th>Subject</th><th>N</th>
<th>A1</th><th>A2</th><th>A3</th><th>A4</th><th>A5</th>
<th>Loose</th><th>Selective</th><th>Exact</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
"""


def render_paper_buckets_table(buckets: dict[str, dict]) -> str:
    """Paper Appendix C reproduction with paper-style inline stacked bars
    matching the visualization in Honeyquest Tables 5, 6, 9.

    Each row gets three small horizontal stacked bars (deceptive, risky,
    neutral query classes), each showing the % distribution across the
    four canonical buckets in the paper's pink/orange/gray/white scheme.
    """
    import html as h

    PAPER_COLORS_HTML = {
        "dEx":   "#E94B8C",   # warm pink — exploit-on-annotated (fell for trap)
        "dTr":   "#F6A85A",   # orange — trap-on-annotated (trap detected)
        "d_and": "#D8D8D8",   # light gray — marks elsewhere
        "d_none":"#FFFFFF",   # white — no marks
    }

    def _seg(pct: float, color: str, label: str = "") -> str:
        """Render one segment of a stacked bar at a given % width."""
        if pct <= 0: return ""
        # Show percentage inside the segment if width is large enough
        text = f"{pct:.0%}" if pct >= 0.12 else ""
        border = "; border:1px solid #999" if color == "#FFFFFF" else ""
        return (f"<div class='pbar-seg' style='width:{pct*100:.1f}%; "
                f"background:{color}{border};' title='{label} {pct:.1%}'>{text}</div>")

    def _bar(d: dict, prefix: str) -> str:
        """Render the inline stacked bar for one query class."""
        total = d.get("total", 0)
        if total == 0:
            return "<div class='pbar empty'>n=0</div>"
        ex_key = {"d": "dEx", "r": "rEx", "n": "nEx"}[prefix]
        tr_key = {"d": "dTr", "r": "rTr", "n": "nTr"}[prefix]
        and_key = {"d": "d_and", "r": "r_and", "n": "n_and"}[prefix]
        none_key = {"d": "d_none", "r": "r_none", "n": "n_none"}[prefix]
        ex = d.get(ex_key, 0) / total
        tr = d.get(tr_key, 0) / total
        an = d.get(and_key, 0) / total
        no = d.get(none_key, 0) / total
        return (
            f"<div class='pbar'>"
            + _seg(ex, PAPER_COLORS_HTML["dEx"], f"{ex_key}")
            + _seg(tr, PAPER_COLORS_HTML["dTr"], f"{tr_key}")
            + _seg(an, PAPER_COLORS_HTML["d_and"], f"{and_key}")
            + _seg(no, PAPER_COLORS_HTML["d_none"], f"{none_key}")
            + f"</div><div class='pbar-n'>n={total}</div>"
        )

    def _row(alias: str, label: str, b: dict, row_cls: str = "") -> str:
        cls_attr = f" class='{row_cls}'" if row_cls else ""
        return (
            f"<tr{cls_attr}>"
            f"<th>{h.escape(label)}</th>"
            f"<td class='bar-cell'>{_bar(b['deceptive'], 'd')}</td>"
            f"<td class='bar-cell'>{_bar(b['risky'], 'r')}</td>"
            f"<td class='bar-cell'>{_bar(b['neutral'], 'n')}</td>"
            f"</tr>"
        )

    rows = []
    if "__human__" in buckets:
        rows.append(_row("__human__", "Humans (Kahlhofer 2024)",
                         buckets["__human__"], row_cls="human"))
    for alias in sorted(k for k in buckets if k != "__human__"):
        rows.append(_row(alias, alias, buckets[alias]))

    legend_html = (
        "<div class='pbar-legend'>"
        f"<span><span class='legend-sw' style='background:{PAPER_COLORS_HTML['dEx']}'></span> "
        "Exploit mark on annotated line (= fell for trap if deceptive, risk-detected if risky, false-positive if neutral)</span>"
        f"<span><span class='legend-sw' style='background:{PAPER_COLORS_HTML['dTr']}'></span> "
        "Trap mark on annotated line (= trap detected, risk-mistaken-for-trap, false-positive trap)</span>"
        f"<span><span class='legend-sw' style='background:{PAPER_COLORS_HTML['d_and']}'></span> "
        "Marks placed elsewhere (none on annotated line, both kinds present for neutral)</span>"
        f"<span><span class='legend-sw' style='background:{PAPER_COLORS_HTML['d_none']}; border:1px solid #999;'></span> "
        "No marks at all</span>"
        "</div>"
    )

    return f"""
<h3>Appendix C reproduction: canonical paper buckets per query class (paper Tables 5 / 6 / 9 visual style)</h3>
<p><em>Mutually exclusive 4-bucket classification of every response within each query class,
rendered as inline horizontal stacked bars in the paper's color convention. Humans row
recomputed from upstream Honeyquest marks CSV restricted to the same qids the agents saw.</em></p>
{legend_html}
<table class="pbars-table">
<thead><tr>
<th>Subject</th>
<th>Deceptive queries (d<sub>Ex</sub> / d<sub>Tr</sub> / d<sub>∧</sub> / d<sub>∅</sub>)</th>
<th>Risky queries (r<sub>Ex</sub> / r<sub>Tr</sub> / r<sub>∧</sub> / r<sub>∅</sub>)</th>
<th>Neutral queries (n<sub>Ex</sub> / n<sub>Tr</sub> / n<sub>∧</sub> / n<sub>∅</sub>)</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
"""


# ---------- Tables 10/11: top exploited / top trapped lines ----------

def top_lines(responses_by_model: dict[str, list], gt: dict[str, dict],
              queries_by_qid: dict[str, dict],
              top_exploit_n: int = 40, top_trap_n: int = 15) -> dict[str, list]:
    """Rank individual (qid, line_idx) pairs by total exploit / trap marks
    across the entire panel of models. Returns paper-style Table 10/11."""
    exploit_counts: dict[tuple[str, int], int] = defaultdict(int)
    trap_counts: dict[tuple[str, int], int] = defaultdict(int)
    for _, rs in responses_by_model.items():
        for r in rs:
            for ln in r.exploit:
                exploit_counts[(r.qid, ln)] += 1
            for ln in r.trap:
                trap_counts[(r.qid, ln)] += 1

    def expand(line_counts: dict[tuple[str, int], int], n: int) -> list[dict]:
        ranked = sorted(line_counts.items(), key=lambda kv: kv[1], reverse=True)
        rows = []
        for (qid, ln), count in ranked[:n]:
            g = gt.get(qid, {})
            q = queries_by_qid.get(qid, {})
            line_text = (q.get("lines") or [""])[ln - 1] if 1 <= ln <= len(q.get("lines") or []) else ""
            dec_lines = set(g.get("deceptive_lines") or [])
            risky_lines = set(g.get("risky_lines") or [])
            line_kind = (
                "deceptive" if ln in dec_lines
                else "risky" if ln in risky_lines
                else "neutral"
            )
            rows.append({
                "qid": qid,
                "qtype": q.get("qtype", ""),
                "line_idx": ln,
                "line_text": line_text,
                "line_kind": line_kind,
                "exploit_count": exploit_counts.get((qid, ln), 0),
                "trap_count": trap_counts.get((qid, ln), 0),
                "applied_honeywire": g.get("applied_honeywire", ""),
            })
        return rows

    return {
        "top_exploit": expand(exploit_counts, top_exploit_n),
        "top_trap": expand(trap_counts, top_trap_n),
    }


# ---------- HTML rendering for the stats tables ----------

def render_b1_table(b1_overall: dict, b1_per_model: dict[str, dict],
                    b1_per_cdt_overall: dict[str, dict],
                    cdt_display: dict[str, str] | None = None) -> str:
    """Aspect B1 per-CDT table (paper Table 9 B1 columns) + per-model overall."""
    import html as h
    cdt_display = cdt_display or {}

    # Per-model overall section
    sorted_models = sorted(b1_per_model.keys(),
                           key=lambda a: (b1_per_model[a].get("rate") or 0))
    rows_m = []
    for a in sorted_models:
        b = b1_per_model[a]
        if b["d_B"] == 0:
            cell_rate = "n/a"
            cell_p = "n/a"
        else:
            cell_rate = f"{b['rate']:.0%} ({b['d_prime_B']}/{b['d_B']})"
            cell_p = f"{b['p_value']:.4g}"
            if b['significant_05']:
                cell_p = f"<strong>{cell_p}</strong> *"
        rows_m.append(
            f"<tr><th>{h.escape(a)}</th><td>{cell_rate}</td><td>{cell_p}</td></tr>"
        )

    # Per-CDT (panel aggregate) section
    cdt_keys = sorted(k for k in b1_per_cdt_overall if k != "__overall__")
    rows_c = []
    ov = b1_per_cdt_overall.get("__overall__", {})
    if ov.get("d_B"):
        p_str = f"{ov['p_value']:.4g}"
        if ov["significant_05"]: p_str = f"<strong>{p_str}</strong> *"
        rows_c.append(
            f"<tr class='overall'><th>All Deceptive Queries (panel)</th>"
            f"<td>{ov['rate']:.0%} ({ov['d_prime_B']}/{ov['d_B']})</td>"
            f"<td>{p_str}</td></tr>"
        )
    for cdt in cdt_keys:
        b = b1_per_cdt_overall[cdt]
        if b["d_B"] == 0:
            cell_rate, cell_p = "n/a", "n/a"
        else:
            cell_rate = f"{b['rate']:.0%} ({b['d_prime_B']}/{b['d_B']})"
            cell_p = f"{b['p_value']:.4g}"
            if b["significant_05"]: cell_p = f"<strong>{cell_p}</strong> *"
        rows_c.append(
            f"<tr><th>{h.escape(cdt_display.get(cdt, cdt))}</th>"
            f"<td>{cell_rate}</td><td>{cell_p}</td></tr>"
        )

    return f"""
<h3>Aspect B1 by model (panel-aggregate, all deceptive queries)</h3>
<p><em>H<sub>0</sub>: first exploit mark on a deceptive line is at p=0.5 chance.
One-sided Binomial test, alternative p&gt;0.5. * = p &lt; 0.05.</em></p>
<table class="stats">
<thead><tr><th>Model</th><th>d'<sub>B</sub>/d<sub>B</sub></th><th>p-value</th></tr></thead>
<tbody>{''.join(rows_m)}</tbody>
</table>

<h3>Aspect B1 per CDT (paper Table 9 B1 columns, panel aggregate)</h3>
<table class="stats">
<thead><tr><th>CDT</th><th>d'<sub>B</sub>/d<sub>B</sub></th><th>p-value</th></tr></thead>
<tbody>{''.join(rows_c)}</tbody>
</table>
"""


def render_b2_table(b2_per_cdt_panel: dict[str, dict],
                    cdt_display: dict[str, str] | None = None) -> str:
    """Aspect B2 per-CDT table (paper Table 8 reproduction, panel-aggregate)."""
    import html as h
    cdt_display = cdt_display or {}

    cdt_keys = sorted(k for k in b2_per_cdt_panel if k != "__overall__")
    rows = []
    ov = b2_per_cdt_panel.get("__overall__", {})
    if ov.get("n_pairs", 0) > 0:
        p_str = f"{ov['p_value']:.4g}"
        if ov["significant_05"]: p_str = f"<strong>{p_str}</strong> *"
        rr_str = f"{ov['delta_RR_pct']:+.0f}%" if ov['delta_RR_pct'] is not None else "—"
        powered = "" if ov["powered"] else " <small>(low power)</small>"
        rows.append(
            f"<tr class='overall'><th>All Techniques (panel)</th>"
            f"<td>{ov['alpha']}</td><td>{ov['beta']}</td>"
            f"<td>{ov['gamma']}</td><td>{ov['delta']}</td>"
            f"<td>{p_str}{powered}</td><td>{rr_str}</td></tr>"
        )
    for cdt in cdt_keys:
        b = b2_per_cdt_panel[cdt]
        if b["n_pairs"] == 0:
            continue
        p_str = f"{b['p_value']:.4g}"
        if b["significant_05"]: p_str = f"<strong>{p_str}</strong> *"
        rr_str = f"{b['delta_RR_pct']:+.0f}%" if b['delta_RR_pct'] is not None else "—"
        powered = "" if b["powered"] else " <small>(low power)</small>"
        rows.append(
            f"<tr><th>{h.escape(cdt_display.get(cdt, cdt))}</th>"
            f"<td>{b['alpha']}</td><td>{b['beta']}</td>"
            f"<td>{b['gamma']}</td><td>{b['delta']}</td>"
            f"<td>{p_str}{powered}</td><td>{rr_str}</td></tr>"
        )

    return f"""
<h3>Aspect B2 per CDT (paper Table 8 reproduction, panel aggregate)</h3>
<p><em>2×2 contingency over (matched risky line in q<sub>R</sub>, matched risky line in q<sub>D</sub>).
H<sub>0</sub>: β = γ (no attention diversion). One-sided McNemar / Binomial on discordant pairs,
alternative β &gt; γ (deception reduces risk-marking). ΔRR = (Pr match in q<sub>D</sub> / Pr match in q<sub>R</sub>) − 1.
* = p &lt; 0.05. "(low power)" = expected cell count &lt; 5 per the paper's rule of thumb.</em></p>
<table class="stats">
<thead><tr>
<th>CDT</th>
<th>α<br><small>✗q<sub>R</sub>, ✗q<sub>D</sub></small></th>
<th>β<br><small>✓q<sub>R</sub>, ✗q<sub>D</sub></small></th>
<th>γ<br><small>✗q<sub>R</sub>, ✓q<sub>D</sub></small></th>
<th>δ<br><small>✓q<sub>R</sub>, ✓q<sub>D</sub></small></th>
<th>p-value</th><th>ΔRR</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
"""


def render_top_lines(top: dict[str, list], top_exploit_n: int = 40,
                     top_trap_n: int = 15) -> str:
    """Tables 10 + 11 reproduction."""
    import html as h

    def fmt_rows(rows: list[dict], rank_key: str) -> str:
        out = []
        for i, r in enumerate(rows, 1):
            kind_cls = {"deceptive": "dec", "risky": "rsk", "neutral": "neu"}[r["line_kind"]]
            cdt_chip = (f" <span class='cdt-chip'>{h.escape(r['applied_honeywire'])}</span>"
                        if r["applied_honeywire"] else "")
            line_text = h.escape(r["line_text"])[:160]
            qtype_short = {"filesystem":"F","htaccess":"H","httpheaders":"P","networkrequests":"S"}.get(r["qtype"], "?")
            out.append(
                f"<tr class='{kind_cls}'>"
                f"<td class='rank'>{i}</td>"
                f"<td>{qtype_short}</td>"
                f"<td><code>{h.escape(r['qid'])}</code>:{r['line_idx']}{cdt_chip}</td>"
                f"<td><code class='line'>{line_text}</code></td>"
                f"<td class='count'>{r['exploit_count']}</td>"
                f"<td class='count'>{r['trap_count']}</td>"
                f"</tr>"
            )
        return "\n".join(out)

    return f"""
<h3>Table 10 reproduction: top {top_exploit_n} individual query lines by exploit-mark count (panel)</h3>
<p><em>Per the paper convention, F = file system, H = .htaccess, P = HTTP response, S = HTTP request.
Row color: <span class='dec-chip'>deceptive</span> · <span class='rsk-chip'>risky</span> · neutral.</em></p>
<table class="lines">
<thead><tr><th>#</th><th>T</th><th>qid : line</th><th>line content</th><th>exploit</th><th>trap</th></tr></thead>
<tbody>{fmt_rows(top['top_exploit'], 'exploit_count')}</tbody>
</table>

<h3>Table 11 reproduction: top {top_trap_n} individual query lines by trap-mark count (panel)</h3>
<table class="lines">
<thead><tr><th>#</th><th>T</th><th>qid : line</th><th>line content</th><th>exploit</th><th>trap</th></tr></thead>
<tbody>{fmt_rows(top['top_trap'], 'trap_count')}</tbody>
</table>
"""
