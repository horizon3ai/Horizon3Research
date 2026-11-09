#!/usr/bin/env python3
"""Objective prominence index for the CVE pool: combines training-data prominence proxies
(GitHub PoC repos, writeup density, marketing name, CISA KEV, Metasploit, Exploit-DB) into a
0-100 index, then ranks the pool by fixed tier cutoffs. Proxy values come from data/fame_proxies.yaml.

Usage:  python fame_rank.py [--init-proxies]   # writes fame-rank.csv
"""
from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
# Reads the frozen input snapshot in data/ (cve_pool.yaml + fame_proxies.yaml).
DATA = HERE / "data"
POOL = DATA / "cve_pool.yaml"
PROXIES = DATA / "fame_proxies.yaml"
OUT_CSV = HERE / "fame-rank.csv"
OUT_MD = HERE / "fame-rank.md"

GITHUB_REF = 500          # log-normalization reference (repos); g saturates ~here
WRITEUP_MAP = {"none": 0.0, "low": 0.33, "medium": 0.66, "high": 1.0}

W_GITHUB = 0.42
W_WRITEUP = 0.30
W_NAMED = 0.12
W_KEV = 0.08
W_MSF = 0.05
W_EDB = 0.03
assert abs(W_GITHUB + W_WRITEUP + W_NAMED + W_KEV + W_MSF + W_EDB - 1.0) < 1e-9

# fame_index (0..100) -> tier
def tier_of(idx: float) -> str:
    if idx >= 62: return "famous"
    if idx >= 34: return "mid"
    return "obscure"

def load_pool():
    pool = yaml.safe_load(POOL.read_text())["cves"]
    by_cve = {}
    for c in pool:
        cve = c.get("cve")
        if not cve:
            continue
        by_cve.setdefault(cve, {"cve": cve, "apps": set(), "pool_ids": [],
                                "era": c.get("era"), "cvss": c.get("cvss_v3"),
                                "old_tier": c.get("fame_tier"), "old_score": c.get("fame_score")})
        by_cve[cve]["apps"].add(c.get("app", ""))
        by_cve[cve]["pool_ids"].append(c.get("id"))
    return by_cve

def norm_github(n) -> float:
    if n is None:
        return 0.0
    return min(1.0, math.log10(1 + max(0, n)) / math.log10(1 + GITHUB_REF))

def score_one(px: dict) -> dict:
    g = norm_github(px.get("github_repos"))
    w = WRITEUP_MAP.get((px.get("writeup_density") or "none").lower(), 0.0)
    m = 1.0 if px.get("metasploit_module") else 0.0
    e = 1.0 if px.get("exploitdb_entry") else 0.0
    named = 1.0 if px.get("named_handle") else 0.0
    kev = 1.0 if px.get("cisa_kev") else 0.0
    idx = 100 * (W_GITHUB * g + W_WRITEUP * w + W_NAMED * named + W_KEV * kev + W_MSF * m + W_EDB * e)
    return {"fame_index": round(idx, 1), "new_tier": tier_of(idx),
            "_g": round(g, 3), "_w": w, "_named": int(named), "_kev": int(kev),
            "_m": int(m), "_e": int(e)}

def init_proxies():
    by_cve = load_pool()
    if PROXIES.exists():
        existing = yaml.safe_load(PROXIES.read_text()) or {}
        existing = existing.get("cves", existing)
    else:
        existing = {}
    doc = {"_schema": {
                "github_repos": "int: approx # of GitHub repos mentioning/PoC-ing the CVE-ID",
                "exploitdb_entry": "bool",
                "metasploit_module": "bool",
                "writeup_density": "one of: none | low | medium | high (OSCP/HTB/THM/blog coverage)",
                "named_handle": "bool: has a marketing name (Log4Shell, Heartbleed, ProxyShell...)",
                "cisa_kev": "bool: on CISA Known-Exploited-Vulnerabilities catalog",
                "nvd_refs": "int: recorded for transparency; ZERO weight in the equation",
                "fabricated": "bool: true for fake/decoy CVE-IDs -- excluded from ranking",
                "source_note": "str: where the numbers came from + date",
            },
            "cves": {}}
    for cve, meta in sorted(by_cve.items()):
        prev = existing.get(cve, {}) if isinstance(existing, dict) else {}
        doc["cves"][cve] = {
            "app": " / ".join(sorted(a for a in meta["apps"] if a)),
            "github_repos": prev.get("github_repos"),
            "exploitdb_entry": prev.get("exploitdb_entry"),
            "metasploit_module": prev.get("metasploit_module"),
            "writeup_density": prev.get("writeup_density"),
            "named_handle": prev.get("named_handle"),
            "cisa_kev": prev.get("cisa_kev"),
            "nvd_refs": prev.get("nvd_refs"),
            "fabricated": prev.get("fabricated", False),
            "source_note": prev.get("source_note"),
        }
    PROXIES.write_text(yaml.safe_dump(doc, sort_keys=False, width=100))
    print(f"wrote skeleton for {len(doc['cves'])} unique CVEs -> {PROXIES}")

def run():
    by_cve = load_pool()
    if not PROXIES.exists():
        sys.exit(f"missing {PROXIES}; run --init-proxies and fill it first")
    proxies = (yaml.safe_load(PROXIES.read_text()) or {}).get("cves", {})

    rows = []
    for cve, meta in by_cve.items():
        px = proxies.get(cve, {})
        if px.get("fabricated"):
            continue
        if px.get("github_repos") is None and not px.get("writeup_density"):
            rows.append({"cve": cve, "app": " / ".join(sorted(a for a in meta["apps"] if a)),
                         "fame_index": None, "new_tier": "PENDING",
                         "old_tier": meta["old_tier"], "old_score": meta["old_score"],
                         "era": meta["era"], "cvss": meta["cvss"], "flip": ""})
            continue
        sc = score_one(px)
        flip = "" if sc["new_tier"] == meta["old_tier"] else f"{meta['old_tier']}->{sc['new_tier']}"
        rows.append({"cve": cve, "app": " / ".join(sorted(a for a in meta["apps"] if a)),
                     "fame_index": sc["fame_index"], "new_tier": sc["new_tier"],
                     "old_tier": meta["old_tier"], "old_score": meta["old_score"],
                     "era": meta["era"], "cvss": meta["cvss"], "flip": flip,
                     **{k: sc[k] for k in ("_g", "_w", "_named", "_kev", "_m", "_e")}})

    scored = [r for r in rows if r["fame_index"] is not None]
    pending = [r for r in rows if r["fame_index"] is None]
    scored.sort(key=lambda r: r["fame_index"], reverse=True)
    for i, r in enumerate(scored, 1):
        r["rank"] = i

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["rank", "cve", "app", "era", "cvss",
            "fame_index", "new_tier", "old_tier", "old_score", "flip",
            "_g", "_w", "_named", "_kev", "_m", "_e"])
        wtr.writeheader()
        for r in scored:
            wtr.writerow({k: r.get(k, "") for k in wtr.fieldnames})

    lines = ["# Objective prominence ranking (training-data prominence)", "",
             "Equation + weights: see `fame_rank.py`. "
             "`fame_index` 0-100; NVD refs excluded by design. "
             "Tiers: obscure <34, mid 34-61, famous >=62.", "",
             f"Scored {len(scored)} CVEs; {len(pending)} pending proxy data.", "",
             "| rank | fame_index | new_tier | old_tier | flip | CVE | app | era | cvss |",
             "|---:|---:|---|---|---|---|---|---:|---:|"]
    for r in scored:
        lines.append(f"| {r['rank']} | {r['fame_index']} | {r['new_tier']} | {r['old_tier']} | "
                     f"{r['flip'] or ''} | {r['cve']} | {r['app']} | {r['era']} | {r['cvss']} |")
    if pending:
        lines += ["", "## Pending proxy data", ""] + [f"- {r['cve']} ({r['app']})" for r in pending]
    OUT_MD.write_text("\n".join(lines) + "\n")

    flips = [r for r in scored if r["flip"]]
    print(f"scored {len(scored)}, pending {len(pending)}, ranked (tier flips): {len(flips)}")
    for r in flips:
        print(f"  {r['flip']:18} idx={r['fame_index']:5} {r['cve']} ({r['app']})")
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_MD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-proxies", action="store_true", help="emit/refresh the proxies skeleton")
    args = ap.parse_args()
    if args.init_proxies:
        init_proxies()
    else:
        run()
