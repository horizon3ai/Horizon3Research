#!/usr/bin/env python3
"""Experiment engine (config-driven). experiment_id ties the config to out/<id>.csv.

Run:  OPENROUTER_API_KEY=... python engine/run_experiment.py --config <config.yaml> [--dry-run | --verify-models]
"""
import argparse
import csv
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
# Experiment root; override with DCE_ROOT (defaults to this engine's parent).
ROOT = Path(os.environ.get("DCE_ROOT", HERE.parent)).resolve()
sys.path.insert(0, str(HERE))
import openrouter  # noqa: E402
import matching    # noqa: E402
import recon       # noqa: E402

HARNESS_VERSION = "1.0.0"

BANNED = [
    "write an exploit", "exploit code", "payload for", "step-by-step exploit",
    "reverse shell code", "msfvenom", "metasploit module", "write malware",
]

CSV_COLUMNS = [
    "harness_version", "methodology_version", "owner", "experiment_id", "cveset_id", "trial_index", "n_cves",
    "bundle_idx", "model", "prompt_id", "nmap", "recon_mode", "info_level", "cve_shown", "show_cvss", "round", "temperature", "seed",
    "presented_order", "bundle_cves", "choice_raw", "choice_matched", "matched_by", "abstain",
    "ranking_raw", "rank1_matched", "rank2_matched", "rank3_matched", "ranking_valid",
    "choice_valid", "parse_ok", "content_was_null", "retries", "n_tool_calls", "logprobs_present",
    "mean_choice_logprob", "in_tokens", "out_tokens", "reasoning_tokens",
    "latency_ms", "provider", "finish_reason", "ts", "error",
]


# ---------------------------------------------------------------- reproducible RNG
class LCG:
    """Linear congruential generator, ported from the reference prototype for reproducibility."""

    def __init__(self, seed):
        self.state = seed & 0x7FFFFFFF

    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return self.state

    def below(self, n):
        return self.next() % n


def shuffled(items, rng):
    idx = list(range(len(items)))
    for i in range(len(idx) - 1, 0, -1):
        j = rng.below(i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return [items[k] for k in idx]


def permute(bundle, base_seed, tindex):
    """Reproducible per-trial permutation of the candidate order, seeded from (run seed, trial index)."""
    out = list(bundle)
    random.Random(f"{int(base_seed)}:{tindex}").shuffle(out)
    return out


# ---------------------------------------------------------------- config / build
def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(cfg_path):
    cfg = load_yaml(cfg_path)
    src = cfg.get("pool") or cfg.get("cveset")   # 'pool' (new) or 'cveset' (legacy)
    cveset = load_yaml(ROOT / src)
    prompts = load_yaml(ROOT / cfg["prompts_file"])
    return cfg, cveset, prompts


def draw_bundles(cves, n_cves, n_bundles, rng):
    return [shuffled(cves, rng)[:n_cves] for _ in range(n_bundles)]


def presentations(cfg):
    """Yield disclosure rungs (info_level, show_cve, recon_mode, show_cvss) from `info_levels`, else the legacy `recon` axis."""
    if cfg.get("info_levels"):
        for lv in cfg["info_levels"]:
            yield (lv["name"], bool(lv.get("show_cve", True)), lv.get("recon", "none"),
                   bool(lv.get("show_cvss", False)))
    else:
        modes = cfg.get("recon") or ["version" if x else "none" for x in cfg.get("nmap", [False])]
        for rm in modes:
            yield (rm, True, rm, False)  # legacy: recon mode, CVE shown, no cvss


def build_trials(cfg, cveset):
    cves = cveset.get("cves")
    by_id = {c["id"]: c for c in cves}
    trials, tindex = [], 0
    temps = cfg.get("temperatures") or [cfg.get("temperature", 0.7)]  # sweep axis
    pres = list(presentations(cfg))  # rungs: (info_level, show_cve, recon_mode, show_cvss)
    base_seed = cfg.get("seed", 20260626)  # seeds the per-trial order permutation

    # Fixed sets: same CVEs every trial, order randomized per trial.
    if cfg.get("fixed_sets"):
        for set_name, ids in cfg["fixed_sets"].items():
            missing = [i for i in ids if i not in by_id]
            if missing:
                raise SystemExit(f"fixed_set '{set_name}' references unknown ids: {missing}")
            bundle = [by_id[i] for i in ids]
            for model in cfg["models"]:
                for prompt_id in cfg["prompts"]:
                    for info_level, show_cve, recon_mode, show_cvss in pres:
                        for temp in temps:
                            for r in range(cfg["n_rounds"]):
                                trials.append({
                                    "trial_index": tindex,
                                    "n_cves": len(bundle),
                                    "bundle_idx": set_name,        # the fixed-set name
                                    "model": model,
                                    "prompt_id": prompt_id,
                                    "nmap": recon_mode != "none",
                                    "recon_mode": recon_mode,
                                    "info_level": info_level,
                                    "show_cve": show_cve,
                                    "show_cvss": show_cvss,
                                    "temperature": temp,
                                    "round": r,
                                    "bundle": permute(bundle, base_seed, tindex),  # per-trial random permutation
                                })
                                tindex += 1
        return trials

    # LEGACY: random balanced-ish draws (kept for backward reproducibility).
    rng = LCG(cfg.get("seed", 20260626))
    for n_cves in cfg["n_cves"]:
        if n_cves > len(cves):
            print(f"  ! skipping n_cves={n_cves}: only {len(cves)} CVEs in set")
            continue
        bundles = draw_bundles(cves, n_cves, cfg["n_bundles"], rng)
        for b_idx, bundle in enumerate(bundles):
            for model in cfg["models"]:
                for prompt_id in cfg["prompts"]:
                    for info_level, show_cve, recon_mode, show_cvss in pres:
                        for temp in temps:
                            for r in range(cfg["n_rounds"]):
                                trials.append({
                                    "trial_index": tindex,
                                    "n_cves": n_cves,
                                    "bundle_idx": b_idx,
                                    "model": model,
                                    "prompt_id": prompt_id,
                                    "nmap": recon_mode != "none",
                                    "recon_mode": recon_mode,
                                    "info_level": info_level,
                                    "show_cve": show_cve,
                                    "show_cvss": show_cvss,
                                    "temperature": temp,
                                    "round": r,
                                    "bundle": permute(bundle, base_seed, tindex),  # per-trial random permutation
                                })
                                tindex += 1
    return trials


# ---------------------------------------------------------------- prompt assembly
def cvss_severity(score):
    """CVSS v3.1 qualitative severity band from the base score (derived; the pool carries only numeric cvss_v3)."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return ""
    if s >= 9.0:
        return "CRITICAL"
    if s >= 7.0:
        return "HIGH"
    if s >= 4.0:
        return "MEDIUM"
    if s >= 0.1:
        return "LOW"
    return "NONE"


def format_targets(order, show_cve=True, show_cvss=False):
    """Numbered candidate list: `N. <app>`, plus `(<cve>)` when shown and `(<cve>, CVSS <score> <SEV>)` when show_cvss."""
    lines = []
    for i, it in enumerate(order, 1):
        if show_cve:
            label = f"{it['app']} ({it['cve']}"
            if show_cvss:
                score = it.get("cvss_v3")
                label += f", CVSS {score} {cvss_severity(score)}".rstrip()
            label += ")"
        else:
            label = it["app"]
        lines.append(f"{i}. {label}")
    return "\n".join(lines)


def build_prompt(template, order, recon_mode, show_cve=True, show_cvss=False):
    recon_block = ""
    if recon_mode and recon_mode != "none":
        labels = {"generic": "Recon (Nmap output):",
                  "version": "Recon (Nmap -sV output):",
                  "scripts": "Recon (Nmap -sV --script vuln output):"}
        recon_block = labels.get(recon_mode, "Recon (Nmap output):") + "\n" + recon.synth_nmap(order, recon_mode) + "\n\n"
    # Only assert "known CVE" when the CVE is shown, so L1 (show_cve=False) stays a pure recognition test.
    lead_in = "Candidate services, each with a known CVE:" if show_cve else "Candidate services:"
    return template.format(recon=recon_block, lead_in=lead_in,
                           targets=format_targets(order, show_cve, show_cvss))


def ethics_violations(text):
    low = text.lower()
    return [b for b in BANNED if b in low]


# ---------------------------------------------------------------- logprob helper
def mean_choice_logprob(logprobs):
    try:
        toks = logprobs.get("content") or []
        vals = [t["logprob"] for t in toks if "logprob" in t]
        return round(sum(vals) / len(vals), 4) if vals else None
    except Exception:
        return None


# ---------------------------------------------------------------- one trial
def run_trial(trial, cfg, prompts, api_key):
    template = prompts[trial["prompt_id"]]
    prompt = build_prompt(template, trial["bundle"], trial["recon_mode"],
                          trial.get("show_cve", True), trial.get("show_cvss", False))
    base_msg = [{"role": "user", "content": prompt}]
    # Task-aware retry nudge: matches the task shape (ranking vs single-pick).
    if '"ranking"' in template:
        corrective = ("\n\nReturn ONLY the single-line JSON object with your top-3 \"ranking\" "
                      "list, exactly as instructed above (each entry the exact app name or CVE ID "
                      "from the list).")
    else:
        corrective = ("\n\nReturn ONLY the JSON object on one line, exactly as instructed above "
                      "(use the exact app name or CVE ID from the list).")

    resp = None
    err = None
    obj = parse_ok = raw_choice = rationale = None
    ranking_list = rank1 = rank2 = rank3 = None
    matched_id = matched_by = None
    choice_valid = False
    retries = 0
    cum_in = cum_out = cum_reason = 0   # token totals across ALL retry attempts (was: last attempt only)
    ep = cfg.get("_endpoints", {}).get(trial["model"], {})
    m_key = ep.get("api_key") or api_key
    m_prefs = ep.get("provider_prefs")                        # pinned OpenRouter provider order

    for attempt in range(cfg.get("max_retries", 2) + 1):
        messages = base_msg if attempt == 0 else [
            {"role": "user", "content": prompt + corrective}]
        rf = {"type": "json_object"} if (cfg.get("structured", True) and attempt == 0) else None
        try:
            resp = openrouter.call_model(
                m_key, trial["model"], messages,
                temperature=trial["temperature"], max_tokens=cfg["max_tokens"],
                want_logprobs=cfg.get("_lp_caps", {}).get(trial["model"], cfg.get("logprobs", True)),
                response_format=rf, provider_prefs=m_prefs)
        except openrouter.ORError as e:
            err = str(e)
            retries = attempt + 1
            time.sleep(1.5 * (attempt + 1))
            continue

        cum_in += resp.get("in_tokens") or 0      # accumulate across attempts
        cum_out += resp.get("out_tokens") or 0
        cum_reason += resp.get("reasoning_tokens") or 0
        err = None
        obj = matching.extract_choice_obj(resp.get("content"), resp.get("reasoning"))
        parse_ok = obj is not None
        rationale = obj.get("rationale") if parse_ok else None
        ranking_list = obj.get("ranking") if (parse_ok and isinstance(obj.get("ranking"), list)) else None
        if ranking_list is not None:                        # ranking task (top-k): rank-1 is the pick
            rank_matches = matching.match_ranking(ranking_list, trial["bundle"], k=3)
            ranks = [m[0] for m in rank_matches] + [None, None, None]
            rank1, rank2, rank3 = ranks[0], ranks[1], ranks[2]
            raw_choice = str(ranking_list[0]) if ranking_list else None
            matched_id = rank1
            matched_by = rank_matches[0][1] if rank_matches else None
        else:                                               # single-pick task
            raw_choice = obj.get("choice") if parse_ok else None
            # Only the *_abstain prompts offer the neutral opt-out (allow_abstain=True); all others are unchanged.
            allow_abstain = str(trial["prompt_id"]).endswith("_abstain")
            matched_id, matched_by = matching.match_choice(raw_choice, trial["bundle"], allow_abstain=allow_abstain)
            rank1 = rank2 = rank3 = None
        choice_valid = matched_id is not None
        if parse_ok and choice_valid:
            break
        retries = attempt + 1  # will retry if attempts remain

    ts = datetime.now(timezone.utc).isoformat()
    bundle_ids = [it["id"] for it in trial["bundle"]]
    row = {
        "harness_version": HARNESS_VERSION,
        "methodology_version": cfg.get("methodology_version", "unspecified"),
        "owner": cfg.get("owner", "anonymous"),
        "experiment_id": cfg["experiment_id"],
        "cveset_id": cfg.get("cveset_id", ""),
        "trial_index": trial["trial_index"],
        "n_cves": trial["n_cves"],
        "bundle_idx": trial["bundle_idx"],
        "model": trial["model"],
        "prompt_id": trial["prompt_id"],
        "nmap": trial["nmap"],
        "recon_mode": trial["recon_mode"],
        "info_level": trial.get("info_level"),
        "cve_shown": trial.get("show_cve"),
        "show_cvss": trial.get("show_cvss"),
        "round": trial["round"],
        "temperature": trial["temperature"],
        "seed": cfg.get("seed", ""),
        "presented_order": "|".join(bundle_ids),
        "bundle_cves": ";".join(it.get("cve", "") for it in trial["bundle"]),
        "choice_raw": raw_choice,
        "choice_matched": matched_id,
        "matched_by": matched_by,
        "abstain": matched_by == "abstain",   # Q5 C7: distinct valid opt-out outcome (never a parse fail)
        "ranking_raw": json.dumps(ranking_list) if ranking_list is not None else None,
        "rank1_matched": rank1,
        "rank2_matched": rank2,
        "rank3_matched": rank3,
        "ranking_valid": (rank1 is not None) if ranking_list is not None else None,
        "choice_valid": choice_valid,
        "parse_ok": bool(parse_ok),
        "content_was_null": (resp.get("content") is None) if resp else None,
        "retries": retries,
        "n_tool_calls": 0,
        "logprobs_present": bool(resp and resp.get("logprobs")),
        "mean_choice_logprob": mean_choice_logprob(resp["logprobs"]) if (resp and resp.get("logprobs")) else None,
        "in_tokens": cum_in,
        "out_tokens": cum_out,
        "reasoning_tokens": cum_reason,
        "latency_ms": resp.get("latency_ms") if resp else None,
        "provider": resp.get("provider") if resp else None,
        "finish_reason": resp.get("finish_reason") if resp else None,
        "ts": ts,
        "error": err,
    }
    raw_rec = dict(row)
    raw_rec.update({
        "ranking": ranking_list,
        "rationale_text": rationale,
        "reasoning_full": resp.get("reasoning") if resp else None,
        "content_full": resp.get("content") if resp else None,
        "logprobs": resp.get("logprobs") if resp else None,
        "request_id": resp.get("request_id") if resp else None,
        "bundle": trial["bundle"],
    })
    return row, raw_rec


# ---------------------------------------------------------------- dry run
def do_dry_run(cfg, cveset, prompts, trials):
    print(f"\n=== DRY RUN, experiment {cfg['experiment_id']} (methodology v{cfg.get('methodology_version','?')}) ===")
    print(f"models={cfg['models']}")
    print(f"temperatures={cfg.get('temperatures') or [cfg.get('temperature', 0.7)]}")
    if cfg.get("info_levels"):
        print("info_levels: " + ", ".join(
            f"{lv['name']}(cve={lv.get('show_cve', True)},recon={lv.get('recon', 'none')}"
            f",cvss={lv.get('show_cvss', False)})"
            for lv in cfg["info_levels"]))
    if cfg.get("fixed_sets"):
        print(f"fixed_sets={ {k: v for k, v in cfg['fixed_sets'].items()} }")
        print(f"prompts={cfg['prompts']}  n_rounds={cfg['n_rounds']}")
    else:
        print(f"prompts={cfg['prompts']}  n_cves={cfg['n_cves']}  n_bundles={cfg['n_bundles']}"
              f"  n_rounds={cfg['n_rounds']}  recon={cfg.get('recon') or cfg.get('nmap')}")
    print(f"TOTAL TRIALS: {len(trials)}")

    # ethics lint across every distinct rendered prompt; keep one sample per disclosure rung
    seen, violations, samples = set(), [], {}
    for t in trials:
        lvl = t.get("info_level", t["recon_mode"])
        key = (t["prompt_id"], lvl, t["n_cves"], t["bundle_idx"])
        if key in seen:
            continue
        seen.add(key)
        p = build_prompt(prompts[t["prompt_id"]], t["bundle"], t["recon_mode"],
                         t.get("show_cve", True), t.get("show_cvss", False))
        violations += [(key, v) for v in ethics_violations(p)]
        samples.setdefault(lvl, p)

    for lvl, p in samples.items():
        print(f"\n--- sample prompt (info_level={lvl}) ---\n" + p)

    if violations:
        print(f"\n!!! ETHICS LINT FAILED: {violations[:5]}")
    else:
        print("\nethics lint: OK (no exploit-content phrases in any rendered prompt)")

    # rough cost band (real numbers come from the run)
    est_in = max((len(p) for p in samples.values()), default=0) // 4 + 40
    lo = len(trials) * (est_in + 300) / 1e6 * 1.0
    hi = len(trials) * (est_in + 2500) / 1e6 * 5.0
    print(f"\nrough cost band: ${lo:.2f}-${hi:.2f} (blended est.; thinking-model "
          f"reasoning tokens dominate, measure for real on the run)")


# ------------------------------------------------------ provider routing (OpenRouter)
def model_ids(specs):
    """A config `models` entry is a plain OpenRouter id (str) or a mapping {id, provider_order}."""
    return [m["id"] if isinstance(m, dict) else m for m in specs]


def model_endpoints(specs, or_key):
    """Map model id -> {api_key, provider_prefs}. Entry is a str (OpenRouter default) or a dict
    with `provider_order` (pinned OpenRouter routing)."""
    eps = {}
    for m in specs:
        if isinstance(m, dict):
            prefs = {"order": list(m["provider_order"])} if m.get("provider_order") else None
            if prefs is not None and "allow_fallbacks" in m:
                prefs["allow_fallbacks"] = bool(m["allow_fallbacks"])
            eps[m["id"]] = {"api_key": or_key, "provider_prefs": prefs}
        else:
            eps[m] = {"api_key": or_key, "provider_prefs": None}
    return eps


# ---------------------------------------------------------------- capability probe
def probe_logprob_caps(models, endpoints):
    """One cheap call per model: does it accept logprobs? Returns {model: bool} (default True on non-logprob errors)."""
    caps = {}
    for m in models:
        ep = endpoints.get(m, {})
        try:
            openrouter.call_model(ep.get("api_key"), m, [{"role": "user", "content": "ok"}],
                                  max_tokens=16, want_logprobs=True)
            caps[m] = True
        except openrouter.ORError as e:
            caps[m] = "logprob" not in str(e).lower()
    return caps


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-models", action="store_true")
    ap.add_argument("--rounds", type=int, help="override n_rounds")
    ap.add_argument("--limit", type=int, help="cap number of trials (smoke test)")
    ap.add_argument("--concurrency", type=int, help="override concurrency (operational; no effect on results)")
    args = ap.parse_args()

    cfg, cveset, prompts = resolve(args.config)
    cfg["cveset_id"] = cveset.get("cveset_id") or cveset.get("pool_id", "")
    cfg.setdefault("methodology_version", "unspecified")
    if args.rounds is not None:
        cfg["n_rounds"] = args.rounds
    if args.concurrency is not None:
        cfg["concurrency"] = args.concurrency
    cfg["_model_specs"] = list(cfg["models"])        # raw specs (str = OpenRouter, or {id, provider_order})
    cfg["models"] = model_ids(cfg["_model_specs"])   # ids only, for build_trials / dry-run / CSV

    trials = build_trials(cfg, cveset)
    # Round-robin across models so concurrency spans all providers. Order does not affect results
    # (per-trial RNG seeded by trial_index; rows re-sorted by trial_index before the CSV is written).
    if len(cfg.get("models", [])) > 1:
        from collections import deque
        buckets = {}
        for t in trials:
            buckets.setdefault(t["model"], deque()).append(t)
        qs = list(buckets.values())
        interleaved = []
        while any(qs):
            for q in qs:
                if q:
                    interleaved.append(q.popleft())
        trials = interleaved
    if args.limit:
        trials = trials[:args.limit]

    if args.dry_run:
        do_dry_run(cfg, cveset, prompts, trials)
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    cfg["_endpoints"] = model_endpoints(cfg["_model_specs"], api_key or "")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set (required for OpenRouter-hosted models)")

    if args.verify_models:
        print("Verifying model ids...")
        for m in cfg["models"]:
            ep = cfg["_endpoints"].get(m, {})
            try:
                openrouter.verify_model(ep.get("api_key") or api_key, m)
                print(f"  OK   {m}")
            except openrouter.ORError as e:
                print(f"  FAIL {m}: {e}")
        return

    out_csv = ROOT / "out" / f"{cfg['experiment_id']}.csv"
    raw_jsonl = ROOT / "raw" / f"{cfg['experiment_id']}.jsonl"
    out_csv.parent.mkdir(exist_ok=True)
    raw_jsonl.parent.mkdir(exist_ok=True)

    if cfg.get("logprobs", True):
        cfg["_lp_caps"] = probe_logprob_caps(cfg["models"], cfg["_endpoints"])
        print("logprob support: " + ", ".join(
            f"{m.split('/')[-1]}={v}" for m, v in cfg["_lp_caps"].items()))
    else:
        cfg["_lp_caps"] = {m: False for m in cfg["models"]}

    print(f"Running experiment {cfg['experiment_id']}: {len(trials)} trials, "
          f"concurrency={cfg.get('concurrency', 6)}")
    rows = []
    lock = threading.Lock()
    done = [0]
    t_start = time.time()

    with open(raw_jsonl, "w") as rawf:
        def worker(tr):
            row, raw_rec = run_trial(tr, cfg, prompts, api_key)
            with lock:
                rawf.write(json.dumps(raw_rec, default=str) + "\n")
                rawf.flush()
                rows.append(row)
                done[0] += 1
                if os.environ.get("DCE_VERBOSE"):   # opt-in live per-trial pick (watch the models choose)
                    _m = str(row.get("model", "")).split("/")[-1]
                    _p = str(row.get("prompt_id", "")).split("_")[-1]
                    _set = str(row.get("bundle_idx", ""))
                    _ch = row.get("choice_matched") or (row.get("choice_raw") or "?")
                    _rat = (raw_rec.get("rationale_text") or "").replace("\n", " ")[:70]
                    _flag = "" if row.get("choice_valid") else " [INVALID]"
                    print(f"    {done[0]:>4} {_m:<20} {_p:<7} {_set:<9} -> {_ch:<10}{_flag}  {_rat}", flush=True)
                if done[0] % 10 == 0 or done[0] == len(trials):
                    ok = sum(1 for r in rows if r["choice_valid"])
                    print(f"  {done[0]}/{len(trials)}  valid={ok}", flush=True)
            return row

        with ThreadPoolExecutor(max_workers=cfg.get("concurrency", 6)) as ex:
            futs = [ex.submit(worker, tr) for tr in trials]
            for _ in as_completed(futs):
                pass

    rows.sort(key=lambda r: r["trial_index"])
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    valid = sum(1 for r in rows if r["choice_valid"])
    parsed = sum(1 for r in rows if r["parse_ok"])
    errs = sum(1 for r in rows if r["error"])
    in_tok = sum(r["in_tokens"] or 0 for r in rows)
    out_tok = sum(r["out_tokens"] or 0 for r in rows)
    print(f"\nDONE in {time.time()-t_start:.0f}s")
    print(f"  trials={len(rows)} parsed={parsed} valid={valid} errors={errs}")
    print(f"  tokens: in={in_tok} out={out_tok}")
    print(f"  csv -> {out_csv}")
    print(f"  raw -> {raw_jsonl}")


if __name__ == "__main__":
    main()
