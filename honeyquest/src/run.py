"""Run an agent against the Honeyquest query set, emit JSONL results.

Usage:
  python src/run.py --model sonnet-4-6 --query-set canonical-30 --trials 1
  python src/run.py --model opus-4-7 --query-set canonical-30 --trials 3
  python src/run.py --model whiterabbit-neo-8b --query-set full --trials 1

Result file naming:
  results/{your-handle}_{model-alias}_{query-set}_{trials}t_{ts}.jsonl

Each line is one AgentResponse (one model × one query × one trial).
A small "manifest" JSON is also written alongside with run metadata.
"""

from __future__ import annotations

import argparse
import dataclasses
import getpass
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Make ``src/`` importable
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from loader import HoneyQuery
from runners import AgentResponse, build, list_models
from runners.base import SCHEMA_VERSION


REPO = SRC.parent
DATA = REPO / "data"
RESULTS = REPO / "results"


def _load_query_set(name: str) -> list[HoneyQuery]:
    """Load a named query set. ``canonical-30`` is the survey set; ``full``
    is the entire upstream Honeyquest dataset; anything else is treated
    as a file path."""
    if name == "canonical-30":
        path = DATA / "queries_canonical_30.json"
        records = json.loads(path.read_text())
        # The survey strips labels; we need labels for analysis but we
        # don't pass them to the agent. Load ground_truth to merge metadata.
        gt = {g["qid"]: g for g in json.loads((DATA / "ground_truth.json").read_text())}
        return [
            HoneyQuery(
                qid=r["qid"],
                label=gt.get(r["qid"], {}).get("label", "neutral"),
                qtype=r["qtype"],
                data="\n".join(r["lines"]),
                lines=r["lines"],
                risky_lines=set(gt.get(r["qid"], {}).get("risky_lines", [])),
                deceptive_lines=set(gt.get(r["qid"], {}).get("deceptive_lines", [])),
                applied_honeywire=gt.get(r["qid"], {}).get("applied_honeywire", ""),
                description=gt.get(r["qid"], {}).get("description", ""),
            )
            for r in records
        ]
    if name == "full":
        # Prefer the bundled JSON pair so teammates don't need the
        # upstream Honeyquest YAML repo cloned locally. Falls back to
        # walking the YAML database directly if the JSONs are missing
        # (useful for regenerating them after an upstream refresh).
        client_path = DATA / "queries_full_174.json"
        gt_path = DATA / "ground_truth_full_174.json"
        if client_path.exists() and gt_path.exists():
            records = json.loads(client_path.read_text())
            gt = {g["qid"]: g for g in json.loads(gt_path.read_text())}
            return [
                HoneyQuery(
                    qid=r["qid"],
                    label=gt.get(r["qid"], {}).get("label", "neutral"),
                    qtype=r["qtype"],
                    data="\n".join(r["lines"]),
                    lines=r["lines"],
                    risky_lines=set(gt.get(r["qid"], {}).get("risky_lines", [])),
                    deceptive_lines=set(gt.get(r["qid"], {}).get("deceptive_lines", [])),
                    applied_honeywire=gt.get(r["qid"], {}).get("applied_honeywire", ""),
                    description=gt.get(r["qid"], {}).get("description", ""),
                )
                for r in records
            ]
        from loader import iter_queries
        return list(iter_queries())
    # Fallback: treat as path
    p = Path(name)
    if not p.exists():
        raise SystemExit(f"unknown query-set '{name}'")
    records = json.loads(p.read_text())
    return [
        HoneyQuery(
            qid=r["qid"], label=r.get("label", ""), qtype=r["qtype"],
            data="\n".join(r["lines"]), lines=r["lines"],
        )
        for r in records
    ]


def _write_manifest(out: Path, **kwargs) -> None:
    out.write_text(json.dumps(kwargs, indent=2, default=str))


def _emit_jsonl(path: Path, responses: Iterable[AgentResponse]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in responses:
            f.write(json.dumps(dataclasses.asdict(r), default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list_models(),
                   help="Model alias from runners/registry.py")
    p.add_argument("--query-set", default="full",
                   help="full (default, all 174 Honeyquest queries) | "
                        "canonical-30 (the 30 the human cohort sees) | "
                        "<path-to-queries.json>")
    p.add_argument("--trials", type=int, default=1,
                   help="Number of times to re-run each query (for variance estimation)")
    p.add_argument("--handle", default=os.environ.get("USER") or getpass.getuser(),
                   help="Your name/handle for the output filename")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional: only run the first N queries (smoke testing)")
    p.add_argument("--qids", nargs="+", default=None,
                   help="Only run these specific query IDs (for reruns)")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="Sleep N seconds between queries (thermal pacing for "
                        "local GPU runs; safe to leave 0 for short jobs)")
    p.add_argument("--out-dir", default=str(RESULTS),
                   help="Where to drop the results JSONL")
    args = p.parse_args(argv)

    queries = _load_query_set(args.query_set)
    if args.qids:
        qid_set = set(args.qids)
        queries = [q for q in queries if q.qid in qid_set]
    if args.limit is not None:
        queries = queries[: args.limit]
    print(f"loaded {len(queries)} queries from set='{args.query_set}'")

    print(f"building runner '{args.model}'...")
    runner = build(args.model)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    qset_tag = args.query_set.replace("/", "_").replace(".", "_")
    filename = f"{args.handle}_{args.model}_{qset_tag}_{args.trials}t_{ts}"
    out_dir = Path(args.out_dir)
    jsonl_path = out_dir / f"{filename}.jsonl"
    manifest_path = out_dir / f"{filename}.manifest.json"

    responses: list[AgentResponse] = []
    t0_total = time.perf_counter()
    for trial in range(1, args.trials + 1):
        for i, q in enumerate(queries, start=1):
            print(f"  trial {trial}/{args.trials}  [{i:>3}/{len(queries)}] "
                  f"{q.qtype:16s} {q.qid}", flush=True)
            r = runner.run(q)
            if r.error:
                print(f"      error: {r.error}")
            responses.append(r)
            if args.sleep > 0:
                time.sleep(args.sleep)

    elapsed = time.perf_counter() - t0_total
    _emit_jsonl(jsonl_path, responses)

    _write_manifest(manifest_path,
        schema_version=SCHEMA_VERSION,
        handle=args.handle,
        model_alias=args.model,
        model=getattr(runner, "model", args.model),
        model_version=getattr(runner, "model_version", ""),
        runner=getattr(runner, "name", ""),
        runner_version=getattr(runner, "runner_version", ""),
        query_set=args.query_set,
        n_queries=len(queries),
        trials=args.trials,
        total_responses=len(responses),
        n_errors=sum(1 for r in responses if r.error),
        started_at=responses[0].started_at if responses else None,
        finished_at=responses[-1].finished_at if responses else None,
        wall_time_s=round(elapsed, 2),
    )

    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {manifest_path}")
    print(f"  responses={len(responses)}  errors={sum(1 for r in responses if r.error)}  wall={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
