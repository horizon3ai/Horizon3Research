"""Honeyquest query loader + stratified sampler.

Reads the multi-document YAMLs under ``honeyquest/querydb/queries/`` and
yields normalized query records. Parses the line-annotation strings
(e.g. ``"L4"``, ``"L3, L5"``, ``"L2-L4"``) into integer line sets
(L_R, L_D in Honeyquest terminology) for comparison against agent
answer marks the same way the paper does.
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import os

import yaml


# Only used by the YAML-walk fallback (regenerating the JSONs after an
# upstream refresh). The run path reads the baked JSON in ``data/`` and
# never touches this. Override with HONEYQUEST_ROOT=/path/to/querydb/queries;
# default points at a sibling ``querydb/queries`` next to this bundle.
HONEYQUEST_ROOT = Path(
    os.environ.get(
        "HONEYQUEST_ROOT",
        Path(__file__).resolve().parent.parent / "honeyquest_dataset" / "queries",
    )
)

_LINE_PATTERN = re.compile(r"L(\d+)(?::\d+-\d+)?(?:-L?(\d+)(?::\d+-\d+)?)?")


def _parse_line_spec(spec: str | None) -> set[int]:
    """Parse Honeyquest line specs like ``"L4"``, ``"L3, L5"``, ``"L2-L4"``,
    and column-range variants like ``"L10:69-79"`` (extracts the line number,
    ignores the column range)."""
    if not spec:
        return set()
    out: set[int] = set()
    for part in re.split(r"[\s,]+", spec.strip()):
        m = _LINE_PATTERN.fullmatch(part)
        if not m:
            continue
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        out.update(range(start, end + 1))
    return out


@dataclass
class HoneyQuery:
    """One normalized Honeyquest query, ready to feed to an agent."""
    qid: str
    label: str               # neutral | risky | deceptive
    qtype: str               # filesystem | htaccess | httpheaders | networkrequests
    data: str                # raw payload as shown to the human
    lines: list[str] = field(default_factory=list)   # data.splitlines(), 0-indexed
    risky_lines: set[int] = field(default_factory=set)     # 1-indexed line numbers (L_R)
    deceptive_lines: set[int] = field(default_factory=set) # 1-indexed line numbers (L_D)
    applied_honeywire: str = ""   # CDT id from honeypatch, when present
    description: str = ""         # honeypatch description for analysis


def _doc_to_query(doc: dict) -> HoneyQuery | None:
    if not isinstance(doc, dict) or "id" not in doc or "data" not in doc:
        return None
    annotations = doc.get("annotations") or []
    risk_spec: str | None = None
    deceptive_spec: str | None = None
    honeywire = ""
    description = ""
    for entry in annotations:
        if not isinstance(entry, dict):
            continue
        for key, val in entry.items():
            if key == "risk/risky-lines":
                risk_spec = val
            elif key == "honeypatch/deceptive-lines":
                deceptive_spec = val
            elif key == "honeypatch/applied-honeywire":
                honeywire = val or ""
            elif key == "honeypatch/description":
                description = val or ""
    data = doc["data"]
    return HoneyQuery(
        qid=doc["id"],
        label=doc.get("label", "neutral"),
        qtype=doc.get("type", ""),
        data=data,
        lines=data.splitlines(),
        risky_lines=_parse_line_spec(risk_spec),
        deceptive_lines=_parse_line_spec(deceptive_spec),
        applied_honeywire=honeywire,
        description=description,
    )


def iter_queries(root: Path = HONEYQUEST_ROOT, exclude_tutorial: bool = True) -> Iterable[HoneyQuery]:
    """Yield every query under ``root``. Tutorial queries are skipped by default."""
    for path in sorted(root.rglob("*.yaml")):
        if exclude_tutorial and "tutorial" in path.parts:
            continue
        try:
            docs = list(yaml.safe_load_all(path.read_text()))
        except yaml.YAMLError:
            continue
        for doc in docs:
            q = _doc_to_query(doc)
            if q is not None:
                yield q


def stratified_sample(
    queries: list[HoneyQuery],
    n_total: int,
    seed: int = 0,
    types: tuple[str, ...] = ("filesystem", "htaccess", "httpheaders", "networkrequests"),
    labels: tuple[str, ...] = ("neutral", "risky", "deceptive"),
) -> list[HoneyQuery]:
    """Stratified sample across (type × label) cells.

    Distributes ``n_total`` items as evenly as possible across the 12 cells.
    Cells with too few queries contribute whatever they have; the remainder
    is allocated round-robin across the cells with surplus, keeping the
    final count = ``n_total`` whenever the dataset is large enough.
    """
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[HoneyQuery]] = {(t, l): [] for t in types for l in labels}
    for q in queries:
        key = (q.qtype, q.label)
        if key in buckets:
            buckets[key].append(q)
    cells = list(buckets.keys())
    base = n_total // len(cells)
    extras = n_total - base * len(cells)
    rng.shuffle(cells)

    picked: list[HoneyQuery] = []
    deficits = 0
    for i, cell in enumerate(cells):
        take = base + (1 if i < extras else 0)
        pool = buckets[cell][:]
        rng.shuffle(pool)
        if len(pool) < take:
            deficits += take - len(pool)
            picked.extend(pool)
        else:
            picked.extend(pool[:take])

    # Backfill from any leftover items if some cells were short
    if deficits > 0:
        leftover = [
            q for cell in cells
            for q in buckets[cell] if q not in picked
        ]
        rng.shuffle(leftover)
        picked.extend(leftover[:deficits])

    rng.shuffle(picked)
    return picked[:n_total]


if __name__ == "__main__":
    all_qs = list(iter_queries())
    print(f"Loaded {len(all_qs)} queries total")
    from collections import Counter
    print("By type x label:", Counter((q.qtype, q.label) for q in all_qs))
    sample = stratified_sample(all_qs, 25, seed=42)
    print(f"\nSampled {len(sample)} queries:")
    for q in sample:
        print(f"  {q.qtype:16s} {q.label:10s} {q.qid}  L_R={sorted(q.risky_lines)} L_D={sorted(q.deceptive_lines)}")
