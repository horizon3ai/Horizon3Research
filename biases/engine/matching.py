"""Choice extraction and exact matching for the engine: pull the {choice, rationale} JSON from a model
reply, and resolve the raw choice to one bundle item by exact app-name or CVE-ID match (no substring)."""
import json
import re


def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def _balanced_objs(text):
    """Yield every top-level balanced {...} substring, in order."""
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]
                start = None


def extract_choice_obj(content, reasoning):
    """Return the choice dict from content (preferred) or reasoning: the last balanced object with a 'choice' key."""
    for text in (content, reasoning):
        if not text:
            continue
        found = None
        for frag in _balanced_objs(text):
            try:
                obj = json.loads(frag)
            except Exception:
                continue
            if isinstance(obj, dict) and ("choice" in obj or "ranking" in obj):
                found = obj  # keep last (single-pick "choice" or top-k "ranking")
        if found is not None:
            return found
    return None


CVE_RE = re.compile(r"(CVE-\d{4}-\d{3,7}|MS\d{2}-\d{3})", re.I)

# Neutral-abstain sentinel and the phrasings a model may echo for it.
ABSTAIN_ID = "__ABSTAIN__"
ABSTAIN_TOKENS = {"none", "none of these", "none yet", "none of the above",
                  "none of these targets", "none of them"}


def match_choice(raw_choice, bundle, allow_abstain=False):
    """Resolve raw_choice to a bundle item id by exact app-name or CVE-ID match (no substring matching).
    Returns (matched_id | None, matched_by | None); allow_abstain maps the opt-out token to ABSTAIN_ID."""
    if raw_choice is None:
        return None, None
    n = _norm(raw_choice)

    if allow_abstain and n in ABSTAIN_TOKENS:  # tier: neutral opt-out (gated)
        return ABSTAIN_ID, "abstain"

    for item in bundle:                       # tier: exact app / cve
        if n == _norm(item["app"]):
            return item["id"], "app"
        if item.get("cve") and n == _norm(item["cve"]):
            return item["id"], "cve"

    for item in bundle:                       # tier: exact displayed label "App (CVE)"
        if item.get("cve") and n == _norm(f"{item['app']} ({item['cve']})"):
            return item["id"], "app_cve_label"

    m = CVE_RE.search(str(raw_choice))        # tier: CVE/MS id token -> exact cve
    if m:
        tok = _norm(m.group(1))
        for item in bundle:
            if item.get("cve") and tok == _norm(item["cve"]):
                return item["id"], "cve_token"

    stripped = _norm(re.sub(r"\s*\([^)]*\)\s*$", "", str(raw_choice)))  # strip trailing (...)
    if stripped:
        for item in bundle:
            if stripped == _norm(item["app"]):
                return item["id"], "app_paren_strip"

    return None, None


def match_ranking(ranking, bundle, k=3):
    """Map a ranked list to bundle ids (top-k) via match_choice; duplicates after the first resolve to (None, "dup")."""
    out, seen = [], set()
    for raw in list(ranking or [])[:k]:
        mid, mby = match_choice(raw, bundle)
        if mid is not None and mid in seen:
            mid, mby = None, "dup"
        elif mid is not None:
            seen.add(mid)
        out.append((mid, mby))
    return out
