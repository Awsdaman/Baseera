"""Human-approved rewrites: a phrasing that is known to confuse routing/retrieval -> the clean question to search for.

data/curated/rewrites.json is edited only by a reviewer (tools/review.py approve-rewrite); nothing here learns by itself.
A rewrite applies only when the user's question matches an approved phrasing (after Arabic normalization) almost exactly.
"""
import json
from pathlib import Path

from rapidfuzz import fuzz

from core.normalize import normalize_ar

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "curated" / "rewrites.json"
MATCH_RATIO = 94  # near-exact: tolerates a different hamza / punctuation / spacing, not a different question

_cache = {"mtime": None, "items": []}


def _items() -> list[dict]:
    if not PATH.exists():
        return []
    m = PATH.stat().st_mtime
    if _cache["mtime"] != m:
        _cache["items"] = [dict(x, _norm=normalize_ar(x["match"])) for x in json.loads(PATH.read_text(encoding="utf-8"))]
        _cache["mtime"] = m
    return _cache["items"]


def lookup(question: str) -> dict | None:
    """The approved rewrite whose phrasing matches `question`, or None."""
    q = normalize_ar(question)
    if not q:
        return None
    best = None
    for it in _items():
        score = 100 if it["_norm"] == q else fuzz.ratio(it["_norm"], q)
        if score >= MATCH_RATIO and (best is None or score > best[0]):
            best = (score, it)
    return {k: v for k, v in best[1].items() if not k.startswith("_")} if best else None


def add(match: str, canonical: str, claim: str | None = None, added: str = "", source: str = "") -> dict:
    items = json.loads(PATH.read_text(encoding="utf-8")) if PATH.exists() else []
    entry = {"match": match.strip(), "canonical": canonical.strip(), "claim": claim, "added": added, "source": source}
    items = [x for x in items if normalize_ar(x["match"]) != normalize_ar(match)] + [entry]
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    _cache["mtime"] = None
    return entry
